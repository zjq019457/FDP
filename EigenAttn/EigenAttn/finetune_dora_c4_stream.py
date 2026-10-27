#!/usr/bin/env python3
import os, math, glob, gzip, json, random, shutil, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import PeftModel
from transformers import TrainerCallback

# ================= CONFIG =================
DATA_DIR = "/diskE/c4-en/en/"
MODEL_PATH = "./eigen_outputs_1.0x/final_model"
TOKENIZER_PATH = "/home/bai-project/jiaqizhang/opt-125m"
ADAPTER_PATH = "./dora_extreme_ppl_model"
OUTPUT_DIR = "/home/bai-project/jiaqizhang/OptFinetune"

BLOCK_SIZE = 1024
PPL_EVAL_STEPS = 2000
PPL_BATCHES = 50

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda"

# ================= DISK CHECK =================
def check_disk(path="/diskF", min_gb=100):
    free = shutil.disk_usage(path).free / (1024**3)
    print(f"💾 Disk free: {free:.1f} GB")
    if free < min_gb:
        raise RuntimeError("Disk almost full")

check_disk("/diskF")

# ================= TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ================= MODEL =================
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
model.to(device)
model.print_trainable_parameters()

# ================= C4 RAW STREAM =================
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json.gz")))
assert len(files) > 0, "No C4 files found"

class C4Stream(torch.utils.data.IterableDataset):
    def __init__(self, files, tokenizer, block_size=1024):
        self.files = files
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        buffer = []
        while True:
            random.shuffle(self.files)
            for path in self.files:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        text = json.loads(line)["text"]
                        ids = self.tokenizer(text, return_attention_mask=False)["input_ids"]
                        buffer.extend(ids)

                        while len(buffer) >= self.block_size:
                            chunk = buffer[:self.block_size]
                            buffer = buffer[self.block_size:]
                            yield {
                                "input_ids": torch.tensor(chunk, dtype=torch.long),
                                "labels": torch.tensor(chunk, dtype=torch.long),
                            }

train_dataset = C4Stream(files, tokenizer, BLOCK_SIZE)

# ================= COLLATOR =================
def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }

# ================= PPL =================
@torch.no_grad()
def compute_ppl(model, dataset, max_batches=50):
    model.eval()
    total_loss, total_tokens = 0, 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k,v in batch.items()}
        out = model(**batch)
        loss = out.loss
        n = batch["input_ids"].numel()
        total_loss += loss.item() * n
        total_tokens += n

    model.train()
    return math.exp(total_loss / total_tokens)

# ================= CALLBACK =================
class PPLCallback(TrainerCallback):
    def __init__(self):
        self.best = 1e9

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % PPL_EVAL_STEPS != 0 or not state.is_world_process_zero:
            return

        print("\n📊 Evaluating PPL...")
        ppl = compute_ppl(model, train_dataset, PPL_BATCHES)
        print(f"📉 PPL @ step {state.global_step}: {ppl:.2f}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "ppl_log.txt"), "a") as f:
            f.write(f"{state.global_step},{ppl}\n")

        if ppl < self.best:
            self.best = ppl
            model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
            print("🏆 New best model saved")

# ================= TRAIN ARGS =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=50000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,
    logging_steps=100,
    bf16=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    report_to="none",
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    callbacks=[PPLCallback()],
)

# ================= RESUME =================
def get_last_ckpt(d):
    if not os.path.isdir(d):
        return None
    cks = [x for x in os.listdir(d) if x.startswith("checkpoint-")]
    if not cks:
        return None
    return os.path.join(d, sorted(cks, key=lambda x:int(x.split("-")[-1]))[-1])

ckpt = get_last_ckpt(OUTPUT_DIR)
trainer.train(resume_from_checkpoint=ckpt)

print("🏁 Training finished")

