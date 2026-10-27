import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ======================================================
# 0️⃣ 基础配置
# ======================================================
os.makedirs("qkv_outputs2", exist_ok=True)

trainer_state = {
    "global_step": 0,
    "max_steps": 999999,
    "buffer": [],                # 存 step 的 QKV（GPU）
    "flush_id": 0                # flush 批次 index
}

SAVE_LAST_N_STEPS = 120
FLUSH_SIZE = 20   # ⭐⭐ 每 20 个 step flush 一次，防止 OOM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# 1️⃣ 加载模型
# ======================================================
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# 2️⃣ LoRA 配置
# ======================================================
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
model = model.to(device)
model.print_trainable_parameters()

# ======================================================
# 3️⃣ 注册 HOOK（GPU 版本）
# ======================================================
TARGET_LAYERS = [f"decoder.layers.{i}" for i in range(12)]

print("Hook will record Q/K/V from last steps:")

def hook_fn(name):
    def hook(module, input, output):
        step = trainer_state["global_step"]
        max_steps = trainer_state["max_steps"]

        if step >= max_steps - SAVE_LAST_N_STEPS:

            # ⭐⭐ 修改：强制统一到 cuda:0
            out = output.detach()

            if out.dim() == 3 and out.shape[-1] == 768:
                trainer_state["buffer"].append(out.to("cuda:0"))


            if len(trainer_state["buffer"]) >= FLUSH_SIZE:
                flush_path = f"qkv_outputs2/flush_{trainer_state['flush_id']}.pt"
                torch.save(torch.cat(trainer_state["buffer"], dim=1), flush_path)
                print(f"[FLUSH] Saved {flush_path}")
                trainer_state["flush_id"] += 1
                trainer_state["buffer"] = []

    return hook


# 注册 hook
for name, module in model.named_modules():
    if any(layer in name for layer in TARGET_LAYERS):
        if any(x in name for x in ["q_proj", "k_proj", "v_proj"]):
            module.register_forward_hook(hook_fn(name))
            print(f"Hook registered on: {name}")

# ======================================================
# 4️⃣ 加载数据
# ======================================================
dataset = load_dataset("tatsu-lab/alpaca")["train"]

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n{example['input']}\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_example)

# ======================================================
# 5️⃣ 训练参数
# ======================================================
training_args = TrainingArguments(
    output_dir="./opt125m_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=True,
    optim="adamw_torch"
)

# ======================================================
# 6️⃣ 带步数跟踪的 Trainer
# ======================================================
class MyTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        trainer_state["global_step"] = self.state.global_step
        trainer_state["max_steps"] = self.state.max_steps
        return super().compute_loss(model, inputs, return_outputs)

# ======================================================
# 7️⃣ Collator
# ======================================================
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="### Response:",
    instruction_template="### Instruction:"
)

# ======================================================
# 8️⃣ 训练
# ======================================================
trainer = MyTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=collator,
    dataset_text_field="text"
)

trainer.train()

# ======================================================
# ⭐⭐⭐ 9️⃣ 训练结束：写入最后一次 flush
# ======================================================
if len(trainer_state["buffer"]) > 0:
    flush_path = f"qkv_outputs2/flush_{trainer_state['flush_id']}.pt"
    torch.save(torch.cat(trainer_state["buffer"], dim=1), flush_path)
    print(f"[FLUSH] Saved last flush: {flush_path}")

# ======================================================
# 🔟 最终扩维 concat
# ======================================================
print("🔗 Combining flush files into final expanded QKV...")

flush_files = [
    f"qkv_outputs2/{f}" for f in os.listdir("qkv_outputs2") if f.startswith("flush_")
]

expanded_list = []
for f in flush_files:
    expanded_list.append(torch.load(f, map_location=device))

expanded_qkv = torch.cat(expanded_list, dim=1)  # GPU concat

torch.save(expanded_qkv.cpu(), "qkv_outputs2/expanded_qkv.pt")
print("🎉 Saved expanded_qkv.pt")
print(f"📐 Final shape = {expanded_qkv.shape}")

# ======================================================
# 🔚 保存模型
# ======================================================
model.save_pretrained("./opt125m_finetuned/final_model")
tokenizer.save_pretrained("./opt125m_finetuned/final_model")

print("🎯 Training complete with safe GPU expansion!")

