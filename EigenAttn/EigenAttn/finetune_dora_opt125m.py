import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os

# ======================================================
# 0️⃣ 基础配置
# ======================================================
os.makedirs("qkv_outputs", exist_ok=True)
trainer_state = {"global_step": 0, "max_steps": 999999}
SAVE_LAST_N_STEPS = 120

# ======================================================
# 1️⃣ 加载模型和分词器
# ======================================================
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# 2️⃣ 配置 LoRA
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
model.print_trainable_parameters()

# ======================================================
# 3️⃣ 注册 hooks（GPU 保存版本）
# ======================================================
TARGET_LAYERS = [f"decoder.layers.{i}" for i in range(12)]
print(f"✅ Hook will record Q/K/V from 12 layers × 3 projections in the last {SAVE_LAST_N_STEPS} steps.")

def hook_fn(name):
    def hook(module, input, output):
        step = trainer_state["global_step"]
        max_steps = trainer_state["max_steps"]
        if step >= max_steps - SAVE_LAST_N_STEPS:
            save_name = name.replace(".", "_")
            save_path = f"qkv_outputs2/{save_name}_step{step}.pt"

            try:
                # 🚀 保存 GPU 张量（保留原设备信息）
                torch.save(output.detach().float(), save_path)
                print(f"💾 [GPU] Saved tensor: {save_path} ({output.device})")

            except RuntimeError as e:
                print(f"⚠️ GPU save failed ({e}), fallback to CPU...")
                torch.save(output.detach().float().cpu(), save_path)
                print(f"💾 [CPU] Saved tensor: {save_path}")

    return hook

# 注册 hook 到所有层的 q/k/v
for name, module in model.named_modules():
    if any(layer in name for layer in TARGET_LAYERS):
        if any(x in name for x in ["q_proj", "k_proj", "v_proj"]):
            module.register_forward_hook(hook_fn(name))
            print(f"🔗 Hook registered on: {name}")

# ======================================================
# 4️⃣ 加载数据集
# ======================================================
dataset = load_dataset("tatsu-lab/alpaca")["train"]

def format_example(example):
    instruction = example["instruction"]
    input_text = example["input"] or ""
    output_text = example["output"]
    return {
        "text": f"### Instruction:\n{instruction}\n{input_text}\n### Response:\n{output_text}"
    }

dataset = dataset.map(format_example)

# ======================================================
# 5️⃣ 训练参数
# ======================================================
training_args = TrainingArguments(
    output_dir="./dora_opt125m_finetune",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=True,
)

# ======================================================
# 6️⃣ 自定义 Trainer：跟踪训练步数
# ======================================================
class MyTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        trainer_state["global_step"] = self.state.global_step
        trainer_state["max_steps"] = self.state.max_steps
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

# ======================================================
# 7️⃣ Collator
# ======================================================
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="### Response:",
    instruction_template="### Instruction:"
)

# ======================================================
# 8️⃣ 启动训练
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
# 9️⃣ 保存最终模型
# ======================================================
model.save_pretrained("./dora_opt125m_finetune/final_model")
tokenizer.save_pretrained("./dora_opt125m_finetune/final_model")

print("🎯 Training complete! Q/K/V tensors for 12 layers × 3 × 8 steps saved to ./qkv_outputs/ (GPU mode)")

