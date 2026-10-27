#!/usr/bin/env python3
"""
finetune_extreme_ppl_local_with_diskcheck.py

特点：
- 本地模型、本地 tokenizer、本地 C4 数据（train 1%）
- DoRA 微调 q/k/v
- 每个 epoch 计算 PPL，自动 early stop
- 自动保存最佳模型
- 增加磁盘监控 check_disk()

"""

import os
import math
import glob
import torch
import shutil
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from transformers.utils import logging

# ================================================================
# 路径配置（本地）
# ================================================================
DATA_DIR       = "/diskE/c4-en/en/"
CACHE_DIR      = "/diskF/c4-cache/"
MODEL_PATH     = "./eigen_outputs_1.0x/final_model"
TOKENIZER_PATH = "/home/bai-project/jiaqizhang/opt-125m"
OUTPUT_DIR     = "/home/bai-project/jiaqizhang/OptFinetune"
RESUME_PATH = "/home/bai-project/jiaqizhang/OptFinetune"
print("==== 路径配置 ====")
print("DATA_DIR:", DATA_DIR)
print("CACHE_DIR:", CACHE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("TOKENIZER_PATH:", TOKENIZER_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)

logging.set_verbosity_info()

# ================================================================
# GPU 设置
# ================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# 监控磁盘空间函数（你提供的版本）
# ================================================================
def check_disk(path="/diskF", min_gb=100):
    free_gb = shutil.disk_usage(path).free / (1024**3)
    print(f"💾 剩余空间: {free_gb:.2f} GB")
    if free_gb < min_gb:
        raise RuntimeError(f"⚠️ 磁盘空间不足！剩余 {free_gb:.2f} GB，低于阈值 {min_gb} GB")


# 在训练前检查一次磁盘空间
check_disk(CACHE_DIR, min_gb=100)


# ================================================================
# 加载 tokenizer 与模型
# ================================================================
print("📦 Loading tokenizer from local disk...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

print("📦 Loading model with EigenAttention & DoRA support...")
if os.path.exists(os.path.join(RESUME_PATH, "adapter_model.bin")):
    print("🔄 检测到已有训练进度，自动从上次保存加载模型（Resume Training）")
    model = AutoModelForCausalLM.from_pretrained(
        RESUME_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
else:
    print("📦 加载初始模型（基于 EigenAttention 的 OPT-125M）")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
model.to(device)


# ================================================================
# 配置 DoRA / LoRA
# ================================================================
print("🧠 Applying DoRA modules...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# ================================================================
# 加载本地 C4 数据
# ================================================================
files = sorted(glob.glob(DATA_DIR + "*.json.gz"))[:10]
print("📚 Loading LOCAL C4 dataset...")
dataset = load_dataset(
    "json",
    data_files=files,
    split="train",
    cache_dir=CACHE_DIR,
)

print("📊 Full dataset size:", len(dataset))

# 取 1% 子集（等价 train[:1%]）
subset_size = int(len(dataset) * 1.0)
dataset = dataset.shuffle(seed=42).select(range(subset_size))
print("📦 Final training subset size:", len(dataset))


# ================================================================
# Tokenizer 映射
# ================================================================
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

train_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
train_dataset.set_format(type="torch")

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ================================================================
# PPL 计算函数
# ================================================================
@torch.no_grad()
def compute_ppl(model, dataset):
    model.eval()
    losses = []
    for batch in dataset:
        batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        if torch.isfinite(outputs.loss):
            losses.append(outputs.loss.item())
    return math.exp(sum(losses) / len(losses))


# ================================================================
# 训练主循环：每个 epoch 训练一次 + PPL 监控 + 模型保存
# ================================================================
TARGET_PPL = 10
best_ppl = float("inf")

for epoch in range(1, MAX_EPOCHS + 1):

    print(f"\n🚀 Epoch {epoch}/{MAX_EPOCHS}")
    check_disk(CACHE_DIR, min_gb=100)  # 每个 epoch 开头检查磁盘空间

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        save_strategy="steps",
        save_steps=1000,              # 每 N step 存一次（按你速度调）
        save_total_limit=2,           # 只保留最近 2 个，防止磁盘爆炸
        logging_steps=100,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=50,
        evaluation_strategy="no",
        bf16 = True,
        fp16=False,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=True)
    trainer.train()

    # 测试 PPL（dataset 前 200 条）
    ppl = compute_ppl(model, train_dataset.select(range(200)))
    print(f"📉 Current PPL: {ppl}")

    # 保存最优模型
    if ppl < best_ppl:
        best_ppl = ppl
        model.save_pretrained(OUTPUT_DIR)
        print(f"✅ Best model saved | PPL={best_ppl}")

    # 达成目标提前终止
    if ppl < TARGET_PPL:
        print("🎯 Target PPL reached. Stopping training.")
        break

print("\n🏁 Training finished")
print("🏆 Best PPL:", best_ppl)
print("📁 Saved at:", OUTPUT_DIR)
