#!/usr/bin/env python3
"""
finetune_extreme_ppl.py

🚀 极限降低 PPL 专用脚本
目标：不断微调直到 PPL < 10 或达到最大 epoch

特性：
- 使用 C4 数据集（更稳定）
- 支持多 GPU：cuda:1, cuda:2
- 自动 Early Stop
- 每个 epoch 输出 PPL
- 自动保存最优模型
- 适配 EigenAttention + DoRA + OPT-125M

使用模型路径：
./eigen_outputs_1.0x/final_model
"""

import os
import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import huggingface_hub
from transformers.utils import logging

print("Visible GPUs:", torch.cuda.device_count())
print("Using device:", torch.cuda.current_device())

print("==== HuggingFace Cache Paths ====")
print("HF_HOME:", os.environ.get("HF_HOME"))
print("cache_home:", huggingface_hub.constants.HF_HUB_CACHE)
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("HUGGINGFACE_HUB_CACHE:", os.environ.get("HUGGINGFACE_HUB_CACHE"))
print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))
print("=================================")

logging.set_verbosity_info()

# ============================
# 环境设置
# ============================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./eigen_outputs_1.0x/final_model"
OUTPUT_DIR = "./dora_extreme_ppl_model"
TARGET_PPL = 10
MAX_EPOCHS = 30

# ============================
# 加载模型与分词器
# ============================
print("📦 Loading model...")
#tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
#tokenizer = AutoTokenizer.from_pretrained(
#    MODEL_PATH,
#    use_fast=True,
#    local_files_only=True,
#    trust_remote_code=True
#)
tokenizer_path = "/home/bai-project/jiaqizhang/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.to(device)

# ============================
# DoRA 配置
# ============================
print("🧠 Applying DoRA...")
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

# ============================
# 加载 C4 数据集
# ============================
print("📚 Loading C4 dataset...")
#dataset = load_dataset("allenai/c4", "en", split="train[:1%]")
#dataset = load_dataset(
#    "allenai/c4",
#    "en",
#    split="train[:1%]",
#    cache_dir="/media/bai3090/Seagate Portable Drive/cache/huggingface"
#)
#dataset = load_dataset(
#    "wikitext",
#    "wikitext-2-raw-v1",
#    split="train",
#    cache_dir="/media/bai3090/Seagate Portable Drive/cache/huggingface"
#)

from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "train": [
            "/home/bai-project/jiaqizhang/wikitext/train-00000-of-00002.parquet",
            "/home/bai-project/jiaqizhang/wikitext/train-00001-of-00002.parquet",
        ],
        "validation": "/home/bai-project/jiaqizhang/wikitext/validation-00000-of-00001.parquet",
        "test": "/home/bai-project/jiaqizhang/wikitext/test-00000-of-00001.parquet",
    }
)

dataset = dataset["train"]

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

#train_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
train_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
train_dataset.set_format(type="torch")

# ============================
# Data Collator
# ============================
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ============================
# PPL 计算函数
# ============================
@torch.no_grad()
def compute_ppl(model, dataloader):
    model.eval()
    losses = []
    for batch in dataloader:
        batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        if torch.isfinite(loss):
            losses.append(loss.item())
    if len(losses) == 0:
        return float('inf')
    return math.exp(sum(losses) / len(losses))

# ============================
# 训练循环（带 Early Stop）
# ============================
best_ppl = float("inf")

for epoch in range(1, MAX_EPOCHS + 1):
    print(f"\n🚀 Epoch {epoch}/{MAX_EPOCHS}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=192,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        save_strategy="no",
        evaluation_strategy="no",
        learning_rate=3e-4,
        fp16 = False,
        bf16 = True,
        dataloader_num_workers=4,  # 加速数据加载
        logging_steps=500,         # 降低打印开销
        ddp_find_unused_parameters=False,  # 必须关闭
        optim="adamw_torch_fused",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    # 计算 PPL
    ppl = compute_ppl(model, train_dataset.select(range(200)))
    print(f"📉 Current PPL: {ppl}")

    if ppl < best_ppl:
        best_ppl = ppl
        model.save_pretrained(OUTPUT_DIR)
        print(f"✅ New best model saved | PPL={best_ppl}")

    if ppl < TARGET_PPL:
        print("🎯 Target PPL achieved! Stop training.")
        break

print("\n🏁 Training completed")
print("🏆 Best PPL achieved:", best_ppl)
print("📁 Model saved at:", OUTPUT_DIR)

