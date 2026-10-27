#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_vscale_only_c4.py

- 只训练每层 self_attn.vscale_log（48个标量），其余参数全部冻结
- 数据：C4 streaming（可离线，依赖你已有 HF cache）
- 训练目标：降低 ppl（让每层能量自动校准），避免手动扫 EIGEN_VSCALE
- 保存：输出 HuggingFace safe_serialization (safetensors + index)

Example:
PYTHONPATH=/home/bai-project/jiaqizhang/EigenAttn/EigenAttn:$PYTHONPATH \
HF_HOME=/diskF/hf HF_DATASETS_CACHE=/diskF/jiaqi_hf_cache/datasets HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python3 train_vscale_only_c4.py \
  --model_dir /diskD/eigen_outputs_opt30b_r448/final_model \
  --tok_dir /diskE/opt-30b \
  --out_dir /diskD/eigen_outputs_opt30b_r448/final_model_vscale_trained \
  --seq_len 128 \
  --batch_size 1 \
  --train_steps 500 \
  --eval_every 50 \
  --eval_steps 20 \
  --lr 5e-2 \
  --init_vscale 0.03
"""

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Iterator, List

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

# 你的自定义模型（必须走它的 forward，才能用到 vscale_log）
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn


# -----------------------------
# Streaming dataset -> fixed-length token blocks
# -----------------------------
class C4TokenBlocks(IterableDataset):
    def __init__(self, tok, seq_len: int, split: str = "train", seed: int = 1234):
        super().__init__()
        self.tok = tok
        self.seq_len = seq_len
        self.split = split
        self.seed = seed

        # 这里用 c4/en，和你常用的 c4-en shard cache 对齐（如果你之前就是这样用的）
        self.ds = load_dataset(
            "c4",
            "en",
            split=split,
            streaming=True,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # 简单文本拼接成 token buffer，再切 block
        buffer: List[int] = []
        for ex in self.ds:
            text = ex.get("text", "")
            if not text:
                continue
            ids = self.tok(text, add_special_tokens=False).input_ids
            if not ids:
                continue
            buffer.extend(ids)

            # 不断吐出 seq_len+1（用于 labels shift）
            while len(buffer) >= self.seq_len + 1:
                block = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                yield {"input_ids": torch.tensor(block, dtype=torch.long)}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # batch_size=1 时最简单；如果你想 >1，也可 pad（这里先不 pad，要求严格同长）
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)  # (B, L+1)
    return {"input_ids": input_ids}


# -----------------------------
# Loss: standard causal LM
# -----------------------------
def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: (B, L, V), labels: (B, L)
    vocab = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), reduction="mean")


@torch.no_grad()
def evaluate_ppl(model, dl, device, steps: int) -> float:
    model.eval()
    losses = []
    it = iter(dl)
    for _ in range(steps):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)  # (B, L+1)
        x = input_ids[:, :-1].contiguous()
        y = input_ids[:, 1:].contiguous()

        out = model(input_ids=x, attention_mask=None, use_cache=False)
        logits = out.logits
        loss = causal_lm_loss(logits, y)
        losses.append(loss.item())

    mean_loss = float(sum(losses) / max(1, len(losses)))
    ppl = float(math.exp(min(50.0, mean_loss)))  # 防止 overflow
    model.train()
    return mean_loss, ppl


def get_vscale_stats(model) -> Dict[str, float]:
    vals = []
    for n, p in model.named_parameters():
        if n.endswith("self_attn.vscale_log"):
            vals.append(float(p.detach().exp().cpu().item()))
    if not vals:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {"min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--tok_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--train_steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--init_vscale", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This script is intended for CUDA GPU (OPT-30B).")

    # 重要：初始化用 env（你 __init__ 里如果用了 EIGEN_VSCALE_INIT，就会在 missing keys 时生效）
    os.environ["EIGEN_VSCALE_INIT"] = str(args.init_vscale)

    # 训练时不要再用全局 env 去乘（避免把可训练参数“盖住”）
    # 你代码里如果还乘了 os.getenv("EIGEN_VSCALE")，这里固定为 1.0
    os.environ["EIGEN_VSCALE"] = "1.0"

    # 其余 debug 环境变量确保关闭
    os.environ["EIGEN_BYPASS_ATTN"] = "0"
    os.environ["EIGEN_BYPASS_BEFORE_UP"] = "0"
    os.environ["EIGEN_BYPASS_PROBS"] = "0"

    # dtype
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print(f"[Load] tokenizer: {args.tok_dir}")
    tok = AutoTokenizer.from_pretrained(args.tok_dir, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[Load] model: {args.model_dir}")
    model = OPTForCausalLM_EigenAttn.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.to(device)
    model.train()

    # Freeze all params
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze only vscale_log
    trainable = []
    for n, p in model.named_parameters():
        if n.endswith("self_attn.vscale_log"):
            p.requires_grad_(True)
            trainable.append(p)

    if len(trainable) == 0:
        raise RuntimeError("No parameters matched *.self_attn.vscale_log. Did you add vscale_log to OPTEigenAttention?")

    print(f"[Params] trainable vscale_log = {len(trainable)} (expected 48 for OPT-30B)")
    print("[vscale init stats]", get_vscale_stats(model))

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    # data
    ds = C4TokenBlocks(tok, seq_len=args.seq_len, split="train", seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    # quick baseline eval
    eval_loss, eval_ppl = evaluate_ppl(model, dl, device, steps=args.eval_steps)
    print(f"[Eval@start] loss={eval_loss:.6f} ppl={eval_ppl:.4f} vscale={get_vscale_stats(model)}")

    it = iter(dl)
    t0 = time.time()

    for step in range(1, args.train_steps + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)  # (B, L+1)
        x = input_ids[:, :-1].contiguous()
        y = input_ids[:, 1:].contiguous()

        out = model(input_ids=x, attention_mask=None, use_cache=False)
        logits = out.logits
        loss = causal_lm_loss(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # 只训练标量，梯度通常很稳；如果你想更保险可以加 clip
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

        opt.step()

        if step % 10 == 0:
            dt = time.time() - t0
            vs = get_vscale_stats(model)
            print(f"[Train] step={step:5d} loss={loss.item():.6f}  time={dt:.1f}s  vscale(min/mean/max)={vs}")
            t0 = time.time()

        if step % args.eval_every == 0:
            eval_loss, eval_ppl = evaluate_ppl(model, dl, device, steps=args.eval_steps)
            vs = get_vscale_stats(model)
            print(f"[Eval] step={step:5d} loss={eval_loss:.6f} ppl={eval_ppl:.4f} vscale={vs}")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[Save] -> {args.out_dir}")
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)

    # Also dump per-layer vscale values for inspection
    vscales = {}
    for n, p in model.named_parameters():
        if n.endswith("self_attn.vscale_log"):
            vscales[n] = float(p.detach().exp().cpu().item())
    with open(os.path.join(args.out_dir, "vscale_layerwise.json"), "w") as f:
        import json
        json.dump(vscales, f, indent=2)

    print("[Done] saved model + vscale_layerwise.json")


if __name__ == "__main__":
    main()
