#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_vscale_opt30b_v2.py

Purpose:
- Train ONLY per-layer vscale_log (48 scalars for OPT-30B) for compressed EigenAttn model.
- Minimal custom training loop (no HF Trainer) -> much less overhead, faster iteration.
- Evaluate PPL every eval_steps (default 50) on fixed eval_blocks.pt
- Save ONLY the latest checkpoint (overwrite) every save_steps (default 50)

Key policy:
- Offline env set inside the script
- Force EIGEN_VSCALE=1.0 during training to avoid env constant overriding learnable vscale
- Use EIGEN_VSCALE_INIT for missing vscale_log init in model __init__
"""

import os
import re
import csv
import json
import math
import glob
import time
import argparse
import logging
import importlib
from typing import Iterator, Dict, Any, List, Optional, Tuple

# -------------------------
# Hard-set offline env
# -------------------------
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

# -------------------------
# Logging
# -------------------------
def setup_logger():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

def is_rank0() -> bool:
    return True  # single-process only

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def append_csv(path: str, row: Dict[str, Any]):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

# -------------------------
# Data
# -------------------------
class TokenBlockIterableDataset(IterableDataset):
    def __init__(self, hf_iterable, tokenizer, block_size: int, text_field: str = "text"):
        super().__init__()
        self.hf_iterable = hf_iterable
        self.tok = tokenizer
        self.block_size = int(block_size)
        self.text_field = text_field
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        eos = self.tok.eos_token_id
        buf: List[int] = []
        for ex in self.hf_iterable:
            text = None
            if isinstance(ex, dict):
                text = ex.get(self.text_field, None)
                if text is None:
                    for _, v in ex.items():
                        if isinstance(v, str):
                            text = v
                            break
            elif isinstance(ex, str):
                text = ex
            if not text:
                continue

            ids = self.tok(text, add_special_tokens=False).input_ids
            if eos is not None:
                ids = ids + [eos]
            buf.extend(ids)

            while len(buf) >= self.block_size:
                block = buf[: self.block_size]
                buf = buf[self.block_size :]

                input_ids = torch.tensor(block, dtype=torch.long)
                attn_mask = torch.ones_like(input_ids, dtype=torch.long)
                labels = input_ids.clone()
                yield {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

def _collect_local_files(data_dir: str) -> Tuple[str, List[str]]:
    patterns = [
        "**/*.jsonl.gz", "**/*.json.gz",
        "**/*.jsonl", "**/*.json",
        "**/*.parquet",
        "**/*.txt.gz", "**/*.txt",
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"[Dataset] No files found under {data_dir}")

    pick = files[0].lower()
    if pick.endswith(".parquet"):
        return "parquet", files
    if any(pick.endswith(suf) for suf in [".json", ".jsonl", ".json.gz", ".jsonl.gz"]):
        return "json", files
    return "text", files

def load_streaming_local_dataset_strict(data_dir: str):
    loader, files = _collect_local_files(data_dir)
    logging.info(f"[Dataset] STRICT local loader={loader} files={len(files)} example={files[0]}")
    if loader == "parquet":
        return load_dataset("parquet", data_files=files, split="train", streaming=True)
    if loader == "json":
        return load_dataset("json", data_files=files, split="train", streaming=True)
    return load_dataset("text", data_files=files, split="train", streaming=True)

# -------------------------
# Model utils
# -------------------------
def get_underlying_lm(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model

def build_device_map_single():
    return {"": 0}

def enable_vscale_only_training(model) -> int:
    for p in model.parameters():
        p.requires_grad_(False)
    cnt = 0
    for n, p in model.named_parameters():
        if n.endswith("self_attn.vscale_log"):
            p.requires_grad_(True)
            cnt += 1
    return cnt
def force_init_vscale(model, vscale_init: float, param_is_log: bool):
    with torch.no_grad():
        init_val = math.log(max(vscale_init, 1e-12)) if param_is_log else float(vscale_init)
        for n, p in model.named_parameters():
            if n.endswith("self_attn.vscale_log"):
                p.fill_(init_val)

def iter_vscale_params(model):
    for n, p in model.named_parameters():
        if n.endswith("self_attn.vscale_log") and p.requires_grad:
            yield n, p

def vscale_stats(model) -> Dict[str, float]:
    vals = []
    grads = []
    with torch.no_grad():
        for _, p in iter_vscale_params(model):
            vals.append(float(p.detach().float().mean().cpu().item()))
            if p.grad is None:
                grads.append(0.0)
            else:
                grads.append(float(p.grad.detach().float().abs().mean().cpu().item()))
    if not vals:
        return {"mean_v": float("nan"), "mean_g": float("nan"), "max_g": float("nan")}
    return {
        "mean_v": sum(vals) / len(vals),
        "mean_g": sum(grads) / len(grads),
        "max_g": max(grads) if grads else 0.0,
    }

def clamp_vscale_inplace(model, param_is_log: bool, gmin: float, gmax: float):
    if gmax <= 0:
        return
    with torch.no_grad():
        if param_is_log:
            lo = math.log(max(gmin, 1e-12))
            hi = math.log(max(gmax, 1e-12))
            for _, p in iter_vscale_params(model):
                p.clamp_(lo, hi)
        else:
            for _, p in iter_vscale_params(model):
                p.clamp_(gmin, gmax)

def set_env_for_training(vscale_init: float, force_env_vscale_one: bool):
    # init missing keys in model init
    os.environ["EIGEN_VSCALE_INIT"] = str(vscale_init)
    # IMPORTANT: during training, do NOT let env constant path distort the learnable vscale
    if force_env_vscale_one:
        os.environ["EIGEN_VSCALE"] = "1.0"
    # keep deterministic behavior
    os.environ.setdefault("EIGEN_SCALE_ORIG", "0")
    os.environ.setdefault("EIGEN_BYPASS_ATTN", "0")

def load_model(args, dtype: torch.dtype):
    cfg = AutoConfig.from_pretrained(args.model_path)
    device_map = build_device_map_single()

    low_rank_config = json.load(open(args.low_rank_json, "r"))
    mod_path, cls_name = args.import_eigen_class.split(":")
    module = importlib.import_module(mod_path)
    EigenCls = getattr(module, cls_name)

    model = EigenCls.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=cfg,
        low_rank_config=low_rank_config,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    return model, cfg

# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def eval_ppl(model, eval_blocks: List[Dict[str, torch.Tensor]], device: torch.device, amp_dtype: torch.dtype) -> float:
    model.eval()
    losses = []
    use_amp = (device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16))
    for b in eval_blocks:
        batch = {k: v.unsqueeze(0).to(device) for k, v in b.items()}
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**batch)
        else:
            out = model(**batch)
        loss = out.loss.detach().float().cpu().item()
        losses.append(loss)
    mean_loss = float(sum(losses) / max(1, len(losses)))
    ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    return mean_loss, ppl

# -------------------------
# Checkpoint (overwrite)
# -------------------------
def save_latest_checkpoint(model, output_dir: str, step: int, param_is_log: bool):
    ensure_dir(output_dir)
    # Save only vscale params (tiny, fast)
    state = {"step": int(step), "param_is_log": bool(param_is_log), "vscale": {}}
    for n, p in model.named_parameters():
        if n.endswith("self_attn.vscale_log"):
            state["vscale"][n] = p.detach().float().cpu()
    tmp = os.path.join(output_dir, "ckpt_latest.pt.tmp")
    path = os.path.join(output_dir, "ckpt_latest.pt")
    torch.save(state, tmp)
    os.replace(tmp, path)

def load_latest_checkpoint_if_any(model, output_dir: str) -> int:
    path = os.path.join(output_dir, "ckpt_latest.pt")
    if not os.path.exists(path):
        return 0
    obj = torch.load(path, map_location="cpu")
    v = obj.get("vscale", {})
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in v and n.endswith("self_attn.vscale_log"):
                p.copy_(v[n].to(p.device, dtype=p.dtype))
    step = int(obj.get("step", 0))
    logging.info(f"[Resume] loaded {path} step={step}")
    return step

# -------------------------
# Args
# -------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--low_rank_json", type=str, required=True)
    p.add_argument("--import_eigen_class", type=str,
                   default="models.modelling_opt_eigen_attn:OPTForCausalLM_EigenAttn")

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--shuffle_buffer", type=int, default=10000)

    p.add_argument("--eval_blocks_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # training
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0)

    # eval/save/log
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=10)

    # dtype
    p.add_argument("--bf16", type=str2bool, default=True, nargs="?", const=True)
    p.add_argument("--fp16", type=str2bool, default=False, nargs="?", const=True)

    # vscale-only
    p.add_argument("--vscale_init", type=float, default=0.03)
    p.add_argument("--vscale_param_is_log", type=str2bool, default=True, nargs="?", const=True)
    p.add_argument("--force_env_vscale_one", type=str2bool, default=True, nargs="?", const=True)

    # clamp (VERY recommended)
    p.add_argument("--vscale_clamp_min", type=float, default=0.005)
    p.add_argument("--vscale_clamp_max", type=float, default=0.2)

    # resume
    p.add_argument("--resume", type=str2bool, default=True, nargs="?", const=True)

    return p.parse_args()

# -------------------------
# Main
# -------------------------
def main():
    setup_logger()
    args = parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("fp16 and bf16 cannot both be True.")

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    amp_dtype = dtype  # autocast dtype

    ensure_dir(args.output_dir)

    # Set env policy for training
    set_env_for_training(vscale_init=args.vscale_init, force_env_vscale_one=bool(args.force_env_vscale_one))
    logging.info(f"[ENV] EIGEN_VSCALE_INIT={os.environ.get('EIGEN_VSCALE_INIT')} EIGEN_VSCALE={os.environ.get('EIGEN_VSCALE')}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Load model
    model, cfg = load_model(args, dtype=dtype)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Freeze -> only vscale trainable
    cnt = enable_vscale_only_training(model)
    logging.info(f"[VSCALE] trainable_params={cnt} param_is_log={bool(args.vscale_param_is_log)} init={args.vscale_init}")

    # Resume vscale checkpoint (optional)
    start_step = 0
    if args.resume:
        start_step = load_latest_checkpoint_if_any(model, args.output_dir)

    # ✅ 如果没有 resume（新跑），就强制把 vscale_log 设为你指定的 init
    if start_step == 0:
        force_init_vscale(model, args.vscale_init, bool(args.vscale_param_is_log))
        logging.info("[VSCALE] force_init applied to all layers.")


    # Load eval blocks
    eval_blocks = torch.load(args.eval_blocks_path, map_location="cpu")
    logging.info(f"[Eval] loaded eval_blocks={len(eval_blocks)} from {args.eval_blocks_path}")

    # Build streaming train dataset
    raw = load_streaming_local_dataset_strict(args.data_dir)
    raw = raw.shuffle(buffer_size=args.shuffle_buffer, seed=2)
    train_iterable = TokenBlockIterableDataset(raw, tok, block_size=args.block_size, text_field=args.text_field)

    dl = DataLoader(train_iterable, batch_size=args.train_batch_size, num_workers=0)

    # Optimizer (AdamW is enough for 48 scalars)
    # Note: use fused adamw if available (torch 2.x) for a bit faster
    params = [p for _, p in iter_vscale_params(model)]
    opt = torch.optim.AdamW(params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    # Warmup scheduler (optional; default 0)
    def lr_for_step(s: int):
        if args.warmup_steps <= 0:
            return args.learning_rate
        if s < args.warmup_steps:
            return args.learning_rate * (s / max(1, args.warmup_steps))
        return args.learning_rate

    ppl_csv = os.path.join(args.output_dir, "ppl_log.csv")

    # Initial eval at start_step (only if start_step==0, to match your logs)
    if start_step == 0:
        ev_loss, ppl = eval_ppl(model, eval_blocks, device=device, amp_dtype=amp_dtype)
        append_csv(ppl_csv, {"step": 0, "eval_loss": ev_loss, "ppl": ppl, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        logging.info(f"[Eval] step=0 eval_loss={ev_loss:.6f} ppl={ppl:.3f}")

    model.train()
    global_step = start_step
    opt.zero_grad(set_to_none=True)

    t0 = time.time()
    for batch in dl:
        # stop
        if global_step >= args.max_steps:
            break

        # LR schedule
        lr = lr_for_step(global_step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # move
        batch = {k: v.to(device) for k, v in batch.items()}

        use_amp = (device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16))
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**batch)
                loss = out.loss / max(1, args.grad_accum)
        else:
            out = model(**batch)
            loss = out.loss / max(1, args.grad_accum)

        loss.backward()

        if (global_step + 1) % args.grad_accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

            # clamp after update
            clamp_vscale_inplace(
                model,
                param_is_log=bool(args.vscale_param_is_log),
                gmin=float(args.vscale_clamp_min),
                gmax=float(args.vscale_clamp_max),
            )

        # logging
        if (global_step + 1) % args.logging_steps == 0:
            st = vscale_stats(model)
            dt = time.time() - t0
            logging.info(
                f"[Train] step={global_step+1} loss={float(loss.detach().float().cpu().item()):.6f} "
                f"lr={lr:.3g} v_mean={st['mean_v']:.6g} g_mean={st['mean_g']:.6g} g_max={st['max_g']:.6g} "
                f"elapsed={dt/60:.1f}min"
            )

        # eval
        if (global_step + 1) % args.eval_steps == 0:
            ev_loss, ppl = eval_ppl(model, eval_blocks, device=device, amp_dtype=amp_dtype)
            append_csv(ppl_csv, {"step": global_step + 1, "eval_loss": ev_loss, "ppl": ppl, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            logging.info(f"[Eval] step={global_step+1} eval_loss={ev_loss:.6f} ppl={ppl:.3f}")

        # save (overwrite, only latest)
        if (global_step + 1) % args.save_steps == 0:
            save_latest_checkpoint(model, args.output_dir, step=global_step + 1, param_is_log=bool(args.vscale_param_is_log))
            logging.info(f"[Save] ckpt_latest.pt updated at step={global_step+1}")

        global_step += 1

    # final save
    save_latest_checkpoint(model, args.output_dir, step=global_step, param_is_log=bool(args.vscale_param_is_log))
    logging.info(f"[Done] reached step={global_step}, saved ckpt_latest.pt")

if __name__ == "__main__":
    main()
