#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_vscale_opt30b.py

- OPT-30B EigenAttention compressed 模型：只训练每层 self_attn.vscale_log（48 scalars）
- 完全离线（HF_DATASETS_OFFLINE / TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE）
- 单 GPU（不 torchrun）
- 固定 eval_blocks（可选：首次自动生成并保存）
- 每 50 step evaluate 一次，并记录 ppl_log.csv
- 只保留最新 checkpoint（save_total_limit=1）
- max_steps=20000（可参数覆盖）
"""

import os
# -------------------------
# Hard-set environment here
# -------------------------
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 关键：避免 env 常量覆盖 learnable vscale_log（训练时必须禁用 env vscale）
# 如果你 forward 内部会读 EIGEN_VSCALE，这里强制清掉/置空
os.environ.pop("EIGEN_VSCALE", None)
os.environ.pop("EIGEN_VSCALE_INIT", None)

import re
import csv
import json
import math
import glob
import time
import argparse
import logging
import importlib
from typing import Iterator, Dict, Any, Optional, List, Tuple

import torch
from torch.utils.data import IterableDataset, Dataset

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.trainer_callback import TrainerCallback


# =========================
# Logging
# =========================

def setup_logger():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

def is_rank0() -> bool:
    return True  # single-process only

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def append_ppl_csv(csv_path: str, row: Dict[str, Any]):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def write_text(path: str, s: str):
    with open(path, "w") as f:
        f.write(s)

def count_trainable_params(model) -> str:
    tot = 0
    trn = 0
    for p in model.parameters():
        n = p.numel()
        tot += n
        if p.requires_grad:
            trn += n
    pct = (trn / tot * 100.0) if tot > 0 else 0.0
    return f"trainable={trn:,} / total={tot:,} ({pct:.6f}%)"


# =========================
# Dataset: local streaming -> token blocks
# =========================

class TokenBlockIterableDataset(IterableDataset):
    def __init__(self, hf_iterable, tokenizer, block_size: int, text_field: str = "text"):
        super().__init__()
        self.hf_iterable = hf_iterable
        self.tok = tokenizer
        self.block_size = block_size
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


class PreloadedEvalDataset(Dataset):
    def __init__(self, blocks: List[Dict[str, torch.Tensor]]):
        self.data = blocks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


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


def build_train_and_eval(args, tok):
    """
    - eval_blocks_path 如果存在：直接加载 eval blocks（固定评估集）
    - 否则：从 data_dir 里 streaming 取前 eval_num_blocks*2 blocks 生成并保存
    """
    ensure_dir(args.output_dir)

    if args.eval_blocks_path and os.path.exists(args.eval_blocks_path):
        blocks = torch.load(args.eval_blocks_path, map_location="cpu")
        eval_ds = PreloadedEvalDataset(blocks)
        logging.info(f"[Data] Loaded eval_blocks from {args.eval_blocks_path} len={len(eval_ds)}")
        raw = load_streaming_local_dataset_strict(args.data_dir).shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        train_raw = raw.skip(len(eval_ds) * 2)
        train_ds = TokenBlockIterableDataset(train_raw, tok, block_size=args.block_size, text_field=args.text_field)
        return train_ds, eval_ds

    raw = load_streaming_local_dataset_strict(args.data_dir).shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    # build eval blocks
    eval_raw = raw.take(args.eval_num_blocks * 2)
    eval_iter = TokenBlockIterableDataset(eval_raw, tok, block_size=args.block_size, text_field=args.text_field)

    blocks = []
    it = iter(eval_iter)
    for _ in range(args.eval_num_blocks):
        try:
            blocks.append(next(it))
        except StopIteration:
            break
    eval_ds = PreloadedEvalDataset(blocks)
    logging.info(f"[Data] Built eval_blocks len={len(eval_ds)}")

    if args.save_eval_blocks:
        path = os.path.join(args.output_dir, "eval_blocks.pt")
        torch.save(blocks, path)
        logging.info(f"[Data] Saved eval_blocks -> {path}")

    train_raw = raw.skip(len(eval_ds) * 2)
    train_ds = TokenBlockIterableDataset(train_raw, tok, block_size=args.block_size, text_field=args.text_field)
    return train_ds, eval_ds


# =========================
# Model loading (EigenAttention class)
# =========================

def load_eigen_model(args, dtype: torch.dtype):
    cfg = AutoConfig.from_pretrained(args.model_path)

    # single-gpu device_map
    device_map = {"": 0}

    # import class
    mod_path, cls_name = args.import_eigen_class.split(":")
    module = importlib.import_module(mod_path)
    EigenCls = getattr(module, cls_name)

    low_rank_config = json.load(open(args.low_rank_json, "r"))

    fn = lambda **kw: EigenCls.from_pretrained(**kw)
    try:
        model, info = fn(
            pretrained_model_name_or_path=args.model_path,
            config=cfg,
            low_rank_config=low_rank_config,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
    except TypeError:
        model = fn(
            pretrained_model_name_or_path=args.model_path,
            config=cfg,
            low_rank_config=low_rank_config,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        info = None

    # HF Trainer single-process model-parallel hints
    model.is_parallelizable = True
    model.model_parallel = True
    if hasattr(model, "model") and model.model is not None:
        model.model.is_parallelizable = True
        model.model.model_parallel = True

    # memory
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.eval()
    return model, info


# =========================
# vscale-only controls
# =========================

def enable_vscale_only_training(model) -> int:
    for p in model.parameters():
        p.requires_grad_(False)
    cnt = 0
    for n, p in model.named_parameters():
        if n.endswith("self_attn.vscale_log"):
            p.requires_grad_(True)
            cnt += 1
    return cnt

def init_vscale_constant(model, val: float):
    """
    强制把每层 vscale_log 填成同一个值（不依赖 missing_keys/env）
    注意：这里假设 vscale_log 存的就是 gamma 本身。
    如果你 forward 用 exp(vscale_log)，那你要把 val 改成 log(val)（见 args.vscale_param_is_log）。
    """
    lm = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        lm = model.base_model.model

    layers = lm.model.decoder.layers
    with torch.no_grad():
        for layer in layers:
            layer.self_attn.vscale_log.data.fill_(val)

def init_vscale_constant_logparam(model, gamma: float):
    lm = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        lm = model.base_model.model
    layers = lm.model.decoder.layers
    v = math.log(max(float(gamma), 1e-12))
    with torch.no_grad():
        for layer in layers:
            layer.self_attn.vscale_log.data.fill_(v)


# =========================
# Callbacks
# =========================

class PPLLogCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.csv_path = os.path.join(output_dir, "ppl_log.csv")

    def log_eval(self, state, eval_loss: float):
        ppl = math.exp(eval_loss) if eval_loss < 50 else float("inf")
        row = {
            "step": int(state.global_step),
            "eval_loss": float(eval_loss),
            "ppl": float(ppl),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_ppl_csv(self.csv_path, row)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not is_rank0():
            return
        metrics = metrics or {}
        ev_loss = metrics.get("eval_loss", None)
        if ev_loss is None:
            return
        self.log_eval(state, float(ev_loss))

class EvalAtStartCallback(TrainerCallback):
    """
    训练开始前先 evaluate 一次，写入 step=0 的 ppl_log.csv（便于对齐你 sweep 的 step=0）
    """
    def __init__(self, ppl_cb: PPLLogCallback):
        self.ppl_cb = ppl_cb
        self._done = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self._done:
            return
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return
        metrics = trainer.evaluate()
        ev_loss = metrics.get("eval_loss", None)
        if ev_loss is not None:
            logging.info(f"[Eval@Start] eval_loss={ev_loss} ppl={math.exp(ev_loss) if ev_loss < 50 else float('inf')}")
            self.ppl_cb.log_eval(state, float(ev_loss))
        self._done = True

class VSCALEDebugCallback(TrainerCallback):
    def __init__(self, every: int = 10):
        self.every = every

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if not is_rank0() or model is None:
            return
        if state.global_step % self.every != 0:
            return
        vals = []
        grads = []
        for n, p in model.named_parameters():
            if n.endswith("self_attn.vscale_log"):
                vals.append(float(p.detach().float().mean().cpu().item()))
                if p.grad is None:
                    grads.append(0.0)
                else:
                    grads.append(float(p.grad.detach().float().abs().mean().cpu().item()))
        if vals:
            logging.info(
                f"[VSCALE-DBG] step={state.global_step} "
                f"mean(v)={sum(vals)/len(vals):.6g} mean(|g|)={sum(grads)/len(grads):.6g} max(|g|)={max(grads):.6g}"
            )

class NoMoveTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        return model


# =========================
# Args
# =========================

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

    # paths
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--low_rank_json", type=str, required=True)
    p.add_argument("--import_eigen_class", type=str,
                   default="models.modelling_opt_eigen_attn:OPTForCausalLM_EigenAttn")

    # data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--shuffle_buffer", type=int, default=10000)
    p.add_argument("--eval_num_blocks", type=int, default=200)
    p.add_argument("--eval_blocks_path", type=str, default=None)
    p.add_argument("--save_eval_blocks", type=str2bool, default=True, nargs="?", const=True)

    # output
    p.add_argument("--output_dir", type=str, required=True)

    # train
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-3)   # vscale-only 推荐 1e-3 ~ 1e-2
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--save_total_limit", type=int, default=1)
    # eval/save cadence
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=200)         # 你可以改更大/更小
    p.add_argument("--logging_steps", type=int, default=10)
    #p.add_argument("--save_total_limit", type=int, default=1)
    # dtype
    p.add_argument("--bf16", type=str2bool, default=True, nargs="?", const=True)
    p.add_argument("--fp16", type=str2bool, default=False, nargs="?", const=True)

    # misc
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--gradient_checkpointing", type=str2bool, default=True, nargs="?", const=True)

    # vscale-only
    p.add_argument("--train_vscale_only", type=str2bool, default=True, nargs="?", const=True)
    p.add_argument("--vscale_init", type=float, default=0.03)
    p.add_argument("--vscale_param_is_log", type=str2bool, default=False, nargs="?", const=True,
                   help="若 forward 用 exp(vscale_log)，设为 True，这样 vscale_init 会按 log(gamma) 写入。")

    # modes
    p.add_argument("--eval_only", type=str2bool, default=False, nargs="?", const=True)

    return p.parse_args()


# =========================
# Main
# =========================

def main():
    setup_logger()
    args = parse_args()
    set_seed(args.seed)

    # hard check: single process only
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError("This script is single-process only. Run without torchrun.")

    ensure_dir(args.output_dir)

    if args.fp16 and args.bf16:
        raise ValueError("fp16 and bf16 cannot both be True.")
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model, loading_info = load_eigen_model(args, dtype=dtype)

    if loading_info is not None:
        mk = loading_info.get("missing_keys", [])
        logging.info(f"[LoadInfo] missing_keys={len(mk)} (sample={mk[:5]})")

    # vscale-only: init + freeze/unfreeze
    if args.train_vscale_only:
        if args.vscale_param_is_log:
            init_vscale_constant_logparam(model, gamma=float(args.vscale_init))
        else:
            init_vscale_constant(model, val=float(args.vscale_init))

        cnt = enable_vscale_only_training(model)
        logging.info(f"[VSCALE] init={args.vscale_init} param_is_log={args.vscale_param_is_log} trainable_params={cnt}")
        logging.info(f"[Params] {count_trainable_params(model)}")

    train_ds, eval_ds = build_train_and_eval(args, tok)

    # TrainingArguments
    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,

        evaluation_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        #save_total_limit=1,                 # ✅ only keep latest checkpoint

        logging_steps=args.logging_steps,
        report_to="none",

        fp16=args.fp16,
        bf16=args.bf16,

        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",

        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataloader_pin_memory=False,

        load_best_model_at_end=False,       # 你要“只保留最新ckpt”，这里就别再best_model逻辑了
    )

    ppl_cb = PPLLogCallback(args.output_dir)
    cbs = [
        ppl_cb,
        EvalAtStartCallback(ppl_cb),
        VSCALEDebugCallback(every=args.logging_steps),
    ]

    trainer = NoMoveTrainer(
       # save_total_limit=args.save_total_limit,
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=default_data_collator,
        callbacks=cbs,
    )

    if args.eval_only:
        metrics = trainer.evaluate()
        ev_loss = metrics.get("eval_loss", None)
        ppl = math.exp(ev_loss) if ev_loss is not None and ev_loss < 50 else None
        logging.info(f"[EvalOnly] eval_loss={ev_loss} ppl={ppl}")
        return

    # train
    trainer.train()

    # final eval
    metrics = trainer.evaluate()
    ev_loss = metrics.get("eval_loss", None)
    ppl = math.exp(ev_loss) if ev_loss is not None and ev_loss < 50 else None
    logging.info(f"[FinalEval] eval_loss={ev_loss} ppl={ppl}")

    # export final model (not best, just final state)
    final_dir = os.path.join(args.output_dir, "final_model")
    ensure_dir(final_dir)
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    write_text(os.path.join(args.output_dir, "DONE.txt"), "done\n")
    logging.info(f"[Export] saved -> {final_dir}")


if __name__ == "__main__":
    main()
