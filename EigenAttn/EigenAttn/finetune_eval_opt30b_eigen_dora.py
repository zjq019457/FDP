#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finetune_eval_opt30b_eigen_dora.py (single-GPU friendly + debug checks)

[MODIFIED]
- Added --train_vscale_only mode:
  Freeze all params and only train each layer self_attn.vscale_log (48 scalars for OPT-30B)
- Added --vscale_init, --vscale_lr, --force_env_vscale_one
- If train_vscale_only:
  - disables PEFT (to avoid training adapters)
  - sets EIGEN_VSCALE_INIT (init missing keys)
  - forces EIGEN_VSCALE=1.0 (prevent env constant overriding learnable vscale)
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
from typing import Iterator, Dict, Any, Optional, List, Tuple

# ✅ Force offline
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from torch.utils.data import IterableDataset, Dataset

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# === Optional: PEFT ===
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    _PEFT_OK = True
except Exception:
    _PEFT_OK = False


# =========================
# Utils
# =========================

def setup_logger():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

def is_rank0() -> bool:
    r = int(os.environ.get("RANK", "0"))
    return r == 0

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

def find_checkpoints(output_dir: str) -> List[str]:
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    def step(p):
        m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return sorted(ckpts, key=step)

def read_best_checkpoint_from_state(output_dir: str) -> Optional[str]:
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return None
    try:
        j = json.load(open(state_path, "r"))
        best = j.get("best_model_checkpoint", None)
        if best and os.path.isdir(best):
            return best
    except Exception:
        return None
    return None

def pick_resume_checkpoint(output_dir: str, policy: str) -> Optional[str]:
    if policy == "none":
        return None
    ckpts = find_checkpoints(output_dir)
    latest = ckpts[-1] if ckpts else None
    if policy == "latest":
        return latest
    if policy == "best":
        best = read_best_checkpoint_from_state(output_dir)
        return best or latest
    return None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_text(path: str, s: str):
    with open(path, "w") as f:
        f.write(s)

def append_ppl_csv(csv_path: str, row: Dict[str, Any]):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def mark_model_parallel(model):
    """
    Tell HF/Trainer that model is already sharded by device_map (single-process model-parallel),
    so it should not try to move the whole model onto one device.
    """
    model.is_parallelizable = True
    model.model_parallel = True
    if hasattr(model, "model") and model.model is not None:
        model.model.is_parallelizable = True
        model.model.model_parallel = True
    return model

def scan_nan_inf(model, only_contains: Optional[List[str]] = None, max_print: int = 50) -> int:
    bad = 0
    shown = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if only_contains:
                ln = n.lower()
                if not any(s.lower() in ln for s in only_contains):
                    continue
            if not torch.isfinite(p).all():
                bad += 1
                if is_rank0() and shown < max_print:
                    shown += 1
                    logging.warning(f"[NaN/Inf] {n} dtype={p.dtype} shape={tuple(p.shape)}")
    return bad

def sanitize_internal_dora_to_zero(model) -> int:
    fixed = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "dora" in n.lower():
                if not torch.isfinite(p).all():
                    p.data.zero_()
                    fixed += 1
    return fixed

def get_underlying_lm(model):
    """
    If wrapped by PEFT, return the underlying HF CausalLM model.
    """
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model

def device_str(x):
    try:
        return str(x.device)
    except Exception:
        return "unknown"

def quick_param_stat(model, name_contains: str):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if name_contains in n:
                x = p.detach().float().flatten()
                if x.numel() == 0:
                    logging.info(f"[ParamStat] {n} EMPTY")
                    return
                mean = x.mean().item()
                std = x.std().item()
                maxabs = x.abs().max().item()
                logging.info(f"[ParamStat] {n} shape={tuple(p.shape)} dtype={p.dtype} "
                             f"mean={mean:.6g} std={std:.6g} maxabs={maxabs:.6g} dev={device_str(p)}")
                return
    logging.info(f"[ParamStat] no param matched '{name_contains}'")

def check_tie_weights(model) -> Dict[str, Any]:
    out = {"ok": False, "same_ptr": None, "allclose": None, "msg": ""}
    lm = get_underlying_lm(model)
    try:
        w1 = lm.lm_head.weight
        w2 = lm.model.decoder.embed_tokens.weight
        same_ptr = (w1.data_ptr() == w2.data_ptr())
        a = w1[:8, :8].detach().float().cpu()
        b = w2[:8, :8].detach().float().cpu()
        allclose = torch.allclose(a, b, atol=1e-5)
        out.update({"ok": True, "same_ptr": same_ptr, "allclose": allclose})
        out["msg"] = f"same_ptr={same_ptr} allclose={allclose} lm_head={tuple(w1.shape)} embed={tuple(w2.shape)}"
    except Exception as e:
        out["msg"] = f"failed: {e}"
    return out

def print_key_module_devices(model):
    lm = get_underlying_lm(model)
    try:
        dec = lm.model.decoder
        mods = {
            "embed_tokens": dec.embed_tokens,
            "embed_positions": getattr(dec, "embed_positions", None),
            "final_ln": getattr(dec, "final_layer_norm", None),
            "lm_head": lm.lm_head,
        }
        if hasattr(dec, "layers") and len(dec.layers) > 0:
            mods["layer0"] = dec.layers[0]
            mods["layerLast"] = dec.layers[-1]
            mods["attn0"] = dec.layers[0].self_attn

        for k, m in mods.items():
            if m is None:
                continue
            dev = "no-params"
            for p in m.parameters(recurse=True):
                dev = str(p.device)
                break
            logging.info(f"[Device] {k:>12} -> {dev}")
    except Exception as e:
        logging.warning(f"[Device] failed: {e}")

def get_embed_device(model) -> torch.device:
    lm = get_underlying_lm(model)
    try:
        p = next(lm.model.decoder.embed_tokens.parameters())
        return p.device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def install_forward_hit_hooks(model, mode: str) -> Tuple[Dict[str, int], List[Any]]:
    hits = {}
    handles = []
    lm = get_underlying_lm(model)

    def add_hook(module, name):
        hits[name] = 0
        def _hook(m, inp, out):
            hits[name] += 1
        handles.append(module.register_forward_hook(_hook))

    try:
        attn0 = lm.model.decoder.layers[0].self_attn
        if mode == "compressed":
            if hasattr(attn0, "q_proj_low"):
                add_hook(attn0.q_proj_low, "q_proj_low")
            if hasattr(attn0, "k_proj_low"):
                add_hook(attn0.k_proj_low, "k_proj_low")
            if hasattr(attn0, "v_proj_low"):
                add_hook(attn0.v_proj_low, "v_proj_low")
        else:
            if hasattr(attn0, "q_proj"):
                add_hook(attn0.q_proj, "q_proj")
            if hasattr(attn0, "k_proj"):
                add_hook(attn0.k_proj, "k_proj")
            if hasattr(attn0, "v_proj"):
                add_hook(attn0.v_proj, "v_proj")
    except Exception as e:
        logging.warning(f"[Hook] install failed: {e}")

    return hits, handles

def remove_hooks(handles: List[Any]):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass
def get_model_autocast_dtype(model):
    # pick first param dtype
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float16

def sanity_forward_one_batch(model, eval_ds, device: torch.device):
    if len(eval_ds) <= 0:
        logging.warning("[Sanity] eval_ds is empty, skip sanity forward.")
        return

    b0 = eval_ds[0]
    batch = {k: v.clone() for k, v in b0.items()}
    for k in list(batch.keys()):
        batch[k] = batch[k].unsqueeze(0).to(device)

    model.eval()
    amp_dtype = get_model_autocast_dtype(model)
    use_amp = (device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16))

    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**batch)
        else:
            out = model(**batch)

        loss = out.loss
        logits = out.logits
        loss_v = float(loss.detach().float().cpu().item()) if loss is not None else None
        finite_logits = bool(torch.isfinite(logits).all().item())
        maxabs = float(logits.detach().float().abs().max().cpu().item())
        logging.info(f"[Sanity] loss={loss_v} logits_finite={finite_logits} logits_maxabs={maxabs:.6g}")

# =========================
# Data: streaming -> token blocks
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


class FiniteEvalDataset(Dataset):
    def __init__(self, iterable_dataset: IterableDataset, num_blocks: int):
        self.data = []
        it = iter(iterable_dataset)
        for _ in range(num_blocks):
            try:
                self.data.append(next(it))
            except StopIteration:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class PreloadedEvalDataset(Dataset):
    def __init__(self, blocks: List[Dict[str, torch.Tensor]]):
        self.data = blocks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


# =========================
# Offline local file loader (STRICT)
# =========================

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
    if is_rank0():
        logging.info(f"[Dataset] STRICT local loader={loader} files={len(files)}")
        logging.info(f"[Dataset] example_file={files[0]}")

    if loader == "parquet":
        return load_dataset("parquet", data_files=files, split="train", streaming=True)
    if loader == "json":
        return load_dataset("json", data_files=files, split="train", streaming=True)
    return load_dataset("text", data_files=files, split="train", streaming=True)


# =========================
# Device map (single-process model parallel)
# =========================

def build_layer_device_map(cfg, mode: str):
    assert torch.cuda.is_available(), "CUDA is required."
    n_gpus = torch.cuda.device_count()
    if mode == "single" or n_gpus == 1:
        return {"": 0}

    n_layers = int(cfg.num_hidden_layers)
    per = (n_layers + n_gpus - 1) // n_gpus

    device_map = {"": 0}
    for i in range(n_layers):
        dev = min(i // per, n_gpus - 1)
        device_map[f"model.decoder.layers.{i}"] = int(dev)

    last = int(n_gpus - 1)
    device_map["model.decoder.embed_tokens"] = last
    device_map["lm_head"] = last
    device_map["model.decoder.final_layer_norm"] = last
    device_map["model.decoder.embed_positions"] = last
    device_map["model.decoder.project_in"] = last
    device_map["model.decoder.project_out"] = last

    return device_map


# =========================
# Trainer callback: log PPL and track best checkpoint
# =========================
from transformers.trainer_callback import TrainerCallback
class VSCALEDebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if not is_rank0():
            return
        if model is None:
            return
        # print every logging_steps
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
            logging.info(f"[VSCALE-DBG] mean(v)={sum(vals)/len(vals):.6g} mean(|g|)={sum(grads)/len(grads):.6g} max(|g|)={max(grads):.6g}")


class PPLLogCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "ppl_log.csv")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not is_rank0():
            return
        metrics = metrics or {}
        ev_loss = metrics.get("eval_loss", None)
        if ev_loss is None:
            return
        ppl = math.exp(ev_loss) if ev_loss < 50 else float("inf")
        row = {
            "step": int(state.global_step),
            "eval_loss": float(ev_loss),
            "ppl": float(ppl),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_ppl_csv(self.csv_path, row)

        best_ckpt = getattr(state, "best_model_checkpoint", None)
        if best_ckpt:
            write_text(os.path.join(self.output_dir, "best_checkpoint.txt"), best_ckpt + "\n")


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

    # mode
    p.add_argument("--mode", choices=["baseline", "compressed"], required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)

    # compressed extra
    p.add_argument("--low_rank_json", type=str, default=None)
    p.add_argument("--import_eigen_class", type=str,
                   default="models.modelling_opt_eigen_attn:OPTForCausalLM_EigenAttn")

    # data (STRICT local)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--shuffle_buffer", type=int, default=10000)
    p.add_argument("--block_size", type=int, default=512)

    # train/eval
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)

    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=120)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=10)

    # dtype
    p.add_argument("--fp16", type=str2bool, default=True, nargs="?", const=True)
    p.add_argument("--bf16", type=str2bool, default=False, nargs="?", const=True)

    p.add_argument("--seed", type=int, default=2)

    # device map
    p.add_argument("--device_map_mode", choices=["single", "layer"], default="single")

    # resume
    p.add_argument("--resume_policy", choices=["none", "latest", "best"], default="latest")

    # memory controls
    p.add_argument("--gradient_checkpointing", type=str2bool, default=True, nargs="?", const=True)
    p.add_argument("--use_cache", type=str2bool, default=False, nargs="?", const=True)

    # optimizer choice (optional)
    p.add_argument("--optim", type=str, default="adamw_torch")

    # peft (LoRA/DoRA)
    p.add_argument("--use_peft", action="store_true", default=False)
    p.add_argument("--use_dora", action="store_true", default=False,
                   help="If set, enable DoRA in PEFT (requires peft supports use_dora).")
    p.add_argument("--peft_r", type=int, default=8)
    p.add_argument("--peft_alpha", type=int, default=16)
    p.add_argument("--peft_dropout", type=float, default=0.05)
    p.add_argument("--peft_target_modules", type=str, default="auto")

    p.add_argument("--load_peft_model", action="store_true", default=False)
    p.add_argument("--peft_model_path", type=str, default=None)

    # eval set
    p.add_argument("--eval_num_blocks", type=int, default=200)
    p.add_argument("--eval_blocks_path", type=str, default=None)
    p.add_argument("--save_eval_blocks", type=str2bool, default=True, nargs="?", const=True)

    # run modes
    p.add_argument("--do_train", type=str2bool, default=True, nargs="?", const=True)
    p.add_argument("--do_eval", type=str2bool, default=True, nargs="?", const=True)
    p.add_argument("--eval_only", action="store_true", default=False)

    # safety
    p.add_argument("--check_nan_inf", type=str2bool, default=True, nargs="?", const=True)
    p.add_argument("--sanitize_internal_dora", type=str2bool, default=False, nargs="?", const=True)

    # debug checks
    p.add_argument("--debug_checks", type=str2bool, default=True, nargs="?", const=True)
    p.add_argument("--auto_tie_weights", type=str2bool, default=True, nargs="?", const=True)

    # ===== NEW: vscale-only training =====
    p.add_argument("--train_vscale_only", action="store_true", default=False,
                   help="Freeze all params and only train each layer self_attn.vscale_log.")
    p.add_argument("--vscale_init", type=float, default=0.03,
                   help="Init value for missing self_attn.vscale_log (via EIGEN_VSCALE_INIT).")
    p.add_argument("--vscale_lr", type=float, default=5e-2,
                   help="Learning rate for vscale-only training (scalars).")
    p.add_argument("--force_env_vscale_one", type=str2bool, default=True, nargs="?", const=True,
                   help="If True, set EIGEN_VSCALE=1.0 so env constant won't override learnable vscale.")
    p.add_argument("--vscale_init_path", type=str, default=None,
               help="Optional: path to layerwise gamma init (json/pt). length=num_layers.")
    p.add_argument("--vscale_param_is_log", type=str2bool, default=False, nargs="?", const=True,
                help="If True, we store log(gamma) in vscale_log; else store gamma directly.")
    p.add_argument("--vscale_clamp_min", type=float, default=0.0)
    p.add_argument("--vscale_clamp_max", type=float, default=0.0,
                help="If >0, clamp vscale values into [min,max].")

    return p.parse_args()


def infer_target_modules(model, mode: str) -> List[str]:
    cand = ["q_proj", "k_proj", "v_proj"] if mode == "baseline" else ["q_proj_low", "k_proj_low", "v_proj_low"]
    names = set()
    lm = get_underlying_lm(model)
    for n, m in lm.named_modules():
        if isinstance(m, torch.nn.Linear):
            for c in cand:
                if n.endswith(c):
                    names.add(c)
    return sorted(list(names)) or cand


# =========================
# Load model/tokenizer
# =========================

def apply_memory_savers(model, args):
    if hasattr(model, "config") and model.config is not None:
        model.config.use_cache = bool(args.use_cache)

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model

def _try_output_loading_info(fn, **kwargs):
    try:
        model, info = fn(output_loading_info=True, **kwargs)
        return model, info
    except TypeError:
        model = fn(**kwargs)
        return model, None

def load_model_and_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(args.model_path)
    device_map = build_layer_device_map(cfg, args.device_map_mode)

    if args.fp16 and args.bf16:
        raise ValueError("fp16 and bf16 cannot both be True.")
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

    loading_info = None

    if args.mode == "baseline":
        fn = lambda **kw: AutoModelForCausalLM.from_pretrained(**kw)
        model, loading_info = _try_output_loading_info(
            fn,
            pretrained_model_name_or_path=args.model_path,
            config=cfg,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        low_rank_config = None
    else:
        if not args.low_rank_json:
            raise ValueError("compressed mode requires --low_rank_json")
        low_rank_config = json.load(open(args.low_rank_json, "r"))

        mod_path, cls_name = args.import_eigen_class.split(":")
        module = importlib.import_module(mod_path)
        EigenCls = getattr(module, cls_name)

        fn = lambda **kw: EigenCls.from_pretrained(**kw)
        model, loading_info = _try_output_loading_info(
            fn,
            pretrained_model_name_or_path=args.model_path,
            config=cfg,
            low_rank_config=low_rank_config,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

    model = mark_model_parallel(model)
    model = apply_memory_savers(model, args)
    model.eval()
    return model, tok, cfg, low_rank_config, loading_info


class NoMoveTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        return model


# =========================
# Build datasets (STRICT local files)
# =========================

def build_datasets(args, tok):
    if args.eval_blocks_path and os.path.exists(args.eval_blocks_path):
        if is_rank0():
            logging.info(f"[Data] Loading fixed eval blocks from {args.eval_blocks_path}")
        blocks = torch.load(args.eval_blocks_path, map_location="cpu")
        eval_ds = PreloadedEvalDataset(blocks)
        raw = load_streaming_local_dataset_strict(args.data_dir).shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        train_raw = raw.skip(args.eval_num_blocks * 2)
        train_iter = TokenBlockIterableDataset(train_raw, tok, block_size=args.block_size, text_field=args.text_field)
        return train_iter, eval_ds

    raw = load_streaming_local_dataset_strict(args.data_dir)
    raw = raw.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    eval_raw = raw.take(args.eval_num_blocks * 2)
    train_raw = raw.skip(args.eval_num_blocks * 2)

    train_iter = TokenBlockIterableDataset(train_raw, tok, block_size=args.block_size, text_field=args.text_field)
    eval_iter  = TokenBlockIterableDataset(eval_raw,  tok, block_size=args.block_size, text_field=args.text_field)

    eval_ds = FiniteEvalDataset(eval_iter, num_blocks=args.eval_num_blocks)

    if args.save_eval_blocks and is_rank0():
        path = os.path.join(args.output_dir, "eval_blocks.pt")
        try:
            torch.save(eval_ds.data, path)
            logging.info(f"[Data] Saved eval blocks to {path} (len={len(eval_ds)})")
        except Exception as e:
            logging.warning(f"[Data] Failed to save eval blocks: {e}")

    return train_iter, eval_ds


# =========================
# NEW: vscale-only parameter selection
# =========================

def enable_vscale_only_training(model) -> Tuple[int, List[str]]:
    """
    Freeze all params; unfreeze only *.self_attn.vscale_log.
    Returns (count, example_names)
    """
    for p in model.parameters():
        p.requires_grad_(False)

    names = []
    for n, p in model.named_parameters():
        if n.endswith("self_attn.vscale_log"):
            p.requires_grad_(True)
            if len(names) < 10:
                names.append(n)

    cnt = sum(1 for n, p in model.named_parameters() if n.endswith("self_attn.vscale_log"))
    return cnt, names

def load_layerwise_vscale_init(path: str, num_layers: int) -> List[float]:
    if path is None:
        return None
    if path.endswith(".pt") or path.endswith(".pth"):
        obj = torch.load(path, map_location="cpu")
        # support {"gamma": [...] } or direct list/tensor
        if isinstance(obj, dict) and "gamma" in obj:
            arr = obj["gamma"]
        else:
            arr = obj
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().tolist()
        return [float(x) for x in arr]
    else:
        j = json.load(open(path, "r"))
        # support {"gamma": [...]} or direct list
        arr = j["gamma"] if isinstance(j, dict) and "gamma" in j else j
        return [float(x) for x in arr]

def init_vscale_from_list(model, gammas: List[float], param_is_log: bool, clamp_min: float, clamp_max: float):
    lm = get_underlying_lm(model)
    layers = lm.model.decoder.layers
    assert len(gammas) == len(layers), f"gamma len {len(gammas)} != num_layers {len(layers)}"
    with torch.no_grad():
        for i, layer in enumerate(layers):
            p = layer.self_attn.vscale_log
            g = float(gammas[i])
            if clamp_max and clamp_max > 0:
                g = max(clamp_min, min(clamp_max, g))
            if param_is_log:
                # store log(gamma)
                g = math.log(max(g, 1e-12))
            p.data.fill_(g)
# =========================
# Main
# =========================

def main():
    setup_logger()
    args = parse_args()
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.eval_only:
        args.do_train = False
        args.do_eval = True

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} detected. This script is SINGLE-PROCESS only.\n"
            f"Run WITHOUT torchrun, e.g.:\n"
            f"  CUDA_VISIBLE_DEVICES=0 python finetune_eval_opt30b_eigen_dora.py ...\n"
        )

    ensure_dir(args.output_dir)

    resume_ckpt = pick_resume_checkpoint(args.output_dir, args.resume_policy)
    if is_rank0():
        logging.info(f"[Resume] policy={args.resume_policy} -> {resume_ckpt}")

    # ===== NEW: vscale init + env control =====
    if args.train_vscale_only:
        # init missing vscale_log from env in your __init__
        os.environ["EIGEN_VSCALE_INIT"] = str(args.vscale_init)
        # prevent env constant from overriding learnable vscale
        if args.force_env_vscale_one:
            os.environ["EIGEN_VSCALE"] = "1.0"
        if is_rank0():
            logging.info(f"[VSCALE] train_vscale_only=1 init={os.environ.get('EIGEN_VSCALE_INIT')} env_vscale={os.environ.get('EIGEN_VSCALE')}")

        # vscale-only training should NOT also train PEFT (unless你明确想联合训练)
        if args.use_peft or args.load_peft_model:
            logging.warning("[VSCALE] train_vscale_only enabled: disabling PEFT flags to avoid training adapters.")
            args.use_peft = False
            args.load_peft_model = False

        # Use scalar-friendly LR
        args.learning_rate = float(args.vscale_lr)
        # Warmup usually unnecessary for 48 scalars; set to 0 for stability
        args.warmup_steps = 0

    # load model
    model, tok, cfg, low_rank_config, loading_info = load_model_and_tokenizer(args)
# ---- NEW: optional layerwise vscale init ----
    if args.train_vscale_only and args.vscale_init_path:
        lm = get_underlying_lm(model)
        num_layers = len(lm.model.decoder.layers)
        gammas = load_layerwise_vscale_init(args.vscale_init_path, num_layers)
        if gammas is None:
            raise ValueError("--vscale_init_path given but failed to load gammas")
        init_vscale_from_list(
            model,
            gammas,
            param_is_log=bool(args.vscale_param_is_log),
            clamp_min=float(args.vscale_clamp_min),
            clamp_max=float(args.vscale_clamp_max),
        )
        if is_rank0():
            logging.info(f"[VSCALE] loaded layerwise init from {args.vscale_init_path} (len={len(gammas)})")
            quick_param_stat(model, "self_attn.vscale_log")
    if is_rank0():
        lm = get_underlying_lm(model)
        attn = lm.model.decoder.layers[0].self_attn
        print("[DEBUG] self_attn class =", type(attn))
        print("[DEBUG] has q_proj_low =", hasattr(attn, "q_proj_low"))
        print("[DEBUG] has q_proj     =", hasattr(attn, "q_proj"))
        print("[DEBUG] scaling        =", getattr(attn, "scaling", None))
        print("[DEBUG] rank_kq/rank_v  =", getattr(attn, "rank_kq", None), getattr(attn, "rank_v", None))
        print("[DEBUG] num_heads      =", getattr(attn, "num_heads", None))
        logging.info(f"[Model] mode={args.mode} class={type(model)}")
        logging.info(f"[Params] {count_trainable_params(model)}")
        logging.info(f"[GPUs] visible={torch.cuda.device_count()} map_mode={args.device_map_mode}")
        logging.info(f"[Mem] use_cache={args.use_cache} grad_ckpt={args.gradient_checkpointing}")

        dmap = getattr(model, "hf_device_map", None)
        if dmap:
            logging.info(f"[DeviceMap] keys={len(dmap)} example={list(dmap.items())[:5]}")

        if loading_info is not None:
            mk = loading_info.get("missing_keys", [])
            uk = loading_info.get("unexpected_keys", [])
            mm = loading_info.get("mismatched_keys", [])
            logging.info(f"[LoadInfo] missing_keys={len(mk)} unexpected_keys={len(uk)} mismatched_keys={len(mm)}")
            if len(mk) > 0:
                logging.warning(f"[LoadInfo] first_missing={mk[:10]}")
            if len(mm) > 0:
                logging.warning(f"[LoadInfo] first_mismatched={mm[:10]}")

    # NaN/Inf scan
    if args.check_nan_inf:
        bad = scan_nan_inf(model, only_contains=["dora", "_low", "eigen", "lm_head", "embed_tokens"], max_print=50)
        if is_rank0():
            logging.info(f"[Check] bad_tensors(dora/_low/eigen/lm_head/embed)={bad}")
        if bad > 0 and args.sanitize_internal_dora:
            fixed = sanitize_internal_dora_to_zero(model)
            model = mark_model_parallel(model)
            model = apply_memory_savers(model, args)
            if is_rank0():
                logging.warning(f"[Sanitize] internal dora fixed_tensors={fixed}")

    # ===== NEW: enable vscale-only training (after load, before trainer) =====
    if args.train_vscale_only:
        cnt, examples = enable_vscale_only_training(model)
        if is_rank0():
            logging.info(f"[VSCALE] enabled vscale-only training params={cnt} examples={examples}")
            logging.info(f"[Params-AfterVSCALE] {count_trainable_params(model)}")

    # build data
    train_ds, eval_ds = build_datasets(args, tok)
    if is_rank0():
        logging.info(f"[Data] eval_num_blocks={len(eval_ds)} block_size={args.block_size}")

    # DEBUG CHECKS
    if is_rank0() and args.debug_checks:
        print_key_module_devices(model)
        quick_param_stat(model, "lm_head.weight")
        quick_param_stat(model, "model.decoder.embed_tokens.weight")
        if args.mode == "compressed":
            quick_param_stat(model, "q_proj_low")
            quick_param_stat(model, "k_proj_low")
            quick_param_stat(model, "v_proj_low")
        else:
            quick_param_stat(model, "q_proj.weight")
            quick_param_stat(model, "k_proj.weight")
            quick_param_stat(model, "v_proj.weight")

        tie = check_tie_weights(model)
        logging.info(f"[TieCheck] {tie['msg']}")
        if tie["ok"] and (tie["same_ptr"] is False and tie["allclose"] is False):
            logging.warning("[TieCheck] lm_head and embed_tokens appear NOT tied / not equal.")
            if args.auto_tie_weights and hasattr(get_underlying_lm(model), "tie_weights"):
                try:
                    get_underlying_lm(model).tie_weights()
                    tie2 = check_tie_weights(model)
                    logging.info(f"[TieCheck-AfterTie] {tie2['msg']}")
                except Exception as e:
                    logging.warning(f"[TieCheck] tie_weights() failed: {e}")

        hits, handles = install_forward_hit_hooks(model, args.mode)
        try:
            device = get_embed_device(model)
            sanity_forward_one_batch(model, eval_ds, device=device)
        except RuntimeError as e:
            logging.error(f"[Sanity] forward RuntimeError: {e}")
        finally:
            if len(hits) > 0:
                logging.info(f"[HookHits] {hits}")
            else:
                logging.warning("[HookHits] no hooks installed; cannot confirm proj path usage.")
            remove_hooks(handles)

    # training args
    targs = TrainingArguments(
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,

        evaluation_strategy="steps" if args.do_eval else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        load_best_model_at_end=True if args.do_eval else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=args.fp16,
        bf16=args.bf16,

        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,

        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    cbs = [PPLLogCallback(args.output_dir)]
    if args.train_vscale_only:
        cbs.append(VSCALEDebugCallback())
    trainer = NoMoveTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds if args.do_eval else None,
        tokenizer=tok,
        data_collator=default_data_collator,
        callbacks=[PPLLogCallback(args.output_dir)],
    )

    # eval only
    if args.eval_only:
        metrics = trainer.evaluate()
        if is_rank0():
            ev_loss = metrics.get("eval_loss", None)
            ppl = math.exp(ev_loss) if ev_loss is not None and ev_loss < 50 else None
            logging.info(f"[EvalOnly] eval_loss={ev_loss} ppl={ppl}")
        return

    # train
    if args.do_train:
        trainer.train(resume_from_checkpoint=resume_ckpt)

    # final eval + export
    if args.do_eval:
        metrics = trainer.evaluate()
        if is_rank0():
            ev_loss = metrics.get("eval_loss", None)
            ppl = math.exp(ev_loss) if ev_loss is not None and ev_loss < 50 else None
            logging.info(f"[FinalEval] eval_loss={ev_loss} ppl={ppl}")

    if is_rank0():
        best_ckpt = trainer.state.best_model_checkpoint
        if best_ckpt:
            write_text(os.path.join(args.output_dir, "best_checkpoint.txt"), best_ckpt + "\n")
            logging.info(f"[Best] best_checkpoint={best_ckpt}")

        best_dir = os.path.join(args.output_dir, "best_model")
        ensure_dir(best_dir)
        trainer.save_model(best_dir)
        tok.save_pretrained(best_dir)

        if args.mode == "compressed" and args.low_rank_json:
            try:
                import shutil
                shutil.copy2(args.low_rank_json, os.path.join(best_dir, os.path.basename(args.low_rank_json)))
            except Exception:
                pass

        logging.info(f"[Export] best_model saved to {best_dir}")
        logging.info("[Done]")


if __name__ == "__main__":
    main()