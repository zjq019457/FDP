#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# main_eigen_attn.py
#
# Includes your OPT-30B / A100-80G multi-GPU + OOM fixes:
# 1) DO NOT force fp32 in evaluate()  (old code did lm.model.float()).
# 2) If args.multigpu: layer-wise placement across ALL visible GPUs
#    via map_layers_to_multi_gpus(...) BEFORE eigenattn() and evaluate().
# 3) If launched with torchrun (WORLD_SIZE>1) + args.multigpu:
#    only rank0 runs and uses all GPUs (single-process multi-GPU).
#    Other ranks exit to avoid "N processes all on GPU0".
# 4) Provide --dtype {bf16,fp16,fp32} and keep model in bf16 by default on A100.
# 5) Avoid overriding HF_HOME/HF_DATASETS_CACHE if user already exported them.
#
# NEW (QKV transform inspection utilities integrated):
# - Optional export of Q/K/V projection weights to qkv_mats.pt
# - Optional analysis of "transform mode" for selected layers:
#     * SVD-optimal rank-r approximation (randomized SVD)
#     * eigen_basis-projection rank-r approximation (if eigen_basis.pt is available)
#     * Activation-level comparison using real hidden-states sampled via forward hooks
# - Results saved to: {output_dir}/qkv_transform_analysis/*.jsonl (+ optional .pt matrices)
#
# Typical usage (baseline-stage analysis before decomposition):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   python main_eigen_attn.py --model /diskE/opt-30b --net opt-30b \
#     --calib_dataset c4 --calib_data_path /diskE/c4-en/en --nsamples 128 --batch_size 1 \
#     --avg_dim 448 --error_budget 0.025 --multigpu --dtype bf16 \
#     --qkv_analyze --qkv_analyze_stage baseline --qkv_layers 46 --qkv_ranks 448,896 \
#     --qkv_export --qkv_export_layers 46
#
# If you already have eigen_basis.pt somewhere:
#   --qkv_eigen_basis /path/to/eigen_basis.pt
# ============================================================

import os
import sys
import time
import json
import math
import random
import re
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoConfig, AutoModelForCausalLM

# Your project imports
from models.LMClass import LMClass
from datautils import get_loaders, _load_local_streaming
from decompose.eigen_attn import eigenattn
import utils
from categories import subcategories, categories

from parallel_utils import map_layers_to_multi_gpus  # keep mapping util

# (Optional) these are imported in your original file; keep them if other modules use them
from models.modelling_mpt_eigen_attn import MptEigenAttention
from models.modelling_llama_eigen_attn import LlamaEigenAttention
from transformers.models.mpt.modeling_mpt import MptAttention

torch.serialization.add_safe_globals([BatchEncoding])
torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "mpt-7b",
    "mpt-30b",
    "Llama-3-8b",
    "Llama-3-80b",
]


# ----------------------------
# Small helpers
# ----------------------------
def _dist_info():
    """Return (rank, local_rank, world_size) from torchrun env vars if present."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_dtype(dtype_str: str):
    s = (dtype_str or "").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _mkdir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _parse_int_list(s: Optional[str], default: Optional[List[int]] = None) -> Optional[Union[List[int], str]]:
    if s is None:
        return default
    s = str(s).strip()
    if s == "":
        return default
    if s.lower() in ("all", "*"):
        return "all"
    out = []
    for x in s.split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(int(x))
    return out if len(out) else default


def _parse_float_list(s: Optional[str], default: Optional[List[float]] = None) -> Optional[List[float]]:
    if s is None:
        return default
    s = str(s).strip()
    if s == "":
        return default
    out = []
    for x in s.split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(float(x))
    return out if len(out) else default


def _parse_str_list(s: Optional[str], default: Optional[List[str]] = None) -> Optional[List[str]]:
    if s is None:
        return default
    s = str(s).strip()
    if s == "":
        return default
    out = []
    for x in s.split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(x)
    return out if len(out) else default


def _get_opt_basis(basis_dict: Dict[str, torch.Tensor], layer_idx: int, kind: str) -> torch.Tensor:
    key = f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{kind}_proj_basis"
    if key not in basis_dict:
        raise KeyError(f"Missing basis tensor: {key}")
    return basis_dict[key].float()


def _set_attn_vscale_one(attn_module: nn.Module):
    for attr_name, fill_value in (("vscale_log", 0.0), ("vscale", 1.0), ("vscale_param", 1.0)):
        value = getattr(attn_module, attr_name, None)
        if torch.is_tensor(value):
            value.fill_(fill_value)


def _set_attn_v_rot_identity(attn_module: nn.Module):
    """用 headwise（或任意新）basis 重建 Q/K/V low 后，必须把 v_rot 重置为单位阵。
    否则 v_rot 仍是 global basis 时代留下的，会破坏输出甚至导致 cos 反相关。"""
    v_rot = getattr(attn_module, "v_rot", None)
    if v_rot is not None and hasattr(v_rot, "weight") and torch.is_tensor(v_rot.weight):
        with torch.no_grad():
            r = v_rot.weight.shape[0]
            v_rot.weight.copy_(torch.eye(r, dtype=v_rot.weight.dtype, device=v_rot.weight.device))


def _extract_text_from_example(example: Any) -> str:
    if isinstance(example, dict):
        if isinstance(example.get("text"), str):
            return example["text"].strip()
        for value in example.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    if isinstance(example, str):
        return example.strip()
    return ""


def _build_local_c4_finetune_dataset(data_dir: str, max_samples: int, min_chars: int = 32):
    from datasets import Dataset

    stream = _load_local_streaming(data_dir)
    texts = []
    for example in stream:
        text = _extract_text_from_example(example)
        if len(text) < min_chars:
            continue
        texts.append(text)
        if len(texts) >= max_samples:
            break
    if not texts:
        raise RuntimeError(f"No usable text samples found under local C4 dir: {data_dir}")
    return Dataset.from_dict({"text": texts})


# ----------------------------
# Low-memory training helpers (adapted from finetune_ppl_opt30b_eigen.py)
# ----------------------------

def _iter_texts_from_dir(data_dir: str, max_files: int = 80, max_lines: int = 300000):
    """Yield raw text strings from txt/jsonl/json/gz files under data_dir."""
    import glob, gzip as _gzip
    exts = ["*.txt", "*.jsonl", "*.json", "*.gz"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", e), recursive=True))
    files = sorted(files)[:max_files]
    n = 0
    for fp in files:
        opener = _gzip.open if fp.endswith(".gz") else open
        try:
            with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
                if fp.endswith(".json") and not fp.endswith(".jsonl") and not fp.endswith(".gz"):
                    try:
                        obj = json.load(f)
                        items = obj if isinstance(obj, list) else [obj]
                        for it in items:
                            t = (it.get("text") or it.get("content")) if isinstance(it, dict) else (it if isinstance(it, str) else None)
                            if isinstance(t, str) and t.strip():
                                yield t.strip()
                                n += 1
                                if n >= max_lines:
                                    return
                    except Exception:
                        pass
                    continue
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t: Optional[str] = None
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            j = json.loads(line)
                            if isinstance(j, dict):
                                t = j.get("text") or j.get("content")
                        except Exception:
                            t = None
                    if t is None:
                        t = line
                    if isinstance(t, str) and t.strip():
                        yield t.strip()
                        n += 1
                        if n >= max_lines:
                            return
        except Exception:
            continue


def _build_token_stream(tokenizer, texts_iter, target_tokens: int = 500000, seed: int = 2) -> Optional[torch.Tensor]:
    """Concatenate tokenized texts into a single CPU LongTensor token stream."""
    random.seed(seed)
    ids: List[int] = []
    total = 0
    for t in texts_iter:
        enc = tokenizer(t, add_special_tokens=False)["input_ids"]
        if not enc:
            continue
        ids.extend(enc + [tokenizer.eos_token_id])
        total += len(enc) + 1
        if total >= target_tokens:
            break
    if total == 0:
        return None
    return torch.tensor(ids, dtype=torch.long)


def _batchify_stream_cpu(stream_cpu: torch.Tensor, seqlen: int, batch_size: int):
    """Infinite random-offset batch generator; tensors stay on CPU."""
    N = stream_cpu.numel()
    assert N > seqlen + 1, f"Token stream ({N}) too short for seqlen={seqlen}"
    while True:
        starts = torch.randint(0, N - seqlen - 1, (batch_size,))
        x = torch.stack([stream_cpu[s : s + seqlen] for s in starts])
        yield x  # shape: [batch_size, seqlen]


@torch.no_grad()
def _eval_ppl_simple(
    model: nn.Module,
    stream_cpu: torch.Tensor,
    device: torch.device,
    seqlen: int = 2048,
    stride: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
) -> float:
    """Sliding-window perplexity over a CPU token stream. No HF Trainer needed."""
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    nll_sum = 0.0
    tok_count = 0
    N = stream_cpu.numel()
    for start in range(0, N, stride):
        end = min(start + seqlen, N)
        if end - start < 2:
            break
        inp = stream_cpu[start:end].unsqueeze(0).to(device, non_blocking=True)
        attn = torch.ones_like(inp, dtype=torch.long)
        labels = inp.clone()
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=inp, attention_mask=attn)
        shift_logits = out.logits[:, :-1, :].contiguous().view(-1, out.logits.size(-1))
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits.float(), shift_labels, ignore_index=-100, reduction="mean")
        nll_sum += loss.detach().float().item() * (inp.numel() - 1)
        tok_count += inp.numel() - 1
        if end >= N:
            break
    return math.exp(nll_sum / max(tok_count, 1))


# -------------------------
# Utkarsh-aligned C4-train valenc + PPL (mirrors sanity_AB_opt30b.py)
# -------------------------
def _iter_jsonl_texts_gz(fp: str):
    import gzip as _gzip
    with _gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                if isinstance(j, dict) and isinstance(j.get("text", None), str):
                    t = j["text"].strip()
                    if t:
                        yield t
            except Exception:
                continue


def _build_doc_pool_from_c4_train_dir(
    data_dir: str,
    glob_pat: str = "c4-train.*.json.gz",
    pool_docs: int = 20000,
    scan_lines: int = 200000,
    seed: int = 2,
):
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(data_dir, glob_pat)))
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pat} under {data_dir}")

    rng = random.Random(seed)
    reservoir = []
    seen = 0

    for fp in files:
        for t in _iter_jsonl_texts_gz(fp):
            seen += 1
            if len(reservoir) < pool_docs:
                reservoir.append(t)
            else:
                j = rng.randint(1, seen)
                if j <= pool_docs:
                    reservoir[j - 1] = t
            if seen >= scan_lines:
                break
        if seen >= scan_lines:
            break

    if not reservoir:
        raise RuntimeError(f"Failed to read any text from {files[0]}")
    return reservoir, {"seen_docs": seen, "pool_docs": len(reservoir), "file": files[0]}


@torch.no_grad()
def _build_utkarsh_like_valenc_from_pool(
    tok,
    doc_pool,
    seqlen: int,
    n_chunks: int = 256,
    val_seed: int = 0,
    max_tries_per_chunk: int = 2000,
):
    rng = random.Random(val_seed)
    valenc = []

    for _k in range(n_chunks):
        ok = False
        for _ in range(max_tries_per_chunk):
            text = doc_pool[rng.randint(0, len(doc_pool) - 1)]
            tmp = tok(text, return_tensors="pt")
            L = tmp.input_ids.shape[1]
            if L > seqlen:
                i = rng.randint(0, L - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"Unable to sample a doc with token_len > seqlen after {max_tries_per_chunk} tries. "
                f"Try increasing --ft_ppl_scan_lines / --ft_ppl_pool_docs."
            )

    return torch.hstack(valenc)  # [1, n_chunks*seqlen]


@torch.no_grad()
def _eval_ppl_utkarsh_opt(model, testenc: torch.Tensor, device: torch.device, seqlen: int, limit: int = -1):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    nsamples = testenc.numel() // seqlen
    nlls = []

    for i in range(nsamples):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(device)

        outputs = model.model.decoder(batch)
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].to(model.lm_head.weight.device)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        nlls.append(loss.float() * seqlen)
        if i == limit:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item(), {"nsamples": nsamples, "used_chunks": len(nlls)}


def _ppl_sanity_utkarsh_c4_train(model, tok, args, logger, tag: str, device: torch.device):
    data_dir = (getattr(args, "ft_data_dir", "") or "").strip()
    if not data_dir or (not os.path.isdir(data_dir)):
        logger.info(f"[ppl][{tag}] ft_data_dir not provided or not a directory. Skip.")
        return None

    pool, pool_info = _build_doc_pool_from_c4_train_dir(
        data_dir,
        glob_pat=str(getattr(args, "ft_ppl_c4_glob", "c4-train.*.json.gz")),
        pool_docs=int(getattr(args, "ft_ppl_pool_docs", 20000)),
        scan_lines=int(getattr(args, "ft_ppl_scan_lines", 200000)),
        seed=int(getattr(args, "seed", 2)),
    )
    logger.info(f"[ppl][{tag}] C4-Pool: {pool_info}")

    seqlen = int(getattr(args, "ft_ppl_seqlen", 2048))
    n_chunks = int(getattr(args, "ft_ppl_val_chunks", 256))
    val_seed = int(getattr(args, "ft_ppl_val_seed", 0))

    testenc = _build_utkarsh_like_valenc_from_pool(tok, pool, seqlen=seqlen, n_chunks=n_chunks, val_seed=val_seed)
    ppl, info = _eval_ppl_utkarsh_opt(
        model,
        testenc,
        device=device,
        seqlen=seqlen,
        limit=int(getattr(args, "limit", -1)),
    )
    logger.info(f"[ppl][{tag}] PPL={ppl:.6f} info={info}")
    return ppl


def _set_trainable_opt(model: nn.Module, mode: str = "qkv_up_vscale", train_last_n: int = 0):
    """
    Freeze all parameters, then selectively unfreeze by mode.

    mode choices:
      vscale_only    – only vscale_log
      up_vscale      – out_proj_up  +  vscale_log
      qkv_up_vscale  – q/k/v_proj_low  +  out_proj_up  +  vscale_log
      v_up_vscale    – v_proj_low   +  out_proj_up  +  vscale_log
      all            – everything

    train_last_n:
      0  → unfreeze the selected params in ALL decoder layers
      >0 → unfreeze only the last N decoder layers
    """
    for p in model.parameters():
        p.requires_grad_(False)

    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)
        return

    def _mark(module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(True)

    layers = list(model.model.decoder.layers)
    if train_last_n and train_last_n > 0:
        layers = layers[-train_last_n:]

    for layer in layers:
        attn = layer.self_attn
        if hasattr(attn, "vscale_log"):
            try:
                attn.vscale_log.requires_grad_(True)
            except Exception:
                pass
        if mode in ("up_vscale", "qkv_up_vscale", "v_up_vscale"):
            if hasattr(attn, "out_proj_up"):
                _mark(attn.out_proj_up)
        if mode == "qkv_up_vscale":
            for nm in ("q_proj_low", "k_proj_low", "v_proj_low"):
                if hasattr(attn, nm):
                    _mark(getattr(attn, nm))
        if mode == "v_up_vscale":
            if hasattr(attn, "v_proj_low"):
                _mark(attn.v_proj_low)


def _top_evecs_from_rows(weight_block: torch.Tensor, rank_per_head: int) -> torch.Tensor:
    """
    Compute a headwise row-space basis from a full-dim effective weight block.
    weight_block: [head_dim, hidden]
    returns: [head_dim, rank_per_head]
    """
    cov = weight_block @ weight_block.t()
    _, evecs = torch.linalg.eigh(cov)
    return evecs[:, -rank_per_head:].contiguous()


@torch.no_grad()
def _refresh_opt_basis_and_reproject(
    model: nn.Module,
    basis_dict: Dict[str, torch.Tensor],
    repair_keqk: int,
    logger,
    force_vscale_one: int = 1,
):
    """
    Scheme C outer step for OPT:
    1) Use previous basis + current low-rank weights to reconstruct effective full-dim Q/K/V/O.
    2) Recompute headwise eigen bases from those effective weights.
    3) Reproject back into low-rank weights under the refreshed basis.

    This keeps the runtime model unchanged while refreshing the latent subspace state.
    """
    layers = model.model.decoder.layers
    hidden_size = int(model.config.hidden_size)
    num_heads = int(model.config.num_attention_heads)
    head_dim = hidden_size // num_heads

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn

        u_q_old = _get_opt_basis(basis_dict, layer_idx, "q")
        u_k_old = u_q_old if repair_keqk == 1 else _get_opt_basis(basis_dict, layer_idx, "k")
        u_v_old = _get_opt_basis(basis_dict, layer_idx, "v")

        q_low_weight = attn.q_proj_low.weight.detach().float().cpu()
        k_low_weight = attn.k_proj_low.weight.detach().float().cpu()
        v_low_weight = attn.v_proj_low.weight.detach().float().cpu()
        out_up_weight = attn.out_proj_up.weight.detach().float().cpu()

        q_low_bias = attn.q_proj_low.bias.detach().float().cpu() if attn.q_proj_low.bias is not None else None
        k_low_bias = attn.k_proj_low.bias.detach().float().cpu() if attn.k_proj_low.bias is not None else None
        v_low_bias = attn.v_proj_low.bias.detach().float().cpu() if attn.v_proj_low.bias is not None else None
        out_up_bias = attn.out_proj_up.bias.detach().float().cpu() if attn.out_proj_up.bias is not None else None

        # Lift the current low-rank weights back to their effective full-dim matrices.
        w_q_eff = (u_q_old @ q_low_weight).contiguous()
        w_k_eff = (u_k_old @ k_low_weight).contiguous()
        w_v_eff = (u_v_old @ v_low_weight).contiguous()
        w_o_eff = (out_up_weight @ u_v_old.t()).contiguous()

        b_q_eff = (u_q_old @ q_low_bias).contiguous() if q_low_bias is not None else None
        b_k_eff = (u_k_old @ k_low_bias).contiguous() if k_low_bias is not None else None
        b_v_eff = (u_v_old @ v_low_bias).contiguous() if v_low_bias is not None else None

        rank_kq = int(attn.rank_kq)
        rank_v = int(attn.rank_v)
        rank_kq_per_head = rank_kq // num_heads
        rank_v_per_head = rank_v // num_heads

        u_q_new = torch.zeros((hidden_size, rank_kq), dtype=torch.float32)
        u_k_new = torch.zeros((hidden_size, rank_kq), dtype=torch.float32)
        u_v_new = torch.zeros((hidden_size, rank_v), dtype=torch.float32)

        for head_idx in range(num_heads):
            row_slice = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
            col_slice_kq = slice(head_idx * rank_kq_per_head, (head_idx + 1) * rank_kq_per_head)
            col_slice_v = slice(head_idx * rank_v_per_head, (head_idx + 1) * rank_v_per_head)

            u_q_head = _top_evecs_from_rows(w_q_eff[row_slice, :], rank_kq_per_head)
            u_v_head = _top_evecs_from_rows(w_v_eff[row_slice, :], rank_v_per_head)

            u_q_new[row_slice, col_slice_kq] = u_q_head
            u_v_new[row_slice, col_slice_v] = u_v_head

            if repair_keqk == 1:
                u_k_new[row_slice, col_slice_kq] = u_q_head
            else:
                u_k_new[row_slice, col_slice_kq] = _top_evecs_from_rows(
                    w_k_eff[row_slice, :], rank_kq_per_head
                )

        basis_dict[f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.q_proj_basis"] = u_q_new
        basis_dict[f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.k_proj_basis"] = u_k_new
        basis_dict[f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.v_proj_basis"] = u_v_new

        attn.q_proj_low.weight.copy_(
            (u_q_new.t() @ w_q_eff).to(dtype=attn.q_proj_low.weight.dtype, device=attn.q_proj_low.weight.device)
        )
        attn.k_proj_low.weight.copy_(
            (u_k_new.t() @ w_k_eff).to(dtype=attn.k_proj_low.weight.dtype, device=attn.k_proj_low.weight.device)
        )
        attn.v_proj_low.weight.copy_(
            (u_v_new.t() @ w_v_eff).to(dtype=attn.v_proj_low.weight.dtype, device=attn.v_proj_low.weight.device)
        )
        attn.out_proj_up.weight.copy_(
            (w_o_eff @ u_v_new).to(dtype=attn.out_proj_up.weight.dtype, device=attn.out_proj_up.weight.device)
        )

        if attn.q_proj_low.bias is not None and b_q_eff is not None:
            attn.q_proj_low.bias.copy_(
                (b_q_eff @ u_q_new).to(dtype=attn.q_proj_low.bias.dtype, device=attn.q_proj_low.bias.device)
            )
        if attn.k_proj_low.bias is not None and b_k_eff is not None:
            attn.k_proj_low.bias.copy_(
                (b_k_eff @ u_k_new).to(dtype=attn.k_proj_low.bias.dtype, device=attn.k_proj_low.bias.device)
            )
        if attn.v_proj_low.bias is not None and b_v_eff is not None:
            attn.v_proj_low.bias.copy_(
                (b_v_eff @ u_v_new).to(dtype=attn.v_proj_low.bias.dtype, device=attn.v_proj_low.bias.device)
            )
        if attn.out_proj_up.bias is not None and out_up_bias is not None:
            attn.out_proj_up.bias.copy_(
                out_up_bias.to(dtype=attn.out_proj_up.bias.dtype, device=attn.out_proj_up.bias.device)
            )

        if force_vscale_one == 1:
            _set_attn_vscale_one(attn)

        if layer_idx % 8 == 0:
            logger.info(f"[fine_tune/bilevel] refreshed OPT basis at layer {layer_idx}")


# ----------------------------
# Model size / KV cache size helpers (your originals)
# ----------------------------
def get_model_size(model):
    num_elements = 0
    for key in model.model.state_dict().keys():
        if "lm_head" in key:
            continue
        param = model.model.state_dict()[key]
        num_elements += param.numel()
    # (assuming 16-bit storage in this reporting)
    model_size = num_elements * 16 / (8 * 1024 * 1024 * 1024)
    return num_elements, model_size


def get_kvcache_size_opt(lm):
    num_elements = 0
    seq_len = lm.seqlen
    for key in lm.model.state_dict().keys():
        if (
            "k_proj.weight" in key
            or "v_proj.weight" in key
            or "v_proj_low.weight" in key
            or "k_proj_low.weight" in key
        ):
            num_elements += seq_len * lm.model.state_dict()[key].shape[0]
    cache_size = num_elements * 16 / (8 * 1024 * 1024 * 1024)
    return cache_size


def get_kvcache_size_mpt(lm, args):
    num_elements = 0
    seq_len = lm.seqlen
    total_rankkq = 0
    total_rankv = 0
    i = 0
    for name, m in lm.model.named_modules():
        if (".attn" in name and isinstance(m, MptAttention)):
            num_elements += seq_len * m.Wqkv.weight.shape[0] * 2 / 3
        elif (".attn" in name and isinstance(m, MptEigenAttention)):
            num_elements += seq_len * (m.rank_kq + m.rank_v)
            total_rankkq += m.rank_kq
            total_rankv += m.rank_v
            i += 1
    cache_size = (num_elements * 16) / (8 * 1024 * 1024 * 1024)
    return cache_size


def get_kvcache_size_llama(lm):
    num_elements = 0
    seq_len = lm.seqlen
    for key in lm.model.state_dict().keys():
        if (
            "k_proj.weight" in key
            or "v_proj.weight" in key
            or "v_proj_low.weight" in key
            or "k_proj_low.weight" in key
        ):
            num_elements += seq_len * lm.model.state_dict()[key].shape[0]
    cache_size = num_elements * 16 / (8 * 1024 * 1024 * 1024)
    return cache_size


# ----------------------------
# Multi-GPU placement util (your original with fixes kept)
# ----------------------------
def _place_model_multigpu(lm, args, logger):
    """
    Layer-wise pipeline placement across all visible GPUs.
    This is the "one layer group per GPU" style mapping (not DDP).
    """
    if not args.multigpu:
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; multigpu mapping skipped.")
        return

    if getattr(lm.model, "_eigen_multigpu_mapped", False):
        logger.info("[multigpu] current model already mapped; skip duplicate placement.")
        return

    layer_gpu_map = None

    if "opt" in args.net.lower():
        layer_gpu_map = map_layers_to_multi_gpus(lm.model.model.decoder.layers, logger=logger)
        input_device = lm.model.model.decoder.layers[0].device
        output_device = lm.model.model.decoder.layers[-1].device
        lm._device = input_device
        assert input_device == output_device
        lm.model.model.decoder.embed_positions.to(input_device)
        lm.model.model.decoder.embed_tokens.to(input_device)
        lm.model.model.decoder.final_layer_norm.to(output_device)
        lm.model.lm_head.to(output_device)

    elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
        layer_gpu_map = map_layers_to_multi_gpus(lm.model.model.layers, logger=logger)
        input_device = lm.model.model.layers[0].device
        output_device = lm.model.model.layers[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.model.embed_tokens.to(input_device)
        lm.model.model.norm.to(output_device)
        lm.model.lm_head.to(output_device)

    elif "falcon" in args.net.lower():
        layer_gpu_map = map_layers_to_multi_gpus(lm.model.transformer.h, logger=logger)
        input_device = lm.model.transformer.h[0].device
        output_device = lm.model.transformer.h[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.transformer.word_embeddings.to(input_device)
        lm.model.transformer.ln_f.to(output_device)
        lm.model.lm_head.to(output_device)

    elif "mpt" in args.net.lower():
        layer_gpu_map = map_layers_to_multi_gpus(lm.model.transformer.blocks, logger=logger)
        input_device = lm.model.transformer.blocks[0].device
        output_device = lm.model.transformer.blocks[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.transformer.wte.to(input_device)
        lm.model.transformer.norm_f.to(output_device)
        lm.model.lm_head.to(output_device)

    else:
        logger.warning(f"multigpu mapping not implemented for net={args.net}")
        return

    lm.model.is_parallelizable = True
    lm.model.model_parallel = True
    lm.model._eigen_multigpu_mapped = True

    if layer_gpu_map:
        gpu_counts: Dict[str, int] = {}
        for layer, gpu_id in layer_gpu_map.items():
            gpu_key = f"cuda:{gpu_id}"
            gpu_counts[gpu_key] = gpu_counts.get(gpu_key, 0) + 1
        summary = ", ".join(f"{gpu}={count} layers" for gpu, count in sorted(gpu_counts.items()))
        logger.info(f"[multigpu] layer placement summary: {summary}")
        logger.info(
            "[multigpu] explanation: this is single-process model parallelism. "
            "Each decoder block stays on one GPU and activations are moved between GPUs by hooks."
        )


def _mark_model_parallel_for_trainer(model: nn.Module, logger=None, note: str = ""):
    model.is_parallelizable = True
    model.model_parallel = True
    if logger is not None:
        suffix = f" ({note})" if note else ""
        logger.info(
            "[multigpu] mark model_parallel=True so Trainer will not move the whole model back to a single GPU"
            f"{suffix}."
        )


def _log_part(logger, scope: str, index: int, total: int, title: str, explanation: str):
    logger.info(f"[{scope}][part {index}/{total}] {title}")
    logger.info(f"[{scope}][explain] {explanation}")


class _TrainerProgressLogger:
    def __init__(self, logger, scope: str, total_steps: int, round_idx: Optional[int] = None):
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self, outer):
                self.outer = outer

            def on_train_begin(self, args, state, control, **kwargs):
                prefix = f"[{self.outer.scope}]"
                if self.outer.round_idx is not None:
                    prefix = f"{prefix}[round {self.outer.round_idx}]"
                self.outer.logger.info(
                    f"{prefix} trainer started: total_steps={self.outer.total_steps}. "
                    "This phase performs LoRA adapter optimization on the current low-rank basis."
                )

            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = logs or {}
                if state.max_steps and state.global_step:
                    percent = 100.0 * state.global_step / max(1, state.max_steps)
                    loss = logs.get("loss")
                    lr = logs.get("learning_rate")
                    extra = []
                    if loss is not None:
                        extra.append(f"loss={loss}")
                    if lr is not None:
                        extra.append(f"lr={lr}")
                    detail = ", ".join(extra) if extra else "metrics pending"
                    prefix = f"[{self.outer.scope}]"
                    if self.outer.round_idx is not None:
                        prefix = f"{prefix}[round {self.outer.round_idx}]"
                    self.outer.logger.info(
                        f"{prefix} training progress: step={state.global_step}/{state.max_steps} "
                        f"({percent:.1f}%), {detail}"
                    )

            def on_train_end(self, args, state, control, **kwargs):
                prefix = f"[{self.outer.scope}]"
                if self.outer.round_idx is not None:
                    prefix = f"{prefix}[round {self.outer.round_idx}]"
                self.outer.logger.info(
                    f"{prefix} trainer finished. Next step is to merge LoRA weights and refresh the basis."
                )

        self.callback_cls = _Callback
        self.logger = logger
        self.scope = scope
        self.total_steps = total_steps
        self.round_idx = round_idx

    def build(self):
        return self.callback_cls(self)


# ----------------------------
# QKV transform analysis (NEW)
# ----------------------------
def _walk(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """Yield (path, value) for nested dict/list/tuple structures."""
    if torch.is_tensor(obj):
        yield prefix, obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from _walk(v, p)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield from _walk(v, p)
        return


def _infer_layer_idx_from_path(path: str) -> Optional[int]:
    # Common patterns: layer_46, layers.46, layers[46], decoder.layers.46, etc.
    p = path.lower()
    m = re.search(r"(?:layer|layers)[_\.\[](\d+)", p)
    if m:
        return int(m.group(1))
    # Another common: "[46]" early in path (when basis is a list indexed by layer)
    # We only accept it if path contains 'layers' or 'decoder' or 'model' words.
    if ("layers" in p) or ("decoder" in p) or ("model" in p):
        m2 = re.search(r"\[(\d+)\]", p)
        if m2:
            return int(m2.group(1))
    return None


def _infer_qkv_kind_from_path(path: str) -> Optional[str]:
    p = path.lower()
    # Strict-ish token matching to avoid accidental matches
    def has_token(tok: str) -> bool:
        return re.search(rf"(^|[^a-z0-9]){tok}([^a-z0-9]|$)", p) is not None

    # Prioritize explicit q/k/v tokens + basis-ish words
    basisish = ("basis" in p) or ("eigen" in p) or ("u_" in p) or (".u" in p) or ("proj" in p)
    if has_token("q") and basisish:
        return "q"
    if has_token("k") and basisish:
        return "k"
    if has_token("v") and basisish:
        return "v"

    # fallback: allow 'q_basis' etc without basisish
    if "q_basis" in p or "qbasis" in p or "qproj" in p:
        return "q"
    if "k_basis" in p or "kbasis" in p or "kproj" in p:
        return "k"
    if "v_basis" in p or "vbasis" in p or "vproj" in p:
        return "v"

    return None


def _build_basis_index(basis_obj: Any) -> Dict[Tuple[int, str], List[Tuple[str, torch.Tensor]]]:
    """
    Returns mapping: (layer_idx, kind) -> list of (path, tensor)
    Only includes 2D tensors.
    """
    idx: Dict[Tuple[int, str], List[Tuple[str, torch.Tensor]]] = {}
    for path, val in _walk(basis_obj):
        if not torch.is_tensor(val) or val.ndim != 2:
            continue
        layer = _infer_layer_idx_from_path(path)
        kind = _infer_qkv_kind_from_path(path)
        if layer is None or kind is None:
            continue
        idx.setdefault((layer, kind), []).append((path, val))
    return idx


def _pick_basis_tensor(cands: List[Tuple[str, torch.Tensor]], d_model: int, logger) -> Optional[Tuple[str, torch.Tensor]]:
    if not cands:
        return None
    # prefer square [d_model, d_model] or [d_model, r]
    scored = []
    for path, t in cands:
        s0, s1 = int(t.shape[0]), int(t.shape[1])
        score = 0
        if s0 == d_model:
            score += 10
        if s1 == d_model:
            score += 5
        if s0 == s1:
            score += 2
        scored.append((score, path, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    if best[0] <= 0:
        logger.warning("Could not confidently pick basis tensor (no shape match). Using the first candidate.")
        return cands[0]
    return best[1], best[2]


@torch.no_grad()
def _svd_lowrank_prepare(W: torch.Tensor, q: int, niter: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD on CPU float32 for stability.
    Returns U,S,V with shapes U:[out,q], S:[q], V:[in,q]
    """
    Wc = W.detach().float().cpu()
    q = min(q, min(Wc.shape))
    U, S, V = torch.svd_lowrank(Wc, q=q, niter=niter)
    return U, S, V


@torch.no_grad()
def _svd_reconstruct(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, r: int) -> torch.Tensor:
    r = min(r, S.numel())
    Ur = U[:, :r]
    Sr = S[:r]
    Vr = V[:, :r]
    return (Ur * Sr) @ Vr.T


@torch.no_grad()
def _proj_reconstruct(W: torch.Tensor, U: torch.Tensor, r: int) -> Tuple[torch.Tensor, str]:
    """
    Efficient projection-based approx without forming dense P:
      left(out):  W_hat = U_r @ (U_r^T @ W)
      right(in): W_hat = (W @ U_r) @ U_r^T
    Automatically decides side based on shape.
    """
    Wc = W.detach().float().cpu()
    Uc = U.detach().float().cpu()
    out_dim, in_dim = Wc.shape
    if Uc.shape[0] == out_dim:
        U_r = Uc[:, :r]
        W_hat = U_r @ (U_r.T @ Wc)
        return W_hat, "left(out)"
    if Uc.shape[0] == in_dim:
        U_r = Uc[:, :r]
        W_hat = (Wc @ U_r) @ U_r.T
        return W_hat, "right(in)"
    raise ValueError(f"basis shape {tuple(Uc.shape)} not compatible with W {tuple(Wc.shape)}")


@torch.no_grad()
def _rel_fro_error(A: torch.Tensor, B: torch.Tensor) -> float:
    A = A.float()
    B = B.float()
    return ((A - B).norm(p="fro") / (A.norm(p="fro") + 1e-12)).item()


@torch.no_grad()
def _cos_rel_stats(y: torch.Tensor, yh: torch.Tensor) -> Dict[str, float]:
    """
    y, yh: [N, D] on CPU float32
    """
    y = y.float()
    yh = yh.float()
    cos = F.cosine_similarity(y, yh, dim=-1)  # [N]
    rel = (y - yh).norm(dim=-1) / (y.norm(dim=-1) + 1e-12)

    def pct(v, p):
        if v.numel() == 0:
            return float("nan")
        k = max(1, int(math.ceil(p * v.numel())))
        return v.kthvalue(k).values.item()

    return {
        "cos_mean": cos.mean().item(),
        "cos_p50": cos.median().item(),
        "cos_p95": pct(cos, 0.95),
        "relL2_mean": rel.mean().item(),
        "relL2_p50": rel.median().item(),
        "relL2_p95": pct(rel, 0.95),
    }


def _get_opt_layer_modules(lm, layer_idx: int):
    layer = lm.model.model.decoder.layers[layer_idx]
    attn = layer.self_attn
    return {
        "q": attn.q_proj,
        "k": attn.k_proj,
        "v": attn.v_proj,
        "Wq": attn.q_proj.weight,
        "Wk": attn.k_proj.weight,
        "Wv": attn.v_proj.weight,
    }


def _get_llama_layer_modules(lm, layer_idx: int):
    layer = lm.model.model.layers[layer_idx]
    attn = layer.self_attn
    return {
        "q": attn.q_proj,
        "k": attn.k_proj,
        "v": attn.v_proj,
        "Wq": attn.q_proj.weight,
        "Wk": attn.k_proj.weight,
        "Wv": attn.v_proj.weight,
    }


def _get_mpt_layer_modules(lm, layer_idx: int):
    # MPT packs qkv; for analysis we can only hook Wqkv (combined).
    # We keep this as a placeholder; you can extend if you split Wqkv.
    layer = lm.model.transformer.blocks[layer_idx]
    attn = layer.attn
    return {
        "qkv": attn.Wqkv,
        "Wqkv": attn.Wqkv.weight,
    }


@torch.no_grad()
def export_qkv_mats_from_model(lm, args, logger, out_path: Path, layer_idx: int):
    """
    Export Wq/Wk/Wv weights (HF Linear.weight layout [out,in]) for a single layer.
    """
    if "opt" in args.net.lower():
        m = _get_opt_layer_modules(lm, layer_idx)
        pack = {"Wq": m["Wq"].detach().cpu(), "Wk": m["Wk"].detach().cpu(), "Wv": m["Wv"].detach().cpu()}
    elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
        m = _get_llama_layer_modules(lm, layer_idx)
        pack = {"Wq": m["Wq"].detach().cpu(), "Wk": m["Wk"].detach().cpu(), "Wv": m["Wv"].detach().cpu()}
    else:
        raise RuntimeError(f"export_qkv not implemented for net={args.net}")
    torch.save(pack, out_path)
    logger.info(f"[qkv_export] saved {out_path} with shapes: {pack['Wq'].shape} {pack['Wk'].shape} {pack['Wv'].shape}")


class _IOSampler:
    """
    Forward hook sampler that collects (x, y) pairs from a Linear module:
      x = input hidden states to the projection (flattened)
      y = output of the projection
    Sampling is done on-device, but stored on CPU float32.
    """
    def __init__(self, max_vecs: int):
        self.max_vecs = int(max_vecs)
        self.count = 0
        self.x_chunks: List[torch.Tensor] = []
        self.y_chunks: List[torch.Tensor] = []

    def hook(self, module, inp, out):
        if self.count >= self.max_vecs:
            return
        if not inp:
            return
        x = inp[0]
        y = out
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            return
        # flatten [bs, seq, d] -> [N, d]
        x2 = x.reshape(-1, x.shape[-1])
        y2 = y.reshape(-1, y.shape[-1])

        remaining = self.max_vecs - self.count
        n = min(remaining, x2.shape[0])
        if n <= 0:
            return
        # random sample indices (no replacement)
        idx = torch.randperm(x2.shape[0], device=x2.device)[:n]
        self.x_chunks.append(x2.index_select(0, idx).detach().float().cpu())
        self.y_chunks.append(y2.index_select(0, idx).detach().float().cpu())
        self.count += n

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.x_chunks:
            return torch.empty(0), torch.empty(0)
        X = torch.cat(self.x_chunks, dim=0)
        Y = torch.cat(self.y_chunks, dim=0)
        return X, Y


@torch.no_grad()
def _run_forward_minibatches(lm, args, dataloader, logger, max_batches: int):
    """
    Run a small number of forward passes to trigger hooks.
    Uses the model family-specific forward used in evaluate().
    """
    use_cache = lm.model.config.use_cache
    lm.model.config.use_cache = False
    lm.model.eval()

    it = iter(dataloader)
    for b in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        # dataloader batches can vary by dataset loader implementation
        if isinstance(batch, (list, tuple)):
            # common: (input_ids, labels, ...)
            input_ids = batch[0]
        elif isinstance(batch, dict):
            input_ids = batch.get("input_ids", None)
            if input_ids is None:
                # fallback: take first tensor
                for v in batch.values():
                    if torch.is_tensor(v):
                        input_ids = v
                        break
        else:
            input_ids = batch

        if not torch.is_tensor(input_ids):
            logger.warning("[qkv_analyze] Could not parse input_ids from batch, skipping.")
            continue

        input_ids = input_ids.to(lm.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        if "opt" in args.net.lower():
            _ = lm.model.model.decoder(input_ids)
        elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
            _ = lm.model.model(input_ids)
        elif "falcon" in args.net.lower():
            _ = lm.model.transformer(input_ids)
        elif "mpt" in args.net.lower():
            _ = lm.model.transformer(input_ids)
        else:
            raise RuntimeError(f"Unsupported net for qkv analyze forward: {args.net}")

    lm.model.config.use_cache = use_cache


@torch.no_grad()
def run_qkv_transform_analysis(
    lm,
    args,
    dataloader,
    logger,
    stage: str,
    out_dir: Path,
):
    """
    Main entry: analyze selected layers' q/k/v projection matrices.

    - SVD approximation: W_hat_svd(r)
    - eigen_basis approximation: W_hat_basis(r) if basis is found
    - Activation comparison: uses real hidden inputs sampled at q_proj/k_proj/v_proj via hooks
    """
    out_dir = _mkdir(out_dir)
    analysis_path = out_dir / f"qkv_transform_{stage}.jsonl"
    logger.info(f"[qkv_analyze] stage={stage} output={analysis_path}")

    # Load eigen_basis if provided / discoverable
    basis_obj = None
    basis_idx = None
    if args.qkv_eigen_basis:
        basis_path = Path(args.qkv_eigen_basis)
    else:
        # auto-search
        candidates = []
        if args.output_dir:
            candidates.append(Path(args.output_dir) / "eigen_basis.pt")
            candidates.append(Path(args.output_dir) / "qkv_outputs" / "eigen_basis.pt")
        if args.save_dir:
            candidates.append(Path(args.save_dir) / "eigen_basis.pt")
        candidates.append(Path.cwd() / "eigen_basis.pt")
        basis_path = None
        for c in candidates:
            if c.exists():
                basis_path = c
                break

    if basis_path and Path(basis_path).exists():
        try:
            basis_obj = torch.load(str(basis_path), map_location="cpu")
            basis_idx = _build_basis_index(basis_obj)
            logger.info(f"[qkv_analyze] loaded eigen_basis from {basis_path} (indexed entries={len(basis_idx)})")
        except Exception as e:
            logger.warning(f"[qkv_analyze] failed to load eigen_basis: {e}")
            basis_obj = None
            basis_idx = None
    else:
        logger.info("[qkv_analyze] eigen_basis not found; basis-projection comparison will be skipped.")

    # Determine target layers
    if args.qkv_layers_obj == "all":
        if "opt" in args.net.lower():
            n_layers = len(lm.model.model.decoder.layers)
        elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
            n_layers = len(lm.model.model.layers)
        elif "mpt" in args.net.lower():
            n_layers = len(lm.model.transformer.blocks)
        else:
            raise RuntimeError(f"[qkv_analyze] layers=all not supported for net={args.net}")
        layer_ids = list(range(n_layers))
    else:
        layer_ids = args.qkv_layers_obj or []

    if not layer_ids:
        logger.warning("[qkv_analyze] No layers specified; skipping.")
        return

    ranks = args.qkv_ranks_obj or []
    if not ranks:
        if args.avg_dim:
            ranks = [int(args.avg_dim)]
        else:
            ranks = [448]

    max_rank = max(ranks)
    oversample = int(getattr(args, "qkv_svd_oversample", 16))

    # For activation sampling, attach hooks for selected layers and run a few minibatches
    # We'll do per-layer to avoid too many hooks at once.
    with open(analysis_path, "a", encoding="utf-8") as f_jsonl:
        for layer_idx in layer_ids:
            logger.info(f"[qkv_analyze] analyzing layer {layer_idx} (ranks={ranks})")

            if "opt" in args.net.lower():
                mods = _get_opt_layer_modules(lm, layer_idx)
                q_mod, k_mod, v_mod = mods["q"], mods["k"], mods["v"]
                Wq, Wk, Wv = mods["Wq"], mods["Wk"], mods["Wv"]
            elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
                mods = _get_llama_layer_modules(lm, layer_idx)
                q_mod, k_mod, v_mod = mods["q"], mods["k"], mods["v"]
                Wq, Wk, Wv = mods["Wq"], mods["Wk"], mods["Wv"]
            else:
                logger.warning(f"[qkv_analyze] net={args.net} not implemented for per-qkv analysis. Skipping layer.")
                continue

            d_model = int(Wq.shape[0])

            # Prepare basis tensors (if possible)
            Uq = Uk = Uv = None
            Uq_path = Uk_path = Uv_path = None
            if basis_idx is not None:
                for kind, ref in [("q", "Uq"), ("k", "Uk"), ("v", "Uv")]:
                    cands = basis_idx.get((layer_idx, kind), [])
                    picked = _pick_basis_tensor(cands, d_model, logger) if cands else None
                    if picked is None:
                        continue
                    pth, ten = picked
                    if kind == "q":
                        Uq_path, Uq = pth, ten
                    elif kind == "k":
                        Uk_path, Uk = pth, ten
                    elif kind == "v":
                        Uv_path, Uv = pth, ten

            # 1) Activation sampling via hooks (real hidden->Q/K/V)
            sam_q = _IOSampler(max_vecs=args.qkv_sample_vecs)
            sam_k = _IOSampler(max_vecs=args.qkv_sample_vecs)
            sam_v = _IOSampler(max_vecs=args.qkv_sample_vecs)

            hq = q_mod.register_forward_hook(sam_q.hook)
            hk = k_mod.register_forward_hook(sam_k.hook)
            hv = v_mod.register_forward_hook(sam_v.hook)

            try:
                _run_forward_minibatches(lm, args, dataloader, logger, max_batches=args.qkv_max_batches)
            finally:
                hq.remove()
                hk.remove()
                hv.remove()

            Xq, Yq = sam_q.get()
            Xk, Yk = sam_k.get()
            Xv, Yv = sam_v.get()

            if Xq.numel() == 0 or Xk.numel() == 0 or Xv.numel() == 0:
                logger.warning(f"[qkv_analyze] No samples collected for layer {layer_idx}. Skipping activation stats.")
            else:
                logger.info(f"[qkv_analyze] samples collected: Q={Xq.shape[0]} K={Xk.shape[0]} V={Xv.shape[0]}")

            # 2) SVD prep once per W (optional)
            do_svd = bool(args.qkv_do_svd)
            svd_q = svd_k = svd_v = None
            if do_svd:
                q_dim = min(max_rank + oversample, min(Wq.shape))
                k_dim = min(max_rank + oversample, min(Wk.shape))
                v_dim = min(max_rank + oversample, min(Wv.shape))
                logger.info(f"[qkv_analyze] randomized SVD q={q_dim} (layer {layer_idx})")
                svd_q = _svd_lowrank_prepare(Wq, q=q_dim, niter=args.qkv_svd_niter)
                svd_k = _svd_lowrank_prepare(Wk, q=k_dim, niter=args.qkv_svd_niter)
                svd_v = _svd_lowrank_prepare(Wv, q=v_dim, niter=args.qkv_svd_niter)

            # 3) For each rank, compute approximations + stats
            for r in ranks:
                record_base = {
                    "time": _now(),
                    "stage": stage,
                    "net": args.net,
                    "layer": int(layer_idx),
                    "rank": int(r),
                    "d_model": d_model,
                    "samples": int(min(Xq.shape[0], Xk.shape[0], Xv.shape[0])) if Xq.numel() else 0,
                    "basis_paths": {"q": Uq_path, "k": Uk_path, "v": Uv_path},
                }

                # --- SVD-optimal approx ---
                if do_svd and svd_q is not None:
                    U, S, V = svd_q
                    Wq_svd = _svd_reconstruct(U, S, V, r=r)
                    werr = _rel_fro_error(Wq.detach().cpu(), Wq_svd)
                    act = {}
                    if Xq.numel():
                        Yh = Xq.float() @ Wq_svd.T
                        act = _cos_rel_stats(Yq, Yh)
                    rec = dict(record_base)
                    rec.update({
                        "method": "svd",
                        "kind": "q",
                        "weight_rel_fro": werr,
                        "svd_top_singular": float(S[0].item()) if S.numel() else float("nan"),
                        "svd_energy_frac_r": float(((S[:r]**2).sum() / (S**2).sum()).item()) if S.numel() else float("nan"),
                        **{f"act_{k}": v for k,v in act.items()},
                    })
                    f_jsonl.write(json.dumps(rec) + "\n")
                    if args.qkv_save_approx:
                        torch.save({"W": Wq.detach().cpu(), "W_hat": Wq_svd}, out_dir / f"layer{layer_idx}_q_svd_r{r}.pt")

                    U, S, V = svd_k
                    Wk_svd = _svd_reconstruct(U, S, V, r=r)
                    werr = _rel_fro_error(Wk.detach().cpu(), Wk_svd)
                    act = {}
                    if Xk.numel():
                        Yh = Xk.float() @ Wk_svd.T
                        act = _cos_rel_stats(Yk, Yh)
                    rec = dict(record_base)
                    rec.update({
                        "method": "svd",
                        "kind": "k",
                        "weight_rel_fro": werr,
                        "svd_top_singular": float(S[0].item()) if S.numel() else float("nan"),
                        "svd_energy_frac_r": float(((S[:r]**2).sum() / (S**2).sum()).item()) if S.numel() else float("nan"),
                        **{f"act_{k}": v for k,v in act.items()},
                    })
                    f_jsonl.write(json.dumps(rec) + "\n")
                    if args.qkv_save_approx:
                        torch.save({"W": Wk.detach().cpu(), "W_hat": Wk_svd}, out_dir / f"layer{layer_idx}_k_svd_r{r}.pt")

                    U, S, V = svd_v
                    Wv_svd = _svd_reconstruct(U, S, V, r=r)
                    werr = _rel_fro_error(Wv.detach().cpu(), Wv_svd)
                    act = {}
                    if Xv.numel():
                        Yh = Xv.float() @ Wv_svd.T
                        act = _cos_rel_stats(Yv, Yh)
                    rec = dict(record_base)
                    rec.update({
                        "method": "svd",
                        "kind": "v",
                        "weight_rel_fro": werr,
                        "svd_top_singular": float(S[0].item()) if S.numel() else float("nan"),
                        "svd_energy_frac_r": float(((S[:r]**2).sum() / (S**2).sum()).item()) if S.numel() else float("nan"),
                        **{f"act_{k}": v for k,v in act.items()},
                    })
                    f_jsonl.write(json.dumps(rec) + "\n")
                    if args.qkv_save_approx:
                        torch.save({"W": Wv.detach().cpu(), "W_hat": Wv_svd}, out_dir / f"layer{layer_idx}_v_svd_r{r}.pt")

                # --- Basis projection approx ---
                if Uq is not None:
                    try:
                        Wq_b, side = _proj_reconstruct(Wq, Uq, r=r)
                        werr = _rel_fro_error(Wq.detach().cpu(), Wq_b)
                        act = {}
                        if Xq.numel():
                            Yh = Xq.float() @ Wq_b.T
                            act = _cos_rel_stats(Yq, Yh)
                        rec = dict(record_base)
                        rec.update({
                            "method": "eigen_basis",
                            "kind": "q",
                            "basis_side": side,
                            "weight_rel_fro": werr,
                            **{f"act_{k}": v for k,v in act.items()},
                        })
                        f_jsonl.write(json.dumps(rec) + "\n")
                        if args.qkv_save_approx:
                            torch.save({"W": Wq.detach().cpu(), "W_hat": Wq_b, "basis_side": side, "basis_path": Uq_path},
                                       out_dir / f"layer{layer_idx}_q_basis_r{r}.pt")
                    except Exception as e:
                        logger.warning(f"[qkv_analyze] basis Q failed layer{layer_idx}: {e}")

                if Uk is not None:
                    try:
                        Wk_b, side = _proj_reconstruct(Wk, Uk, r=r)
                        werr = _rel_fro_error(Wk.detach().cpu(), Wk_b)
                        act = {}
                        if Xk.numel():
                            Yh = Xk.float() @ Wk_b.T
                            act = _cos_rel_stats(Yk, Yh)
                        rec = dict(record_base)
                        rec.update({
                            "method": "eigen_basis",
                            "kind": "k",
                            "basis_side": side,
                            "weight_rel_fro": werr,
                            **{f"act_{k}": v for k,v in act.items()},
                        })
                        f_jsonl.write(json.dumps(rec) + "\n")
                        if args.qkv_save_approx:
                            torch.save({"W": Wk.detach().cpu(), "W_hat": Wk_b, "basis_side": side, "basis_path": Uk_path},
                                       out_dir / f"layer{layer_idx}_k_basis_r{r}.pt")
                    except Exception as e:
                        logger.warning(f"[qkv_analyze] basis K failed layer{layer_idx}: {e}")

                if Uv is not None:
                    try:
                        Wv_b, side = _proj_reconstruct(Wv, Uv, r=r)
                        werr = _rel_fro_error(Wv.detach().cpu(), Wv_b)
                        act = {}
                        if Xv.numel():
                            Yh = Xv.float() @ Wv_b.T
                            act = _cos_rel_stats(Yv, Yh)
                        rec = dict(record_base)
                        rec.update({
                            "method": "eigen_basis",
                            "kind": "v",
                            "basis_side": side,
                            "weight_rel_fro": werr,
                            **{f"act_{k}": v for k,v in act.items()},
                        })
                        f_jsonl.write(json.dumps(rec) + "\n")
                        if args.qkv_save_approx:
                            torch.save({"W": Wv.detach().cpu(), "W_hat": Wv_b, "basis_side": side, "basis_path": Uv_path},
                                       out_dir / f"layer{layer_idx}_v_basis_r{r}.pt")
                    except Exception as e:
                        logger.warning(f"[qkv_analyze] basis V failed layer{layer_idx}: {e}")

            f_jsonl.flush()

    logger.info(f"[qkv_analyze] done. Results: {analysis_path}")


# ----------------------------
# Evaluation (your fixed version kept)
# ----------------------------
@torch.no_grad()
def evaluate(lm, args, logger):
    """
    IMPORTANT: Do NOT convert the whole model to fp32 here.
    That was the biggest VRAM killer in your old code.
    """
    results = {}

    # Ensure correct placement for evaluation too (safe even if already placed)
    if args.multigpu:
        _place_model_multigpu(lm, args, logger)
    else:
        # single GPU / single device
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "mpt" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)

    if args.eval_ppl:
        for dataset in ["wikitext2", "c4"]:
            cache_testloader = f"{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache"
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)

            testenc = testloader if "c4" in dataset else testloader.input_ids
            nsamples = testenc.numel() // lm.seqlen

            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()

            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen): ((i + 1) * lm.seqlen)].to(lm.device)

                if "opt" in args.net.lower():
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
                    outputs = lm.model.model(batch)
                elif "falcon" in args.model:
                    outputs = lm.model.transformer(batch)
                elif "mpt" in args.net.lower():
                    outputs = lm.model.transformer(batch)
                else:
                    raise RuntimeError(f"Unsupported net for eval_ppl: {args.net}")

                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)

                # numeric guard
                logits = torch.clamp(logits, -20, 20)

                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen): ((i + 1) * lm.seqlen)][:, 1:].to(lm.device)

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("⚠ NaN/Inf detected, skipping batch")
                    continue

                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)

                if i == args.limit:
                    break

            if len(nlls) == 0:
                logger.error("❌ All batches produced NaN. PPL invalid for this model.")
                results[dataset] = float("nan")
            else:
                total_tokens = len(nlls) * lm.seqlen
                ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
                logger.info(f"{dataset} : {ppl.item()}")
                results[dataset] = ppl.item()

            lm.model.config.use_cache = use_cache

    if args.tasks != "":
        import lm_eval
        task_manager = lm_eval.tasks.TaskManager()
        t_results = lm_eval.simple_evaluate(
            model=lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            task_manager=task_manager,
        )

        results.update(t_results)
        logger.info(results)
        pprint(results)

        if "mmlu" in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}

            for key in t_results["results"].keys():
                if "hendrycksTest" not in key:
                    continue
                subject = key.split("-")[-1]
                cors = t_results["results"][key]["acc"]
                cors_norm = t_results["results"][key]["acc_norm"]
                subcats = subcategories[subject]

                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for cat in categories.keys():
                        if subcat in categories[cat]:
                            cat_cors[cat].append(cors)
                            cat_cors_norm[cat].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)

            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))

    return results


# ----------------------------
# Main
# ----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="model name or model path")
    parser.add_argument("--cache_dir", default="./HF_cache", type=str, help="cache dir for datasets / hf")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix", "pile", "alpaca"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")

    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)

    # IMPORTANT: this is "single-process multi-GPU layer mapping"
    parser.add_argument("--multigpu", action="store_true", help="map model layers to multiple gpus (single-process)")

    parser.add_argument(
        "--attn_implementation",
        type=str,
        required=False,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)

    parser.add_argument("--load_low_rank", action="store_true", help="load already compressed model")
    parser.add_argument("--load_peft_model", action="store_true", help="load the peft model")
    parser.add_argument("--peft_model_path", default=None, help="path to finetuned model")
    parser.add_argument("--save_dir", default=None, help="path to save compressed/finetuned model")
    parser.add_argument("--output_dir", default=None, help="path to save log file")

    parser.add_argument("--evaluate_baseline", action="store_true")
    parser.add_argument("--avg_dim", type=int)
    parser.add_argument("--error_budget", type=float, default=0.025)
    parser.add_argument("--fine_tune", action="store_true", help="finetune model after compressing")
    parser.add_argument("--ft_target_modules", type=str, default="", help="comma-separated PEFT target modules; default is model-specific")
    parser.add_argument("--ft_lora_r", type=int, default=64, help="LoRA rank for PEFT fine-tuning")
    parser.add_argument("--ft_lora_alpha", type=int, default=64, help="LoRA alpha for PEFT fine-tuning")
    parser.add_argument("--ft_lora_dropout", type=float, default=0.05, help="LoRA dropout for PEFT fine-tuning")
    parser.add_argument("--ft_learning_rate", type=float, default=2e-4, help="learning rate for PEFT fine-tuning")
    parser.add_argument("--ft_batch_size", type=int, default=4, help="per-device batch size for PEFT fine-tuning")
    parser.add_argument("--ft_grad_accum", type=int, default=4, help="gradient accumulation steps for PEFT fine-tuning")
    parser.add_argument("--ft_num_epochs", type=float, default=1.0, help="number of epochs for PEFT fine-tuning")
    parser.add_argument("--ft_max_steps", type=int, default=2000, help="max training steps for PEFT fine-tuning")
    parser.add_argument("--ft_save_steps", type=int, default=500, help="save interval for PEFT fine-tuning")
    parser.add_argument("--ft_logging_steps", type=int, default=10, help="logging interval for PEFT fine-tuning")
    parser.add_argument("--ft_seqlen", type=int, default=1024, help="sequence length used only for PEFT fine-tuning tokenization")
    parser.add_argument("--ft_dataset", type=str, default="c4", choices=["c4", "alpaca"], help="dataset used for PEFT fine-tuning")
    parser.add_argument("--ft_data_dir", type=str, default="", help="local data dir for fine-tuning; defaults to C4_DATA_DIR when ft_dataset=c4")
    parser.add_argument("--ft_max_train_samples", type=int, default=20000, help="maximum number of local text samples to collect for PEFT fine-tuning")
    parser.add_argument("--ft_bilevel_refresh", action="store_true", help="OPT-only scheme C: alternate inner recovery with outer eigen-basis refresh (manual loop, no LoRA)")
    parser.add_argument("--ft_steps_per_outer", type=int, default=50, help="inner fine-tune steps per outer basis-refresh round when --ft_bilevel_refresh is enabled")
    parser.add_argument("--ft_outer_rounds", type=int, default=0, help="optional number of outer basis-refresh rounds; 0 derives it from ft_max_steps / ft_steps_per_outer")
    # bilevel manual-loop controls (inspired by finetune_ppl_opt30b_eigen.py)
    parser.add_argument("--ft_inner_mode", type=str, default="lora_qkv",
                        choices=["lora_qkv", "vscale_only", "up_vscale", "qkv_up_vscale", "v_up_vscale", "basis_v", "all"],
                        help="inner training mode for scheme-C: LoRA on q/k/v or direct low-rank weight training")
    parser.add_argument("--ft_train_mode", type=str, default="qkv_up_vscale",
                        choices=["vscale_only", "up_vscale", "qkv_up_vscale", "all"],
                        help="which params to unfreeze per round: vscale_only / up_vscale / qkv_up_vscale / all")
    parser.add_argument("--ft_train_last_n", type=int, default=0,
                        help="only unfreeze last N decoder layers per round (0 = all layers)")
    parser.add_argument("--ft_grad_clip", type=float, default=0.5,
                        help="gradient clip norm for manual training loop (0 = no clip)")
    parser.add_argument("--ft_logit_clip", type=float, default=20.0,
                        help="(stability) clamp LM logits to [-clip, clip] before CE loss. 0 = disable")
    parser.add_argument("--ft_max_nonfinite", type=int, default=20,
                        help="(stability) if loss is non-finite this many times in a round, abort the round early")
    parser.add_argument("--ft_token_budget", type=int, default=500000,
                        help="token budget when building the CPU token stream for bilevel training")
    parser.add_argument("--ft_eval_every", type=int, default=0,
                        help="evaluate PPL every N inner steps during bilevel (0 = disable)")
    parser.add_argument("--save_pre_ft_checkpoint", action="store_true",
                        help="save the compressed model BEFORE bilevel fine-tuning starts (default: skip to save VRAM)")
    parser.add_argument("--ft_skip_refresh", action="store_true",
                        help="when --ft_bilevel_refresh is enabled, skip the outer eigen-basis refresh/reproject step (LoRA->merge only)")
    parser.add_argument("--ft_calib_strong", action="store_true",
                        help="strong calibration (direct-weight only): 2000 steps/round, all layers, 600k tokens, lr=2e-5 (matches finetune_ppl_opt30b_eigen.py)")
    parser.add_argument("--ft_compare_ppl", action="store_true",
                        help="print Utkarsh-aligned C4-train PPL for (compressed pre-FT) vs (post-FT) inside this run")
    # Utkarsh-aligned PPL controls (mirrors sanity_AB_opt30b.py)
    parser.add_argument("--ft_ppl_seqlen", type=int, default=2048, help="seqlen used for Utkarsh-aligned PPL valenc")
    parser.add_argument("--ft_ppl_c4_glob", type=str, default="c4-train.*.json.gz", help="glob for C4 shards under --ft_data_dir")
    parser.add_argument("--ft_ppl_scan_lines", type=int, default=200000, help="how many docs to scan when building C4 pool")
    parser.add_argument("--ft_ppl_pool_docs", type=int, default=20000, help="reservoir pool size for C4")
    parser.add_argument("--ft_ppl_val_chunks", type=int, default=256, help="number of valenc chunks for PPL")
    parser.add_argument("--ft_ppl_val_seed", type=int, default=0, help="random seed for valenc sampling")

    # NEW: dtype control (VRAM!)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="model compute dtype (A100 recommended: bf16)")

    # NEW: seqlen control
    parser.add_argument("--seqlen", type=int, default=2048, help="sequence length used for calibration/eval")

    # -------------------------
    # NEW: QKV export + analysis flags
    # -------------------------
    parser.add_argument("--qkv_export", action="store_true", help="export qkv weight matrices to .pt")
    parser.add_argument("--qkv_export_layers", type=str, default="", help='layers to export, e.g. "46" or "0,46"; "all" allowed but huge')
    parser.add_argument("--qkv_export_dir", type=str, default="", help="dir to save exported qkv mats (default: {output_dir}/qkv_exports)")

    parser.add_argument("--qkv_analyze", action="store_true", help="run qkv transform analysis (svd + eigen_basis proj + activation stats)")
    parser.add_argument("--qkv_analyze_stage", type=str, default="baseline", choices=["baseline", "after"], help="run analysis before or after eigenattn()")
    parser.add_argument("--qkv_layers", type=str, default="", help='layers to analyze, e.g. "46" or "0,46" or "all"')
    parser.add_argument("--qkv_ranks", type=str, default="", help='ranks to test, e.g. "448,896"; default uses --avg_dim')
    parser.add_argument("--qkv_eigen_basis", type=str, default="", help="path to eigen_basis.pt (optional; auto-discover if empty)")
    parser.add_argument("--qkv_max_batches", type=int, default=2, help="how many calibration minibatches to run for hook sampling")
    parser.add_argument("--qkv_sample_vecs", type=int, default=2048, help="how many token-vectors to sample per (Q/K/V) for activation stats")
    parser.add_argument("--qkv_do_svd", action="store_true", help="also compute SVD-optimal rank-r approximation (randomized SVD)")
    parser.add_argument("--qkv_save_approx", action="store_true", help="save W_hat matrices to .pt (can be large!)")
    parser.add_argument("--qkv_svd_niter", type=int, default=4, help="niter for torch.svd_lowrank")
    parser.add_argument("--qkv_svd_oversample", type=int, default=16, help="oversampling for randomized SVD")
    parser.add_argument("--repair_basis_pt", type=str, default="", help="first basis .pt for 降秩: rebuild low-rank from base (e.g. eigen_basis.pt)")
    parser.add_argument("--repair_basis_pt_headwise", type=str, default="", help="second basis .pt for 修复: headwise repair after vscale fix (e.g. eigen_basis_headwise_5712.pt). With both: 降秩 -> vscale fix -> 修复")
    parser.add_argument("--repair_keqk", type=int, default=0, choices=[0, 1], help="when rebuilding from basis, reuse q basis for k")
    parser.add_argument("--force_vscale_one", type=int, default=0, choices=[0, 1], help="force vscale/vscale_log to 1 before saving")

    args = parser.parse_args()

    # normalize empty strings
    args.qkv_export_layers_obj = _parse_int_list(args.qkv_export_layers, default=None)
    args.qkv_layers_obj = _parse_int_list(args.qkv_layers, default=None)
    args.qkv_ranks_obj = _parse_int_list(args.qkv_ranks, default=None)
    args.ft_target_modules_obj = _parse_str_list(args.ft_target_modules, default=None)
    args.qkv_eigen_basis = args.qkv_eigen_basis.strip() if args.qkv_eigen_basis else ""
    args.repair_basis_pt = args.repair_basis_pt.strip() if args.repair_basis_pt else ""
    args.repair_basis_pt_headwise = args.repair_basis_pt_headwise.strip() if args.repair_basis_pt_headwise else ""
    # 若只传了 headwise，默认用同目录下的 eigen_basis.pt 做降秩（三步流程）
    if args.repair_basis_pt_headwise and not args.repair_basis_pt:
        default_first = Path(args.repair_basis_pt_headwise).parent / "eigen_basis.pt"
        if default_first.exists():
            args.repair_basis_pt = str(default_first)
    # default: if user didn't pass --qkv_do_svd, keep it OFF (avoid extra compute)
    # You can enable with --qkv_do_svd
    args.qkv_do_svd = bool(args.qkv_do_svd)

    # ----- torchrun safety for multigpu layer mapping -----
    rank, local_rank, world_size = _dist_info()
    if world_size > 1 and args.multigpu:
        if rank != 0:
            return
    elif world_size > 1 and not args.multigpu:
        # 危险：torchrun 起了多进程但没开 --multigpu → 每个进程都会加载整份模型 → CPU/GPU 多份拷贝 → 极易 OOM
        import sys
        print(
            "[main_eigen_attn] 错误: 当前用 torchrun 启动了 {} 个进程，但未传入 --multigpu。\n"
            "每个进程都会加载完整模型，会导致 CPU/GPU 内存暴增并 OOM。\n"
            "请加上 --multigpu，这样只有 rank0 运行并把层分布到多张卡上。\n"
            "示例: torchrun ... main_eigen_attn.py ... --multigpu ...".format(world_size),
            file=sys.stderr,
        )
        sys.exit(1)

    _set_seed(args.seed)

    # Respect user-exported HF env; only set if missing.
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
        os.environ.setdefault("HF_HOME", args.cache_dir)

    if args.net is None:
        args.net = args.model.split("/")[-1]

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir) if args.output_dir else Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = utils.create_logger(output_dir)

    logger.info(args)
    logger.info(f"[dist] world_size={world_size} rank={rank} local_rank={local_rank}")

    # ----- load model -----
    _log_part(
        logger,
        "pipeline",
        1,
        7,
        "load base model",
        "Load the original checkpoint on CPU first, then cast to the requested dtype so VRAM usage stays predictable.",
    )
    args.model_family = args.net.split("-")[0]
    lm = LMClass(args)

    lm.seqlen = int(args.seqlen)
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    # enforce dtype (VRAM)
    dtype = _resolve_dtype(args.dtype)
    try:
        lm.model = lm.model.to(dtype=dtype)
    except Exception as e:
        logger.warning(f"Could not cast model to dtype={args.dtype} via .to(dtype=..): {e}")

    logger.info(f"error_budget : {args.error_budget}")
    logger.info(str(lm.model.config))

    # If baseline-only, do it early
    if args.evaluate_baseline:
        logger.info("=== Evaluating baselines ===")
        model_params, _ = get_model_size(lm)

        if "opt" in args.net.lower():
            cache_size = get_kvcache_size_opt(lm)
        elif "mpt" in args.net.lower():
            cache_size = get_kvcache_size_mpt(lm, args)
        elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
            cache_size = get_kvcache_size_llama(lm)
        else:
            cache_size = float("nan")

        logger.info(f"Baseline model parameters : {model_params/1e9} Billion")
        logger.info(f"Baseline model KV Cache Size : {cache_size} GB for batch size of 1")

        if args.multigpu:
            _place_model_multigpu(lm, args, logger)

        evaluate(lm, args, logger)
        return

    # ----- MULTI-GPU LAYER PLACEMENT BEFORE eigenattn() -----
    if args.multigpu:
        _log_part(
            logger,
            "pipeline",
            2,
            7,
            "place model across visible GPUs",
            "This step distributes decoder layers across GPUs for single-process model parallelism before decomposition starts.",
        )
        _place_model_multigpu(lm, args, logger)

    logger.info("=== start low rank decomposition ===")
    tick = time.time()

    # ----- load calibration dataset -----
    _log_part(
        logger,
        "pipeline",
        3,
        7,
        "load calibration data",
        "Calibration samples are used to estimate the low-rank subspace and determine the EigenAttn projection.",
    )
    cache_dataloader = f"{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache"
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        logger.info(f"load calibration from {cache_dataloader}")
    else:
        dataloader, _ = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
        )
        torch.save(dataloader, cache_dataloader)

    # ----- QKV export (baseline weights) -----
    if args.qkv_export:
        export_dir = Path(args.qkv_export_dir) if args.qkv_export_dir else (output_dir / "qkv_exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        layers_to_export = args.qkv_export_layers_obj
        if layers_to_export == "all":
            if "opt" in args.net.lower():
                layers_to_export = list(range(len(lm.model.model.decoder.layers)))
            elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
                layers_to_export = list(range(len(lm.model.model.layers)))
            else:
                logger.warning(f"[qkv_export] layers=all not supported for net={args.net}; skipping.")
                layers_to_export = []
        elif layers_to_export is None:
            # default: export layer 0 if nothing specified
            layers_to_export = [0]

        for li in layers_to_export:
            out_path = export_dir / f"qkv_mats_layer{li}.pt"
            export_qkv_mats_from_model(lm, args, logger, out_path=out_path, layer_idx=int(li))

    # ----- QKV analysis (baseline stage) -----
    if args.qkv_analyze and args.qkv_analyze_stage == "baseline":
        qkv_out = output_dir / "qkv_transform_analysis"
        run_qkv_transform_analysis(
            lm=lm,
            args=args,
            dataloader=dataloader,
            logger=logger,
            stage="baseline",
            out_dir=qkv_out,
        )

    args.eigen_attn_params = {
        "threshold": 1.0,
        "avg_dim_features": args.avg_dim,
        "error_budget": args.error_budget,
    }

    if not args.load_low_rank:
        _log_part(
            logger,
            "pipeline",
            4,
            7,
            "run EigenAttn decomposition",
            "This phase computes the low-rank attention representation and replaces the dense attention blocks.",
        )
        eigenattn(lm, args, dataloader, logger)
    else:
        logger.info("You are already loading a low rank compressed model.")

    # ----- QKV analysis (after stage) -----
    if args.qkv_analyze and args.qkv_analyze_stage == "after":
        qkv_out = output_dir / "qkv_transform_analysis"
        run_qkv_transform_analysis(
            lm=lm,
            args=args,
            dataloader=dataloader,
            logger=logger,
            stage="after",
            out_dir=qkv_out,
        )

    # ----- build a clean low-rank model (your original behavior) -----
    _log_part(
        logger,
        "pipeline",
        5,
        7,
        "rebuild clean low-rank checkpoint",
        "A fresh low-rank model is instantiated so the compressed weights are saved in a clean and reusable form.",
    )
    if "opt" in args.net.lower():
        layers = lm.model.model.decoder.layers
        low_rank_config = torch.zeros(len(layers), 2)
        for i in range(len(layers)):
            low_rank_config[i][0] = layers[i].self_attn.rank_kq
            low_rank_config[i][1] = layers[i].self_attn.rank_v

        config = AutoConfig.from_pretrained(
            args.model,
            attn_implementation=args.attn_implementation,
            cache_dir=args.cache_dir,
        )
        from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

        model2 = OPTForCausalLM_EigenAttn.from_pretrained(
            args.model,
            config=config,
            low_rank_config=low_rank_config,
            device_map="cpu",
            torch_dtype=torch.float16,   # store weights as fp16
            cache_dir=args.cache_dir,
        )

        layers2 = model2.model.decoder.layers
        with torch.no_grad():
            if args.repair_basis_pt or args.repair_basis_pt_headwise:
                # Load full base model on CPU once (shared by 降秩 and 修复)
                base_cpu = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=_resolve_dtype(args.dtype),
                    device_map="cpu",
                    cache_dir=args.cache_dir,
                ).eval()

                def _rebuild_layer_from_basis(base_attn, attn2, basis_dict, layer_idx, keqk):
                    uq = _get_opt_basis(basis_dict, layer_idx, "q")
                    uk = uq if keqk == 1 else _get_opt_basis(basis_dict, layer_idx, "k")
                    uv = _get_opt_basis(basis_dict, layer_idx, "v")
                    for basis, proj_name in ((uq, "q_proj"), (uk, "k_proj"), (uv, "v_proj")):
                        base_proj = getattr(base_attn, proj_name)
                        low_proj = getattr(attn2, f"{proj_name}_low")
                        w_low = (basis.t() @ base_proj.weight.detach().float()).contiguous()
                        b_low = (base_proj.bias.detach().float() @ basis).contiguous()
                        low_proj.weight.copy_(w_low.to(dtype=low_proj.weight.dtype, device=low_proj.weight.device))
                        low_proj.bias.copy_(b_low.to(dtype=low_proj.bias.dtype, device=low_proj.bias.device))
                    w_up = (base_attn.out_proj.weight.detach().float() @ uv).contiguous()
                    attn2.out_proj_up.weight.copy_(
                        w_up.to(dtype=attn2.out_proj_up.weight.dtype, device=attn2.out_proj_up.weight.device)
                    )
                    attn2.out_proj_up.bias.copy_(
                        base_attn.out_proj.bias.detach().to(
                            dtype=attn2.out_proj_up.bias.dtype,
                            device=attn2.out_proj_up.bias.device,
                        )
                    )

                use_first_basis = bool(args.repair_basis_pt)
                use_headwise_basis = bool(args.repair_basis_pt_headwise)
                if use_first_basis and use_headwise_basis:
                    # 三步流程: 1) 用 eigen_basis.pt 降秩  2) vscale fix  3) 用 eigen_basis_headwise_5712.pt 修复
                    basis_path_1 = Path(args.repair_basis_pt)
                    if not basis_path_1.exists():
                        raise FileNotFoundError(f"--repair_basis_pt not found: {basis_path_1}")
                    basis_path_2 = Path(args.repair_basis_pt_headwise)
                    if not basis_path_2.exists():
                        raise FileNotFoundError(f"--repair_basis_pt_headwise not found: {basis_path_2}")
                    logger.info("[repair] flow: 降秩 (repair_basis_pt) -> vscale fix -> 修复 (repair_basis_pt_headwise)")
                    basis_dict_1 = torch.load(basis_path_1, map_location="cpu")
                    logger.info(f"[repair] step 1/3: rebuild from basis (降秩): {basis_path_1}")
                    for i in range(len(layers2)):
                        base_attn = base_cpu.model.decoder.layers[i].self_attn
                        attn2 = layers2[i].self_attn
                        _rebuild_layer_from_basis(base_attn, attn2, basis_dict_1, i, args.repair_keqk)
                    logger.info("[repair] step 2/3: fix vscale to 1 for all layers")
                    for i in range(len(layers2)):
                        _set_attn_vscale_one(layers2[i].self_attn)
                    basis_dict_2 = torch.load(basis_path_2, map_location="cpu")
                    logger.info(f"[repair] step 3/3: rebuild from basis (修复): {basis_path_2}")
                    for i in range(len(layers2)):
                        base_attn = base_cpu.model.decoder.layers[i].self_attn
                        attn2 = layers2[i].self_attn
                        _rebuild_layer_from_basis(base_attn, attn2, basis_dict_2, i, args.repair_keqk)
                        _set_attn_v_rot_identity(attn2)
                elif use_first_basis:
                    basis_path = Path(args.repair_basis_pt)
                    if not basis_path.exists():
                        raise FileNotFoundError(f"--repair_basis_pt not found: {basis_path}")
                    logger.info(f"[repair] rebuild OPT low-rank weights from basis (single): {basis_path}")
                    basis_dict = torch.load(basis_path, map_location="cpu")
                    for i in range(len(layers2)):
                        base_attn = base_cpu.model.decoder.layers[i].self_attn
                        attn2 = layers2[i].self_attn
                        _rebuild_layer_from_basis(base_attn, attn2, basis_dict, i, args.repair_keqk)
                        _set_attn_vscale_one(attn2)
                else:
                    basis_path = Path(args.repair_basis_pt_headwise)
                    if not basis_path.exists():
                        raise FileNotFoundError(f"--repair_basis_pt_headwise not found: {basis_path}")
                    logger.info(f"[repair] rebuild OPT low-rank weights from basis (single, headwise): {basis_path}")
                    basis_dict = torch.load(basis_path, map_location="cpu")
                    for i in range(len(layers2)):
                        base_attn = base_cpu.model.decoder.layers[i].self_attn
                        attn2 = layers2[i].self_attn
                        _rebuild_layer_from_basis(base_attn, attn2, basis_dict, i, args.repair_keqk)
                        _set_attn_vscale_one(attn2)
                        _set_attn_v_rot_identity(attn2)
            else:
                for i in range(len(layers2)):
                    src_attn = layers[i].self_attn
                    dst_attn = layers2[i].self_attn

                    for proj_name in ("k_proj", "q_proj", "v_proj"):
                        src_proj = getattr(src_attn, proj_name)
                        dst_proj = getattr(dst_attn, f"{proj_name}_low")
                        dst_proj.weight.copy_(src_proj.weight.detach().to(dtype=dst_proj.weight.dtype, device=dst_proj.weight.device))
                        dst_proj.bias.copy_(src_proj.bias.detach().to(dtype=dst_proj.bias.dtype, device=dst_proj.bias.device))

                    dst_attn.out_proj_up.weight.copy_(
                        src_attn.out_proj_up.weight.detach().to(
                            dtype=dst_attn.out_proj_up.weight.dtype,
                            device=dst_attn.out_proj_up.weight.device,
                        )
                    )
                    dst_attn.out_proj_up.bias.copy_(
                        src_attn.out_proj_up.bias.detach().to(
                            dtype=dst_attn.out_proj_up.bias.dtype,
                            device=dst_attn.out_proj_up.bias.device,
                        )
                    )

                    if args.force_vscale_one == 1:
                        _set_attn_vscale_one(dst_attn)

        lm.model = model2

        if args.save_dir:
            lm.model.save_pretrained(args.save_dir)
            lm.tokenizer.save_pretrained(args.save_dir)
            torch.save(low_rank_config, os.path.join(args.save_dir, "low_rank_config.pt"))

    elif "mpt" in args.net.lower():
        layers = lm.model.transformer.blocks
        low_rank_config = torch.zeros(len(layers), 2)
        for i in range(len(layers)):
            low_rank_config[i][0] = layers[i].attn.rank_kq
            low_rank_config[i][1] = layers[i].attn.rank_v

        from models.modelling_mpt_eigen_attn import MptForCausalLM_EigenAttn
        config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation, cache_dir=args.cache_dir)
        model2 = MptForCausalLM_EigenAttn.from_pretrained(
            args.model,
            config=config,
            low_rank_config=low_rank_config,
            device_map="cpu",
            torch_dtype=torch.float16,
            cache_dir=args.cache_dir,
        )
        layers2 = model2.transformer.blocks
        for i in range(len(layers2)):
            layers2[i].attn.Wqkv_low.weight.data = layers[i].attn.Wqkv.weight.data
            layers2[i].attn.out_proj_up.weight.data = layers[i].attn.out_proj.weight.data

        lm.model = model2

        if args.save_dir:
            lm.model.save_pretrained(args.save_dir)
            lm.tokenizer.save_pretrained(args.save_dir)
            torch.save(low_rank_config, os.path.join(args.save_dir, "low_rank_config.pt"))

    elif "llama" in args.net.lower():
        layers = lm.model.model.layers
        low_rank_config = []
        for i in range(len(layers)):
            low_rank_config.append([int(layers[i].self_attn.rank_kq), int(layers[i].self_attn.rank_v)])

        from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
        config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation, cache_dir=args.cache_dir)

        model2 = LlamaForCausalLM_EigenAttn.from_pretrained(
            args.model,
            config=config,
            low_rank_config=low_rank_config,
            device_map="cpu",
            torch_dtype=torch.float16,
            cache_dir=args.cache_dir,
        )

        layers2 = model2.model.layers
        bias = config.attention_bias
        for i in range(len(layers2)):
            layers2[i].self_attn.k_proj_low.weight.data = layers[i].self_attn.k_proj_low.weight.data
            if bias:
                layers2[i].self_attn.k_proj_low.bias.data = layers[i].self_attn.k_proj_low.bias.data
            layers2[i].self_attn.k_proj_up.weight.data = layers[i].self_attn.k_proj_up.weight.data

            layers2[i].self_attn.v_proj_low.weight.data = layers[i].self_attn.v_proj_low.weight.data
            if bias:
                layers2[i].self_attn.v_proj_low.bias.data = layers[i].self_attn.v_proj_low.bias.data

            layers2[i].self_attn.o_proj_up.weight.data = layers[i].self_attn.o_proj_up.weight.data
            if bias:
                layers2[i].self_attn.o_proj_up.bias.data = layers[i].self_attn.o_proj_up.bias.data

        lm.model = model2

        if args.save_dir:
            lm.model.save_pretrained(args.save_dir)
            lm.tokenizer.save_pretrained(args.save_dir)
            torch.save(low_rank_config, os.path.join(args.save_dir, "low_rank_config.pt"))

    logger.info(f"Decomposition time: {time.time() - tick:.2f}s")

    # ------------------------------------------------------------------
    # When bilevel fine-tuning is requested we skip the pre-FT evaluate
    # and save, going straight into scheme-C training to conserve VRAM.
    # The user can opt back in with --save_pre_ft_checkpoint.
    # ------------------------------------------------------------------
    skip_pre_eval = args.fine_tune and args.ft_bilevel_refresh and not getattr(args, "save_pre_ft_checkpoint", False)

    if not skip_pre_eval:
        # ----- Evaluate compressed model -----
        _log_part(logger, "pipeline", 6, 7, "evaluate compressed model",
                  "Post-compression evaluation before optional fine-tuning.")
        logger.info("=== Evaluating compressed model ===")
        model_params, _ = get_model_size(lm)
        logger.info(f"Compressed model parameters : {model_params/1e9} Billion")

        if "opt" in args.net.lower():
            cache_size = get_kvcache_size_opt(lm)
        elif "mpt" in args.net.lower():
            cache_size = get_kvcache_size_mpt(lm, args)
        elif "llama" in args.net.lower():
            cache_size = get_kvcache_size_llama(lm)
        else:
            cache_size = float("nan")
        logger.info(f"Compressed model KV Cache Size : {cache_size} GB for batch size of 1")

        if args.multigpu:
            _place_model_multigpu(lm, args, logger)
        evaluate(lm, args, logger)

        # ----- Save compressed model -----
        _log_part(logger, "pipeline", 7, 7, "save compressed checkpoint",
                  "Persist the compressed model so fine-tuning can resume from a stable checkpoint.")
        save_dir = Path(args.output_dir) / "final_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving compressed model to: {save_dir}")
        lm.model.save_pretrained(save_dir, safe_serialization=True)
        lm.tokenizer.save_pretrained(save_dir)
        logger.info("Model saved successfully.")
    else:
        logger.info(
            "[pipeline] --ft_bilevel_refresh active and --save_pre_ft_checkpoint not set: "
            "skipping pre-FT evaluate/save to conserve VRAM."
        )
        if args.multigpu:
            _place_model_multigpu(lm, args, logger)

    # ================================================================
    # Optional fine-tune
    # ================================================================
    if args.fine_tune:
        tokenizer = lm.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        ft_data_dir = (args.ft_data_dir or "").strip() or os.environ.get("C4_DATA_DIR", "").strip()
        if not ft_data_dir:
            raise RuntimeError(
                "Fine-tune requires a local data dir. Set --ft_data_dir or export C4_DATA_DIR."
            )
        # keep args.ft_data_dir normalized for downstream PPL compare
        args.ft_data_dir = ft_data_dir

        # Optional: compare (compressed pre-FT) PPL using Utkarsh-aligned logic from sanity_AB_opt30b.py
        if args.ft_compare_ppl:
            try:
                if "opt" in args.net.lower():
                    _ppl_dev = lm.model.model.decoder.layers[0].device
                else:
                    _ppl_dev = next(lm.model.parameters()).device
                _ppl_sanity_utkarsh_c4_train(lm.model, tokenizer, args, logger, tag="pre_ft(compressed)", device=_ppl_dev)
            except Exception as e:
                logger.warning(f"[ppl][pre_ft] failed: {e}")

        if args.ft_bilevel_refresh:
            # ============================================================
            # Scheme C – bilevel refresh
            # Inner loop: LoRA adapters or direct low-rank weight training (no HF Trainer)
            # Token stream kept on CPU to save VRAM.
            # After each inner loop: optionally merge_and_unload → refresh eigen basis.
            # ============================================================
            if "opt" not in args.net.lower():
                raise RuntimeError("--ft_bilevel_refresh is currently implemented for OPT models only.")
            if not args.repair_basis_pt:
                raise RuntimeError("--ft_bilevel_refresh requires --repair_basis_pt.")

            use_lora_inner = args.ft_inner_mode == "lora_qkv"
            use_direct_weight = not use_lora_inner
            if args.ft_calib_strong and use_direct_weight:
                logger.info(
                    "[fine_tune/bilevel] --ft_calib_strong: 2000 steps/round, all layers (train_last_n=0), "
                    "token_budget=600000, lr=2e-5 (align finetune_ppl_opt30b_eigen.py)"
                )
            if use_lora_inner:
                from peft import LoraConfig, get_peft_model
                # --- LoRA target modules ---
                if args.ft_target_modules_obj is not None:
                    target_modules = args.ft_target_modules_obj
                else:
                    target_modules = ["q_proj_low", "k_proj_low", "v_proj_low"]
                peft_config = LoraConfig(
                    lora_alpha=args.ft_lora_alpha,
                    lora_dropout=args.ft_lora_dropout,
                    target_modules=target_modules,
                    r=args.ft_lora_r,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                logger.info(
                    f"[fine_tune/bilevel] inner_mode=lora_qkv LoRA config: "
                    f"target_modules={target_modules} r={args.ft_lora_r} "
                    f"alpha={args.ft_lora_alpha} dropout={args.ft_lora_dropout}"
                )
            else:
                logger.info(
                    f"[fine_tune/bilevel] inner_mode={args.ft_inner_mode} "
                    f"(direct low-rank weight training via _set_trainable_opt; train_last_n={args.ft_train_last_n})"
                )

            _log_part(logger, "fine_tune", 1, 3, "build CPU token stream",
                      "Tokenize once, keep on CPU; each step slices a random window → no GPU padding overhead.")
            basis_state = torch.load(
                args.repair_basis_pt_headwise if args.repair_basis_pt_headwise else args.repair_basis_pt,
                map_location="cpu",
            )
            dtype = _resolve_dtype(args.dtype)

            token_budget_stream = (
                600000 if (getattr(args, "ft_calib_strong", False) and use_direct_weight)
                else args.ft_token_budget
            )
            texts_iter = _iter_texts_from_dir(ft_data_dir)
            stream_cpu = _build_token_stream(
                tokenizer, texts_iter,
                target_tokens=token_budget_stream,
                seed=args.seed,
            )
            if stream_cpu is None:
                raise RuntimeError(f"No usable tokens found in: {ft_data_dir}")
            logger.info(
                f"[fine_tune/bilevel] token_stream={stream_cpu.numel()} "
                f"seqlen={args.ft_seqlen} batch_size={args.ft_batch_size}"
            )

            texts_eval = _iter_texts_from_dir(ft_data_dir)
            eval_stream_cpu = _build_token_stream(
                tokenizer, texts_eval,
                target_tokens=min(200000, args.ft_token_budget),
                seed=args.seed + 1,
            )

            if getattr(args, "ft_calib_strong", False) and use_direct_weight:
                inner_steps = 2000
                outer_rounds = 1
                total_budget_steps = 2000
            else:
                inner_steps = max(1, int(args.ft_steps_per_outer))
                outer_rounds = (
                    int(args.ft_outer_rounds)
                    if args.ft_outer_rounds > 0
                    else max(1, math.ceil(int(args.ft_max_steps) / inner_steps))
                )
                total_budget_steps = max(1, int(args.ft_max_steps))

            output_dir_ft = (
                os.path.join(args.save_dir, "bilevel_model")
                if args.save_dir
                else str(Path(args.output_dir) / "bilevel_model")
            )
            Path(output_dir_ft).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"[fine_tune/bilevel] SCHEME C – LoRA + manual loop | "
                f"total_steps={total_budget_steps} steps_per_outer={inner_steps} "
                f"outer_rounds={outer_rounds} lr={args.ft_learning_rate} "
                f"grad_accum={args.ft_grad_accum} grad_clip={args.ft_grad_clip} "
                f"seqlen={args.ft_seqlen}"
            )

            _log_part(logger, "fine_tune", 2, 3, "run bilevel rounds",
                      "Each outer round: attach LoRA → manual inner loop → merge_and_unload → refresh eigen basis.")

            model = lm.model
            model.config.use_cache = False
            finished_steps = 0

            if "opt" in args.net.lower():
                _train_device = lm.model.model.decoder.layers[0].device
            else:
                _train_device = next(lm.model.parameters()).device

            for round_idx in range(1, outer_rounds + 1):
                if finished_steps >= total_budget_steps:
                    break
                current_steps = min(inner_steps, total_budget_steps - finished_steps)
                round_dir = Path(output_dir_ft) / f"round_{round_idx:02d}"
                round_dir.mkdir(parents=True, exist_ok=True)

                logger.info(
                    f"[fine_tune/bilevel] round={round_idx}/{outer_rounds} "
                    f"inner_steps={current_steps} completed={finished_steps}/{total_budget_steps}"
                )
                logger.info(
                    "[fine_tune/bilevel][explain] "
                    "attach LoRA → manual SGD inner loop → merge LoRA → refresh eigen basis."
                )

                # --- part 1/5: attach trainable modules ---
                if use_lora_inner:
                    logger.info(f"[fine_tune/bilevel][round {round_idx}][part 1/5] attach LoRA adapters")
                    logger.info("[fine_tune/bilevel][explain] Only LoRA delta weights are trainable; backbone stays frozen.")
                    for p in model.parameters():
                        p.requires_grad_(False)
                    if hasattr(model, "gradient_checkpointing_enable"):
                        model.gradient_checkpointing_enable()
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if args.multigpu:
                        lm.model = model
                        _place_model_multigpu(lm, args, logger)
                        model = lm.model
                        _train_device = lm.model.model.decoder.layers[0].device

                    peft_model = get_peft_model(model, peft_config)
                    # 可选：只训练最后 N 层的 LoRA 参数
                    if args.ft_train_last_n and args.ft_train_last_n > 0:
                        num_layers = len(peft_model.model.model.decoder.layers)
                        cutoff = max(0, num_layers - args.ft_train_last_n)
                        for name, param in peft_model.named_parameters():
                            if "lora_" in name and "decoder.layers." in name:
                                try:
                                    idx_str = name.split("decoder.layers.")[1].split(".")[0]
                                    layer_idx = int(idx_str)
                                except Exception:
                                    layer_idx = None
                                if layer_idx is not None and layer_idx < cutoff:
                                    param.requires_grad_(False)
                    if args.multigpu:
                        _mark_model_parallel_for_trainer(peft_model, logger, note=f"round {round_idx}")
                    peft_model.print_trainable_parameters()
                    train_model = peft_model
                else:
                    if hasattr(model, "gradient_checkpointing_enable"):
                        model.gradient_checkpointing_enable()
                        logger.info("[fine_tune/bilevel] gradient_checkpointing enabled (direct-weight, match finetune_ppl)")
                    calib_train_last_n = (
                        0 if (getattr(args, "ft_calib_strong", False) and use_direct_weight)
                        else args.ft_train_last_n
                    )
                    logger.info(f"[fine_tune/bilevel][round {round_idx}][part 1/5] set trainable params via _set_trainable_opt (inner_mode={args.ft_inner_mode}, train_last_n={calib_train_last_n})")
                    _set_trainable_opt(model, mode=args.ft_inner_mode, train_last_n=calib_train_last_n)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if args.multigpu:
                        lm.model = model
                        _place_model_multigpu(lm, args, logger)
                        model = lm.model
                        _train_device = lm.model.model.decoder.layers[0].device
                    train_model = model

                # --- part 2/5: manual inner training loop ---
                logger.info(f"[fine_tune/bilevel][round {round_idx}][part 2/5] inner training loop ({current_steps} steps)")
                logger.info("[fine_tune/bilevel][explain] Manual autocast+backward; token stream stays on CPU → no GPU padding VRAM cost.")

                trainable_params = [p for p in train_model.parameters() if p.requires_grad]
                calib_lr = (
                    2e-5 if (getattr(args, "ft_calib_strong", False) and use_direct_weight)
                    else args.ft_learning_rate
                )
                optimizer = torch.optim.AdamW(
                    trainable_params, lr=calib_lr, weight_decay=0.0
                )
                loader = _batchify_stream_cpu(stream_cpu, args.ft_seqlen, args.ft_batch_size)

                train_model.train()
                optimizer.zero_grad(set_to_none=True)
                step_start = time.time()
                nonfinite_count = 0
                round_aborted = False

                for step in range(1, current_steps + 1):
                    x_cpu = next(loader)
                    x = x_cpu.to(_train_device, non_blocking=True)
                    attn = torch.ones_like(x)
                    labels = x

                    with torch.autocast(device_type="cuda", dtype=dtype):
                        out = train_model(input_ids=x, attention_mask=attn)
                    # Compute CE loss in float32 to avoid bf16 overflow -> nan (logits may be large)
                    shift_logits = out.logits[:, :-1, :].contiguous().view(-1, out.logits.size(-1))
                    shift_labels = labels[:, 1:].contiguous().view(-1)
                    if args.ft_logit_clip and args.ft_logit_clip > 0:
                        shift_logits = shift_logits.clamp(min=-args.ft_logit_clip, max=args.ft_logit_clip)
                    loss = F.cross_entropy(
                        shift_logits.float(),
                        shift_labels,
                        ignore_index=-100,
                        reduction="mean",
                    ) / max(args.ft_grad_accum, 1)

                    if not torch.isfinite(loss).all():
                        nonfinite_count += 1
                        logger.warning(
                            f"[fine_tune/bilevel][round {round_idx}] step={step} loss non-finite (nan/inf), skipping backward/step"
                        )
                        if args.ft_max_nonfinite and nonfinite_count >= args.ft_max_nonfinite:
                            logger.warning(
                                f"[fine_tune/bilevel][round {round_idx}] too many non-finite losses "
                                f"({nonfinite_count}/{args.ft_max_nonfinite}); aborting round early"
                            )
                            round_aborted = True
                            break
                    else:
                        loss.backward()

                    if step % args.ft_grad_accum == 0 or step == current_steps:
                        do_step = True
                        if args.ft_grad_clip and args.ft_grad_clip > 0:
                            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, args.ft_grad_clip)
                            if not (torch.isfinite(gnorm).item() if gnorm.numel() == 1 else torch.isfinite(gnorm).all().item()):
                                logger.warning(f"[fine_tune/bilevel][round {round_idx}] step={step} grad_norm non-finite, skipping optimizer.step()")
                                do_step = False
                        if do_step:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    nll_for_log = loss.detach().float().item() * max(args.ft_grad_accum, 1)
                    if step % args.ft_logging_steps == 0 or step == 1 or step == current_steps:
                        elapsed = time.time() - step_start
                        pct = 100.0 * step / current_steps
                        logger.info(
                            f"[fine_tune/bilevel][round {round_idx}] "
                            f"step={step}/{current_steps} ({pct:.1f}%) "
                            f"nll={nll_for_log:.6f} "
                            f"elapsed={elapsed:.1f}s"
                        )

                    if args.ft_eval_every > 0 and step % args.ft_eval_every == 0 and eval_stream_cpu is not None:
                        train_model.eval()
                        ppl = _eval_ppl_simple(
                            train_model, eval_stream_cpu, _train_device,
                            seqlen=min(2048, args.ft_seqlen * 4), stride=min(2048, args.ft_seqlen * 4),
                            dtype=dtype,
                        )
                        logger.info(f"[fine_tune/bilevel][round {round_idx}] mid-round PPL @ step {step}: {ppl:.4f}")
                        train_model.train()

                del optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if round_aborted:
                    logger.warning(
                        f"[fine_tune/bilevel][round {round_idx}] aborted due to too many non-finite losses; "
                        "skip merge/refresh/save for this round and stop bilevel fine-tuning early."
                    )
                    break

                # --- part 3/5: merge LoRA weights back into backbone (if using LoRA) ---
                logger.info(f"[fine_tune/bilevel][round {round_idx}][part 3/5] merge trainable params into backbone")
                if use_lora_inner:
                    logger.info("[fine_tune/bilevel][explain] Fold LoRA delta weights into backbone so basis refresh sees the updated effective weights.")
                    if not hasattr(peft_model, "merge_and_unload"):
                        raise RuntimeError("PEFT model does not support merge_and_unload().")
                    model = peft_model.merge_and_unload()
                    del peft_model
                else:
                    logger.info("[fine_tune/bilevel][explain] Direct low-rank weights were trained; no LoRA merge needed.")
                model.eval()
                model.config.use_cache = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # --- part 4/5: refresh eigen basis (optional) ---
                if args.ft_skip_refresh:
                    logger.info(f"[fine_tune/bilevel][round {round_idx}][part 4/5] skip eigen basis refresh (--ft_skip_refresh)")
                else:
                    logger.info(f"[fine_tune/bilevel][round {round_idx}][part 4/5] refresh eigen basis")
                    logger.info(
                        "[fine_tune/bilevel][explain] "
                        "Recompute headwise eigen subspace from merged weights and reproject q/k/v back into new basis."
                    )
                    for p in model.parameters():
                        p.requires_grad_(False)
                    _refresh_opt_basis_and_reproject(
                        model=model,
                        basis_dict=basis_state,
                        repair_keqk=args.repair_keqk,
                        logger=logger,
                        force_vscale_one=args.force_vscale_one,
                    )
                    if args.multigpu:
                        model._eigen_multigpu_mapped = False
                        lm.model = model
                        _place_model_multigpu(lm, args, logger)
                        model = lm.model
                        _train_device = lm.model.model.decoder.layers[0].device

                # --- part 5/5: end-of-round eval + save ---
                logger.info(f"[fine_tune/bilevel][round {round_idx}][part 5/5] end-of-round eval + save")
                if eval_stream_cpu is not None:
                    if args.ft_compare_ppl:
                        try:
                            ppl, info = _eval_ppl_utkarsh_opt(
                                model,
                                _build_utkarsh_like_valenc_from_pool(
                                    tokenizer,
                                    _build_doc_pool_from_c4_train_dir(
                                        args.ft_data_dir,
                                        glob_pat=str(getattr(args, "ft_ppl_c4_glob", "c4-train.*.json.gz")),
                                        pool_docs=int(getattr(args, "ft_ppl_pool_docs", 20000)),
                                        scan_lines=int(getattr(args, "ft_ppl_scan_lines", 200000)),
                                        seed=int(getattr(args, "seed", 2)),
                                    )[0],
                                    seqlen=int(getattr(args, "ft_ppl_seqlen", 2048)),
                                    n_chunks=int(getattr(args, "ft_ppl_val_chunks", 64)),
                                    val_seed=int(getattr(args, "ft_ppl_val_seed", 0)),
                                ),
                                device=_train_device,
                                seqlen=int(getattr(args, "ft_ppl_seqlen", 2048)),
                                limit=int(getattr(args, "limit", -1)),
                            )
                            logger.info(f"[fine_tune/bilevel][round {round_idx}] end-of-round Utkarsh-style PPL: {ppl:.6f} info={info}")
                        except Exception as e:
                            logger.warning(f"[fine_tune/bilevel][round {round_idx}] Utkarsh-style PPL eval failed: {e}")
                    else:
                        ppl = _eval_ppl_simple(
                            model, eval_stream_cpu, _train_device,
                            seqlen=min(2048, args.ft_seqlen * 4), stride=min(2048, args.ft_seqlen * 4),
                            dtype=dtype,
                        )
                        logger.info(f"[fine_tune/bilevel][round {round_idx}] end-of-round PPL: {ppl:.4f}")

                merged_dir = round_dir / "model"
                merged_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(merged_dir, safe_serialization=True)
                tokenizer.save_pretrained(merged_dir)
                # Save low_rank_config so sanity_AB can load CMP with matching rank
                _low_rank_list = [
                    [int(model.model.decoder.layers[i].self_attn.rank_kq), int(model.model.decoder.layers[i].self_attn.rank_v)]
                    for i in range(len(model.model.decoder.layers))
                ]
                with open(merged_dir / "low_rank_config.json", "w") as f:
                    json.dump(_low_rank_list, f, indent=2)
                torch.save(basis_state, round_dir / "eigen_basis_refreshed.pt")
                logger.info(f"[fine_tune/bilevel][round {round_idx}] saved to {merged_dir}")

                lm.model = model
                finished_steps += current_steps

            # --- save final ---
            _log_part(logger, "fine_tune", 3, 3, "save bilevel final checkpoint",
                      "Store the final merged model and refreshed basis after all outer rounds.")
            final_dir = Path(output_dir_ft) / "final_model"
            final_dir.mkdir(parents=True, exist_ok=True)
            lm.model.save_pretrained(final_dir, safe_serialization=True)
            tokenizer.save_pretrained(final_dir)
            _low_rank_list = [
                [int(lm.model.model.decoder.layers[i].self_attn.rank_kq), int(lm.model.model.decoder.layers[i].self_attn.rank_v)]
                for i in range(len(lm.model.model.decoder.layers))
            ]
            with open(final_dir / "low_rank_config.json", "w") as f:
                json.dump(_low_rank_list, f, indent=2)
            torch.save(basis_state, Path(output_dir_ft) / "final_basis.pt")
            logger.info(f"[fine_tune/bilevel] saved final model to: {final_dir}")

            if args.ft_compare_ppl:
                try:
                    _ppl_dev = lm.model.model.decoder.layers[0].device
                    _ppl_sanity_utkarsh_c4_train(lm.model, tokenizer, args, logger, tag="post_ft(final_model)", device=_ppl_dev)
                except Exception as e:
                    logger.warning(f"[ppl][post_ft] failed: {e}")

        else:
            # ============================================================
            # Plain fine-tune (HF Trainer + LoRA, unchanged)
            # ============================================================
            from datasets import load_dataset
            from peft import LoraConfig, get_peft_model
            import transformers
            from time import perf_counter

            _log_part(logger, "fine_tune", 1, 2, "load and tokenize dataset",
                      "Build supervised dataset for LoRA plain fine-tuning.")

            if args.ft_dataset == "c4":
                logger.info(f"[fine_tune] loading local C4 texts from: {ft_data_dir}")
                traindata = _build_local_c4_finetune_dataset(
                    data_dir=ft_data_dir, max_samples=args.ft_max_train_samples
                )
            else:
                ds = load_dataset("tatsu-lab/alpaca")
                ds = ds.remove_columns(["input", "output", "instruction"])
                traindata = ds["train"]

            model = lm.model
            model.config.use_cache = False

            if args.ft_target_modules_obj is not None:
                target_modules = args.ft_target_modules_obj
            elif "opt" in args.net.lower():
                target_modules = ["q_proj_low", "k_proj_low", "v_proj_low"]
            elif "mpt" in args.net.lower():
                target_modules = ["Wqkv_low", "out_proj_up"]
            elif "llama" in args.net.lower():
                target_modules = ["k_proj_low", "k_proj_up", "v_proj_low", "o_proj_up"]
            else:
                raise RuntimeError("fine_tune target_modules are not configured for this model family.")

            peft_config = LoraConfig(
                lora_alpha=args.ft_lora_alpha,
                lora_dropout=args.ft_lora_dropout,
                target_modules=target_modules,
                r=args.ft_lora_r,
                bias="none",
                task_type="CAUSAL_LM",
            )
            logger.info(f"[fine_tune] target_modules={target_modules}")
            logger.info(
                f"[fine_tune] lora_r={args.ft_lora_r} alpha={args.ft_lora_alpha} "
                f"dropout={args.ft_lora_dropout} lr={args.ft_learning_rate}"
            )

            def _tokenize_batch(batch):
                encoded = tokenizer(
                    batch["text"], truncation=True,
                    padding="max_length", max_length=args.ft_seqlen,
                )
                encoded["labels"] = [ids[:] for ids in encoded["input_ids"]]
                return encoded

            train_dataset = traindata.map(
                _tokenize_batch, batched=True,
                remove_columns=traindata.column_names,
                desc="Tokenizing",
            )

            output_dir_ft = (
                os.path.join(args.save_dir, "peft_model")
                if args.save_dir
                else str(Path(args.output_dir) / "peft_model")
            )
            training_arguments = transformers.TrainingArguments(
                output_dir=output_dir_ft,
                per_device_train_batch_size=args.ft_batch_size,
                gradient_accumulation_steps=args.ft_grad_accum,
                optim="adamw_torch",
                learning_rate=args.ft_learning_rate,
                lr_scheduler_type="linear",
                save_strategy="steps",
                save_steps=args.ft_save_steps,
                logging_steps=args.ft_logging_steps,
                num_train_epochs=args.ft_num_epochs,
                max_steps=args.ft_max_steps,
                fp16=(args.dtype == "fp16"),
                bf16=(args.dtype == "bf16"),
                push_to_hub=False,
                remove_unused_columns=False,
            )

            for param in model.parameters():
                param.requires_grad = False
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.multigpu:
                lm.model = model
                _place_model_multigpu(lm, args, logger)
                model = lm.model
                _mark_model_parallel_for_trainer(model, logger, note="plain fine-tune base model")

            model = get_peft_model(model, peft_config)
            if args.multigpu:
                _mark_model_parallel_for_trainer(model, logger, note="plain fine-tune peft wrapper")
            model.print_trainable_parameters()

            _log_part(logger, "fine_tune", 2, 2, "train LoRA adapters",
                      "HF Trainer runs LoRA fine-tuning on the compressed model.")
            trainer = transformers.Trainer(
                model=model,
                args=training_arguments,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=transformers.default_data_collator,
                callbacks=[_TrainerProgressLogger(logger, "fine_tune/plain", int(args.ft_max_steps)).build()],
            )
            t0 = time.time()
            trainer.train()
            logger.info(f"[fine_tune] training finished in {time.time()-t0:.1f}s")
            trainer.model.save_pretrained(output_dir_ft)
            tokenizer.save_pretrained(output_dir_ft)


if __name__ == "__main__":
    print(sys.argv)
    main()