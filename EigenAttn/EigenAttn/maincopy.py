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
from transformers import AutoConfig

# Your project imports
from models.LMClass import LMClass
from datautils import get_loaders
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

    if "opt" in args.net.lower():
        map_layers_to_multi_gpus(lm.model.model.decoder.layers)
        input_device = lm.model.model.decoder.layers[0].device
        output_device = lm.model.model.decoder.layers[-1].device
        lm._device = input_device
        assert input_device == output_device
        lm.model.model.decoder.embed_positions.to(input_device)
        lm.model.model.decoder.embed_tokens.to(input_device)
        lm.model.model.decoder.final_layer_norm.to(output_device)
        lm.model.lm_head.to(output_device)

    elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
        map_layers_to_multi_gpus(lm.model.model.layers)
        input_device = lm.model.model.layers[0].device
        output_device = lm.model.model.layers[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.model.embed_tokens.to(input_device)
        lm.model.model.norm.to(output_device)
        lm.model.lm_head.to(output_device)

    elif "falcon" in args.net.lower():
        map_layers_to_multi_gpus(lm.model.transformer.h)
        input_device = lm.model.transformer.h[0].device
        output_device = lm.model.transformer.h[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.transformer.word_embeddings.to(input_device)
        lm.model.transformer.ln_f.to(output_device)
        lm.model.lm_head.to(output_device)

    elif "mpt" in args.net.lower():
        map_layers_to_multi_gpus(lm.model.transformer.blocks)
        input_device = lm.model.transformer.blocks[0].device
        output_device = lm.model.transformer.blocks[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.transformer.wte.to(input_device)
        lm.model.transformer.norm_f.to(output_device)
        lm.model.lm_head.to(output_device)

    else:
        logger.warning(f"multigpu mapping not implemented for net={args.net}")


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

    args = parser.parse_args()

    # normalize empty strings
    args.qkv_export_layers_obj = _parse_int_list(args.qkv_export_layers, default=None)
    args.qkv_layers_obj = _parse_int_list(args.qkv_layers, default=None)
    args.qkv_ranks_obj = _parse_int_list(args.qkv_ranks, default=None)
    args.qkv_eigen_basis = args.qkv_eigen_basis.strip() if args.qkv_eigen_basis else ""
    # default: if user didn't pass --qkv_do_svd, keep it OFF (avoid extra compute)
    # You can enable with --qkv_do_svd
    args.qkv_do_svd = bool(args.qkv_do_svd)

    # ----- torchrun safety for multigpu layer mapping -----
    rank, local_rank, world_size = _dist_info()
    if world_size > 1 and args.multigpu:
        if rank != 0:
            return

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
        _place_model_multigpu(lm, args, logger)

    logger.info("=== start low rank decomposition ===")
    tick = time.time()

    # ----- load calibration dataset -----
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
        for i in range(len(layers2)):
            layers2[i].self_attn.k_proj_low.weight.data = layers[i].self_attn.k_proj.weight.data
            layers2[i].self_attn.k_proj_low.bias.data = layers[i].self_attn.k_proj.bias.data

            layers2[i].self_attn.q_proj_low.weight.data = layers[i].self_attn.q_proj.weight.data
            layers2[i].self_attn.q_proj_low.bias.data = layers[i].self_attn.q_proj.bias.data

            layers2[i].self_attn.v_proj_low.weight.data = layers[i].self_attn.v_proj.weight.data
            layers2[i].self_attn.v_proj_low.bias.data = layers[i].self_attn.v_proj.bias.data

            layers2[i].self_attn.out_proj_up.weight.data = layers[i].self_attn.out_proj_up.weight.data
            layers2[i].self_attn.out_proj_up.bias.data = layers[i].self_attn.out_proj_up.bias.data

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

    # ----- Evaluate compressed model -----
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

    # Put compressed model on requested devices for eval
    if args.multigpu:
        _place_model_multigpu(lm, args, logger)

    evaluate(lm, args, logger)

    # ----- Save final model -----
    save_dir = Path(args.output_dir) / "final_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving compressed model to: {save_dir}")

    lm.model.save_pretrained(save_dir, safe_serialization=True)
    lm.tokenizer.save_pretrained(save_dir)
    logger.info("✅ Model saved successfully.")

    # Optional fine-tune (kept as-is, but note: this section can OOM easily on 30B)
    if args.fine_tune:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import transformers
        from time import perf_counter

        ds = load_dataset("tatsu-lab/alpaca")
        ds = ds.remove_columns(["input", "output", "instruction"])
        traindata = ds["train"]

        model = lm.model
        model.config.use_cache = False

        tokenizer = lm.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        lora_alpha = 64
        lora_dropout = 0.05
        lora_r = 64

        if "mpt" in args.net.lower():
            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["Wqkv_low", "out_proj_up"],
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif "llama" in args.net.lower():
            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["k_proj_low", "k_proj_up", "v_proj_low", "o_proj_up"],
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            raise RuntimeError("fine_tune only implemented for mpt/llama in this script.")

        output_dir_ft = os.path.join(args.save_dir, "peft_model") if args.save_dir else str(Path(args.output_dir) / "peft_model")

        training_arguments = transformers.TrainingArguments(
            output_dir=output_dir_ft,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            optim="adamw_torch",
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            save_strategy="steps",
            save_steps=500,
            logging_steps=10,
            num_train_epochs=1,
            max_steps=2000,
            fp16=True,
            push_to_hub=False,
        )

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        from trl import SFTTrainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=traindata,
            peft_config=peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=args.seqlen,
        )

        start_time = perf_counter()
        trainer.train()
        end_time = perf_counter()
        print(f"Time taken for training: {end_time - start_time} seconds")


if __name__ == "__main__":
    print(sys.argv)
    main()