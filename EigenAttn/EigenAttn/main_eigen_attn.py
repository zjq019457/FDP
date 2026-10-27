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
import shutil
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

from models.low_rank_balance import (
    OrthogonalRebalanceRank,
    LowRankColumnBalancer,
    apply_low_rank_balance_to_opt_model,
    collect_orth_balance_regularizers,
    parse_balance_target,
)

# (Optional) these are imported in your original file; keep them if other modules use them
from models.modelling_mpt_eigen_attn import MptEigenAttention
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
    "Llama-3.1-8B",
    "Llama-3.1-8b",
    "Llama-3-80b",
    "Llama-3.1-70B",
    "Llama-3.1-70b",
    "Qwen2.5-7B",
    "qwen2.5-7b",
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
        # Support both "1,2,3" and range tokens like "32-47" (inclusive).
        if "-" in x:
            a, b = x.split("-", 1)
            a = a.strip()
            b = b.strip()
            if a == "" or b == "":
                raise ValueError(f"invalid range token: {x!r}")
            ia = int(a)
            ib = int(b)
            if ia <= ib:
                out.extend(list(range(ia, ib + 1)))
            else:
                out.extend(list(range(ia, ib - 1, -1)))
        else:
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


class LowRankDoRALinear(nn.Module):
    """
    DoRA for compressed low-rank projection.
    W_eff = normalize_out(W_base + B@A) * diag(s)
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
        use_lora_scaling: bool = False,
        s_min_ratio: float = 0.25,
        s_max_ratio: float = 4.0,
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LowRankDoRALinear expects nn.Linear, got {type(base_linear)}")
        self.base = base_linear
        self.eps = float(eps)
        self.r = int(max(1, r))
        self.alpha = float(alpha)
        self.use_lora_scaling = bool(use_lora_scaling)
        self.scaling = (self.alpha / float(self.r)) if self.use_lora_scaling else 1.0
        self.s_min_ratio = float(s_min_ratio)
        self.s_max_ratio = float(s_max_ratio)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        out_features, in_features = self.base.weight.shape
        base_device = self.base.weight.device
        # freeze base low-rank weight; only DoRA params are trainable
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # A: [r, in], B: [out, r]
        self.dora_A = nn.Parameter(torch.zeros(self.r, in_features, dtype=self.base.weight.dtype, device=base_device))
        self.dora_B = nn.Parameter(torch.zeros(out_features, self.r, dtype=self.base.weight.dtype, device=base_device))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.dora_A, a=math.sqrt(5.0))
            self.dora_B.zero_()
            # magnitude initialized by output-channel norm of base weight
            mag0 = self.base.weight.detach().float().norm(dim=1).clamp_min(self.eps)
        self.register_buffer("dora_s0", mag0.clone(), persistent=False)
        self.dora_s = nn.Parameter(mag0.to(dtype=self.base.weight.dtype))

    def _effective_weight(self) -> torch.Tensor:
        w = self.base.weight
        if self.orth_core is not None:
            w = self.orth_core.left_multiply(w)
        if self.col_bal is not None:
            w = w * self.col_bal.column_scale_for_weight().unsqueeze(0)
        delta = torch.matmul(self.dora_B, self.dora_A) * self.scaling
        w = w + delta
        # output-channel normalization (row-wise, out_features axis)
        w_norm = w.float().norm(dim=1, keepdim=True).clamp_min(self.eps).to(dtype=w.dtype)
        w_hat = w / w_norm
        # Keep magnitude in a safe envelope to avoid score explosions.
        s_min = (self.dora_s0 * self.s_min_ratio).to(dtype=self.dora_s.dtype, device=self.dora_s.device)
        s_max = (self.dora_s0 * self.s_max_ratio).to(dtype=self.dora_s.dtype, device=self.dora_s.device)
        s_safe = torch.clamp(self.dora_s, min=s_min, max=s_max)
        w_eff = w_hat * s_safe.unsqueeze(1)
        return w_eff

    @property
    def weight(self) -> torch.Tensor:
        # Compatibility: some attention implementations read `.weight` directly.
        return self._effective_weight()

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.dropout(x)
        w_eff = self._effective_weight()
        return F.linear(x_in, w_eff, self.base.bias)

    @torch.no_grad()
    def merge_into_base(self):
        w_eff = self._effective_weight()
        self.base.weight.copy_(w_eff.to(dtype=self.base.weight.dtype, device=self.base.weight.device))
        if self.orth_core is not None:
            self.orth_core.reset_to_identity()
        if self.col_bal is not None:
            self.col_bal.log_gamma.zero_()


def _build_dora_param_groups_opt(model: nn.Module, args, base_lr: float):
    """Q/K/V (and optional s) use separate LR groups for stability."""
    lr_q = float(args.ft_dora_lr_q) if float(args.ft_dora_lr_q) > 0 else float(base_lr)
    lr_k = float(args.ft_dora_lr_k) if float(args.ft_dora_lr_k) > 0 else float(base_lr) * 0.5
    lr_v = float(args.ft_dora_lr_v) if float(args.ft_dora_lr_v) > 0 else float(base_lr)
    lr_s = float(args.ft_dora_lr_s) if float(args.ft_dora_lr_s) > 0 else min(lr_q, lr_k, lr_v) * 0.5
    orth_lr = float(getattr(args, "ft_orth_lr", -1.0))
    if orth_lr <= 0:
        orth_lr = float(base_lr) * 0.5
    bqk = float(getattr(args, "balance_lr_mult_k", 0.5))
    bqv = float(getattr(args, "balance_lr_mult_v", 1.0))

    # Re-build by named_parameters for orth/col separation
    q_params, k_params, v_params = [], [], []
    s_params = []
    orth_q, orth_k, orth_v = [], [], []
    col_q, col_k, col_v = [], [], []
    other = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "dora_s" in n:
            s_params.append(p)
        elif ".q_proj_low." in n:
            if "orth_core" in n:
                orth_q.append(p)
            elif "col_bal" in n:
                col_q.append(p)
            else:
                q_params.append(p)
        elif ".k_proj_low." in n:
            if "orth_core" in n:
                orth_k.append(p)
            elif "col_bal" in n:
                col_k.append(p)
            else:
                k_params.append(p)
        elif ".v_proj_low." in n:
            if "orth_core" in n:
                orth_v.append(p)
            elif "col_bal" in n:
                col_v.append(p)
            else:
                v_params.append(p)
        elif ".self_attn.out_proj_up." in n:
            v_params.append(p)
        else:
            other.append(p)

    groups = []
    wd = float(getattr(args, "ft_dora_weight_decay", 0.0))
    if q_params:
        groups.append({"params": q_params, "lr": lr_q, "weight_decay": wd})
    if k_params:
        groups.append({"params": k_params, "lr": lr_k, "weight_decay": wd})
    if v_params:
        groups.append({"params": v_params, "lr": lr_v, "weight_decay": wd})
    if s_params:
        groups.append({"params": s_params, "lr": lr_s, "weight_decay": 0.0})
    # Orthogonal + column balancer LRs (K more conservative on column)
    if orth_q:
        groups.append({"params": orth_q, "lr": orth_lr, "weight_decay": 0.0})
    if orth_k:
        groups.append({"params": orth_k, "lr": orth_lr * bqk, "weight_decay": 0.0})
    if orth_v:
        groups.append({"params": orth_v, "lr": orth_lr * bqv, "weight_decay": 0.0})
    if col_q:
        groups.append({"params": col_q, "lr": lr_q * 0.5, "weight_decay": 0.0})
    if col_k:
        groups.append({"params": col_k, "lr": lr_k * bqk * 0.5, "weight_decay": 0.0})
    if col_v:
        groups.append({"params": col_v, "lr": lr_v * bqv * 0.5, "weight_decay": 0.0})
    if other:
        groups.append({"params": other, "lr": float(base_lr), "weight_decay": wd})
    return groups


def _balance_branch_from_name(nm: str) -> Optional[str]:
    return {"q_proj_low": "q", "k_proj_low": "k", "v_proj_low": "v"}.get(nm, None)


def _maybe_enable_out_proj_up_trainable(model: nn.Module, args) -> None:
    if not int(getattr(args, "train_o_proj_up_with_v_balance", 0)):
        return
    last_n = int(getattr(args, "ft_train_last_n", 0) or 0)
    layers = list(model.model.decoder.layers)
    if last_n > 0:
        layers = layers[-last_n:]
    for layer in layers:
        ou = layer.self_attn.out_proj_up
        ou.weight.requires_grad_(True)
        if ou.bias is not None:
            ou.bias.requires_grad_(True)


def _transfer_balance_modules_into_dora(attn: nn.Module, dora_mod: LowRankDoRALinear, nm: str) -> None:
    br = _balance_branch_from_name(nm)
    if br is None:
        return
    core = getattr(attn, f"balance_{br}_core", None)
    col = getattr(attn, f"balance_{br}_col", None)
    if core is not None:
        dora_mod.register_module("orth_core", core)
        setattr(attn, f"balance_{br}_core", None)
    if col is not None:
        dora_mod.register_module("col_bal", col)
        setattr(attn, f"balance_{br}_col", None)


def _attach_lowrank_dora_opt(model: nn.Module, args):
    """
    Inject LowRankDoRALinear into OPT self-attn low-rank Q/K/V (and optional out_proj_up).
    """
    target_names = _parse_str_list(args.ft_target_modules, default=["q_proj_low", "k_proj_low", "v_proj_low"]) or []
    include_out_up = bool(int(getattr(args, "ft_dora_include_out_up", 0)))
    for p in model.parameters():
        p.requires_grad_(False)

    wrapped = 0
    layers = list(model.model.decoder.layers)
    if int(getattr(args, "ft_train_last_n", 0)) > 0:
        layers = layers[-int(args.ft_train_last_n):]
    for layer in layers:
        attn = layer.self_attn
        names = list(target_names)
        if include_out_up and "out_proj_up" not in names:
            names.append("out_proj_up")
        for nm in names:
            if not hasattr(attn, nm):
                continue
            mod = getattr(attn, nm)
            if isinstance(mod, LowRankDoRALinear):
                continue
            if not isinstance(mod, nn.Linear):
                continue
            dora_mod = LowRankDoRALinear(
                base_linear=mod,
                r=int(args.ft_dora_r),
                alpha=float(args.ft_dora_alpha),
                dropout=float(args.ft_dora_dropout),
                eps=float(args.ft_dora_eps),
                use_lora_scaling=bool(int(getattr(args, "ft_dora_use_lora_scaling", 0))),
                s_min_ratio=float(args.ft_dora_k_s_min_ratio) if nm == "k_proj_low" else float(args.ft_dora_s_min_ratio),
                s_max_ratio=float(args.ft_dora_k_s_max_ratio) if nm == "k_proj_low" else float(args.ft_dora_s_max_ratio),
            )
            _transfer_balance_modules_into_dora(attn, dora_mod, nm)
            setattr(attn, nm, dora_mod)
            wrapped += 1
            # 若环境变量 EIGEN_USE_DORA=1，OPTEigenAttention 在 q/k/v_proj_low 之后还会再过一层 q_dora/k_dora/v_dora。
            # 这里已在 low-rank 投影内做 DoRA，必须关掉后置 DoRA，否则双重 DoRA 极易几步内 NaN/Inf。
            if nm == "q_proj_low" and hasattr(attn, "q_dora"):
                attn.q_dora = nn.Identity()
            elif nm == "k_proj_low" and hasattr(attn, "k_dora"):
                attn.k_dora = nn.Identity()
            elif nm == "v_proj_low" and hasattr(attn, "v_dora"):
                attn.v_dora = nn.Identity()
    return wrapped


@torch.no_grad()
def _merge_and_unload_lowrank_dora_opt(model: nn.Module):
    merged = 0
    for layer in model.model.decoder.layers:
        attn = layer.self_attn
        for nm in ("q_proj_low", "k_proj_low", "v_proj_low", "out_proj_up"):
            if not hasattr(attn, nm):
                continue
            mod = getattr(attn, nm)
            if isinstance(mod, LowRankDoRALinear):
                br = _balance_branch_from_name(nm)
                orth = getattr(mod, "orth_core", None)
                col = getattr(mod, "col_bal", None)
                mod.merge_into_base()
                setattr(attn, nm, mod.base)
                if br is not None and orth is not None:
                    attn.register_module(f"balance_{br}_core", orth)
                if br is not None and col is not None:
                    attn.register_module(f"balance_{br}_col", col)
                merged += 1
    return merged


@torch.no_grad()
def _collect_dora_stats_opt(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Collect lightweight DoRA magnitude stats for q/k/v branches.
    Returns per-branch min/median/max and ratio-to-init statistics.
    """
    buckets: Dict[str, List[torch.Tensor]] = {"q": [], "k": [], "v": []}
    ratios: Dict[str, List[torch.Tensor]] = {"q": [], "k": [], "v": []}
    for layer in model.model.decoder.layers:
        attn = layer.self_attn
        for branch, nm in (("q", "q_proj_low"), ("k", "k_proj_low"), ("v", "v_proj_low")):
            if not hasattr(attn, nm):
                continue
            mod = getattr(attn, nm)
            if not isinstance(mod, LowRankDoRALinear):
                continue
            s = mod.dora_s.detach().float().cpu()
            s0 = mod.dora_s0.detach().float().cpu().clamp_min(1e-9)
            buckets[branch].append(s)
            ratios[branch].append((s / s0).clamp_min(0.0))

    out: Dict[str, Dict[str, float]] = {}
    for branch in ("q", "k", "v"):
        if not buckets[branch]:
            continue
        s_all = torch.cat(buckets[branch], dim=0)
        r_all = torch.cat(ratios[branch], dim=0)
        out[branch] = {
            "s_min": float(s_all.min().item()),
            "s_med": float(s_all.median().item()),
            "s_max": float(s_all.max().item()),
            "ratio_min": float(r_all.min().item()),
            "ratio_med": float(r_all.median().item()),
            "ratio_max": float(r_all.max().item()),
        }
    return out


def _load_repair_teacher_opt(teacher_path: str, dtype_str: str, logger) -> nn.Module:
    """Load full-rank OPT teacher entirely on CPU (minimal GPU VRAM). RAM ~60GB for 30B fp16/bf16."""
    if dtype_str == "bf16":
        td = torch.bfloat16
    elif dtype_str == "fp32":
        td = torch.float32
    else:
        td = torch.float16
    logger.info(
        f"[repair/teacher] Loading teacher from {teacher_path!r} on CPU, dtype={td} "
        f"(OPT-30B needs ~60GB system RAM at fp16/bf16)."
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=td,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _decoder_last_layer(model: nn.Module):
    """Works for OPTForCausalLM and OPTForCausalLM_EigenAttn."""
    return model.model.decoder.layers[-1]


def forward_student_with_last_hidden_capture(train_model, x, attn_mask, dtype, autocast_enabled: bool):
    """Single student forward; returns (output, last_layer_hidden [B,L,D])."""
    cap: List[torch.Tensor] = []

    def _hook(_m, _inp, out_l):
        t = out_l[0] if isinstance(out_l, tuple) else out_l
        cap.append(t)

    lay = _decoder_last_layer(train_model)
    h = lay.register_forward_hook(_hook)
    try:
        if autocast_enabled and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = train_model(input_ids=x, attention_mask=attn_mask, use_cache=False, return_dict=True)
        else:
            out = train_model(input_ids=x, attention_mask=attn_mask, use_cache=False, return_dict=True)
    finally:
        h.remove()
    if not cap:
        raise RuntimeError("last-layer hook did not capture hidden states")
    return out, cap[0]


def _teacher_forward_prefix_cpu(teacher_model: nn.Module, x_cpu: torch.Tensor, attn_cpu: torch.Tensor):
    """Teacher on CPU; returns (logits, last_hidden on CPU)."""
    cap: List[torch.Tensor] = []

    def _hook(_m, _inp, out_l):
        t = out_l[0] if isinstance(out_l, tuple) else out_l
        cap.append(t)

    lay = _decoder_last_layer(teacher_model)
    h = lay.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            tout = teacher_model(
                input_ids=x_cpu,
                attention_mask=attn_cpu,
                use_cache=False,
                return_dict=True,
            )
    finally:
        h.remove()
    if not cap:
        raise RuntimeError("teacher hook failed")
    return tout.logits, cap[0]


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


def _build_fixed_eval_batches_cpu(
    stream_cpu: torch.Tensor,
    seqlen: int,
    batch_size: int,
    num_batches: int,
    seed: int,
) -> List[torch.Tensor]:
    """Build deterministic anchor batches on CPU for direction calibration / checks."""
    N = stream_cpu.numel()
    assert N > seqlen + 1, f"Token stream ({N}) too short for seqlen={seqlen}"
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    batches: List[torch.Tensor] = []
    for _ in range(max(1, int(num_batches))):
        starts = torch.randint(0, N - seqlen - 1, (batch_size,), generator=g)
        x = torch.stack([stream_cpu[s : s + seqlen] for s in starts])
        batches.append(x)
    return batches


@torch.no_grad()
def _collect_reference_logits(
    model: nn.Module,
    batches_cpu: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    logit_clip: float = 20.0,
) -> List[torch.Tensor]:
    """Cache frozen reference logits from the pre-FT compressed model."""
    refs: List[torch.Tensor] = []
    was_training = model.training
    use_cache = getattr(model.config, "use_cache", None) if hasattr(model, "config") else None
    if use_cache is not None:
        model.config.use_cache = False
    model.eval()
    for x_cpu in batches_cpu:
        x = x_cpu.to(device, non_blocking=True)
        attn = torch.ones_like(x)
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=x, attention_mask=attn)
        logits = out.logits[:, :-1, :].detach().float().cpu()
        if logit_clip and logit_clip > 0:
            logits = logits.clamp(min=-logit_clip, max=logit_clip)
        refs.append(logits.contiguous())
    if use_cache is not None:
        model.config.use_cache = use_cache
    if was_training:
        model.train()
    return refs


def _save_opt_checkpoint_with_metadata(
    model: nn.Module,
    tokenizer,
    target_dir: Path,
    basis_state: Optional[Dict[str, torch.Tensor]] = None,
    basis_path: Optional[Path] = None,
):
    """Save OPT low-rank model plus low_rank_config metadata."""
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target_dir, safe_serialization=True)
    tokenizer.save_pretrained(target_dir)
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        low_rank_list = [
            [int(model.model.decoder.layers[i].self_attn.rank_kq), int(model.model.decoder.layers[i].self_attn.rank_v)]
            for i in range(len(model.model.decoder.layers))
        ]
        with open(target_dir / "low_rank_config.json", "w") as f:
            json.dump(low_rank_list, f, indent=2)
    if basis_state is not None and basis_path is not None:
        torch.save(basis_state, basis_path)


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
        shift_logits = shift_logits.clamp(min=-20.0, max=20.0)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits.float(), shift_labels, ignore_index=-100, reduction="mean")
        if not torch.isfinite(loss):
            continue
        nll_sum += loss.detach().float().item() * (inp.numel() - 1)
        tok_count += inp.numel() - 1
        if end >= N:
            break
    if tok_count == 0:
        return float("nan")
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
    used = 0

    for i in range(nsamples):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(device)

        outputs = model.model.decoder(batch)
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)

        # numeric guard (align with your main evaluate() logic)
        logits = torch.clamp(logits, -20, 20)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].to(model.lm_head.weight.device)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        if not torch.isfinite(loss).all():
            continue
        nlls.append(loss.float() * seqlen)
        used += 1
        if i == limit:
            break

    if used == 0:
        return float("nan"), {"nsamples": nsamples, "used_chunks": 0}

    ppl = torch.exp(torch.stack(nlls).sum() / (used * seqlen))
    return ppl.item(), {"nsamples": nsamples, "used_chunks": used}


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
      up_only        – out_proj_up only
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
        if mode in ("vscale_only", "up_vscale", "qkv_up_vscale", "v_up_vscale") and hasattr(attn, "vscale_log"):
            try:
                attn.vscale_log.requires_grad_(True)
            except Exception:
                pass
        if mode in ("up_only", "up_vscale", "qkv_up_vscale", "v_up_vscale"):
            if hasattr(attn, "out_proj_up"):
                _mark(attn.out_proj_up)
        if mode == "qkv_up_vscale":
            for nm in ("q_proj_low", "k_proj_low", "v_proj_low"):
                if hasattr(attn, nm):
                    _mark(getattr(attn, nm))
        if mode == "v_up_vscale":
            if hasattr(attn, "v_proj_low"):
                _mark(attn.v_proj_low)


def _set_trainable_opt_layers(model: nn.Module, layer_indices: List[int], mode: str = "up_vscale"):
    """Like _set_trainable_opt, but only unfreezes selected decoder layer indices."""
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
    n = len(layers)
    idxs = sorted({int(i) for i in layer_indices if 0 <= int(i) < n})
    for i in idxs:
        attn = layers[i].self_attn
        if mode in ("vscale_only", "up_vscale", "qkv_up_vscale", "v_up_vscale") and hasattr(attn, "vscale_log"):
            try:
                attn.vscale_log.requires_grad_(True)
            except Exception:
                pass
        if mode in ("up_only", "up_vscale", "qkv_up_vscale", "v_up_vscale"):
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
    # Align with EigenAttn_base: state_dict sweep + 16/8 GB normalization (bf16).
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
        # HF LlamaModel keeps shared RoPE on the model (not inside each decoder layer). map_layers_to_multi_gpus
        # only moves layers/embed/lm_head; if rotary_emb stays on CPU, rotary forward hits CPU/GPU mix (inv_freq vs positions).
        if hasattr(lm.model.model, "rotary_emb") and lm.model.model.rotary_emb is not None:
            lm.model.model.rotary_emb.to(input_device)
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
            basis_obj = torch.load(str(basis_path), map_location="cpu", weights_only=False)
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
                testloader = torch.load(cache_testloader, weights_only=False)
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

                # OPT needs logit clamp for fp16 CE stability; Llama/MPT large vocab must not clamp
                # (matches sanity_AB_opt30b.eval_ppl_utkarsh_opt / paper PPL protocol).
                if "opt" in args.net.lower():
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
        _inc = (
            os.environ.get("LM_EVAL_INCLUDE_PATH", "").strip()
            or os.environ.get("EIGENATTN_TASK_INCLUDE_PATH", "").strip()
        )
        tm_kw = {}
        if _inc:
            tm_kw["include_path"] = _inc
            logger.info(
                f"[lm_eval] include_path={_inc!r} "
                f"(from LM_EVAL_INCLUDE_PATH or EIGENATTN_TASK_INCLUDE_PATH)"
            )
        task_manager = lm_eval.tasks.TaskManager(**tm_kw)
        _simple_kw: dict = dict(
            model=lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            task_manager=task_manager,
            batch_size=int(getattr(args, "batch_size", 1) or 1),
        )
        if int(getattr(args, "limit", -1)) >= 0:
            _simple_kw["limit"] = int(args.limit)
        t_results = lm_eval.simple_evaluate(**_simple_kw)

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


def _default_eigen_basis_pt() -> str:
    """Absolute path to packaged default: EigenAttn/qkv_outputs/eigen_basis.pt (not CWD-relative)."""
    return str(Path(__file__).resolve().parent / "qkv_outputs" / "eigen_basis.pt")


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
    parser.add_argument(
        "--wikitext2_data_dir",
        type=str,
        default="",
        help="Local folder with wiki.train.raw / wiki.test.raw (wikitext-2-raw-v1). "
        "Sets WIKITEXT2_DATA_DIR so calibration/eval can run without the HF Hub.",
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
    parser.add_argument(
        "--compress_only",
        action="store_true",
        help="run compression/repair/eval/save only; force skip all fine-tuning branches",
    )
    # Residual mixer (方案1)：静态 MLP 补偿（不依赖 block summary）
    parser.add_argument(
        "--resid_mixer_alpha",
        type=float,
        default=0.0,
        help="residual mixer strength; 0 disables (after attention residual)",
    )
    parser.add_argument(
        "--resid_mixer_hidden_dim",
        type=int,
        default=0,
        help="residual mixer hidden dim; 0=auto(embed_dim//8)",
    )
    parser.add_argument(
        "--resid_mixer_ln_eps",
        type=float,
        default=1e-5,
        help="LayerNorm eps for residual mixer",
    )
    parser.add_argument("--ft_target_modules", type=str, default="", help="comma-separated PEFT target modules; default is model-specific")
    parser.add_argument("--ft_lora_r", type=int, default=64, help="LoRA rank for PEFT fine-tuning")
    parser.add_argument("--ft_lora_alpha", type=int, default=64, help="LoRA alpha for PEFT fine-tuning")
    parser.add_argument("--ft_lora_dropout", type=float, default=0.05, help="LoRA dropout for PEFT fine-tuning")
    parser.add_argument("--ft_use_dora", type=int, default=0, choices=[0, 1],
                        help="enable DoRA adapters for PEFT (requires a PEFT version that supports use_dora)")
    parser.add_argument("--ft_dora_r", type=int, default=8, help="DoRA rank for low-rank QKV adapters")
    parser.add_argument("--ft_dora_alpha", type=float, default=16.0, help="DoRA alpha for low-rank QKV adapters")
    parser.add_argument("--ft_dora_use_lora_scaling", type=int, default=0, choices=[0, 1],
                        help="whether to apply classic LoRA alpha/r scaling in DoRA delta; default 0 is more stable on low-rank QKV")
    parser.add_argument("--ft_dora_dropout", type=float, default=0.0, help="DoRA dropout on adapter input")
    parser.add_argument("--ft_dora_eps", type=float, default=1e-6, help="DoRA output-channel norm epsilon")
    parser.add_argument("--ft_dora_s_min_ratio", type=float, default=0.25, help="global lower clamp ratio for DoRA magnitude s")
    parser.add_argument("--ft_dora_s_max_ratio", type=float, default=4.0, help="global upper clamp ratio for DoRA magnitude s")
    parser.add_argument("--ft_dora_k_s_min_ratio", type=float, default=0.5, help="K-branch lower clamp ratio for DoRA magnitude s")
    parser.add_argument("--ft_dora_k_s_max_ratio", type=float, default=2.0, help="K-branch upper clamp ratio for DoRA magnitude s")
    parser.add_argument("--ft_dora_k_warmup_steps", type=int, default=20, help="freeze K-DoRA updates for first N inner steps each round")
    parser.add_argument("--ft_dora_include_out_up", type=int, default=0, choices=[0, 1],
                        help="also inject DoRA into out_proj_up (optional; less stable)")
    parser.add_argument("--ft_dora_weight_decay", type=float, default=0.0, help="weight decay for DoRA A/B params")
    parser.add_argument("--ft_dora_lr_q", type=float, default=-1.0, help="Q-branch DoRA lr; <=0 means use ft_learning_rate")
    parser.add_argument("--ft_dora_lr_k", type=float, default=-1.0, help="K-branch DoRA lr; <=0 means 0.5x ft_learning_rate")
    parser.add_argument("--ft_dora_lr_v", type=float, default=-1.0, help="V-branch DoRA lr; <=0 means use ft_learning_rate")
    parser.add_argument("--ft_dora_lr_s", type=float, default=-1.0, help="DoRA magnitude(s) lr; <=0 means conservative auto")
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
                        choices=["lora_qkv", "dora_qkv", "vscale_only", "up_vscale", "qkv_up_vscale", "v_up_vscale", "basis_v", "all"],
                        help="inner training mode for scheme-C: LoRA/DoRA on low-rank q/k/v or direct low-rank weight training")
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
    parser.add_argument("--ft_keep_best", type=int, default=1, choices=[0, 1],
                        help="export the best checkpoint by eval PPL (includes pre-FT baseline as fallback)")
    # Layerwise fine-tune (stable repair loop): train one layer at a time, save, then move to next.
    parser.add_argument("--ft_layerwise", action="store_true", help="enable layerwise local fine-tune (one decoder layer at a time)")
    parser.add_argument("--ft_layerwise_layers", type=str, default="",
                        help='layers to run layerwise FT, e.g. "32-47" or "32,33,34" (default: last 16 layers for OPT)')
    parser.add_argument("--ft_layerwise_steps", type=int, default=200,
                        help="inner steps per layer for layerwise FT")
    parser.add_argument("--ft_layerwise_mode", type=str, default="up_vscale",
                        choices=["vscale_only", "up_only", "up_vscale", "qkv_up_vscale", "v_up_vscale", "all"],
                        help="trainable param mode for each layerwise stage")
    parser.add_argument("--ft_dircalib", type=int, default=0, choices=[0, 1],
                        help="teacher-free directional calibration: anchor current logits to frozen pre-FT reference logits")
    parser.add_argument("--ft_dircalib_weight", type=float, default=0.02,
                        help="weight of cosine directional calibration loss")
    parser.add_argument("--ft_dircalib_every", type=int, default=10,
                        help="apply directional calibration every N inner steps")
    parser.add_argument("--ft_dircalib_batches", type=int, default=8,
                        help="number of fixed anchor batches cached before fine-tuning")
    parser.add_argument("--ft_dircalib_seqlen", type=int, default=256,
                        help="sequence length for fixed directional-calibration anchor batches")
    parser.add_argument("--ft_kd", type=int, default=0, choices=[0, 1],
                        help="enable lightweight logit KD against frozen pre-FT reference logits")
    parser.add_argument("--ft_kd_weight", type=float, default=0.05,
                        help="weight for KD KL loss term")
    parser.add_argument("--ft_kd_temp", type=float, default=2.0,
                        help="temperature for KD soft targets")
    parser.add_argument("--ft_kd_every", type=int, default=10,
                        help="apply KD every N inner steps")
    parser.add_argument("--ft_kd_batches", type=int, default=8,
                        help="number of fixed anchor batches for KD")
    parser.add_argument("--ft_kd_seqlen", type=int, default=256,
                        help="sequence length for fixed KD anchor batches")
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

    # Low-rank QKV orthogonal + column rebalancing (see models/low_rank_balance.py)
    parser.add_argument("--use_core_rebalance", type=int, default=0, choices=[0, 1],
                        help="headwise factored Cayley H1@H2 on left of low-rank W (rank space)")
    parser.add_argument("--use_column_balance", type=int, default=0, choices=[0, 1],
                        help="per-input-column scaling of W (W * gamma.unsqueeze(0))")
    parser.add_argument("--core_rebalance_trainable", type=int, default=1, choices=[0, 1],
                        help="whether H1/H2 skew factors are trainable")
    parser.add_argument("--core_rebalance_orth_rank", type=int, default=8,
                        help="factored skew width rank (<< head dim)")
    parser.add_argument("--balance_strength", type=float, default=0.25,
                        help="column rebalance strength; 0 disables effect (gamma stays ~1)")
    parser.add_argument("--balance_mode", type=str, default="trainable", choices=["static", "trainable"],
                        help="static: freeze gamma after init from W; trainable: log_gamma is a parameter")
    parser.add_argument("--balance_target", type=str, default="qkv",
                        help="qkv | qk | qv | q | k | v | all — which branches get balance modules")
    parser.add_argument("--balance_lr_mult_k", type=float, default=0.5, help="K-branch LR multiplier for orth/col")
    parser.add_argument("--balance_lr_mult_v", type=float, default=1.0, help="V-branch LR multiplier for orth/col")
    parser.add_argument("--ft_orth_reg_weight", type=float, default=0.0, help="weight of ||H^T H - I||^2 regularizer")
    parser.add_argument("--ft_balance_reg_weight", type=float, default=0.0, help="weight of ||log_gamma||^2 toward 1")
    parser.add_argument("--ft_orth_lr", type=float, default=-1.0, help="LR for orth_core params; <=0 means 0.5x ft_learning_rate")
    parser.add_argument("--train_o_proj_up_with_v_balance", type=int, default=0, choices=[0, 1],
                        help="if 1, also train out_proj_up on last ft_train_last_n layers (V path coupling)")
    parser.add_argument("--ft_stage", type=str, default="both", choices=["repair", "task", "both"],
                        help="training stage: repair emphasizes distillation hooks (if enabled); task = LM CE; both = same loop")
    parser.add_argument("--ft_ce_weight", type=float, default=1.0,
                        help="weight for LM CE term in fine-tuning loss; set 0 for distillation-only optimization")
    parser.add_argument("--ft_repair_teacher", type=str, default="",
                        help="HF/local path to base (full-rank) OPT teacher; loaded on CPU to save GPU VRAM")
    parser.add_argument("--ft_repair_hidden_weight", type=float, default=0.0,
                        help="weight for last-layer hidden cosine loss vs teacher (requires --ft_repair_teacher)")
    parser.add_argument("--ft_repair_logits_weight", type=float, default=0.0,
                        help="weight for KL(student||teacher) on logits (short prefix, requires teacher)")
    parser.add_argument("--ft_repair_logits_temp", type=float, default=2.0, help="temperature for repair logit KL")
    parser.add_argument("--ft_repair_attn_weight", type=float, default=0.0,
                        help="attention-map loss (currently unsupported for EigenAttn; keep 0)")
    parser.add_argument("--ft_repair_every", type=int, default=8, help="run teacher repair loss every N steps (and step 1)")
    parser.add_argument("--ft_repair_seqlen", type=int, default=128,
                        help="prefix length for teacher CPU forward (smaller = less RAM/time)")
    parser.add_argument("--ft_repair_teacher_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                        help="teacher weights on CPU; fp16/bf16 ~60GB for OPT-30B")
    parser.add_argument("--ft_save_round_ckpt", type=int, default=0, choices=[0, 1],
                        help="save per-round merged model to round_XX/model (0 saves disk space)")
    parser.add_argument("--ft_use_safetensors", type=int, default=1, choices=[0, 1],
                        help="1: save_pretrained(..., safe_serialization=True), 0: .bin format (often less peak temp IO)")

    # NEW: dtype control (VRAM!)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="model compute dtype (A100 recommended: bf16)")

    # NEW: seqlen control
    parser.add_argument("--seqlen", type=int, default=2048, help="sequence length used for calibration/eval")
    parser.add_argument(
        "--eigen_basis_path",
        type=str,
        default=_default_eigen_basis_pt(),
        help=(
            "Llama GQA EigenAttn: path to eigen_basis.pt. "
            "Default: this repo's qkv_outputs/eigen_basis.pt (absolute path). Override on CLI."
        ),
    )

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

    _wt2 = (getattr(args, "wikitext2_data_dir", "") or "").strip()
    if _wt2:
        os.environ["WIKITEXT2_DATA_DIR"] = _wt2

    # normalize empty strings
    args.qkv_export_layers_obj = _parse_int_list(args.qkv_export_layers, default=None)
    args.qkv_layers_obj = _parse_int_list(args.qkv_layers, default=None)
    args.qkv_ranks_obj = _parse_int_list(args.qkv_ranks, default=None)
    args.ft_target_modules_obj = _parse_str_list(args.ft_target_modules, default=None)
    args.qkv_eigen_basis = args.qkv_eigen_basis.strip() if args.qkv_eigen_basis else ""
    args.repair_basis_pt = args.repair_basis_pt.strip() if args.repair_basis_pt else ""
    args.repair_basis_pt_headwise = args.repair_basis_pt_headwise.strip() if args.repair_basis_pt_headwise else ""
    _ebp = (getattr(args, "eigen_basis_path", "") or "").strip()
    args.eigen_basis_path = _ebp if _ebp else _default_eigen_basis_pt()
    # 若只传了 headwise，默认用同目录下的 eigen_basis.pt 做降秩（三步流程）
    if args.repair_basis_pt_headwise and not args.repair_basis_pt:
        default_first = Path(args.repair_basis_pt_headwise).parent / "eigen_basis.pt"
        if default_first.exists():
            args.repair_basis_pt = str(default_first)
    # default: if user didn't pass --qkv_do_svd, keep it OFF (avoid extra compute)
    # You can enable with --qkv_do_svd
    args.qkv_do_svd = bool(args.qkv_do_svd)
    # Loss-stage sanity checks (for distillation-only runs).
    if float(getattr(args, "ft_ce_weight", 1.0)) == 0.0 and str(getattr(args, "ft_stage", "both")).lower() == "task":
        raise ValueError("--ft_ce_weight 0 with --ft_stage task would disable all task loss; use --ft_stage repair/both.")
    if float(getattr(args, "ft_ce_weight", 1.0)) == 0.0:
        has_any_distill = (
            int(getattr(args, "ft_kd", 0)) == 1
            or float(getattr(args, "ft_repair_hidden_weight", 0.0)) > 0.0
            or float(getattr(args, "ft_repair_logits_weight", 0.0)) > 0.0
            or float(getattr(args, "ft_dircalib_weight", 0.0)) > 0.0
        )
        if not has_any_distill:
            raise ValueError("--ft_ce_weight 0 requires at least one distillation signal: ft_kd / repair_* / ft_dircalib_weight.")

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

    # Auto-detect a pre-compressed low-rank checkpoint directory.
    # This allows resuming fine-tune from --model <.../best_model> without running decomposition again.
    model_dir = Path(args.model)
    has_low_rank_cfg = (model_dir / "low_rank_config.pt").exists() or (model_dir / "low_rank_config.json").exists()
    if has_low_rank_cfg and not bool(getattr(args, "load_low_rank", False)):
        args.load_low_rank = True
    auto_low_rank_save_dir = False
    if bool(getattr(args, "load_low_rank", False)) and not args.save_dir:
        # Needed by LMClass loader only; do NOT use this as fine-tune output root.
        args.save_dir = args.model
        auto_low_rank_save_dir = True

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
    if auto_low_rank_save_dir:
        # Prevent nested outputs: resume-from-model should still write to --output_dir.
        args.save_dir = ""

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
    need_calib = (not bool(getattr(args, "load_low_rank", False))) or args.qkv_analyze or args.qkv_export
    if need_calib:
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
            dataloader = torch.load(cache_dataloader, weights_only=False)
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
    else:
        dataloader = None
        logger.info("[pipeline] load_low_rank model detected: skip calibration loading.")

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
    if bool(getattr(args, "load_low_rank", False)):
        logger.info("[pipeline] load_low_rank model detected: skip clean low-rank rebuild/repair flow.")
    elif "opt" in args.net.lower():
        _log_part(
            logger,
            "pipeline",
            5,
            7,
            "rebuild clean low-rank checkpoint",
            "A fresh low-rank model is instantiated so the compressed weights are saved in a clean and reusable form.",
        )
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
                        r_model = int(low_proj.weight.shape[0])
                        r_basis = int(basis.shape[1])
                        if r_basis != r_model:
                            raise RuntimeError(
                                f"[repair] layer={layer_idx} {proj_name}_low: model expects rank {r_model}, "
                                f"but basis has {r_basis} columns (shape {tuple(basis.shape)}). "
                                f"Step-3 --repair_basis_pt_headwise must match the same low_rank as the loaded "
                                f"checkpoint (and as step-1 --repair_basis_pt). "
                                f"Regenerate stable basis at rank {r_model}, or load a checkpoint with rank {r_basis}."
                            )
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
                    basis_dict_1 = torch.load(basis_path_1, map_location="cpu", weights_only=False)
                    logger.info(f"[repair] step 1/3: rebuild from basis (降秩): {basis_path_1}")
                    for i in range(len(layers2)):
                        base_attn = base_cpu.model.decoder.layers[i].self_attn
                        attn2 = layers2[i].self_attn
                        _rebuild_layer_from_basis(base_attn, attn2, basis_dict_1, i, args.repair_keqk)
                    logger.info("[repair] step 2/3: fix vscale to 1 for all layers")
                    for i in range(len(layers2)):
                        _set_attn_vscale_one(layers2[i].self_attn)
                    basis_dict_2 = torch.load(basis_path_2, map_location="cpu", weights_only=False)
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
                    basis_dict = torch.load(basis_path, map_location="cpu", weights_only=False)
                    for i in range(len(layers2)):
                        base_attn = base_cpu.model.decoder.layers[i].self_attn
                        attn2 = layers2[i].self_attn
                        _rebuild_layer_from_basis(base_attn, attn2, basis_dict, i, args.repair_keqk)
                        _set_attn_vscale_one(attn2)
                        # model2 来自 from_pretrained(low_cpu_mem_usage=True)，v_rot 不在 dense ckpt 中，
                        # 会被 _init_weights 用 N(0, init_std=0.02) 随机覆盖，必须复位为单位阵，
                        # 否则 V 通道被 ~0.02 的随机矩阵打散，PPL 直接崩。
                        _set_attn_v_rot_identity(attn2)
                else:
                    basis_path = Path(args.repair_basis_pt_headwise)
                    if not basis_path.exists():
                        raise FileNotFoundError(f"--repair_basis_pt_headwise not found: {basis_path}")
                    logger.info(f"[repair] rebuild OPT low-rank weights from basis (single, headwise): {basis_path}")
                    basis_dict = torch.load(basis_path, map_location="cpu", weights_only=False)
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

                    # 同上：src_attn (OPTEigenAttention from decompose_modules) 没有 v_rot，
                    # 不会覆盖 dst_attn.v_rot；而 dst_attn 来自 from_pretrained 的随机初始化，
                    # 必须显式复位 v_rot=I，否则 V 通道被随机矩阵打散。
                    _set_attn_v_rot_identity(dst_attn)

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

    elif "llama" in args.net.lower() or "qwen" in args.net.lower():
        layers = lm.model.model.layers
        low_rank_config = []
        for i in range(len(layers)):
            low_rank_config.append([int(layers[i].self_attn.rank_kq), int(layers[i].self_attn.rank_v)])

        config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation, cache_dir=args.cache_dir)
        if "qwen" in args.net.lower():
            from models.modelling_qwen2_eigen_attn import Qwen2ForCausalLM_EigenAttn
            if not hasattr(config, "attention_bias"):
                config.attention_bias = True
            model2 = Qwen2ForCausalLM_EigenAttn.from_pretrained(
                args.model,
                config=config,
                low_rank_config=low_rank_config,
                device_map="cpu",
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir,
            )
            qkv_bias = True
            o_bias = False
        else:
            from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
            model2 = LlamaForCausalLM_EigenAttn.from_pretrained(
                args.model,
                config=config,
                low_rank_config=low_rank_config,
                device_map="cpu",
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir,
            )
            qkv_bias = bool(getattr(config, "attention_bias", False))
            o_bias = qkv_bias

        layers2 = model2.model.layers
        for i in range(len(layers2)):
            layers2[i].self_attn.k_proj_low.weight.data = layers[i].self_attn.k_proj_low.weight.data
            if qkv_bias and layers[i].self_attn.k_proj_low.bias is not None:
                layers2[i].self_attn.k_proj_low.bias.data = layers[i].self_attn.k_proj_low.bias.data
            layers2[i].self_attn.k_proj_up.weight.data = layers[i].self_attn.k_proj_up.weight.data

            layers2[i].self_attn.v_proj_low.weight.data = layers[i].self_attn.v_proj_low.weight.data
            if qkv_bias and layers[i].self_attn.v_proj_low.bias is not None:
                layers2[i].self_attn.v_proj_low.bias.data = layers[i].self_attn.v_proj_low.bias.data

            layers2[i].self_attn.o_proj_up.weight.data = layers[i].self_attn.o_proj_up.weight.data
            if o_bias and getattr(layers[i].self_attn.o_proj_up, "bias", None) is not None:
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
    do_fine_tune = bool(args.fine_tune) and not bool(getattr(args, "compress_only", False))
    if args.fine_tune and bool(getattr(args, "compress_only", False)):
        logger.info("[pipeline] --compress_only is set: ignore --fine_tune and skip all fine-tuning steps.")
    skip_pre_eval = do_fine_tune and args.ft_bilevel_refresh and not getattr(args, "save_pre_ft_checkpoint", False)

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
        elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
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
        # Sidecar ranks next to final_model/ (same convention as hand-copied eb024 artifacts;
        # sanity_AB_opt30b.py prefers low_rank_config.json / .pt in --cmp dir).
        if "opt" in args.net.lower():
            try:
                layers = lm.model.model.decoder.layers
                lr_cfg = [
                    [int(layers[i].self_attn.rank_kq), int(layers[i].self_attn.rank_v)]
                    for i in range(len(layers))
                ]
                lr_json = save_dir / "low_rank_config.json"
                with open(lr_json, "w", encoding="utf-8") as f:
                    json.dump(lr_cfg, f, indent=2)
                torch.save(torch.tensor(lr_cfg, dtype=torch.long), save_dir / "low_rank_config.pt")
                logger.info(f"Wrote {lr_json} and low_rank_config.pt ({len(lr_cfg)} layers).")
            except Exception as e:
                logger.warning(f"[save] could not write low_rank_config sidecars: {e}")
        elif "llama" in args.net.lower() or "mistral" in args.net.lower() or "qwen2" in args.net.lower():
            try:
                layers = lm.model.model.layers
                lr_cfg = [
                    [int(layers[i].self_attn.rank_kq), int(layers[i].self_attn.rank_v)]
                    for i in range(len(layers))
                ]
                lr_json = save_dir / "low_rank_config.json"
                with open(lr_json, "w", encoding="utf-8") as f:
                    json.dump(lr_cfg, f, indent=2)
                torch.save(torch.tensor(lr_cfg, dtype=torch.long), save_dir / "low_rank_config.pt")
                logger.info(f"Wrote {lr_json} and low_rank_config.pt ({len(lr_cfg)} layers).")
            except Exception as e:
                logger.warning(f"[save] could not write low_rank_config sidecars: {e}")
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
    if do_fine_tune:
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

        # ------------------------------------------------------------
        # Layerwise local fine-tune (stable): train one layer at a time
        # ------------------------------------------------------------
        if bool(getattr(args, "ft_layerwise", False)):
            if "opt" not in args.net.lower():
                raise RuntimeError("--ft_layerwise currently implemented for OPT models only.")
            model = lm.model
            model.config.use_cache = False
            if args.multigpu:
                _place_model_multigpu(lm, args, logger)
                model = lm.model
                _train_device = lm.model.model.decoder.layers[0].device
            else:
                _train_device = next(model.parameters()).device

            # Build token stream once (CPU)
            texts_iter = _iter_texts_from_dir(ft_data_dir)
            stream_cpu = _build_token_stream(tokenizer, texts_iter, target_tokens=int(args.ft_token_budget), seed=int(args.seed))
            if stream_cpu is None:
                raise RuntimeError(f"No usable tokens found in: {ft_data_dir}")
            loader = _batchify_stream_cpu(stream_cpu, int(args.ft_seqlen), int(args.ft_batch_size))

            # Eval stream (optional)
            texts_eval = _iter_texts_from_dir(ft_data_dir)
            eval_stream_cpu = _build_token_stream(tokenizer, texts_eval, target_tokens=min(200000, int(args.ft_token_budget)), seed=int(args.seed) + 1)

            # Parse layers
            n_layers = len(model.model.decoder.layers)
            lay_s = str(getattr(args, "ft_layerwise_layers", "") or "").strip()
            if lay_s:
                layer_list = _parse_int_list(lay_s, default=None)
                if layer_list is None:
                    raise RuntimeError(f"Invalid --ft_layerwise_layers: {lay_s!r}")
            else:
                # default: last 16 layers
                layer_list = list(range(max(0, n_layers - 16), n_layers))

            out_root = Path(args.output_dir) / "layerwise_model"
            out_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"[ft_layerwise] layers={layer_list} steps_per_layer={int(args.ft_layerwise_steps)} mode={args.ft_layerwise_mode}")

            dtype = _resolve_dtype(args.dtype)
            best_ppl = float("inf")
            best_dir = out_root / "best_model"
            best_basis_path = out_root / "best_basis.pt"
            basis_state = {}

            # baseline best snapshot (optional, matches bilevel)
            if bool(int(getattr(args, "ft_keep_best", 1))) and eval_stream_cpu is not None:
                ppl0 = _eval_ppl_simple(model, eval_stream_cpu, _train_device, dtype=dtype)
                best_ppl = float(ppl0) if math.isfinite(float(ppl0)) else float("inf")
                _save_opt_checkpoint_with_metadata(model=model, tokenizer=tokenizer, target_dir=best_dir, basis_state=basis_state, basis_path=best_basis_path)
                logger.info(f"[ft_layerwise] baseline PPL={best_ppl:.6f}; best_model initialized.")

            for li in layer_list:
                li = int(li)
                if not (0 <= li < n_layers):
                    continue
                logger.info(f"[ft_layerwise] ===== layer {li}/{n_layers-1} =====")
                # Save layer-start checkpoint first (before training this layer).
                layer_pre_dir = out_root / f"layer_{li:02d}_pretrain"
                _save_opt_checkpoint_with_metadata(
                    model=model,
                    tokenizer=tokenizer,
                    target_dir=layer_pre_dir,
                    basis_state=basis_state,
                    basis_path=(layer_pre_dir / "eigen_basis.pt"),
                )
                logger.info(f"[ft_layerwise] saved layer pretrain checkpoint: {layer_pre_dir}")

                _set_trainable_opt_layers(model, [li], mode=str(args.ft_layerwise_mode))
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                logger.info(f"[ft_layerwise] trainable params={sum(p.numel() for p in trainable_params):,}")
                optimizer = torch.optim.AdamW(trainable_params, lr=float(args.ft_learning_rate), weight_decay=0.0)
                optimizer.zero_grad(set_to_none=True)
                nonfinite_count = 0

                steps_here = max(1, int(args.ft_layerwise_steps))
                for step in range(1, steps_here + 1):
                    x_cpu = next(loader)
                    x = x_cpu.to(_train_device, non_blocking=True)
                    attn = torch.ones_like(x)
                    labels = x
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        # Align with finetune_ppl_opt30b_eigen.py: use model's labels-path loss.
                        out = model(input_ids=x, attention_mask=attn, labels=labels, use_cache=False, return_dict=True)
                        raw_loss = float(out.loss.detach().float().item())
                        loss = out.loss / max(args.ft_grad_accum, 1)
                    if not torch.isfinite(loss).all():
                        nonfinite_count += 1
                        optimizer.zero_grad(set_to_none=True)
                        if args.ft_max_nonfinite and nonfinite_count >= int(args.ft_max_nonfinite):
                            logger.warning(
                                f"[ft_layerwise][layer {li}] too many consecutive non-finite losses "
                                f"(count={nonfinite_count}); abort this layer."
                            )
                            break
                        continue
                    # Count only consecutive non-finite steps; reset after a valid loss.
                    nonfinite_count = 0
                    loss.backward()
                    if step % int(args.ft_grad_accum) == 0 or step == steps_here:
                        do_step = True
                        if args.ft_grad_clip and args.ft_grad_clip > 0:
                            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, args.ft_grad_clip)
                            if not torch.isfinite(gnorm).all().item():
                                do_step = False
                        if do_step:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    if step == 1 or step % max(1, int(args.ft_logging_steps)) == 0 or step == steps_here:
                        logger.info(f"[ft_layerwise][layer {li}] step={step}/{steps_here} nll={raw_loss:.6f}")

                # Save layer checkpoint
                layer_dir = out_root / f"layer_{li:02d}"
                _save_opt_checkpoint_with_metadata(model=model, tokenizer=tokenizer, target_dir=layer_dir, basis_state=basis_state, basis_path=(layer_dir / "eigen_basis.pt"))
                logger.info(f"[ft_layerwise] saved layer checkpoint: {layer_dir}")

                # Update best by eval PPL (optional)
                if bool(int(getattr(args, "ft_keep_best", 1))) and eval_stream_cpu is not None:
                    ppl = _eval_ppl_simple(model, eval_stream_cpu, _train_device, dtype=dtype)
                    if math.isfinite(float(ppl)) and float(ppl) < best_ppl:
                        best_ppl = float(ppl)
                        _save_opt_checkpoint_with_metadata(model=model, tokenizer=tokenizer, target_dir=best_dir, basis_state=basis_state, basis_path=best_basis_path)
                        logger.info(f"[ft_layerwise] new best_model by eval PPL={best_ppl:.6f} at layer {li}")

            # Export final_model
            final_dir = out_root / "final_model"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            if bool(int(getattr(args, "ft_keep_best", 1))) and best_dir.exists():
                shutil.copytree(best_dir, final_dir)
                logger.info(f"[ft_layerwise] exported final_model from best_model PPL={best_ppl:.6f}: {final_dir}")
            else:
                final_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(final_dir, safe_serialization=True)
                tokenizer.save_pretrained(final_dir)
                logger.info(f"[ft_layerwise] saved final_model to: {final_dir}")
            return

        if args.ft_bilevel_refresh:
            # ============================================================
            # Scheme C – bilevel refresh
            # Inner loop: LoRA adapters or direct low-rank weight training (no HF Trainer)
            # Token stream kept on CPU to save VRAM.
            # After each inner loop: optionally merge_and_unload → refresh eigen basis.
            # ============================================================
            if "opt" not in args.net.lower():
                raise RuntimeError("--ft_bilevel_refresh is currently implemented for OPT models only.")
            need_refresh_basis = not bool(getattr(args, "ft_skip_refresh", False))
            if need_refresh_basis and not args.repair_basis_pt:
                raise RuntimeError("--ft_bilevel_refresh requires --repair_basis_pt unless --ft_skip_refresh is set.")

            use_lora_inner = args.ft_inner_mode == "lora_qkv"
            use_dora_inner = args.ft_inner_mode == "dora_qkv"
            use_direct_weight = not (use_lora_inner or use_dora_inner)
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
                _lora_kwargs = dict(
                    lora_alpha=args.ft_lora_alpha,
                    lora_dropout=args.ft_lora_dropout,
                    target_modules=target_modules,
                    r=args.ft_lora_r,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                if int(getattr(args, "ft_use_dora", 0)) == 1:
                    _lora_kwargs["use_dora"] = True
                try:
                    peft_config = LoraConfig(**_lora_kwargs)
                except TypeError as e:
                    if "use_dora" in _lora_kwargs:
                        raise RuntimeError(
                            "Current PEFT version does not support DoRA (use_dora). "
                            "Please upgrade peft or set --ft_use_dora 0."
                        ) from e
                    raise
                logger.info(
                    f"[fine_tune/bilevel] inner_mode=lora_qkv adapter config: "
                    f"target_modules={target_modules} r={args.ft_lora_r} "
                    f"alpha={args.ft_lora_alpha} dropout={args.ft_lora_dropout} "
                    f"use_dora={bool(int(getattr(args, 'ft_use_dora', 0)))}"
                )
            elif use_dora_inner:
                if args.ft_target_modules_obj is not None:
                    dora_targets = args.ft_target_modules_obj
                else:
                    dora_targets = ["q_proj_low", "k_proj_low", "v_proj_low"]
                logger.info(
                    f"[fine_tune/bilevel] inner_mode=dora_qkv config: "
                    f"target_modules={dora_targets} r={args.ft_dora_r} alpha={args.ft_dora_alpha} "
                    f"dropout={args.ft_dora_dropout} include_out_up={bool(int(args.ft_dora_include_out_up))} "
                    f"lr_q={args.ft_dora_lr_q} lr_k={args.ft_dora_lr_k} lr_v={args.ft_dora_lr_v} lr_s={args.ft_dora_lr_s}"
                )
            else:
                logger.info(
                    f"[fine_tune/bilevel] inner_mode={args.ft_inner_mode} "
                    f"(direct low-rank weight training via _set_trainable_opt; train_last_n={args.ft_train_last_n})"
                )

            _log_part(logger, "fine_tune", 1, 3, "build CPU token stream",
                      "Tokenize once, keep on CPU; each step slices a random window → no GPU padding overhead.")
            if bool(getattr(args, "ft_skip_refresh", False)):
                basis_state = {}
            else:
                basis_state = torch.load(
                    args.repair_basis_pt_headwise if args.repair_basis_pt_headwise else args.repair_basis_pt,
                    map_location="cpu",
                    weights_only=False,
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

            # Always write fine-tune artifacts under --output_dir to avoid nesting into input checkpoint path.
            output_dir_ft = str(Path(args.output_dir) / "bilevel_model")
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

            keep_best_ckpt = bool(int(getattr(args, "ft_keep_best", 1)))
            best_ppl = float("inf")
            best_dir = Path(output_dir_ft) / "best_model"
            best_basis_path = Path(output_dir_ft) / "best_basis.pt"

            utkarsh_doc_pool = None
            utkarsh_valenc = None
            if args.ft_compare_ppl:
                utkarsh_doc_pool, _ = _build_doc_pool_from_c4_train_dir(
                    args.ft_data_dir,
                    glob_pat=str(getattr(args, "ft_ppl_c4_glob", "c4-train.*.json.gz")),
                    pool_docs=int(getattr(args, "ft_ppl_pool_docs", 20000)),
                    scan_lines=int(getattr(args, "ft_ppl_scan_lines", 200000)),
                    seed=int(getattr(args, "seed", 2)),
                )
                utkarsh_valenc = _build_utkarsh_like_valenc_from_pool(
                    tokenizer,
                    utkarsh_doc_pool,
                    seqlen=int(getattr(args, "ft_ppl_seqlen", 2048)),
                    n_chunks=int(getattr(args, "ft_ppl_val_chunks", 64)),
                    val_seed=int(getattr(args, "ft_ppl_val_seed", 0)),
                )

            dir_batches_cpu: List[torch.Tensor] = []
            dir_ref_logits: List[torch.Tensor] = []
            if int(getattr(args, "ft_dircalib", 0)) == 1:
                dir_source_stream = eval_stream_cpu if eval_stream_cpu is not None else stream_cpu
                dir_seqlen = max(8, min(int(args.ft_dircalib_seqlen), int(args.ft_seqlen)))
                dir_batches_cpu = _build_fixed_eval_batches_cpu(
                    dir_source_stream,
                    seqlen=dir_seqlen,
                    batch_size=1,
                    num_batches=max(1, int(args.ft_dircalib_batches)),
                    seed=int(args.seed) + 1234,
                )
                dir_ref_logits = _collect_reference_logits(
                    model=model,
                    batches_cpu=dir_batches_cpu,
                    device=_train_device,
                    dtype=dtype,
                    logit_clip=float(args.ft_logit_clip),
                )
                logger.info(
                    f"[fine_tune/bilevel] directional calibration enabled: "
                    f"anchors={len(dir_batches_cpu)} seqlen={dir_seqlen} "
                    f"weight={args.ft_dircalib_weight} every={args.ft_dircalib_every}"
                )

            kd_batches_cpu: List[torch.Tensor] = []
            kd_ref_logits: List[torch.Tensor] = []
            if int(getattr(args, "ft_kd", 0)) == 1:
                kd_source_stream = eval_stream_cpu if eval_stream_cpu is not None else stream_cpu
                kd_seqlen = max(8, min(int(args.ft_kd_seqlen), int(args.ft_seqlen)))
                kd_batches_cpu = _build_fixed_eval_batches_cpu(
                    kd_source_stream,
                    seqlen=kd_seqlen,
                    batch_size=1,
                    num_batches=max(1, int(args.ft_kd_batches)),
                    seed=int(args.seed) + 4321,
                )
                kd_ref_logits = _collect_reference_logits(
                    model=model,
                    batches_cpu=kd_batches_cpu,
                    device=_train_device,
                    dtype=dtype,
                    logit_clip=float(args.ft_logit_clip),
                )
                logger.info(
                    f"[fine_tune/bilevel] KD enabled: anchors={len(kd_batches_cpu)} "
                    f"seqlen={kd_seqlen} weight={args.ft_kd_weight} temp={args.ft_kd_temp} "
                    f"every={args.ft_kd_every}"
                )

            if keep_best_ckpt and eval_stream_cpu is not None:
                if args.ft_compare_ppl and utkarsh_valenc is not None:
                    best_ppl, info = _eval_ppl_utkarsh_opt(
                        model,
                        utkarsh_valenc,
                        device=_train_device,
                        seqlen=int(getattr(args, "ft_ppl_seqlen", 2048)),
                        limit=int(getattr(args, "limit", -1)),
                    )
                    logger.info(
                        f"[fine_tune/bilevel] baseline Utkarsh-style eval PPL before training: "
                        f"{best_ppl:.6f} info={info}"
                    )
                else:
                    best_ppl = _eval_ppl_simple(
                        model, eval_stream_cpu, _train_device,
                        seqlen=min(2048, args.ft_seqlen * 4), stride=min(2048, args.ft_seqlen * 4),
                        dtype=dtype,
                    )
                _save_opt_checkpoint_with_metadata(
                    model=model,
                    tokenizer=tokenizer,
                    target_dir=best_dir,
                    basis_state=basis_state,
                    basis_path=best_basis_path,
                )
                logger.info(
                    f"[fine_tune/bilevel] baseline eval PPL before training: {best_ppl:.6f}; "
                    f"best_model initialized from pre-FT checkpoint"
                )

            # Base OPT teacher on CPU (minimal GPU VRAM); not used with LoRA inner mode
            teacher_model: Optional[nn.Module] = None
            repair_attn_noted = [False]
            _tpath = str(getattr(args, "ft_repair_teacher", "") or "").strip()
            _rw_h = float(getattr(args, "ft_repair_hidden_weight", 0.0))
            _rw_l = float(getattr(args, "ft_repair_logits_weight", 0.0))
            _rw_a = float(getattr(args, "ft_repair_attn_weight", 0.0))
            if _tpath and (_rw_h > 0 or _rw_l > 0 or _rw_a > 0):
                if use_lora_inner:
                    logger.warning(
                        "[repair/teacher] LoRA inner mode: skip loading teacher "
                        "(decoder hook path not wired for PEFT wrapper)."
                    )
                else:
                    try:
                        teacher_model = _load_repair_teacher_opt(
                            _tpath,
                            str(getattr(args, "ft_repair_teacher_dtype", "fp16")),
                            logger,
                        )
                        if _rw_a > 0:
                            logger.warning(
                                "[repair/teacher] ft_repair_attn_weight>0 is ignored for EigenAttn vs base OPT."
                            )
                            repair_attn_noted[0] = True
                    except Exception as e:
                        logger.warning(f"[repair/teacher] failed to load teacher: {e}")
                        teacher_model = None

            for round_idx in range(1, outer_rounds + 1):
                if finished_steps >= total_budget_steps:
                    break
                current_steps = min(inner_steps, total_budget_steps - finished_steps)
                round_dir = Path(output_dir_ft) / f"round_{round_idx:02d}"
                if int(getattr(args, "ft_save_round_ckpt", 0)) == 1:
                    round_dir.mkdir(parents=True, exist_ok=True)

                logger.info(
                    f"[fine_tune/bilevel] round={round_idx}/{outer_rounds} "
                    f"inner_steps={current_steps} completed={finished_steps}/{total_budget_steps}"
                )
                logger.info(
                    "[fine_tune/bilevel][explain] "
                    "attach LoRA → manual SGD inner loop → merge LoRA → refresh eigen basis."
                )

                # Optional: headwise orthogonal + column balance on low-rank W (idempotent)
                if int(getattr(args, "use_core_rebalance", 0)) or int(getattr(args, "use_column_balance", 0)):
                    apply_low_rank_balance_to_opt_model(model, args, logger)

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
                elif use_dora_inner:
                    logger.info(f"[fine_tune/bilevel][round {round_idx}][part 1/5] attach low-rank DoRA adapters")
                    logger.info("[fine_tune/bilevel][explain] Freeze compressed backbone, train DoRA(A/B/s) on q/k/v low-rank projections.")
                    if hasattr(model, "gradient_checkpointing_enable"):
                        model.gradient_checkpointing_enable()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if args.multigpu:
                        lm.model = model
                        _place_model_multigpu(lm, args, logger)
                        model = lm.model
                        _train_device = lm.model.model.decoder.layers[0].device
                    # After final layer placement, attach optional orth/col on low-rank W (idempotent)
                    if int(getattr(args, "use_core_rebalance", 0)) or int(getattr(args, "use_column_balance", 0)):
                        apply_low_rank_balance_to_opt_model(model, args, logger)
                    wrapped = _attach_lowrank_dora_opt(model, args)
                    logger.info(f"[fine_tune/bilevel][round {round_idx}] injected DoRA wrappers: {wrapped}")
                    _maybe_enable_out_proj_up_trainable(model, args)
                    train_model = model
                else:
                    if hasattr(model, "gradient_checkpointing_enable"):
                        model.gradient_checkpointing_enable()
                        logger.info("[fine_tune/bilevel] gradient_checkpointing enabled (direct-weight, match finetune_ppl)")
                    calib_train_last_n = (
                        0 if (getattr(args, "ft_calib_strong", False) and use_direct_weight)
                        else args.ft_train_last_n
                    )
                    logger.info(f"[fine_tune/bilevel][round {round_idx}][part 1/5] set trainable params via _set_trainable_opt (inner_mode={args.ft_inner_mode}, train_last_n={calib_train_last_n})")
                    if int(getattr(args, "use_core_rebalance", 0)) or int(getattr(args, "use_column_balance", 0)):
                        apply_low_rank_balance_to_opt_model(model, args, logger)
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
                k_dora_params = []
                if use_dora_inner:
                    for n, p in train_model.named_parameters():
                        if p.requires_grad and ".k_proj_low." in n and ("dora_A" in n or "dora_B" in n or "dora_s" in n):
                            k_dora_params.append(p)
                if not trainable_params:
                    raise RuntimeError(
                        "[fine_tune/bilevel] 没有可训练参数：LoRA/DoRA 未挂上、或全被 ft_train_last_n / 配置冻住。"
                        "请检查 --ft_inner_mode、--ft_train_last_n、PEFT target_modules，以及是否走了 direct-weight 但未解冻任何层。"
                    )
                calib_lr = (
                    2e-5 if (getattr(args, "ft_calib_strong", False) and use_direct_weight)
                    else args.ft_learning_rate
                )
                if use_dora_inner:
                    param_groups = _build_dora_param_groups_opt(train_model, args, base_lr=calib_lr)
                    optimizer = torch.optim.AdamW(param_groups)
                else:
                    optimizer = torch.optim.AdamW(
                        trainable_params, lr=calib_lr, weight_decay=0.0
                    )
                loader = _batchify_stream_cpu(stream_cpu, args.ft_seqlen, args.ft_batch_size)

                train_model.train()
                if teacher_model is not None and hasattr(train_model, "gradient_checkpointing_disable"):
                    train_model.gradient_checkpointing_disable()
                    logger.info("[repair/teacher] gradient_checkpointing disabled (required for last-layer hook).")
                optimizer.zero_grad(set_to_none=True)
                step_start = time.time()
                nonfinite_count = 0
                round_aborted = False

                for step in range(1, current_steps + 1):
                    x_cpu = next(loader)
                    x = x_cpu.to(_train_device, non_blocking=True)
                    attn = torch.ones_like(x)
                    labels = x

                    rw_h = float(getattr(args, "ft_repair_hidden_weight", 0.0))
                    rw_l = float(getattr(args, "ft_repair_logits_weight", 0.0))
                    rw_a = float(getattr(args, "ft_repair_attn_weight", 0.0))
                    rep_every = max(1, int(getattr(args, "ft_repair_every", 8)))
                    repair_on = (
                        teacher_model is not None
                        and (rw_h > 0 or rw_l > 0 or rw_a > 0)
                        and (step % rep_every == 0 or step == 1)
                    )

                    last_h_s = None
                    if repair_on and rw_h > 0:
                        out, last_h_s = forward_student_with_last_hidden_capture(
                            train_model, x, attn, dtype, autocast_enabled=torch.cuda.is_available()
                        )
                    else:
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            out = train_model(
                                input_ids=x,
                                attention_mask=attn,
                                use_cache=False,
                                return_dict=True,
                            )
                    stage = str(getattr(args, "ft_stage", "both")).lower()
                    enable_ce = stage in ("task", "both")
                    enable_distill = stage in ("repair", "both")
                    ce_w = float(getattr(args, "ft_ce_weight", 1.0))
                    # Compute CE only when enabled; avoid 0*NaN contamination in distillation-only runs.
                    if enable_ce and ce_w != 0.0:
                        shift_logits = out.logits[:, :-1, :].contiguous().view(-1, out.logits.size(-1))
                        shift_labels = labels[:, 1:].contiguous().view(-1)
                        if args.ft_logit_clip and args.ft_logit_clip > 0:
                            shift_logits = shift_logits.clamp(min=-args.ft_logit_clip, max=args.ft_logit_clip)
                        ce_loss = F.cross_entropy(
                            shift_logits.float(),
                            shift_labels,
                            ignore_index=-100,
                            reduction="mean",
                        ) / max(args.ft_grad_accum, 1)
                        loss = ce_loss * ce_w
                    else:
                        ce_loss = torch.zeros((), device=x.device, dtype=torch.float32)
                        loss = torch.zeros((), device=x.device, dtype=torch.float32)
                    dir_loss_value = None
                    kd_loss_value = None
                    repair_loss_value = None

                    if enable_distill and repair_on and (rw_h > 0 or rw_l > 0):
                        Lrep = min(x.size(1), int(getattr(args, "ft_repair_seqlen", 128)))
                        x_cpu = x[:, :Lrep].detach().cpu()
                        attn_cpu = attn[:, :Lrep].detach().cpu().float()
                        try:
                            t_logits_cpu, last_h_t_cpu = _teacher_forward_prefix_cpu(teacher_model, x_cpu, attn_cpu)
                        except Exception as e:
                            logger.warning(f"[repair/teacher] teacher forward failed at step={step}: {e}")
                            t_logits_cpu, last_h_t_cpu = None, None
                        if t_logits_cpu is not None:
                            rloss = torch.zeros((), device=x.device, dtype=torch.float32)
                            if rw_h > 0 and last_h_s is not None and last_h_t_cpu is not None:
                                hs = last_h_s[:, :Lrep].float().reshape(-1, last_h_s.size(-1))
                                ht = last_h_t_cpu.to(device=hs.device, dtype=torch.float32).reshape_as(hs)
                                cos = F.cosine_similarity(hs, ht, dim=-1).clamp(-1.0, 1.0)
                                rloss = rloss + rw_h * (1.0 - cos).mean()
                            if rw_l > 0:
                                tau = max(1e-3, float(getattr(args, "ft_repair_logits_temp", 2.0)))
                                sl = out.logits[:, :Lrep].float()
                                tl = t_logits_cpu.to(device=sl.device, dtype=torch.float32)
                                s_log = F.log_softmax(sl / tau, dim=-1).view(-1, sl.size(-1))
                                t_prob = F.softmax(tl / tau, dim=-1).view(-1, tl.size(-1))
                                kl = F.kl_div(s_log, t_prob, reduction="batchmean") * (tau * tau)
                                rloss = rloss + rw_l * kl
                            loss = loss + rloss.to(dtype=loss.dtype) / max(args.ft_grad_accum, 1)
                            repair_loss_value = float(rloss.detach().float().item())

                    if enable_distill and dir_ref_logits and (step % max(1, int(args.ft_dircalib_every)) == 0 or step == 1):
                        anchor_idx = (step - 1) % len(dir_ref_logits)
                        dir_x = dir_batches_cpu[anchor_idx].to(_train_device, non_blocking=True)
                        dir_attn = torch.ones_like(dir_x)
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            dir_out = train_model(input_ids=dir_x, attention_mask=dir_attn)
                        dir_logits = dir_out.logits[:, :-1, :].contiguous()
                        if args.ft_logit_clip and args.ft_logit_clip > 0:
                            dir_logits = dir_logits.clamp(min=-args.ft_logit_clip, max=args.ft_logit_clip)
                        ref_logits = dir_ref_logits[anchor_idx].to(dir_logits.device, non_blocking=True)
                        dir_cos = F.cosine_similarity(
                            dir_logits.float().view(1, -1),
                            ref_logits.view(1, -1),
                            dim=-1,
                        ).mean()
                        dir_loss_value = (1.0 - dir_cos).clamp_min(0.0)
                        loss = loss + (
                            float(args.ft_dircalib_weight) * dir_loss_value / max(args.ft_grad_accum, 1)
                        )

                    if enable_distill and kd_ref_logits and (step % max(1, int(args.ft_kd_every)) == 0 or step == 1):
                        kd_idx = (step - 1) % len(kd_ref_logits)
                        kd_x = kd_batches_cpu[kd_idx].to(_train_device, non_blocking=True)
                        kd_attn = torch.ones_like(kd_x)
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            kd_out = train_model(input_ids=kd_x, attention_mask=kd_attn)
                        kd_s_logits = kd_out.logits[:, :-1, :].contiguous()
                        if args.ft_logit_clip and args.ft_logit_clip > 0:
                            kd_s_logits = kd_s_logits.clamp(min=-args.ft_logit_clip, max=args.ft_logit_clip)
                        kd_t_logits = kd_ref_logits[kd_idx].to(kd_s_logits.device, non_blocking=True)
                        t = max(1e-3, float(args.ft_kd_temp))
                        s_log_prob = F.log_softmax((kd_s_logits.float() / t).view(-1, kd_s_logits.size(-1)), dim=-1)
                        t_prob = F.softmax((kd_t_logits / t).view(-1, kd_t_logits.size(-1)), dim=-1)
                        kd_loss_value = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (t * t)
                        if torch.isfinite(kd_loss_value).all():
                            loss = loss + (
                                float(args.ft_kd_weight) * kd_loss_value / max(args.ft_grad_accum, 1)
                            )
                        else:
                            kd_loss_value = None

                    ow = float(getattr(args, "ft_orth_reg_weight", 0.0))
                    bw = float(getattr(args, "ft_balance_reg_weight", 0.0))
                    if ow > 0 or bw > 0:
                        o_reg, b_reg = collect_orth_balance_regularizers(train_model)
                        o_reg = o_reg.to(device=loss.device, dtype=loss.dtype)
                        b_reg = b_reg.to(device=loss.device, dtype=loss.dtype)
                        if ow > 0:
                            loss = loss + ow * o_reg / max(args.ft_grad_accum, 1)
                        if bw > 0:
                            loss = loss + bw * b_reg / max(args.ft_grad_accum, 1)

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
                        if not getattr(loss, "requires_grad", False):
                            raise RuntimeError(
                                "[fine_tune/bilevel] loss 未接入 autograd（无 grad_fn），无法 backward。"
                                "常见原因：(1) 关闭了 CE 且 ft_kd_weight=0 等导致总损失仍为常数 0；"
                                "(2) 无可训练参数导致 KL/CE 对参数无梯度；"
                                "(3) 日志里的 kd_loss 为原始 KL，若 ft_kd_weight=0 则不会乘进 loss。"
                                f" step={step} ft_stage={getattr(args, 'ft_stage', '')} "
                                f"ft_ce_weight={getattr(args, 'ft_ce_weight', 1.0)} "
                                f"ft_kd_weight={getattr(args, 'ft_kd_weight', 0.0)} "
                                f"trainable_params={len(trainable_params)}"
                            )
                        loss.backward()
                        # Conservative K branch warmup to stabilize QK^T.
                        if use_dora_inner and int(getattr(args, "ft_dora_k_warmup_steps", 0)) > 0 and step <= int(args.ft_dora_k_warmup_steps):
                            for _p in k_dora_params:
                                _p.grad = None

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

                    nll_for_log = ce_loss.detach().float().item() * max(args.ft_grad_accum, 1)
                    if step % args.ft_logging_steps == 0 or step == 1 or step == current_steps:
                        elapsed = time.time() - step_start
                        pct = 100.0 * step / current_steps
                        dir_msg = ""
                        if dir_loss_value is not None:
                            dir_msg = f" dir_loss={dir_loss_value.detach().float().item():.6f}"
                        kd_msg = ""
                        if kd_loss_value is not None:
                            kd_msg = f" kd_loss={kd_loss_value.detach().float().item():.6f}"
                        rep_msg = ""
                        if repair_loss_value is not None:
                            rep_msg = f" repair={repair_loss_value:.6f}"
                        logger.info(
                            f"[fine_tune/bilevel][round {round_idx}] "
                            f"step={step}/{current_steps} ({pct:.1f}%) "
                            f"nll={nll_for_log:.6f} "
                            f"{dir_msg}"
                            f"{kd_msg}"
                            f"{rep_msg}"
                            f"elapsed={elapsed:.1f}s"
                        )
                        if use_dora_inner:
                            dora_stats = _collect_dora_stats_opt(train_model)
                            if "q" in dora_stats:
                                q = dora_stats["q"]
                                logger.info(
                                    f"[fine_tune/bilevel][round {round_idx}] DoRA-Q s[min/med/max]="
                                    f"{q['s_min']:.4f}/{q['s_med']:.4f}/{q['s_max']:.4f} "
                                    f"ratio[min/med/max]={q['ratio_min']:.3f}/{q['ratio_med']:.3f}/{q['ratio_max']:.3f}"
                                )
                            if "k" in dora_stats:
                                k = dora_stats["k"]
                                logger.info(
                                    f"[fine_tune/bilevel][round {round_idx}] DoRA-K s[min/med/max]="
                                    f"{k['s_min']:.4f}/{k['s_med']:.4f}/{k['s_max']:.4f} "
                                    f"ratio[min/med/max]={k['ratio_min']:.3f}/{k['ratio_med']:.3f}/{k['ratio_max']:.3f}"
                                )
                            if "v" in dora_stats:
                                v = dora_stats["v"]
                                logger.info(
                                    f"[fine_tune/bilevel][round {round_idx}] DoRA-V s[min/med/max]="
                                    f"{v['s_min']:.4f}/{v['s_med']:.4f}/{v['s_max']:.4f} "
                                    f"ratio[min/med/max]={v['ratio_min']:.3f}/{v['ratio_med']:.3f}/{v['ratio_max']:.3f}"
                                )

                    if args.ft_eval_every > 0 and step % args.ft_eval_every == 0 and eval_stream_cpu is not None:
                        train_model.eval()
                        if args.ft_compare_ppl and utkarsh_valenc is not None:
                            ppl, info = _eval_ppl_utkarsh_opt(
                                train_model,
                                utkarsh_valenc,
                                device=_train_device,
                                seqlen=int(getattr(args, "ft_ppl_seqlen", 2048)),
                                limit=int(getattr(args, "limit", -1)),
                            )
                            logger.info(
                                f"[fine_tune/bilevel][round {round_idx}] mid-round Utkarsh PPL @ step {step}: {ppl:.6f} info={info}"
                            )
                        else:
                            ppl = _eval_ppl_simple(
                                train_model, eval_stream_cpu, _train_device,
                                seqlen=min(2048, args.ft_seqlen * 4), stride=min(2048, args.ft_seqlen * 4),
                                dtype=dtype,
                            )
                            logger.info(f"[fine_tune/bilevel][round {round_idx}] mid-round PPL @ step {step}: {ppl:.4f}")

                        # mid-round best snapshot: allows escaping later divergence
                        if keep_best_ckpt and ppl is not None and math.isfinite(float(ppl)) and float(ppl) < best_ppl:
                            best_ppl = float(ppl)
                            # Note: only safe/meaningful to snapshot direct-weight mode (no adapter merge mid-step).
                            if use_direct_weight:
                                _save_opt_checkpoint_with_metadata(
                                    model=train_model,
                                    tokenizer=tokenizer,
                                    target_dir=best_dir,
                                    basis_state=basis_state,
                                    basis_path=best_basis_path,
                                )
                                logger.info(
                                    f"[fine_tune/bilevel][round {round_idx}] updated best_model mid-round by PPL={best_ppl:.6f} at step {step}"
                                )
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
                elif use_dora_inner:
                    merged = _merge_and_unload_lowrank_dora_opt(model)
                    logger.info(f"[fine_tune/bilevel][explain] merged+unloaded DoRA wrappers: {merged}")
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
                round_ppl = None
                if eval_stream_cpu is not None:
                    if args.ft_compare_ppl:
                        try:
                            ppl, info = _eval_ppl_utkarsh_opt(
                                model,
                                utkarsh_valenc,
                                device=_train_device,
                                seqlen=int(getattr(args, "ft_ppl_seqlen", 2048)),
                                limit=int(getattr(args, "limit", -1)),
                            )
                            round_ppl = float(ppl)
                            logger.info(f"[fine_tune/bilevel][round {round_idx}] end-of-round Utkarsh-style PPL: {ppl:.6f} info={info}")
                        except Exception as e:
                            logger.warning(f"[fine_tune/bilevel][round {round_idx}] Utkarsh-style PPL eval failed: {e}")
                    else:
                        ppl = _eval_ppl_simple(
                            model, eval_stream_cpu, _train_device,
                            seqlen=min(2048, args.ft_seqlen * 4), stride=min(2048, args.ft_seqlen * 4),
                            dtype=dtype,
                        )
                        round_ppl = float(ppl)
                        logger.info(f"[fine_tune/bilevel][round {round_idx}] end-of-round PPL: {ppl:.4f}")

                if keep_best_ckpt and round_ppl is not None and math.isfinite(round_ppl) and round_ppl < best_ppl:
                    best_ppl = round_ppl
                    _save_opt_checkpoint_with_metadata(
                        model=model,
                        tokenizer=tokenizer,
                        target_dir=best_dir,
                        basis_state=basis_state,
                        basis_path=best_basis_path,
                    )
                    logger.info(
                        f"[fine_tune/bilevel][round {round_idx}] new best_model by eval PPL: {best_ppl:.6f}"
                    )

                if int(getattr(args, "ft_save_round_ckpt", 0)) == 1:
                    merged_dir = round_dir / "model"
                    merged_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(
                        merged_dir,
                        safe_serialization=bool(int(getattr(args, "ft_use_safetensors", 1))),
                    )
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
            final_basis_path = Path(output_dir_ft) / "final_basis.pt"
            if keep_best_ckpt and best_dir.exists():
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                shutil.copytree(best_dir, final_dir)
                if best_basis_path.exists():
                    shutil.copy2(best_basis_path, final_basis_path)
                logger.info(
                    f"[fine_tune/bilevel] exported final_model from best_model with eval PPL={best_ppl:.6f}: {final_dir}"
                )
            else:
                final_dir.mkdir(parents=True, exist_ok=True)
                lm.model.save_pretrained(
                    final_dir,
                    safe_serialization=bool(int(getattr(args, "ft_use_safetensors", 1))),
                )
                tokenizer.save_pretrained(final_dir)
                _low_rank_list = [
                    [int(lm.model.model.decoder.layers[i].self_attn.rank_kq), int(lm.model.model.decoder.layers[i].self_attn.rank_v)]
                    for i in range(len(lm.model.model.decoder.layers))
                ]
                with open(final_dir / "low_rank_config.json", "w") as f:
                    json.dump(_low_rank_list, f, indent=2)
                torch.save(basis_state, final_basis_path)
                logger.info(f"[fine_tune/bilevel] saved final model to: {final_dir}")

            if args.ft_compare_ppl:
                try:
                    # Evaluate PPL on the exported final_model (disk) to avoid using a potentially corrupted in-memory model.
                    eval_model = lm.model
                    if keep_best_ckpt and best_dir.exists():
                        from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn
                        final_dir = Path(output_dir_ft) / "final_model"
                        low_rank_cfg_path = final_dir / "low_rank_config.json"
                        if low_rank_cfg_path.exists():
                            low_rank_config_re = json.load(open(low_rank_cfg_path, "r"))
                        else:
                            low_rank_config_re = json.load(open(best_dir / "low_rank_config.json", "r"))
                        config_re = AutoConfig.from_pretrained(
                            final_dir,
                            attn_implementation=args.attn_implementation,
                            cache_dir=args.cache_dir,
                        )
                        config_re.use_cache = False
                        eval_model = OPTForCausalLM_EigenAttn.from_pretrained(
                            final_dir,
                            config=config_re,
                            low_rank_config=low_rank_config_re,
                            device_map="cpu",
                            torch_dtype=dtype,
                            cache_dir=args.cache_dir,
                        )

                        lm.model = eval_model
                        if args.multigpu:
                            _place_model_multigpu(lm, args, logger)
                            eval_model = lm.model

                    _ppl_dev = eval_model.model.decoder.layers[0].device
                    _ppl_sanity_utkarsh_c4_train(eval_model, tokenizer, args, logger, tag="post_ft(final_model)", device=_ppl_dev)
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

            _lora_kwargs = dict(
                lora_alpha=args.ft_lora_alpha,
                lora_dropout=args.ft_lora_dropout,
                target_modules=target_modules,
                r=args.ft_lora_r,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if int(getattr(args, "ft_use_dora", 0)) == 1:
                _lora_kwargs["use_dora"] = True
            try:
                peft_config = LoraConfig(**_lora_kwargs)
            except TypeError as e:
                if "use_dora" in _lora_kwargs:
                    raise RuntimeError(
                        "Current PEFT version does not support DoRA (use_dora). "
                        "Please upgrade peft or set --ft_use_dora 0."
                    ) from e
                raise
            logger.info(f"[fine_tune] target_modules={target_modules}")
            logger.info(
                f"[fine_tune] lora_r={args.ft_lora_r} alpha={args.ft_lora_alpha} "
                f"dropout={args.ft_lora_dropout} lr={args.ft_learning_rate} "
                f"use_dora={bool(int(getattr(args, 'ft_use_dora', 0)))}"
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