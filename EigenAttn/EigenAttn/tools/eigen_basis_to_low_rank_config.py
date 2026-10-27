#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 eigen_basis.pt 推导每层 low_rank_config [[rank_kq, rank_v], ...]，并估算压缩率。

- OPT（build_eigen_basis_opt_error_budget_offline.py）：q/k basis [hidden, rank_kq]，v [hidden, rank_v]。
- Llama/GQA（build_eigen_basis_llama_error_budget_offline.py）：q/k basis [head_dim, kcols]，
  rank_kq = kcols * num_kv；v 块状 [num_kv*head_dim, rank_v]。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch


def parse_keys(keys: List[str]) -> Dict[int, Dict[str, str]]:
    """layer_idx -> {"q": full_key, ...}"""
    by_layer: Dict[int, Dict[str, str]] = {}
    for k in keys:
        m = None
        for pat in (
            r"base_model\.model\.model\.decoder\.layers\.(\d+)\.self_attn\.(q|k|v)_proj_basis",
            r"model\.decoder\.layers\.(\d+)\.self_attn\.(q|k|v)_proj_basis",
            r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj_basis",
        ):
            m = re.match(pat, k)
            if m:
                break
        if not m:
            continue
        li = int(m.group(1))
        typ = m.group(2)
        by_layer.setdefault(li, {})[typ] = k
    return by_layer


def rank_from_tensor(t: torch.Tensor) -> int:
    """U shape [hidden, r] or [r, hidden] -> rank r."""
    if t.dim() != 2:
        raise ValueError(f"expected 2D basis matrix, got shape {tuple(t.shape)}")
    d0, d1 = t.shape
    return int(d1) if d0 >= d1 else int(d0)


def eigen_attn_param_count(rkq: int, rv: int, hidden: int) -> int:
    """
    EigenAttn 单侧近似（不含 bias / DoRA / adapter）：
    q_proj_low [rkq, h], k_proj_low [rkq, h], v_proj_low [rv, h],
    v_rot [rv, rv], out_proj_up [h, rv]
    """
    return (
        int(rkq) * hidden * 2
        + int(rv) * hidden * 2
        + int(rv) * int(rv)
    )


def full_opt_attn_linear_params(hidden: int) -> int:
    """OPT attention：q_proj,k_proj,v_proj,out_proj 四块权重矩阵。"""
    h = int(hidden)
    return 4 * h * h


def llama_ranks_from_qv(Uq: torch.Tensor, Uv: torch.Tensor) -> tuple:
    """Return (rank_kq, rank_v), num_kv inferred from shapes."""
    if Uq.dim() != 2 or Uv.dim() != 2:
        raise ValueError(f"expected 2D q/v basis, got {tuple(Uq.shape)}, {tuple(Uv.shape)}")
    head_dim = int(Uq.shape[0])
    kcols = int(Uq.shape[1])
    if Uv.shape[0] % head_dim != 0:
        raise ValueError(f"v rows {Uv.shape[0]} not divisible by head_dim {head_dim}")
    num_kv = int(Uv.shape[0] // head_dim)
    rank_kq = kcols * num_kv
    rank_v = int(Uv.shape[1])
    if rank_kq % num_kv != 0 or rank_v % num_kv != 0:
        raise ValueError(f"rank_kq={rank_kq}, rank_v={rank_v} not compatible with num_kv={num_kv}")
    return rank_kq, rank_v, num_kv, head_dim


def eigen_attn_llama_param_count(
    rkq: int, rv: int, hidden: int, num_kv: int, n_heads: int
) -> int:
    """
    Llama EigenAttn（EigenAttn_base）：全秩 q_proj + k/v 低秩 + o_proj_up。
    不计 bias 小节；与快速估算一致。
    """
    h = int(hidden)
    head_dim = h // int(n_heads)
    kv = int(rv) // int(num_kv)
    return (
        h * h
        + int(rkq) * h
        + (int(rkq) // int(num_kv)) * head_dim
        + int(rv) * h
        + h * (kv * int(n_heads))
    )


def full_llama_gqa_attn_linear_params(hidden: int, num_kv: int, n_heads: int) -> int:
    h = int(hidden)
    head_dim = h // int(n_heads)
    kv_dim = int(num_kv) * head_dim
    return h * h + 2 * h * kv_dim + h * h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", required=True, type=str, help="path to eigen_basis.pt")
    ap.add_argument(
        "--out_json",
        type=str,
        default="",
        help="write low_rank_config JSON path",
    )
    ap.add_argument(
        "--hidden_size",
        type=int,
        default=0,
        help="hidden size (OPT default 7168 if arch=opt; Llama-8B 常用 4096；0=auto 仅 Llama 可从 basis 推断 head_dim×n_heads 需与模型一致)",
    )
    ap.add_argument(
        "--num_attention_heads",
        type=int,
        default=0,
        help="Llama：注意力头数（如 32）；0 则尝试用 hidden/head_dim 推断",
    )
    ap.add_argument("--map_location", type=str, default="cpu")
    args = ap.parse_args()

    pt_path = Path(args.basis)
    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)

    obj = torch.load(pt_path, map_location=args.map_location, weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict checkpoint, got {type(obj)}")

    keys = sorted(obj.keys())
    by_layer = parse_keys(keys)
    if not by_layer:
        raise RuntimeError(
            "No recognized *_proj_basis keys. Keys sample: " + repr(keys[:8])
        )

    layers_sorted = sorted(by_layer.keys())
    n_layers = len(layers_sorted)
    expected = list(range(n_layers))
    if layers_sorted != expected:
        missing = set(expected) - set(layers_sorted)
        raise RuntimeError(f"non-contiguous layer indices or missing layers: {missing}")

    sample_key = keys[0]
    arch_llama = "decoder.layers" not in sample_key and ".layers." in sample_key

    low_rank_config: List[List[int]] = []
    num_kv = 0
    head_dim = 0

    if arch_llama:
        for li in range(n_layers):
            blk = by_layer[li]
            if "q" not in blk or "v" not in blk:
                raise KeyError(f"layer {li} missing q or v key: {blk}")
            Uq = obj[blk["q"]]
            Uv = obj[blk["v"]]
            if not torch.is_tensor(Uq) or not torch.is_tensor(Uv):
                raise TypeError(f"layer {li}: expected tensors")
            rkq, rv, nkv, hd = llama_ranks_from_qv(Uq, Uv)
            if num_kv == 0:
                num_kv = nkv
                head_dim = hd
            elif nkv != num_kv or hd != head_dim:
                raise RuntimeError(
                    f"inconsistent num_kv/head_dim: layer0 num_kv={num_kv} hd={head_dim}, "
                    f"layer{li} num_kv={nkv} hd={hd}"
                )
            Uk = obj.get(blk.get("k"), None)
            if Uk is not None and torch.is_tensor(Uk):
                rk2, _, _, _ = llama_ranks_from_qv(Uk, Uv)
                if rk2 != rkq:
                    print(
                        f"[warn] layer {li}: rank_kq from q ({rkq}) != k ({rk2}); using q.",
                        flush=True,
                    )
            low_rank_config.append([rkq, rv])

        if args.num_attention_heads and args.hidden_size:
            hidden = int(args.hidden_size)
            n_heads = int(args.num_attention_heads)
        elif args.hidden_size:
            hidden = int(args.hidden_size)
            n_heads = hidden // head_dim
        elif args.num_attention_heads:
            n_heads = int(args.num_attention_heads)
            hidden = head_dim * n_heads
        elif head_dim == 128:
            n_heads = 32
            hidden = head_dim * n_heads
            print(
                "[info] Llama: inferred n_heads=32, hidden_size=4096 from head_dim=128 "
                "(set --hidden_size / --num_attention_heads if your model differs).",
                flush=True,
            )
        else:
            raise ValueError(
                "Llama basis: pass --hidden_size and/or --num_attention_heads "
                f"(head_dim from basis is {head_dim})."
            )
        if hidden != head_dim * n_heads:
            raise ValueError(
                f"hidden_size={hidden} must equal head_dim×n_heads ({head_dim}×{n_heads}={head_dim * n_heads})"
            )

        sum_kq = sum(x[0] for x in low_rank_config)
        sum_v = sum(x[1] for x in low_rank_config)
        full_kv_rank_units = n_layers * (2 * hidden)
        rank_budget_ratio = (sum_kq + sum_v) / max(full_kv_rank_units, 1)

        param_lr = sum(
            eigen_attn_llama_param_count(a, b, hidden, num_kv, n_heads)
            for a, b in low_rank_config
        )
        param_full = n_layers * full_llama_gqa_attn_linear_params(hidden, num_kv, n_heads)
        param_ratio = param_lr / max(param_full, 1)
        compression_vs_full = 1.0 - param_ratio

        summary = {
            "arch": "llama_gqa",
            "basis_file": str(pt_path.resolve()),
            "num_layers": n_layers,
            "hidden_size": hidden,
            "num_attention_heads": n_heads,
            "num_key_value_heads": num_kv,
            "head_dim_from_basis": head_dim,
            "low_rank_config": low_rank_config,
            "sum_rank_kq": sum_kq,
            "sum_rank_v": sum_v,
            "sum_ranks": sum_kq + sum_v,
            "full_kv_rank_units_baseline": full_kv_rank_units,
            "rank_budget_vs_2h_per_layer": rank_budget_ratio,
            "approx_attn_weight_params_low_rank": param_lr,
            "approx_attn_weight_params_full_dense": param_full,
            "param_ratio_low_over_full": param_ratio,
            "compression_fraction_attn_weights": compression_vs_full,
            "note": "compression 指注意力线性权重的参数量比（不含 bias/LM head/MLP）；q 仍为全秩。",
        }
    else:
        hidden = int(args.hidden_size) if args.hidden_size else 7168
        for li in range(n_layers):
            blk = by_layer[li]
            if "q" not in blk or "v" not in blk:
                raise KeyError(f"layer {li} missing q or v key: {blk}")
            Uq = obj[blk["q"]]
            Uk = obj.get(blk.get("k"), None)
            Uv = obj[blk["v"]]
            if not torch.is_tensor(Uq) or not torch.is_tensor(Uv):
                raise TypeError(f"layer {li}: expected tensors")
            rkq = rank_from_tensor(Uq)
            rv = rank_from_tensor(Uv)
            if Uk is not None and torch.is_tensor(Uk):
                rkq_k = rank_from_tensor(Uk)
                if rkq_k != rkq:
                    print(
                        f"[warn] layer {li}: rank_kq from q ({rkq}) != k ({rkq_k}); using q.",
                        flush=True,
                    )
            low_rank_config.append([rkq, rv])

        sum_kq = sum(x[0] for x in low_rank_config)
        sum_v = sum(x[1] for x in low_rank_config)

        full_kv_rank_units = n_layers * (2 * hidden)
        rank_budget_ratio = (sum_kq + sum_v) / max(full_kv_rank_units, 1)

        param_lr = sum(eigen_attn_param_count(a, b, hidden) for a, b in low_rank_config)
        param_full = n_layers * full_opt_attn_linear_params(hidden)
        param_ratio = param_lr / max(param_full, 1)
        compression_vs_full = 1.0 - param_ratio

        summary = {
            "arch": "opt",
            "basis_file": str(pt_path.resolve()),
            "num_layers": n_layers,
            "hidden_size": hidden,
            "low_rank_config": low_rank_config,
            "sum_rank_kq": sum_kq,
            "sum_rank_v": sum_v,
            "sum_ranks": sum_kq + sum_v,
            "full_kv_rank_units_baseline": full_kv_rank_units,
            "rank_budget_vs_2h_per_layer": rank_budget_ratio,
            "approx_attn_weight_params_low_rank": param_lr,
            "approx_attn_weight_params_full_dense": param_full,
            "param_ratio_low_over_full": param_ratio,
            "compression_fraction_attn_weights": compression_vs_full,
        }

    print(json.dumps(summary, indent=2))
    print()
    print(
        f"[summary] 近似注意力线性权重：低秩参数量 / 全稠密 = {param_ratio:.4f} "
        f"→ 压缩约 {100.0 * compression_vs_full:.2f}% （剩余 {100.0 * param_ratio:.2f}%）"
    )
    print(
        f"[summary] 秩预算 (sum_kq+sum_v) / (L×2h) = {rank_budget_ratio:.4f} "
        f"（可与 EB024 retarget 口径对照）"
    )

    out_path = args.out_json.strip()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(low_rank_config, f, indent=2)
            f.write("\n")
        meta_path = outp.with_name(outp.stem + "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"[saved] {outp}")
        print(f"[saved] {meta_path}")


if __name__ == "__main__":
    main()
