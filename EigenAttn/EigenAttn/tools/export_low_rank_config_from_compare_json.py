#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 build_activation_eigen_basis_opt.py 写出的 compare_json 中读取 per-head 统一 rank，
生成 low_rank_config（每层 [rank_kq, rank_v] = [num_heads * r_kq, num_heads * r_v]），
并可选择按层截断 eigen_basis.pt 中 q/k/v 的列（headwise 块内截断）。

compare_json 字段优先级：
  1) rank_allocation.kq_layer_unified_after_global / v_layer_unified_after_global
  2) rank_allocation.kq_layer_unified_before_global / v_layer_unified_before_global
  3) per_layer[].layer_rank_candidate_kq_unified / layer_rank_candidate_v_unified

若某层未在 compare 的 layers 中，使用 --fallback_ref_json 中同层 rank，或 --default_r_kq / --default_r_v。

用法（在 EigenAttn 目录下，PYTHONPATH 含当前目录）:
  python tools/export_low_rank_config_from_compare_json.py \\
    --compare_json qkv_outputs/eigen_basis_unbind_qk_full_compare.json \\
    --out_low_rank low_rank_config_adaptive_opt30b.json \\
    --num_heads 56 \\
    --n_layers 48

  # 可选：同时截断 basis
  python tools/export_low_rank_config_from_compare_json.py \\
    --compare_json ... \\
    --out_low_rank ... \\
    --basis_in qkv_outputs/eigen_basis.pt \\
    --basis_out /diskD/jiaqizhang/eigen_basis_adaptive.pt

  # compare 分配的 per-head rank 大于 ref basis 列数时，默认钳制到 ref（--clamp_to_ref 0 则报错退出）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def _basis_key(layer_idx: int, branch: str) -> str:
    return f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{branch}_proj_basis"


def _load_ranks_from_compare(data: dict, prefer_before: bool = False) -> Tuple[List[int], List[int], List[int]]:
    """Returns (layer_indices, r_kq_per_head, r_v_per_head) aligned by layer index."""
    ra = data.get("rank_allocation")
    if isinstance(ra, dict):
        layers = ra.get("layers")
        kq_after = ra.get("kq_layer_unified_after_global")
        rv_after = ra.get("v_layer_unified_after_global")
        kq_before = ra.get("kq_layer_unified_before_global")
        rv_before = ra.get("v_layer_unified_before_global")
        if prefer_before and isinstance(kq_before, list) and isinstance(rv_before, list) and isinstance(layers, list):
            if len(kq_before) == len(rv_before) == len(layers):
                return [int(x) for x in layers], [int(x) for x in kq_before], [int(x) for x in rv_before]
        kq, rv = kq_after, rv_after
        if kq is None or rv is None:
            kq, rv = kq_before, rv_before
        if (
            isinstance(kq, list)
            and isinstance(rv, list)
            and isinstance(layers, list)
            and len(kq) == len(rv) == len(layers)
        ):
            return [int(x) for x in layers], [int(x) for x in kq], [int(x) for x in rv]

    per = data.get("per_layer")
    if isinstance(per, list) and per:
        layers = []
        rkq = []
        rv = []
        for rec in per:
            if not isinstance(rec, dict):
                continue
            li = rec.get("layer")
            if li is None:
                continue
            layers.append(int(li))
            rkq.append(int(rec["layer_rank_candidate_kq_unified"]))
            rv.append(int(rec["layer_rank_candidate_v_unified"]))
        if layers and len(layers) == len(rkq) == len(rv):
            return layers, rkq, rv

    raise ValueError(
        "compare_json 中未找到 rank_allocation（after/before global）或 per_layer 中的 layer_rank_candidate_*"
    )


def _load_fallback_ranks(path: str, n_layers: int) -> List[Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, list) or len(cfg) < n_layers:
        raise ValueError(f"fallback ref json invalid or shorter than n_layers={n_layers}")
    out: List[Tuple[int, int]] = []
    for i in range(n_layers):
        row = cfg[i]
        out.append((int(row[0]), int(row[1])))
    return out


def _slice_headwise(U: torch.Tensor, num_heads: int, r_old_per_head: int, r_new_per_head: int) -> torch.Tensor:
    """U: [hidden, num_heads * r_old] -> [hidden, num_heads * r_new]（每 head 只保留前 r_new 列）。"""
    if r_new_per_head > r_old_per_head:
        raise ValueError(
            f"目标 per-head rank {r_new_per_head} > 参考 per-head {r_old_per_head}，无法仅截断得到，请换更大 ref basis 或重跑统计。"
        )
    parts: List[torch.Tensor] = []
    for h in range(num_heads):
        block = U[:, h * r_old_per_head : (h + 1) * r_old_per_head]
        parts.append(block[:, :r_new_per_head].contiguous())
    return torch.cat(parts, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare_json", type=str, required=True)
    ap.add_argument("--out_low_rank", type=str, required=True, help="输出的 low_rank_config JSON 路径")
    ap.add_argument("--num_heads", type=int, default=56)
    ap.add_argument("--n_layers", type=int, default=48)
    ap.add_argument(
        "--use_after_global",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: 优先用 rank_allocation 的 *_after_global；0: 若存在 before 则优先 before（仅当 after 不存在时仍会用 per_layer）",
    )
    ap.add_argument(
        "--fallback_ref_json",
        type=str,
        default="",
        help="若 compare 未覆盖全部层，用该文件每层 [rank_kq,rank_v] 补全未出现层",
    )
    ap.add_argument(
        "--default_r_kq",
        type=int,
        default=0,
        help="无 fallback 时，缺失层 per-head r_kq（0 表示用 ref 该层推导）",
    )
    ap.add_argument("--default_r_v", type=int, default=0)
    ap.add_argument("--basis_in", type=str, default="", help="可选：输入 eigen_basis.pt（用于推导每头最大 rank，并可截断）")
    ap.add_argument("--basis_out", type=str, default="", help="可选：截断后保存路径（需同时给 --basis_in）")
    ap.add_argument(
        "--clamp_to_ref",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: compare 的 per-head rank 超过 ref basis 时钳制到 ref 并告警；0: 保持原值并在截断时报错",
    )
    args = ap.parse_args()

    cmp_path = Path(args.compare_json)
    if not cmp_path.is_file():
        raise FileNotFoundError(cmp_path)

    with open(cmp_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prefer_before = int(args.use_after_global) == 0
    layers_idx, r_kq_ph, r_v_ph = _load_ranks_from_compare(data, prefer_before=prefer_before)

    by_layer: Dict[int, Tuple[int, int]] = {}
    for li, a, b in zip(layers_idx, r_kq_ph, r_v_ph):
        by_layer[int(li)] = (int(a), int(b))

    fallback: Optional[List[Tuple[int, int]]] = None
    if args.fallback_ref_json.strip():
        fallback = _load_fallback_ranks(args.fallback_ref_json.strip(), args.n_layers)

    if args.basis_out.strip() and not args.basis_in.strip():
        raise ValueError("--basis_out 需要同时指定 --basis_in")

    basis_in: Optional[Dict[str, torch.Tensor]] = None
    r_old_kq: Optional[int] = None
    r_old_v: Optional[int] = None
    if args.basis_in.strip():
        basis_in = torch.load(args.basis_in, map_location="cpu")
        if not isinstance(basis_in, dict):
            raise TypeError("basis_in must be a dict")
        Uq0 = basis_in[_basis_key(0, "q")]
        hidden = int(Uq0.shape[0])
        rank_kq0 = int(Uq0.shape[1])
        rank_v0 = int(basis_in[_basis_key(0, "v")].shape[1])
        if rank_kq0 % args.num_heads or rank_v0 % args.num_heads:
            raise ValueError("ref basis rank not divisible by num_heads")
        r_old_kq = rank_kq0 // args.num_heads
        r_old_v = rank_v0 // args.num_heads
        if hidden != int(basis_in[_basis_key(0, "v")].shape[0]):
            raise ValueError("q/v basis hidden mismatch")

    low_rank: List[List[int]] = []
    out_basis: Dict[str, torch.Tensor] = {}
    clamp_notes: List[str] = []

    for li in range(args.n_layers):
        if li in by_layer:
            r_kq, r_v = by_layer[li]
        elif fallback is not None:
            r_kq = fallback[li][0] // args.num_heads
            r_v = fallback[li][1] // args.num_heads
        else:
            if args.default_r_kq > 0 and args.default_r_v > 0:
                r_kq, r_v = args.default_r_kq, args.default_r_v
            elif r_old_kq is not None and r_old_v is not None:
                r_kq, r_v = r_old_kq, r_old_v
            else:
                raise ValueError(
                    f"layer {li} 无 rank，请提供 --fallback_ref_json 或 --basis_in 以推导默认 per-head rank"
                )

        if r_old_kq is not None and r_old_v is not None and int(args.clamp_to_ref) == 1:
            if int(r_kq) > r_old_kq or int(r_v) > r_old_v:
                nk, nv = min(int(r_kq), r_old_kq), min(int(r_v), r_old_v)
                clamp_notes.append(f"layer {li}: r_kq {r_kq}->{nk}, r_v {r_v}->{nv} (ref per-head kq={r_old_kq}, v={r_old_v})")
                r_kq, r_v = nk, nv

        rk_tot = int(r_kq) * args.num_heads
        rv_tot = int(r_v) * args.num_heads
        low_rank.append([rk_tot, rv_tot])

        if args.basis_out.strip() and basis_in is not None and r_old_kq is not None and r_old_v is not None:
            for br, rph, rold in [("q", r_kq, r_old_kq), ("k", r_kq, r_old_kq), ("v", r_v, r_old_v)]:
                key = _basis_key(li, br)
                if key not in basis_in:
                    raise KeyError(f"missing key {key}")
                U = basis_in[key].float()
                out_basis[key] = _slice_headwise(U, args.num_heads, rold, int(rph)).to(dtype=U.dtype)

    out_lr = Path(args.out_low_rank)
    out_lr.parent.mkdir(parents=True, exist_ok=True)
    with open(out_lr, "w", encoding="utf-8") as f:
        json.dump(low_rank, f, indent=2)
    print(f"[saved] low_rank_config -> {out_lr}")
    if clamp_notes:
        print(f"[warn] 已将 per-head rank 钳制到 ref basis 上限（共 {len(clamp_notes)} 层），示例：")
        for s in clamp_notes[:8]:
            print(f"       {s}")
        if len(clamp_notes) > 8:
            print(f"       ... 另有 {len(clamp_notes) - 8} 条，略")
    print(
        f"[info] rank_kq sum (per-head over layers) ~= sum of r_kq_ph: "
        f"check compare_json; layerwise total dims: kq={sum(x[0] for x in low_rank)}, v={sum(x[1] for x in low_rank)}"
    )

    if args.basis_out.strip() and out_basis:
        # 若 basis_in 有未截断层（例如只 partial layers），从原 dict 拷贝其余键
        for k, v in basis_in.items():
            if k not in out_basis:
                out_basis[k] = v
        outp = Path(args.basis_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_basis, outp)
        print(f"[saved] sliced basis -> {outp}")


if __name__ == "__main__":
    main()
