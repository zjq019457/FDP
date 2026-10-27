#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两份 eigen_basis.pt（或同类 dict checkpoint）的结构与数值差异。

用法:
  cd EigenAttn/EigenAttn
  PYTHONPATH=$(pwd):$PYTHONPATH python3 tools/compare_eigen_basis_files.py \\
    --a qkv_outputs/eigen_basis.pt \\
    --b qkv_outputs/eigen_basis_5712.pt \\
    --json_out qkv_outputs/basis_ab_report.json

检查项（尽量多）:
  - 键集合、缺失/多余键
  - 每层 q/k/v 的 shape、dtype、非 finite
  - 列范数、是否近似列正交 (||U^T U - I||)
  - 奇异值分布 (SVD of U)、有效秩
  - 同键时: Fro 差、相对误差、逐列 cos、主角度、投影矩阵差
  - 层间: 相邻层 basis 相似度 (可选)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# key parsing
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(
    r"^base_model\.model\.model\.decoder\.layers\.(\d+)\.self_attn\.(q|k|v)_proj_basis$"
)


def parse_basis_key(k: str) -> Optional[Tuple[int, str]]:
    m = _LAYER_RE.match(k)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def walk_tensors(obj: Any, prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
    out: List[Tuple[str, torch.Tensor]] = []
    if torch.is_tensor(obj):
        out.append((prefix or "<root>", obj))
        return out
    if isinstance(obj, dict):
        for kk, vv in obj.items():
            p = f"{prefix}.{kk}" if prefix else str(kk)
            out.extend(walk_tensors(vv, p))
        return out
    return out


# ---------------------------------------------------------------------------
# metrics on one matrix U [d, r] (columns = basis directions)
# ---------------------------------------------------------------------------

@torch.no_grad()
def orthogonality_error(U: torch.Tensor) -> Dict[str, float]:
    Uf = U.float().cpu()
    d, r = Uf.shape
    if r == 0:
        return {"col_ortho_fro": float("nan"), "col_ortho_rel": float("nan")}
    G = Uf.T @ Uf
    I = torch.eye(r, dtype=G.dtype)
    err = torch.linalg.matrix_norm(G - I, ord="fro")
    rel = float(err / max(torch.linalg.matrix_norm(I, ord="fro").item(), 1e-12))
    return {"col_ortho_fro": float(err.item()), "col_ortho_rel": rel}


@torch.no_grad()
def column_norm_stats(U: torch.Tensor) -> Dict[str, float]:
    Uf = U.float().cpu()
    if Uf.numel() == 0:
        return {"col_norm_mean": float("nan"), "col_norm_std": float("nan"), "col_norm_min": float("nan"), "col_norm_max": float("nan")}
    cn = torch.linalg.norm(Uf, dim=0)
    return {
        "col_norm_mean": float(cn.mean().item()),
        "col_norm_std": float(cn.std().item()),
        "col_norm_min": float(cn.min().item()),
        "col_norm_max": float(cn.max().item()),
    }


@torch.no_grad()
def svd_stats(U: torch.Tensor) -> Dict[str, float]:
    Uf = U.float().cpu()
    if Uf.numel() == 0:
        return {}
    # thin SVD: U = Q S Vt  with Q [d,r], S [r]
    try:
        Q, S, _ = torch.linalg.svd(Uf, full_matrices=False)
    except Exception:
        return {"svd_fail": 1.0}
    S = S.clamp(min=0)
    s0 = float(S[0].item()) if S.numel() else 0.0
    smin = float(S[-1].item()) if S.numel() else 0.0
    ssum = float((S**2).sum().item())
    rat = float(S[0] / (S[-1] + 1e-12)) if S.numel() else float("inf")
    # 有效秩: 能量 90%/99%
    c = torch.cumsum(S**2, dim=0) / max((S**2).sum().item(), 1e-12)
    eff90 = int((c >= 0.90).nonzero(as_tuple=False)[0].item()) + 1 if S.numel() else 0
    eff99 = int((c >= 0.99).nonzero(as_tuple=False)[0].item()) + 1 if S.numel() else 0
    return {
        "sing_max": s0,
        "sing_min": smin,
        "sing_cond": rat,
        "sing_energy_ratio_1": float((S[0] ** 2 / max(ssum, 1e-12))) if S.numel() else float("nan"),
        "eff_rank_energy90": eff90,
        "eff_rank_energy99": eff99,
    }


@torch.no_grad()
def compare_pair(Ua: torch.Tensor, Ub: torch.Tensor) -> Dict[str, float]:
    """Same shape assumed."""
    A = Ua.float().cpu()
    B = Ub.float().cpu()
    if A.shape != B.shape:
        return {"shape_match": 0.0, "fro_diff": float("nan")}
    diff = A - B
    fro = float(torch.linalg.norm(diff, ord="fro").item())
    na = float(torch.linalg.norm(A, ord="fro").item())
    nb = float(torch.linalg.norm(B, ord="fro").item())
    rel = fro / max(0.5 * (na + nb), 1e-12)
    # column-wise cosine
    ca = torch.linalg.norm(A, dim=0).clamp(min=1e-12)
    cb = torch.linalg.norm(B, dim=0).clamp(min=1e-12)
    cos = ((A * B).sum(dim=0) / (ca * cb)).clamp(-1, 1)
    # principal angles via SVD of A^T B (orthonormalized)
    def ortho(X: torch.Tensor) -> torch.Tensor:
        q, _ = torch.linalg.qr(X, mode="reduced")
        return q[:, : X.shape[1]]

    Ao = ortho(A)
    Bo = ortho(B)
    M = Ao.T @ Bo
    s = torch.linalg.svdvals(M).clamp(-1, 1)
    ang = torch.acos(s) * (180.0 / math.pi)
    # projector difference
    Pa = Ao @ Ao.T
    Pb = Bo @ Bo.T
    pf = float(torch.linalg.matrix_norm(Pa - Pb, ord="fro").item())
    r = A.shape[1]
    pfn = pf / max(math.sqrt(2.0 * r), 1e-12)
    return {
        "shape_match": 1.0,
        "fro_diff": fro,
        "fro_rel_mean_norm": rel,
        "cos_col_mean": float(cos.mean().item()),
        "cos_col_min": float(cos.min().item()),
        "cos_col_max": float(cos.max().item()),
        "princ_angle_mean_deg": float(ang.mean().item()),
        "princ_angle_max_deg": float(ang.max().item()),
        "proj_fro_diff": pf,
        "proj_fro_diff_norm_sqrt2r": pfn,
    }


def load_state(path: Path) -> Dict[str, Any]:
    x = torch.load(str(path), map_location="cpu")
    if not isinstance(x, dict):
        raise TypeError(f"{path} root must be dict, got {type(x)}")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=str, required=True, help="第一个 basis 文件")
    ap.add_argument("--b", type=str, required=True, help="第二个 basis 文件")
    ap.add_argument("--json_out", type=str, default="", help="可选：写出完整 JSON 报告")
    ap.add_argument("--max_layers_print", type=int, default=8, help="终端打印每层对比条数上限（0=不打印每层）")
    args = ap.parse_args()

    pa, pb = Path(args.a), Path(args.b)
    if not pa.exists():
        print(f"ERROR: missing {pa}", file=sys.stderr)
        sys.exit(1)
    if not pb.exists():
        print(f"ERROR: missing {pb}", file=sys.stderr)
        sys.exit(1)

    A = load_state(pa)
    B = load_state(pb)

    keys_a = set(A.keys())
    keys_b = set(B.keys())
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    report: Dict[str, Any] = {
        "file_a": str(pa.resolve()),
        "file_b": str(pb.resolve()),
        "n_keys_a": len(keys_a),
        "n_keys_b": len(keys_b),
        "n_common": len(common),
        "keys_only_in_a": only_a[:200],  # cap
        "keys_only_in_b": only_b[:200],
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "per_key": {},
        "per_layer_summary": [],
        "global_a": {},
        "global_b": {},
    }

    # non-tensor entries
    for name, state in [("a", A), ("b", B)]:
        non_tensor = [k for k, v in state.items() if not torch.is_tensor(v)]
        report[f"non_tensor_keys_{name}"] = non_tensor[:100]
        report[f"n_non_tensor_{name}"] = len(non_tensor)

    # aggregate stats for common tensor keys
    fro_list: List[float] = []
    ang_list: List[float] = []

    for k in common:
        va, vb = A[k], B[k]
        if not torch.is_tensor(va) or not torch.is_tensor(vb):
            report["per_key"][k] = {"skip": "non-tensor"}
            continue
        entry: Dict[str, Any] = {
            "shape_a": list(va.shape),
            "shape_b": list(vb.shape),
            "dtype_a": str(va.dtype),
            "dtype_b": str(vb.dtype),
        }
        if va.shape != vb.shape:
            entry["match"] = False
            report["per_key"][k] = entry
            continue
        entry["match"] = True
        entry["a_stats"] = {
            **orthogonality_error(va),
            **column_norm_stats(va),
            **svd_stats(va),
            "finite": bool(torch.isfinite(va.float()).all().item()),
        }
        entry["b_stats"] = {
            **orthogonality_error(vb),
            **column_norm_stats(vb),
            **svd_stats(vb),
            "finite": bool(torch.isfinite(vb.float()).all().item()),
        }
        cmp = compare_pair(va, vb)
        entry["a_vs_b"] = cmp
        report["per_key"][k] = entry
        if cmp.get("shape_match") == 1.0:
            fro_list.append(cmp["fro_diff"])
            ang_list.append(cmp["princ_angle_mean_deg"])

        parsed = parse_basis_key(k)
        if parsed is not None:
            layer, br = parsed
            report["per_layer_summary"].append(
                {
                    "layer": layer,
                    "branch": br,
                    "key": k,
                    "fro_diff": cmp.get("fro_diff"),
                    "mean_deg": cmp.get("princ_angle_mean_deg"),
                    "cos_col_min": cmp.get("cos_col_min"),
                }
            )

    if fro_list:
        report["global_a_vs_b"] = {
            "common_tensors_compared": len(fro_list),
            "fro_diff_mean": float(sum(fro_list) / len(fro_list)),
            "fro_diff_max": float(max(fro_list)),
            "princ_angle_mean_deg_mean": float(sum(ang_list) / len(ang_list)),
            "princ_angle_mean_deg_max": float(max(ang_list)),
        }

    # sort per_layer by layer then q k v
    order = {"q": 0, "k": 1, "v": 2}
    report["per_layer_summary"].sort(key=lambda x: (x["layer"], order.get(x["branch"], 9)))

    # optional: hidden_size / ranks from layer 0 q
    k0 = "base_model.model.model.decoder.layers.0.self_attn.q_proj_basis"
    if k0 in A and k0 in B and torch.is_tensor(A[k0]):
        d, r = A[k0].shape
        report["inferred_from_layer0_q"] = {"hidden_size": d, "rank_kq": r}

    # print human summary
    print("=== eigen basis 对比 ===")
    print(f"A: {pa}")
    print(f"B: {pb}")
    print(f"键: A={len(keys_a)} B={len(keys_b)} 交集={len(common)} 仅A={len(only_a)} 仅B={len(only_b)}")
    if only_a[:5]:
        print("仅 A 有(示例):", only_a[:5])
    if only_b[:5]:
        print("仅 B 有(示例):", only_b[:5])
    if "global_a_vs_b" in report:
        g = report["global_a_vs_b"]
        print(
            f"同形状张量: {g['common_tensors_compared']} | "
            f"Fro 差 mean/max: {g['fro_diff_mean']:.6f} / {g['fro_diff_max']:.6f} | "
            f"主角度 mean(均值)/max: {g['princ_angle_mean_deg_mean']:.4f}° / {g['princ_angle_mean_deg_max']:.4f}°"
        )

    if args.max_layers_print > 0:
        pls = report["per_layer_summary"]
        for row in pls[: args.max_layers_print * 3]:
            print(
                f"  L{row['layer']:02d} {row['branch']}  "
                f"fro={row['fro_diff']:.4f}  mean_ang={row['mean_deg']:.4f}°  cos_min={row['cos_col_min']:.6f}"
            )
        if len(pls) > args.max_layers_print * 3:
            print(f"  ... ({len(pls)} 条 per-layer 记录，省略；见 JSON)")

    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)

        def json_safe(x: Any) -> Any:
            if isinstance(x, dict):
                return {str(kk): json_safe(vv) for kk, vv in x.items()}
            if isinstance(x, list):
                return [json_safe(vv) for vv in x]
            if isinstance(x, float):
                if math.isnan(x) or math.isinf(x):
                    return None
                return x
            if isinstance(x, (int, str, bool)) or x is None:
                return x
            return str(x)

        with open(outp, "w", encoding="utf-8") as f:
            json.dump(json_safe(report), f, indent=2, ensure_ascii=False)
        print(f"[saved] {outp.resolve()}")


if __name__ == "__main__":
    main()
