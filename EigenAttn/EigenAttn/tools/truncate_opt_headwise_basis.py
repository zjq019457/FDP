#!/usr/bin/env python3
"""Truncate OPT headwise block-diagonal eigen bases to target ranks.

OPT bases from build_eigen_basis_opt_error_budget_offline._pack_headwise_basis are
stored as U ∈ R^{D×R} with per-head blocks on the diagonal:
  columns [h*k : (h+1)*k] only nonzero in rows [h*hd : (h+1)*hd],
  where k = R / num_heads and hd = head_dim.

Naive column slice U[:, :r_new] destroys that packing whenever r_new < R.
Correct truncation keeps the first k_new columns from EACH head block and
re-packs into U' ∈ R^{D×r_new}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

Pair = Tuple[int, int]


def truncate_headwise_packed(
    U: torch.Tensor,
    new_rank: int,
    *,
    num_heads: int = 56,
    head_dim: int = 128,
) -> torch.Tensor:
    """Re-pack a headwise block-diagonal basis to a smaller total rank."""
    if U.ndim == 3 and U.shape[0] == 1:
        U = U.squeeze(0)
    if U.ndim != 2:
        raise ValueError(f"expected 2D basis, got shape {tuple(U.shape)}")
    d, r = int(U.shape[0]), int(U.shape[1])
    if d != num_heads * head_dim:
        raise ValueError(
            f"basis rows {d} != num_heads*head_dim={num_heads * head_dim}"
        )
    if new_rank <= 0:
        raise ValueError(f"new_rank must be positive, got {new_rank}")
    if new_rank % num_heads != 0:
        raise ValueError(
            f"new_rank={new_rank} not divisible by num_heads={num_heads}"
        )
    if r % num_heads != 0:
        raise ValueError(
            f"basis width {r} not divisible by num_heads={num_heads} "
            "(not a headwise packed OPT basis?)"
        )
    if new_rank > r:
        raise ValueError(f"cannot expand basis width {r} -> {new_rank}")
    if new_rank == r:
        return U.contiguous()

    k_old = r // num_heads
    k_new = new_rank // num_heads
    if k_new > head_dim:
        raise ValueError(f"per-head rank {k_new} exceeds head_dim={head_dim}")

    out = torch.zeros((d, new_rank), dtype=U.dtype, device=U.device)
    for h in range(num_heads):
        rs = slice(h * head_dim, (h + 1) * head_dim)
        cs_old = slice(h * k_old, h * k_old + k_new)
        cs_new = slice(h * k_new, (h + 1) * k_new)
        out[rs, cs_new] = U[rs, cs_old]
    return out.contiguous()


def offblock_fraction(
    U: torch.Tensor,
    *,
    num_heads: int = 56,
    head_dim: int = 128,
) -> float:
    """Fraction of Frobenius energy outside headwise blocks."""
    if U.ndim == 3 and U.shape[0] == 1:
        U = U.squeeze(0)
    d, r = U.shape
    if r % num_heads != 0:
        return float("nan")
    k = r // num_heads
    mask = torch.zeros_like(U, dtype=torch.bool)
    for h in range(num_heads):
        rs = slice(h * head_dim, (h + 1) * head_dim)
        cs = slice(h * k, (h + 1) * k)
        mask[rs, cs] = True
    total = float(U.float().pow(2).sum().item())
    if total <= 0:
        return 0.0
    off = float(U.float().pow(2).masked_fill(mask, 0.0).sum().item())
    return off / total


def truncate_opt_basis_dict(
    obj: dict,
    pairs: Sequence[Pair],
    *,
    num_heads: int = 56,
    head_dim: int = 128,
    key_prefix: str = "base_model.model.model.decoder.layers",
    naive: bool = False,
) -> Tuple[dict, List[dict]]:
    """Truncate all Q/K/V bases in an OPT eigen-basis .pt dict.

    naive=True reproduces the broken U[:, :want] path (for ablation only).
    """
    out = dict(obj)
    report: List[dict] = []
    for i, (kq, v) in enumerate(pairs):
        prefix = f"{key_prefix}.{i}.self_attn"
        row: Dict[str, object] = {"layer": i, "kq": int(kq), "v": int(v), "naive": naive}
        for key, want in (
            (f"{prefix}.q_proj_basis", int(kq)),
            (f"{prefix}.k_proj_basis", int(kq)),
            (f"{prefix}.v_proj_basis", int(v)),
        ):
            if key not in out:
                raise KeyError(key)
            t = out[key]
            if not torch.is_tensor(t):
                t = torch.as_tensor(t)
            if t.ndim == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.shape[1] < want:
                raise ValueError(f"{key}: width {t.shape[1]} < want {want}")
            if naive:
                t2 = t[:, :want].contiguous()
            else:
                t2 = truncate_headwise_packed(
                    t, want, num_heads=num_heads, head_dim=head_dim
                )
            out[key] = t2
            name = key.rsplit(".", 1)[-1]
            row[f"{name}_shape"] = list(t2.shape)
            row[f"{name}_offblock"] = offblock_fraction(
                t2, num_heads=num_heads, head_dim=head_dim
            )
        report.append(row)
    return out, report


def truncate_opt_basis_file(
    src_pt: Union[str, Path],
    pairs: Sequence[Pair],
    dst_pt: Union[str, Path],
    *,
    num_heads: int = 56,
    head_dim: int = 128,
    naive: bool = False,
) -> dict:
    src_pt = Path(src_pt)
    dst_pt = Path(dst_pt)
    obj = torch.load(src_pt, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"unexpected basis type {type(obj)}")
    out, layers = truncate_opt_basis_dict(
        obj, pairs, num_heads=num_heads, head_dim=head_dim, naive=naive
    )
    dst_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst_pt)
    del obj, out
    return {
        "src": str(src_pt),
        "dst": str(dst_pt),
        "naive": naive,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "layers": layers,
    }


def load_pairs(path: Union[str, Path]) -> List[Pair]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "low_rank_config" in raw:
        raw = raw["low_rank_config"]
    elif isinstance(raw, dict) and "schedule" in raw:
        return [
            (int(L["k_proj"]["rank"]), int(L["v_proj"]["rank"]))
            for L in raw["schedule"]
        ]
    return [(int(a), int(b)) for a, b in raw]


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--ranks_json", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--num_heads", type=int, default=56)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument(
        "--naive",
        action="store_true",
        help="broken column-slice truncate (ablation only)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    pairs = load_pairs(args.ranks_json)
    rep = truncate_opt_basis_file(
        args.src,
        pairs,
        args.dst,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        naive=args.naive,
    )
    # compact stdout
    bad = sum(
        1
        for L in rep["layers"]
        for k in ("q_proj_basis_offblock", "k_proj_basis_offblock", "v_proj_basis_offblock")
        if float(L.get(k, 0.0)) > 1e-6
    )
    print(
        json.dumps(
            {
                "ok": True,
                "naive": rep["naive"],
                "n_layers": len(rep["layers"]),
                "layers_with_offblock_energy": bad,
                "dst": rep["dst"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
