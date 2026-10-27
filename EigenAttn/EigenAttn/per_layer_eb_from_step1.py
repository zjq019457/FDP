#!/usr/bin/env python3
"""
Per-layer error_budget (EB) from Step-1 fragility, for pipelines that use
EigenAttn-style ``error_budget`` (e.g. build_eigen_basis_opt_error_budget_offline.py).

Baseline: uniform EB024 → every layer uses the same EB (default 0.024).

Fragile layers (high Step-1 s_tilde) get a *smaller* error_budget → rank search
stops earlier → more conservative / higher rank for that layer.

Mapping (default):
    eb_l = eb_base / (1 + lambda_eb * s_tilde_l)

where s_tilde_l is min-max normalized fragility across 48 layers (same weighted
sum as realloc_rank_from_fragility.py).

Optional ``--normalize mean``: scale all eb_l by eb_base / mean(eb_l) so the
**average** EB across layers equals ``eb_base`` (global aggressiveness comparable
to uniform 0.024, but fragile layers still **relatively** stricter).

Usage:
  python per_layer_eb_from_step1.py \\
    --step1 /path/step1_eb024.json \\
    --out /path/eb024_per_layer_error_budget.json \\
    --eb_base 0.024 --lambda_eb 1.5 --normalize mean
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

# same directory import
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from realloc_rank_from_fragility import (  # noqa: E402
    fragility_vector,
    load_json,
    minmax,
)


_STEP1_METRIC_KEYS = (
    "Z_rel_mse",
    "score_rel_mse_causal",
    "value_only_Z_rel_mse",
    "score_only_AV_rel_mse",
    "entropy_diff_A",
)


def _sanitize_step1_per_layer(per_layer: Dict[str, dict]) -> None:
    """NaN/Inf in CMP-side tensors poison fragility + min-max; replace with 0 for mapping."""
    for _lid, d in per_layer.items():
        if not isinstance(d, dict):
            continue
        for k in _STEP1_METRIC_KEYS:
            if k not in d:
                continue
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                d[k] = 0.0
                continue
            if not math.isfinite(v):
                d[k] = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eb_base", type=float, default=0.024, help="Uniform EB024 reference")
    ap.add_argument(
        "--lambda_eb",
        type=float,
        default=1.5,
        help="Fragility tilt: larger → fragile layers get smaller EB",
    )
    ap.add_argument(
        "--normalize",
        choices=["none", "mean"],
        default="mean",
        help="After inverse map, scale so mean EB equals eb_base (recommended)",
    )
    ap.add_argument("--eb_min", type=float, default=None, help="Clamp lower bound after normalize")
    ap.add_argument("--eb_max", type=float, default=None, help="Clamp upper bound after normalize")
    ap.add_argument(
        "--eb_min_ratio",
        type=float,
        default=None,
        help="After normalize: enforce eb_l >= eb_base * eb_min_ratio (e.g. 0.75 keeps floor vs uniform).",
    )
    ap.add_argument(
        "--freeze_layers",
        type=str,
        default="",
        help='Comma-separated layer indices forced to eb_base after mapping (e.g. "0").',
    )
    ap.add_argument("--wz", type=float, default=1.0)
    ap.add_argument("--ws", type=float, default=1.0)
    ap.add_argument("--wv", type=float, default=1.0)
    ap.add_argument("--wa", type=float, default=1.0)
    ap.add_argument("--wh", type=float, default=1.0)
    ap.add_argument("--zscore", action="store_true")
    args = ap.parse_args()

    data = load_json(args.step1)
    per_layer: Dict[str, dict] = data["per_layer"]
    _sanitize_step1_per_layer(per_layer)
    layers = sorted(int(k) for k in per_layer.keys())

    s_raw, breakdown = fragility_vector(
        per_layer,
        layers,
        args.wz,
        args.ws,
        args.wv,
        args.wa,
        args.wh,
        args.zscore,
    )
    s_tilde = minmax(s_raw)

    eb_inv: List[float] = []
    for i in range(len(layers)):
        st = s_tilde[i]
        eb_inv.append(args.eb_base / (1.0 + args.lambda_eb * st))

    eb_final = list(eb_inv)
    if args.normalize == "mean":
        m = sum(eb_final) / len(eb_final)
        if m > 0:
            scale = args.eb_base / m
            eb_final = [e * scale for e in eb_final]

    if args.eb_min is not None:
        eb_final = [max(args.eb_min, e) for e in eb_final]
    if args.eb_max is not None:
        eb_final = [min(args.eb_max, e) for e in eb_final]

    frozen: List[int] = []
    if (args.freeze_layers or "").strip():
        for t in args.freeze_layers.split(","):
            t = t.strip()
            if t != "":
                frozen.append(int(t))
    for idx in frozen:
        if 0 <= idx < len(eb_final):
            eb_final[idx] = float(args.eb_base)

    if args.eb_min_ratio is not None:
        lo = float(args.eb_base) * float(args.eb_min_ratio)
        eb_final = [max(lo, e) for e in eb_final]

    rows = []
    for i, l in enumerate(layers):
        rows.append(
            {
                "layer": l,
                "fragility_raw": s_raw[i],
                "fragility_s_tilde": s_tilde[i],
                "error_budget_uniform": args.eb_base,
                "error_budget_inverse_only": eb_inv[i],
                "error_budget": eb_final[i],
                "delta_vs_uniform": eb_final[i] - args.eb_base,
                "ratio_vs_uniform": eb_final[i] / args.eb_base,
                **breakdown[l],
            }
        )

    out = {
        "meta": {
            "description": "Per-layer error_budget from Step-1 fragility; fragile → lower EB",
            "step1": args.step1,
            "eb_base_uniform": args.eb_base,
            "lambda_eb": args.lambda_eb,
            "normalize": args.normalize,
            "weights": {"wz": args.wz, "ws": args.ws, "wv": args.wv, "wa": args.wa, "wh": args.wh},
            "zscore_metrics": args.zscore,
            "freeze_layers": frozen,
            "eb_min_ratio": args.eb_min_ratio,
            "mean_eb_final": sum(eb_final) / len(eb_final),
            "min_eb_final": min(eb_final),
            "max_eb_final": max(eb_final),
        },
        "per_layer": rows,
        "error_budget_by_layer": eb_final,
        "error_budget_by_layer_csv": ",".join(f"{e:.6f}" for e in eb_final),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"[OK] {args.out}")
    print(f"     mean EB = {out['meta']['mean_eb_final']:.6f}  (target base {args.eb_base})")
    print(f"     range   = [{out['meta']['min_eb_final']:.6f}, {out['meta']['max_eb_final']:.6f}]")


if __name__ == "__main__":
    main()
