"""
Fragility-weighted rank reallocation under a fixed total budget.

Reads Step 1 diagnostic JSON (step1_diagnose.py output) and a baseline
low_rank_config (e.g. EB024). Builds per-layer fragility scores from the
same ingredients discussed with the user:

  s_l = wz*Z_rel + ws*Score_rel + wv*Value_rel + wa*AV_rel + wh*entropy_penalty

where entropy_penalty = max(0, -entropy_diff_A) (attention collapse / sharpening).

Then allocates new ranks so that high-fragility layers receive a larger share
of the *same* global budgets as the baseline:

  B_kq = sum_l rank_kq_l^(baseline)
  B_v  = sum_l rank_v_l^(baseline)

Allocation (default):
  raw_l = baseline_l * (1 + lambda * s_tilde_l)
  float_target_l = B * raw_l / sum(raw)
  snap to multiples of num_heads, clamp [min_rank, max_rank], then greedy fix
  so sums match B_kq and B_v exactly.

Usage:
  python realloc_rank_from_fragility.py \\
    --step1 /path/step1_eb024.json \\
    --baseline_low /diskF/jiaqizhang/eb024/final_model/low_rank_config_opt30b_paper_eb024.json \\
    --out_json /path/low_rank_realloc_eb024.json \\
    --report /path/realloc_report.json \\
    --num_heads 56 --max_rank 7168 \\
    --lambda_frag 2.0 \\
    --freeze_layers 0

`--freeze_layers` keeps selected indices at the baseline [rkq,rv] (recommended for L0
when Step~1 fragility is low but the layer is structurally load-bearing).
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Sequence, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def zscore(x: List[float]) -> List[float]:
    n = len(x)
    if n == 0:
        return x
    m = sum(x) / n
    v = sum((t - m) ** 2 for t in x) / max(1, n - 1)
    s = math.sqrt(v) if v > 0 else 1.0
    return [(t - m) / s for t in x]


def minmax(x: List[float], eps: float = 1e-12) -> List[float]:
    lo, hi = min(x), max(x)
    span = hi - lo
    if span < eps:
        return [0.5 for _ in x]
    return [(t - lo) / span for t in x]


def snap_to_grid(x: float, step: int, lo: int, hi: int) -> int:
    v = int(round(x / step)) * step
    return max(lo, min(hi, v))


def fragility_vector(
    per_layer: Dict[str, dict],
    layers: List[int],
    wz: float,
    ws: float,
    wv: float,
    wa: float,
    wh: float,
    zscore_norm: bool,
) -> Tuple[List[float], Dict[int, dict]]:
    Z_raw = [per_layer[str(l)]["Z_rel_mse"] for l in layers]
    Sc_raw = [per_layer[str(l)]["score_rel_mse_causal"] for l in layers]
    Va_raw = [per_layer[str(l)]["value_only_Z_rel_mse"] for l in layers]
    Av_raw = [per_layer[str(l)]["score_only_AV_rel_mse"] for l in layers]
    dH = [per_layer[str(l)]["entropy_diff_A"] for l in layers]
    Hpen_raw = [max(0.0, -h) for h in dH]

    if zscore_norm:
        Z, Sc, Va, Av, Hpen = (
            zscore(Z_raw),
            zscore(Sc_raw),
            zscore(Va_raw),
            zscore(Av_raw),
            zscore(Hpen_raw),
        )
    else:
        Z, Sc, Va, Av, Hpen = Z_raw, Sc_raw, Va_raw, Av_raw, Hpen_raw

    s: List[float] = []
    breakdown: Dict[int, dict] = {}
    for i, l in enumerate(layers):
        sl = (
            wz * Z[i]
            + ws * Sc[i]
            + wv * Va[i]
            + wa * Av[i]
            + wh * Hpen[i]
        )
        s.append(sl)
        breakdown[l] = {
            "Z_rel_mse_raw": Z_raw[i],
            "score_rel_mse_raw": Sc_raw[i],
            "value_only_Z_rel_mse_raw": Va_raw[i],
            "score_only_AV_rel_mse_raw": Av_raw[i],
            "entropy_diff_A_raw": dH[i],
            "entropy_penalty_raw": Hpen_raw[i],
            "used_in_sum": {
                "Z": Z[i],
                "score": Sc[i],
                "value": Va[i],
                "AV": Av[i],
                "Hpen": Hpen[i],
            },
            "fragility_weighted": sl,
        }
    return s, breakdown


def project_budget(
    baseline: List[int],
    s_tilde: List[float],
    B_target: int,
    num_heads: int,
    min_rank: int,
    max_rank: int,
    lam: float,
    power: float,
    frozen: set,
) -> List[int]:
    """
    baseline[i], s_tilde[i] in [0,1]. Higher s_tilde -> larger multiplier.
    raw_i = baseline_i * (1 + lam * s_tilde_i) ** power

    Indices in `frozen` keep baseline[i] exactly; remaining layers absorb
    B_target - sum(baseline[i] for i in frozen).
    """
    n = len(baseline)
    cur = list(baseline)
    if not frozen:
        active = list(range(n))
        B_rem = B_target
        base_rem = baseline
        st_rem = s_tilde
    else:
        for i in frozen:
            cur[i] = baseline[i]
        active = [i for i in range(n) if i not in frozen]
        B_rem = B_target - sum(baseline[i] for i in frozen)
        base_rem = [baseline[i] for i in active]
        st_rem = [s_tilde[i] for i in active]

    m = len(active)
    if m == 0:
        return cur
    raw = [
        float(base_rem[j]) * (1.0 + lam * st_rem[j]) ** power
        for j in range(m)
    ]
    sraw = sum(raw)
    if sraw <= 0:
        return cur
    floats_full = [0.0] * n
    for j, idx in enumerate(active):
        floats_full[idx] = B_rem * raw[j] / sraw
    cur = list(baseline) if not frozen else cur
    for i in range(n):
        if i in frozen:
            cur[i] = baseline[i]
            continue
        cur[i] = snap_to_grid(floats_full[i], num_heads, min_rank, max_rank)

    def fix(delta: int) -> None:
        step = num_heads
        guard = 0
        while delta != 0 and guard < 100000:
            guard += 1
            if delta > 0:
                best_i = -1
                best_gap = -1e9
                for i in active:
                    if i in frozen:
                        continue
                    if cur[i] + step > max_rank:
                        continue
                    gap = floats_full[i] - cur[i]
                    if gap > best_gap:
                        best_gap = gap
                        best_i = i
                if best_i < 0:
                    break
                cur[best_i] += step
                delta -= step
            else:
                best_i = -1
                best_gap = 1e9
                for i in active:
                    if i in frozen:
                        continue
                    if cur[i] - step < min_rank:
                        continue
                    gap = floats_full[i] - cur[i]
                    if gap < best_gap:
                        best_gap = gap
                        best_i = i
                if best_i < 0:
                    break
                cur[best_i] -= step
                delta += step

    delta = B_target - sum(cur)
    fix(delta)
    return cur


def main():
    ap = argparse.ArgumentParser(description="Fragility-based rank reallocation (fixed B_kq, B_v).")
    ap.add_argument("--step1", required=True, help="step1_diagnose.py JSON output")
    ap.add_argument("--baseline_low", required=True, help="Baseline low_rank_config.json (list of [rkq, rv])")
    ap.add_argument("--out_json", required=True, help="Output new low_rank_config JSON path")
    ap.add_argument("--report", default=None, help="Optional detailed report JSON")

    ap.add_argument("--num_heads", type=int, default=56)
    ap.add_argument("--max_rank", type=int, default=7168, help="Clamp rank_kq and rank_v (OPT-30B hidden)")
    ap.add_argument("--min_rank", type=int, default=None, help="Default: num_heads")

    ap.add_argument("--wz", type=float, default=1.0, help="Weight on Z_rel_mse")
    ap.add_argument("--ws", type=float, default=1.0, help="Weight on score_rel_mse_causal")
    ap.add_argument("--wv", type=float, default=1.0, help="Weight on value_only_Z_rel_mse")
    ap.add_argument("--wa", type=float, default=1.0, help="Weight on score_only_AV_rel_mse")
    ap.add_argument("--wh", type=float, default=1.0, help="Weight on max(0,-entropy_diff_A)")
    ap.add_argument("--zscore", action="store_true", help="Z-score each metric across layers before weighting")

    ap.add_argument("--lambda_frag", type=float, default=2.0, help="Strength of fragility tilt (0 = keep baseline shape)")
    ap.add_argument("--power", type=float, default=1.0, help="Exponent on (1 + lambda * s_tilde)")
    ap.add_argument(
        "--v_frag_blend",
        type=float,
        default=0.35,
        help="For V budget only: blend s_tilde with minmax(value_only); 0=pure s_tilde, 1=pure value stress",
    )
    ap.add_argument(
        "--freeze_layers",
        type=str,
        default="",
        help="Comma-separated layer indices to keep at baseline rank (e.g. '0' to preserve L0)",
    )

    args = ap.parse_args()
    min_rank = args.min_rank if args.min_rank is not None else args.num_heads
    h = args.num_heads
    frozen: set = set()
    if args.freeze_layers.strip():
        for tok in args.freeze_layers.split(","):
            tok = tok.strip()
            if tok:
                frozen.add(int(tok))

    data = load_json(args.step1)
    per_layer: Dict[str, dict] = data["per_layer"]
    layers = sorted(int(k) for k in per_layer.keys())

    baseline = load_json(args.baseline_low)
    if len(baseline) != len(layers):
        raise SystemExit(
            f"baseline_low has {len(baseline)} layers, step1 has {len(layers)}"
        )

    base_kq = [int(baseline[i][0]) for i in range(len(layers))]
    base_v = [int(baseline[i][1]) for i in range(len(layers))]
    for i, l in enumerate(layers):
        if base_kq[i] != per_layer[str(l)]["rank_kq"] or base_v[i] != per_layer[str(l)]["rank_v"]:
            # warn but continue — step1 may be from a different run
            pass

    B_kq = sum(base_kq)
    B_v = sum(base_v)

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
    val_only = [per_layer[str(l)]["value_only_Z_rel_mse"] for l in layers]
    val_tilde = minmax(val_only)
    s_tilde_v = [
        (1.0 - args.v_frag_blend) * s_tilde[i] + args.v_frag_blend * val_tilde[i]
        for i in range(len(layers))
    ]

    new_kq = project_budget(
        base_kq,
        s_tilde,
        B_kq,
        h,
        min_rank,
        args.max_rank,
        args.lambda_frag,
        args.power,
        frozen,
    )
    new_v = project_budget(
        base_v,
        s_tilde_v,
        B_v,
        h,
        min_rank,
        args.max_rank,
        args.lambda_frag,
        args.power,
        frozen,
    )

    out_cfg = [[new_kq[i], new_v[i]] for i in range(len(layers))]
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2)
        f.write("\n")

    report = {
        "args": vars(args),
        "frozen_layers": sorted(frozen),
        "budget": {
            "B_kq_baseline": B_kq,
            "B_v_baseline": B_v,
            "B_kq_new": sum(new_kq),
            "B_v_new": sum(new_v),
        },
        "per_layer": [],
    }
    eps0 = 1.0
    for i, l in enumerate(layers):
        lam = args.lambda_frag
        st = s_tilde[i]
        epsilon_l = eps0 / (1.0 + lam * st) if lam * st >= 0 else eps0
        row = {
            "layer": l,
            "fragility_raw": s_raw[i],
            "s_tilde": st,
            "s_tilde_v_blend": s_tilde_v[i],
            "epsilon_l_form": epsilon_l,
            "baseline": [base_kq[i], base_v[i]],
            "new": [new_kq[i], new_v[i]],
            "delta_kq": new_kq[i] - base_kq[i],
            "delta_v": new_v[i] - base_v[i],
        }
        row.update(breakdown[l])
        report["per_layer"].append(row)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    # concise stdout
    changed = sum(
        1
        for i in range(len(layers))
        if new_kq[i] != base_kq[i] or new_v[i] != base_v[i]
    )
    print(f"[OK] Wrote {args.out_json}")
    print(f"     B_kq: {B_kq} -> {sum(new_kq)}  B_v: {B_v} -> {sum(new_v)}")
    print(f"     Layers with any rank change: {changed}/{len(layers)}")
    print(f"     lambda_frag={args.lambda_frag}  v_frag_blend={args.v_frag_blend}")


if __name__ == "__main__":
    main()
