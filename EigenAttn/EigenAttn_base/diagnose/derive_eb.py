"""Per-layer error_budget from Step-1 fragility (paper-aligned formula).

Direct port of ``per_layer_eb_from_step1.py`` from the larger workspace, with
``freeze_layers=[0]`` and ``normalize="mean"`` as defaults so the produced
schedule has the same global aggressiveness as the uniform baseline (only the
*relative* allocation across layers shifts toward more conservative values for
fragile layers).

Mapping:

    eb_l = eb_base / (1 + lambda_eb * s_tilde_l)

where ``s_tilde_l`` is min-max normalised fragility across layers using a
weighted sum of the Step-1 metrics:

    s_l = wz*Z_rel_mse + ws*score_rel_mse_causal + wv*value_only_Z_rel_mse
        + wa*score_only_AV_rel_mse + wh*entropy_diff_A

After mapping, all eb_l can be uniformly rescaled so that ``mean(eb_l)`` equals
``eb_base`` (``normalize="mean"``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_STEP1_METRIC_KEYS = (
    "Z_rel_mse",
    "score_rel_mse_causal",
    "value_only_Z_rel_mse",
    "score_only_AV_rel_mse",
    "entropy_diff_A",
)


def _sanitize(per_layer: Dict[str, dict]) -> None:
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


def _zscore(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    sd = math.sqrt(var) if var > 0 else 1.0
    return [(v - m) / sd for v in values]


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def fragility_vector(
    per_layer: Dict[str, dict],
    layers: Sequence[int],
    wz: float,
    ws: float,
    wv: float,
    wa: float,
    wh: float,
    use_zscore: bool,
) -> Tuple[List[float], Dict[int, dict]]:
    Z = [float(per_layer[str(l)].get("Z_rel_mse", 0.0)) for l in layers]
    S = [float(per_layer[str(l)].get("score_rel_mse_causal", 0.0)) for l in layers]
    V = [float(per_layer[str(l)].get("value_only_Z_rel_mse", 0.0)) for l in layers]
    A = [float(per_layer[str(l)].get("score_only_AV_rel_mse", 0.0)) for l in layers]
    H = [float(per_layer[str(l)].get("entropy_diff_A", 0.0)) for l in layers]

    if use_zscore:
        Z, S, V, A, H = _zscore(Z), _zscore(S), _zscore(V), _zscore(A), _zscore(H)

    s_raw = [
        wz * Z[i] + ws * S[i] + wv * V[i] + wa * A[i] + wh * H[i]
        for i in range(len(layers))
    ]

    breakdown: Dict[int, dict] = {}
    for i, l in enumerate(layers):
        breakdown[l] = {
            "Z_rel_mse_raw": Z[i],
            "score_rel_mse_raw": S[i],
            "value_only_Z_rel_mse_raw": V[i],
            "score_only_AV_rel_mse_raw": A[i],
            "entropy_diff_A_raw": H[i],
            "entropy_penalty_raw": 0.0,
            "used_in_sum": {"Z": Z[i], "score": S[i], "value": V[i], "AV": A[i], "Hpen": 0.0},
            "fragility_weighted": s_raw[i],
        }
    return s_raw, breakdown


def derive_per_layer_eb(
    step1_report: Dict[str, dict],
    *,
    eb_base: float,
    lambda_eb: float = 1.2,
    normalize: str = "mean",
    freeze_layers: Optional[Iterable[int]] = (0,),
    eb_min: Optional[float] = None,
    eb_max: Optional[float] = None,
    eb_min_ratio: Optional[float] = None,
    wz: float = 1.0,
    ws: float = 1.0,
    wv: float = 1.0,
    wa: float = 1.0,
    wh: float = 1.0,
    use_zscore: bool = False,
) -> Dict[str, dict]:
    """Build the per-layer EB schedule from a step1 report.

    Returns a JSON-compatible dict with ``meta``, ``per_layer``, and
    ``error_budget_by_layer`` (a plain Python list ready for
    ``args.per_layer_error_budget``).
    """
    if "per_layer" not in step1_report:
        raise ValueError("step1 report missing 'per_layer'")

    per_layer = dict(step1_report["per_layer"])
    _sanitize(per_layer)
    layers = sorted(int(k) for k in per_layer.keys())
    s_raw, breakdown = fragility_vector(per_layer, layers, wz, ws, wv, wa, wh, use_zscore)
    s_tilde = _minmax(s_raw)

    eb_inv = [eb_base / (1.0 + lambda_eb * st) for st in s_tilde]
    eb_final = list(eb_inv)
    if normalize == "mean":
        m = sum(eb_final) / len(eb_final) if eb_final else 0.0
        if m > 0:
            scale = eb_base / m
            eb_final = [e * scale for e in eb_final]
    if eb_min is not None:
        eb_final = [max(eb_min, e) for e in eb_final]
    if eb_max is not None:
        eb_final = [min(eb_max, e) for e in eb_final]
    frozen = list(freeze_layers or [])
    for idx in frozen:
        if 0 <= idx < len(eb_final):
            eb_final[idx] = float(eb_base)
    if eb_min_ratio is not None:
        lo = float(eb_base) * float(eb_min_ratio)
        eb_final = [max(lo, e) for e in eb_final]

    rows = []
    for i, l in enumerate(layers):
        rows.append(
            {
                "layer": l,
                "fragility_raw": s_raw[i],
                "fragility_s_tilde": s_tilde[i],
                "error_budget_uniform": eb_base,
                "error_budget_inverse_only": eb_inv[i],
                "error_budget": eb_final[i],
                "delta_vs_uniform": eb_final[i] - eb_base,
                "ratio_vs_uniform": eb_final[i] / eb_base if eb_base else 0.0,
                **breakdown[l],
            }
        )

    return {
        "meta": {
            "description": "Per-layer EB from Step-1 fragility; fragile -> lower EB",
            "eb_base_uniform": eb_base,
            "lambda_eb": lambda_eb,
            "normalize": normalize,
            "weights": {"wz": wz, "ws": ws, "wv": wv, "wa": wa, "wh": wh},
            "zscore_metrics": use_zscore,
            "freeze_layers": frozen,
            "eb_min_ratio": eb_min_ratio,
            "mean_eb_final": sum(eb_final) / len(eb_final) if eb_final else 0.0,
            "min_eb_final": min(eb_final) if eb_final else 0.0,
            "max_eb_final": max(eb_final) if eb_final else 0.0,
            "step1_meta": step1_report.get("meta", {}),
        },
        "per_layer": rows,
        "error_budget_by_layer": eb_final,
        "error_budget_by_layer_csv": ",".join(f"{e:.6f}" for e in eb_final),
    }


def save_dynamic_eb(report: Dict[str, dict], out_json: str) -> None:
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")


__all__ = ["fragility_vector", "derive_per_layer_eb", "save_dynamic_eb"]
