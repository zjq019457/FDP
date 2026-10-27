"""Strict MPT-side protocol helpers for the frozen FDP-EB campaign.

This module contains only protocol validation/accounting helpers. It does not
load weights, select quality-driven settings, launch jobs, or modify the
official EigenAttention source tree.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence


PROFILE_SOURCE_FIELDS = (
    "Z_rel_mse",
    "score_rel_mse_causal",
    "entropy_diff_A",
)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_mpt_config(config: Any) -> dict[str, Any]:
    """Validate architecture from config fields, never from a path/name."""
    get = config.get if isinstance(config, Mapping) else lambda k, d=None: getattr(config, k, d)
    model_type = get("model_type")
    layers = get("n_layers")
    width = get("d_model")
    heads = get("n_heads")
    context = get("max_seq_len")
    attn = get("attn_config", {}) or {}
    if not isinstance(attn, Mapping):
        attn = vars(attn)
    if model_type != "mpt":
        raise ValueError(f"expected model_type='mpt', got {model_type!r}")
    if (layers, width, heads, context) != (32, 4096, 32, 2048):
        raise ValueError(
            "checkpoint config differs from frozen MPT-7B binding: "
            f"layers={layers}, width={width}, heads={heads}, context={context}"
        )
    if attn.get("alibi") is not True:
        raise ValueError("frozen MPT-7B attention requires ALiBi=True")
    if attn.get("attn_type", "multihead_attention") != "multihead_attention":
        raise ValueError("unexpected MPT attention type")
    return {
        "model_type": model_type,
        "layers": int(layers),
        "hidden_size": int(width),
        "attention_heads": int(heads),
        "max_seq_len": int(context),
        "head_dim": int(width) // int(heads),
        "alibi": True,
        "attn_type": attn.get("attn_type", "multihead_attention"),
    }


def bind_mpt_eval_context_limit(hflm: Any) -> int:
    """Bind lm-eval HFLM's explicit context cap to MPT config.max_seq_len.

    The upstream HFLM fallback does not inspect MptConfig.max_seq_len. This
    helper sets its explicit ``_max_length`` so token preparation cannot rely
    on a tokenizer sentinel or a generic fallback.
    """
    model = getattr(hflm, "model", None)
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("HFLM has no loaded model config")
    identity = validate_mpt_config(config)
    context = identity["max_seq_len"]
    existing = getattr(hflm, "_max_length", None)
    if existing not in (None, 0, context):
        raise ValueError(f"explicit HFLM max_length={existing} conflicts with MPT max_seq_len={context}")
    hflm._max_length = context
    if getattr(hflm, "max_length", None) != context:
        raise ValueError("HFLM context limit did not resolve to MPT max_seq_len")
    return context


def _layer_rows(profile: Mapping[str, Any], expected_layers: int) -> list[dict[str, Any]]:
    raw = profile.get("per_layer")
    if isinstance(raw, Mapping):
        rows = []
        for key, record in raw.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"per_layer[{key!r}] is not a record")
            row = dict(record)
            row.setdefault("layer", int(key))
            rows.append(row)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        rows = [dict(x) for x in raw]
    else:
        raise ValueError("profile must contain per_layer mapping or row sequence")
    rows.sort(key=lambda r: int(r["layer"]))
    ids = [int(r["layer"]) for r in rows]
    if ids != list(range(expected_layers)):
        raise ValueError(f"expected contiguous layers 0..{expected_layers-1}, got {ids}")
    return rows


def derive_three_component_profile(
    profile: Mapping[str, Any],
    *,
    epsilon_base: float,
    lambda_value: float = 1.2,
    expected_layers: int = 32,
    freeze_layer0_at_base: bool = True,
) -> dict[str, Any]:
    """Sum raw Z/causal-score/entropy-penalty, min-max once, then map epsilon.

    The mean normalization is applied to mapped epsilons before the frozen
    Layer-0 override, matching the Llama-3 final-method ordering.
    """
    if not (math.isfinite(epsilon_base) and epsilon_base > 0):
        raise ValueError("epsilon_base must be finite and positive")
    if not (math.isfinite(lambda_value) and lambda_value >= 0):
        raise ValueError("lambda must be finite and nonnegative")
    rows = _layer_rows(profile, expected_layers)
    raw_scores: list[float] = []
    component_rows: list[dict[str, float | int]] = []
    for row in rows:
        vals: dict[str, float] = {}
        for field in PROFILE_SOURCE_FIELDS:
            try:
                v = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"layer {row['layer']} missing/invalid {field}") from exc
            if not math.isfinite(v):
                raise ValueError(f"layer {row['layer']} has non-finite {field}")
            vals[field] = v
        if vals["Z_rel_mse"] < 0 or vals["score_rel_mse_causal"] < 0:
            raise ValueError("relative reconstruction/score errors must be nonnegative")
        entropy_penalty = max(0.0, -vals["entropy_diff_A"])
        score = vals["Z_rel_mse"] + vals["score_rel_mse_causal"] + entropy_penalty
        raw_scores.append(score)
        component_rows.append({
            "layer": int(row["layer"]),
            "relative_z": vals["Z_rel_mse"],
            "causal_score": vals["score_rel_mse_causal"],
            "entropy_penalty": entropy_penalty,
            "fragility_sum_raw": score,
        })

    lo, hi = min(raw_scores), max(raw_scores)
    q = [0.0] * expected_layers if hi == lo else [(x - lo) / (hi - lo) for x in raw_scores]
    mapped = [epsilon_base / (1.0 + lambda_value * x) for x in q]
    mean = sum(mapped) / expected_layers
    if mean <= 0 or not math.isfinite(mean):
        raise ValueError("invalid mapped epsilon mean")
    scale = epsilon_base / mean
    mean_normalized = [x * scale for x in mapped]
    final = list(mean_normalized)
    if freeze_layer0_at_base:
        final[0] = epsilon_base

    result_rows = []
    for i, row in enumerate(component_rows):
        result_rows.append({
            **row,
            "q": q[i],
            "epsilon_raw": mapped[i],
            "epsilon_mean_normalized": mean_normalized[i],
            "epsilon_final": final[i],
        })
    return {
        "schema": "MPT7B_FDP_EB_THREE_COMPONENT_PROFILE_V1",
        "epsilon_base": float(epsilon_base),
        "lambda": float(lambda_value),
        "aggregation": "sum_raw_then_layerwise_minmax",
        "mean_normalization": "mean(mapped_epsilon)=epsilon_base_before_layer0_override",
        "layer0_effective_epsilon": "epsilon_base" if freeze_layer0_at_base else "mapped",
        "mean_epsilon_before_layer0_override": sum(mean_normalized) / expected_layers,
        "mean_epsilon_final": sum(final) / expected_layers,
        "profile_sha256": canonical_json_sha256(result_rows),
        "per_layer": result_rows,
    }


def strict_threshold_grid(start: str = "1.00", stop: str = "0.30", step: str = "0.02") -> list[float]:
    """Exact-decimal descending grid, inclusive at both endpoints."""
    a, b, d = Decimal(start), Decimal(stop), Decimal(step)
    if d <= 0 or a < b or (a - b) % d != 0:
        raise ValueError("grid must descend in exact equal steps and include stop")
    n = int((a - b) / d)
    return [float(a - i * d) for i in range(n + 1)]


def select_last_passing_before_first_failure(
    ordered_trials: Sequence[Mapping[str, Any]],
    *,
    epsilon: float,
    expected_grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Apply frozen first-failure semantics; never resume after first failure."""
    if not ordered_trials:
        raise ValueError("no threshold trials")
    if expected_grid is not None:
        actual = [float(t["threshold"]) for t in ordered_trials]
        if actual != list(expected_grid):
            raise ValueError("trials do not exactly match frozen threshold grid")
    last_pass = None
    audited = []
    failed = False
    for t in ordered_trials:
        threshold = float(t["threshold"])
        error = float(t["relative_error"])
        if not math.isfinite(threshold) or not math.isfinite(error) or error < 0:
            raise ValueError("non-finite threshold/error or negative error")
        if failed:
            raise ValueError("trials continued after first failure")
        passed = error <= epsilon
        audited.append({**dict(t), "pass": passed})
        if passed:
            last_pass = dict(t)
        else:
            failed = True
    return {
        "epsilon": float(epsilon),
        "selected": last_pass,
        "stopped_at_first_failure": failed,
        "trials": audited,
    }


def run_mpt_strict_layer_search(
    *,
    epsilon: float,
    hidden_states: Sequence[Any],
    sample_ids: Sequence[str],
    sample_input_sha256: Sequence[str],
    position_bias: Any,
    attention_mask: Any,
    operator_binding: Mapping[str, Any],
    build_candidate: Any,
    dense_forward: Any,
    compressed_forward: Any,
    n_heads: int = 32,
    hidden_size: int = 4096,
    rank_floor: int = 64,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate one MPT layer over one frozen set of inputs, stopping first fail.

    ``build_candidate(threshold)`` must return a mapping containing
    ``rank_kq``, ``rank_v``, and ``candidate``. Both forward callbacks receive
    the same hidden state, ALiBi ``position_bias``, and causal mask. The caller
    must bind these callbacks to the checkpoint-specific MPT block and its
    frozen calibration activations.
    """
    import torch

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not hidden_states or len(hidden_states) != len(sample_ids) or len(hidden_states) != len(sample_input_sha256):
        raise ValueError("frozen inputs, sample IDs, and input hashes must be nonempty and aligned")
    if len(set(sample_ids)) != len(sample_ids) or any(not x for x in sample_input_sha256):
        raise ValueError("duplicate sample IDs or missing input hash")
    required_binding = {
        "model_type": "mpt",
        "attention_impl": "mpt_native",
        "alibi": True,
        "same_checkpoint_for_dense_and_compressed": True,
        "same_position_bias_and_mask_for_all_candidates": True,
    }
    for key, expected in required_binding.items():
        if operator_binding.get(key) != expected:
            raise ValueError(f"operator binding mismatch for {key}: expected {expected!r}")
    grid = list(thresholds) if thresholds is not None else strict_threshold_grid()
    if not grid or grid[0] != 1.0 or grid[-1] != 0.30:
        raise ValueError("strict MPT search must use the frozen 1.00..0.30 grid")
    if any(abs((grid[i] - grid[i + 1]) - 0.02) > 1e-12 for i in range(len(grid) - 1)):
        raise ValueError("strict MPT threshold grid must descend by 0.02")

    with torch.no_grad():
        refs = [dense_forward(x, position_bias, attention_mask) for x in hidden_states]
    ref_shapes = [list(x.shape) for x in refs]
    if any(not torch.isfinite(x).all().item() for x in refs):
        raise ValueError("non-finite dense MPT block output")
    denominator_sq = sum(float(x.float().square().sum().item()) for x in refs)
    denominator = math.sqrt(max(denominator_sq, 1e-30))

    trials = []
    selected = None
    stopped = False
    for threshold in grid:
        built = build_candidate(float(threshold))
        rkq, rv = int(built["rank_kq"]), int(built["rank_v"])
        validate_rank_pair(rkq, rv, n_heads=n_heads, hidden_size=hidden_size, rank_floor=rank_floor)
        with torch.no_grad():
            got = [compressed_forward(built["candidate"], x, position_bias, attention_mask)
                   for x in hidden_states]
        if [list(x.shape) for x in got] != ref_shapes:
            raise ValueError(f"candidate threshold {threshold} changed output shape")
        if any(not torch.isfinite(x).all().item() for x in got):
            raise ValueError(f"non-finite compressed MPT output at threshold {threshold}")
        error_sq = sum(float((a.float() - b.float()).square().sum().item()) for a, b in zip(refs, got))
        relative_error = math.sqrt(error_sq) / denominator
        passed = relative_error <= epsilon
        row = {
            "threshold": float(threshold), "rank_kq": rkq, "rank_v": rv,
            "relative_error": relative_error, "pass": passed,
        }
        trials.append(row)
        if passed:
            selected = row
        else:
            stopped = True
            break
    return {
        "schema": "MPT_NATIVE_STRICT_FIRST_FAILURE_LAYER_SEARCH_V1",
        "epsilon": float(epsilon),
        "selected": selected,
        "stopped_at_first_failure": stopped,
        "sample_ids": list(sample_ids),
        "sample_input_sha256": list(sample_input_sha256),
        "operator_binding": dict(operator_binding),
        "trials": trials,
    }


def validate_rank_pair(
    rank_kq: int,
    rank_v: int,
    *,
    n_heads: int = 32,
    hidden_size: int = 4096,
    rank_floor: int = 64,
) -> None:
    for label, rank in (("rank_kq", rank_kq), ("rank_v", rank_v)):
        if isinstance(rank, bool) or int(rank) != rank:
            raise ValueError(f"{label} must be an integer")
        rank = int(rank)
        if rank < rank_floor or rank > hidden_size or rank % n_heads != 0:
            raise ValueError(
                f"illegal {label}={rank}; require {rank_floor} <= rank <= {hidden_size} "
                f"and rank divisible by n_heads={n_heads}"
            )


def validate_rank_schedule(
    schedule: Sequence[Sequence[int]],
    *,
    n_layers: int = 32,
    n_heads: int = 32,
    hidden_size: int = 4096,
    rank_floor: int = 64,
) -> list[tuple[int, int]]:
    if len(schedule) != n_layers:
        raise ValueError(f"expected {n_layers} layer rank pairs, got {len(schedule)}")
    rows: list[tuple[int, int]] = []
    for i, pair in enumerate(schedule):
        if len(pair) != 2:
            raise ValueError(f"layer {i} rank record must be [rank_kq,rank_v]")
        rkq, rv = int(pair[0]), int(pair[1])
        validate_rank_pair(rkq, rv, n_heads=n_heads, hidden_size=hidden_size, rank_floor=rank_floor)
        rows.append((rkq, rv))
    return rows


def rank_schedule_sha256(schedule: Sequence[Sequence[int]]) -> str:
    normalized = [[int(pair[0]), int(pair[1])] for pair in schedule]
    return canonical_json_sha256({"rank_kq_rank_v_by_layer": normalized})


def predicted_cache_bytes(
    schedule: Sequence[Sequence[int]],
    *,
    sequence_length: int = 2048,
    batch_size: int = 1,
    bytes_per_element: int = 2,
) -> int:
    """Geometry estimate only; must not be called an actual materialized lock."""
    return int(sequence_length * batch_size * bytes_per_element * sum(int(k) + int(v) for k, v in schedule))


def dense_cache_bytes(
    *,
    n_layers: int = 32,
    hidden_size: int = 4096,
    sequence_length: int = 2048,
    batch_size: int = 1,
    bytes_per_element: int = 2,
) -> int:
    """Dense K+V cache bytes for MPT MHA, not checkpoint or weight bytes."""
    return int(n_layers * 2 * hidden_size * sequence_length * batch_size * bytes_per_element)


def actual_cache_census(
    past_key_values: Sequence[Sequence[Any]],
    schedule: Sequence[Sequence[int]],
    *,
    n_layers: int = 32,
    n_heads: int = 32,
    sequence_length: int = 2048,
) -> dict[str, Any]:
    """Count and shape-check actual per-layer MPT (K,V) cache tensors."""
    ranks = validate_rank_schedule(schedule, n_layers=n_layers, n_heads=n_heads)
    if len(past_key_values) != n_layers:
        raise ValueError(f"expected {n_layers} cache layers, got {len(past_key_values)}")
    layer_rows = []
    total = 0
    dtypes = set()
    devices = set()
    for i, (kv, ranks_i) in enumerate(zip(past_key_values, ranks)):
        if len(kv) != 2:
            raise ValueError(f"layer {i} cache must be (K,V)")
        k, v = kv
        if len(k.shape) != 4 or len(v.shape) != 4:
            raise ValueError(f"layer {i} K/V must be rank-4 [B,H,S,D] tensors")
        rkq, rv = ranks_i
        expected_k = (k.shape[0], n_heads, sequence_length, rkq // n_heads)
        expected_v = (v.shape[0], n_heads, sequence_length, rv // n_heads)
        if tuple(k.shape) != expected_k or tuple(v.shape) != expected_v:
            raise ValueError(
                f"layer {i} actual cache shapes K={tuple(k.shape)} V={tuple(v.shape)} "
                f"do not match expected K={expected_k} V={expected_v}"
            )
        layer_bytes = int(k.numel() * k.element_size() + v.numel() * v.element_size())
        total += layer_bytes
        dtypes.update((str(k.dtype), str(v.dtype)))
        devices.update((str(k.device), str(v.device)))
        layer_rows.append({
            "layer": i,
            "rank_kq": rkq,
            "rank_v": rv,
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "dtype": str(k.dtype),
            "device": str(k.device),
            "bytes": layer_bytes,
        })
    return {
        "status": "ACTUAL_TENSOR_CENSUS_PASS",
        "accounting": "sum(K.numel*K.element_size + V.numel*V.element_size) over all layers",
        "b_live_bytes": total,
        "layers": n_layers,
        "sequence_length": sequence_length,
        "batch_size": int(past_key_values[0][0].shape[0]),
        "dtypes": sorted(dtypes),
        "devices": sorted(devices),
        "rank_schedule_sha256": rank_schedule_sha256(ranks),
        "per_layer": layer_rows,
    }


__all__ = [
    "PROFILE_SOURCE_FIELDS", "actual_cache_census", "canonical_json_sha256",
    "bind_mpt_eval_context_limit", "dense_cache_bytes", "derive_three_component_profile", "predicted_cache_bytes",
    "rank_schedule_sha256", "run_mpt_strict_layer_search", "select_last_passing_before_first_failure",
    "strict_threshold_grid", "validate_mpt_config", "validate_rank_pair",
    "validate_rank_schedule",
]
