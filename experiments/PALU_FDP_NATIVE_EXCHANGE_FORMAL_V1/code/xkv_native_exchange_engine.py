"""Pure schedule/resource engine for XKV_FDP_NATIVE_EXCHANGE_FORMAL_V1.

This module performs no model evaluation.  It is deliberately shared by the
q-prior, native-phi, and random-signal arms; only the input ranking signal can
vary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

GROUP_COUNT = 8
GROUP_SIZE = 4
CONTROL_K = 384
CONTROL_V = 576
DELTA_K = 32
DELTA_V = 48
FACTOR_SEQ_LEN = 65536
BITS = 16
KV_HEADS = 8
HEAD_DIM = 128
BATCH = 1


@dataclass(frozen=True)
class NativeSchedule:
    signal_name: str
    seed: int | None
    T: int
    donors: tuple[int, ...]
    receivers: tuple[int, ...]
    rank_k: tuple[int, ...]
    rank_v: tuple[int, ...]


def group_factor_bytes(rank_k: int, rank_v: int) -> int:
    """Official xKV grouped factor formula for one W=4 group."""
    elements = (
        BATCH * FACTOR_SEQ_LEN * (rank_k + rank_v)
        + GROUP_SIZE * BATCH * KV_HEADS * HEAD_DIM * (rank_k + rank_v)
    )
    return elements * (BITS // 8)


def total_factor_bytes(rank_k: Sequence[int], rank_v: Sequence[int]) -> int:
    if len(rank_k) != GROUP_COUNT or len(rank_v) != GROUP_COUNT:
        raise ValueError("xKV native schedules must contain 8 groups")
    return sum(group_factor_bytes(k, v) for k, v in zip(rank_k, rank_v))


def order_signal(signal: Sequence[float]) -> list[int]:
    if len(signal) != GROUP_COUNT:
        raise ValueError("signal must have exactly 8 group values")
    # Stable deterministic tie break by native group id.
    return sorted(range(GROUP_COUNT), key=lambda g: (float(signal[g]), g))


def candidate_pairs(signal: Sequence[float]) -> list[tuple[int, int]]:
    order = order_signal(signal)
    low = order[:4]
    high = list(reversed(order[-4:]))
    return list(zip(low, high))


def schedule_for_signal(signal_name: str, signal: Sequence[float], T: int, seed: int | None = None) -> NativeSchedule:
    if T not in (0, 1, 2, 3, 4):
        raise ValueError("T must be one of 0,1,2,3,4")
    pairs = candidate_pairs(signal)
    selected = pairs[:T]
    donors = tuple(d for d, _ in selected)
    receivers = tuple(r for _, r in selected)
    rk = [CONTROL_K] * GROUP_COUNT
    rv = [CONTROL_V] * GROUP_COUNT
    for donor, receiver in selected:
        rk[donor] -= DELTA_K
        rv[donor] -= DELTA_V
        rk[receiver] += DELTA_K
        rv[receiver] += DELTA_V
    if total_factor_bytes(rk, rv) != total_factor_bytes([CONTROL_K] * 8, [CONTROL_V] * 8):
        raise ValueError("RESOURCE_PROJECTION_FAILED: factor bytes are not exact")
    return NativeSchedule(signal_name, seed, T, donors, receivers, tuple(rk), tuple(rv))


def nested_schedules(signal_name: str, signal: Sequence[float], seed: int | None = None) -> list[NativeSchedule]:
    return [schedule_for_signal(signal_name, signal, T, seed=seed) for T in range(5)]


def random_signal(seed: int) -> list[float]:
    import random
    if seed not in (0, 1, 2):
        raise ValueError("formal random ablation seeds are 0,1,2")
    rng = random.Random(seed)
    values = list(range(GROUP_COUNT))
    rng.shuffle(values)
    # A permutation signal makes the ranking explicit and reproducible.
    score = [0.0] * GROUP_COUNT
    for rank, group in enumerate(values):
        score[group] = float(rank)
    return score


def audit_all(signal_map: dict[str, Sequence[float]]) -> list[dict]:
    out = []
    for name, signal in signal_map.items():
        seed = int(name.rsplit("_", 1)[-1]) if name.startswith("random_seed_") else None
        for schedule in nested_schedules(name, signal, seed=seed):
            out.append({
                "signal": schedule.signal_name,
                "seed": schedule.seed,
                "T": schedule.T,
                "donors": list(schedule.donors),
                "receivers": list(schedule.receivers),
                "rank_k": list(schedule.rank_k),
                "rank_v": list(schedule.rank_v),
                "total_native_rank": sum(schedule.rank_k) + sum(schedule.rank_v),
                "factor_bytes": total_factor_bytes(schedule.rank_k, schedule.rank_v),
                "resource_match_exact": total_factor_bytes(schedule.rank_k, schedule.rank_v) == total_factor_bytes([CONTROL_K] * 8, [CONTROL_V] * 8),
            })
    return out
