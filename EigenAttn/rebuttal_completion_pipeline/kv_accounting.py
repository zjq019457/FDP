#!/usr/bin/env python3
"""Canonical KV-byte accounting for Llama-3-8B EigenAttn ranks.

Rank entries are absolute stored dims (already aggregated across KV heads).
Do NOT multiply again by num_kv_heads * head_dim.

realized = batch * seqlen * bytes_per_elem * sum_l (r_kq_l + r_v_l)
dense    = n_layers * 2 * n_kv_heads * head_dim * seqlen * bytes_per_elem
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

LLAMA3_8B = {
    "n_layers": 32,
    "n_kv_heads": 8,
    "n_heads": 32,
    "head_dim": 128,
    "hidden_size": 4096,
    "full_rank_per_proj": 1024,  # n_kv_heads * head_dim
}


def normalize_low_rank(low: Any) -> List[Tuple[int, int]]:
    if isinstance(low, dict) and "low_rank_config" in low:
        low = low["low_rank_config"]
    out: List[Tuple[int, int]] = []
    if not isinstance(low, list):
        raise TypeError(f"unexpected low_rank_config type: {type(low)}")
    for layer in low:
        if isinstance(layer, (list, tuple)) and len(layer) >= 2:
            out.append((int(layer[0]), int(layer[1])))
        elif isinstance(layer, dict):
            kq = layer.get("k_proj") or layer.get("kq") or {}
            v = layer.get("v_proj") or layer.get("v") or {}
            if isinstance(kq, dict):
                kq = kq.get("rank", 0)
            if isinstance(v, dict):
                v = v.get("rank", 0)
            out.append((int(kq), int(v)))
        else:
            raise TypeError(f"unexpected layer entry: {layer!r}")
    return out


def load_low_rank(path: Union[str, Path]) -> List[Tuple[int, int]]:
    path = Path(path)
    if path.suffix == ".pt":
        import torch

        raw = torch.load(path, map_location="cpu")
        return normalize_low_rank(raw.tolist() if hasattr(raw, "tolist") else raw)
    return normalize_low_rank(json.loads(path.read_text(encoding="utf-8")))


def dense_kv_bytes(*, seqlen: int = 2048, bytes_per_elem: int = 2, batch: int = 1) -> int:
    return (
        batch
        * LLAMA3_8B["n_layers"]
        * 2
        * LLAMA3_8B["n_kv_heads"]
        * LLAMA3_8B["head_dim"]
        * seqlen
        * bytes_per_elem
    )


def realized_kv_bytes(
    low: Any,
    *,
    seqlen: int = 2048,
    bytes_per_elem: int = 2,
    batch: int = 1,
) -> Dict[str, Any]:
    ranks = normalize_low_rank(low)
    rank_sum = sum(kq + v for kq, v in ranks)
    realized = batch * seqlen * bytes_per_elem * rank_sum
    dense = dense_kv_bytes(seqlen=seqlen, bytes_per_elem=bytes_per_elem, batch=batch)
    return {
        "batch": batch,
        "seqlen": seqlen,
        "bytes_per_elem": bytes_per_elem,
        "n_layers": len(ranks),
        "total_rank_sum": rank_sum,
        "realized_kv_bytes": int(realized),
        "dense_kv_bytes": int(dense),
        "retained_ratio": (realized / dense) if dense else None,
        "formula": "batch*seqlen*bytes_per_elem*sum_l(r_kq+r_v)",
        "ranks_sha256": hashlib.sha256(
            json.dumps([[kq, v] for kq, v in ranks], separators=(",", ":")).encode()
        ).hexdigest(),
    }


def audit_from_path(path: Union[str, Path], **kwargs: Any) -> Dict[str, Any]:
    path = Path(path)
    ranks = load_low_rank(path)
    out = realized_kv_bytes(ranks, **kwargs)
    out["source"] = str(path)
    return out


def gb_from_bytes(n: int) -> float:
    return n / (1024**3)


def compare_to_runtime_cache_gb(realized_bytes: int, runtime_cache_gb: float, tol: float = 1e-6) -> Dict[str, Any]:
    expected_gb = gb_from_bytes(realized_bytes)
    return {
        "static_gb": expected_gb,
        "runtime_cache_gb": runtime_cache_gb,
        "abs_diff_gb": abs(expected_gb - runtime_cache_gb),
        "match": abs(expected_gb - runtime_cache_gb) <= tol,
    }
