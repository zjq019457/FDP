#!/usr/bin/env python3
"""Truncate a precomputed eigen basis to target [[kq,v],...] ranks (exact)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch


def truncate_basis(
    src_pt: Path,
    pairs: List[Tuple[int, int]],
    dst_pt: Path,
    *,
    num_kv_heads: int = 8,
) -> dict:
    obj = torch.load(src_pt, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"unexpected basis type {type(obj)}")
    report = {"layers": [], "src": str(src_pt), "dst": str(dst_pt)}
    out = dict(obj)
    for i, (kq, v) in enumerate(pairs):
        kcols = int(kq) // num_kv_heads
        if int(kq) % num_kv_heads != 0:
            raise ValueError(f"layer {i}: kq={kq} not divisible by {num_kv_heads}")
        if int(v) % num_kv_heads != 0:
            raise ValueError(f"layer {i}: v={v} not divisible by {num_kv_heads}")
        prefix = f"base_model.model.model.layers.{i}.self_attn"
        qk = f"{prefix}.q_proj_basis"
        kk = f"{prefix}.k_proj_basis"
        vk = f"{prefix}.v_proj_basis"
        for key, want in ((qk, kcols), (kk, kcols), (vk, int(v))):
            if key not in out:
                raise KeyError(key)
            t = out[key]
            if not torch.is_tensor(t):
                t = torch.as_tensor(t)
            if t.ndim == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.shape[1] < want:
                raise ValueError(
                    f"{key}: basis width {t.shape[1]} < requested {want}. "
                    "Need a higher-capacity source basis."
                )
            out[key] = t[:, :want].contiguous()
        report["layers"].append({"layer": i, "kq": kq, "v": v, "kcols": kcols})
    dst_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst_pt)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ranks_json", required=True, help="JSON list of [kq,v] or schedule file")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    raw = json.loads(Path(args.ranks_json).read_text())
    if isinstance(raw, dict) and "schedule" in raw:
        pairs = [(int(L["k_proj"]["rank"]), int(L["v_proj"]["rank"])) for L in raw["schedule"]]
    else:
        pairs = [(int(a), int(b)) for a, b in raw]
    rep = truncate_basis(Path(args.src), pairs, Path(args.dst))
    print(json.dumps({"ok": True, **rep}, ensure_ascii=False))


if __name__ == "__main__":
    main()
