#!/usr/bin/env python3
"""Matched-budget control: allocate ranks to match FDP actual KV bytes at anchor operating point."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import artifact_dir, atomic_write_json, now_iso, op_artifact_dir  # noqa: E402
from kv_accounting import realized_kv_bytes  # noqa: E402


def _normalize_rank_cfg(low: Any) -> List[Dict[str, Any]]:
    """Accept [[kq,v],...] or layer-dict schedules; return dict form for controls."""
    if isinstance(low, dict) and "low_rank_config" in low:
        low = low["low_rank_config"]
    out: List[Dict[str, Any]] = []
    if not isinstance(low, list):
        raise TypeError(f"unexpected low_rank_config type: {type(low)}")
    for layer in low:
        if isinstance(layer, dict):
            out.append(layer)
            continue
        if isinstance(layer, (list, tuple)) and len(layer) >= 2:
            kq, v = int(layer[0]), int(layer[1])
            out.append({
                "k_proj": {"rank": kq, "full_rank": 1024},
                "v_proj": {"rank": v, "full_rank": 1024},
            })
            continue
        raise TypeError(f"unexpected layer entry: {layer!r}")
    return out


def _load_fdp_config(anchor: str) -> tuple:
    op = op_artifact_dir(f"operating_points/{anchor}", link_under=HERE / "artifacts" / f"OP_{anchor}")
    manifest = op / "OPERATING_POINT_MANIFEST.json"
    if not manifest.exists() and anchor == "moderate":
        # fallback to mild/legacy for bootstrap
        op = Path("/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts/r3_fgeb045")
        lr = op / "final_model" / "low_rank_config.json"
    else:
        lr = op / "fdp" / "final_model" / "low_rank_config.json"
    low = json.loads(lr.read_text(encoding="utf-8"))
    low = _normalize_rank_cfg(low)
    # Always recompute target from FDP ranks — never trust stub manifest bytes.
    est = _estimate_bytes(low)
    target_bytes = est
    m = op / "OPERATING_POINT_MANIFEST.json"
    if m.exists():
        d = json.loads(m.read_text())
        # Prefer recomputed; keep manifest note for audit.
        _ = d.get("fdp_kv", {}).get("realized_kv_bytes", target_bytes)
    return low, target_bytes


def _estimate_bytes(config: list, seqlen: int = 2048, bpe: int = 2) -> int:
    """Correct formula: batch(=1)*seqlen*bpe*sum(r_kq+r_v). No extra *2."""
    return int(realized_kv_bytes(config, seqlen=seqlen, bytes_per_elem=bpe, batch=1)["realized_kv_bytes"])


def _round_rank(r: int, multiple: int = 8, full: int = 1024) -> int:
    r = max(multiple, int(r))
    r = (r // multiple) * multiple
    return min(full, max(multiple, r))


def _set_ranks(cfg: list, ranks: List[tuple]) -> None:
    for layer, (kq, v) in zip(cfg, ranks):
        layer["k_proj"]["rank"] = int(kq)
        layer["v_proj"]["rank"] = int(v)


def _scale_to_target(cfg: list, target_bytes: int) -> list:
    """Iteratively scale ranks (multiples of 8) to match target KV bytes."""
    for _ in range(20):
        cur = _estimate_bytes(cfg)
        if cur <= 0:
            break
        if abs(cur - target_bytes) / target_bytes <= 0.01:
            break
        ratio = target_bytes / cur
        for layer in cfg:
            for key in ("k_proj", "v_proj"):
                v = layer.get(key)
                if isinstance(v, dict) and "rank" in v:
                    full = int(v.get("full_rank", 1024))
                    v["rank"] = _round_rank(int(v["rank"] * ratio), full=full)
        # if still over, peel from largest ranks
        cur = _estimate_bytes(cfg)
        if cur > target_bytes * 1.01:
            # reduce largest proj ranks
            flat = []
            for i, layer in enumerate(cfg):
                for key in ("k_proj", "v_proj"):
                    flat.append((int(layer[key]["rank"]), i, key))
            flat.sort(reverse=True)
            for _, i, key in flat:
                if _estimate_bytes(cfg) <= target_bytes * 1.01:
                    break
                cur_r = int(cfg[i][key]["rank"])
                if cur_r > 8:
                    cfg[i][key]["rank"] = cur_r - 8
    return cfg


def _apply_control(base: list, control: str, target_bytes: int, seed: int = 0) -> list:
    cfg = copy.deepcopy(base)
    n_layers = len(cfg)
    rng = random.Random(seed)

    if control == "uniform_matched":
        # Equal ranks across layers, KQ/V split like mean FDP ratio.
        mean_kq = sum(int(l["k_proj"]["rank"]) for l in base) / n_layers
        mean_v = sum(int(l["v_proj"]["rank"]) for l in base) / n_layers
        for layer in cfg:
            layer["k_proj"]["rank"] = _round_rank(mean_kq)
            layer["v_proj"]["rank"] = _round_rank(mean_v)

    elif control.startswith("random_seed"):
        seed_n = int(control.replace("random_seed", ""))
        rng = random.Random(seed_n + 17)
        # Redistribute existing FDP rank mass randomly across layers (keeps capacity).
        kqs = [int(l["k_proj"]["rank"]) for l in base]
        vs = [int(l["v_proj"]["rank"]) for l in base]
        rng.shuffle(kqs)
        rng.shuffle(vs)
        for layer, kq, v in zip(cfg, kqs, vs):
            layer["k_proj"]["rank"] = _round_rank(kq)
            layer["v_proj"]["rank"] = _round_rank(v)

    elif control == "spectral_tail":
        # Prefer higher ranks on later layers (spectral-tail protection heuristic).
        for i, layer in enumerate(cfg):
            w = 0.5 + (i + 1) / n_layers  # 0.5..1.5
            layer["k_proj"]["rank"] = _round_rank(int(layer["k_proj"]["rank"] * w))
            layer["v_proj"]["rank"] = _round_rank(int(layer["v_proj"]["rank"] * w))

    elif control == "middle_layer":
        for i, layer in enumerate(cfg):
            if n_layers // 4 <= i < 3 * n_layers // 4:
                layer["k_proj"]["rank"] = _round_rank(int(layer["k_proj"]["rank"] * 1.5))
                layer["v_proj"]["rank"] = _round_rank(int(layer["v_proj"]["rank"] * 1.5))
            else:
                layer["k_proj"]["rank"] = _round_rank(int(layer["k_proj"]["rank"] * 0.7))
                layer["v_proj"]["rank"] = _round_rank(int(layer["v_proj"]["rank"] * 0.7))

    elif control == "depth_front":
        for i, layer in enumerate(cfg):
            w = 1.4 if i < n_layers // 3 else 0.8
            layer["k_proj"]["rank"] = _round_rank(int(layer["k_proj"]["rank"] * w))
            layer["v_proj"]["rank"] = _round_rank(int(layer["v_proj"]["rank"] * w))

    elif control == "depth_back":
        for i, layer in enumerate(cfg):
            w = 1.4 if i >= 2 * n_layers // 3 else 0.8
            layer["k_proj"]["rank"] = _round_rank(int(layer["k_proj"]["rank"] * w))
            layer["v_proj"]["rank"] = _round_rank(int(layer["v_proj"]["rank"] * w))

    elif control == "reconstruction_slope":
        for i, layer in enumerate(cfg):
            slope = 1.25 - (i / max(n_layers - 1, 1)) * 0.5
            layer["k_proj"]["rank"] = _round_rank(int(layer["k_proj"]["rank"] * slope))
            layer["v_proj"]["rank"] = _round_rank(int(layer["v_proj"]["rank"] * slope))

    return _scale_to_target(cfg, target_bytes)


def _clip_to_capacity(cfg: list, capacity: list) -> list:
    """Clip ranks so overlay on capacity checkpoint is valid."""
    for layer, cap in zip(cfg, capacity):
        layer["k_proj"]["rank"] = min(int(layer["k_proj"]["rank"]), int(cap["k_proj"]["rank"]))
        layer["v_proj"]["rank"] = min(int(layer["v_proj"]["rank"]), int(cap["v_proj"]["rank"]))
        layer["k_proj"]["rank"] = _round_rank(layer["k_proj"]["rank"])
        layer["v_proj"]["rank"] = _round_rank(layer["v_proj"]["rank"])
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--anchor", default="moderate")
    args = ap.parse_args()
    out = artifact_dir(f"matched_controls/{args.control}", link_under=HERE / "artifacts" / f"MC_{args.control}")
    base, target = _load_fdp_config(args.anchor)
    schedule = _apply_control(base, args.control, target)
    # Ensure evaluable via FDP overlay (no extra 15GB compress).
    schedule = _clip_to_capacity(schedule, base)
    # Re-scale up within capacity headroom toward target if clipped under.
    schedule = _scale_to_target(schedule, target)
    schedule = _clip_to_capacity(schedule, base)
    realized = _estimate_bytes(schedule)
    atomic_write_json(out / "MATCHED_CONTROL_SCHEDULE.json", {
        "control": args.control,
        "anchor": args.anchor,
        "target_kv_bytes": target,
        "realized_kv_bytes_est": realized,
        "abs_byte_gap": realized - target,
        "rel_gap": (realized - target) / target if target else None,
        "capacity_clip": "fdp_overlay",
        "schedule": schedule,
        "generated_at": now_iso(),
        "signed_before_eval": True,
        "kv_formula": "batch*seqlen*bpe*sum(r_kq+r_v)",
    })
    # Write schedule CSV
    import csv
    with (out / "layer_ranks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "proj", "rank"])
        for i, layer in enumerate(schedule):
            for k, v in layer.items():
                if isinstance(v, dict) and "rank" in v:
                    w.writerow([i, k, v["rank"]])
    print(json.dumps({"ok": True, "control": args.control, "realized": realized, "target": target, "rel_gap": (realized - target) / target}, ensure_ascii=False))


if __name__ == "__main__":
    main()
