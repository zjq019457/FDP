#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
from formula_engine import TIER_CONF, load_traces, match_ranks, compute_rank_sha

OUT_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks")

# Preload traces and step1 profiles
TIERS_DATA = {}
for tier_name, conf in TIER_CONF.items():
    traces = load_traces(conf["report"])
    with open(conf["step1"]) as f:
        step1 = json.load(f)
    TIERS_DATA[tier_name] = {"traces": traces, "step1": step1, "conf": conf}

low_weights = [
    {"cid": "half_d1_z", "desc": "Half d1 (wz=0.5)", "rho": 1.0, "lambda": 1.2, "weights": (0.5, 1.0, 1.0, 1.0, 1.0)},
    {"cid": "half_d2_s", "desc": "Half d2 (ws=0.5)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 0.5, 1.0, 1.0, 1.0)},
    {"cid": "half_d3_v", "desc": "Half d3 (wv=0.5)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 1.0, 0.5, 1.0, 1.0)},
    {"cid": "half_d4_a", "desc": "Half d4 (wa=0.5)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 1.0, 1.0, 0.5, 1.0)},
    {"cid": "half_d5_h", "desc": "Half d5 (wh=0.5)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 1.0, 1.0, 1.0, 0.5)},
]

manifest_path = OUT_DIR / "MASTER_RESOURCE_LOCKS_MANIFEST.json"
with open(manifest_path) as mf:
    manifest = json.load(mf)

manifest["family4_low_weights"] = []

print("=== Generating Family 4 Low Weights (0.5x) Locks (20 locks) ===")
for c in low_weights:
    cand_entry = {"cid": c["cid"], "desc": c["desc"], "tiers": {}}
    for tier_name in ["light", "medium", "aggressive", "ultra"]:
        tdata = TIERS_DATA[tier_name]
        conf = tdata["conf"]
        ranks, best_c, diff = match_ranks(
            step1_data=tdata["step1"],
            traces=tdata["traces"],
            eb_base=conf["eb_base"],
            target_rank=conf["target_rank"],
            weights=c["weights"],
            lambda_eb=c["lambda"],
            rho=c["rho"]
        )
        tot_r = sum(sum(r) for r in ranks)
        b_live = tot_r * 4096
        gap_pct = ((b_live - conf["target_b_live"]) / conf["target_b_live"]) * 100.0
        r_sha = compute_rank_sha(ranks)
        floor_count = sum(1 for r in ranks if r[0] == 64 or r[1] == 64)
        
        lock_data = {
            "candidate_id": c["cid"],
            "family": "family4_low_weights",
            "tier": tier_name,
            "description": c["desc"],
            "rho": c["rho"],
            "lambda": c["lambda"],
            "weights": list(c["weights"]),
            "c_scale": best_c,
            "target_rank": conf["target_rank"],
            "matched_rank": tot_r,
            "target_b_live": conf["target_b_live"],
            "matched_b_live": b_live,
            "gap_pct": gap_pct,
            "rank_sha": r_sha,
            "floor_count": floor_count,
            "ranks": ranks
        }
        lock_path = OUT_DIR / f"family4_low_{c['cid']}_{tier_name}.json"
        with open(lock_path, "w") as lf:
            json.dump(lock_data, lf, indent=2)
            
        cand_entry["tiers"][tier_name] = {
            "matched_rank": tot_r,
            "b_live": b_live,
            "gap_pct": round(gap_pct, 4),
            "rank_sha": r_sha,
            "c_scale": round(best_c, 4),
            "floor_count": floor_count,
            "lock_file": lock_path.name
        }
    print(f"  {c['cid']:15s} | M: {cand_entry['tiers']['medium']['gap_pct']:+.2f}% | A: {cand_entry['tiers']['aggressive']['gap_pct']:+.2f}% | U: {cand_entry['tiers']['ultra']['gap_pct']:+.2f}% | L: {cand_entry['tiers']['light']['gap_pct']:+.2f}%")
    manifest["family4_low_weights"].append(cand_entry)

with open(manifest_path, "w") as mf:
    json.dump(manifest, mf, indent=2)

print("Successfully generated all 20 low weight locks! Total weight locks now: 40/40.")
