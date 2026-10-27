#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
from formula_engine import TIER_CONF, load_traces, match_ranks, compute_rank_sha

OUT_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Preload traces and step1 profiles for all 4 tiers
TIERS_DATA = {}
for tier_name, conf in TIER_CONF.items():
    print(f"Loading {tier_name.upper()} traces and profile...")
    traces = load_traces(conf["report"])
    with open(conf["step1"]) as f:
        step1 = json.load(f)
    TIERS_DATA[tier_name] = {
        "traces": traces,
        "step1": step1,
        "conf": conf
    }

families = {}

# Family 1: Rho
families["family1_rho"] = []
for rho in [0.00, 0.25, 0.50, 0.75, 1.00]:
    families["family1_rho"].append({
        "cid": f"rho_{int(rho*100):02d}",
        "desc": f"Layer-0 rho={rho:.2f}",
        "rho": rho,
        "lambda": 1.2,
        "weights": (1.0, 1.0, 1.0, 1.0, 1.0)
    })

# Family 2: Component LOO
families["family2_loo"] = [
    {"cid": "minus_d1_z", "desc": "Drop d1 (delta_z)", "rho": 1.0, "lambda": 1.2, "weights": (0.0, 1.0, 1.0, 1.0, 1.0)},
    {"cid": "minus_d2_s", "desc": "Drop d2 (delta_score)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 0.0, 1.0, 1.0, 1.0)},
    {"cid": "minus_d3_v", "desc": "Drop d3 (delta_v)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 1.0, 0.0, 1.0, 1.0)},
    {"cid": "minus_d4_a", "desc": "Drop d4 (attn_entropy)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 1.0, 1.0, 0.0, 1.0)},
    {"cid": "minus_d5_h", "desc": "Drop d5 (entropy_collapse)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 1.0, 1.0, 1.0, 0.0)},
]

# Family 3: Lambda
families["family3_lambda"] = []
for lam in [0.6, 0.9, 1.2, 1.5, 1.8]:
    families["family3_lambda"].append({
        "cid": f"lambda_{str(lam).replace('.', 'p')}",
        "desc": f"Coupling lambda={lam:.1f}",
        "rho": 1.0,
        "lambda": lam,
        "weights": (1.0, 1.0, 1.0, 1.0, 1.0)
    })

# Family 4: Weights
families["family4_weights"] = [
    {"cid": "double_d1_z", "desc": "Double d1 (wz=2.0)", "rho": 1.0, "lambda": 1.2, "weights": (2.0, 1.0, 1.0, 1.0, 1.0)},
    {"cid": "double_d2_s", "desc": "Double d2 (ws=2.0)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 2.0, 1.0, 1.0, 1.0)},
    {"cid": "double_d3_v", "desc": "Double d3 (wv=2.0)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 1.0, 2.0, 1.0, 1.0)},
    {"cid": "double_d4_a", "desc": "Double d4 (wa=2.0)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 1.0, 1.0, 2.0, 1.0)},
    {"cid": "double_d5_h", "desc": "Double d5 (wh=2.0)", "rho": 1.0, "lambda": 1.2, "weights": (1.0, 1.0, 1.0, 1.0, 2.0)},
]

all_locks_summary = {}

for fam_name, cand_list in families.items():
    print(f"\n=== Processing {fam_name} ({len(cand_list)} candidates x 4 tiers = {len(cand_list)*4} locks) ===")
    all_locks_summary[fam_name] = []
    
    for c in cand_list:
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
            
            # Floor saturation count (ranks where rank == 64)
            floor_count = sum(1 for r in ranks if r[0] == 64 or r[1] == 64)
            
            lock_data = {
                "candidate_id": c["cid"],
                "family": fam_name,
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
            
            lock_path = OUT_DIR / f"{fam_name}_{c['cid']}_{tier_name}.json"
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
        all_locks_summary[fam_name].append(cand_entry)

with open(OUT_DIR / "MASTER_RESOURCE_LOCKS_MANIFEST.json", "w") as mf:
    json.dump(all_locks_summary, mf, indent=2)

print(f"\nAll resource locks generated in {OUT_DIR}!")
