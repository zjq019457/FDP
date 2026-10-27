
import json
from pathlib import Path

OUT_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks")
TIERS = ["light", "medium", "aggressive", "ultra"]
CIDS = ["minus_d1_z", "minus_d2_s", "minus_d3_v", "minus_d4_a", "minus_d5_h"]

# Load baseline F_CURRENT SHAs
BASELINES = {
    "light": "7ecf6eda0d60b4a4d01c7838f0736b47167320cd7640254816cd4101f417ad59",
    "medium": "870786eb6b8e5ca5c196ae0a421e384686a011e80a8be7a95f1fd6b0fd1a6064",
    "aggressive": "9fe83e2b2bf28336bf18a7d8532aa8f033fc2d717e2f4be389b8f7c458f46910",
    "ultra": "58ed4bf50c1632c9922904c41b0d425b2f1fa4a7b61541c207c02b17ac4c7676"
}

print("=== LOO CANDIDATE LOCKS AUDIT ===")
total_evals_needed = 0
for tier in TIERS:
    print(f"\nTier: {tier.upper()}")
    base_sha = BASELINES[tier]
    for cid in CIDS:
        lock_path = OUT_DIR / f"family2_loo_{cid}_{tier}.json"
        with open(lock_path) as f:
            data = json.load(f)
        sha = data["rank_sha"]
        matches_base = (sha == base_sha)
        gap = data["gap_pct"]
        rk = data["matched_rank"]
        print(f"  {cid:12s} | rk={rk} | gap={gap:+.2f}% | matches_base={matches_base} | sha={sha[:12]}...")
        if not matches_base:
            total_evals_needed += 1

print(f"\nTotal candidate states differing from baseline: {total_evals_needed} / 20")
