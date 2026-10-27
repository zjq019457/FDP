
import json
from pathlib import Path

lock_dir = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks")
tiers = ["medium", "aggressive", "ultra", "light"]
rhos = ["rho_100", "rho_75", "rho_50", "rho_25", "rho_00"]

for t in tiers:
    print(f"=== TIER: {t.upper()} ===")
    for r in rhos:
        fn = lock_dir / f"family1_rho_{r}_{t}.json"
        with open(fn) as f:
            d = json.load(f)
        rk, rv = d["ranks"][0]
        sha = d["rank_sha"][:16]
        c = d["c_scale"]
        tot = d["matched_rank"]
        print(f"  {r}: sum={tot}, c={c:.3f}, L0=[{rk:4d}, {rv:4d}], SHA={sha}...")
