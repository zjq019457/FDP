#!/usr/bin/env python3
import json
from pathlib import Path

# Load audit results
with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks/LOO_RHO75_AUDIT.json") as f:
    loo_audit = json.load(f)

OUT_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks")

updated_count = 0
reused_count = 0

for tier, entries in loo_audit.items():
    for item in entries:
        cid = item["cid"]
        lock_path = OUT_DIR / f"family2_loo_{cid}_{tier}.json"
        
        with open(lock_path) as lf:
            lock_data = json.load(lf)
            
        # Update rho field to 0.75
        lock_data["rho"] = 0.75
        
        if not item["is_identical"]:
            # Update affected lock with new matched rank, sha, b_live, c_scale, ranks
            lock_data["matched_rank"] = item["new_matched_rank"]
            lock_data["matched_b_live"] = item["new_b_live"]
            lock_data["gap_pct"] = item["new_gap_pct"]
            lock_data["c_scale"] = item["new_c_scale"]
            lock_data["rank_sha"] = item["new_sha"]
            lock_data["ranks"] = item["new_ranks"]
            lock_data["floor_count"] = sum(1 for r in item["new_ranks"] if r[0] == 64 or r[1] == 64)
            updated_count += 1
            print(f"UPDATED affected lock: family2_loo_{cid}_{tier}.json (new SHA: {item['new_sha'][:12]}...)")
        else:
            reused_count += 1
            print(f"REUSED bit-identical lock: family2_loo_{cid}_{tier}.json (SHA: {item['old_sha'][:12]}...)")
            
        with open(lock_path, "w") as lf:
            json.dump(lock_data, lf, indent=2)

print(f"\nResource locks summary: {updated_count} updated, {reused_count} bit-identical reused.")

# Also update MASTER_RESOURCE_LOCKS_MANIFEST.json
manifest_path = OUT_DIR / "MASTER_RESOURCE_LOCKS_MANIFEST.json"
with open(manifest_path) as mf:
    manifest = json.load(mf)

for cand_entry in manifest.get("family2_loo", []):
    cid = cand_entry["cid"]
    for tier in ["light", "medium", "aggressive", "ultra"]:
        lock_path = OUT_DIR / f"family2_loo_{cid}_{tier}.json"
        with open(lock_path) as lf:
            current_lock = json.load(lf)
        cand_entry["tiers"][tier]["matched_rank"] = current_lock["matched_rank"]
        cand_entry["tiers"][tier]["b_live"] = current_lock["matched_b_live"]
        cand_entry["tiers"][tier]["gap_pct"] = round(current_lock["gap_pct"], 4)
        cand_entry["tiers"][tier]["rank_sha"] = current_lock["rank_sha"]
        cand_entry["tiers"][tier]["c_scale"] = current_lock["c_scale"]
        cand_entry["tiers"][tier]["floor_count"] = current_lock["floor_count"]

with open(manifest_path, "w") as mf:
    json.dump(manifest, mf, indent=2)

print("Updated MASTER_RESOURCE_LOCKS_MANIFEST.json successfully.")

# Save authority freeze file for DEV_RHO_SELECTED
authority_rho = {
    "promoted_parameter": "DEV_RHO_SELECTED",
    "selected_value": 0.75,
    "status": "DEV_PROMOTED_FORMULA",
    "historical_fallback": "F0_HISTORICAL_FDP (rho=1.00)",
    "light_guard": {
        "status": "PASSED",
        "delta_j": 0.0,
        "threshold": 0.003
    },
    "j_strong": {
        "rho_100_baseline": 3.3419083983947835,
        "rho_075_selected": 3.3396348521734276,
        "delta_j_strong": -0.0022735462213558932
    },
    "note": "Dev selection only. Not final method. Formal statistical claims deferred to formal test on frozen candidate."
}

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/authority/DEV_RHO_SELECTED_LOCK.json", "w") as af:
    json.dump(authority_rho, af, indent=2)

print("Saved authority/DEV_RHO_SELECTED_LOCK.json.")
