#!/usr/bin/env python3
import csv
import json
from pathlib import Path

csv_path = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/authority/FDP_ABLATION_CAMPAIGN_STATE.csv")
manifest_path = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks/MASTER_RESOURCE_LOCKS_MANIFEST.json")

with open(manifest_path) as f:
    manifest = json.load(f)

# Read existing CSV rows
rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        rows.append(r)

# Map candidate entries to rows
lock_lookup = {}
for fam, cands in manifest.items():
    for c in cands:
        cid = c["cid"]
        for tier, tdata in c["tiers"].items():
            key = (cid, tier)
            lock_lookup[key] = tdata

updated_count = 0
for r in rows:
    var = r["variant"].lower()
    tier = r["tier"].lower()
    
    # match candidate
    matched_data = None
    for k, v in lock_lookup.items():
        if k[1] == tier and (k[0] in var or var.endswith(k[0])):
            matched_data = v
            break
            
    if matched_data:
        r["B_live"] = str(matched_data["b_live"])
        r["rank_SHA"] = matched_data["rank_sha"]
        r["gap_vs_full"] = f"{matched_data['gap_pct']:+.4f}"
        if r["status"] == "NOT_STARTED":
            r["status"] = "RESOURCE_LOCKED"
        updated_count += 1

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {updated_count} rows in {csv_path} with authoritative resource locks!")
