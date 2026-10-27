
import json, shutil
from pathlib import Path

p = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/formal_ablation/CORE_FORMAL_PER_WINDOW")
with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/formal_ablation/CORE_RESOURCE_AUTHORITY.json") as f:
    core = json.load(f)

core_shas = set()
for r in core:
    core_shas.add(r["natural_rank_sha"])
    core_shas.add(r["matched_rank_sha"])

deleted = 0
kept = 0
for sdir in list(p.iterdir()):
    if not sdir.is_dir(): continue
    sha = sdir.name
    if sha not in core_shas:
        shutil.rmtree(sdir)
        print(f"DELETED polluted dir: {sha[:12]}")
        deleted += 1
    else:
        kept += 1

print(f"Cleanup done: kept {kept} true core formal states, deleted {deleted} polluted dirs.")
