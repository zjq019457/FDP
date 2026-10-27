#!/usr/bin/env python3
import json
with open('/diskF/jiaqizhang/fdp_clean_ablation_v1/formal_ablation/CORE_RESOURCE_AUTHORITY.json') as f:
    d = json.load(f)
print("Keys in row 0:", list(d[0].keys()))
for k, v in d[0].items():
    if not isinstance(v, (list, dict)):
        print(f"  {k}: {v}")
