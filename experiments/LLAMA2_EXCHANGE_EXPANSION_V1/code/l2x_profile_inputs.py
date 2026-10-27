#!/usr/bin/env python3
"""Adapter inputs for the frozen Llama-2 d1/d2/d5 profile runner (llama2_clean_redesign_profile_uniform068_v5.py, rank-CSV mode):
writes a per-layer rank CSV (legacy column names) and a resource manifest binding it, for a Uniform operating point. The profile
computation itself is unchanged. usage: l2x_profile_inputs.py OP"""
import csv, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l2x_lib as X

op = sys.argv[1]
pairs = X.uniform_pairs(op)
d = X.ROOT / "profiles" / op / "inputs"; d.mkdir(parents=True, exist_ok=True)
c = d / f"UNIFORM_{op}_PER_LAYER.csv"
with open(c, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["layer", "uniform068_QK_rank", "uniform068_V_rank"])
    for i, (k, v) in enumerate(pairs): w.writerow([i, k, v])
m = d / f"UNIFORM_{op}_RESOURCE_MANIFEST.json"
X.wjson(m, {"note": f"adapter: legacy column/field names 'uniform068_*' hold the {op} Uniform schedule", "op": op,
            "outputs": {"per_layer_csv": {"path": str(c), "sha256": X.sha_file(c)}},
            "resource_accounting": {"uniform068_B_live_rank_implied_bytes": X.B_of(pairs)},
            "schedule_sha256": {"uniform068_rank_schedule": X.sha_json(pairs)}})
print(c, m, X.B_of(pairs))
