import json
import hashlib
from pathlib import Path

# 1. Load FDP-120 report
fdp_rep_path = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb_report.json")
with open(fdp_rep_path) as f:
    fdp_rep = json.load(f)

fdp_ranks = []
for l in fdp_rep["layers"]:
    r = l["runs"][0]
    fdp_ranks.append([int(r["rank_kq_final"]), int(r["rank_v_final"])])

tot_rank_fdp = sum(sum(r) for r in fdp_ranks)
b_live_fdp = tot_rank_fdp * 4096
fdp_sha = hashlib.sha256(json.dumps(fdp_ranks, sort_keys=True).encode('utf-8')).hexdigest()

all_fdp_r = [x for layer in fdp_ranks for x in layer]
floor_count_fdp = sum(1 for x in all_fdp_r if x <= 64)

# Dense baseline KV params: 32 layers * (num_kv=8 * head_dim=128 for K + 8*128 for V) = 32 * 2048 = 65536 total rank
# Dense B_live = 65536 * 4096 = 268,435,456 bytes
dense_rank = 65536
dense_bytes = dense_rank * 4096
kv_retention = b_live_fdp / dense_bytes
kv_compression = 1.0 - kv_retention

ultra_lock = {
    "tier": "ultra",
    "scalar_eb": 0.120,
    "method": "FDP-HIST",
    "provenance": {
        "pass1_source": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass1_uniform/final_model/pass1_meta.json",
        "step1_diagnostic": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb120.json",
        "eb_schedule": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/per_layer_eb_ultra_120.json",
        "operator": "build_eigen_basis_llama_error_budget_offline.py",
        "operator_config": {
            "floor": 0.30,
            "min_rank": 64,
            "threshold_step": 0.02,
            "nsamples": 64,
            "seqlen": 2048,
            "seed": 2,
            "dtype": "bf16"
        },
        "formula_parameters": {
            "lambda": 1.2,
            "rho": 1.0,
            "weights": {"wz": 1.0, "ws": 1.0, "wv": 1.0, "wa": 1.0, "wh": 1.0},
            "normalization": "mean",
            "layer0_override": 0.120
        }
    },
    "metrics": {
        "total_rank": tot_rank_fdp,
        "B_live": b_live_fdp,
        "rank_sha": fdp_sha,
        "dense_rank": dense_rank,
        "dense_bytes": dense_bytes,
        "kv_retention_ratio": kv_retention,
        "kv_compression_ratio": kv_compression,
        "floor_count": floor_count_fdp,
        "total_heads": len(all_fdp_r),
        "floor_percentage": (floor_count_fdp / len(all_fdp_r)) * 100.0
    },
    "ranks": fdp_ranks
}

out_lock_fdp = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/ULTRA_FDP_HIST_LOCK.json")
with open(out_lock_fdp, "w") as f:
    json.dump(ultra_lock, f, indent=2)
print("Saved ULTRA_FDP_HIST_LOCK.json successfully!")

# 2. Parse Uniform Lattice and generate Comparator Lock
uni_rep_path = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/lattice_uniform/eigen_basis_llama3_8b_wikitext2_uniform_report.json")
with open(uni_rep_path) as f:
    uni_rep = json.load(f)

budgets = uni_rep.get("budgets", [0.108, 0.110, 0.112, 0.114, 0.116, 0.118, 0.120])
lattice_entries = []
closest = None
min_gap = float("inf")

for bi, eb in enumerate(budgets):
    uni_ranks = []
    for l in uni_rep["layers"]:
        r = l["runs"][bi]
        uni_ranks.append([int(r["rank_kq_final"]), int(r["rank_v_final"])])
    
    tot_r = sum(sum(r) for r in uni_ranks)
    b_live = tot_r * 4096
    u_sha = hashlib.sha256(json.dumps(uni_ranks, sort_keys=True).encode('utf-8')).hexdigest()
    all_u_r = [x for layer in uni_ranks for x in layer]
    fl_cnt = sum(1 for x in all_u_r if x <= 64)
    
    gap_pct = ((b_live - b_live_fdp) / b_live_fdp) * 100.0
    abs_gap = abs(gap_pct)
    
    entry = {
        "scalar_eb": eb,
        "total_rank": tot_r,
        "B_live": b_live,
        "rank_sha": u_sha,
        "gap_pct": gap_pct,
        "abs_gap_pct": abs_gap,
        "floor_count": fl_cnt,
        "floor_pct": (fl_cnt / len(all_u_r)) * 100.0,
        "ranks": uni_ranks
    }
    lattice_entries.append(entry)
    if abs_gap < min_gap:
        min_gap = abs_gap
        closest = entry

if closest["abs_gap_pct"] <= 0.15:
    match_tier = "STRICT"
elif closest["abs_gap_pct"] <= 1.0:
    match_tier = "NEAR"
else:
    match_tier = "FRONTIER"

comp_lock = {
    "tier": "ultra",
    "comparator_type": "UNIFORM_LATTICE_CLOSEST_BYTES",
    "target_fdp_B_live": b_live_fdp,
    "target_fdp_total_rank": tot_rank_fdp,
    "target_fdp_sha": fdp_sha,
    "closest_comparator": {
        "scalar_eb": closest["scalar_eb"],
        "total_rank": closest["total_rank"],
        "B_live": closest["B_live"],
        "rank_sha": closest["rank_sha"],
        "gap_pct": closest["gap_pct"],
        "abs_gap_pct": closest["abs_gap_pct"],
        "floor_count": closest["floor_count"],
        "floor_pct": closest["floor_pct"],
        "match_tier": match_tier,
        "ranks": closest["ranks"]
    },
    "lattice_table": [
        {
            "scalar_eb": e["scalar_eb"],
            "total_rank": e["total_rank"],
            "B_live": e["B_live"],
            "gap_pct": e["gap_pct"],
            "floor_count": e["floor_count"],
            "rank_sha": e["rank_sha"]
        }
        for e in lattice_entries
    ]
}

out_comp_lock = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/ULTRA_UNIFORM_COMPARATOR_LOCK.json")
with open(out_comp_lock, "w") as f:
    json.dump(comp_lock, f, indent=2)
print("Saved ULTRA_UNIFORM_COMPARATOR_LOCK.json successfully!")
