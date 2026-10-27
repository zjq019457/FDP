
import json
import hashlib
from pathlib import Path

rep_path = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/lattice_fine/eigen_basis_llama3_8b_wikitext2_uniform_fine_report.json")
with open(rep_path) as f:
    rep = json.load(f)

target_fdp_b_live = 124223488
target_fdp_rank = 30328

budgets = [0.1125, 0.1130, 0.1135]
results = {}

for bi, eb in enumerate(budgets):
    ranks = []
    floor_count = 0
    for li, layer in enumerate(rep["layers"]):
        run = layer["runs"][bi]
        rk = run["rank_kq_final"]
        rv = run["rank_v_final"]
        ranks.append([rk, rv])
        if rk == 64 or rv == 64:
            floor_count += 1
            
    tot_rank = sum(sum(r) for r in ranks)
    b_live = tot_rank * 4096
    gap_pct = ((b_live - target_fdp_b_live) / target_fdp_b_live) * 100.0
    r_sha = hashlib.sha256(json.dumps(ranks, separators=(',', ':')).encode('utf-8')).hexdigest()
    
    results[eb] = {
        "scalar_eb": eb,
        "total_rank": tot_rank,
        "B_live": b_live,
        "gap_pct": gap_pct,
        "abs_gap_pct": abs(gap_pct),
        "floor_count": floor_count,
        "floor_pct": (floor_count / 64.0) * 100.0,
        "rank_sha": r_sha,
        "ranks": ranks
    }
    print(f"Fine eb={eb:.4f}: Rank={tot_rank}, B_live={b_live}, Gap={gap_pct:+.4f}%, FloorCount={floor_count} ({floor_count/64*100:.1f}%), SHA={r_sha[:16]}...")

# Compare against coarse best eb=0.112 (+1.5036%)
with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/ULTRA_UNIFORM_COMPARATOR_LOCK.json") as f:
    coarse_lock = json.load(f)

coarse_best = coarse_lock["closest_comparator"]
print(f"Coarse best eb={coarse_best['scalar_eb']}: Rank={coarse_best['total_rank']}, B_live={coarse_best['B_live']}, Gap={coarse_best['gap_pct']:+.4f}%")

# Find overall closest across coarse + fine
all_candidates = list(results.values()) + [coarse_best]
best_cand = min(all_candidates, key=lambda x: abs(x["gap_pct"]))
print(f"\nOVERALL CLOSEST UNIFORM COMPARATOR: eb={best_cand['scalar_eb']}, Gap={best_cand['gap_pct']:+.4f}%, B_live={best_cand['B_live']}")

# Classify match tier
abs_gap = abs(best_cand["gap_pct"])
if abs_gap <= 0.5:
    m_tier = "STRICT_MATCH (<=0.5%)"
elif abs_gap <= 1.0:
    m_tier = "CLOSE_MATCH (<=1.0%)"
elif abs_gap <= 2.0:
    m_tier = "FRONTIER (<=2.0%)"
else:
    m_tier = "COARSE"

final_lock = {
    "tier": "ultra",
    "comparator_type": "UNIFORM_FINE_LATTICE_OPTIMAL",
    "target_fdp_B_live": target_fdp_b_live,
    "target_fdp_total_rank": target_fdp_rank,
    "target_fdp_sha": "6c5f695f4256c9c01504137b8f6ace79afd625884d8f86a07ebef2e41e1e5239",
    "status": "FINAL_DONE_VALID",
    "closest_comparator": {
        "scalar_eb": best_cand["scalar_eb"],
        "total_rank": best_cand["total_rank"],
        "B_live": best_cand["B_live"],
        "rank_sha": best_cand["rank_sha"],
        "gap_pct": best_cand["gap_pct"],
        "abs_gap_pct": abs_gap,
        "floor_count": best_cand["floor_count"],
        "floor_pct": best_cand["floor_pct"],
        "match_tier": m_tier,
        "ranks": best_cand["ranks"]
    },
    "fine_lattice_table": list(results.values()),
    "coarse_lattice_table": coarse_lock.get("lattice_table", [])
}

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/authority/ULTRA_UNIFORM_COMPARATOR_LOCK.json", "w") as f:
    json.dump(final_lock, f, indent=2)
with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/ULTRA_UNIFORM_COMPARATOR_LOCK.json", "w") as f:
    json.dump(final_lock, f, indent=2)

print("Saved authoritative ULTRA_UNIFORM_COMPARATOR_LOCK.json with status FINAL_DONE_VALID!")
