import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
from formula_engine import TIER_CONF, load_traces, match_ranks, compute_rank_sha, derive_eb_schedule

# Load evaluation results
with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection/RHO_DEV_EVAL_PROGRESS.json") as f:
    eval_results = json.load(f)

TIERS = ["light", "medium", "aggressive", "ultra"]
RHOS = [1.00, 0.75, 0.50, 0.25, 0.00]

tier_data = {}
for tier in TIERS:
    conf = TIER_CONF[tier]
    traces = load_traces(conf["report"])
    with open(conf["step1"]) as f:
        step1 = json.load(f)
    tier_data[tier] = {
        "conf": conf,
        "traces": traces,
        "step1": step1
    }

# 1. Gather all data per tier and per rho
full_report = {}

for tier in TIERS:
    conf = tier_data[tier]["conf"]
    traces = tier_data[tier]["traces"]
    step1 = tier_data[tier]["step1"]
    eb_base = conf["eb_base"]
    
    weights = (1.0, 1.0, 1.0, 1.0, 1.0)
    
    tier_report = []
    for rho in RHOS:
        cid = f"rho_{int(rho*100):02d}"
        lock_file = Path(f"/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks/family1_rho_{cid}_{tier}.json")
        with open(lock_file) as f:
            lock = json.load(f)
            
        c_scale = lock["c_scale"]
        
        # Layer 0 EB before rho interpolation: rho=0.0 schedule at L0
        eb_sched_unblended = derive_eb_schedule(step1, eb_base, weights, 1.2, rho=0.0, c_scale=c_scale)
        eb_0_before = eb_sched_unblended[0]
        
        # Layer 0 EB after rho interpolation:
        eb_sched_blended = derive_eb_schedule(step1, eb_base, weights, 1.2, rho=rho, c_scale=c_scale)
        eb_0_after = eb_sched_blended[0]
        
        rho_key = str(rho)
        metrics = eval_results[tier][rho_key]
        
        tier_report.append({
            "rho": rho,
            "cid": cid,
            "eb_0_before": eb_0_before,
            "eb_0_after": eb_0_after,
            "l0_kq_rank": lock["ranks"][0][0],
            "l0_v_rank": lock["ranks"][0][1],
            "rank_sha": lock["rank_sha"],
            "total_rank": lock["matched_rank"],
            "b_live": lock["matched_b_live"],
            "gap_pct": lock["gap_pct"],
            "dev_wiki_nll": metrics["dev_wiki_nll"],
            "dev_wiki_ppl": metrics["dev_wiki_ppl"],
            "dev_c4_nll": metrics["dev_c4_nll"],
            "dev_c4_ppl": metrics["dev_c4_ppl"],
            "j_dev": metrics["J_dev"],
            "delta_j": metrics["delta_J"],
            "ci_low": metrics["ci_low"],
            "ci_high": metrics["ci_high"],
            "p_val": metrics["p_value"]
        })
    full_report[tier] = tier_report

# 2. Deduplicate unique physical rank states per tier
unique_states_by_tier = {}
for tier in TIERS:
    seen = {}
    for entry in full_report[tier]:
        sha = entry["rank_sha"]
        if sha not in seen:
            seen[sha] = {
                "rhos": [entry["rho"]],
                "entry": entry
            }
        else:
            seen[sha]["rhos"].append(entry["rho"])
    unique_states_by_tier[tier] = seen

print("================================================================================")
print("SECTION 4: UNIQUE PHYSICAL RHO STATES PER TIER (DEDUPLICATED)")
print("================================================================================")

for tier in TIERS:
    print(f"\n=================== TIER: {tier.upper()} ===================")
    for sha, u in unique_states_by_tier[tier].items():
        e = u["entry"]
        rhos_str = ", ".join([f"{r:.2f}" for r in u["rhos"]])
        print(f"Bound rho values: [{rhos_str}]")
        print(f"  Layer0 EB before interpolation: {e['eb_0_before']:.6f}")
        print(f"  Layer0 EB after interpolation:  {e['eb_0_after']:.6f}")
        print(f"  Layer0 KQ rank: {e['l0_kq_rank']} | Layer0 V rank: {e['l0_v_rank']}")
        print(f"  Full Rank SHA:  {e['rank_sha']}")
        print(f"  Total Rank:     {e['total_rank']} | B_live: {e['b_live']} bytes (gap: {e['gap_pct']:+.4f}%)")
        print(f"  DEV Wiki NLL:   {e['dev_wiki_nll']:.4f} (PPL: {e['dev_wiki_ppl']:.2f})")
        print(f"  DEV C4 NLL:     {e['dev_c4_nll']:.4f} (PPL: {e['dev_c4_ppl']:.2f})")
        print(f"  J_{tier[0].upper()}(rho):         {e['j_dev']:.6f} (Delta vs rho=1.00: {e['delta_j']:+.6f})")

# 3. J_STRONG Synthesis
print("\n================================================================================")
print("SECTION 5: J_STRONG SYNTHESIS & LIGHT GUARD")
print("================================================================================")
j_strong_table = []
for rho in RHOS:
    rho_key = str(rho)
    jm = eval_results["medium"][rho_key]["J_dev"]
    ja = eval_results["aggressive"][rho_key]["J_dev"]
    ju = eval_results["ultra"][rho_key]["J_dev"]
    jl = eval_results["light"][rho_key]["J_dev"]
    
    j_strong = (jm + ja + ju) / 3.0
    j_strong_table.append({
        "rho": rho,
        "j_light": jl,
        "j_medium": jm,
        "j_aggr": ja,
        "j_ultra": ju,
        "j_strong": j_strong
    })

baseline_strong = j_strong_table[0]["j_strong"]
baseline_light = j_strong_table[0]["j_light"]

for row in j_strong_table:
    delta_s = row["j_strong"] - baseline_strong
    delta_l = row["j_light"] - baseline_light
    print(f"rho={row['rho']:.2f}: J_STRONG = {row['j_strong']:.6f} (Delta_STRONG = {delta_s:+.6f}) | Light J_L = {row['j_light']:.6f} (Delta_L = {delta_l:+.6f})")

# 4. Audit Component LOO under rho=0.75 vs rho=1.00
print("\n================================================================================")
print("SECTION 7: DOWNSTREAM DEPENDENCY AUDIT — COMPONENT LOO (rho=0.75 vs rho=1.00)")
print("================================================================================")

loo_defs = [
    ("minus_d1_z", "Drop d1 (delta_z)", (0.0, 1.0, 1.0, 1.0, 1.0)),
    ("minus_d2_s", "Drop d2 (delta_score)", (1.0, 0.0, 1.0, 1.0, 1.0)),
    ("minus_d3_v", "Drop d3 (delta_v)", (1.0, 1.0, 0.0, 1.0, 1.0)),
    ("minus_d4_a", "Drop d4 (attn_entropy)", (1.0, 1.0, 1.0, 0.0, 1.0)),
    ("minus_d5_h", "Drop d5 (attn_energy)", (1.0, 1.0, 1.0, 1.0, 0.0)),
]

loo_audit_results = {}
for tier in TIERS:
    conf = tier_data[tier]["conf"]
    traces = tier_data[tier]["traces"]
    step1 = tier_data[tier]["step1"]
    eb_base = conf["eb_base"]
    target_rank = conf["target_rank"]
    
    tier_audit = []
    print(f"\n--- Tier: {tier.upper()} (Target Rank: {target_rank}) ---")
    for cid, desc, weights in loo_defs:
        old_lock_path = Path(f"/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks/family2_loo_{cid}_{tier}.json")
        with open(old_lock_path) as f:
            old_lock = json.load(f)
            
        new_ranks, new_c, diff = match_ranks(
            step1, traces, eb_base, target_rank, weights, lambda_eb=1.2, rho=0.75
        )
        new_tot_r = sum(sum(r) for r in new_ranks)
        new_b = new_tot_r * 4096
        new_gap_pct = ((new_b - conf["target_b_live"]) / conf["target_b_live"]) * 100.0
        new_sha = compute_rank_sha(new_ranks)
        old_sha = old_lock["rank_sha"]
        
        is_identical = (new_sha == old_sha)
        tier_audit.append({
            "cid": cid,
            "desc": desc,
            "weights": list(weights),
            "is_identical": is_identical,
            "old_sha": old_sha,
            "new_sha": new_sha,
            "old_matched_rank": old_lock["matched_rank"],
            "new_matched_rank": new_tot_r,
            "old_b_live": old_lock["matched_b_live"],
            "new_b_live": new_b,
            "new_gap_pct": new_gap_pct,
            "new_c_scale": new_c,
            "new_ranks": new_ranks
        })
        print(f"  {cid:12s}: identical={str(is_identical):5s} | old_sha={old_sha[:10]}... new_sha={new_sha[:10]}... | new_rank={new_tot_r} (gap={new_gap_pct:+.2f}%)")
    loo_audit_results[tier] = tier_audit

# Save LOO audit results
with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks/LOO_RHO75_AUDIT.json", "w") as f:
    json.dump(loo_audit_results, f, indent=2)

# Save unique physical states
unique_states_serializable = {}
for tier in TIERS:
    unique_states_serializable[tier] = []
    for sha, u in unique_states_by_tier[tier].items():
        unique_states_serializable[tier].append({
            "rhos": u["rhos"],
            "details": u["entry"]
        })

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection/RHO_UNIQUE_PHYSICAL_STATES.json", "w") as f:
    json.dump(unique_states_serializable, f, indent=2)

print("\nAudit and unique state analysis complete and saved.")
