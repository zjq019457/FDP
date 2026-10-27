#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
from formula_engine import TIER_CONF, load_traces, match_ranks, compute_rank_sha, compute_b_live

results_file = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection/rho_eval_results.json")
if not results_file.exists():
    print("rho_eval_results.json not found")
    sys.exit(1)

with open(results_file) as f:
    eval_results = json.load(f)

TIERS = ["medium", "aggressive", "ultra", "light"]
RHOS = [1.00, 0.75, 0.50, 0.25, 0.00]

tier_reports = {}

for tier in TIERS:
    conf = TIER_CONF[tier]
    traces = load_traces(conf["report"])
    with open(conf["step1"]) as f:
        step1 = json.load(f)
    eb_base = conf["eb_base"]
    target_rank = conf["target_rank"]
    
    tier_reports[tier] = []
    
    for rho in RHOS:
        cid = f"rho_{int(rho*100):02d}"
        lock_file = Path(f"/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks/family1_rho_{cid}_{tier}.json")
        with open(lock_file) as f:
            lock = json.load(f)
            
        c_scale = lock["c_scale"]
        
        # Calculate raw Layer 0 EB before interpolation:
        w = [1.0, 1.0, 1.0, 1.0, 1.0]
        components = ["delta_z", "delta_score", "delta_v", "attn_entropy", "attn_energy"]
        comp_scores = []
        for i, c in enumerate(components):
            c_val = step1[c]
            mean_val = sum(c_val) / len(c_val)
            std_val = (sum((x - mean_val)**2 for x in c_val) / len(c_val))**0.5
            norm_val = [(x - mean_val) / (std_val + 1e-8) for x in c_val]
            comp_scores.append([x * w[i] for x in norm_val])
        combined = [sum(comp_scores[j][l] for j in range(5)) for l in range(32)]
        mean_c = sum(combined) / len(combined)
        std_c = (sum((x - mean_c)**2 for x in combined) / len(combined))**0.5
        norm_comb = [(x - mean_c) / (std_c + 1e-8) for x in combined]
        
        s = [math.exp(-1.2 * x) for x in norm_comb]
        mean_s = sum(s) / len(s)
        s_norm = [x / mean_s for x in s]
        
        eb_0_before = eb_base * c_scale * s_norm[0]
        
        if rho >= 0.999:
            eb_0_after = eb_base * c_scale
        else:
            eb_0_after = math.exp(rho * math.log(eb_base * c_scale) + (1.0 - rho) * math.log(eb_0_before))
            
        l0_kq_rank = lock["ranks"][0][0]
        l0_v_rank = lock["ranks"][0][1]
        
        res_key = f"{tier}_{cid}"
        metrics = eval_results.get(res_key, {})
        
        tier_reports[tier].append({
            "rho": rho,
            "cid": cid,
            "eb_0_before": eb_0_before,
            "eb_0_after": eb_0_after,
            "l0_kq_rank": l0_kq_rank,
            "l0_v_rank": l0_v_rank,
            "rank_sha": lock["rank_sha"],
            "total_rank": lock["matched_rank"],
            "b_live": lock["matched_b_live"],
            "gap_pct": lock["gap_pct"],
            "dev_wiki_nll": metrics.get("dev_wiki_nll"),
            "dev_wiki_ppl": metrics.get("dev_wiki_ppl"),
            "dev_c4_nll": metrics.get("dev_c4_nll"),
            "dev_c4_ppl": metrics.get("dev_c4_ppl"),
            "j_dev": metrics.get("j_dev"),
            "delta_j": metrics.get("delta_j"),
            "ci_95": metrics.get("ci_95"),
            "p_val": metrics.get("p_val")
        })

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection/detailed_rho_report.json", "w") as f:
    json.dump(tier_reports, f, indent=2)

print("Report saved to detailed_rho_report.json")
