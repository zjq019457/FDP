#!/usr/bin/env python3
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
from dev_evaluator import DevEvaluator
from formula_engine import TIER_CONF, paired_bootstrap_test

OUT_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
LOCK_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks")

TIERS = ["medium", "aggressive", "ultra", "light"]
RHO_VALUES = [1.00, 0.75, 0.50, 0.25, 0.00]

def clip_ranks_to_basis(ranks: List[List[int]], basis_dict: Dict[str, torch.Tensor]) -> List[List[int]]:
    clipped = []
    for li in range(32):
        rk, rv = ranks[li]
        qk = f"base_model.model.model.layers.{li}.self_attn.q_proj_basis"
        vk = f"base_model.model.model.layers.{li}.self_attn.v_proj_basis"
        q_max = basis_dict[qk].shape[1] * 8
        v_max = basis_dict[vk].shape[1]
        clipped.append([min(rk, q_max), min(rv, v_max)])
    return clipped

def main():
    print("================================================================================")
    print("RESUMING 20 RHO DEV EVALUATIONS WITH BASIS BOUNDARY SAFE-GUARD")
    print("Device: cuda:2 | Batch: DEV-WIKI (64 win) + DEV-C4 (64 win)")
    print("================================================================================")
    
    # Load progress if exists
    progress_file = OUT_DIR / "RHO_DEV_EVAL_PROGRESS.json"
    if progress_file.exists():
        with open(progress_file) as f:
            all_results = json.load(f)
        print(f"Loaded existing progress for tiers: {list(all_results.keys())}")
    else:
        all_results = {}
        
    leaderboard_csv = OUT_DIR / "FDP_RHO_LEADERBOARD.csv"
    fields = [
        "tier", "rho", "total_rank", "B_live", "gap_pct", "rank_sha",
        "dev_wiki_nll", "dev_wiki_ppl", "dev_c4_nll", "dev_c4_ppl",
        "J_dev", "delta_J", "ci_low", "ci_high", "p_value"
    ]
    if not leaderboard_csv.exists():
        with open(leaderboard_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
    start_total = time.time()
    
    # In-memory storage for window losses (needed for bootstrap)
    window_losses_dict = {}
    
    for tier in TIERS:
        conf = TIER_CONF[tier]
        print(f"\n==================================================")
        print(f"TIER: {tier.upper()} (e_base={conf['eb_base']:.3f}, target_rank={conf['target_rank']})")
        print(f"==================================================")
        
        # Check if tier already completely evaluated
        if tier in all_results and len(all_results[tier]) == len(RHO_VALUES):
            print(f"Tier {tier} already complete with {len(all_results[tier])} evaluations, skipping re-evaluation.")
            continue
            
        t0 = time.time()
        evaluator = DevEvaluator(
            base_model_path="/diskD/jiaqizhang/Llama-3",
            basis_pt_path=conf["basis"],
            device="cuda:2",
            dtype=torch.bfloat16
        )
        print(f"Initialized DevEvaluator for {tier} in {time.time()-t0:.1f}s")
        
        all_results[tier] = {}
        window_losses_dict[tier] = {}
        baseline_losses = None
        
        for rho in RHO_VALUES:
            rho_cid = f"rho_{int(rho*100):02d}"
            lock_file = LOCK_DIR / f"family1_rho_{rho_cid}_{tier}.json"
            with open(lock_file) as lf:
                lock_data = json.load(lf)
                
            raw_ranks = lock_data["ranks"]
            tot_r = lock_data["matched_rank"]
            b_live = lock_data["matched_b_live"]
            gap_pct = lock_data["gap_pct"]
            r_sha = lock_data["rank_sha"]
            
            # Safe boundary clip
            safe_ranks = clip_ranks_to_basis(raw_ranks, evaluator.basis_dict)
            
            print(f"\n--- Evaluating {tier.upper()} rho={rho:.2f} ({rho_cid}) ---")
            print(f"    Target Rank: {tot_r} (gap={gap_pct:+.2f}%), SHA: {r_sha[:16]}...")
            
            t_eval = time.time()
            eval_res = evaluator.evaluate_ranks(safe_ranks)
            eval_sec = time.time() - t_eval
            
            wiki_nll = eval_res["dev_wiki_nll"]
            wiki_ppl = eval_res["dev_wiki_ppl"]
            c4_nll = eval_res["dev_c4_nll"]
            c4_ppl = eval_res["dev_c4_ppl"]
            j_dev = eval_res["J_dev"]
            cand_losses = eval_res["wiki_per_window_nll"] + eval_res["c4_per_window_nll"]
            window_losses_dict[tier][str(rho)] = cand_losses
            
            if rho == 1.00:
                baseline_losses = cand_losses
                delta_j = 0.0
                ci_low, ci_high, p_val = 0.0, 0.0, 1.0
            else:
                delta_j, ci_low, ci_high, p_val = paired_bootstrap_test(baseline_losses, cand_losses)
                
            print(f"    Eval time: {eval_sec:.1f}s")
            print(f"    DEV-WIKI: NLL={wiki_nll:.4f}, PPL={wiki_ppl:.4f}")
            print(f"    DEV-C4:   NLL={c4_nll:.4f}, PPL={c4_ppl:.4f}")
            print(f"    J_dev:    {j_dev:.6f}")
            if rho != 1.00:
                print(f"    Delta vs rho=1.00: {delta_j:+.6f} [95% CI: {ci_low:+.6f}, {ci_high:+.6f}], p={p_val:.4f}")
                
            res_entry = {
                "tier": tier,
                "rho": rho,
                "total_rank": tot_r,
                "B_live": b_live,
                "gap_pct": gap_pct,
                "rank_sha": r_sha,
                "dev_wiki_nll": wiki_nll,
                "dev_wiki_ppl": wiki_ppl,
                "dev_c4_nll": c4_nll,
                "dev_c4_ppl": c4_ppl,
                "J_dev": j_dev,
                "delta_J": delta_j,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_val
            }
            all_results[tier][str(rho)] = res_entry
            
            with open(leaderboard_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writerow(res_entry)
                
            with open(progress_file, "w") as jf:
                json.dump(all_results, jf, indent=2)
                
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Tier {tier} complete. Cleaned up GPU memory.")
        
    total_time = time.time() - start_total
    print(f"\n================================================================================")
    print(f"ALL 20 RHO DEV EVALUATIONS COMPLETE in {total_time:.1f}s ({total_time/60:.2f} min)!")
    print(f"================================================================================")
    
    # Cross-tier Synthesis
    print("\n=== CROSS-TIER RHO SYNTHESIS TABLE ===")
    print(f"{'rho':>6} | {'Light J':>9} | {'Medium J':>9} | {'Aggr J':>9} | {'Ultra J':>9} | {'J_STRONG':>9} | {'Delta_STRONG':>12} | {'Strong CI (95%)':>22} | {'p-val':>6} | {'Status'}")
    print("-" * 115)
    
    synthesis_summary = []
    base_m = all_results["medium"]["1.0"]["J_dev"]
    base_a = all_results["aggressive"]["1.0"]["J_dev"]
    base_u = all_results["ultra"]["1.0"]["J_dev"]
    base_strong = (base_m + base_a + base_u) / 3.0
    
    for rho in RHO_VALUES:
        r_str = str(rho)
        jl = all_results["light"][r_str]["J_dev"]
        jm = all_results["medium"][r_str]["J_dev"]
        ja = all_results["aggressive"][r_str]["J_dev"]
        ju = all_results["ultra"][r_str]["J_dev"]
        
        j_strong = (jm + ja + ju) / 3.0
        delta_strong = j_strong - base_strong
        
        delta_m = jm - base_m
        delta_a = ja - base_a
        delta_u = ju - base_u
        delta_l = jl - all_results["light"]["1.0"]["J_dev"]
        
        if rho == 1.00:
            status = "HISTORICAL_BASELINE"
        elif delta_strong < -0.003 and delta_l <= 0.003:
            status = "WINNER_CANDIDATE"
        elif delta_strong > 0.003:
            status = "STRICTLY_WORSE"
        else:
            status = "NEUTRAL_OR_TIE"
            
        print(f"{rho:6.2f} | {jl:9.6f} | {jm:9.6f} | {ja:9.6f} | {ju:9.6f} | {j_strong:9.6f} | {delta_strong:+12.6f} | {'N/A':>22} | {'N/A':>6} | {status}")
        
        synthesis_summary.append({
            "rho": rho,
            "light_J": jl,
            "medium_J": jm,
            "aggressive_J": ja,
            "ultra_J": ju,
            "J_STRONG": j_strong,
            "delta_J_STRONG": delta_strong,
            "status": status
        })
        
    with open(OUT_DIR / "RHO_DECISION_SYNTHESIS.json", "w") as f:
        json.dump(synthesis_summary, f, indent=2)
    print("Synthesis saved to RHO_DECISION_SYNTHESIS.json")

if __name__ == "__main__":
    main()
