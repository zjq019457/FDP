#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
from dev_evaluator import DevEvaluator
from formula_engine import TIER_CONF, paired_bootstrap_test

OUT_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
LOCK_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/resource_locks")

CIDS = [
    ("minus_d1_z", "Drop d1 (delta_z)"),
    ("minus_d2_s", "Drop d2 (delta_score)"),
    ("minus_d3_v", "Drop d3 (delta_v)"),
    ("minus_d4_a", "Drop d4 (attn_entropy)"),
    ("minus_d5_h", "Drop d5 (attn_energy)")
]

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", type=str, required=True, help="Comma-separated tiers, e.g. 'medium,aggressive'")
    parser.add_argument("--gpu", type=int, required=True, help="CUDA device index, e.g. 2 or 3")
    args = parser.parse_args()
    
    tiers_to_run = [t.strip() for t in args.tiers.split(",")]
    device = f"cuda:{args.gpu}"
    
    print(f"================================================================================")
    print(f"COMPONENT LOO EVALUATIONS (Family 2) on {device}")
    print(f"Tiers: {tiers_to_run}")
    print(f"================================================================================")
    
    out_json = OUT_DIR / f"LOO_EVAL_PROGRESS_GPU{args.gpu}.json"
    if out_json.exists():
        with open(out_json) as f:
            tier_results = json.load(f)
    else:
        tier_results = {}
        
    for tier in tiers_to_run:
        conf = TIER_CONF[tier]
        print(f"\n==================================================")
        print(f"TIER: {tier.upper()} (e_base={conf['eb_base']:.3f}, target_rank={conf['target_rank']})")
        print(f"==================================================")
        
        t0 = time.time()
        evaluator = DevEvaluator(
            base_model_path="/diskD/jiaqizhang/Llama-3",
            basis_pt_path=conf["basis"],
            device=device,
            dtype=torch.bfloat16
        )
        print(f"Initialized DevEvaluator for {tier} in {time.time()-t0:.1f}s")
        
        # Load baseline lock (F_CURRENT: rho=0.75)
        base_lock_file = LOCK_DIR / f"family1_rho_rho_75_{tier}.json"
        with open(base_lock_file) as lf:
            base_lock = json.load(lf)
            
        base_ranks = clip_ranks_to_basis(base_lock["ranks"], evaluator.basis_dict)
        base_sha = base_lock["rank_sha"]
        
        print(f"\n--- Evaluating Baseline F_CURRENT (rho=0.75) for {tier.upper()} ---")
        t_base = time.time()
        base_res = evaluator.evaluate_ranks(base_ranks)
        base_sec = time.time() - t_base
        base_losses = base_res["wiki_per_window_nll"] + base_res["c4_per_window_nll"]
        print(f"    Base Eval time: {base_sec:.1f}s | J_dev: {base_res['J_dev']:.6f}")
        print(f"    DEV-WIKI: NLL={base_res['dev_wiki_nll']:.4f}, PPL={base_res['dev_wiki_ppl']:.4f}")
        print(f"    DEV-C4:   NLL={base_res['dev_c4_nll']:.4f}, PPL={base_res['dev_c4_ppl']:.4f}")
        
        tier_results[tier] = {
            "baseline": {
                "cid": "baseline_f_current",
                "desc": "Promoted DEV Baseline F(rho=0.75)",
                "total_rank": base_lock["matched_rank"],
                "B_live": base_lock["matched_b_live"],
                "gap_pct": base_lock["gap_pct"],
                "rank_sha": base_sha,
                "dev_wiki_nll": base_res["dev_wiki_nll"],
                "dev_wiki_ppl": base_res["dev_wiki_ppl"],
                "dev_c4_nll": base_res["dev_c4_nll"],
                "dev_c4_ppl": base_res["dev_c4_ppl"],
                "J_dev": base_res["J_dev"]
            },
            "candidates": {}
        }
        
        for cid, desc in CIDS:
            lock_file = LOCK_DIR / f"family2_loo_{cid}_{tier}.json"
            with open(lock_file) as lf:
                lock_data = json.load(lf)
                
            raw_ranks = lock_data["ranks"]
            tot_r = lock_data["matched_rank"]
            b_live = lock_data["matched_b_live"]
            gap_pct = lock_data["gap_pct"]
            r_sha = lock_data["rank_sha"]
            
            print(f"\n--- Evaluating {tier.upper()} LOO: {cid} ({desc}) ---")
            print(f"    Target Rank: {tot_r} (gap={gap_pct:+.2f}%), SHA: {r_sha[:16]}...")
            
            if r_sha == base_sha:
                print(f"    [IDENTICAL TO BASELINE] Reusing baseline metrics directly.")
                eval_sec = 0.0
                wiki_nll = base_res["dev_wiki_nll"]
                wiki_ppl = base_res["dev_wiki_ppl"]
                c4_nll = base_res["dev_c4_nll"]
                c4_ppl = base_res["dev_c4_ppl"]
                j_dev = base_res["J_dev"]
                delta_j = 0.0
                ci_low, ci_high, p_val = 0.0, 0.0, 1.0
            else:
                safe_ranks = clip_ranks_to_basis(raw_ranks, evaluator.basis_dict)
                t_eval = time.time()
                eval_res = evaluator.evaluate_ranks(safe_ranks)
                eval_sec = time.time() - t_eval
                wiki_nll = eval_res["dev_wiki_nll"]
                wiki_ppl = eval_res["dev_wiki_ppl"]
                c4_nll = eval_res["dev_c4_nll"]
                c4_ppl = eval_res["dev_c4_ppl"]
                j_dev = eval_res["J_dev"]
                cand_losses = eval_res["wiki_per_window_nll"] + eval_res["c4_per_window_nll"]
                delta_j, ci_low, ci_high, p_val = paired_bootstrap_test(base_losses, cand_losses)
                
            print(f"    Eval time: {eval_sec:.1f}s")
            print(f"    DEV-WIKI: NLL={wiki_nll:.4f}, PPL={wiki_ppl:.4f}")
            print(f"    DEV-C4:   NLL={c4_nll:.4f}, PPL={c4_ppl:.4f}")
            print(f"    J_dev:    {j_dev:.6f}")
            print(f"    Delta vs F_CURRENT: {delta_j:+.6f} [95% CI: {ci_low:+.6f}, {ci_high:+.6f}], p={p_val:.4f}")
            
            tier_results[tier]["candidates"][cid] = {
                "cid": cid,
                "desc": desc,
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
                "p_value": p_val,
                "eval_sec": eval_sec
            }
            
            # Save progress incrementally
            with open(out_json, "w") as f:
                json.dump(tier_results, f, indent=2)
                
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Tier {tier} complete. Cleaned up GPU memory.")

    print(f"\n==================================================")
    print(f"All requested tiers {tiers_to_run} complete on {device}!")
    print(f"Results saved to {out_json}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
