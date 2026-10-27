#!/usr/bin/env python3
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from formula_engine import TIER_CONF, load_traces, match_ranks, paired_bootstrap_test
from dev_evaluator import DevEvaluator

def main():
    print("=== STARTING PHASE 4: DEVELOPMENT FORMULA SEARCH ===")
    tier = "medium"
    conf = TIER_CONF[tier]
    
    print(f"Primary Development Tier: {tier.upper()} (e={conf['eb_base']:.3f}, target_rank={conf['target_rank']})")
    traces = load_traces(conf["report"])
    with open(conf["step1"]) as f:
        step1_data = json.load(f)
        
    # Initialize DevEvaluator on GPU 3
    evaluator = DevEvaluator(
        base_model_path="/diskD/jiaqizhang/Llama-3",
        basis_pt_path=conf["basis"],
        device="cuda:3",
        dtype=torch.bfloat16
    )
    
    leaderboard_csv = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection/FDP_FORMULA_LEADERBOARD.csv")
    fields = [
        "candidate_id", "family", "tier", "description",
        "rho", "lambda", "weights", "c_scale",
        "total_rank", "B_live", "target_gap_pct",
        "dev_wiki_nll", "dev_wiki_ppl",
        "dev_c4_nll", "dev_c4_ppl",
        "J_dev", "delta_J_dev", "ci_low", "ci_high", "p_value"
    ]
    
    # Check if CSV exists, else create with header
    if not leaderboard_csv.exists():
        with open(leaderboard_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
    # Define candidate suite
    candidates = []
    
    # F0: Historical Baseline
    candidates.append({
        "id": "F0_HIST",
        "family": "Baseline",
        "desc": "Historical May Baseline (lambda=1.2, rho=1.0, w=(1,1,1,1,1))",
        "rho": 1.0, "lambda": 1.2, "weights": (1.0, 1.0, 1.0, 1.0, 1.0)
    })
    
    # Family 1: Layer-0 Policy (rho sweep)
    for rho_val in [0.00, 0.25, 0.50, 0.75]:
        candidates.append({
            "id": f"F1_rho_{int(rho_val*100):02d}",
            "family": "Family 1 (Layer-0)",
            "desc": f"Layer-0 blending rho={rho_val:.2f}",
            "rho": rho_val, "lambda": 1.2, "weights": (1.0, 1.0, 1.0, 1.0, 1.0)
        })
        
    # Family 2: Leave-One-Out Diagnostics
    loo_weights = [
        ("F2_minus_d1_z", "Drop d1 (delta_z)", (0.0, 1.0, 1.0, 1.0, 1.0)),
        ("F2_minus_d2_s", "Drop d2 (delta_score)", (1.0, 0.0, 1.0, 1.0, 1.0)),
        ("F2_minus_d3_v", "Drop d3 (delta_v)", (1.0, 1.0, 0.0, 1.0, 1.0)),
        ("F2_minus_d4_a", "Drop d4 (attn_entropy)", (1.0, 1.0, 1.0, 0.0, 1.0)),
        ("F2_minus_d5_h", "Drop d5 (entropy_collapse)", (1.0, 1.0, 1.0, 1.0, 0.0)),
    ]
    for cid, cdesc, w in loo_weights:
        candidates.append({
            "id": cid,
            "family": "Family 2 (Leave-One-Out)",
            "desc": cdesc,
            "rho": 1.0, "lambda": 1.2, "weights": w
        })
        
    # Family 3: Lambda Sensitivity
    for lam in [0.6, 0.9, 1.5, 1.8]:
        candidates.append({
            "id": f"F3_lambda_{str(lam).replace('.', 'p')}",
            "family": "Family 3 (Lambda)",
            "desc": f"Coupling strength lambda={lam:.1f}",
            "rho": 1.0, "lambda": lam, "weights": (1.0, 1.0, 1.0, 1.0, 1.0)
        })
        
    # Family 4: Coordinate Perturbations
    pert_weights = [
        ("F4_wz_2x", "Double d1 (wz=2.0)", (2.0, 1.0, 1.0, 1.0, 1.0)),
        ("F4_ws_2x", "Double d2 (ws=2.0)", (1.0, 2.0, 1.0, 1.0, 1.0)),
        ("F4_wv_2x", "Double d3 (wv=2.0)", (1.0, 1.0, 2.0, 1.0, 1.0)),
        ("F4_wa_2x", "Double d4 (wa=2.0)", (1.0, 1.0, 1.0, 2.0, 1.0)),
        ("F4_wh_2x", "Double d5 (wh=2.0)", (1.0, 1.0, 1.0, 1.0, 2.0)),
    ]
    for cid, cdesc, w in pert_weights:
        candidates.append({
            "id": cid,
            "family": "Family 4 (Perturbation)",
            "desc": cdesc,
            "rho": 1.0, "lambda": 1.2, "weights": w
        })
        
    print(f"Total candidates to evaluate: {len(candidates)}")
    
    # Store F0 window losses for paired bootstrap
    f0_window_losses = None
    
    for idx, c in enumerate(candidates):
        cid = c["id"]
        print(f"
--- [{idx+1}/{len(candidates)}] Evaluating {cid}: {c['desc']} ---")
        
        # 1. Match ranks
        ranks, best_c, diff = match_ranks(
            step1_data=step1_data,
            traces=traces,
            eb_base=conf["eb_base"],
            target_rank=conf["target_rank"],
            weights=c["weights"],
            lambda_eb=c["lambda"],
            rho=c["rho"]
        )
        tot_r = sum(sum(r) for r in ranks)
        b_live = tot_r * 4096
        gap_pct = ((b_live - conf["target_b_live"]) / conf["target_b_live"]) * 100.0
        print(f"  Matched rank: {tot_r} / {conf['target_rank']} (gap={gap_pct:+.4f}%, diff={diff} ranks, c={best_c:.4f})")
        
        # 2. Evaluate on DEV
        eval_res = evaluator.evaluate_ranks(ranks)
        wiki_nll = eval_res["dev_wiki_nll"]
        wiki_ppl = eval_res["dev_wiki_ppl"]
        c4_nll = eval_res["dev_c4_nll"]
        c4_ppl = eval_res["dev_c4_ppl"]
        j_dev = eval_res["J_dev"]
        
        # Combined window losses: 64 wiki + 64 c4
        cand_window_losses = eval_res["wiki_per_window_nll"] + eval_res["c4_per_window_nll"]
        
        if cid == "F0_HIST":
            f0_window_losses = cand_window_losses
            delta_j = 0.0
            ci_low, ci_high, p_val = 0.0, 0.0, 1.0
        else:
            delta_j, ci_low, ci_high, p_val = paired_bootstrap_test(f0_window_losses, cand_window_losses)
            
        print(f"  DEV-WIKI PPL: {wiki_ppl:.4f} | DEV-C4 PPL: {c4_ppl:.4f} | J_dev: {j_dev:.6f}")
        print(f"  Delta J_dev: {delta_j:+.6f} [95% CI: {ci_low:+.6f}, {ci_high:+.6f}], p-value: {p_val:.4f}")
        
        # Write to CSV immediately
        row = {
            "candidate_id": cid,
            "family": c["family"],
            "tier": tier,
            "description": c["desc"],
            "rho": c["rho"],
            "lambda": c["lambda"],
            "weights": str(c["weights"]),
            "c_scale": best_c,
            "total_rank": tot_r,
            "B_live": b_live,
            "target_gap_pct": gap_pct,
            "dev_wiki_nll": wiki_nll,
            "dev_wiki_ppl": wiki_ppl,
            "dev_c4_nll": c4_nll,
            "dev_c4_ppl": c4_ppl,
            "J_dev": j_dev,
            "delta_J_dev": delta_j,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_val
        }
        with open(leaderboard_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(row)
            
    print("
=== PHASE 4 DEVELOPMENT FORMULA SEARCH COMPLETE! ===")

if __name__ == "__main__":
    main()
