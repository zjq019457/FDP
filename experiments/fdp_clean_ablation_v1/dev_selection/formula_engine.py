#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
from realloc_rank_from_fragility import fragility_vector, minmax

DEV_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
TIER_CONF = {
    "light": {
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb045.json",
        "eb_base": 0.045,
        "target_b_live": 218988544, # 53464 rank
        "target_rank": 53464
    },
    "medium": {
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb060.json",
        "eb_base": 0.060,
        "target_b_live": 198377472, # 48432 rank
        "target_rank": 48432
    },
    "aggressive": {
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb090.json",
        "eb_base": 0.090,
        "target_b_live": 158105600, # 38600 rank
        "target_rank": 38600
    },
    "ultra": {
        "report": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb_report.json",
        "basis": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb120.json",
        "eb_base": 0.120,
        "target_b_live": 124223488, # 30328 rank
        "target_rank": 30328
    }
}

def compute_rank_sha(ranks: List[List[int]]) -> str:
    ranks_str = json.dumps(ranks, separators=(',', ':'))
    return hashlib.sha256(ranks_str.encode('utf-8')).hexdigest()

def load_traces(report_path: str):
    with open(report_path) as f:
        rep = json.load(f)
    return [l["runs"][0]["threshold_trace"] for l in rep["layers"]]

def oracle_lookup(trace, eb, svd_floor=0.30):
    best_rk, best_rv = 1024, 1024
    for step in trace:
        th = step["threshold"]
        if step["err"] <= eb and th >= svd_floor:
            best_rk = step["rank_kq"]
            best_rv = step["rank_v"]
        else:
            break
    return best_rk, best_rv

def derive_eb_schedule(step1_data, eb_base, weights, lambda_eb, rho, c_scale=1.0):
    wz, ws, wv, wa, wh = weights
    per_layer = step1_data["per_layer"]
    layers = sorted(int(k) for k in per_layer.keys())
    s_raw, _ = fragility_vector(per_layer, layers, wz, ws, wv, wa, wh, False)
    s_tilde = minmax(s_raw)
    
    eb_inv = [eb_base / (1.0 + lambda_eb * st) for st in s_tilde]
    m = sum(eb_inv) / len(eb_inv)
    scale = (eb_base / m) if m > 0 else 1.0
    eb_norm = [e * scale for e in eb_inv]
    
    # Layer 0 blending
    eps0 = math.exp(rho * math.log(eb_base) + (1.0 - rho) * math.log(eb_norm[0]))
    eb_norm[0] = eps0
    
    eb_final = [e * c_scale for e in eb_norm]
    if rho >= 0.999:
        eb_final[0] = eb_base
    return eb_final

def match_ranks(step1_data, traces, eb_base, target_rank, weights, lambda_eb, rho):
    best_ranks = None
    best_diff = float("inf")
    best_c = 1.0
    
    # Search c grid around 1.0
    c_candidates = [1.0] + [1.0 + x * 0.005 for x in range(-40, 41) if x != 0]
    for c in c_candidates:
        eb_sched = derive_eb_schedule(step1_data, eb_base, weights, lambda_eb, rho, c)
        ranks = []
        for li in range(32):
            rk, rv = oracle_lookup(traces[li], eb_sched[li])
            ranks.append([rk, rv])
        tot_r = sum(sum(r) for r in ranks)
        diff = abs(tot_r - target_rank)
        if diff < best_diff:
            best_diff = diff
            best_ranks = ranks
            best_c = c
            if diff == 0:
                break
    return best_ranks, best_c, best_diff

def paired_bootstrap_test(f0_losses, cand_losses, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    f0 = np.array(f0_losses)
    cand = np.array(cand_losses)
    diff = cand - f0 # negative is improvement
    obs_mean = float(np.mean(diff))
    
    n = len(diff)
    boot_means = rng.choice(diff, size=(n_boot, n), replace=True).mean(axis=1)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    
    # one-sided p-value: probability that candidate is NOT better (mean >= 0)
    p_val = float(np.mean(boot_means >= 0))
    return obs_mean, ci_low, ci_high, p_val

print("Execution engine loaded successfully with all 4 tiers.")
