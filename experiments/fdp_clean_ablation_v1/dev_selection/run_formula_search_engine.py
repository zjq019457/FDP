#!/usr/bin/env python3
"""
FDP FORMULA RESEARCH V1 - COMPREHENSIVE FORMULA FAMILY SEARCH & TUNING
Governed by FORMULA-SEARCH PHASE BOUNDARY PATCH.

Executes:
1. Step C: Formula family comparison (F_BASE, F_PROP, F_SRP)
2. Step D: Layer 0 protection internalization audit (sweep rho in [0, 1])
3. Step E: Lambda sweep & component weight coordinate descent
4. Ablation cleanliness & bootstrap verification
5. CSVs & markdown reports export
"""
import copy
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, "/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/eigenattention_upstream")
sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")

from dev_evaluator import DevEvaluator, _compat_rope
from formula_engine import TIER_CONF, load_traces, oracle_lookup, paired_bootstrap_test
from realloc_rank_from_fragility import fragility_vector

DEV_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
CAUSAL_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/causal")
FORMULA_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/formula_search")
FINAL_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/final")
FORMULA_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

TIERS = ["light", "medium", "aggressive", "ultra"]
COMPONENTS = ["d1_z", "d2_s", "d3_v", "d4_a", "d5_h"]

def minmax(x: List[float], eps: float = 1e-12) -> List[float]:
    lo, hi = min(x), max(x)
    span = hi - lo
    if span < eps:
        return [0.5 for _ in x]
    return [(t - lo) / span for t in x]

def compute_rank_sha(ranks: List[List[int]]) -> str:
    ranks_str = json.dumps(ranks, separators=(',', ':'))
    return hashlib.sha256(ranks_str.encode('utf-8')).hexdigest()

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

def compute_H(step1_data, G_tier, weights, mode="prop", eta=0.5, alpha_s=0.5, alpha_r=0.0, alpha_p=0.5):
    per_layer = step1_data["per_layer"]
    layers = sorted(int(k) for k in per_layer.keys())
    wz, ws, wv, wa, wh = weights
    
    # Baseline observed fragility S_l
    s_raw, _ = fragility_vector(per_layer, layers, wz, ws, wv, wa, wh, False)
    s_tilde = minmax(s_raw)
    
    if mode == "base":
        return s_tilde
        
    # Downstream propagation P_l
    P_raw = []
    for l in range(32):
        if l < 31:
            val = 0.0
            for j, w in enumerate(weights):
                pos_down = np.maximum(0.0, G_tier[j, (l+1):32, l])
                val += w * float(np.mean(pos_down))
            P_raw.append(val)
        else:
            P_raw.append(0.0)
    P_tilde = minmax(P_raw)
    
    if mode == "prop":
        return [(1.0 - eta) * s_tilde[l] + eta * P_tilde[l] for l in range(32)]
        
    # Self-response R_l (chi_l)
    R_raw = []
    for l in range(32):
        val = 0.0
        for j, w in enumerate(weights):
            pos_self = max(0.0, float(G_tier[j, l, l]))
            val += w * pos_self
        R_raw.append(val)
    R_tilde = minmax(R_raw)
    
    if mode == "srp":
        return [alpha_s * s_tilde[l] + alpha_r * R_tilde[l] + alpha_p * P_tilde[l] for l in range(32)]
        
    raise ValueError(f"Unknown mode {mode}")

def derive_eb_schedule(H, eb_base, lambda_eb, rho, c_scale=1.0):
    eb_inv = [eb_base / (1.0 + lambda_eb * hl) for hl in H]
    m = sum(eb_inv) / len(eb_inv)
    scale = (eb_base / m) if m > 0 else 1.0
    eb_norm = [e * scale for e in eb_inv]
    
    # Layer 0 geometric blending
    eps0 = math.exp(rho * math.log(eb_base) + (1.0 - rho) * math.log(eb_norm[0]))
    eb_norm[0] = eps0
    
    eb_final = [e * c_scale for e in eb_norm]
    if rho >= 0.999:
        eb_final[0] = eb_base
    return eb_final

def match_ranks(H, traces, basis_dict, eb_base, target_rank, lambda_eb, rho):
    best_ranks = None
    best_diff = float("inf")
    best_c = 1.0
    
    c_candidates = [1.0] + [1.0 + x * 0.002 for x in range(-50, 51) if x != 0]
    for c in c_candidates:
        eb_sched = derive_eb_schedule(H, eb_base, lambda_eb, rho, c)
        ranks = []
        for li in range(32):
            rk, rv = oracle_lookup(traces[li], eb_sched[li])
            ranks.append([rk, rv])
        ranks_safe = clip_ranks_to_basis(ranks, basis_dict)
        tot_r = sum(sum(r) for r in ranks_safe)
        diff = abs(tot_r - target_rank)
        if diff < best_diff:
            best_diff = diff
            best_ranks = ranks_safe
            best_c = c
            if diff == 0:
                break
    return best_ranks, best_c, best_diff

class FormulaSearchRunner:
    def __init__(self, device: str = "cuda:2"):
        self.device = device
        self.traces = {}
        self.step1_data = {}
        self.basis_dicts = {}
        self.evaluators = {}
        
        print("[FormulaSearchRunner] Loading tier configs and oracle traces...")
        for tier in TIERS:
            conf = TIER_CONF[tier]
            self.traces[tier] = load_traces(conf["report"])
            with open(conf["step1"]) as f:
                self.step1_data[tier] = json.load(f)
            self.basis_dicts[tier] = torch.load(conf["basis"], map_location="cpu")
            
        print("[FormulaSearchRunner] Loading Causal Response Tensor...")
        tensor_path = CAUSAL_DIR / "causal_response_tensor.pt"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Causal tensor not found at {tensor_path}. Run synthesize_causal_tensor.py first.")
        causal_data = torch.load(tensor_path, map_location="cpu")
        # G_4d: [5, 32, 32, 4]
        self.G_4d = causal_data["tensor"].numpy().astype(np.float64)
        
        # Load or initialize DEV eval cache
        self.cache_path = DEV_DIR / "formula_search_dev_cache.json"
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                self.dev_cache = json.load(f)
            print(f"[FormulaSearchRunner] Loaded {sum(len(v) for v in self.dev_cache.values())} cached evaluations.")
        else:
            self.dev_cache = {t: {} for t in TIERS}
            
        # Baseline F_CURRENT (rho=0.75, lambda=1.2, unit weights) values from component LOO
        self.f_current_j = {
            "light": 2.135747,
            "medium": 2.270080,
            "aggressive": 2.619581,
            "ultra": 5.129243
        }
        self.f_current_j_strong = (self.f_current_j["medium"] + self.f_current_j["aggressive"] + self.f_current_j["ultra"]) / 3.0

    def evaluate_tier_ranks(self, tier: str, ranks: List[List[int]]) -> Dict[str, object]:
        sha = compute_rank_sha(ranks)
        if sha in self.dev_cache[tier]:
            return self.dev_cache[tier][sha]
            
        # Materialize evaluator on demand for this tier
        conf = TIER_CONF[tier]
        evaluator = DevEvaluator(
            base_model_path="/diskD/jiaqizhang/Llama-3",
            basis_pt_path=conf["basis"],
            device=self.device,
            dtype=torch.bfloat16
        )
        
        evaluator.set_ranks(ranks)
        w_nll, w_per_win = evaluator.evaluate_tokens(evaluator.dev_wiki)
        c_nll, c_per_win = evaluator.evaluate_tokens(evaluator.dev_c4)
        
        j_tier = (w_nll + c_nll) / 2.0
        tot_r = sum(sum(r) for r in ranks)
        gap_pct = abs(tot_r - conf["target_rank"]) / conf["target_rank"] * 100.0
        
        res = {
            "sha": sha,
            "ranks": ranks,
            "tot_rank": tot_r,
            "target_rank": conf["target_rank"],
            "gap_pct": gap_pct,
            "wiki_nll": w_nll,
            "c4_nll": c_nll,
            "j_tier": j_tier,
            "wiki_per_win": w_per_win,
            "c4_per_win": c_per_win
        }
        self.dev_cache[tier][sha] = res
        
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()
        
        # Save cache
        with open(self.cache_path, "w") as f:
            json.dump(self.dev_cache, f)
            
        return res

    def evaluate_candidate(self, candidate_name: str, mode: str, weights: List[float], lambda_eb: float, rho: float,
                           eta: float = 0.5, alpha_s: float = 0.5, alpha_r: float = 0.0, alpha_p: float = 0.5) -> Dict[str, object]:
        tier_results = {}
        for e_idx, tier in enumerate(TIERS):
            conf = TIER_CONF[tier]
            G_tier = self.G_4d[:, :, :, e_idx]
            H = compute_H(self.step1_data[tier], G_tier, weights, mode, eta, alpha_s, alpha_r, alpha_p)
            ranks, c_val, diff = match_ranks(H, self.traces[tier], self.basis_dicts[tier], conf["eb_base"], conf["target_rank"], lambda_eb, rho)
            
            t_eval = self.evaluate_tier_ranks(tier, ranks)
            tier_results[tier] = {
                "c_val": c_val,
                "diff": diff,
                "ranks": ranks,
                "sha": t_eval["sha"],
                "tot_rank": t_eval["tot_rank"],
                "gap_pct": t_eval["gap_pct"],
                "wiki_nll": t_eval["wiki_nll"],
                "c4_nll": t_eval["c4_nll"],
                "j_tier": t_eval["j_tier"],
                "wiki_per_win": t_eval["wiki_per_win"],
                "c4_per_win": t_eval["c4_per_win"]
            }
            
        j_strong = (tier_results["medium"]["j_tier"] + tier_results["aggressive"]["j_tier"] + tier_results["ultra"]["j_tier"]) / 3.0
        delta_j_strong = j_strong - self.f_current_j_strong
        delta_j_light = tier_results["light"]["j_tier"] - self.f_current_j["light"]
        light_guard_pass = (delta_j_light <= 0.0030)
        
        summary = {
            "name": candidate_name,
            "mode": mode,
            "weights": weights,
            "lambda": lambda_eb,
            "rho": rho,
            "eta": eta,
            "alpha_s": alpha_s,
            "alpha_r": alpha_r,
            "alpha_p": alpha_p,
            "j_strong": j_strong,
            "delta_j_strong": delta_j_strong,
            "delta_j_light": delta_j_light,
            "light_guard_pass": light_guard_pass,
            "tiers": tier_results
        }
        return summary

print("[FormulaSearchRunner] Module defined successfully.")
