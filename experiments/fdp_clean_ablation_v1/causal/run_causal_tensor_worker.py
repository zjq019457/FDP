#!/usr/bin/env python3
"""
Compute component-level causal EB intervention tensor G[j, k <- l](e)
for target source layer l:
  intervene by changing epsilon_l to e + delta and e - delta.
  keep all non-target layers at tier base EB e.
  for each functional component j (d1..d5) and measured layer k >= l:
    G[j, k <- l](e) = (d[j, k](e + delta) - d[j, k](e - delta)) / (2 * delta)
  for k < l:
    G[j, k <- l](e) = 0.0 (strictly causal)
"""
import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, "/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/eigenattention_upstream")
sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from dev_evaluator import DevEvaluator, _compat_rope
from formula_engine import TIER_CONF, load_traces, oracle_lookup
from step1_diagnose import DEFAULT_PROMPTS, build_inputs, unpatch
from step1_diagnose_llama import patch_full_llama, make_cmp_diag_forward_llama

OUT_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/causal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENTS = ["d1_z", "d2_s", "d3_v", "d4_a", "d5_h"]
TIER_NAMES = ["light", "medium", "aggressive", "ultra"]
DELTA = 0.015

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

def extract_diagnostics(metrics_dict: Dict[int, dict]) -> np.ndarray:
    """Extract 5 components across 32 layers: shape [5, 32]."""
    d = np.zeros((5, 32), dtype=np.float64)
    for li in range(32):
        m = metrics_dict[li]
        d[0, li] = m["Z_rel_mse"]                      # d1: delta_z
        d[1, li] = m["score_rel_mse_causal"]           # d2: delta_score
        d[2, li] = m["value_only_Z_rel_mse"]           # d3: delta_v
        d[3, li] = m["score_only_AV_rel_mse"]          # d4: attn_entropy_distortion
        d[4, li] = max(0.0, -m["entropy_diff_A"])      # d5: entropy_collapse_penalty
    return d

def run_diagnostic_forward(cmp_model, full_store, input_ids):
    cmp_store = {}
    metrics = {}
    saved = []
    for li, layer in enumerate(cmp_model.model.layers):
        attn = layer.self_attn
        saved.append((attn, attn.forward))
        attn.forward = make_cmp_diag_forward_llama(attn, cmp_store, full_store, metrics, li)
    try:
        with torch.no_grad():
            cmp_model.model(input_ids)
    finally:
        unpatch(saved)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", type=str, required=True, help="Comma-separated tiers, e.g. 'medium,aggressive'")
    parser.add_argument("--gpu", type=int, required=True, help="CUDA device index, e.g. 2 or 3")
    args = parser.parse_args()

    tiers_to_run = [t.strip() for t in args.tiers.split(",")]
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"================================================================================")
    print(f"CAUSAL EB INTERVENTION TENSOR WORKER")
    print(f"Device: {device} | Tiers: {tiers_to_run} | Delta: {DELTA}")
    print(f"================================================================================")

    # 1. Capture base dense model full_store once
    print(f"[{time.strftime('%H:%M:%S')}] Loading tokenizer & base dense model...")
    tok = AutoTokenizer.from_pretrained("/diskD/jiaqizhang/Llama-3", local_files_only=True, use_fast=False)
    input_ids = build_inputs(tok, DEFAULT_PROMPTS[:2], 512, device)

    base = AutoModelForCausalLM.from_pretrained(
        "/diskD/jiaqizhang/Llama-3",
        local_files_only=True,
        torch_dtype=dtype,
        device_map=device
    )
    base.eval()
    base.config.use_cache = False

    print(f"[{time.strftime('%H:%M:%S')}] Running Phase 1 dense baseline...")
    full_store = {}
    saved = patch_full_llama(base, full_store)
    try:
        with torch.no_grad():
            base.model(input_ids)
    finally:
        unpatch(saved)

    print(f"[{time.strftime('%H:%M:%S')}] Captured {len(full_store)} layers from dense baseline.")
    for li, d in full_store.items():
        for k, v in list(d.items()):
            if isinstance(v, torch.Tensor):
                d[k] = v.detach().to("cpu")

    del base
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Iterate through assigned tiers
    worker_results = {}
    for tier in tiers_to_run:
        conf = TIER_CONF[tier]
        eb_base = conf["eb_base"]
        print(f"\n==================================================")
        print(f"TIER: {tier.upper()} (eb_base={eb_base:.3f}, delta={DELTA:.3f})")
        print(f"==================================================")

        t_tier_start = time.time()
        evaluator = DevEvaluator(
            base_model_path="/diskD/jiaqizhang/Llama-3",
            basis_pt_path=conf["basis"],
            device=device,
            dtype=dtype
        )
        traces = load_traces(conf["report"])

        # Base Uniform rank configuration
        base_ranks = []
        for li in range(32):
            rk, rv = oracle_lookup(traces[li], eb_base)
            base_ranks.append([rk, rv])
        base_ranks_safe = clip_ranks_to_basis(base_ranks, evaluator.basis_dict)

        # Baseline diagnostics at base EB
        evaluator.set_ranks(base_ranks_safe)
        fs_copy = {li: {k: v for k, v in full_store[li].items()} for li in full_store}
        m_base = run_diagnostic_forward(evaluator.cmp_model, fs_copy, input_ids)
        d_base = extract_diagnostics(m_base) # shape [5, 32]
        print(f"  Uniform base pass complete. L0 d1={d_base[0,0]:.4e}, L31 d1={d_base[0,31]:.4e}")

        # G tensor for this tier: shape [5, 32, 32] -> G[j, k (measured), l (source)]
        G_tier = np.zeros((5, 32, 32), dtype=np.float64)

        for l in range(32):
            t_l0 = time.time()
            eb_plus = eb_base + DELTA
            eb_minus = max(0.005, eb_base - DELTA)

            # Plus intervention
            ranks_plus = [list(r) for r in base_ranks_safe]
            rk_p, rv_p = oracle_lookup(traces[l], eb_plus)
            qk = f"base_model.model.model.layers.{l}.self_attn.q_proj_basis"
            vk = f"base_model.model.model.layers.{l}.self_attn.v_proj_basis"
            q_max = evaluator.basis_dict[qk].shape[1] * 8
            v_max = evaluator.basis_dict[vk].shape[1]
            ranks_plus[l] = [min(rk_p, q_max), min(rv_p, v_max)]

            evaluator.set_ranks(ranks_plus)
            fs_copy = {li: {k: v for k, v in full_store[li].items()} for li in full_store}
            m_plus = run_diagnostic_forward(evaluator.cmp_model, fs_copy, input_ids)
            d_plus = extract_diagnostics(m_plus) # [5, 32]

            # Minus intervention
            ranks_minus = [list(r) for r in base_ranks_safe]
            rk_m, rv_m = oracle_lookup(traces[l], eb_minus)
            ranks_minus[l] = [min(rk_m, q_max), min(rv_m, v_max)]

            evaluator.set_ranks(ranks_minus)
            fs_copy = {li: {k: v for k, v in full_store[li].items()} for li in full_store}
            m_minus = run_diagnostic_forward(evaluator.cmp_model, fs_copy, input_ids)
            d_minus = extract_diagnostics(m_minus) # [5, 32]

            actual_delta_eb = (eb_plus - eb_minus)
            # Derivative for k >= l
            for k in range(l, 32):
                G_tier[:, k, l] = (d_plus[:, k] - d_minus[:, k]) / actual_delta_eb
            # For k < l: strictly causal 0.0
            for k in range(0, l):
                G_tier[:, k, l] = 0.0

            sec_l = time.time() - t_l0
            # Diagnostics for display
            self_resp = G_tier[:, l, l].mean()
            down_resp = G_tier[:, (l+1):, l].mean() if l < 31 else 0.0
            print(f"  Layer {l:2d}/31 | Ranks: [-]={ranks_minus[l]} [+]={ranks_plus[l]} | Time: {sec_l:.2f}s | Self: {self_resp:+.4e} | Downstream: {down_resp:+.4e}")

        tier_sec = time.time() - t_tier_start
        print(f"Tier {tier.upper()} completed in {tier_sec:.1f}s.")
        worker_results[tier] = {
            "G_tier": G_tier.tolist(),
            "d_base": d_base.tolist(),
            "eb_base": eb_base,
            "delta": DELTA
        }

        del evaluator
        torch.cuda.empty_cache()
        gc.collect()

    # Save worker results
    out_file = OUT_DIR / f"causal_worker_results_gpu{args.gpu}.json"
    with open(out_file, "w") as f:
        json.dump(worker_results, f)
    print(f"\nWorker completed! Saved results to {out_file}")

if __name__ == "__main__":
    main()
