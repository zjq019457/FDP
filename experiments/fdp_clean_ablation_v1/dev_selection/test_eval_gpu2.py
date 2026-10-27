#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
from dev_evaluator import DevEvaluator
from formula_engine import TIER_CONF

print("Initializing DevEvaluator on cuda:2 for medium tier...")
t0 = time.time()
conf = TIER_CONF["medium"]
evaluator = DevEvaluator(
    base_model_path="/diskD/jiaqizhang/Llama-3",
    basis_pt_path=conf["basis"],
    device="cuda:2",
    dtype=torch.bfloat16
)
print(f"Loaded in {time.time()-t0:.1f}s")

# Load historical medium ranks
with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/authority/FDP_HIST_AUTHORITY.json") as f:
    auth = json.load(f)
ranks = auth["historical_ranks"]["medium"]["ranks"]

print("Running test evaluation on DEV (64 wiki + 64 c4)...")
t0 = time.time()
res = evaluator.evaluate_ranks(ranks)
elapsed = time.time() - t0

print(f"Evaluation completed in {elapsed:.1f}s!")
print(f"DEV-WIKI NLL: {res['dev_wiki_nll']:.4f} (PPL: {res['dev_wiki_ppl']:.4f})")
print(f"DEV-C4   NLL: {res['dev_c4_nll']:.4f} (PPL: {res['dev_c4_ppl']:.4f})")
print(f"J_dev:        {res['J_dev']:.6f}")
