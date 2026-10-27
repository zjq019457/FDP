#!/usr/bin/env python3
"""Phase 4: Development Search & Formula Selection Engine.
Evaluates candidate FDP formulas on DEV-WIKI and DEV-C4 under byte-matched resources.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
from realloc_rank_from_fragility import fragility_vector, minmax

from dev_evaluator import DevEvaluator

DEV_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
RAW_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles")
AUTH_PATH = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/authority/FDP_HIST_AUTHORITY.json")

TIERS = {
    "light": {"base_eb": 0.045, "step1_file": "step1_llama3_8b_eb045.json"},
    "medium": {"base_eb": 0.060, "step1_file": "step1_llama3_8b_eb060.json"},
    "aggressive": {"base_eb": 0.090, "step1_file": "step1_llama3_8b_eb090.json"},
    "ultra": {"base_eb": 0.120, "step1_file": "step1_llama3_8b_eb120.json"},
}

def derive_schedule(
    step1_data: dict,
    base_eb: float,
    weights: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
    lambda_eb: float = 1.2,
    rho: float = 1.0,
    c_scale: float = 1.0,
) -> List[float]:
    wz, ws, wv, wa, wh = weights
    per_layer = step1_data["per_layer"]
    layers = sorted(int(k) for k in per_layer.keys())

    s_raw, _ = fragility_vector(per_layer, layers, wz, ws, wv, wa, wh, False)
    s_tilde = minmax(s_raw)

    eb_inv = [base_eb / (1.0 + lambda_eb * st) for st in s_tilde]
    m = sum(eb_inv) / len(eb_inv)
    scale = (base_eb / m) if m > 0 else 1.0
    eb_norm = [e * scale for e in eb_inv]

    # Layer 0 blending with rho
    # log eps0(rho) = rho * log(base_eb) + (1 - rho) * log(eb_norm[0])
    eps0 = math.exp(rho * math.log(base_eb) + (1.0 - rho) * math.log(eb_norm[0]))
    eb_norm[0] = eps0

    # Apply scalar c to non-frozen layers (or all layers if rho < 1)
    eb_scaled = [e * c_scale for e in eb_norm]
    if rho >= 0.999:
        eb_scaled[0] = base_eb
    return eb_scaled

print("[dev_formula_search] Helper functions defined successfully.")
