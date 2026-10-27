#!/usr/bin/env python3
"""
PARALLEL BATCH EVALUATION OF FRESH CORE PHYSICAL STATES
======================================================
Evaluates the 8 fresh physical rank states across safe GPUs:
- Tier LIGHT states: GPU 3
- Tier AGGRESSIVE states: GPU 3 / GPU 0
- Tier ULTRA states: GPU 3 / GPU 2

For each state:
1. In-memory fast materialize & project weights with RoPE patch
2. Full WikiText-2 PPL (141 windows, save per-window NLLs)
3. Full C4 PPL (256 windows, save per-window NLLs)
4. 5-task Zero-Shot (PIQA, ARC-e, ARC-c, HellaSwag, WinoGrande) using lm_eval_primary authority
5. Atomic save into formal_ablation/CORE_FORMAL_PER_WINDOW/{sha}/
"""
import os
import sys
import json
import time
import math
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"
PER_WIN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_BASE = "/diskD/jiaqizhang/Llama-3"
UPSTREAM = Path("/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/eigenattention_upstream")
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

_orig_rope = LlamaRotaryEmbedding.forward
def _compat_rope(self, x, position_ids):
    if hasattr(self, "inv_freq") and torch.is_tensor(self.inv_freq):
        if self.inv_freq.device != position_ids.device:
            self.inv_freq = self.inv_freq.to(position_ids.device)
    return _orig_rope(self, x, position_ids)
LlamaRotaryEmbedding.forward = _compat_rope

from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"[{utc()}] STARTING PARALLEL FRESH STATE RUNNER...")
