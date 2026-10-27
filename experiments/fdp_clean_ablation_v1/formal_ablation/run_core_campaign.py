#!/usr/bin/env python3
"""
FORMAL DATA-FIRST CORE CAMPAIGN DISPATCHER & EVALUATOR
======================================================
Evaluates the 16 Core Rows:
- Dual reporting: Natural state and Matched-control state
- WikiText-2 formal PPL + per-window NLLs
- C4 formal PPL + per-window NLLs
- 5 Zero-Shot tasks (PIQA, ARC-e, ARC-c, HellaSwag, WinoGrande) using lm_eval_primary authority
- Saves full artifacts to formal_ablation/CORE_FORMAL_PER_WINDOW/
- Outputs:
  CORE_RESOURCE_AUTHORITY.csv
  CORE_FORMAL_RESULTS.csv
  CORE_ZERO_SHOT_RESULTS.csv
  CORE_RESOURCE_MATCHING.csv
  CORE_BOOTSTRAP_RESULTS.csv
  CORE_CAMPAIGN_STATUS.json
  CORE_FORMAL_REPORT.md
"""
import os
import sys
import json
import time
import math
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

import torch
import numpy as np

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"
PER_WIN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_BASE = "/diskD/jiaqizhang/Llama-3"
PY_BIN = "/home/bai-project/anaconda3/envs/eigen/bin/python"
PPL_SCRIPT = "/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/runners/ppl_instrumented.py"
ZS_SCRIPT = "/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/runners/zero_shot_instrumented.py"
WIKI_DIR = "/home/bai-project/jiaqizhang/wikitext2"
C4_TEXTS = "/diskF/jiaqizhang/fdp_eb_orth_paper_v1_results/offline_calib/c4/c4_ppl_selected_texts.jsonl"
ZS_YAML_DIR = "/diskF/jiaqizhang/a1804298693/offline_datasets"
CACHE_DIR = "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn/HF_cache"

UPSTREAM = Path("/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/eigenattention_upstream")
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
_orig_rope = LlamaRotaryEmbedding.forward
def _compat_rope(self, x, position_ids):
    if hasattr(self, "inv_freq") and torch.is_tensor(self.inv_freq):
        if self.inv_freq.device != position_ids.device:
            self.inv_freq = self.inv_freq.to(position_ids.device)
    return _orig_rope(self, x, position_ids)
LlamaRotaryEmbedding.forward = _compat_rope

from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
from transformers import LlamaForCausalLM, AutoConfig

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"[{utc()}] LOADING 16 CORE ROWS FROM CORE_RESOURCE_AUTHORITY.json...")
with open(FORMAL_DIR / "CORE_RESOURCE_AUTHORITY.json") as f:
    core_rows = json.load(f)

# Load existing progress if any
STATUS_FILE = FORMAL_DIR / "CORE_CAMPAIGN_STATUS.json"
status_dict = {}
if STATUS_FILE.exists():
    try:
        status_dict = json.loads(STATUS_FILE.read_text())
    except Exception:
        status_dict = {}

status_dict["total_rows"] = len(core_rows)
status_dict["last_updated"] = utc()
status_dict["stage"] = "CORE_FORMAL_DISPATCH_ACTIVE"

with open(STATUS_FILE, "w") as f:
    json.dump(status_dict, f, indent=2)

print(f"[{utc()}] CORE STATUS INITIALIZED: 16 ROWS READY.")
