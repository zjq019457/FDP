#!/usr/bin/env python3
"""
FORMAL EVALUATION RUNNER
========================
Dispatches and evaluates all audited configurations on formal test sets.
Records:
- Natural state metrics
- Matched state metrics
- Per-window NLLs for paired two-sided bootstrap
- Wiki PPL, C4 PPL
- 5-task zero-shot (PIQA, ARC-e, ARC-c, HellaSwag, WinoGrande) using lm_eval_primary authority
Strict one-way execution: no parameter updates allowed.
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
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR = WORKSPACE / "formal_ablation"
AUTHORITY_FILE = WORKSPACE / "authority" / "FDP_FORMULA_RESEARCH_AUTHORITY.json"
AUDIT_FILE = FORMAL_DIR / "FORMAL_RESOURCE_AUDIT.json"

UPSTREAM = Path("/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/eigenattention_upstream")
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

# RoPE compatibility
_orig_rope = LlamaRotaryEmbedding.forward
def _compat_rope(self, x, position_ids):
    if hasattr(self, "inv_freq") and torch.is_tensor(self.inv_freq):
        if self.inv_freq.device != position_ids.device:
            self.inv_freq = self.inv_freq.to(position_ids.device)
    return _orig_rope(self, x, position_ids)
LlamaRotaryEmbedding.forward = _compat_rope

from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
from transformers import LlamaForCausalLM

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"[{utc()}] STARTING FORMAL EVALUATION RUNNER...")

# Verify authority state
with open(AUTHORITY_FILE, "r") as f:
    auth = json.load(f)

if auth.get("status") != "FORMAL_RESOURCE_AUDIT_PASS":
    raise RuntimeError(f"FATAL: Authority status is NOT FORMAL_RESOURCE_AUDIT_PASS: {auth.get('status')}")

# Load audit table
with open(AUDIT_FILE, "r") as f:
    audit_data = json.load(f)

configs = audit_data["configurations"]
print(f"[{utc()}] Loaded {len(configs)} audited formal configurations.")

# We will save completed formal evaluations in FORMAL_DIR / "FORMAL_EVAL_PROGRESS.json"
PROGRESS_FILE = FORMAL_DIR / "FORMAL_EVAL_PROGRESS.json"
completed_evals = {}
if PROGRESS_FILE.exists():
    try:
        completed_evals = json.loads(PROGRESS_FILE.read_text())
    except Exception:
        completed_evals = {}

print(f"[{utc()}] Currently completed formal configurations: {len(completed_evals)} / {len(configs)}")

# Save runner state confirmation
auth["status"] = "FORMAL_EVALUATION_IN_PROGRESS"
auth["formal_eval_started_at"] = utc()
with open(AUTHORITY_FILE, "w") as f:
    json.dump(auth, f, indent=2)

print(f"[{utc()}] Authority updated to FORMAL_EVALUATION_IN_PROGRESS.")
