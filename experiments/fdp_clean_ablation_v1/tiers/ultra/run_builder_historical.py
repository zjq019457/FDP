#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

# Apply RoPE patch
_orig_rope_forward = LlamaRotaryEmbedding.forward

def _compat_rope_forward(self, x, position_ids):
    if hasattr(self, "inv_freq") and torch.is_tensor(self.inv_freq):
        if self.inv_freq.device != position_ids.device:
            self.inv_freq = self.inv_freq.to(position_ids.device)
    return _orig_rope_forward(self, x, position_ids)

LlamaRotaryEmbedding.forward = _compat_rope_forward

# Add repo to sys.path
sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn/tools")

import build_eigen_basis_llama_error_budget_offline as builder

if __name__ == "__main__":
    builder.main()
