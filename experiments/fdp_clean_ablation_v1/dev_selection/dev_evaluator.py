#!/usr/bin/env python3
"""In-memory fast development evaluator for Llama-3-8B low-rank configurations.
Evaluates NLL on DEV-WIKI (64 windows) and DEV-C4 (64 windows).
Strictly isolated: does not touch test sets.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

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

class DevEvaluator:
    def __init__(self, base_model_path: str, basis_pt_path: str, device: str = "cuda:0", dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.base_model_path = base_model_path
        
        print(f"[DevEvaluator] Loading dense base model weights from {base_model_path}...")
        self.dense_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=device
        )
        self.dense_model.eval()
        
        print(f"[DevEvaluator] Loading basis from {basis_pt_path}...")
        self.basis_dict = torch.load(basis_pt_path, map_location="cpu")
        
        self.config = AutoConfig.from_pretrained(base_model_path)
        self.num_heads = self.config.num_attention_heads
        self.num_kv = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.groups = self.num_heads // self.num_kv
        
        # Load DEV datasets
        dev_dir = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
        print(f"[DevEvaluator] Loading DEV datasets from {dev_dir}...")
        self.dev_wiki = torch.load(dev_dir / "dev_wiki_tokens.pt", map_location="cpu")
        self.dev_c4 = torch.load(dev_dir / "dev_c4_tokens.pt", map_location="cpu")
        print(f"[DevEvaluator] DEV-WIKI shape: {self.dev_wiki.shape}, DEV-C4 shape: {self.dev_c4.shape}")
        
        # Pre-cache dense weights on CPU/GPU to speed up dynamic projection
        self.dense_w_k = [self.dense_model.model.layers[l].self_attn.k_proj.weight.data.clone() for l in range(32)]
        self.dense_w_v = [self.dense_model.model.layers[l].self_attn.v_proj.weight.data.clone() for l in range(32)]
        self.dense_w_o = [self.dense_model.model.layers[l].self_attn.o_proj.weight.data.clone() for l in range(32)]
        
        self.cmp_model = None
        self.current_ranks = None

    def set_ranks(self, low_rank_config: List[List[int]]):
        """Materialize or update compressed model weights in-place with low_rank_config."""
        if self.cmp_model is None or self.current_ranks != low_rank_config:
            if self.cmp_model is not None:
                del self.cmp_model
                torch.cuda.empty_cache()
            
            self.cmp_model = LlamaForCausalLM_EigenAttn.from_pretrained(
                self.base_model_path,
                config=self.config,
                low_rank_config=low_rank_config,
                device_map=self.device,
                torch_dtype=self.dtype
            )
            self.cmp_model.eval()
            self.current_ranks = [list(r) for r in low_rank_config]

        # Project weights
        for li in range(32):
            cmp_attn = self.cmp_model.model.layers[li].self_attn
            rkq, rv = low_rank_config[li]
            k_col = rkq // self.num_kv
            v_col = rv // self.num_kv

            q_key = f"base_model.model.model.layers.{li}.self_attn.q_proj_basis"
            v_key = f"base_model.model.model.layers.{li}.self_attn.v_proj_basis"

            U_kq = self.basis_dict[q_key][:, :k_col].to(dtype=self.dtype, device=self.device)
            U_v_packed = self.basis_dict[v_key].to(dtype=self.dtype, device=self.device)

            org_w_k = self.dense_w_k[li].to(self.device)
            for h in range(self.num_kv):
                slice_w = org_w_k[h * self.head_dim : (h + 1) * self.head_dim, :]
                cmp_attn.k_proj_low.weight.data[h * k_col : (h + 1) * k_col, :] = torch.matmul(U_kq.t(), slice_w)
            cmp_attn.k_proj_up.weight.data = U_kq.contiguous()

            org_w_v = self.dense_w_v[li].to(self.device)
            for h in range(self.num_kv):
                u_v_head = U_v_packed[h * self.head_dim : (h + 1) * self.head_dim, h * v_col : (h + 1) * v_col]
                slice_v = org_w_v[h * self.head_dim : (h + 1) * self.head_dim, :]
                cmp_attn.v_proj_low.weight.data[h * v_col : (h + 1) * v_col, :] = torch.matmul(u_v_head.t(), slice_v)

            org_w_o = self.dense_w_o[li].to(self.device)
            for h in range(self.num_kv):
                u_v_head = U_v_packed[h * self.head_dim : (h + 1) * self.head_dim, h * v_col : (h + 1) * v_col]
                for g in range(self.groups):
                    idx = h * self.groups + g
                    w_o_slice = org_w_o[:, idx * self.head_dim : (idx + 1) * self.head_dim]
                    cmp_attn.o_proj_up.weight.data[:, idx * v_col : (idx + 1) * v_col] = torch.matmul(w_o_slice, u_v_head)

    @torch.no_grad()
    def evaluate_tokens(self, tokens_tensor: torch.Tensor) -> Tuple[float, List[float]]:
        """Compute average NLL and per-window NLL for a tensor of shape [N, seqlen]."""
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        nll_list = []
        n_windows = tokens_tensor.shape[0]
        
        for i in range(n_windows):
            input_ids = tokens_tensor[i : i + 1].to(self.device)
            target_ids = input_ids.clone()
            
            outputs = self.cmp_model(input_ids)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            window_nll = loss.mean().item()
            nll_list.append(window_nll)
            
        mean_nll = float(sum(nll_list) / len(nll_list))
        return mean_nll, nll_list

    def evaluate_ranks(self, low_rank_config: List[List[int]]) -> Dict[str, object]:
        self.set_ranks(low_rank_config)
        
        wiki_mean, wiki_per_win = self.evaluate_tokens(self.dev_wiki)
        c4_mean, c4_per_win = self.evaluate_tokens(self.dev_c4)
        
        j_dev = 0.5 * wiki_mean + 0.5 * c4_mean
        tot_rank = sum(k + v for k, v in low_rank_config)
        b_live = 4096 * tot_rank
        
        return {
            "B_live": b_live,
            "total_rank": tot_rank,
            "dev_wiki_nll": wiki_mean,
            "dev_wiki_ppl": math.exp(wiki_mean),
            "dev_c4_nll": c4_mean,
            "dev_c4_ppl": math.exp(c4_mean),
            "J_dev": j_dev,
            "wiki_per_window_nll": wiki_per_win,
            "c4_per_window_nll": c4_per_win,
        }
