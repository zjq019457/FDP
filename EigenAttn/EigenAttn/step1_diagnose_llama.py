#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-1 diagnostic for Llama (dense) vs Llama EigenAttn CMP, same *metric definitions*
as step1_diagnose.py (OPT) so downstream fragility tools keep a single formula:

  per_layer_eb_from_step1.py  +  realloc_rank_from_fragility.py
  expect keys: Z_rel_mse, score_rel_mse_causal, value_only_Z_rel_mse,
  score_only_AV_rel_mse, entropy_diff_A

Requirements:
  * Base model: standard Llama / Llama-3 with HF eager attention.
  * CMP: models.modelling_llama_eigen_attn.LlamaForCausalLM_EigenAttn, attn_implementation eager.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, Optional

import torch
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from step1_diagnose import (
    DEFAULT_PROMPTS,
    build_inputs,
    shannon_entropy,
    unpatch,
)


def cpu_fp32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float32)


def _causal_valid_mask(scores: torch.Tensor) -> torch.Tensor:
    """Lower-triangular valid region for self-attn scores [B,H,T,S]."""
    T_q, S_k = scores.shape[-2], scores.shape[-1]
    causal = torch.ones(T_q, S_k, dtype=torch.bool, device=scores.device).tril(diagonal=S_k - T_q)
    causal = causal & torch.isfinite(scores)
    return causal


def make_full_diag_forward_llama(self_attn, store: Dict[int, dict], layer_idx: int):
    """Record dense LlamaAttention tensors (same layout as OPT step1 where applicable)."""

    def diag_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        m = self_attn
        bsz, q_len, _ = hidden_states.size()

        query_states = m.q_proj(hidden_states)
        key_states = m.k_proj(hidden_states)
        value_states = m.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, m.num_heads, m.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = m.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, m.layer_idx, cache_kwargs)

        from transformers.models.llama.modeling_llama import repeat_kv

        key_states = repeat_kv(key_states, m.num_key_value_groups)
        value_states = repeat_kv(value_states, m.num_key_value_groups)

        attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(m.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_logits = attn_logits + causal_mask

        scores_full = attn_logits.detach().view(bsz, m.num_heads, q_len, key_states.shape[-2])

        attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=m.attention_dropout, training=m.training)
        attn_output = torch.matmul(attn_weights, value_states)

        probs_full = attn_weights.detach().view(bsz, m.num_heads, q_len, value_states.shape[-2])

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        AV_full = attn_output.detach()
        attn_output = m.o_proj(attn_output)
        Z_full = attn_output

        store[layer_idx] = {
            "scores": cpu_fp32(scores_full).half(),
            "probs": cpu_fp32(probs_full).half(),
            "V": cpu_fp32(value_states.detach()).half(),
            "AV": cpu_fp32(AV_full).half(),
            "Z": cpu_fp32(Z_full.detach()).half(),
        }

        if not output_attentions:
            attn_weights = None
        return Z_full, attn_weights, past_key_value

    return diag_forward


def make_cmp_diag_forward_llama(
    self_attn,
    cmp_store: Dict[int, dict],
    full_store: Dict[int, dict],
    metrics: Dict[int, dict],
    layer_idx: int,
):
    def diag_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        m = self_attn
        bsz, q_len, _ = hidden_states.size()

        query_states = m.q_proj(hidden_states)
        key_states_low = m.k_proj_low(hidden_states)
        value_states_low = m.v_proj_low(hidden_states)

        query_states = query_states.view(bsz, q_len, m.num_heads, m.head_dim).transpose(1, 2)
        key_states_low = key_states_low.view(
            bsz, q_len, m.num_key_value_heads, m.rank_kq // m.num_key_value_heads
        ).transpose(1, 2)
        value_states_low = value_states_low.view(
            bsz, q_len, m.num_key_value_heads, m.rank_v // m.num_key_value_heads
        ).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = m.rotary_emb(value_states_low, position_ids)
        else:
            cos, sin = position_embeddings
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

        past_key_value = getattr(m, "past_key_value", past_key_value)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states_low, value_states_low = past_key_value.update(
                key_states_low, value_states_low, m.layer_idx, cache_kwargs
            )

        key_states = m.k_proj_up(key_states_low)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, m.num_key_value_groups)
        value_states_low = repeat_kv(value_states_low, m.num_key_value_groups)

        attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(m.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_logits = attn_logits + causal_mask

        scores_cmp = attn_logits.detach().view(bsz, m.num_heads, q_len, key_states.shape[-2])

        attn_weights = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=m.attention_dropout, training=m.training)
        attn_output = torch.matmul(attn_weights, value_states_low)

        probs_cmp = attn_weights.detach().view(bsz, m.num_heads, q_len, value_states_low.shape[-2])

        # GQA: per-head value width is rank_v // num_kv_heads (see LlamaEigenAttention + o_proj_up in_edges).
        head_dim_v = m.rank_v // m.num_key_value_heads
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, m.num_heads * head_dim_v)
        Z_cmp = m.o_proj_up(attn_output)

        full = full_store[layer_idx]
        Z_full = full["Z"].to(device=Z_cmp.device, dtype=torch.float32)
        scores_full = full["scores"].to(device=Z_cmp.device, dtype=torch.float32)
        probs_full = full["probs"].to(device=Z_cmp.device, dtype=torch.float32)
        AV_full = full["AV"].to(device=Z_cmp.device, dtype=torch.float32)
        V_full = full["V"].to(device=Z_cmp.device, dtype=torch.float32)

        Z_cmp_f = Z_cmp.detach().float()
        scores_cmp_f = scores_cmp.float()
        probs_cmp_f = probs_cmp.float()

        std_full = Z_full.std().item()
        std_cmp = Z_cmp_f.std().item()
        std_ratio = std_cmp / max(std_full, 1e-12)

        norm_full_t = Z_full.norm(dim=-1)
        norm_cmp_t = Z_cmp_f.norm(dim=-1)
        norm_ratio = (norm_cmp_t / norm_full_t.clamp_min(1e-8)).mean().item()

        ent_full = shannon_entropy(probs_full)
        ent_cmp = shannon_entropy(probs_cmp_f)
        ent_diff = (ent_cmp - ent_full).mean().item()
        ent_full_mean = ent_full.mean().item()
        ent_cmp_mean = ent_cmp.mean().item()

        causal = (
            _causal_valid_mask(scores_full)
            & _causal_valid_mask(scores_cmp_f)
            & (scores_full > -1e4)
            & (scores_cmp_f > -1e4)
        )
        nvalid = causal.sum().clamp_min(1).item()
        score_diff_sq = (scores_cmp_f - scores_full).pow(2)
        score_diff_sq = torch.where(causal, score_diff_sq, torch.zeros_like(score_diff_sq))
        scores_full_sq = torch.where(causal, scores_full.pow(2), torch.zeros_like(scores_full))
        score_mse = (score_diff_sq.sum() / nvalid).item()
        scores_full_var = (scores_full_sq.sum() / nvalid).item()
        score_rel_mse = score_mse / max(scores_full_var, 1e-12)
        score_mean_full = (torch.where(causal, scores_full, torch.zeros_like(scores_full)).sum() / nvalid).item()
        score_mean_cmp = (torch.where(causal, scores_cmp_f, torch.zeros_like(scores_cmp_f)).sum() / nvalid).item()
        scores_full_centered_sq = (scores_full - score_mean_full).pow(2)
        scores_full_centered_sq = torch.where(causal, scores_full_centered_sq, torch.zeros_like(scores_full_centered_sq))
        scores_full_std_valid = (scores_full_centered_sq.sum() / nvalid).sqrt().item()
        scores_cmp_centered_sq = (scores_cmp_f - score_mean_cmp).pow(2)
        scores_cmp_centered_sq = torch.where(causal, scores_cmp_centered_sq, torch.zeros_like(scores_cmp_centered_sq))
        scores_cmp_std_valid = (scores_cmp_centered_sq.sum() / nvalid).sqrt().item()

        Z_diff_sq = (Z_cmp_f - Z_full).pow(2).mean().item()
        Z_full_var = Z_full.pow(2).mean().item()
        Z_rel_mse = Z_diff_sq / max(Z_full_var, 1e-12)

        tgt_len = q_len
        src_len = key_states.shape[-2]
        bsz_h = bsz * m.num_heads
        head_dim_v = m.rank_v // m.num_key_value_heads
        # [B,H,S,Dv] -> [B*H,S,Dv] for matmul: [B*H,T,S] @ [B*H,S,Dv]
        value_states_flat = value_states_low.reshape(bsz_h, src_len, head_dim_v)
        probs_full_flat = probs_full.reshape(bsz_h, tgt_len, src_len)
        ow = m.o_proj_up.weight

        with torch.no_grad():
            probs_full_flat_dt = probs_full_flat.to(dtype=value_states_flat.dtype)
            ao = torch.matmul(probs_full_flat_dt, value_states_flat)
            ao = ao.view(bsz, m.num_heads, tgt_len, head_dim_v).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            ao = ao.to(device=ow.device, dtype=ow.dtype)
            Z_oracle_A = m.o_proj_up(ao).float()
            value_only_mse = (Z_oracle_A - Z_full).pow(2).mean().item()
            value_only_rel_mse = value_only_mse / max(Z_full_var, 1e-12)

        with torch.no_grad():
            V_full_flat = V_full.reshape(bsz * m.num_heads, src_len, V_full.size(-1))
            probs_cmp_flat = probs_cmp_f.view(bsz * m.num_heads, tgt_len, src_len)
            AV_with_cmp_A = torch.matmul(probs_cmp_flat, V_full_flat)
            AV_with_cmp_A = (
                AV_with_cmp_A.view(bsz, m.num_heads, tgt_len, V_full.size(-1))
                .transpose(1, 2)
                .contiguous()
                .view(bsz, q_len, -1)
            )
            AV_full_var = AV_full.pow(2).mean().item()
            score_only_mse = (AV_with_cmp_A - AV_full).pow(2).mean().item()
            score_only_rel_mse = score_only_mse / max(AV_full_var, 1e-12)

        metrics[layer_idx] = {
            "rank_kq": int(m.rank_kq),
            "rank_v": int(m.rank_v),
            "std_full": std_full,
            "std_cmp": std_cmp,
            "std_ratio_Z": std_ratio,
            "norm_ratio_Z": norm_ratio,
            "entropy_full_mean": ent_full_mean,
            "entropy_cmp_mean": ent_cmp_mean,
            "entropy_diff_A": ent_diff,
            "score_mse_causal": score_mse,
            "score_rel_mse_causal": score_rel_mse,
            "score_std_full_causal": scores_full_std_valid,
            "score_std_cmp_causal": scores_cmp_std_valid,
            "score_std_ratio_causal": scores_cmp_std_valid / max(scores_full_std_valid, 1e-12),
            "Z_mse": Z_diff_sq,
            "Z_rel_mse": Z_rel_mse,
            "value_only_Z_rel_mse": value_only_rel_mse,
            "score_only_AV_rel_mse": score_only_rel_mse,
        }

        cmp_store[layer_idx] = {"rank_kq": int(m.rank_kq), "rank_v": int(m.rank_v)}
        for k in list(full.keys()):
            full[k] = None
        del full_store[layer_idx]

        attn_out = None if not output_attentions else attn_weights
        return Z_cmp, attn_out, past_key_value

    return diag_forward


def patch_full_llama(model, store):
    saved = []
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        saved.append((attn, attn.forward))
        attn.forward = make_full_diag_forward_llama(attn, store, li)
    return saved


def patch_cmp_llama(cmp, cmp_store, full_store, metrics):
    saved = []
    for li, layer in enumerate(cmp.model.layers):
        attn = layer.self_attn
        saved.append((attn, attn.forward))
        attn.forward = make_cmp_diag_forward_llama(attn, cmp_store, full_store, metrics, li)
    return saved


def _resolve_low_rank_llama(cmp_dir: str, low_json: Optional[str]):
    low_rank_config = None
    cfg_p = os.path.join(cmp_dir, "config.json")
    if os.path.isfile(cfg_p):
        with open(cfg_p) as f:
            cfgj = json.load(f)
        if isinstance(cfgj.get("low_rank_config"), list):
            low_rank_config = cfgj["low_rank_config"]
    if low_rank_config is None:
        low_p = os.path.join(cmp_dir, "low_rank_config.json")
        if os.path.isfile(low_p):
            low_rank_config = json.load(open(low_p))
    if low_rank_config is None and low_json:
        low_rank_config = json.load(open(low_json))
    if low_rank_config is None:
        from safetensors.torch import safe_open

        idx = json.load(open(os.path.join(cmp_dir, "model.safetensors.index.json")))
        wmap = idx["weight_map"]
        files = sorted(set(wmap[k] for k in wmap if "proj_low.weight" in k))
        shapes: Dict[str, list] = {}
        for fn in files:
            with safe_open(os.path.join(cmp_dir, fn), framework="pt") as f:
                for k in f.keys():
                    if "proj_low.weight" in k:
                        shapes[k] = list(f.get_slice(k).get_shape())
        layer_to: Dict[int, Dict[str, int]] = {}
        for k, sh in shapes.items():
            mm = re.match(r"model\.layers\.(\d+)\.self_attn\.([kv])_proj_low\.weight", k)
            if mm:
                li = int(mm.group(1))
                kind = mm.group(2)
                layer_to.setdefault(li, {})[kind] = sh[0]
        low_rank_config = []
        for li in sorted(layer_to.keys()):
            d = layer_to[li]
            if "k" not in d or "v" not in d:
                continue
            low_rank_config.append([d["k"], d["v"]])
        if not low_rank_config:
            return None
        print("[Phase 2] Inferred low_rank_config from safetensors (k/v proj_low).", flush=True)
    return low_rank_config


def run_full_phase_llama(args, prompts):
    print("[Phase 1] Loading FULL Llama model:", args.base, flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    if args.device == "cpu":
        base = AutoModelForCausalLM.from_pretrained(
            args.base,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation,
            device_map={"": "cpu"},
        )
    elif args.device_map_auto:
        base = AutoModelForCausalLM.from_pretrained(
            args.base,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation,
            device_map="auto",
            max_memory=args.max_memory,
        )
    else:
        dev = torch.device(f"cuda:{int(args.gpu)}")
        base = AutoModelForCausalLM.from_pretrained(
            args.base,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation,
            device_map={"": dev},
        )
    base.eval()
    base.config.use_cache = False

    full_store: Dict[int, dict] = {}
    saved = patch_full_llama(base, full_store)
    try:
        input_ids = build_inputs(tok, prompts, args.seqlen, next(base.parameters()).device)
        with torch.no_grad():
            base.model(input_ids)
    finally:
        unpatch(saved)

    print(f"[Phase 1] Captured {len(full_store)} layers from FULL.", flush=True)
    for li, d in full_store.items():
        for k, v in list(d.items()):
            if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                d[k] = v.detach().to("cpu", non_blocking=True)

    del base
    torch.cuda.empty_cache()
    return tok, full_store, input_ids.detach().cpu()


def run_cmp_phase_llama(args, tok, full_store, input_ids_cpu):
    print("[Phase 2] Loading CMP Llama EigenAttn:", args.cmp, flush=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn

    low_rank_config = _resolve_low_rank_llama(args.cmp, args.low_json)
    if low_rank_config is None:
        raise SystemExit("Could not resolve low_rank_config; pass --low_json")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    if args.device == "cpu":
        cmp = LlamaForCausalLM_EigenAttn.from_pretrained(
            args.cmp,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
            low_rank_config=low_rank_config,
        )
    elif args.device_map_auto:
        cmp = LlamaForCausalLM_EigenAttn.from_pretrained(
            args.cmp,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=args.max_memory,
            low_rank_config=low_rank_config,
        )
    else:
        dev = torch.device(f"cuda:{int(args.gpu)}")
        cmp = LlamaForCausalLM_EigenAttn.from_pretrained(
            args.cmp,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": dev},
            low_rank_config=low_rank_config,
        )
    cmp.eval()
    cmp.config.use_cache = False

    cmp_store: Dict[int, dict] = {}
    metrics: Dict[int, dict] = {}
    saved = patch_cmp_llama(cmp, cmp_store, full_store, metrics)
    try:
        input_ids = input_ids_cpu.to(next(cmp.parameters()).device)
        with torch.no_grad():
            cmp.model(input_ids)
    finally:
        unpatch(saved)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Dense Llama checkpoint dir")
    ap.add_argument("--cmp", required=True, help="Llama EigenAttn final_model dir")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--low_json", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--num_seqs", type=int, default=2)
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa"],
        help="Must match how base/CMP were trained; EigenAttn CMP is eager-only.",
    )
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Single-GPU index for device_map (default: 0). With CUDA_VISIBLE_DEVICES=3 use 0.",
    )
    ap.add_argument(
        "--device_map_auto",
        action="store_true",
        help="Use accelerate device_map=auto + --max_memory_gb (splits layers; often breaks this diagnostic).",
    )
    ap.add_argument(
        "--max_memory_gb",
        default="0:60",
        help="Only used with --device_map_auto: GPU:GiB,... caps",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.prompts_file and os.path.isfile(args.prompts_file):
        with open(args.prompts_file) as f:
            pl = [ln.strip() for ln in f if ln.strip()]
        prompts = pl[: args.num_seqs] if pl else DEFAULT_PROMPTS[: args.num_seqs]
    else:
        prompts = DEFAULT_PROMPTS[: args.num_seqs]
    if len(prompts) < args.num_seqs:
        prompts = (prompts * args.num_seqs)[: args.num_seqs]

    mm = {}
    for tok in args.max_memory_gb.split(","):
        if not tok.strip():
            continue
        gpu, gb = tok.split(":")
        mm[int(gpu)] = f"{int(gb)}GiB"
    args.max_memory = mm if mm else None

    if args.attn_implementation != "eager":
        print("[warn] CMP Llama EigenAttn is eager-only; forcing base to eager for aligned numerics.", flush=True)
        args.attn_implementation = "eager"

    tok, full_store, input_ids_cpu = run_full_phase_llama(args, prompts)
    metrics = run_cmp_phase_llama(args, tok, full_store, input_ids_cpu)

    out = {
        "args": {k: v for k, v in vars(args).items() if k not in ("max_memory",)},
        "per_layer": {str(li): metrics[li] for li in sorted(metrics.keys())},
        "summary": {},
    }
    if metrics:
        keys = [
            "std_ratio_Z",
            "norm_ratio_Z",
            "entropy_diff_A",
            "score_rel_mse_causal",
            "score_std_ratio_causal",
            "Z_rel_mse",
            "value_only_Z_rel_mse",
            "score_only_AV_rel_mse",
        ]
        for k in keys:
            vals = [metrics[li][k] for li in sorted(metrics.keys()) if k in metrics[li]]
            if vals:
                vals_t = torch.tensor(vals, dtype=torch.float64)
                out["summary"][k] = {
                    "mean": vals_t.mean().item(),
                    "median": vals_t.median().item(),
                    "min": vals_t.min().item(),
                    "max": vals_t.max().item(),
                }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[Done] wrote metrics for {len(metrics)} layers -> {args.out}")

    print()
    print(
        f"{'L':>3} {'rkq':>5} {'rv':>5}  {'std_r':>7}  {'nrm_r':>7}  "
        f"{'dH(A)':>8}  {'sc_rel':>9}  {'sc_str':>7}  {'Z_rel':>9}  {'V|A_rel':>9}  {'A|V_rel':>9}"
    )
    for li in sorted(metrics.keys()):
        m = metrics[li]
        print(
            f"{li:>3} {m['rank_kq']:>5} {m['rank_v']:>5}  "
            f"{m['std_ratio_Z']:>7.3f}  {m['norm_ratio_Z']:>7.3f}  "
            f"{m['entropy_diff_A']:>+8.3f}  "
            f"{m['score_rel_mse_causal']:>9.3e}  "
            f"{m['score_std_ratio_causal']:>7.3f}  "
            f"{m['Z_rel_mse']:>9.3e}  "
            f"{m['value_only_Z_rel_mse']:>9.3e}  {m['score_only_AV_rel_mse']:>9.3e}"
        )


if __name__ == "__main__":
    main()
