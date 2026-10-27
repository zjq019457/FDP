"""Step-1 fragility diagnostic for OPT / Llama / MPT, using **forward-replace**.

Direct port of ``EigenAttn/EigenAttn/step1_diagnose.py`` (OPT) and
``step1_diagnose_llama.py`` (Llama), with all v3-only fields removed
(``q_dora`` / ``k_dora`` / ``v_dora`` / ``balance_*_core`` / ``balance_*_col`` /
``vscale_log`` / ``v_rot``) — the EigenAttn modules in ``EigenAttn_base`` do not
have any of those, so the simplified forward = the standard ``_base`` EigenAttn
forward written out inline. MPT is implemented in the same style here (no
legacy reference existed); the schema is identical.

Per-layer schema (consumed by ``derive_eb.derive_per_layer_eb``):

  Z_rel_mse              = mean((Z_cmp - Z_full)^2) / mean(Z_full^2)
  score_rel_mse_causal   = causal-tri rel MSE of pre-softmax logits
  value_only_Z_rel_mse   = oracle-A: keep CMP V/out_proj_up, use FULL probs.
                           rel MSE between Z_oracle_A and Z_full.
  score_only_AV_rel_mse  = oracle-V: keep FULL V/out_proj, use CMP probs.
                           rel MSE between AV_with_cmp_A and AV_full
                           (in FULL embed_dim space; no out_proj of cmp involved).
  entropy_diff_A         = mean(H(probs_cmp) - H(probs_full)) over all heads/tokens.

Plus diagnostic extras kept verbatim from legacy: ``rank_kq``, ``rank_v``,
``std_full``, ``std_cmp``, ``std_ratio_Z``, ``norm_ratio_Z``, ``entropy_full_mean``,
``entropy_cmp_mean``, ``score_mse_causal``, ``score_std_full_causal``,
``score_std_cmp_causal``, ``score_std_ratio_causal``, ``Z_mse``.

Two-phase flow (memory-friendly):
  Phase 1: install hooks on BASE (dense) attention -> capture (scores/probs/V/AV/Z)
           per layer to CPU/fp16; run model forward; un-install.
  Phase 2: install hooks on CMP (EigenAttn) attention -> for each layer compute
           the 5 metrics on-the-fly, drop FULL captures right after consumption.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


DEFAULT_PROMPTS: List[str] = [
    "The history of the Roman Empire spans more than a thousand years and includes many "
    "significant transitions of power, ideology, and culture. From the early Republic, "
    "through the rise of Augustus and the Pax Romana, to the eventual fall of the Western "
    "Empire, scholars have analyzed each era to understand the long arc of European "
    "civilization. Modern research often combines archaeology with newer methods such as "
    "genetic analysis and climate reconstruction.",
    "Quantum mechanics describes nature at the smallest scales of energy levels of atoms "
    "and subatomic particles. Classical mechanics, the description of physical phenomena "
    "at non-relativistic velocities and at scales larger than atoms, breaks down at these "
    "tiny distances. The earliest version of quantum mechanics was formulated in the first "
    "decade of the 20th century, when Max Planck explained the radiation emitted by an "
    "ideal blackbody, and Einstein proposed that light is also composed of discrete "
    "packets of energy called photons.",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def cpu_fp32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float32)


def shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def _causal_valid_mask(scores: torch.Tensor) -> torch.Tensor:
    """Lower-triangular (i>=j) valid region of pre-softmax logits ``[..., T, S]``.

    Tokens above the diagonal get clamped to fp16-min in BOTH dense and cmp models;
    they contribute 0 to MSE but inflate denominators. Restrict the rel-MSE to the
    causal lower-triangle to keep the metric meaningful (also drop any stray
    non-finite entries). Identical to legacy step1.
    """
    T_q, S_k = scores.shape[-2], scores.shape[-1]
    causal = torch.ones(T_q, S_k, dtype=torch.bool, device=scores.device).tril(diagonal=S_k - T_q)
    causal = causal & torch.isfinite(scores)
    return causal


def build_inputs(
    tokenizer,
    prompts: Sequence[str],
    seqlen: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack ``prompts`` into a single ``[B, T]`` tensor (left-truncated, EOS-padded).

    Returns the tokenized ``input_ids`` (no attention_mask kwarg — the model's
    own attention mask construction handles padding implicitly via causal mask).
    """
    enc_list = []
    for p in prompts:
        ids = tokenizer(
            p, return_tensors="pt", truncation=True, max_length=seqlen
        ).input_ids[0]
        if ids.numel() < seqlen:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 1
            pad = torch.full((seqlen - ids.numel(),), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)
        enc_list.append(ids[:seqlen])
    input_ids = torch.stack(enc_list, dim=0).to(device)
    return input_ids


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------
def detect_arch(model) -> str:
    cfg_mt = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    if cfg_mt == "opt":
        return "opt"
    if cfg_mt == "mpt":
        return "mpt"
    if cfg_mt in ("llama", "mistral", "qwen2"):
        return "llama"
    cls = model.__class__.__name__
    if "OPT" in cls:
        return "opt"
    if "Mpt" in cls or "MPT" in cls:
        return "mpt"
    return "llama"


def _layers(model, arch: str):
    if arch == "opt":
        return model.model.decoder.layers
    if arch == "mpt":
        return model.transformer.blocks
    return model.model.layers


def _attn_attr(arch: str) -> str:
    return "attn" if arch == "mpt" else "self_attn"


def _inner_forward(model, arch: str, input_ids: torch.Tensor) -> None:
    """Run the inner (no-LM-head) forward on the pre-loaded model.

    Going through the inner module skips ``lm_head`` and loss computation —
    we only need the attention forward to fire on each layer.
    """
    if arch == "opt":
        model.model.decoder(input_ids, use_cache=False)
    elif arch == "mpt":
        model.transformer(input_ids, use_cache=False)
    else:
        model.model(input_ids, use_cache=False)


# ---------------------------------------------------------------------------
# Metric kernel: identical formulas across archs, given full-store + cmp tensors
# ---------------------------------------------------------------------------
def _compute_metrics(
    full: Dict[str, torch.Tensor],
    Z_cmp: torch.Tensor,
    scores_cmp: torch.Tensor,
    probs_cmp: torch.Tensor,
    *,
    rank_kq: int,
    rank_v: int,
    bsz: int,
    num_heads: int,
    tgt_len: int,
    src_len: int,
    head_dim_v: int,
    value_states_low_flat: torch.Tensor,    # [B*H, S, head_dim_v]  CMP value states pre-out_proj_up
    out_proj_up: nn.Module,                 # CMP's out_proj_up / o_proj_up
) -> Dict[str, float]:
    """Final 5-metric computation from FULL captures + CMP intermediates.

    All inputs are expected on the same device as ``Z_cmp`` (we move FULL on demand).
    Identical to legacy ``step1_diagnose.py`` / ``step1_diagnose_llama.py``.
    """
    Z_full = full["Z"].to(device=Z_cmp.device, dtype=torch.float32)
    scores_full = full["scores"].to(device=Z_cmp.device, dtype=torch.float32)
    probs_full = full["probs"].to(device=Z_cmp.device, dtype=torch.float32)
    AV_full = full["AV"].to(device=Z_cmp.device, dtype=torch.float32)
    V_full = full["V"].to(device=Z_cmp.device, dtype=torch.float32)

    Z_cmp_f = Z_cmp.detach().float()
    scores_cmp_f = scores_cmp.float()
    probs_cmp_f = probs_cmp.float()

    # std / norm of Z (residual-stream level)
    std_full = Z_full.std().item()
    std_cmp = Z_cmp_f.std().item()
    std_ratio = std_cmp / max(std_full, 1e-12)

    norm_full_t = Z_full.norm(dim=-1)
    norm_cmp_t = Z_cmp_f.norm(dim=-1)
    norm_ratio = (norm_cmp_t / norm_full_t.clamp_min(1e-8)).mean().item()

    # softmax entropy
    ent_full = shannon_entropy(probs_full)
    ent_cmp = shannon_entropy(probs_cmp_f)
    ent_diff = (ent_cmp - ent_full).mean().item()
    ent_full_mean = ent_full.mean().item()
    ent_cmp_mean = ent_cmp.mean().item()

    # score (pre-softmax logits) MSE on the causal lower-tri only
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
    score_mean_full = (
        torch.where(causal, scores_full, torch.zeros_like(scores_full)).sum() / nvalid
    ).item()
    score_mean_cmp = (
        torch.where(causal, scores_cmp_f, torch.zeros_like(scores_cmp_f)).sum() / nvalid
    ).item()
    scores_full_centered_sq = (scores_full - score_mean_full).pow(2)
    scores_full_centered_sq = torch.where(
        causal, scores_full_centered_sq, torch.zeros_like(scores_full_centered_sq)
    )
    scores_full_std_valid = (scores_full_centered_sq.sum() / nvalid).sqrt().item()
    scores_cmp_centered_sq = (scores_cmp_f - score_mean_cmp).pow(2)
    scores_cmp_centered_sq = torch.where(
        causal, scores_cmp_centered_sq, torch.zeros_like(scores_cmp_centered_sq)
    )
    scores_cmp_std_valid = (scores_cmp_centered_sq.sum() / nvalid).sqrt().item()

    # Z MSE (residual-stream-level total error)
    Z_diff_sq = (Z_cmp_f - Z_full).pow(2).mean().item()
    Z_full_var = Z_full.pow(2).mean().item()
    Z_rel_mse = Z_diff_sq / max(Z_full_var, 1e-12)

    # value_only (oracle-A): keep CMP's V + out_proj_up, but use FULL's probs
    ow = out_proj_up.weight
    with torch.no_grad():
        probs_full_flat = probs_full.view(bsz * num_heads, tgt_len, src_len)
        probs_full_flat_dt = probs_full_flat.to(dtype=value_states_low_flat.dtype)
        ao = torch.matmul(probs_full_flat_dt, value_states_low_flat)  # [B*H, T, Dh_v]
        ao = (
            ao.view(bsz, num_heads, tgt_len, head_dim_v)
            .transpose(1, 2)
            .contiguous()
            .reshape(bsz, tgt_len, num_heads * head_dim_v)
        )
        ao = ao.to(device=ow.device, dtype=ow.dtype)
        Z_oracle_A = out_proj_up(ao).float()
        value_only_mse = (Z_oracle_A - Z_full).pow(2).mean().item()
        value_only_rel_mse = value_only_mse / max(Z_full_var, 1e-12)

    # score_only (oracle-V): keep FULL's V (in FULL embed_dim space), use CMP's probs.
    # Compared at AV-level (no CMP out_proj involved) since FULL's out_proj is gone in CMP.
    with torch.no_grad():
        Dh_full = V_full.size(-1)
        V_full_flat = V_full.reshape(bsz * num_heads, src_len, Dh_full)  # [B*H, S, Dh_full]
        probs_cmp_flat = probs_cmp_f.view(bsz * num_heads, tgt_len, src_len)
        AV_with_cmp_A = torch.matmul(probs_cmp_flat, V_full_flat)
        AV_with_cmp_A = (
            AV_with_cmp_A.view(bsz, num_heads, tgt_len, Dh_full)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, num_heads * Dh_full)
        )
        AV_full_var = AV_full.pow(2).mean().item()
        score_only_mse = (AV_with_cmp_A - AV_full).pow(2).mean().item()
        score_only_rel_mse = score_only_mse / max(AV_full_var, 1e-12)

    return {
        "rank_kq": int(rank_kq),
        "rank_v": int(rank_v),
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


# ===========================================================================
# OPT
# ===========================================================================
def _make_full_diag_forward_opt(self_attn: nn.Module, store: Dict[int, dict], li: int):
    """Replace OPTAttention.forward with a recording version (no behaviour change)."""

    def diag_forward(
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        m = self_attn
        bsz, tgt_len, _ = hidden_states.size()
        query_states = m.q_proj(hidden_states) * m.scaling
        key_states = m._shape(m.k_proj(hidden_states), -1, bsz)
        value_states = m._shape(m.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * m.num_heads, -1, m.head_dim)
        query_states_flat = m._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states_flat = key_states.view(*proj_shape)
        value_states_flat = value_states.view(*proj_shape)

        src_len = key_states_flat.size(1)
        attn_weights = torch.bmm(query_states_flat, key_states_flat.transpose(1, 2))

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, m.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )
            attn_weights = attn_weights.view(bsz * m.num_heads, tgt_len, src_len)

        scores_full = attn_weights.detach().view(bsz, m.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_probs = F.softmax(attn_weights, dim=-1)
        probs_full = attn_probs.detach().view(bsz, m.num_heads, tgt_len, src_len)

        attn_output = torch.bmm(attn_probs, value_states_flat)
        attn_output = attn_output.view(bsz, m.num_heads, tgt_len, m.head_dim).transpose(1, 2)
        AV_full = attn_output.detach().reshape(bsz, tgt_len, m.embed_dim)

        attn_output = attn_output.reshape(bsz, tgt_len, m.embed_dim)
        Z_full = m.out_proj(attn_output)

        # value_states is shape [B, H, S, Dh] (post m._shape). Store on CPU/fp16.
        store[li] = {
            "scores": cpu_fp32(scores_full).half(),
            "probs": cpu_fp32(probs_full).half(),
            "V": cpu_fp32(value_states.detach()).half(),  # [B,H,S,Dh]
            "AV": cpu_fp32(AV_full.detach()).half(),       # [B,T,E]
            "Z": cpu_fp32(Z_full.detach()).half(),
        }
        return Z_full, None, past_key_value

    return diag_forward


def _make_cmp_diag_forward_opt(
    self_attn: nn.Module,
    cmp_store: Dict[int, dict],
    full_store: Dict[int, dict],
    metrics: Dict[int, dict],
    li: int,
):
    """Replicate ``OPTEigenAttention.forward`` (sans v3 fields) and compute the 5 metrics."""

    def diag_forward(
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        m = self_attn
        bsz, tgt_len, _ = hidden_states.size()

        query_states = m.q_proj_low(hidden_states) * m.scaling
        key_states_full = m.k_proj_low(hidden_states)   # [B, T, rank_kq]
        value_states_full = m.v_proj_low(hidden_states)  # [B, T, rank_v]

        head_dim_kq = m.rank_kq // m.num_heads
        head_dim_v = m.rank_v // m.num_heads

        # reshape to [B, H, T, head_dim_*]
        query_states = (
            query_states.view(bsz, tgt_len, m.num_heads, head_dim_kq).transpose(1, 2).contiguous()
        )
        key_states = (
            key_states_full.view(bsz, tgt_len, m.num_heads, head_dim_kq).transpose(1, 2).contiguous()
        )
        value_states = (
            value_states_full.view(bsz, tgt_len, m.num_heads, head_dim_v).transpose(1, 2).contiguous()
        )

        # flatten heads into batch for bmm
        query_flat = query_states.view(bsz * m.num_heads, tgt_len, head_dim_kq)
        key_flat = key_states.view(bsz * m.num_heads, tgt_len, head_dim_kq)
        value_flat = value_states.view(bsz * m.num_heads, tgt_len, head_dim_v)
        src_len = key_flat.size(1)

        attn_weights = torch.bmm(query_flat, key_flat.transpose(1, 2))

        if attention_mask is not None:
            attn_weights = (
                attn_weights.view(bsz, m.num_heads, tgt_len, src_len) + attention_mask
            )
            finfo = torch.finfo(attn_weights.dtype)
            attn_weights = torch.nan_to_num(attn_weights, neginf=finfo.min, posinf=finfo.max)
            attn_weights = attn_weights.view(bsz * m.num_heads, tgt_len, src_len)

        scores_cmp = attn_weights.detach().view(bsz, m.num_heads, tgt_len, src_len)

        attn_probs = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_flat.dtype)
        probs_cmp = attn_probs.detach().view(bsz, m.num_heads, tgt_len, src_len)

        attn_output = torch.bmm(attn_probs, value_flat)  # [B*H, T, head_dim_v]
        attn_output_2d = (
            attn_output.view(bsz, m.num_heads, tgt_len, head_dim_v)
            .transpose(1, 2)
            .contiguous()
            .reshape(bsz, tgt_len, m.rank_v)
        )

        ow = m.out_proj_up.weight
        if (attn_output_2d.device != ow.device) or (attn_output_2d.dtype != ow.dtype):
            attn_output_2d = attn_output_2d.to(device=ow.device, dtype=ow.dtype)
        Z_cmp = m.out_proj_up(attn_output_2d)  # [B,T,E]

        # ---- compute metrics
        full = full_store[li]
        metrics[li] = _compute_metrics(
            full=full,
            Z_cmp=Z_cmp,
            scores_cmp=scores_cmp,
            probs_cmp=probs_cmp,
            rank_kq=m.rank_kq,
            rank_v=m.rank_v,
            bsz=bsz,
            num_heads=m.num_heads,
            tgt_len=tgt_len,
            src_len=src_len,
            head_dim_v=head_dim_v,
            value_states_low_flat=value_flat,
            out_proj_up=m.out_proj_up,
        )
        cmp_store[li] = {"rank_kq": int(m.rank_kq), "rank_v": int(m.rank_v)}

        # free FULL captures for this layer ASAP
        for k in list(full.keys()):
            full[k] = None
        del full_store[li]

        return Z_cmp, None, past_key_value

    return diag_forward


# ===========================================================================
# Llama (handles GQA: num_key_value_heads <= num_heads)
# ===========================================================================
def _make_full_diag_forward_llama(self_attn: nn.Module, store: Dict[int, dict], li: int):
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
        value_states = value_states.view(
            bsz, q_len, m.num_key_value_heads, m.head_dim
        ).transpose(1, 2)

        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

        if position_embeddings is None:
            cos, sin = m.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, m.layer_idx, cache_kwargs
            )

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
        Z_full = m.o_proj(attn_output)

        store[li] = {
            "scores": cpu_fp32(scores_full).half(),
            "probs": cpu_fp32(probs_full).half(),
            "V": cpu_fp32(value_states.detach()).half(),  # [B, H_full(=num_heads after repeat), T, head_dim]
            "AV": cpu_fp32(AV_full).half(),               # [B, T, E]
            "Z": cpu_fp32(Z_full.detach()).half(),        # [B, T, E]
        }
        return Z_full, None, past_key_value

    return diag_forward


def _make_cmp_diag_forward_llama(
    self_attn: nn.Module,
    cmp_store: Dict[int, dict],
    full_store: Dict[int, dict],
    metrics: Dict[int, dict],
    li: int,
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

        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

        if position_embeddings is None:
            cos, sin = m.rotary_emb(value_states_low, position_ids)
        else:
            cos, sin = position_embeddings

        # legacy parity: mirror the EigenAttn forward which checks for a stale
        # past_key_value attribute on the module (left over from cache-mode runs).
        past_key_value = getattr(m, "past_key_value", past_key_value)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states_low, value_states_low = past_key_value.update(
                key_states_low, value_states_low, m.layer_idx, cache_kwargs
            )

        key_states = m.k_proj_up(key_states_low)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQA broadcast (KV heads -> Q heads). value_states stays in low-rank space.
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

        # GQA per-head value width: rank_v // num_kv_heads (matches o_proj_up.in_features design).
        head_dim_v = m.rank_v // m.num_key_value_heads
        attn_output_2d = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, q_len, m.num_heads * head_dim_v)
        )
        Z_cmp = m.o_proj_up(attn_output_2d)

        # Flatten value_states (post-repeat_kv) into [B*H, S, Dh_v] for oracle-A.
        src_len = key_states.shape[-2]
        value_flat = value_states_low.reshape(bsz * m.num_heads, src_len, head_dim_v)

        full = full_store[li]
        metrics[li] = _compute_metrics(
            full=full,
            Z_cmp=Z_cmp,
            scores_cmp=scores_cmp,
            probs_cmp=probs_cmp,
            rank_kq=m.rank_kq,
            rank_v=m.rank_v,
            bsz=bsz,
            num_heads=m.num_heads,
            tgt_len=q_len,
            src_len=src_len,
            head_dim_v=head_dim_v,
            value_states_low_flat=value_flat,
            out_proj_up=m.o_proj_up,
        )
        cmp_store[li] = {"rank_kq": int(m.rank_kq), "rank_v": int(m.rank_v)}

        for k in list(full.keys()):
            full[k] = None
        del full_store[li]

        return Z_cmp, None, past_key_value

    return diag_forward


# ===========================================================================
# MPT (no legacy reference; mirrors the dense MptAttention.forward, ALiBi position_bias)
# ===========================================================================
def _make_full_diag_forward_mpt(self_attn: nn.Module, store: Dict[int, dict], li: int):
    def diag_forward(
        hidden_states,
        position_bias=None,
        past_key_value=None,
        attention_mask=None,
        **kwargs,
    ):
        m = self_attn
        bsz, q_len = hidden_states.shape[:2]

        mixed_qkv = m.Wqkv(hidden_states)
        if getattr(m, "clip_qkv", None):
            mixed_qkv = mixed_qkv.clamp(min=-m.clip_qkv, max=m.clip_qkv)
        query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
        query_states = query_states.reshape(bsz, q_len, m.n_heads, m.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, m.n_heads, m.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, m.n_heads, m.head_dim).transpose(1, 2)

        attention_scores = (
            torch.matmul(query_states, key_states.transpose(-1, -2)) * m.softmax_scale
        )
        src_len = key_states.shape[-2]
        if position_bias is not None:
            position_bias_query_index = max(0, position_bias.size(1) - q_len)
            position_bias_key_index = max(0, position_bias.size(2) - src_len)
            position_bias = position_bias[
                :, position_bias_query_index:, position_bias_key_index:
            ]
            attention_scores = attention_scores + position_bias
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask, torch.finfo(query_states.dtype).min
            )
        scores_full = attention_scores.detach().view(bsz, m.n_heads, q_len, src_len)

        attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(value_states.dtype)
        # MPT's training-mode dropout is no-op since we call model.eval(); skip dropout.
        probs_full = attn_weights.detach().view(bsz, m.n_heads, q_len, src_len)

        context_states = torch.matmul(attn_weights, value_states)
        context_states = (
            context_states.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, -1)
        )
        AV_full = context_states.detach()
        Z_full = m.out_proj(context_states)

        store[li] = {
            "scores": cpu_fp32(scores_full).half(),
            "probs": cpu_fp32(probs_full).half(),
            "V": cpu_fp32(value_states.detach()).half(),  # [B, H, S, Dh]
            "AV": cpu_fp32(AV_full).half(),               # [B, T, E]
            "Z": cpu_fp32(Z_full.detach()).half(),
        }
        return Z_full, None, past_key_value

    return diag_forward


def _make_cmp_diag_forward_mpt(
    self_attn: nn.Module,
    cmp_store: Dict[int, dict],
    full_store: Dict[int, dict],
    metrics: Dict[int, dict],
    li: int,
):
    def diag_forward(
        hidden_states,
        position_bias=None,
        past_key_value=None,
        attention_mask=None,
        **kwargs,
    ):
        m = self_attn
        bsz, q_len = hidden_states.shape[:2]

        mixed_qkv = m.Wqkv_low(hidden_states)
        query_states = mixed_qkv[:, :, : m.rank_kq]
        key_states = mixed_qkv[:, :, m.rank_kq : 2 * m.rank_kq]
        value_states = mixed_qkv[:, :, 2 * m.rank_kq :]

        head_dim_kq = m.rank_kq // m.n_heads
        head_dim_v = m.rank_v // m.n_heads

        query_states = query_states.reshape(bsz, q_len, m.n_heads, head_dim_kq).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, m.n_heads, head_dim_kq).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, m.n_heads, head_dim_v).transpose(1, 2)

        attention_scores = (
            torch.matmul(query_states, key_states.transpose(-1, -2)) * m.softmax_scale
        )
        src_len = key_states.shape[-2]
        if position_bias is not None:
            position_bias_query_index = max(0, position_bias.size(1) - q_len)
            position_bias_key_index = max(0, position_bias.size(2) - src_len)
            position_bias = position_bias[
                :, position_bias_query_index:, position_bias_key_index:
            ]
            attention_scores = attention_scores + position_bias
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask, torch.finfo(query_states.dtype).min
            )
        scores_cmp = attention_scores.detach().view(bsz, m.n_heads, q_len, src_len)

        attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(value_states.dtype)
        probs_cmp = attn_weights.detach().view(bsz, m.n_heads, q_len, src_len)

        context_states = torch.matmul(attn_weights, value_states)
        context_states_2d = (
            context_states.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, m.rank_v)
        )
        Z_cmp = m.out_proj_up(context_states_2d)

        value_flat = value_states.reshape(bsz * m.n_heads, src_len, head_dim_v)

        full = full_store[li]
        metrics[li] = _compute_metrics(
            full=full,
            Z_cmp=Z_cmp,
            scores_cmp=scores_cmp,
            probs_cmp=probs_cmp,
            rank_kq=m.rank_kq,
            rank_v=m.rank_v,
            bsz=bsz,
            num_heads=m.n_heads,
            tgt_len=q_len,
            src_len=src_len,
            head_dim_v=head_dim_v,
            value_states_low_flat=value_flat,
            out_proj_up=m.out_proj_up,
        )
        cmp_store[li] = {"rank_kq": int(m.rank_kq), "rank_v": int(m.rank_v)}

        for k in list(full.keys()):
            full[k] = None
        del full_store[li]

        return Z_cmp, None, past_key_value

    return diag_forward


# ---------------------------------------------------------------------------
# Patch / unpatch dispatcher
# ---------------------------------------------------------------------------
def _patch_full(model, arch: str, store: Dict[int, dict]):
    saved = []
    factory = {
        "opt": _make_full_diag_forward_opt,
        "llama": _make_full_diag_forward_llama,
        "mpt": _make_full_diag_forward_mpt,
    }[arch]
    attn_attr = _attn_attr(arch)
    for li, layer in enumerate(_layers(model, arch)):
        attn = getattr(layer, attn_attr)
        saved.append((attn, attn.forward))
        attn.forward = factory(attn, store, li)
    return saved


def _patch_cmp(model, arch: str, cmp_store, full_store, metrics):
    saved = []
    factory = {
        "opt": _make_cmp_diag_forward_opt,
        "llama": _make_cmp_diag_forward_llama,
        "mpt": _make_cmp_diag_forward_mpt,
    }[arch]
    attn_attr = _attn_attr(arch)
    for li, layer in enumerate(_layers(model, arch)):
        attn = getattr(layer, attn_attr)
        saved.append((attn, attn.forward))
        attn.forward = factory(attn, cmp_store, full_store, metrics, li)
    return saved


def _unpatch(saved):
    for attn, fwd in saved:
        attn.forward = fwd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_step1_diagnose(
    base_model,
    cmp_model,
    tokenizer,
    *,
    prompts: Optional[Sequence[str]] = None,
    seqlen: int = 512,
    base_device: Optional[str] = None,
    cmp_device: Optional[str] = None,
    log_fn=print,
) -> Dict[str, dict]:
    """Run a 2-phase forward-replace fragility diagnosis.

    Phase 1 (BASE): captures (scores, probs, V, AV, Z) per layer to CPU/fp16.
    Phase 2 (CMP): replays the same inputs through the EigenAttn model,
                   uses the FULL captures to compute the 5 schema metrics.

    ``base_device`` / ``cmp_device`` default to whatever device each model is
    already on. The two models can sit on different GPUs (paper-aligned setup
    used by the legacy script).
    """
    prompts = list(prompts or DEFAULT_PROMPTS)
    arch = detect_arch(cmp_model)
    if arch != detect_arch(base_model):
        log_fn(
            f"[step1] WARNING: base arch != cmp arch (base={detect_arch(base_model)}, cmp={arch}); "
            f"continuing with cmp's arch dispatch"
        )

    base_dev = torch.device(base_device or next(base_model.parameters()).device)
    cmp_dev = torch.device(cmp_device or next(cmp_model.parameters()).device)
    log_fn(f"[step1] arch={arch} prompts={len(prompts)} seqlen={seqlen}")
    log_fn(f"[step1] base_device={base_dev} cmp_device={cmp_dev}")

    # ---- Phase 1: BASE
    base_model.config.use_cache = False
    full_store: Dict[int, dict] = {}
    saved_full = _patch_full(base_model, arch, full_store)
    try:
        input_ids = build_inputs(tokenizer, prompts, seqlen, base_dev)
        _inner_forward(base_model, arch, input_ids)
    finally:
        _unpatch(saved_full)

    n_layers = len(full_store)
    log_fn(f"[step1] phase 1 done: captured {n_layers} layers from BASE")

    # explicitly migrate captures to CPU (already on CPU by cpu_fp32(); belt-and-suspenders)
    for li, d in full_store.items():
        for k, v in list(d.items()):
            if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                d[k] = v.detach().to("cpu", non_blocking=True)

    torch.cuda.empty_cache()

    # ---- Phase 2: CMP
    cmp_model.config.use_cache = False
    cmp_store: Dict[int, dict] = {}
    metrics: Dict[int, dict] = {}
    saved_cmp = _patch_cmp(cmp_model, arch, cmp_store, full_store, metrics)
    try:
        input_ids_cmp = build_inputs(tokenizer, prompts, seqlen, cmp_dev)
        _inner_forward(cmp_model, arch, input_ids_cmp)
    finally:
        _unpatch(saved_cmp)

    log_fn(f"[step1] phase 2 done: scored {len(metrics)} layers")
    if len(metrics) != n_layers:
        log_fn(
            f"[step1] WARNING: metric layer count {len(metrics)} != FULL capture {n_layers}; "
            f"some layers may have been skipped"
        )

    # ---- Aggregate
    per_layer: Dict[str, dict] = {str(li): metrics[li] for li in sorted(metrics.keys())}
    summary_keys = [
        "std_ratio_Z",
        "norm_ratio_Z",
        "entropy_diff_A",
        "score_rel_mse_causal",
        "score_std_ratio_causal",
        "Z_rel_mse",
        "value_only_Z_rel_mse",
        "score_only_AV_rel_mse",
    ]
    summary = {}
    for k in summary_keys:
        vals = [metrics[li][k] for li in sorted(metrics.keys()) if k in metrics[li]]
        if not vals:
            continue
        vt = torch.tensor(vals, dtype=torch.float64)
        summary[k] = {
            "mean": float(vt.mean()),
            "median": float(vt.median()),
            "min": float(vt.min()),
            "max": float(vt.max()),
        }

    # short per-layer log for quick eyeballing
    for li in sorted(metrics.keys()):
        m = metrics[li]
        log_fn(
            f"[step1] L{li:02d} rkq={m['rank_kq']} rv={m['rank_v']} "
            f"Z_rel_mse={m['Z_rel_mse']:.4e} score_rel={m['score_rel_mse_causal']:.4e} "
            f"value={m['value_only_Z_rel_mse']:.4e} score={m['score_only_AV_rel_mse']:.4e} "
            f"dH={m['entropy_diff_A']:+.4f}"
        )

    return {
        "meta": {
            "arch": arch,
            "n_layers": n_layers,
            "n_metrics": len(metrics),
            "seqlen": seqlen,
            "n_prompts": len(prompts),
            "schema": "Z_rel_mse / score_rel_mse_causal / value_only_Z_rel_mse / score_only_AV_rel_mse / entropy_diff_A",
            "method": "forward-replace (port of EigenAttn/EigenAttn/step1_diagnose{,_llama}.py, v3-only fields removed)",
        },
        "per_layer": per_layer,
        "summary": summary,
    }


def save_step1(report: Dict[str, dict], out_json: str) -> None:
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


__all__ = [
    "DEFAULT_PROMPTS",
    "detect_arch",
    "build_inputs",
    "shannon_entropy",
    "run_step1_diagnose",
    "save_step1",
]
