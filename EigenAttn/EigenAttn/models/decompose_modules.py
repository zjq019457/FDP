# models/decompose_modules.py
# ============================================================
# EigenAttention decomposed modules for OPT / MPT (stable)
# - OPT: low-rank Q/K/V + correct out_proj_up init + basis_side switch
# - MPT: keep original-style headwise projection
# - LLaMA: placeholder (import-safe). Recommend separate file + dedicated fix.
# ============================================================

import math
import warnings
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# MPT / LLaMA imports are kept for compatibility with your existing codepaths.
from transformers.models.mpt.modeling_mpt import *  # noqa: F401,F403
from transformers.models.mpt.configuration_mpt import *  # noqa: F401,F403
from transformers.models.llama.modeling_llama import *  # noqa: F401,F403
from transformers.models.llama.configuration_llama import *  # noqa: F401,F403
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb as qwen2_apply_rotary_pos_emb


# ------------------------------------------------------------
# Common helpers
# ------------------------------------------------------------

def _normalize_basis(B: torch.Tensor, d: int, r: int, name: str) -> torch.Tensor:
    """
    Acceptable shapes:
      - [d, r]
      - [1, d, r]
      - [r, d]  (will transpose)
      - [1, r, d] (will squeeze then transpose)
    Return: [d, r] contiguous
    """
    if B is None:
        raise ValueError(f"{name}: basis is None")

    if not torch.is_tensor(B):
        B = torch.as_tensor(B)

    if B.ndim == 3 and B.shape[0] == 1:
        B = B.squeeze(0)

    if B.ndim != 2:
        raise ValueError(f"{name}: basis must be 2D after squeeze, got shape={tuple(B.shape)}")

    if B.shape == (r, d):
        B = B.t().contiguous()

    if B.shape != (d, r):
        raise ValueError(f"{name}: basis shape mismatch got={tuple(B.shape)} expect=({d},{r})")

    return B.contiguous()


def _make_low_linear(W: torch.Tensor, b: Optional[torch.Tensor], B: torch.Tensor, side: str):
    """
    Build low-rank projection (d -> r) from full Linear weight W (out=d, in=d) and basis B ([d,r]).

    side == "left":  P = W^T @ B   (interpreting B as output subspace; left/out)
    side == "right": P = W   @ B   (interpreting B as input  subspace; right/in)

    Return:
      w_low: [r, d]  (for nn.Linear(d -> r))
      b_low: [r] or None
    """
    if side == "left":
        P = W.t().matmul(B)           # [d, r]
    elif side == "right":
        P = W.matmul(B)               # [d, r]
    else:
        raise ValueError(f"Unknown basis_side={side}. Use 'left' or 'right'.")

    w_low = P.t().contiguous()        # [r, d]
    b_low = b.matmul(B) if b is not None else None
    if b_low is not None:
        b_low = b_low.contiguous()
    return w_low, b_low


# ------------------------------------------------------------
# OPT
# ------------------------------------------------------------

class OPTEigenAttention(nn.Module):
    """
    OPT attention with low-rank Q/K and low-rank V, attention computed in low dims.
    Key fixes vs your current version:
      - basis is passed PER-LAYER as [d,r], no basis[layer_index] indexing
      - supports basis_side switch (args.basis_side: "left" or "right")
      - out_proj_up is initialized as W_out @ V (crucial for sane PPL)
    """
    def __init__(
        self,
        org_module: nn.Module,
        basis_kq: torch.Tensor,          # [d, r_kq] for this layer
        basis_v: torch.Tensor,           # [d, r_v]  for this layer
        rank_kq: int,
        rank_v: int,
        embed_dim: int,
        num_heads: int,
        layer_name: str,
        dropout: float = 0.0,
        is_decoder: bool = True,
        bias: bool = True,
        args=None,
        disable_act_quant: bool = False,  # kept for compatibility
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.rank_kq = int(rank_kq)
        self.rank_v = int(rank_v)

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        # basis_side discovered by your diag script
        #   - "left"  == left(out) projection
        #   - "right" == right(in) projection
        self.basis_side = getattr(args, "basis_side", "left")

        # original weights
        Wq, bq = org_module.q_proj.weight, org_module.q_proj.bias
        Wk, bk = org_module.k_proj.weight, org_module.k_proj.bias
        Wv, bv = org_module.v_proj.weight, org_module.v_proj.bias
        Wo, bo = org_module.out_proj.weight, org_module.out_proj.bias

        # normalize basis
        U = _normalize_basis(basis_kq.to(Wq.device), embed_dim, self.rank_kq, f"{layer_name}.basis_kq")
        V = _normalize_basis(basis_v.to(Wv.device),  embed_dim, self.rank_v,  f"{layer_name}.basis_v")

        # low rank q/k/v  (d -> r)
        q_w, q_b = _make_low_linear(Wq, bq, U, self.basis_side)
        k_w, k_b = _make_low_linear(Wk, bk, U, self.basis_side)
        v_w, v_b = _make_low_linear(Wv, bv, V, self.basis_side)

        self.q_proj = nn.Linear(embed_dim, self.rank_kq, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.rank_kq, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.rank_v,  bias=bias)

        self.q_proj.weight.data.copy_(q_w)
        self.k_proj.weight.data.copy_(k_w)
        self.v_proj.weight.data.copy_(v_w)
        if bias and q_b is not None: self.q_proj.bias.data.copy_(q_b)
        if bias and k_b is not None: self.k_proj.bias.data.copy_(k_b)
        if bias and v_b is not None: self.v_proj.bias.data.copy_(v_b)

        # map low-dim attention output (rank_v) back to embed_dim
        # Deterministic good init: out_proj_up.weight = Wo @ V
        self.out_proj_up = nn.Linear(self.rank_v, embed_dim, bias=bias)
        out_w = Wo.matmul(V).contiguous()    # [d, r_v]
        self.out_proj_up.weight.data.copy_(out_w)
        if bias and bo is not None:
            self.out_proj_up.bias.data.copy_(bo)

    def _new_shape(self, tensor, seq_len, bsz, low_dim):
        if low_dim < self.num_heads or low_dim % self.num_heads != 0:
            raise ValueError(
                f"low_dim={low_dim} must be >= num_heads={self.num_heads} and divisible by num_heads."
            )
        return tensor.view(bsz, seq_len, self.num_heads, low_dim // self.num_heads).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        if is_cross_attention and past_key_value is not None:
            key_states, value_states = past_key_value
        elif is_cross_attention:
            key_states = self._new_shape(self.k_proj(key_value_states), -1, bsz, self.rank_kq)
            value_states = self._new_shape(self.v_proj(key_value_states), -1, bsz, self.rank_v)
        elif past_key_value is not None:
            key_states = self._new_shape(self.k_proj(hidden_states), -1, bsz, self.rank_kq)
            value_states = self._new_shape(self.v_proj(hidden_states), -1, bsz, self.rank_v)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._new_shape(self.k_proj(hidden_states), -1, bsz, self.rank_kq)
            value_states = self._new_shape(self.v_proj(hidden_states), -1, bsz, self.rank_v)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape_kq = (bsz * self.num_heads, -1, self.rank_kq // self.num_heads)
        proj_shape_v  = (bsz * self.num_heads, -1, self.rank_v  // self.num_heads)

        query_states = self._new_shape(query_states, tgt_len, bsz, self.rank_kq).view(*proj_shape_kq)
        key_states   = key_states.view(*proj_shape_kq)
        value_states = value_states.view(*proj_shape_v)

        src_len = key_states.size(1)
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            # attention_mask expected [bsz, 1, tgt, src]
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # stable softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) if output_attentions else None

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.rank_v // self.num_heads).transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.rank_v)
        attn_output = self.out_proj_up(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OPTEigenAttnDecoderLayer(nn.Module):
    def __init__(
        self,
        ori_layer,
        args,
        basis_kq,   # [d,r_kq] for this layer
        rank_kq,
        basis_v,    # [d,r_v] for this layer
        rank_v,
        config,
        layer_name: Optional[str] = None,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        # Residual mixer (方案1): 不依赖 block summary，只用当前层的 hidden_states 做一个轻量补偿。
        # alpha=0 默认关闭（保证不影响原有行为）。
        self.resid_mixer_alpha = float(getattr(args, "resid_mixer_alpha", 0.0) or 0.0)
        if self.resid_mixer_alpha != 0.0:
            resid_hidden = int(getattr(args, "resid_mixer_hidden_dim", 0) or 0)
            if resid_hidden == 0:
                resid_hidden = max(256, self.embed_dim // 8)
            resid_hidden = max(64, min(resid_hidden, self.embed_dim * 2))  # clamp 合理范围
            resid_ln_eps = float(getattr(args, "resid_mixer_ln_eps", 1e-5))
            self.resid_mixer_norm = nn.LayerNorm(self.embed_dim, eps=resid_ln_eps)
            self.resid_mixer = nn.Sequential(
                nn.Linear(self.embed_dim, resid_hidden),
                nn.GELU(),
                nn.Linear(resid_hidden, self.embed_dim),
            )

        self.self_attn = OPTEigenAttention(
            org_module=ori_layer.self_attn,
            basis_kq=basis_kq,
            basis_v=basis_v,
            rank_kq=rank_kq,
            rank_v=rank_v,
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
            args=args,
            layer_name=layer_name or "opt_layer",
        )

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.self_attn_layer_norm = ori_layer.self_attn_layer_norm

        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc1.weight.data.copy_(ori_layer.fc1.weight.data)
        self.fc1.bias.data.copy_(ori_layer.fc1.bias.data)

        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.fc2.weight.data.copy_(ori_layer.fc2.weight.data)
        self.fc2.bias.data.copy_(ori_layer.fc2.bias.data)

        self.final_layer_norm = ori_layer.final_layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=bool(output_attentions),
        )

        hidden_states = F.dropout(hidden_states, p=0.0, training=False)
        hidden_states = residual + hidden_states
        if self.resid_mixer_alpha != 0.0:
            # 方案1：在 attention residual 后做轻量补偿（不额外计算 attention）
            mix = self.resid_mixer_norm(hidden_states)
            mix = self.resid_mixer(mix)
            hidden_states = hidden_states + self.resid_mixer_alpha * mix

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # FFN
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.fc2(hidden_states)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# ------------------------------------------------------------
# MPT (keep your original approach; minor guards)
# ------------------------------------------------------------

class MptEigenAttention(nn.Module):
    """MPT attention with headwise low-rank projection (kept close to your original)."""

    def __init__(self, basis_kq, rank_kq, basis_v, rank_v, ori_layer, config: MptConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.max_seq_length = config.max_seq_len
        self.head_dim = self.hidden_size // self.n_heads
        self.softmax_scale = config.attn_config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)

        self.attn_dropout_p = config.attn_config.attn_pdrop
        self.rank_kq = int(rank_kq)
        self.rank_v = int(rank_v)

        if self.rank_kq % self.n_heads != 0 or self.rank_v % self.n_heads != 0:
            raise ValueError(f"MPT: ranks must be divisible by n_heads. rank_kq={rank_kq}, rank_v={rank_v}, n_heads={self.n_heads}")

        self.Wqkv = nn.Linear(self.hidden_size, (2 * self.rank_kq + self.rank_v), bias=False)

        ori_weight_q, ori_weight_k, ori_weight_v = ori_layer.Wqkv.weight.data.chunk(3, dim=0)

        # Q/K
        kq_per_head = self.rank_kq // self.n_heads
        for i in range(self.n_heads):
            u = basis_kq[i][:, :kq_per_head]  # [head_dim, kq_per_head] expected
            self.Wqkv.weight.data[i * kq_per_head:(i + 1) * kq_per_head, :] = u.t().matmul(
                ori_weight_q[i * self.head_dim:(i + 1) * self.head_dim, :]
            )
            self.Wqkv.weight.data[self.rank_kq + i * kq_per_head:self.rank_kq + (i + 1) * kq_per_head, :] = u.t().matmul(
                ori_weight_k[i * self.head_dim:(i + 1) * self.head_dim, :]
            )

        # V
        v_per_head = self.rank_v // self.n_heads
        for i in range(self.n_heads):
            u = basis_v[i][:, :v_per_head]
            self.Wqkv.weight.data[2 * self.rank_kq + i * v_per_head:2 * self.rank_kq + (i + 1) * v_per_head, :] = u.t().matmul(
                ori_weight_v[i * self.head_dim:(i + 1) * self.head_dim, :]
            )

        # out proj
        bias = ori_layer.out_proj.bias is not None
        self.out_proj = nn.Linear(v_per_head * self.n_heads, self.hidden_size, bias=bias)
        org_weight_out = ori_layer.out_proj.weight
        org_bias_out = ori_layer.out_proj.bias

        for i in range(self.n_heads):
            u = basis_v[i][:, :v_per_head]
            self.out_proj.weight.data[:, i * v_per_head:(i + 1) * v_per_head] = org_weight_out[:, self.head_dim * i:(i + 1) * self.head_dim].matmul(u)
        if bias and org_bias_out is not None:
            self.out_proj.bias.data.copy_(org_bias_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        mixed_qkv = self.Wqkv(hidden_states)
        query_states = mixed_qkv[:, :, 0:self.rank_kq]
        key_states = mixed_qkv[:, :, self.rank_kq:2 * self.rank_kq]
        value_states = mixed_qkv[:, :, 2 * self.rank_kq:]

        query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.rank_kq // self.n_heads).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.rank_kq // self.n_heads).transpose(1, 2)
        value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.rank_v // self.n_heads).transpose(1, 2)

        if past_key_value is not None and len(past_key_value) != 0:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states)

        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale

        if position_bias is not None:
            key_length = key_states.shape[-2]
            query_length = seq_length
            position_bias_query_index = max(0, position_bias.size(1) - query_length)
            position_bias_key_index = max(0, position_bias.size(2) - key_length)
            position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]
            attention_scores = attention_scores + position_bias

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, torch.finfo(query_states.dtype).min)

        attn_weights = F.softmax(attention_scores.float(), dim=-1).to(value_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

        context_states = torch.matmul(attn_weights, value_states)
        context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.out_proj(context_states)

        return attn_output, attn_weights, past_key_value


class MptBlockEigenAttn(nn.Module):
    def __init__(self, ori_layer, args, basis_kq, rank_kq, basis_v, rank_v, config: MptConfig):
        super().__init__()
        self.norm_1 = ori_layer.norm_1
        self.norm_2 = ori_layer.norm_2
        self.attn = MptEigenAttention(basis_kq, rank_kq, basis_v, rank_v, ori_layer.attn, config)
        self.ffn = ori_layer.ffn
        self.resid_attn_dropout = ori_layer.resid_attn_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        layernorm_output = self.norm_1(hidden_states)
        residual = hidden_states

        attn_outputs, attn_weights, past_key_value = self.attn(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            past_key_value=layer_past,
        )

        hidden_states = self.resid_attn_dropout(attn_outputs) + residual
        layernorm_output = self.norm_2(hidden_states)
        residual = hidden_states

        output = self.ffn(layernorm_output, residual)
        outputs = (output,)

        if use_cache:
            outputs += (past_key_value,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


# ------------------------------------------------------------
# LLaMA (restored from decompose_modules copy)
# ------------------------------------------------------------

class LlamaEigenAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, basis_kq, rank_kq, basis_v, rank_v, ori_layer, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self._use_qwen2_rope = str(getattr(config, "model_type", "")).lower() == "qwen2"

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # assert self.num_key_value_heads == self.num_heads #Does not support Grouped Query Attn now.
        
        self.rank_kq = rank_kq
        self.rank_v = rank_v
        bias = config.attention_bias

        org_weight_k = ori_layer.k_proj.weight
        org_bias_k = ori_layer.k_proj.bias


        self.k_proj_low = torch.nn.Linear(self.hidden_size, self.rank_kq, bias = config.attention_bias)
        self.k_proj_up = torch.nn.Linear(self.rank_kq // self.num_key_value_heads, self.head_dim, bias = False)
        
        self.q_proj = ori_layer.q_proj

        k = self.rank_kq//self.num_key_value_heads
        u = basis_kq
        # u = torch.matmul(hadamard, u)
        u = u[:, 0:k]
        for i in range(self.num_key_value_heads):
            # per attention head decomposition for K and Q
            self.k_proj_low.weight.data[i*k:(i+1)*k, :] = torch.matmul(u.t(), org_weight_k[i*self.head_dim:(i+1)*self.head_dim, :])
            if bias:
                self.k_proj_low.bias.data[i*k:(i+1)*k] = torch.matmul(u.t(), org_bias_k[i*self.head_dim:(i+1)*self.head_dim]) 

        self.k_proj_up.weight.data = u.contiguous()
        
        # decompose V
        org_weight_v = ori_layer.v_proj.weight
        org_bias_v = ori_layer.v_proj.bias

        k = self.rank_v//self.num_key_value_heads
        self.v_proj_low = torch.nn.Linear(self.hidden_size, self.rank_v, bias = bias)
        
        for i in range(self.num_key_value_heads):
            u = basis_v[i]
            # u = torch.matmul(hadamard, u)
            u = u[:, 0:k]
            self.v_proj_low.weight.data[i*k:(i+1)*k, :] = torch.matmul(u.t(), org_weight_v[i*self.head_dim:(i+1)*self.head_dim, :])
            if bias:
                self.v_proj_low.bias.data[i*k:(i+1)*k] = torch.matmul(u.t(), org_bias_v[i*self.head_dim:(i+1)*self.head_dim])

        bias = ori_layer.o_proj.bias is not None
        self.o_proj_up = torch.nn.Linear(k * self.num_heads, self.hidden_size, bias = bias)
        org_weight_out = ori_layer.o_proj.weight
        org_bias_out = ori_layer.o_proj.bias

        k = self.rank_v//self.num_key_value_heads
        for i in range(self.num_key_value_heads):
            u = basis_v[i]
            # u = torch.matmul(hadamard, u)
            u = u[:, 0:k]
            for j in range(self.num_key_value_groups):
                idx = i*self.num_key_value_groups + j
                self.o_proj_up.weight.data[:, idx*k:(idx+1)*k] = torch.matmul(org_weight_out[:, self.head_dim*idx : (idx+1)*self.head_dim], u)
        self.o_proj_up.bias = org_bias_out


        self._init_rope()

    def _init_rope(self):
        if self._use_qwen2_rope:
            if getattr(self.config, "rope_scaling", None) is not None:
                raise NotImplementedError("Qwen2 RoPE scaling is not covered by this frozen adapter.")
            self.rotary_emb = Qwen2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        elif getattr(self.config, "rope_scaling", None) is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states_low = self.k_proj_low(hidden_states)
            value_states_low = self.v_proj_low(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_low = key_states_low.view(bsz, q_len, self.num_key_value_heads, self.rank_kq // self.num_key_value_heads).transpose(1, 2)
        value_states_low = value_states_low.view(bsz, q_len, self.num_key_value_heads, self.rank_v // self.num_key_value_heads).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        if self._use_qwen2_rope:
            kv_seq_len = key_states_low.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            cos, sin = self.rotary_emb(value_states_low, seq_len=kv_seq_len)
        else:
            cos, sin = self.rotary_emb(value_states_low, position_ids)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states_low, value_states_low = past_key_value.update(
                key_states_low, value_states_low, self.layer_idx, cache_kwargs
            )

        key_states = self.k_proj_up(key_states_low)
        if self._use_qwen2_rope:
            query_states, key_states = qwen2_apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states_low = repeat_kv(value_states_low, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_low)

        # attn_output = self.v_proj_up(attn_output)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.rank_v //self.num_key_value_heads):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.rank_v // self.num_key_value_heads)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len,self.num_heads * (self.rank_v // self.num_key_value_heads))

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj_up(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class LlamaEigenAttnDecoderLayer(nn.Module):
    def __init__(self, 
                 ori_layer,
                 args,
                 basis_kq,
                 rank_kq,
                 basis_v,
                 rank_v,
                 config: LlamaConfig, 
                 layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.self_attn = LlamaEigenAttention(basis_kq, rank_kq, basis_v, rank_v, ori_layer.self_attn, config=config, layer_idx=layer_idx)


        # self.mlp = LlamaMLP(config)
        self.mlp = ori_layer.mlp
        # self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = ori_layer.input_layernorm
        # self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ori_layer.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
  
