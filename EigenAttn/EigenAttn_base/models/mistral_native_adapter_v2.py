"""Native Mistral adapter for the shared Eigen low-rank attention factors.

The V4 Mistral candidate bundles use the same per-KV-head factorization as the
shared Eigen attention implementation.  Mistral differs at runtime in that its
KV cache stores one position-rotated key per token.  This adapter instead keeps
the compressed, unrotated K factor in DynamicCache and reapplies RoPE with each
cached token's own position after reconstructing K.  This avoids rotating every
past key with the current decode position.

This implementation intentionally supports only the frozen Mistral-v0.2
contract used by the V4 audit: eager attention, GQA, unscaled Mistral RoPE,
pretraining_tp=1, and Transformers DynamicCache.  Other cache/RoPE contracts
fail closed rather than silently falling back to Llama behavior.
"""
from copy import deepcopy
import math

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.mistral.modeling_mistral import repeat_kv

from .decompose_modules import LlamaEigenAttention


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MistralEigenAttention(LlamaEigenAttention):
    """Eigen low-rank attention with native Mistral RoPE and cache semantics."""

    def __init__(
        self,
        basis_kq: torch.Tensor,
        rank_kq: int,
        basis_v: torch.Tensor,
        rank_v: int,
        ori_layer,
        config,
        layer_idx: int,
    ):
        if str(getattr(config, "model_type", "")).lower() != "mistral":
            raise ValueError("MistralEigenAttention requires config.model_type='mistral'")
        if getattr(config, "rope_scaling", None) is not None:
            raise NotImplementedError("This frozen adapter does not cover scaled Mistral RoPE")
        if int(getattr(config, "pretraining_tp", 1)) != 1:
            raise NotImplementedError("This adapter requires pretraining_tp=1")
        nkv = int(config.num_key_value_heads)
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
        if int(rank_kq) % nkv or int(rank_v) % nkv:
            raise ValueError("KQ and V ranks must be divisible by the Mistral KV-head count")

        dtype = ori_layer.k_proj.weight.dtype
        device = ori_layer.k_proj.weight.device
        basis_kq = torch.as_tensor(basis_kq, device=device, dtype=dtype)
        basis_v = torch.as_tensor(basis_v, device=device, dtype=dtype)
        if basis_kq.ndim != 2 or basis_kq.shape[0] != head_dim:
            raise ValueError(f"basis_kq must have shape [{head_dim}, rank_per_kv]")
        if basis_v.ndim == 2 and basis_v.shape[0] == nkv * head_dim:
            if int(rank_v) % nkv:
                raise ValueError("rank_v must be divisible by num_key_value_heads")
            # The frozen Eigen builder exports V bases as a block-diagonal
            # [nkv*head_dim, total_rank] matrix, not a row-major stack of
            # [nkv, head_dim, per_head_rank]. Each KV head owns a disjoint
            # column block whose stride is the exported per-head capacity.
            per_head_rank = int(rank_v) // nkv
            if basis_v.shape[1] % nkv:
                raise ValueError("packed V basis width must divide evenly across KV heads")
            per_head_capacity = basis_v.shape[1] // nkv
            if per_head_capacity < per_head_rank:
                raise ValueError("packed V basis does not contain the requested per-head rank")
            basis_v = torch.stack([
                basis_v[h * head_dim:(h + 1) * head_dim,
                        h * per_head_capacity:h * per_head_capacity + per_head_rank]
                for h in range(nkv)
            ], dim=0).contiguous()
        if basis_v.ndim != 3 or tuple(basis_v.shape[:2]) != (nkv, head_dim):
            raise ValueError(f"basis_v must have shape [{nkv}, {head_dim}, rank_per_kv]")

        super().__init__(basis_kq, rank_kq, basis_v, rank_v, ori_layer, config, layer_idx)

        native_rope = getattr(ori_layer, "rotary_emb", None)
        if native_rope is None:
            raise ValueError("The source Mistral attention has no native rotary_emb")
        # Keep Mistral's own FP32 inv_freq; do not cast this buffer with low-rank
        # projection weights.
        self.rotary_emb = deepcopy(native_rope)
        for name in ("k_proj_low", "k_proj_up", "v_proj_low"):
            getattr(self, name).to(device=device, dtype=dtype)
        self.o_proj_up.weight.data = self.o_proj_up.weight.data.to(device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        batch, query_len, _ = hidden_states.shape
        if attention_mask is None or attention_mask.ndim != 4:
            raise ValueError("Pass the native Mistral 4-D causal/sliding-window attention mask")

        query = self.q_proj(hidden_states)
        query = query.view(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_low = self.k_proj_low(hidden_states)
        key_low = key_low.view(
            batch, query_len, self.num_key_value_heads,
            self.rank_kq // self.num_key_value_heads,
        ).transpose(1, 2)
        value_low = self.v_proj_low(hidden_states)
        value_low = value_low.view(
            batch, query_len, self.num_key_value_heads,
            self.rank_v // self.num_key_value_heads,
        ).transpose(1, 2)

        past = past_key_value
        if position_ids is None:
            if cache_position is not None:
                position_ids = torch.as_tensor(cache_position, device=hidden_states.device).reshape(1, -1)
                position_ids = position_ids.expand(batch, -1)
            else:
                if isinstance(past, DynamicCache):
                    old_len = int(past.get_seq_length(self.layer_idx))
                elif past is not None:
                    old_len = int(past[0].shape[-2])
                else:
                    old_len = 0
                position_ids = torch.arange(
                    old_len, old_len + query_len, device=hidden_states.device
                ).reshape(1, -1).expand(batch, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.reshape(1, -1).expand(batch, -1)
        if position_ids.shape != (batch, query_len):
            raise ValueError("position_ids must have shape [batch, query_len]")
        if query_len > 1 and not bool(
            torch.all(position_ids[:, 1:] - position_ids[:, :-1] == 1)
        ):
            raise NotImplementedError("Only contiguous Mistral position_ids are supported")

        if past is not None and hasattr(past, "update") and not isinstance(past, DynamicCache):
            raise NotImplementedError("Only Transformers DynamicCache is supported")
        if isinstance(past, DynamicCache):
            key_low, value_low = past.update(
                key_low, value_low, self.layer_idx, {"cache_position": cache_position}
            )
        elif past is not None:
            key_low = torch.cat((past[0], key_low), dim=2)
            value_low = torch.cat((past[1], value_low), dim=2)
            past = (key_low, value_low)

        kv_len = key_low.shape[-2]
        key = self.k_proj_up(key_low)
        cos_q, sin_q = self.rotary_emb(value_low[..., -query_len:, :], position_ids)
        key_start = position_ids[:, :1] - (kv_len - query_len)
        key_positions = key_start + torch.arange(kv_len, device=hidden_states.device)[None, :]
        cos_k, sin_k = self.rotary_emb(key_low, key_positions)
        query = query * cos_q.unsqueeze(1) + _rotate_half(query) * sin_q.unsqueeze(1)
        key = key * cos_k.unsqueeze(1) + _rotate_half(key) * sin_k.unsqueeze(1)

        key = repeat_kv(key, self.num_key_value_groups)
        value_low = repeat_kv(value_low, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + attention_mask[:, :, :query_len, :kv_len]
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        output = torch.matmul(weights, value_low)
        output = output.transpose(1, 2).contiguous().reshape(
            batch, query_len,
            self.num_heads * (self.rank_v // self.num_key_value_heads),
        )
        output = self.o_proj_up(output)
        return output, (weights if output_attentions else None), past


def replace_mistral_attention(
    decoder_layer,
    basis_kq: torch.Tensor,
    rank_kq: int,
    basis_v: torch.Tensor,
    rank_v: int,
    config,
    layer_idx: int,
) -> MistralEigenAttention:
    """Replace only one decoder layer's attention; keep native Mistral block logic."""
    original = decoder_layer.self_attn
    adapter = MistralEigenAttention(
        basis_kq, rank_kq, basis_v, rank_v, original, config, layer_idx
    )
    decoder_layer.self_attn = adapter
    return adapter
