# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union, Dict, List, Any, Set, Tuple
import functools
import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple, logging

try:  # transformers < 4.55 exported LossKwargs; it was removed upstream afterwards.
    from transformers.utils import LossKwargs  # type: ignore
except ImportError:  # pragma: no cover - depends on installed transformers version
    from typing import TypedDict

    class LossKwargs(TypedDict, total=False):
        """Local stand-in for the removed upstream ``transformers.utils.LossKwargs``."""

        num_items_in_batch: Optional[int]

from .configuration_llama import LlamaConfig
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from utils.cache_utils import SlimInferCache, HsCache
from utils.cache_utils import _create_sliminfer_attention_mask
import utils.cache_utils as sp_utils
from utils.processor import SlimInferLayerProcessor, SlimInferInferenceState

CAPTURED_LAYER_ATTENTION_LIST = []
ENABLE_LAYER_ATTENTION_CAPTURING = False

DEFAULT_SLIMINFER_CONFIG = {
    "block_size": 128,
    "semantic_kernel_size": None,
    "swap_threshold": 0.9,
    "pruning_window_len": 32,
    "pruning_method": "ratio",
    "pruning_config": {},
}

logger = logging.get_logger(__name__)

@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.config.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.config.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.config.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.config.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.config.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.config.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
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

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sliminfer_config: Optional[Dict[str, Any]] = None,
        hs_cache: Optional[HsCache] = None,
        all_original_prompt_indices_abs: Optional[List[int]] = None,
        is_decode_phase: bool = False,
        decoded_token_abs_idx: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if isinstance(past_key_values, SlimInferCache) or sliminfer_config is not None:
            return self.slim_llama_model_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                sliminfer_config=sliminfer_config,
                hs_cache=hs_cache,
                all_original_prompt_indices_abs=all_original_prompt_indices_abs,
                is_decode_phase=is_decode_phase,
                decoded_token_abs_idx=decoded_token_abs_idx,
                **flash_attn_kwargs,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        if isinstance(past_key_values, SlimInferCache):
            return self.slim_llama_model_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                sliminfer_config=sliminfer_config,
                hs_cache=hs_cache,
                all_original_prompt_indices_abs=all_original_prompt_indices_abs,
                is_decode_phase=is_decode_phase,
                decoded_token_abs_idx=decoded_token_abs_idx,
                **flash_attn_kwargs,
            )

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def slim_llama_model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[SlimInferCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sliminfer_config: Optional[Dict[str, Any]] = None,
        hs_cache: Optional[HsCache] = None,
        all_original_prompt_indices_abs: Optional[List[int]] = None,
        is_decode_phase: bool = False,
        decoded_token_abs_idx: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        config = DEFAULT_SLIMINFER_CONFIG.copy()
        if sliminfer_config is not None:
            config.update(sliminfer_config)
        
        global CAPTURED_LAYER_ATTENTION_LIST
            
        if config["block_size"] is not None:
            sp_utils._BLOCK_SIZE = config["block_size"]
        if config["swap_threshold"] is not None:
            sp_utils._SWAP_THRESHOLD = config["swap_threshold"]
        if config["pruning_window_len"] is not None:
            sp_utils._PRUNING_WINDOW_LEN = config["pruning_window_len"]
        if config["semantic_kernel_size"] is not None:
            sp_utils._SEMANTIC_KERNEL_SIZE = config["semantic_kernel_size"]

        # --- 1. 初始设置和输入验证 ---
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            bsz, seq_length = input_ids.shape[:2]
            device = input_ids.device
        elif inputs_embeds is not None:
            bsz, seq_length = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        else:
            raise ValueError("input_ids and inputs_embeds cannot both be None")

        if bsz > 1:
            warnings.warn("currently Sliminfer only supports batch size = 1 during decoding phase.")
            
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # --- 2. initial cache and layer processor ---
        slim_kv_cache = past_key_values if past_key_values is not None else SlimInferCache(device=device)
        current_hs_cache = hs_cache if hs_cache is not None else HsCache()
        layer_processor = SlimInferLayerProcessor()

        _current_input_abs_indices_list: List[int]
        if cache_position is not None:
            _current_input_abs_indices_list = cache_position.squeeze(0).tolist()
        elif is_decode_phase and decoded_token_abs_idx is not None:
            _current_input_abs_indices_list = [decoded_token_abs_idx]
        else:
            _current_input_abs_indices_list = list(range(seq_length))
        
        initial_inputs = {
            "inputs_embeds": inputs_embeds,
            "current_input_abs_indices_list": _current_input_abs_indices_list,
            "input_abs_to_batch_idx_map": {idx: i for i, idx in enumerate(_current_input_abs_indices_list)},
        }
        
        assert all_original_prompt_indices_abs is not None, "all_original_prompt_indices_abs must be provided"
        inference_state = SlimInferInferenceState(all_original_prompt_indices_abs, is_decode_phase, device, bsz, decoded_token_abs_idx)

        # --- 3. prepare attention mask ---
        inference_state.current_attention_mask = attention_mask
        if not is_decode_phase:
            # Prefill phase doesn't use attn mask
            inference_state.current_attention_mask = None
        elif is_decode_phase and decoded_token_abs_idx is not None:
            # build a simple all-1 attention mask for the decoded token in decode phase, will rebuild later if needed
            total_seq_len = decoded_token_abs_idx + 1
            inference_state.current_attention_mask = torch.ones((1, 1, 1, total_seq_len), dtype=inputs_embeds.dtype, device=device)

        all_layer_hidden_states_list = [] if output_hidden_states else None
        all_layer_self_attns_list = [] if output_attentions else None

        # --- 4. Inference loop ---
        for layer_idx, decoder_layer in enumerate(self.layers):
            # 4.1 prepare layer inputs
            layer_inputs = layer_processor.prepare_layer_inputs(
                layer_idx, inference_state, initial_inputs, slim_kv_cache, current_hs_cache
            )
            hs_for_query = layer_inputs["hs_for_query"]
            query_L_abs_indices_list = layer_inputs["query_abs_indices_list"]
            query_L_len = len(query_L_abs_indices_list)
            
            # 4.2 update Position ID
            restored_pos_ids = layer_inputs["restored_position_ids"]
            
            # --- Attention block ---
            layer_attention = decoder_layer.self_attn
            hs_norm_for_attn = decoder_layer.input_layernorm(hs_for_query)
            q_squery, k_squery, v_squery = (
                layer_attention.q_proj(hs_norm_for_attn),
                layer_attention.k_proj(hs_norm_for_attn),
                layer_attention.v_proj(hs_norm_for_attn),
            )
            q_squery = q_squery.view(bsz, query_L_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
            k_squery = k_squery.view(bsz, query_L_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
            v_squery = v_squery.view(bsz, query_L_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)

            cos, sin = None, None
            if (inference_state.current_position_embeddings is None and query_L_abs_indices_list) or \
                inference_state.need_to_rebuild_position_embeddings_flag:
                # Case 1: creaate position embeddings for the first time, or need to rebuild
                current_position_ids = torch.tensor(query_L_abs_indices_list, device=device, dtype=torch.long).unsqueeze(0)
                cos, sin = self.rotary_emb(v_squery, position_ids=current_position_ids)
                inference_state.current_position_embeddings = (cos, sin)
                inference_state.need_to_rebuild_position_embeddings_flag = False
            elif restored_pos_ids is not None:
                # Case 2: restored tokens from HsCache, need to supplement position_ids and embeddings, and no need to load from kv cache
                cos_new, sin_new = self.rotary_emb(v_squery, position_ids=restored_pos_ids)
                if inference_state.current_position_embeddings is not None:
                    old_cos, old_sin = inference_state.current_position_embeddings
                    cos = torch.cat([old_cos, cos_new], dim=1)
                    sin = torch.cat([old_sin, sin_new], dim=1)
                else:
                    cos, sin = cos_new, sin_new
                inference_state.current_position_embeddings = (cos, sin)
            
            if inference_state.current_position_embeddings is not None:
                cos, sin = inference_state.current_position_embeddings
            q_squery, k_squery = apply_rotary_pos_emb(q_squery, k_squery, cos, sin)
            # 4.4 update kv cache
            if use_cache:
                if is_decode_phase and decoded_token_abs_idx is not None:
                    slim_kv_cache.update(
                        key_states=k_squery[:, :, 0:1, :], # update only the current decoding token
                        value_states=v_squery[:, :, 0:1, :], # update only the current decoding token
                        layer_idx=layer_idx,
                        is_prompt_token=False,
                        abs_or_block_ids=decoded_token_abs_idx
                    )
                    if inference_state.query_block_id_list:
                        slim_kv_cache.update(
                            key_states=k_squery[:, :, 1:, :], # update the remaining prompt K
                            value_states=v_squery[:, :, 1:, :], # update the remaining prompt V
                            layer_idx=layer_idx,
                            is_prompt_token=True,
                            abs_or_block_ids=inference_state.query_block_id_list
                        )
                elif not is_decode_phase: # Prefill phase
                    slim_kv_cache.update(
                        key_states=k_squery,
                        value_states=v_squery,
                        layer_idx=layer_idx,
                        is_prompt_token=True,
                        abs_or_block_ids=sorted(inference_state.current_APIA_block_set),
                        prompt_seq_len=inference_state.original_prompt_len
                    )

            full_k, full_v, kv_pool_indices = slim_kv_cache.get_kv_pool_for_attention(layer_idx)

            if is_decode_phase and inference_state.need_to_rebuild_attention_mask_flag:
                inference_state.current_attention_mask = _create_sliminfer_attention_mask(query_L_abs_indices_list, kv_pool_indices, device, dtype=q_squery.dtype)
                inference_state.need_to_rebuild_attention_mask_flag = False

            # 4.6 attention calculation
            n_rep = self.config.num_attention_heads // self.config.num_key_value_heads
            use_flash_attention_2 = self.config._attn_implementation == "flash_attention_2" and \
                (inference_state.query_block_id_list is None or inference_state.query_block_id_list == [])
                
            # pre_exp: capture attention weights for the target token during decode phase
            is_capturing_target_token = (
                ENABLE_LAYER_ATTENTION_CAPTURING and
                is_decode_phase and
                decoded_token_abs_idx is not None and
                all_original_prompt_indices_abs is not None and
                decoded_token_abs_idx == len(all_original_prompt_indices_abs)
            )
            if is_capturing_target_token:
                attn_key_pool, attn_value_pool = repeat_kv(full_k, n_rep), repeat_kv(full_v, n_rep)
                attn_weights = torch.matmul(q_squery, attn_key_pool.transpose(2, 3)) / math.sqrt(self.config.head_dim)
                weights_after_mask = attn_weights + inference_state.current_attention_mask
                final_weights = nn.functional.softmax(weights_after_mask, dim=-1, dtype=torch.bfloat16)
                avg_head_weights = final_weights[0, :, 0, :].mean(dim=0).cpu()
                indices_for_weights = torch.tensor(kv_pool_indices, dtype=torch.long, device='cpu')
                CAPTURED_LAYER_ATTENTION_LIST.append((avg_head_weights, indices_for_weights))
            
            if self.config._attn_implementation == "eager":
                attn_key_pool, attn_value_pool = repeat_kv(full_k, n_rep), repeat_kv(full_v, n_rep)
                attn_weights = torch.matmul(q_squery, attn_key_pool.transpose(2, 3)) / math.sqrt(self.config.head_dim)
                attn_weights = attn_weights + inference_state.current_attention_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_squery.dtype)
                attn_output = torch.matmul(attn_weights, attn_value_pool)
            elif self.config._attn_implementation == "sdpa" or self.config._attn_implementation == "flash_attention_2" and use_flash_attention_2 == False:
                q_squery = q_squery.contiguous()
                attn_key_pool, attn_value_pool = repeat_kv(full_k, n_rep), repeat_kv(full_v, n_rep)
                attn_key_pool = attn_key_pool.contiguous()
                attn_value_pool = attn_value_pool.contiguous()
                sdpa_mask = inference_state.current_attention_mask
                # Prefill leaves current_attention_mask as None because the
                # flash_attention_2 branch supplies causality through is_causal=True.
                # SDPA must do the same: with a None mask and is_causal=False it would
                # compute *bidirectional* prefill attention, letting every prompt token
                # see its own future and destroying the LM conditioning.
                # Query and KV cover the same ordered token set here (q_len == kv_len), so
                # SDPA's top-left aligned causal mask is exactly the standard causal mask.
                sdpa_is_causal = False
                if sdpa_mask is None:
                    if q_squery.shape[-2] != attn_key_pool.shape[-2]:
                        raise ValueError(
                            "SlimInfer SDPA attention received no explicit mask with "
                            f"q_len={q_squery.shape[-2]} != kv_len={attn_key_pool.shape[-2]}; "
                            "causal alignment would be ambiguous. Refusing to compute "
                            "silently-wrong attention."
                        )
                    sdpa_is_causal = True
                attn_output = F.scaled_dot_product_attention(
                    q_squery,
                    attn_key_pool,
                    attn_value_pool,
                    attn_mask=sdpa_mask,
                    dropout_p=0.0,
                    is_causal=sdpa_is_causal
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
            elif self.config._attn_implementation == "flash_attention_2" and use_flash_attention_2:
                seq_len = q_squery.shape[2]
                query = q_squery.transpose(1, 2)
                key = full_k.transpose(1, 2)
                value = full_v.transpose(1, 2)
                target_dtype = None
                if query.dtype == torch.float32:
                    if torch.is_autocast_enabled():
                        target_dtype = torch.get_autocast_gpu_dtype()

                attn_output = _flash_attention_forward(
                    query,
                    key,
                    value,
                    None,
                    query_length=seq_len,
                    is_causal=True,
                    dropout=0.0,
                    softmax_scale=None,
                    sliding_window=None,
                    softcap=None,
                    use_top_left_mask=None,
                    target_dtype=target_dtype,
                    attn_implementation="flash_attention_2",
                    **flash_attn_kwargs,
                )
            else:
                raise ValueError("Unsupported Attention implementation.")            
            attn_output = attn_output.view(bsz, query_L_len, -1)
            attn_output = layer_attention.o_proj(attn_output)
            
            next_APIA_set, next_APIA_block_set = layer_processor.plan_next_layer_state(
                layer_idx, inference_state, q_squery, n_rep,
                layer_inputs, config, slim_kv_cache
            )
            
            if inference_state.need_to_swap_kv_cache_flag and layer_idx < self.config.num_hidden_layers - 1:
                slim_kv_cache.swap_prompt_cache(layer_idx + 1, next_APIA_block_set)

            # 4.7 residual and MLP
            hs_after_attn = hs_for_query + attn_output
            layer_output_hs = hs_after_attn + decoder_layer.mlp(decoder_layer.post_attention_layernorm(hs_after_attn))
            
            layer_processor.apply_pruning_and_cache_hs(
                layer_idx, inference_state, layer_output_hs, layer_inputs,
                current_hs_cache, kv_pool_indices,
                next_APIA_set, next_APIA_block_set
            )

        # --- 5. final process ---
        final_hs_for_model_output = inference_state.hidden_states_prev_layer
        if not is_decode_phase and final_hs_for_model_output is not None:
            print(f"Prefill last_hs shape: {final_hs_for_model_output.shape}")
        last_input_abs_idx = max(_current_input_abs_indices_list) 
        
        if final_hs_for_model_output is not None:
            last_token_relative_idx = inference_state.prev_mlp_abs_to_relative_idx_map.get(last_input_abs_idx)
            if last_token_relative_idx is not None:
                final_hs_for_model_output = final_hs_for_model_output[:, last_token_relative_idx:last_token_relative_idx+1, :]
            else:
                raise ValueError("The hidden state for the final token was not found in the output, it may have been pruned incorrectly.")
        else:
            raise ValueError("No valid final hidden state for model output.")

        final_hs_for_model_output = self.norm(final_hs_for_model_output)
        
        if is_decode_phase:
            past_key_values.seq_len += 1

        if not return_dict:
            return tuple(v for v in [final_hs_for_model_output, slim_kv_cache, all_layer_hidden_states_list, all_layer_self_attns_list, current_hs_cache] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=final_hs_for_model_output,
            past_key_values=slim_kv_cache,
            hidden_states=all_layer_hidden_states_list,
            attentions=all_layer_self_attns_list,
        )

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        sliminfer_config: Optional[Dict[str, Any]] = None,
        hs_cache: Optional[HsCache] = None,
        all_original_prompt_indices_abs: Optional[List[int]] = None,
        is_decode_phase: bool = False,
        decoded_token_abs_idx: Optional[int] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        slim_kwargs = {}
        if isinstance(past_key_values, SlimInferCache) or sliminfer_config is not None:
            slim_kwargs = {
                "sliminfer_config": sliminfer_config,
                "hs_cache": hs_cache,
                "all_original_prompt_indices_abs": all_original_prompt_indices_abs,
                "is_decode_phase": is_decode_phase,
                "decoded_token_abs_idx": decoded_token_abs_idx,
            }

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
            **slim_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None and isinstance(past_key_values, SlimInferCache) and past_key_values.prompt_total_len:
            model_inputs = {"input_ids": input_ids[:, -1:]}
            decoded_token_abs_idx = past_key_values.get_seq_length()

            model_inputs.update({
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": kwargs.get("attention_mask"),
                "is_decode_phase": True,
                "decoded_token_abs_idx": decoded_token_abs_idx,
                "all_original_prompt_indices_abs": kwargs.get("all_original_prompt_indices_abs"),
                "sliminfer_config": kwargs.get("sliminfer_config"),
                "hs_cache": kwargs.get("hs_cache"),
                "cache_position": torch.tensor([[decoded_token_abs_idx]], device=input_ids.device)
            })
            return model_inputs
        else:
            model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values, **kwargs)
            if kwargs.get("sliminfer_config") is not None and past_key_values is None:
                prompt_len = input_ids.shape[1]
                model_inputs.update({
                    # "past_key_values": SlimInferCache(device=input_ids.device),
                    # "hs_cache": HsCache(),
                    # "all_original_prompt_indices_abs": list(range(prompt_len)),
                    "is_decode_phase": False,
                    "sliminfer_config": kwargs.get("sliminfer_config"),
                    "cache_position": torch.arange(prompt_len, device=input_ids.device).unsqueeze(0)
                })
            return model_inputs


class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
