# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch OPT model."""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.opt.configuration_opt import *

from transformers.models.opt.modeling_opt import *
import time

from models.dora_linear import DoRALinear

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]
class OPTEigenAttention(nn.Module):
    """Multi-headed attention (OPT-style) with low-rank Q/K/V + optional DoRA on low-rank projections."""

    def __init__(
        self,
        config,              # OPTConfig
        rank_kq: int,
        rank_v: int,
        is_decoder: bool = False,
        dora_rank: int = 8,  # DoRA internal rank
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias

        # --- Sanity checks ---
        if (self.embed_dim % self.num_heads) != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, num_heads={self.num_heads})."
            )
        if rank_kq % self.num_heads != 0:
            raise ValueError(
                f"rank_kq must be divisible by num_heads (got rank_kq={rank_kq}, num_heads={self.num_heads})."
            )
        if rank_v % self.num_heads != 0:
            raise ValueError(
                f"rank_v must be divisible by num_heads (got rank_v={rank_v}, num_heads={self.num_heads})."
            )
        if rank_kq < self.num_heads:
            raise ValueError(f"rank_kq must be >= num_heads (got rank_kq={rank_kq}, num_heads={self.num_heads}).")
        if rank_v < self.num_heads:
            raise ValueError(f"rank_v must be >= num_heads (got rank_v={rank_v}, num_heads={self.num_heads}).")

        # --- Original (full) per-head dim (kept for reference) ---
        self.head_dim = self.embed_dim // self.num_heads

        # --- Low-rank per-head dim (this is the TRUE dot-product dim!) ---
        self.rank_kq = rank_kq
        self.rank_v = rank_v
        #self.low_head_dim = rank_kq // self.num_heads

        # ✅ CRITICAL FIX: scaling must match low_head_dim, not original head_dim.
        #self.scaling = self.low_head_dim ** -0.5
        import os
        self.low_head_dim = rank_kq // self.num_heads

        if os.getenv("EIGEN_SCALE_ORIG", "0") == "1":
            # 原 OPT scaling: 1/sqrt(head_dim)
            self.scaling = (self.embed_dim // self.num_heads) ** -0.5
        else:
            # low-rank scaling: 1/sqrt(low_head_dim)
            self.scaling = self.low_head_dim ** -0.5

        self.is_decoder = is_decoder
        self.is_causal = True

        # --- Low-rank projections ---
        self.q_proj_low = nn.Linear(self.embed_dim, rank_kq, bias=self.enable_bias)
        self.k_proj_low = nn.Linear(self.embed_dim, rank_kq, bias=self.enable_bias)
        self.v_proj_low = nn.Linear(self.embed_dim, rank_v, bias=self.enable_bias)

        # --- Output projection: low-rank -> full ---
        self.out_proj_up = nn.Linear(rank_v, self.embed_dim, bias=self.enable_bias)
        import math
        self.vscale_log = torch.nn.Parameter(torch.tensor(math.log(0.03), dtype=torch.float32))
        import math, os
        init = float(os.getenv("EIGEN_VSCALE_INIT", os.getenv("EIGEN_VSCALE", "1.0")))
        init = max(init, 1e-6)
        #self.vscale_log = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float32))
        self.vscale_log = nn.Parameter(torch.tensor([math.log(init)], dtype=torch.float32))
        # --- DoRA modules (assumed defined elsewhere in your codebase) ---
        # If you sometimes want to disable DoRA, you can wrap with identity or gate it outside.
        #self.q_dora = DoRALinear(rank_kq, rank_kq, rank=dora_rank)
        #self.k_dora = DoRALinear(rank_kq, rank_kq, rank=dora_rank)
        #self.v_dora = DoRALinear(rank_v, rank_v, rank=dora_rank)

        self.q_dora = nn.Identity()
        self.k_dora = nn.Identity()
        self.v_dora = nn.Identity()

    def _new_shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, low_dim: int) -> torch.Tensor:
        # low_dim is total dim across heads in low-rank space (e.g., rank_kq or rank_v)
        if low_dim % self.num_heads != 0:
            raise ValueError(f"low_dim must be divisible by num_heads (low_dim={low_dim}, num_heads={self.num_heads})")
        return (
            tensor.view(bsz, seq_len, self.num_heads, low_dim // self.num_heads)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        hidden_states: (bsz, tgt_len, embed_dim)
        attention_mask: (bsz, 1, tgt_len, src_len) additive mask (0 or very negative)
        returns: (attn_output, attn_weights_reshaped or None, past_key_value or None)
        """
        import os
        import torch
        import torch.nn.functional as F

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        dev = hidden_states.device

        # -------------------------
        # Safety: move optional tensors to correct device
        # -------------------------
        if key_value_states is not None and key_value_states.device != dev:
            key_value_states = key_value_states.to(dev, non_blocking=True)
        if attention_mask is not None and attention_mask.device != dev:
            attention_mask = attention_mask.to(dev, non_blocking=True)
        if layer_head_mask is not None and layer_head_mask.device != dev:
            layer_head_mask = layer_head_mask.to(dev, non_blocking=True)
        if past_key_value is not None:
            pk, pv = past_key_value
            if pk.device != dev:
                pk = pk.to(dev, non_blocking=True)
            if pv.device != dev:
                pv = pv.to(dev, non_blocking=True)
            past_key_value = (pk, pv)

        # -------------------------
        # Q: proj -> (optional DoRA) -> scale
        # -------------------------
                # --- dtype align (avoid Float vs BF16 matmul) ---
       # if hidden_states.dtype != self.q_proj_low.weight.dtype:
       #     hidden_states = hidden_states.to(self.q_proj_low.weight.dtype)

        query_states = self.q_proj_low(hidden_states)
        query_states = self.q_dora(query_states)
        #query_states = query_states * query_states.new_tensor(self.scaling)
        scale = query_states.new_tensor(self.scaling)      # same dtype/device as query_states
        query_states = query_states * scale
        #query_states = query_states * self.scaling  # should be 1/sqrt(low_head_dim) (or env-switched in __init__)

        # -------------------------
        # K/V
        # -------------------------
        if is_cross_attention and past_key_value is not None:
            key_states, value_states = past_key_value

        elif is_cross_attention:
            kv_len = key_value_states.size(1)  # avoid -1 ambiguity
            key_states = self.k_proj_low(key_value_states)
            key_states = self.k_dora(key_states)
            key_states = self._new_shape(key_states, kv_len, bsz, self.rank_kq)

            value_states = self.v_proj_low(key_value_states)
            value_states = self.v_dora(value_states)
            value_states = self._new_shape(value_states, kv_len, bsz, self.rank_v)

        elif past_key_value is not None:
            key_states = self.k_proj_low(hidden_states)
            key_states = self.k_dora(key_states)
            key_states = self._new_shape(key_states, tgt_len, bsz, self.rank_kq)

            value_states = self.v_proj_low(hidden_states)
            value_states = self.v_dora(value_states)
            value_states = self._new_shape(value_states, tgt_len, bsz, self.rank_v)

            # concat past along sequence dim
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        else:
            key_states = self.k_proj_low(hidden_states)
            key_states = self.k_dora(key_states)
            key_states = self._new_shape(key_states, tgt_len, bsz, self.rank_kq)

            value_states = self.v_proj_low(hidden_states)
            value_states = self.v_dora(value_states)
            value_states = self._new_shape(value_states, tgt_len, bsz, self.rank_v)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        # -------------------------
        # reshape to (bsz*num_heads, seq, low_head_dim)
        # -------------------------
        head_dim_kq = self.rank_kq // self.num_heads
        head_dim_v  = self.rank_v  // self.num_heads

        proj_shape_kq = (bsz * self.num_heads, -1, head_dim_kq)
        proj_shape_v  = (bsz * self.num_heads, -1, head_dim_v)

        query_states = self._new_shape(query_states, tgt_len, bsz, self.rank_kq).contiguous().view(*proj_shape_kq)
        key_states   = key_states.contiguous().view(*proj_shape_kq)
        value_states = value_states.contiguous().view(*proj_shape_v)

        src_len = key_states.size(1)

        # -------------------------
        # attention scores
        # -------------------------
        if os.getenv("EIGEN_DTYPE_DEBUG", "0") == "1":
            if not hasattr(self, "_printed_dtype"):
                self._printed_dtype = True
                print("[DTYPE] hidden_states", hidden_states.dtype, hidden_states.device)
                print("[DTYPE] query_states", query_states.dtype, query_states.device)
                print("[DTYPE] key_states  ", key_states.dtype, key_states.device)
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f"Attention weights shape mismatch: {attn_weights.size()}")

        # -------------------------
        # mask
        # -------------------------
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f"Attention mask shape mismatch: {attention_mask.size()}")

            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask

            # clamp to min finite
            min_val = torch.finfo(attn_weights.dtype).min
            attn_weights = torch.maximum(attn_weights, attn_weights.new_full((), min_val))

            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # -------------------------
        # softmax
        # -------------------------
        if attn_weights.dtype in (torch.float16, torch.bfloat16):
            attn_probs = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
        else:
            attn_probs = torch.softmax(attn_weights, dim=-1)
        # debug: force uniform attention probs
        if os.getenv("EIGEN_BYPASS_PROBS", "0") == "1":
            attn_probs = torch.full_like(attn_probs, 1.0 / float(src_len))

        if self.dropout and self.training:
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=True)

        # head mask
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f"Layer head mask shape mismatch: {layer_head_mask.size()}")
            attn_probs = layer_head_mask.view(1, -1, 1, 1) * attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

        # optional return attentions
        if output_attentions:
            attn_weights_reshaped = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        # -------------------------
        # attention output
        # -------------------------
        attn_output = torch.matmul(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, head_dim_v):
            raise ValueError(f"attn_output shape mismatch: {attn_output.size()}")

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, head_dim_v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.rank_v)

        import os

        if os.getenv("EIGEN_SAVE_LOW", "0") == "1":
            # store low-dim value aggregation BEFORE out_proj_up
            self._last_attn_low = attn_output.detach()

        # ---- trainable vscale (log-space) ----
        # IMPORTANT: clamp in log-space BEFORE exp to avoid inf/NaN
        # vscale range: exp([-12, -0.5]) ≈ [6e-6, 0.607]
        vlog = self.vscale_log.clamp(min=-12.0, max=2.0)
        scale = torch.exp(vlog).to(attn_output.dtype)
        attn_output = attn_output * scale
        #上面两行02/23添加
        vscale_train = vlog.exp().to(dtype=attn_output.dtype, device=attn_output.device)

        # env vscale OPTIONAL multiplier (ablation), default=1.0
        env_v = float(os.getenv("EIGEN_VSCALE", "1.0"))
        env_v = attn_output.new_tensor(env_v)

        attn_output = attn_output * vscale_train * env_v

        # debug: zero BEFORE out_proj_up (isolates out_proj_up bias/behavior)
        if os.getenv("EIGEN_BYPASS_BEFORE_UP", "0") == "1":
            attn_output = torch.zeros_like(attn_output)

        # output proj
        attn_output = attn_output.to(self.out_proj_up.weight.dtype)
        attn_output = self.out_proj_up(attn_output)

        # debug: zero AFTER out_proj_up (full bypass)
        if os.getenv("EIGEN_BYPASS_ATTN", "0") == "1":
            attn_output = torch.zeros_like(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
'''
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj_low(hidden_states) * self.scaling

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self.k_proj_low(key_value_states)
            key_states = self._new_shape(key_states, -1, bsz, self.rank_kq)
            value_states = self._new_shape(self.v_proj_low(key_value_states), -1, bsz, self.rank_v)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            # bsz, seq_len, self.num_heads, self.head_dim -> bsz, self.num_heads, seq_len, self.head_dim
            key_states = self.k_proj_low(hidden_states)
            key_states = self._new_shape(key_states, -1, bsz, self.rank_kq)
            value_states = self.v_proj_low(hidden_states)
            value_states = self._new_shape(value_states, -1, bsz, self.rank_v)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self.k_proj_low(hidden_states)
            key_states = self._new_shape(key_states, -1, bsz, self.rank_kq)

            value_states = self.v_proj_low(hidden_states)
            value_states = self._new_shape(value_states, -1, bsz, self.rank_v)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape_kq = (bsz * self.num_heads, -1, self.rank_kq//self.num_heads) 
        proj_shape_v = (bsz * self.num_heads, -1, self.rank_v//self.num_heads) 

        query_states = self._new_shape(query_states, tgt_len, bsz, self.rank_kq).view(*proj_shape_kq)
        key_states = key_states.view(*proj_shape_kq)
        value_states = value_states.view(*proj_shape_v)
        # value_states = self.v_proj_up(value_states)

        src_len = key_states.size(1)
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(torch.float16)
        else:
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attn_weights.dtype)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        # attention shape bsz * self.num_heads, tgt_len, src_len
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.rank_v // self.num_heads):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.rank_v // self.num_heads)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.rank_v // self.num_heads)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.rank_v)
        attn_output = self.out_proj_up(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
'''


OPT_ATTENTION_CLASSES = {
    "eager": OPTEigenAttention,
    "flash_attention_2": OptFlashAttention2,
}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# 你工程里应该已经有：
# - OPTConfig
# - ACT2FN
# - OPT_ATTENTION_CLASSES  (eager -> OPTEigenAttention)
# 这里不重复定义。

import torch
import torch.nn as nn
from typing import Optional, Tuple

class OPTDecoderLayer_EigenAttn(nn.Module):
    def __init__(self, low_rank_config, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        # only eager
        assert config._attn_implementation == "eager"
        self.self_attn = OPT_ATTENTION_CLASSES[config._attn_implementation](
            config=config,
            is_decoder=True,
            rank_kq=int(low_rank_config[0]),
            rank_v=int(low_rank_config[1]),
        )

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        # --- eps fallback (你的 OPTConfig 没 layer_norm_eps 就用 layer_norm_epsilon) ---
        ln_eps = getattr(config, "layer_norm_eps", None)
        if ln_eps is None:
            ln_eps = getattr(config, "layer_norm_epsilon", 1e-5)
        affine = getattr(config, "layer_norm_elementwise_affine", True)

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            eps=ln_eps,
            elementwise_affine=affine,
        )

        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)

        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            eps=ln_eps,
            elementwise_affine=affine,
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        import torch
        import torch.nn as nn

        def _move(x, dev, dtype=None):
            if x is None:
                return None
            if dtype is None:
                if x.device != dev:
                    return x.to(dev, non_blocking=True)
                return x
            if x.device != dev or x.dtype != dtype:
                return x.to(device=dev, dtype=dtype, non_blocking=True)
            return x

        # 以本层 LN 的权重设备为“默认层设备”
        layer_dev = self.self_attn_layer_norm.weight.device
        if layer_dev.type == "meta":
            layer_dev = hidden_states.device

        layer_dtype = self.self_attn_layer_norm.weight.dtype

        hidden_states = _move(hidden_states, layer_dev, layer_dtype)
        attention_mask = _move(attention_mask, layer_dev)
        layer_head_mask = _move(layer_head_mask, layer_dev)
        if past_key_value is not None:
            past_key_value = tuple(_move(t, layer_dev) for t in past_key_value)

        residual = hidden_states

        # ---- LN before attn ----
        if self.do_layer_norm_before:
            ln = self.self_attn_layer_norm
            hs_dev = ln.weight.device
            hidden_states = _move(hidden_states, hs_dev, ln.weight.dtype)
            hidden_states = ln(hidden_states)

        # ---- Self-attn ----
        # 对齐到 attn 参数所在设备（更稳）
        attn_dev = next(self.self_attn.parameters()).device
        hidden_states = _move(hidden_states, attn_dev, next(self.self_attn.parameters()).dtype)
        if attention_mask is not None:
            attention_mask = _move(attention_mask, attn_dev)
        if layer_head_mask is not None:
            layer_head_mask = _move(layer_head_mask, attn_dev)
        if past_key_value is not None:
            past_key_value = tuple(_move(t, attn_dev) for t in past_key_value)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # ---- LN after attn ----
        if not self.do_layer_norm_before:
            ln = self.self_attn_layer_norm
            hs_dev = ln.weight.device
            hidden_states = _move(hidden_states, hs_dev, ln.weight.dtype)
            hidden_states = ln(hidden_states)

        # ---- FFN ----
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        if self.do_layer_norm_before:
            ln = self.final_layer_norm
            hs_dev = ln.weight.device
            hidden_states = _move(hidden_states, hs_dev, ln.weight.dtype)
            hidden_states = ln(hidden_states)

        # fc1
        fc1_dev = self.fc1.weight.device
        hidden_states = _move(hidden_states, fc1_dev, self.fc1.weight.dtype)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        # fc2
        fc2_dev = self.fc2.weight.device
        hidden_states = _move(hidden_states, fc2_dev, self.fc2.weight.dtype)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual.to(hidden_states.device) + hidden_states).view(hidden_states_shape)

        if not self.do_layer_norm_before:
            ln = self.final_layer_norm
            hs_dev = ln.weight.device
            hidden_states = _move(hidden_states, hs_dev, ln.weight.dtype)
            hidden_states = ln(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn

from transformers.models.opt.configuration_opt import OPTConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.opt.modeling_opt import (
    OPTLearnedPositionalEmbedding,
    _prepare_4d_causal_attention_mask,
)
from transformers.models.opt.modeling_opt import OPTPreTrainedModel

import logging
logger = logging.getLogger(__name__)


class OPTDecoder_EigenAttn(OPTPreTrainedModel):
    """
    OPT decoder with EigenAttn low-rank attention layers.

    NOTE:
    - forward() 必须是类方法（你之前 forward 写到 class 外会直接导致 accelerate hook 异常）。
    - 不要在 decoder 顶层再挂 q_dora/k_dora/v_dora（会导致你日志里 decoder.k_dora.* 以及 NaN/Inf 噪声）。
    """

    def __init__(self, config: OPTConfig, low_rank_config, use_internal_dora: bool = False):
        super().__init__(config)

        self.low_rank_config = low_rank_config
        self.use_internal_dora = use_internal_dora

        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        # embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        # proj in/out
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_in = None
            self.project_out = None

        # LayerNorm eps 兼容：OPTConfig 有的版本叫 layer_norm_epsilon
        ln_eps = getattr(config, "layer_norm_eps", None)
        if ln_eps is None:
            ln_eps = getattr(config, "layer_norm_epsilon", 1e-5)

        affine = getattr(config, "layer_norm_elementwise_affine", True)

        if config.do_layer_norm_before and not getattr(config, "_remove_final_layer_norm", False):
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                eps=ln_eps,
                elementwise_affine=affine,
            )
        else:
            self.final_layer_norm = None

        # layers
        self.layers = nn.ModuleList(
            [
                OPTDecoderLayer_EigenAttn(low_rank_config=low_rank_config[i], config=config)
                for i in range(config.num_hidden_layers)
            ]
        )

        self._use_flash_attention_2 = getattr(config, "_attn_implementation", "eager") == "flash_attention_2"
        self.gradient_checkpointing = False

        # ✅ 不要在 decoder 顶层再挂 DoRA，也不要在这里覆盖 layer.self_attn
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        import torch

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ---- retrieve input_ids and inputs_embeds ----
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # ---- move ids/mask onto embedding device ----
        emb_dev = self.embed_tokens.weight.device
        if input_ids is not None and input_ids.device != emb_dev:
            input_ids = input_ids.to(emb_dev, non_blocking=True)
        if attention_mask is not None and attention_mask.device != emb_dev:
            attention_mask = attention_mask.to(emb_dev, non_blocking=True)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        mask_seq_length = past_key_values_length + seq_length

        # ✅ 防止 embed_positions index 越界（你之前那个 indexSelectLargeIndex 就是这类）
        if mask_seq_length > self.max_target_positions:
            raise ValueError(
                f"mask_seq_length={mask_seq_length} > max_position_embeddings={self.max_target_positions} "
                f"(past={past_key_values_length}, cur={seq_length})"
            )

        # ---- attention mask ----
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        else:
            if attention_mask.device != inputs_embeds.device:
                attention_mask = attention_mask.to(inputs_embeds.device, non_blocking=True)
            if attention_mask.shape[1] != mask_seq_length:
                raise ValueError(f"attention_mask length {attention_mask.shape[1]} != required {mask_seq_length}")

        # ---- causal mask ----
        if self._use_flash_attention_2:
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # ---- position embeds ----
        # 直接把 pos_embeds 搬回 inputs_embeds.device
        pos_mask = attention_mask
        pos_embeds = self.embed_positions(pos_mask, past_key_values_length)
        if pos_embeds.device != inputs_embeds.device:
            pos_embeds = pos_embeds.to(inputs_embeds.device, non_blocking=True)

        # ---- project_in ----
        if self.project_in is not None:
            pin_dev = self.project_in.weight.device
            if inputs_embeds.device != pin_dev:
                inputs_embeds = inputs_embeds.to(pin_dev, non_blocking=True)
            if pos_embeds.device != inputs_embeds.device:
                pos_embeds = pos_embeds.to(inputs_embeds.device, non_blocking=True)
            if causal_attention_mask is not None and causal_attention_mask.device != inputs_embeds.device:
                causal_attention_mask = causal_attention_mask.to(inputs_embeds.device, non_blocking=True)
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        # ---- gradient checkpointing vs cache ----
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting use_cache=False.")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if head_mask is not None and head_mask.size()[0] != len(self.layers):
            raise ValueError(f"head_mask should have {len(self.layers)} layers, got {head_mask.size()[0]}")

        # =========================================================
        # ✅ 核心修复：layer_dev 直接用该层 LN 的 weight.device
        # =========================================================
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([], device=hidden_states.device)
                if dropout_probability < self.layerdrop:
                    continue

            # ✅ 不要用 hook 推断，直接以 LN 权重所在卡为准
            layer_dev = decoder_layer.self_attn_layer_norm.weight.device
            if layer_dev.type == "meta":
                layer_dev = hidden_states.device  # 兜底

            if hidden_states.device != layer_dev:
                hidden_states = hidden_states.to(layer_dev, non_blocking=True)

            if causal_attention_mask is not None and causal_attention_mask.device != layer_dev:
                causal_attention_mask = causal_attention_mask.to(layer_dev, non_blocking=True)

            layer_hm = head_mask[idx] if head_mask is not None else None
            if layer_hm is not None and layer_hm.device != layer_dev:
                layer_hm = layer_hm.to(layer_dev, non_blocking=True)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if past_key_value is not None:
                past_key_value = tuple(t.to(layer_dev, non_blocking=True) for t in past_key_value)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_attention_mask,
                layer_head_mask=layer_hm,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # ---- final LN / project_out ----
        if self.final_layer_norm is not None:
            ln_dev = self.final_layer_norm.weight.device
            if hidden_states.device != ln_dev:
                hidden_states = hidden_states.to(ln_dev, non_blocking=True)
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            po_dev = self.project_out.weight.device
            if hidden_states.device != po_dev:
                hidden_states = hidden_states.to(po_dev, non_blocking=True)
            hidden_states = self.project_out(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel_EigenAttn(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig, low_rank_config):
        super().__init__(config)
        self.decoder = OPTDecoder_EigenAttn(config = config, low_rank_config = low_rank_config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM_EigenAttn(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, low_rank_config):
        super().__init__(config, low_rank_config)
        self.model = OPTModel_EigenAttn(config, low_rank_config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2] ###################################################################################################################################################
            # past_length = past_key_values[-1][-1]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

