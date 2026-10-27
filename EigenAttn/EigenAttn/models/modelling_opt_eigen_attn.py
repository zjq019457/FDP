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
import os
import math
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

# Optional low-rank rebalancing (orthogonal + column); forward uses duck-typing for DoRA wrapper.
try:
    from models.low_rank_balance import OrthogonalRebalanceRank, LowRankColumnBalancer  # noqa: F401
except Exception:
    OrthogonalRebalanceRank = None  # type: ignore
    LowRankColumnBalancer = None  # type: ignore

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
        self.dropout = float(getattr(config, "attention_dropout", 0.0))
        self.enable_bias = bool(getattr(config, "enable_bias", True))

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

        # --- dims ---
        self.head_dim = self.embed_dim // self.num_heads  # 原 OPT head_dim
        self.rank_kq = int(rank_kq)
        self.rank_v = int(rank_v)

        # low-rank per-head dim (真正 QK dot-product 的维度)
        self.low_head_dim = self.rank_kq // self.num_heads

        # scaling：默认用原 OPT / EigenAttn 论文的 1/sqrt(head_dim)。
        # 因为共享 Q/K basis 下 Q·K^T 在低秩空间的数值幅度与 full-dim 基本一致，
        # 若改用 1/sqrt(low_head_dim) 会把 attention logits 错误放大（rank_kq=280 时 ~5×），
        # softmax 退化成 near-one-hot，PPL 会从 +0.06 炸到 +2.6（已验证）。
        # 想切到低秩维度 scaling 方便对照时，设 EIGEN_SCALE_LOW=1。
        if os.getenv("EIGEN_SCALE_LOW", "0") == "1":
            self.scaling = (self.low_head_dim) ** -0.5
        else:
            self.scaling = (self.head_dim) ** -0.5

        self.is_decoder = bool(is_decoder)
        self.is_causal = True

        # --- Low-rank projections ---
        self.q_proj_low = nn.Linear(self.embed_dim, self.rank_kq, bias=self.enable_bias)
        self.k_proj_low = nn.Linear(self.embed_dim, self.rank_kq, bias=self.enable_bias)
        self.v_proj_low = nn.Linear(self.embed_dim, self.rank_v,  bias=self.enable_bias)

        # --- Learnable rotation in V low-rank space (acts as learnable basis) ---
        self.v_rot = nn.Linear(self.rank_v, self.rank_v, bias=False)
        with torch.no_grad():
            self.v_rot.weight.copy_(torch.eye(self.rank_v))

        # --- Output projection: low-rank -> full ---
        self.out_proj_up = nn.Linear(self.rank_v, self.embed_dim, bias=self.enable_bias)

        # --- trainable vscale (log-space) ---
        # init: prefer EIGEN_VSCALE_INIT, fallback EIGEN_VSCALE (若你把 EIGEN_VSCALE 当 multiplier 用，建议只设置 INIT)
        init = float(os.getenv("EIGEN_VSCALE_INIT", "0.03"))
        init = max(init, 1e-6)
        self.vscale_log = nn.Parameter(torch.tensor([math.log(init)], dtype=torch.float32))

        # --- DoRA (optional) ---
        use_dora = os.getenv("EIGEN_USE_DORA", "0") == "1"
        if use_dora and (DoRALinear is not None):
            self.q_dora = DoRALinear(self.rank_kq, self.rank_kq, rank=dora_rank)
            self.k_dora = DoRALinear(self.rank_kq, self.rank_kq, rank=dora_rank)
            self.v_dora = DoRALinear(self.rank_v,  self.rank_v,  rank=dora_rank)
        else:
            self.q_dora = nn.Identity()
            self.k_dora = nn.Identity()
            self.v_dora = nn.Identity()

        # --- Optional residual on full hidden dim after out_proj_up (low-rank bottleneck) ---
        # 构建时：EIGEN_O_RESIDUAL=1 才创建参数；前向时可用 EIGEN_O_RESIDUAL=0 临时关闭（模块仍在）。
        self.use_o_residual = os.getenv("EIGEN_O_RESIDUAL", "0") == "1"
        r_o = max(1, int(os.getenv("EIGEN_O_RESIDUAL_RANK", "32")))
        if self.use_o_residual:
            self.o_residual_down = nn.Linear(self.embed_dim, r_o, bias=False)
            self.o_residual_up = nn.Linear(r_o, self.embed_dim, bias=False)
            # 稳定性：初始化为全 0，使开启残差前向等价“不加残差”
            with torch.no_grad():
                self.o_residual_down.weight.zero_()
                self.o_residual_up.weight.zero_()
            _s = float(os.getenv("EIGEN_O_RESIDUAL_SCALE", "0.0"))
            # 形状 [1]：避免 transformers lazy init 里 torch.empty(*size) 对标量参数报错
            self.o_residual_scale = nn.Parameter(torch.tensor([_s], dtype=torch.float32))
        else:
            self.o_residual_down = None  # type: ignore[assignment]
            self.o_residual_up = None  # type: ignore[assignment]
            self.o_residual_scale = None  # type: ignore[assignment]

        # --- 方案 C：attention 输出侧低秩 adapter（out_proj_up 之后；可与 o_residual 叠用）---
        self.use_output_adapter = os.getenv("EIGEN_OUTPUT_ADAPTER", "0") == "1"
        r_ad = max(1, int(os.getenv("EIGEN_OUTPUT_ADAPTER_RANK", "64")))
        if self.use_output_adapter:
            self.output_adapter_down = nn.Linear(self.embed_dim, r_ad, bias=False)
            self.output_adapter_up = nn.Linear(r_ad, self.embed_dim, bias=False)
            # 稳定性：初始化为全 0，使开启 adapter 前向等价“不加 adapter”
            with torch.no_grad():
                self.output_adapter_down.weight.zero_()
                self.output_adapter_up.weight.zero_()
            _as = float(os.getenv("EIGEN_OUTPUT_ADAPTER_SCALE", "0.0"))
            self.output_adapter_scale = nn.Parameter(torch.tensor([_as], dtype=torch.float32))
        else:
            self.output_adapter_down = None  # type: ignore[assignment]
            self.output_adapter_up = None  # type: ignore[assignment]
            self.output_adapter_scale = None  # type: ignore[assignment]
        self.layer_idx = int(kwargs.pop("layer_idx", -1))
        self.calib_num_buckets = 0
        self.register_buffer("attention_calib_alpha", torch.empty(0), persistent=False)
        self.register_buffer("attention_calib_gamma", torch.empty(0), persistent=False)
        self.register_buffer("attention_calib_bucket_edges", torch.empty(0), persistent=False)

    def set_attention_calibration(
        self,
        alpha: Optional[torch.Tensor],
        gamma: Optional[torch.Tensor],
        bucket_edges: Optional[torch.Tensor],
    ) -> None:
        if alpha is None or gamma is None or bucket_edges is None:
            self.attention_calib_alpha = torch.empty(0, device=self.q_proj_low.weight.device)
            self.attention_calib_gamma = torch.empty(0, device=self.q_proj_low.weight.device)
            self.attention_calib_bucket_edges = torch.empty(0, device=self.q_proj_low.weight.device)
            self.calib_num_buckets = 0
            return
        self.attention_calib_alpha = alpha.detach()
        self.attention_calib_gamma = gamma.detach()
        self.attention_calib_bucket_edges = bucket_edges.detach()
        self.calib_num_buckets = int(self.attention_calib_alpha.shape[-1])

    def _shape(self, x: torch.Tensor, bsz: int, seq_len: int, low_dim: int) -> torch.Tensor:
        # (bsz, seq, low_dim) -> (bsz, num_heads, seq, low_dim/num_heads)
        if low_dim % self.num_heads != 0:
            raise ValueError(f"low_dim must be divisible by num_heads (low_dim={low_dim}, num_heads={self.num_heads})")
        return x.view(bsz, seq_len, self.num_heads, low_dim // self.num_heads).transpose(1, 2).contiguous()

    def _proj_low_branch(
        self,
        x: torch.Tensor,
        proj: nn.Module,
        dora_mod: nn.Module,
        branch: str,
    ) -> torch.Tensor:
        """
        Low-rank linear path with optional orth/core and column rebalance.
        If proj is LowRankDoRALinear (duck-typed by dora_A), DoRA is already fused in proj.
        """
        if getattr(proj, "dora_A", None) is not None:
            return proj(x)
        W = proj.weight
        # PEFT/LoRA 或 bias=False 的 Linear 可能没有 .bias 属性；F.linear 允许 bias=None
        b = getattr(proj, "bias", None)
        core = getattr(self, f"balance_{branch}_core", None)
        col = getattr(self, f"balance_{branch}_col", None)
        if core is not None:
            W = core.left_multiply(W)
        if col is not None:
            W = W * col.column_scale_for_weight().unsqueeze(0)
        h = F.linear(x, W, b)
        return dora_mod(h)

    @staticmethod
    def _check_finite(name: str, t: torch.Tensor):
        if not torch.isfinite(t).all():
            # 只打印一点点统计，避免刷屏
            with torch.no_grad():
                tt = t.detach()
                msg = (
                    f"[EigenAttn NaN/Inf] {name}: "
                    f"dtype={tt.dtype} device={tt.device} "
                    f"absmax={tt.abs().max().item():.6g} mean={tt.float().mean().item():.6g}"
                )
            raise RuntimeError(msg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        hidden_states: (bsz, tgt_len, embed_dim)
        attention_mask: additive mask, typically (bsz, 1, tgt_len, src_len) or (bsz, 1, 1, src_len)
                       with 0 for keep and very negative for masked.
        returns: (attn_output, attn_probs_reshaped or None, present_key_value or None)
        """
        check_finite = os.getenv("EIGEN_CHECK_FINITE", "0") == "1"

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # ------------------------------------------------------------
        # Align device + dtype to q_proj_low weight (fix multi-GPU + bf16/fp32 mismatch)
        # ------------------------------------------------------------
        # For LowRankDoRALinear wrappers, use frozen base weight only for device/dtype alignment
        # to avoid materializing effective DoRA weight twice in one forward.
        if hasattr(self.q_proj_low, "base") and hasattr(self.q_proj_low.base, "weight"):
            w = self.q_proj_low.base.weight
        else:
            w = self.q_proj_low.weight
        if (hidden_states.device != w.device) or (hidden_states.dtype != w.dtype):
            hidden_states = hidden_states.to(device=w.device, dtype=w.dtype)

        if key_value_states is not None and key_value_states.device != w.device:
            key_value_states = key_value_states.to(w.device, non_blocking=True)
        if attention_mask is not None and attention_mask.device != w.device:
            attention_mask = attention_mask.to(w.device, non_blocking=True)
        if layer_head_mask is not None and layer_head_mask.device != w.device:
            layer_head_mask = layer_head_mask.to(w.device, non_blocking=True)
        if past_key_value is not None:
            pk, pv = past_key_value
            if pk.device != w.device:
                pk = pk.to(w.device, non_blocking=True)
            if pv.device != w.device:
                pv = pv.to(w.device, non_blocking=True)
            past_key_value = (pk, pv)

        # ------------------------------------------------------------
        # Q projection + DoRA + scaling  (order: low-rank linear → orth/col → env DoRA → scaling)
        # ------------------------------------------------------------
        query_states = self._proj_low_branch(hidden_states, self.q_proj_low, self.q_dora, "q")
        query_states = query_states * query_states.new_tensor(self.scaling)

        if check_finite:
            self._check_finite("query_states", query_states)

        # ------------------------------------------------------------
        # K/V projection (self-attn or cross-attn) + past concat
        # Store K/V in shape (bsz, nheads, seq, head_dim_low)
        # ------------------------------------------------------------
        if is_cross_attention:
            # cross-attn: keys/values come from key_value_states
            kv_len = key_value_states.size(1)
            if past_key_value is not None:
                key_states, value_states = past_key_value
            else:
                key_states = self._proj_low_branch(key_value_states, self.k_proj_low, self.k_dora, "k")
                key_states = self._shape(key_states, bsz, kv_len, self.rank_kq)

                value_states = self._proj_low_branch(key_value_states, self.v_proj_low, self.v_dora, "v")
                value_states = self.v_rot(value_states)
                value_states = self._shape(value_states, bsz, kv_len, self.rank_v)
        else:
            # self-attn: keys/values come from hidden_states
            key_states = self._proj_low_branch(hidden_states, self.k_proj_low, self.k_dora, "k")
            key_states = self._shape(key_states, bsz, tgt_len, self.rank_kq)

            value_states = self._proj_low_branch(hidden_states, self.v_proj_low, self.v_dora, "v")
            value_states = self.v_rot(value_states)
            value_states = self._shape(value_states, bsz, tgt_len, self.rank_v)

            if past_key_value is not None:
                pk, pv = past_key_value
                key_states = torch.cat([pk, key_states], dim=2)    # concat on seq_len dim
                value_states = torch.cat([pv, value_states], dim=2)

        if check_finite:
            self._check_finite("key_states", key_states)
            self._check_finite("value_states", value_states)

        present_key_value = (key_states, value_states) if self.is_decoder else None

        # ------------------------------------------------------------
        # Flatten heads: (bsz*nheads, seq, head_dim)
        # ------------------------------------------------------------
        head_dim_kq = self.rank_kq // self.num_heads
        head_dim_v  = self.rank_v  // self.num_heads

        query_states = self._shape(query_states, bsz, tgt_len, self.rank_kq).view(bsz * self.num_heads, tgt_len, head_dim_kq)
        key_states   = key_states.view(bsz * self.num_heads, -1, head_dim_kq)
        value_states = value_states.view(bsz * self.num_heads, -1, head_dim_v)

        src_len = key_states.size(1)

        # ------------------------------------------------------------
        # Attention scores
        # ------------------------------------------------------------
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))  # (bsz*nheads, tgt, src)

        if check_finite:
            self._check_finite("attn_weights_pre_mask", attn_weights)

        # ------------------------------------------------------------
        # Additive mask
        # ------------------------------------------------------------
        if attention_mask is not None:
            # accept (bsz,1,tgt,src) or (bsz,1,1,src)
            if attention_mask.dim() != 4:
                raise ValueError(f"attention_mask must be 4D, got {attention_mask.dim()}D")
            if attention_mask.size(0) != bsz or attention_mask.size(1) != 1 or attention_mask.size(3) != src_len:
                raise ValueError(f"attention_mask shape mismatch: got {attention_mask.size()}, expect (bsz,1,*,{src_len})")
            if attention_mask.size(2) == 1 and tgt_len != 1:
                attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            elif attention_mask.size(2) != tgt_len:
                raise ValueError(f"attention_mask tgt_len mismatch: got {attention_mask.size(2)}, expect {tgt_len}")

            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attention_mask  # broadcast on heads

            # 处理 -inf / +inf / NaN，避免 softmax NaN 传播
            finfo = torch.finfo(attn_weights.dtype)
            attn_weights = torch.nan_to_num(attn_weights, neginf=finfo.min, posinf=finfo.max)

            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if check_finite:
            self._check_finite("attn_weights_post_mask", attn_weights)

        bucket_idx = None
        use_calib = (
            self.attention_calib_alpha.numel() > 0
            and self.attention_calib_gamma.numel() > 0
            and self.attention_calib_bucket_edges.numel() > 0
            and int(self.attention_calib_alpha.shape[0]) == int(self.num_heads)
        )
        if use_calib:
            logits_view = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            probs_for_entropy = torch.softmax(logits_view.float(), dim=-1)
            entropy = -(probs_for_entropy * torch.log(probs_for_entropy.clamp_min(1e-8))).sum(dim=-1)
            edges = self.attention_calib_bucket_edges.to(device=entropy.device, dtype=entropy.dtype)
            cmp = entropy.unsqueeze(-1) > edges.unsqueeze(0).unsqueeze(2)
            bucket_idx = cmp.sum(dim=-1).long().clamp_(0, self.calib_num_buckets - 1)
            alpha = self.attention_calib_alpha.to(device=entropy.device, dtype=entropy.dtype)
            alpha_tok = alpha.unsqueeze(0).unsqueeze(2).expand(bsz, -1, tgt_len, -1)
            alpha_tok = torch.gather(alpha_tok, dim=-1, index=bucket_idx.unsqueeze(-1)).squeeze(-1)
            logits_view = logits_view * alpha_tok.unsqueeze(-1).to(dtype=logits_view.dtype)
            attn_weights = logits_view.view(bsz * self.num_heads, tgt_len, src_len)

        # ------------------------------------------------------------
        # Softmax:
        # - eval path keeps fp32 for stability
        # - training path defaults to attention dtype to cut peak VRAM
        #   set EIGEN_TRAIN_SOFTMAX_FP32=1 to force the old behavior
        # ------------------------------------------------------------
        softmax_dtype = torch.float32
        if self.training and os.getenv("EIGEN_TRAIN_SOFTMAX_FP32", "0") != "1":
            softmax_dtype = attn_weights.dtype
        attn_probs = torch.softmax(attn_weights, dim=-1, dtype=softmax_dtype).to(value_states.dtype)

        if os.getenv("EIGEN_BYPASS_PROBS", "0") == "1":
            attn_probs = torch.full_like(attn_probs, 1.0 / float(src_len))

        if self.dropout > 0 and self.training:
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=True)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f"layer_head_mask shape mismatch: got {layer_head_mask.size()}, expect ({self.num_heads},)")
            attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs * layer_head_mask.view(1, -1, 1, 1)
            attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

        if check_finite:
            self._check_finite("attn_probs", attn_probs)

        attn_probs_reshaped = attn_probs.view(bsz, self.num_heads, tgt_len, src_len) if output_attentions else None

        # ------------------------------------------------------------
        # Attention output
        # ------------------------------------------------------------
        attn_output = torch.matmul(attn_probs, value_states)  # (bsz*nheads, tgt, head_dim_v)
        if use_calib and bucket_idx is not None:
            gamma = self.attention_calib_gamma.to(device=attn_output.device, dtype=attn_output.dtype)
            gamma_tok = gamma.unsqueeze(0).unsqueeze(2).expand(bsz, -1, tgt_len, -1)
            gamma_tok = torch.gather(gamma_tok, dim=-1, index=bucket_idx.unsqueeze(-1)).squeeze(-1)
            attn_output = attn_output.view(bsz, self.num_heads, tgt_len, head_dim_v)
            attn_output = attn_output * gamma_tok.unsqueeze(-1)
            attn_output = attn_output.view(bsz * self.num_heads, tgt_len, head_dim_v)

        if check_finite:
            self._check_finite("attn_output_heads", attn_output)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, head_dim_v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.rank_v)

        # ------------------------------------------------------------
        # vscale (apply ONCE!)
        # vscale range clamp in log-space to avoid inf
        # ------------------------------------------------------------
        # --- vscale: sanitize NaN/Inf in vscale_log (clamp does NOT fix NaN) ---
        vlog = self.vscale_log
        if not torch.isfinite(vlog).all():
            vlog = torch.nan_to_num(vlog, nan=math.log(0.03), posinf=2.0, neginf=-12.0)
        vlog = vlog.squeeze(0).clamp(min=-12.0, max=2.0)
        vscale = torch.exp(vlog).to(dtype=attn_output.dtype, device=attn_output.device)

        # optional env multiplier (ablation)
        env_v = float(os.getenv("EIGEN_VSCALE", "1.0"))
        vscale = vscale * attn_output.new_tensor(env_v)

        attn_output = attn_output * vscale

        if os.getenv("EIGEN_BYPASS_BEFORE_UP", "0") == "1":
            attn_output = torch.zeros_like(attn_output)

        # ------------------------------------------------------------
        # out proj
        # ------------------------------------------------------------
        # align dtype/device to out_proj_up
        ow = self.out_proj_up.weight
        if (attn_output.device != ow.device) or (attn_output.dtype != ow.dtype):
            attn_output = attn_output.to(device=ow.device, dtype=ow.dtype)

        # --- sanitize vscale_log (clamp does NOT fix NaN) ---
        if not hasattr(self, "_vscale_sanitized"):
            self._vscale_sanitized = True
            if not torch.isfinite(self.vscale_log).all():
                with torch.no_grad():
                    # default log(0.03) unless overridden
                    _d = float(os.getenv("EIGEN_VSCALE_SANITIZE", "0.03"))
                    _d = max(_d, 1e-6)
                    _default = math.log(_d)
                    self.vscale_log.data = torch.nan_to_num(
                        self.vscale_log.data, nan=_default, posinf=2.0, neginf=-12.0
                    )
                    self.vscale_log.data.clamp_(-12.0, 2.0)

        if check_finite:
            self._check_finite("vscale_log", self.vscale_log)
            self._check_finite("attn_output_pre_up", attn_output)
        attn_output = self.out_proj_up(attn_output)
        # 低秩残差：h' = h + scale * Up(Down(h))；关闭：export EIGEN_O_RESIDUAL=0（若已创建模块）
        if self.o_residual_down is not None and os.getenv("EIGEN_O_RESIDUAL", "1") == "1":
            delta = self.o_residual_up(self.o_residual_down(attn_output))
            sc = self.o_residual_scale.to(dtype=delta.dtype, device=delta.device)  # type: ignore[union-attr]
            attn_output = attn_output + sc * delta
        # 方案 C / STI-B：attention 输出侧低秩 adapter
        # delta = β · Up(Down(x))，其中 x = hidden_states（attention 的输入）。
        # 读 hidden_states 而不是 attn_output，因为残差 E = Attn_full(x) - Attn_cmp(x) 依赖输入 x。
        # 若需回退到旧实现（读 attn_output），设 EIGEN_OUTPUT_ADAPTER_SRC=attn_output。
        _oad = getattr(self, "output_adapter_down", None)
        if _oad is not None and os.getenv("EIGEN_OUTPUT_ADAPTER", "1") == "1":
            _src_mode = os.getenv("EIGEN_OUTPUT_ADAPTER_SRC", "hidden_states")
            if _src_mode == "attn_output":
                ad_in = attn_output
            else:
                ad_in = hidden_states.to(dtype=attn_output.dtype, device=attn_output.device)
            ad = self.output_adapter_up(self.output_adapter_down(ad_in))
            asc = self.output_adapter_scale.to(dtype=ad.dtype, device=ad.device)  # type: ignore[union-attr]
            attn_output = attn_output + asc * ad
        if os.getenv("EIGEN_BYPASS_ATTN", "0") == "1":
            attn_output = torch.zeros_like(attn_output)

        if check_finite:
            self._check_finite("attn_output_final", attn_output)

        return attn_output, attn_probs_reshaped, present_key_value
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
    def __init__(self, low_rank_config, config, layer_idx: int = -1):
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
            layer_idx=int(layer_idx),
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
                OPTDecoderLayer_EigenAttn(low_rank_config=low_rank_config[i], config=config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self._use_flash_attention_2 = getattr(config, "_attn_implementation", "eager") == "flash_attention_2"
        self.gradient_checkpointing = False

        # ✅ 不要在 decoder 顶层再挂 DoRA，也不要在这里覆盖 layer.self_attn
        self.post_init()

    def load_attention_calibration(self, attention_calib_pt: str) -> None:
        if not attention_calib_pt:
            return
        state = torch.load(attention_calib_pt, map_location="cpu")
        alpha = state.get("alpha", None)
        gamma = state.get("gamma", None)
        edges = state.get("bucket_edges", None)
        if alpha is None or gamma is None or edges is None:
            raise ValueError(f"Invalid calibration file: {attention_calib_pt}")
        if alpha.dim() != 3 or gamma.dim() != 3 or edges.dim() != 3:
            raise ValueError("Calibration tensors must be [num_layers, num_heads, num_buckets/*]")
        for li, layer in enumerate(self.layers):
            layer.self_attn.set_attention_calibration(
                alpha=alpha[li],
                gamma=gamma[li],
                bucket_edges=edges[li],
            )

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

    def __init__(self, config, low_rank_config, attention_calib_pt: Optional[str] = None):
        super().__init__(config, low_rank_config)
        self.model = OPTModel_EigenAttn(config, low_rank_config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        if attention_calib_pt:
            self.model.decoder.load_attention_calibration(attention_calib_pt)

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

