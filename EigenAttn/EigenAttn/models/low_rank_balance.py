# -*- coding: utf-8 -*-
"""
Low-rank QKV rebalancing: headwise factored orthogonal left-factors + per-input-column scaling.

Math (PyTorch nn.Linear weight shape [out_features, in_features] = [d, r_in]):
  - Column j is weight[:, j] ∈ R^d. Scaling columns by gamma_j is equivalent to
    W_eff = W * gamma.view(1, -1)  (broadcast on rows).
  - Left orthogonal rebalancing: W_eff = H1 @ H2 @ W with H1,H2 ∈ R^{d×d}.
    For large d we use block-diagonal factors per attention head on row blocks of W.

Orthogonal map: Cayley transform  H = (I - A)(I + A)^{-1}  for skew-symmetric A.
Factored skew per head block: A_h = U_h V_h^T - V_h U_h^T  with small width orth_rank << d_h.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _skew_from_factored_uv(
    u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Build skew matrix A = U V^T - V U^T with u,v shape [d_h, r]."""
    return torch.matmul(u, v.transpose(-1, -2)) - torch.matmul(v, u.transpose(-1, -2))


def cayley_from_skew(A: torch.Tensor) -> torch.Tensor:
    """Cayley: H = (I - A)(I + A)^{-1}. A must be skew-symmetric. Supports batch dims."""
    # MAGMA batched LU does not support bf16; do solve in fp32 for stability/compat.
    A32 = A.float()
    n = A32.size(-1)
    I = torch.eye(n, dtype=A32.dtype, device=A32.device, layout=A32.layout)
    for _ in range(A.dim() - 2):
        I = I.unsqueeze(0)
    I = I.expand_as(A32)
    num = I - A32
    den = I + A32
    H32 = torch.linalg.solve(den, num)
    return H32.to(dtype=A.dtype)


def orthogonal_reg_frobenius(H: torch.Tensor) -> torch.Tensor:
    """||H^T H - I||_F^2 for last two dims."""
    n = H.size(-1)
    I = torch.eye(n, dtype=H.dtype, device=H.device)
    g = torch.matmul(H.transpose(-1, -2), H) - I
    return (g.float().pow(2).sum(dim=(-2, -1))).mean()


class OrthogonalRebalanceRank(nn.Module):
    """
    W_eff = H1 @ H2 @ W  with headwise block-orthogonal H1,H2 on output rows of W.

    - W: [rank_kq, embed_dim] for Q/K or [rank_v, embed_dim] for V.
    - Rows are assumed split as num_heads blocks of size d_h = rank // num_heads.

    Uses factored skew per head to keep parameter count small (orth_rank << d_h).
    """

    def __init__(
        self,
        rank: int,
        num_heads: int,
        orth_rank: int = 8,
        trainable: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if rank % num_heads != 0:
            raise ValueError(f"rank {rank} not divisible by num_heads {num_heads}")
        self.rank = int(rank)
        self.num_heads = int(num_heads)
        self.d_h = self.rank // self.num_heads
        self.orth_rank = max(1, int(orth_rank))
        self.trainable = bool(trainable)

        dev = device or torch.device("cpu")
        dt = dtype or torch.float32

        # Factored parameters per head, for H1 and H2 each: U,V ∈ R^{d_h × orth_rank}
        self.u1 = nn.Parameter(torch.zeros(self.num_heads, self.d_h, self.orth_rank, dtype=dt, device=dev))
        self.v1 = nn.Parameter(torch.zeros(self.num_heads, self.d_h, self.orth_rank, dtype=dt, device=dev))
        self.u2 = nn.Parameter(torch.zeros(self.num_heads, self.d_h, self.orth_rank, dtype=dt, device=dev))
        self.v2 = nn.Parameter(torch.zeros(self.num_heads, self.d_h, self.orth_rank, dtype=dt, device=dev))

        if not self.trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def _head_H(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """H: [num_heads, d_h, d_h]"""
        A = _skew_from_factored_uv(u, v)
        return cayley_from_skew(A)

    def left_multiply(self, W: torch.Tensor) -> torch.Tensor:
        """
        W: [rank, in_features]
        Returns H1 @ H2 @ W (same shape as W).
        """
        in_features = W.size(1)
        # [num_heads, d_h, in_features]
        w = W.view(self.num_heads, self.d_h, in_features)
        H1 = self._head_H(self.u1, self.v1)
        H2 = self._head_H(self.u2, self.v2)
        H = torch.matmul(H1, H2)
        out = torch.matmul(H, w)
        return out.view(self.rank, in_features)

    def orthogonality_regularizer(self) -> torch.Tensor:
        H1 = self._head_H(self.u1, self.v1)
        H2 = self._head_H(self.u2, self.v2)
        return orthogonal_reg_frobenius(H1) + orthogonal_reg_frobenius(H2)

    @torch.no_grad()
    def reset_to_identity(self) -> None:
        """Zero skew factors -> H1 = H2 = I after Cayley from zero skew gives I."""
        self.u1.zero_()
        self.v1.zero_()
        self.u2.zero_()
        self.v2.zero_()


class LowRankColumnBalancer(nn.Module):
    """
    Per-input-column scaling of weight matrix: W_eff[:, j] = gamma_j * W[:, j]
    with gamma_j = exp(log_gamma_j) * strength blend toward uniform column norms at init.

    balance_strength=0 -> gamma = 1 (when log_gamma initialized to 0 and blend handles init).
    """

    def __init__(
        self,
        in_features: int,
        balance_strength: float = 0.25,
        mode: str = "trainable",
        init_from_weight: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        conservative_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.balance_strength = float(balance_strength)
        self.mode = str(mode)
        self.conservative_scale = float(conservative_scale)

        dev = device or torch.device("cpu")
        dt = dtype or torch.float32

        self.register_buffer("_gamma_init_done", torch.tensor(0, dtype=torch.int8))
        self.log_gamma = nn.Parameter(torch.zeros(self.in_features, dtype=dt, device=dev))

        if init_from_weight is not None and self.balance_strength > 1e-8:
            self.init_from_weight_matrix(init_from_weight)

        if self.mode == "static":
            self.log_gamma.requires_grad_(False)

    @torch.no_grad()
    def init_from_weight_matrix(self, W: torch.Tensor, blend: Optional[float] = None) -> None:
        """W: [out, in]. Column norms along dim=0 -> per input feature."""
        blend = self.balance_strength if blend is None else float(blend)
        blend = blend * self.conservative_scale
        blend = max(0.0, min(1.0, blend))
        w = W.detach().float()
        col = w.norm(dim=0).clamp_min(1e-6)
        m = col.mean()
        # target: pull norms toward mean; gamma_j = (m / col_j)^blend
        gamma = (m / col).pow(blend)
        # soften toward 1
        gamma = gamma.pow(0.5)  # gentler
        gamma = gamma.clamp(0.25, 4.0)
        self.log_gamma.copy_(gamma.log().to(dtype=self.log_gamma.dtype, device=self.log_gamma.device))

    def gamma_vec(self) -> torch.Tensor:
        """Positive multipliers [in_features], applied as x * gamma for F.linear(x, W)."""
        return torch.exp(self.log_gamma.clamp(-4.0, 4.0))

    def column_scale_for_weight(self) -> torch.Tensor:
        """For W_eff = W * gamma.view(1,-1) — same as scaling input x by gamma before W."""
        g = self.gamma_vec()
        return g

    def balance_regularizer(self) -> torch.Tensor:
        """Keep log_gamma near 0 -> gamma near 1."""
        return (self.log_gamma.float().pow(2)).mean()


def collect_orth_balance_regularizers(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum orthogonality and balance regs over all OrthogonalRebalanceRank / LowRankColumnBalancer submodules."""
    orth_sum: Optional[torch.Tensor] = None
    bal_sum: Optional[torch.Tensor] = None
    for m in model.modules():
        if isinstance(m, OrthogonalRebalanceRank):
            reg = m.orthogonality_regularizer()
            if orth_sum is None:
                orth_sum = reg
            else:
                orth_sum = orth_sum + reg.to(device=orth_sum.device)
        if isinstance(m, LowRankColumnBalancer):
            reg = m.balance_regularizer()
            if bal_sum is None:
                bal_sum = reg
            else:
                bal_sum = bal_sum + reg.to(device=bal_sum.device)
    if orth_sum is None:
        orth_sum = torch.tensor(0.0)
    if bal_sum is None:
        bal_sum = torch.tensor(0.0)
    return orth_sum, bal_sum


def parse_balance_target(s: str) -> Set[str]:
    s = (s or "qkv").strip().lower()
    if s in ("qkv", "all"):
        return {"q", "k", "v"}
    if s == "qv":
        return {"q", "v"}
    if s == "qk":
        return {"q", "k"}
    if s in ("q", "k", "v"):
        return {s}
    return {"q", "k", "v"}


def apply_low_rank_balance_to_opt_model(model: nn.Module, args: Any, logger: Optional[Any] = None) -> None:
    """
    在 OPT EigenAttn 的 q/k/v_proj_low 上挂载 OrthogonalRebalanceRank / LowRankColumnBalancer（幂等）。
    需在 DoRA 包装之前调用；与 main_eigen_attn.apply_low_rank_balance_to_opt_model 行为一致。

    依赖 args:
      use_core_rebalance, use_column_balance, balance_target, core_rebalance_trainable,
      balance_mode, balance_strength, core_rebalance_orth_rank, balance_lr_mult_k, balance_lr_mult_v,
      ft_train_last_n（仅最后 N 层；0 表示全部层）
    """
    use_core = bool(int(getattr(args, "use_core_rebalance", 0)))
    use_col = bool(int(getattr(args, "use_column_balance", 0)))
    if not use_core and not use_col:
        return

    targets = parse_balance_target(str(getattr(args, "balance_target", "qkv")))
    trainable_core = bool(int(getattr(args, "core_rebalance_trainable", 1)))
    mode = str(getattr(args, "balance_mode", "trainable"))
    strength = float(getattr(args, "balance_strength", 0.25))
    orth_rank = int(getattr(args, "core_rebalance_orth_rank", 8))
    km = float(getattr(args, "balance_lr_mult_k", 0.5))
    vm = float(getattr(args, "balance_lr_mult_v", 1.0))

    layers = list(model.model.decoder.layers)
    last_n = int(getattr(args, "ft_train_last_n", 0) or 0)
    if last_n > 0:
        layers = layers[-last_n:]

    for layer in layers:
        attn = layer.self_attn
        embed_dim = int(attn.embed_dim)
        num_heads = int(attn.num_heads)
        rank_kq = int(attn.rank_kq)
        rank_v = int(attn.rank_v)

        for br, nm in (("q", "q_proj_low"), ("k", "k_proj_low"), ("v", "v_proj_low")):
            if br not in targets:
                continue
            if not hasattr(attn, nm):
                continue
            proj = getattr(attn, nm)
            if getattr(proj, "dora_A", None) is not None:
                continue
            if not isinstance(proj, nn.Linear):
                continue

            rank = rank_kq if br in ("q", "k") else rank_v
            W = proj.weight
            dev = W.device
            dt = W.dtype

            cons = km if br == "k" else vm
            if use_core and getattr(attn, f"balance_{br}_core", None) is None:
                core = OrthogonalRebalanceRank(
                    rank=rank,
                    num_heads=num_heads,
                    orth_rank=orth_rank,
                    trainable=trainable_core,
                    dtype=dt,
                    device=dev,
                )
                if not trainable_core:
                    for p in core.parameters():
                        p.requires_grad_(False)
                attn.register_module(f"balance_{br}_core", core)

            if use_col and getattr(attn, f"balance_{br}_col", None) is None:
                col = LowRankColumnBalancer(
                    in_features=embed_dim,
                    balance_strength=strength * cons,
                    mode="static" if mode == "static" else "trainable",
                    init_from_weight=W.detach(),
                    dtype=dt,
                    device=dev,
                    conservative_scale=cons,
                )
                if mode == "static":
                    for p in col.parameters():
                        p.requires_grad_(False)
                attn.register_module(f"balance_{br}_col", col)

    if logger is not None:
        msg = (
            f"[low_rank_balance] attached core={use_core} col={use_col} target={targets} "
            f"mode={mode} strength={strength} trainable_core={trainable_core}"
        )
        if hasattr(logger, "info"):
            logger.info(msg)
        else:
            logger(msg)
