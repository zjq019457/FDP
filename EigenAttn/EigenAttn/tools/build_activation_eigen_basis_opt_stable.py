#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立稳定版 basis 构建脚本（目标：优先提升 PPL 稳定性）。

相对 build_activation_eigen_basis_opt.py 的升级点：
1) score scale/shift 对齐：在 score loss 内对 S_hat 做线性回归校准 S≈a*S_hat+b。
2) value/output path 保真：约束 (softmax(S)@V) 与 (softmax(S_hat_fit)@V_hat)。
3) branch coupling：约束 Q^T K 与 K^T V 的二阶耦合统计。
4) 层内漂移抑制：约束投影后 Q/K/V 的均值与标准差接近原激活。
5) 轻量优化：仅优化每层每头的 U_q/U_k/U_v，不更新模型权重；每步 QR 保持列正交。
6) 层深加权 + 范数锚点：用本层 cache 上 Q/K/V 的平均 L2 作为标量锚，抑制「沿深度累积的幅度漂移」；
   另对逐 token L2 的均值/方差与 log 尺度做约束（对真实 residual hidden 的代理，非完整反传 student）。
7) W_O-aware（可选）：各 head 的 O=A@V 沿 head 维拼接后，用冻结的 self_attn.out_proj 对齐投影前后输出（进残差前）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tools.build_activation_eigen_basis_opt import (
    _basis_key,
    _parse_layers,
    build_token_stream,
    collect_activation_basis_one_layer,
    iter_texts_from_dir,
    principal_angle_stats,
)


def _orth(U: torch.Tensor) -> torch.Tensor:
    if U.numel() == 0:
        return U
    q, _ = torch.linalg.qr(U, mode="reduced")
    return q[:, : U.shape[1]].contiguous()


def _resolve_tokenizer_path(model_path: str, tokenizer: str) -> str:
    t = (tokenizer or "").strip()
    return t if t else model_path


def _gather_qkv_cache(
    model: torch.nn.Module,
    layer_idx: int,
    batch_list: List[torch.Tensor],
    input_device: torch.device,
    max_batches: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    layer = model.model.decoder.layers[layer_idx]
    attn = layer.self_attn
    hook_out: Dict[str, torch.Tensor] = {}

    def qh(_m, _inp, out):
        hook_out["q"] = out

    def kh(_m, _inp, out):
        hook_out["k"] = out

    def vh(_m, _inp, out):
        hook_out["v"] = out

    hq = attn.q_proj.register_forward_hook(qh)
    hk = attn.k_proj.register_forward_hook(kh)
    hv = attn.v_proj.register_forward_hook(vh)
    cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        model.eval()
        for batch in batch_list[: max(1, int(max_batches))]:
            hook_out.clear()
            ids = batch.to(input_device)
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            with torch.no_grad():
                _ = model(input_ids=ids, use_cache=False, return_dict=True)
            if "q" in hook_out and "k" in hook_out and "v" in hook_out:
                cache.append(
                    (
                        hook_out["q"].detach().cpu().float(),
                        hook_out["k"].detach().cpu().float(),
                        hook_out["v"].detach().cpu().float(),
                    )
                )
    finally:
        hq.remove()
        hk.remove()
        hv.remove()
        model.config.use_cache = use_cache
    return cache


def _cache_mean_row_l2(t: torch.Tensor) -> float:
    """t: [bs, seq, d] -> 全 token 行 L2 的均值（CPU float）。"""
    x = t.reshape(-1, t.shape[-1]).float().norm(dim=-1).mean().item()
    return float(x)


def _fit_affine(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # y ≈ a*x + b
    xm = x.mean()
    ym = y.mean()
    xv = x - xm
    den = (xv * xv).mean().clamp(min=1e-8)
    a = ((x - xm) * (y - ym)).mean() / den
    b = ym - a * xm
    return a, b


def _refine_layer_bases_stable(
    *,
    u_q: torch.Tensor,
    u_k: torch.Tensor,
    u_v: torch.Tensor,
    cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    num_heads: int,
    head_dim: int,
    rank_kq_per_head: int,
    rank_v_per_head: int,
    repair_keqk: int,
    layer_idx: int,
    n_layers: int,
    steps: int,
    lr: float,
    sample_m: int,
    sample_n: int,
    seed: int,
    loss_score: float,
    loss_value: float,
    loss_couple: float,
    loss_drift: float,
    loss_resnorm: float,
    loss_stream_ratio: float,
    loss_anchor: float,
    depth_boost: float,
    anchor_q: float,
    anchor_k: float,
    anchor_v: float,
    loss_wo: float,
    out_proj_weight: Optional[torch.Tensor],
    out_proj_bias: Optional[torch.Tensor],
    loss_prox: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    loss_sum = (
        float(loss_score)
        + float(loss_value)
        + float(loss_couple)
        + float(loss_drift)
        + float(loss_resnorm)
        + float(loss_stream_ratio)
        + float(loss_anchor)
        + float(loss_wo)
    )
    if steps <= 0 or loss_sum <= 0:
        return u_q, u_k, u_v, {"skipped": 1.0}
    if not cache:
        return u_q, u_k, u_v, {"skipped_empty_cache": 1.0}

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_wo: Optional[torch.Tensor] = None
    b_wo: Optional[torch.Tensor] = None
    if float(loss_wo) > 0.0:
        if out_proj_weight is None:
            raise ValueError("loss_wo > 0 requires out_proj_weight from self_attn.out_proj")
        exp_in = int(num_heads * head_dim)
        if int(out_proj_weight.shape[1]) != exp_in:
            raise ValueError(
                f"out_proj expects in_features={exp_in}, got weight.shape[1]={int(out_proj_weight.shape[1])}"
            )
        W_wo = out_proj_weight.detach().to(device=dev, dtype=torch.float32)
        b_wo = (
            out_proj_bias.detach().to(device=dev, dtype=torch.float32)
            if out_proj_bias is not None
            else None
        )

    U0_q = []
    U0_k = []
    U0_v = []
    for h in range(num_heads):
        r = slice(h * head_dim, (h + 1) * head_dim)
        ckq = slice(h * rank_kq_per_head, (h + 1) * rank_kq_per_head)
        cv = slice(h * rank_v_per_head, (h + 1) * rank_v_per_head)
        U0_q.append(u_q[r, ckq].float().to(dev).clone())
        U0_k.append(u_k[r, ckq].float().to(dev).clone())
        U0_v.append(u_v[r, cv].float().to(dev).clone())

    if int(repair_keqk) == 1:
        Uq = [torch.nn.Parameter(x.clone()) for x in U0_q]
        Uk = Uq
        params = list(Uq)
    else:
        Uq = [torch.nn.Parameter(x.clone()) for x in U0_q]
        Uk = [torch.nn.Parameter(x.clone()) for x in U0_k]
        params = list(Uq) + list(Uk)
    Uv = [torch.nn.Parameter(x.clone()) for x in U0_v]
    params.extend(Uv)

    opt = torch.optim.Adam(params, lr=float(lr))
    scale = 1.0 / math.sqrt(float(head_dim))

    def _pick_indices(T: int, bidx: int, step_i: int):
        m = int(min(max(1, sample_m), T))
        n = int(min(max(1, sample_n), T))
        g = torch.Generator()
        g.manual_seed(int(seed) + bidx * 1009 + step_i * 100003)
        idx_i = torch.randperm(T, generator=g)[:m].to(dev)
        idx_j = torch.randperm(T, generator=g)[:n].to(dev)
        return idx_i, idx_j

    depth_mult = 1.0 + float(depth_boost) * (float(layer_idx) / float(max(n_layers - 1, 1)))

    def _loss_total(step_i: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        ls = torch.zeros((), device=dev)
        lv = torch.zeros((), device=dev)
        lc = torch.zeros((), device=dev)
        ld = torch.zeros((), device=dev)
        lrn = torch.zeros((), device=dev)
        lsr = torch.zeros((), device=dev)
        la = torch.zeros((), device=dev)
        lwo = torch.zeros((), device=dev)
        nb = 0
        for bidx, (q_cpu, k_cpu, v_cpu) in enumerate(cache):
            q = q_cpu.to(dev)
            k = k_cpu.to(dev)
            v = v_cpu.to(dev)
            bs, seq, hs = q.shape
            T = bs * seq
            idx_i, idx_j = _pick_indices(T, bidx, step_i)
            nb += 1
            if float(loss_wo) > 0.0:
                O_heads: List[torch.Tensor] = []
                Oh_heads: List[torch.Tensor] = []
            for h in range(num_heads):
                sl = slice(h * head_dim, (h + 1) * head_dim)
                qh = q[..., sl].reshape(T, head_dim)
                kh = k[..., sl].reshape(T, head_dim)
                vh = v[..., sl].reshape(T, head_dim)
                Qi = qh.index_select(0, idx_i)
                Kj = kh.index_select(0, idx_j)
                Vj = vh.index_select(0, idx_j)

                Pq = Uq[h] @ Uq[h].t()
                Pk = Uk[h] @ Uk[h].t()
                Pv = Uv[h] @ Uv[h].t()
                Qi_h = Qi @ Pq
                Kj_h = Kj @ Pk
                Vj_h = Vj @ Pv

                S = (Qi @ Kj.t()) * scale
                Sh = (Qi_h @ Kj_h.t()) * scale
                a, b = _fit_affine(Sh, S)
                Sh_fit = a * Sh + b
                ls = ls + F.mse_loss(Sh_fit, S)

                # value/output path consistency
                A = torch.softmax(S, dim=-1)
                Ah = torch.softmax(Sh_fit, dim=-1)
                O = A @ Vj
                Oh = Ah @ Vj_h
                lv = lv + F.mse_loss(Oh, O)
                if float(loss_wo) > 0.0:
                    O_heads.append(O)
                    Oh_heads.append(Oh)

                # cross-branch coupling
                Cqk = (Qi.t() @ Kj) / float(max(1, Qi.shape[0]))
                Cqk_h = (Qi_h.t() @ Kj_h) / float(max(1, Qi_h.shape[0]))
                Ckv = (Kj.t() @ Vj) / float(max(1, Kj.shape[0]))
                Ckv_h = (Kj_h.t() @ Vj_h) / float(max(1, Kj_h.shape[0]))
                lc = lc + F.mse_loss(Cqk_h, Cqk) + F.mse_loss(Ckv_h, Ckv)

                # activation drift (mean/std) on projected branches
                for orig, proj in ((Qi, Qi_h), (Kj, Kj_h), (Vj, Vj_h)):
                    mu_o = orig.mean(dim=0)
                    mu_p = proj.mean(dim=0)
                    sd_o = orig.std(dim=0, unbiased=False)
                    sd_p = proj.std(dim=0, unbiased=False)
                    ld = ld + F.mse_loss(mu_p, mu_o) + F.mse_loss(sd_p, sd_o)
                    if float(loss_resnorm) > 0.0 or float(loss_stream_ratio) > 0.0:
                        no = orig.norm(dim=-1)
                        np_ = proj.norm(dim=-1)
                        if float(loss_resnorm) > 0.0:
                            lrn = lrn + F.mse_loss(np_.mean(), no.mean()) + F.mse_loss(
                                np_.std(unbiased=False), no.std(unbiased=False)
                            )
                        if float(loss_stream_ratio) > 0.0:
                            lsr = lsr + (
                                torch.log(np_.mean().clamp(min=1e-8))
                                - torch.log(no.mean().clamp(min=1e-8))
                            ).pow(2)

                if float(loss_anchor) > 0.0:
                    aq_t = torch.as_tensor(anchor_q, device=dev, dtype=torch.float32)
                    ak_t = torch.as_tensor(anchor_k, device=dev, dtype=torch.float32)
                    av_t = torch.as_tensor(anchor_v, device=dev, dtype=torch.float32)
                    la = la + F.mse_loss(Qi_h.norm(dim=-1).mean(), aq_t)
                    la = la + F.mse_loss(Kj_h.norm(dim=-1).mean(), ak_t)
                    la = la + F.mse_loss(Vj_h.norm(dim=-1).mean(), av_t)

            if float(loss_wo) > 0.0 and W_wo is not None and O_heads:
                O_cat = torch.cat(O_heads, dim=-1)
                Oh_cat = torch.cat(Oh_heads, dim=-1)
                y = F.linear(O_cat, W_wo, b_wo)
                yh = F.linear(Oh_cat, W_wo, b_wo)
                lwo = lwo + F.mse_loss(yh, y)

        div = float(max(1, nb * num_heads))
        ls = ls / div
        lv = lv / div
        lc = lc / div
        ld = ld / div
        lrn = lrn / div
        lsr = lsr / div
        la = la / div
        lwo = lwo / float(max(1, nb))

        lp = torch.zeros((), device=dev)
        if loss_prox > 0:
            for h in range(num_heads):
                lp = lp + (Uq[h] - U0_q[h]).pow(2).mean()
                lp = lp + (Uv[h] - U0_v[h]).pow(2).mean()
                if int(repair_keqk) != 1:
                    lp = lp + (Uk[h] - U0_k[h]).pow(2).mean()

        geo = (
            float(loss_couple) * lc
            + float(loss_drift) * ld
            + float(loss_resnorm) * lrn
            + float(loss_stream_ratio) * lsr
            + float(loss_anchor) * la
            + float(loss_wo) * lwo
        )
        total = float(loss_score) * ls + float(loss_value) * lv + float(depth_mult) * geo + float(loss_prox) * lp
        log = {
            "score": float(ls.detach().item()),
            "value": float(lv.detach().item()),
            "wo": float(lwo.detach().item()),
            "couple": float(lc.detach().item()),
            "drift": float(ld.detach().item()),
            "resnorm": float(lrn.detach().item()),
            "stream_ratio": float(lsr.detach().item()),
            "anchor": float(la.detach().item()),
            "depth_mult": float(depth_mult),
            "prox": float(lp.detach().item()),
        }
        return total, log

    @torch.no_grad()
    def _retract():
        for h in range(num_heads):
            Uq[h].copy_(_orth(Uq[h]))
            Uv[h].copy_(_orth(Uv[h]))
            if int(repair_keqk) != 1:
                Uk[h].copy_(_orth(Uk[h]))

    @torch.no_grad()
    def _pack() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        oq = u_q.clone()
        ok = u_k.clone()
        ov = u_v.clone()
        for h in range(num_heads):
            r = slice(h * head_dim, (h + 1) * head_dim)
            ckq = slice(h * rank_kq_per_head, (h + 1) * rank_kq_per_head)
            cv = slice(h * rank_v_per_head, (h + 1) * rank_v_per_head)
            oq[r, ckq] = Uq[h].float().cpu()
            ok[r, ckq] = Uk[h].float().cpu()
            ov[r, cv] = Uv[h].float().cpu()
        return oq, ok, ov

    _, before = _loss_total(step_i=-1)
    for si in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        total, _ = _loss_total(step_i=si)
        total.backward()
        opt.step()
        _retract()
    _, after = _loss_total(step_i=steps + 1)
    uq, uk, uv = _pack()
    stats = {f"before_{k}": v for k, v in before.items()}
    stats.update({f"after_{k}": v for k, v in after.items()})
    return uq, uk, uv, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--ref_basis", type=str, required=True)
    ap.add_argument("--out_basis", type=str, required=True)
    ap.add_argument("--compare_json", type=str, default="")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default="")
    ap.add_argument("--token_budget", type=int, default=1_200_000)
    ap.add_argument("--max_data_files", type=int, default=200)
    ap.add_argument("--max_data_lines", type=int, default=2_000_000)
    ap.add_argument("--layers", type=str, default="all")
    ap.add_argument("--max_batches", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--device_map", type=str, default="cuda:0")
    ap.add_argument("--repair_keqk", type=int, default=0, choices=[0, 1])
    ap.add_argument("--cov_residual_to_ref", type=int, default=0, choices=[0, 1])
    ap.add_argument("--weight_mode", type=str, default="nll", choices=["uniform", "nll", "nll_fisher", "nll_attn_entropy", "nll_teacher_attn"])
    ap.add_argument("--nll_power", type=float, default=1.0)
    ap.add_argument("--nll_clip_percentile", type=float, default=99.5)
    ap.add_argument("--nll_mix_lambda", type=float, default=0.9)
    ap.add_argument("--nll_power_qk", type=float, default=1.0)
    ap.add_argument("--nll_power_v", type=float, default=1.0)
    ap.add_argument("--nll_clip_percentile_qk", type=float, default=99.5)
    ap.add_argument("--nll_clip_percentile_v", type=float, default=100.0)
    ap.add_argument("--nll_mix_lambda_qk", type=float, default=0.95)
    ap.add_argument("--nll_mix_lambda_v", type=float, default=0.25)
    # stable refinement
    ap.add_argument("--refine_steps", type=int, default=20)
    ap.add_argument("--refine_lr", type=float, default=0.05)
    ap.add_argument(
        "--adaptive_refine",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: 按层号自适应 refine 强度（前层更弱后层更强），减少全层回滚",
    )
    ap.add_argument(
        "--adaptive_front_scale",
        type=float,
        default=0.40,
        help="adaptive_refine=1 时前层相对基准步数/学习率缩放；后层逐步线性过渡到 1.0",
    )
    ap.add_argument("--refine_batches", type=int, default=4)
    ap.add_argument("--refine_sample_m", type=int, default=64)
    ap.add_argument("--refine_sample_n", type=int, default=64)
    ap.add_argument("--loss_score", type=float, default=1.0)
    ap.add_argument("--loss_value", type=float, default=0.8)
    ap.add_argument("--loss_couple", type=float, default=0.15)
    ap.add_argument("--loss_drift", type=float, default=0.3)
    ap.add_argument(
        "--loss_resnorm",
        type=float,
        default=0.10,
        help="逐 token L2 范数的均值/方差：投影前后对齐（抑制幅度漂移）",
    )
    ap.add_argument(
        "--loss_stream_ratio",
        type=float,
        default=0.08,
        help="log(||proj||_mean/||orig||_mean)^2，保持几何尺度（配合 score 仿射）",
    )
    ap.add_argument(
        "--loss_anchor",
        type=float,
        default=0.06,
        help="将投影后 Q/K/V 的行 L2 均值锚到本层 cache 上教师激活的均值（层间深度代理）",
    )
    ap.add_argument(
        "--depth_boost",
        type=float,
        default=0.30,
        help="随层号放大 couple/drift/resnorm/stream_ratio/anchor/wo（0 关闭）",
    )
    ap.add_argument(
        "--loss_wo",
        type=float,
        default=0.0,
        help="W_O-aware：各 head O=A@V 拼接后 MSE(out_proj(Oh), out_proj(O))，权重；0=关闭",
    )
    ap.add_argument("--loss_prox", type=float, default=0.01)
    ap.add_argument(
        "--accept_bad_layer",
        type=int,
        default=0,
        choices=[0, 1],
        help="0: 若层内 score/value 变差则回滚到 refine 前（推荐）；1: 无条件接受",
    )
    ap.add_argument(
        "--accept_score_ratio",
        type=float,
        default=1.01,
        help="层接收阈值：after_score <= before_score * ratio",
    )
    ap.add_argument(
        "--accept_value_ratio",
        type=float,
        default=1.02,
        help="层接收阈值：after_value <= before_value * ratio",
    )
    ap.add_argument(
        "--accept_mode",
        type=str,
        default="strict",
        choices=["strict", "composite"],
        help="strict: score/value 双阈值；composite: 按加权相对变化综合判定",
    )
    ap.add_argument(
        "--accept_comp_w_score",
        type=float,
        default=0.7,
        help="composite 模式中 score 相对变化权重",
    )
    ap.add_argument(
        "--accept_comp_w_value",
        type=float,
        default=0.3,
        help="composite 模式中 value 相对变化权重",
    )
    ap.add_argument(
        "--accept_comp_w_wo",
        type=float,
        default=0.0,
        help="composite 模式中 wo 相对变化权重（loss_wo>0 时建议 0.1~0.3）",
    )
    ap.add_argument(
        "--accept_comp_max_delta",
        type=float,
        default=0.0,
        help=(
            "composite 接收阈值（见 --accept_comp_metric）："
            "ratio 时为相对比例加权和；delta_over_floor 时为归一化差分加权和"
        ),
    )
    ap.add_argument(
        "--accept_comp_metric",
        type=str,
        default="delta_over_floor",
        choices=["ratio", "delta_over_floor"],
        help=(
            "ratio: after/before-1（baseline 极小时易误杀）；"
            "delta_over_floor: (after-before)/max(before,floor)，推荐 all-layer"
        ),
    )
    ap.add_argument(
        "--accept_baseline_floor",
        type=float,
        default=1e-4,
        help="delta_over_floor 模式下 score/value/wo 分母下限，抑制极小 baseline 放大",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ref_path = Path(args.ref_basis)
    ref_state = torch.load(str(ref_path), map_location="cpu")
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    n_layers = int(cfg.num_hidden_layers)
    num_heads = int(cfg.num_attention_heads)
    hidden = int(cfg.hidden_size)
    head_dim = hidden // num_heads
    layer_ids = _parse_layers(args.layers, n_layers)

    tok = AutoTokenizer.from_pretrained(_resolve_tokenizer_path(args.model, args.tokenizer), local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    texts = iter_texts_from_dir(args.data_dir, max_files=args.max_data_files, max_lines=args.max_data_lines)
    stream = build_token_stream(tok, texts, target_tokens=args.token_budget, seed=args.seed)
    if stream is None:
        raise RuntimeError("empty token stream")

    torch.manual_seed(args.seed)
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.device_map != "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dt,
            device_map={"": args.device_map},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dt,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    input_device = model.get_input_embeddings().weight.device

    def make_batches():
        g = torch.Generator()
        g.manual_seed(args.seed)
        N = stream.numel()
        for _ in range(args.max_batches):
            starts = torch.randint(low=0, high=N - args.seqlen - 1, size=(args.batch_size,), generator=g)
            x = torch.stack([stream[s : s + args.seqlen] for s in starts], dim=0)
            yield x

    batch_list = list(make_batches())
    out_state: Dict[str, torch.Tensor] = {}
    report: Dict[str, object] = {
        "model": args.model,
        "layers": layer_ids,
        "stable_refine": {
            "steps": int(args.refine_steps),
            "lr": float(args.refine_lr),
            "adaptive_refine": int(args.adaptive_refine),
            "adaptive_front_scale": float(args.adaptive_front_scale),
            "loss_score": float(args.loss_score),
            "loss_value": float(args.loss_value),
            "loss_couple": float(args.loss_couple),
            "loss_drift": float(args.loss_drift),
            "loss_resnorm": float(args.loss_resnorm),
            "loss_stream_ratio": float(args.loss_stream_ratio),
            "loss_anchor": float(args.loss_anchor),
            "depth_boost": float(args.depth_boost),
            "loss_wo": float(args.loss_wo),
            "loss_prox": float(args.loss_prox),
            "accept_bad_layer": int(args.accept_bad_layer),
            "accept_score_ratio": float(args.accept_score_ratio),
            "accept_value_ratio": float(args.accept_value_ratio),
            "accept_mode": str(args.accept_mode),
            "accept_comp_w_score": float(args.accept_comp_w_score),
            "accept_comp_w_value": float(args.accept_comp_w_value),
            "accept_comp_w_wo": float(args.accept_comp_w_wo),
            "accept_comp_max_delta": float(args.accept_comp_max_delta),
            "accept_comp_metric": str(args.accept_comp_metric),
            "accept_baseline_floor": float(args.accept_baseline_floor),
        },
        "per_layer": [],
    }

    for li in layer_ids:
        r_q = ref_state[_basis_key(li, "q")].float()
        r_k = ref_state[_basis_key(li, "k")].float()
        r_v = ref_state[_basis_key(li, "v")].float()
        rank_kq_per_head = int(r_q.shape[1]) // num_heads
        rank_v_per_head = int(r_v.shape[1]) // num_heads
        u_q, u_k, u_v, n_tok, _ = collect_activation_basis_one_layer(
            model,
            layer_idx=li,
            num_heads=num_heads,
            head_dim=head_dim,
            rank_kq_per_head=rank_kq_per_head,
            rank_v_per_head=rank_v_per_head,
            repair_keqk=args.repair_keqk,
            data_iter=batch_list,
            input_device=input_device,
            weight_mode=args.weight_mode,
            nll_power=float(args.nll_power),
            nll_clip_percentile=float(args.nll_clip_percentile),
            nll_mix_lambda=float(args.nll_mix_lambda),
            nll_power_qk=float(args.nll_power_qk),
            nll_power_v=float(args.nll_power_v),
            nll_clip_percentile_qk=float(args.nll_clip_percentile_qk),
            nll_clip_percentile_v=float(args.nll_clip_percentile_v),
            nll_mix_lambda_qk=float(args.nll_mix_lambda_qk),
            nll_mix_lambda_v=float(args.nll_mix_lambda_v),
            residual_wrt_ref=bool(int(args.cov_residual_to_ref)),
            ref_u_q=r_q if int(args.cov_residual_to_ref) == 1 else None,
            ref_u_k=r_k if int(args.cov_residual_to_ref) == 1 else None,
            ref_u_v=r_v if int(args.cov_residual_to_ref) == 1 else None,
            score_lambda=0.0,
        )
        cache = _gather_qkv_cache(
            model=model,
            layer_idx=li,
            batch_list=batch_list,
            input_device=input_device,
            max_batches=args.refine_batches,
        )
        nbc = max(len(cache), 1)
        aq = ak = av = 0.0
        for qc, kc, vc in cache:
            aq += _cache_mean_row_l2(qc)
            ak += _cache_mean_row_l2(kc)
            av += _cache_mean_row_l2(vc)
        aq /= float(nbc)
        ak /= float(nbc)
        av /= float(nbc)
        if float(args.loss_wo) > 0.0:
            op = model.model.decoder.layers[li].self_attn.out_proj
            w_wo = op.weight.detach().float().cpu()
            b_wo = op.bias.detach().float().cpu() if op.bias is not None else None
        else:
            w_wo, b_wo = None, None
        if int(args.adaptive_refine) == 1:
            li_ratio = float(li) / float(max(n_layers - 1, 1))
            front = float(max(0.05, min(1.0, args.adaptive_front_scale)))
            layer_scale = front + (1.0 - front) * li_ratio
            steps_li = max(1, int(round(float(args.refine_steps) * layer_scale)))
            lr_li = float(args.refine_lr) * layer_scale
        else:
            steps_li = int(args.refine_steps)
            lr_li = float(args.refine_lr)

        u_q2, u_k2, u_v2, st = _refine_layer_bases_stable(
            u_q=u_q,
            u_k=u_k,
            u_v=u_v,
            cache=cache,
            num_heads=num_heads,
            head_dim=head_dim,
            rank_kq_per_head=rank_kq_per_head,
            rank_v_per_head=rank_v_per_head,
            repair_keqk=args.repair_keqk,
            layer_idx=int(li),
            n_layers=int(n_layers),
            steps=steps_li,
            lr=lr_li,
            sample_m=args.refine_sample_m,
            sample_n=args.refine_sample_n,
            seed=args.seed + li * 13,
            loss_score=args.loss_score,
            loss_value=args.loss_value,
            loss_couple=args.loss_couple,
            loss_drift=args.loss_drift,
            loss_resnorm=args.loss_resnorm,
            loss_stream_ratio=args.loss_stream_ratio,
            loss_anchor=args.loss_anchor,
            depth_boost=args.depth_boost,
            anchor_q=aq,
            anchor_k=ak,
            anchor_v=av,
            loss_wo=float(args.loss_wo),
            out_proj_weight=w_wo,
            out_proj_bias=b_wo,
            loss_prox=args.loss_prox,
        )
        b_score = float(st.get("before_score", float("inf")))
        a_score = float(st.get("after_score", float("inf")))
        b_value = float(st.get("before_value", float("inf")))
        a_value = float(st.get("after_value", float("inf")))
        b_wo = float(st.get("before_wo", 0.0))
        a_wo = float(st.get("after_wo", 0.0))
        score_ok = a_score <= b_score * float(args.accept_score_ratio)
        value_ok = a_value <= b_value * float(args.accept_value_ratio)
        bs = max(b_score, 1e-12)
        bv = max(b_value, 1e-12)
        fl = float(max(args.accept_baseline_floor, 1e-12))
        wo_term = 0.0
        if str(args.accept_comp_metric) == "ratio":
            if float(args.loss_wo) > 0.0 and abs(b_wo) > 1e-12:
                wo_term = (a_wo / max(abs(b_wo), 1e-12)) - 1.0
            comp_delta = (
                float(args.accept_comp_w_score) * (a_score / bs - 1.0)
                + float(args.accept_comp_w_value) * (a_value / bv - 1.0)
                + float(args.accept_comp_w_wo) * wo_term
            )
            score_term_dbg = float(a_score / bs - 1.0)
            value_term_dbg = float(a_value / bv - 1.0)
        else:
            score_term_dbg = float((a_score - b_score) / max(bs, fl))
            value_term_dbg = float((a_value - b_value) / max(bv, fl))
            if float(args.loss_wo) > 0.0:
                wo_term = float((a_wo - b_wo) / max(abs(b_wo), fl))
            comp_delta = (
                float(args.accept_comp_w_score) * score_term_dbg
                + float(args.accept_comp_w_value) * value_term_dbg
                + float(args.accept_comp_w_wo) * wo_term
            )
        if str(args.accept_mode) == "composite":
            accept_by_rule = comp_delta <= float(args.accept_comp_max_delta)
        else:
            accept_by_rule = score_ok and value_ok
        accept_layer = bool(int(args.accept_bad_layer) == 1 or accept_by_rule)
        if accept_layer:
            out_state[_basis_key(li, "q")] = u_q2
            out_state[_basis_key(li, "k")] = u_k2
            out_state[_basis_key(li, "v")] = u_v2
        else:
            out_state[_basis_key(li, "q")] = u_q
            out_state[_basis_key(li, "k")] = u_k
            out_state[_basis_key(li, "v")] = u_v
        rec = {
            "layer": li,
            "n_tokens_seen": int(n_tok),
            "anchor_mean_l2_q": float(aq),
            "anchor_mean_l2_k": float(ak),
            "anchor_mean_l2_v": float(av),
            "stable_refine": st,
            "accepted_refined_basis": bool(accept_layer),
            "accept_check": {
                "score_ok": bool(score_ok),
                "value_ok": bool(value_ok),
                "comp_delta": float(comp_delta),
                "accept_mode": str(args.accept_mode),
                "before_score": b_score,
                "after_score": a_score,
                "before_value": b_value,
                "after_value": a_value,
                "before_wo": b_wo,
                "after_wo": a_wo,
                "wo_term": float(wo_term),
                "score_term": float(score_term_dbg),
                "value_term": float(value_term_dbg),
                "accept_comp_metric": str(args.accept_comp_metric),
                "accept_baseline_floor": float(fl),
                "score_ratio_limit": float(args.accept_score_ratio),
                "value_ratio_limit": float(args.accept_value_ratio),
                "comp_max_delta": float(args.accept_comp_max_delta),
            },
            "vs_ref_q": principal_angle_stats(r_q, u_q2),
            "vs_ref_k": principal_angle_stats(r_k, u_k2),
            "vs_ref_v": principal_angle_stats(r_v, u_v2),
        }
        report["per_layer"].append(rec)
        status = "ACCEPT" if accept_layer else "ROLLBACK"
        print(
            f"[layer {li}] {status} "
            f"score {b_score:.4e}->{a_score:.4e} "
            f"value {b_value:.4e}->{a_value:.4e}"
        )

    for li in range(n_layers):
        for br in ("q", "k", "v"):
            key = _basis_key(li, br)
            if key not in out_state:
                out_state[key] = ref_state[key].float().clone()

    out_path = Path(args.out_basis)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, out_path)
    print(f"[saved] {out_path}")

    if args.compare_json:
        cp = Path(args.compare_json)
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[compare] {cp}")


if __name__ == "__main__":
    main()
