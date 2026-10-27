#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从校准数据上的真实 Q/K/V 输出（按 head）做加权二阶矩 + PCA，生成与 eigen_basis.pt 同键、同 rank 的 basis，
并与参考 basis 做子空间/投影矩阵对比（rank 不变）。

- weight_mode=uniform（默认）: 等价于经典 Σ ∝ Y^T Y / n（最大方差方向）。
- weight_mode=nll（A）: α_i = 逐位置 next-token CE。
- weight_mode=nll_fisher（A+B）: γ_i = α_i · β_i，β_i = ||∂L/∂y_q||²（需 backward）。
- weight_mode=nll_attn_entropy（A+C）: Q 用 α_i（NLL）；K/V 用 α_i · c_i，c_i 为该层 attention 对 query 分布的熵（head 平均、batch 归一），前向 output_attentions 即可。注意 repair_keqk=1 时 K 基与 Q 相同，熵权主要落在 V 上；若要让 K 的统计也分离，用 --repair_keqk 0。

路线命名（与 A/B/C 并列）:
  · A — 任务 NLL 加权（对应 weight_mode=nll）
  · B — 在 A 上再乘 ∂L/∂y 能量（nll_fisher 中的 Fisher 项）
  · C — 在 A 上对 K/V 乘 attention 熵等自统计（nll_attn_entropy）
  · D — 与 OPT-30B 参照模型对齐（teacher）：weight_mode=nll_teacher_attn（A+D），见下方。
- weight_mode=nll_teacher_attn（A+D）: Q 用 α_i（NLL，来自 --model 学生）；K/V 用 α_i · d_i，d_i 为学生与教师在当层 attention 概率的逐 query 差异（head 平均后 batch 归一）。需第二套权重时设 --teacher_model 指向教师（如全秩 OPT-30B）；与 --model 相同或未设置时 d_i≡1，退化为纯 A。

- score-aware（可选）: PCA 之后对 Q/K 基做轻量 Adam 精调，直接最小化采样子矩阵上 ||QK^T/sqrt(d)-(P_q Q)(P_k K)^T/sqrt(d)||^2（P=UU^T）。见 --score_lambda 等参数；V 仍仅由协方差 PCA 得到。

用法（在 EigenAttn 目录下，PYTHONPATH 含当前目录）:
  python tools/build_activation_eigen_basis_opt.py \\
    --model /path/to/opt-30b \\
    --ref_basis qkv_outputs/eigen_basis.pt \\
    --out_basis qkv_outputs/eigen_basis_activation.pt \\
    --data_dir /path/to/c4-or-texts \\
    --compare_json qkv_outputs/basis_activation_compare.json

  默认已加大统计量；可用 --preset large_stats；可用 --preset aggressive_nll_qkv。
  层间异质：手动 --hetero_nll_layers；或数据驱动 --hetero_auto_metric attn_entropy|q_grad_energy
  （在 --layers 内按分数取 top_fraction 为强分支，见 --hetero_auto_*）。
  hetero 弱层 baseline=nll 时可用 --baseline_nll_mix_lambda_qk/v 等与全层 nll_*_qk/v 对齐。

依赖: transformers, torch；大模型建议 accelerate（device_map=auto）。
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _resolve_tokenizer_pretrained_path(args) -> str:
    """
    压缩/PEFT 目录常只有权重无 tokenizer。
    优先级: --tokenizer > --teacher_model > --model。
    """
    t = (getattr(args, "tokenizer", None) or "").strip()
    if t:
        return t
    tm = (getattr(args, "teacher_model", None) or "").strip()
    if tm:
        return tm
    return args.model


def _basis_key(layer_idx: int, branch: str) -> str:
    return f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{branch}_proj_basis"


def _parse_layers(spec: str, n_layers: int) -> List[int]:
    s = spec.strip().lower()
    if s == "all" or s == "":
        return list(range(n_layers))
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted({i for i in out if 0 <= i < n_layers})
    if not out:
        raise ValueError(f"no valid layers in {spec!r} (n_layers={n_layers})")
    return out


def _parse_hetero_nll_layers(spec: str, n_layers: int) -> Optional[set]:
    """非空则返回「使用完整 --weight_mode + NLL 超参」的层索引集合；空串表示关闭层间异质。"""
    s = (spec or "").strip()
    if not s:
        return None
    return set(_parse_layers(s, n_layers))


@torch.no_grad()
def _probe_scores_attn_entropy(
    model: torch.nn.Module,
    batch_list: List[torch.Tensor],
    n_layers: int,
    input_device: torch.device,
    max_batches: int,
) -> List[float]:
    """每层平均 attention 熵（对 head/query 平均），在若干 batch 上再平均。scores[li] 越大表示分布越平。"""
    sums = [0.0] * n_layers
    nb = min(len(batch_list), max(1, max_batches))
    for bidx in range(nb):
        batch = batch_list[bidx].to(input_device)
        attn_m = torch.ones_like(batch, dtype=torch.long)
        out = model(
            input_ids=batch,
            attention_mask=attn_m,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        if out.attentions is None:
            raise RuntimeError("model returned attentions=None; need output_attentions=True support")
        for li in range(n_layers):
            if li >= len(out.attentions):
                break
            a = out.attentions[li]
            if not torch.is_tensor(a):
                continue
            if a.dim() == 5:
                a = a.squeeze(2)
            if a.dim() != 4:
                continue
            p = a.float().clamp(min=1e-12)
            p = p / p.sum(dim=-1, keepdim=True)
            H = -(p * torch.log(p)).sum(dim=-1)
            sums[li] += float(H.mean().item())
    return [s / nb for s in sums]


def _probe_scores_q_grad_energy(
    model: torch.nn.Module,
    batch_list: List[torch.Tensor],
    n_layers: int,
    input_device: torch.device,
) -> List[float]:
    """
    每层 q_proj 输出上对 CE 的梯度能量 mean(||∂L/∂q||²)（单 batch，显存/时间开销大，仅作可选）。
    """
    if not batch_list:
        raise RuntimeError("empty batch_list for q_grad_energy probe")
    model.eval()
    batch = batch_list[0].to(input_device)
    attn_m = torch.ones_like(batch, dtype=torch.long)
    stor: Dict[int, torch.Tensor] = {}
    hooks: List[object] = []

    def _mk_hook(li: int):
        def _h(_m, _inp, out):
            if torch.is_tensor(out):
                out.retain_grad()
                stor[li] = out

        return _h

    for li in range(n_layers):
        attn = model.model.decoder.layers[li].self_attn
        if not hasattr(attn, "q_proj"):
            raise RuntimeError(f"layer {li}: self_attn has no q_proj")
        hooks.append(attn.q_proj.register_forward_hook(_mk_hook(li)))

    emb = model.get_input_embeddings()
    ie = emb(batch)
    ie = ie.detach().requires_grad_(True)
    try:
        with torch.enable_grad():
            out_lm = model(
                inputs_embeds=ie,
                attention_mask=attn_m,
                use_cache=False,
                return_dict=True,
                labels=batch,
            )
            loss = out_lm.loss
            model.zero_grad(set_to_none=True)
            loss.backward()
        scores: List[float] = []
        for li in range(n_layers):
            t = stor.get(li)
            if t is None or t.grad is None:
                scores.append(0.0)
            else:
                scores.append(float(t.grad.detach().float().pow(2).mean().item()))
    finally:
        for h in hooks:
            h.remove()
        model.zero_grad(set_to_none=True)

    return scores


def _select_hetero_auto_layers(
    layer_ids: List[int],
    scores: List[float],
    top_fraction: float,
    strong_side: str,
) -> set:
    """在 layer_ids 子集内按 scores 排序，取前 ceil(len*top_fraction) 层为强分支。"""
    if not layer_ids:
        return set()
    tf = float(max(0.0, min(1.0, top_fraction)))
    pairs = [(li, scores[li]) for li in layer_ids if 0 <= li < len(scores)]
    reverse = strong_side.lower() != "low"
    pairs.sort(key=lambda x: x[1], reverse=reverse)
    k = max(1, int(math.ceil(len(layer_ids) * tf)))
    return {p[0] for p in pairs[:k]}


def iter_texts_from_dir(data_dir: str, max_files: int = 120, max_lines: int = 800000):
    exts = ["*.txt", "*.jsonl", "*.json", "*.gz"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", e), recursive=True))
    files = sorted(files)[:max_files]
    if not files:
        return
    n = 0
    for fp in files:
        opener = gzip.open if fp.endswith(".gz") else open
        try:
            with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
                if fp.endswith(".json") and (not fp.endswith(".gz")):
                    try:
                        obj = json.load(f)
                        if isinstance(obj, list):
                            for it in obj:
                                t = None
                                if isinstance(it, dict):
                                    t = it.get("text") or it.get("content")
                                elif isinstance(it, str):
                                    t = it
                                if isinstance(t, str) and t.strip():
                                    yield t.strip()
                                    n += 1
                                    if n >= max_lines:
                                        return
                        elif isinstance(obj, dict):
                            t = obj.get("text") or obj.get("content")
                            if isinstance(t, str) and t.strip():
                                yield t.strip()
                    except Exception:
                        pass
                    continue

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = None
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            j = json.loads(line)
                            if isinstance(j, dict):
                                t = j.get("text") or j.get("content")
                        except Exception:
                            t = None
                    if t is None:
                        t = line
                    if isinstance(t, str) and t.strip():
                        yield t.strip()
                        n += 1
                        if n >= max_lines:
                            return
        except Exception:
            continue


def build_token_stream(tok, texts_iter, target_tokens: int, seed: int):
    random.seed(seed)
    ids: List[int] = []
    total = 0
    for t in texts_iter:
        enc = tok(t, add_special_tokens=False)["input_ids"]
        if not enc:
            continue
        ids.extend(enc + [tok.eos_token_id])
        total += len(enc) + 1
        if total >= target_tokens:
            break
    if total == 0:
        return None
    return torch.tensor(ids, dtype=torch.long)


def batchify_stream_cpu(stream_cpu: torch.Tensor, seqlen: int, batch_size: int):
    N = stream_cpu.numel()
    assert N > seqlen + 1
    while True:
        starts = torch.randint(low=0, high=N - seqlen - 1, size=(batch_size,))
        x = torch.stack([stream_cpu[s : s + seqlen] for s in starts], dim=0)
        attn = torch.ones_like(x, dtype=torch.long)
        yield x, attn


def _per_position_nll_weights(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    power: float = 1.0,
    clip_percentile: float = 100.0,
    mix_lambda: float = 1.0,
) -> torch.Tensor:
    """逐位置 next-token CE，与 q/k/v 在 (b,t) 对齐；最后一位置无 next token，权重为 0。"""
    bs, seq, v = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    nll = F.cross_entropy(
        shift_logits.view(-1, v),
        shift_labels.view(-1),
        reduction="none",
    ).view(bs, seq - 1)
    w = torch.zeros(bs, seq, device=logits.device, dtype=torch.float32)
    w[:, : seq - 1] = nll.float()
    if power != 1.0:
        w = torch.pow(torch.clamp(w, min=0.0), power)
    w = torch.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0).clamp(min=0.0, max=1e12)
    # 百分位裁剪：抑制极端高 NLL token 的过强主导
    if clip_percentile < 100.0:
        cp = float(max(0.0, min(100.0, clip_percentile)))
        q = cp / 100.0
        try:
            upper = torch.quantile(w.reshape(-1), q)
            w = torch.clamp(w, max=float(upper.item()))
        except RuntimeError:
            pass
    # batch 内归一化，避免不同 batch 尺度漂移
    wm = w.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    w = w / wm
    # 与均匀权重混合：alpha'=(1-lambda)*1 + lambda*alpha
    lm = float(max(0.0, min(1.0, mix_lambda)))
    if lm < 1.0:
        w = (1.0 - lm) + lm * w
    return w


def _attention_entropy_factor_per_position(
    attentions: Optional[Tuple[torch.Tensor, ...]],
    layer_idx: int,
    ref_shape: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    当前层 attention 概率对 query 维的熵（对 head 平均），shape [bs, seq]。
    再按 batch 归一化使均值为 1，作为 C 中与 NLL 相乘的标量（无 teacher）。
    """
    bs, seq = ref_shape
    fallback = torch.ones(bs, seq, device=device, dtype=torch.float32)
    if attentions is None or layer_idx >= len(attentions):
        return fallback
    a = attentions[layer_idx]
    if not torch.is_tensor(a):
        return fallback
    a = a.float()
    if a.dim() == 5:
        a = a.squeeze(2)
    # [bs, nh, sq, sk]
    if a.dim() != 4:
        return fallback
    p = a.clamp(min=1e-12)
    p = p / p.sum(dim=-1, keepdim=True)
    H = -(p * torch.log(p)).sum(dim=-1)  # [bs, nh, sq]
    Hm = H.mean(dim=1)  # [bs, sq]
    Hm = torch.nan_to_num(Hm, nan=1.0, posinf=1.0, neginf=1.0)
    Hm = Hm / (Hm.mean(dim=1, keepdim=True).clamp(min=1e-8))
    if tuple(Hm.shape) != (bs, seq):
        return fallback
    return Hm.to(device=device, dtype=torch.float32)


def _attn_student_teacher_discrepancy_per_query(
    att_s: torch.Tensor,
    att_t: torch.Tensor,
    ref_shape: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    学生/教师 attention 在逐 query 上的差异度，shape [bs, seq]，batch 内归一到均值 1。
    att_*: [bs, nh, sq, sk]（已 softmax）。
    """
    bs, seq = ref_shape
    fallback = torch.ones(bs, seq, device=device, dtype=torch.float32)
    if not (torch.is_tensor(att_s) and torch.is_tensor(att_t)):
        return fallback
    if att_s.shape != att_t.shape:
        return fallback
    a = att_s.float().clamp(min=0.0)
    b = att_t.float().clamp(min=0.0)
    # 每个 query：各 head 上对 key 维求和方差再平均
    diff = (a - b).pow(2).sum(dim=-1)  # [bs, nh, sq]
    d = diff.mean(dim=1)
    d = torch.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)
    d = d / (d.mean(dim=1, keepdim=True).clamp(min=1e-8))
    if tuple(d.shape) != (bs, seq):
        return fallback
    return d.to(device=device, dtype=torch.float32)


@torch.no_grad()
def _pca_cols_from_cov(YtY: torch.Tensor, n: float, r: int) -> torch.Tensor:
    """YtY: [d,d] symmetric; n 为归一化量（token 数或 Σ α）。Return V [d,r] orthonormal。"""
    d = YtY.shape[0]
    if n <= 0:
        return torch.eye(d, dtype=YtY.dtype, device=YtY.device)[:, :r]
    C = (YtY / float(n)).float()
    C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    # 对角抖动，减轻 Fisher/bf16 带来的病态与 eigh 不收敛
    tr = torch.diagonal(C).sum().abs().clamp(min=1e-12)
    jitter = float(1e-5 * (tr.item() / max(d, 1)))
    C = C + jitter * torch.eye(d, dtype=C.dtype, device=C.device)
    try:
        _w, V = torch.linalg.eigh(C)
    except RuntimeError:
        return torch.eye(d, dtype=torch.float32, device=C.device)[:, :r]
    V = torch.nan_to_num(V.float(), nan=0.0)
    out = V[:, -r:].contiguous()
    if not torch.isfinite(out).all():
        return torch.eye(d, dtype=torch.float32, device=C.device)[:, :r]
    return out


@torch.no_grad()
def _sorted_eigvals_from_cov(YtY: torch.Tensor, n: float) -> torch.Tensor:
    """Return descending nonnegative eigenvalues of normalized covariance."""
    d = YtY.shape[0]
    if n <= 0:
        return torch.ones(d, dtype=torch.float32)
    C = (YtY / float(n)).float()
    C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    tr = torch.diagonal(C).sum().abs().clamp(min=1e-12)
    jitter = float(1e-5 * (tr.item() / max(d, 1)))
    C = C + jitter * torch.eye(d, dtype=C.dtype, device=C.device)
    try:
        w = torch.linalg.eigvalsh(C)
    except RuntimeError:
        w = torch.ones(d, dtype=torch.float32, device=C.device)
    w = torch.nan_to_num(w.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    return torch.flip(w, dims=[0]).contiguous()


def _rank_from_energy_threshold(evals_desc: torch.Tensor, keep_energy: float) -> int:
    """Minimum r such that cumulative explained energy >= keep_energy."""
    keep = float(max(0.0, min(1.0, keep_energy)))
    d = int(evals_desc.numel())
    if d <= 0:
        return 0
    s = evals_desc.sum().item()
    if s <= 1e-20:
        return 1
    if keep <= 0.0:
        return 1
    if keep >= 1.0:
        return d
    c = torch.cumsum(evals_desc, dim=0) / float(s)
    idx = int(torch.searchsorted(c, torch.tensor(keep, dtype=c.dtype), right=False).item())
    return int(min(max(idx + 1, 1), d))


def _tail_ratio_at_rank(evals_desc: torch.Tensor, r: int) -> float:
    d = int(evals_desc.numel())
    if d <= 0:
        return 0.0
    rr = int(min(max(r, 0), d))
    s = float(evals_desc.sum().item())
    if s <= 1e-20:
        return 0.0
    kept = float(evals_desc[:rr].sum().item()) if rr > 0 else 0.0
    return float(max(0.0, min(1.0, 1.0 - kept / s)))


def _rank_from_tail_budget(evals_desc: torch.Tensor, tail_budget: float) -> int:
    """Minimum r such that residual energy ratio <= tail_budget."""
    tb = float(max(0.0, min(1.0, tail_budget)))
    d = int(evals_desc.numel())
    for r in range(1, d + 1):
        if _tail_ratio_at_rank(evals_desc, r) <= tb:
            return r
    return d


def _allocate_global_budget_greedy(
    evals_per_layer: List[torch.Tensor],
    init_ranks: List[int],
    total_rank_budget: int,
) -> List[int]:
    """
    Reduce per-layer ranks under a global rank budget with minimum proxy loss increase.
    Proxy loss is residual energy ratio from eigen spectrum.
    """
    L = len(init_ranks)
    ranks = [int(max(1, r)) for r in init_ranks]
    budget = int(max(total_rank_budget, L))
    cur = int(sum(ranks))
    if cur <= budget:
        return ranks
    while cur > budget:
        best_li = -1
        best_delta = None
        for li in range(L):
            if ranks[li] <= 1:
                continue
            ev = evals_per_layer[li]
            old_r = ranks[li]
            new_r = old_r - 1
            delta = _tail_ratio_at_rank(ev, new_r) - _tail_ratio_at_rank(ev, old_r)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_li = li
        if best_li < 0:
            break
        ranks[best_li] -= 1
        cur -= 1
    return ranks


def _orthonormalize(U: torch.Tensor) -> torch.Tensor:
    """U: [d,r] -> QR thin Q."""
    if U.numel() == 0:
        return U
    q, _ = torch.linalg.qr(U, mode="reduced")
    return q[:, : U.shape[1]].contiguous()


def principal_angle_stats(U_old: torch.Tensor, U_new: torch.Tensor) -> Dict[str, float]:
    """
    U_old, U_new: [d,r] column orthonormal (approximately).
    Principal angles via SVD of U_old^T U_new.
    """
    nan_pack = {
        "mean_deg": float("nan"),
        "max_deg": float("nan"),
        "min_cos": float("nan"),
        "proj_fro_diff": float("nan"),
        "proj_fro_diff_norm": float("nan"),
    }
    Uo = torch.nan_to_num(U_old.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    Un = torch.nan_to_num(U_new.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    Uo = _orthonormalize(Uo)
    Un = _orthonormalize(Un)
    r = min(Uo.shape[1], Un.shape[1])
    if r == 0:
        return nan_pack
    M = torch.nan_to_num((Uo.t() @ Un), nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(M).all():
        return nan_pack
    try:
        s = torch.linalg.svdvals(M)
    except RuntimeError:
        return nan_pack
    s = torch.clamp(s, -1.0, 1.0)
    ang = torch.acos(s) * (180.0 / math.pi)
    Po = Uo @ Uo.t()
    Pn = Un @ Un.t()
    fro = torch.linalg.matrix_norm(Po - Pn, ord="fro").item()
    r_eff = float(r)
    return {
        "mean_deg": float(ang.mean().item()),
        "max_deg": float(ang.max().item()),
        "min_cos": float(s.min().item()),
        "proj_fro_diff": float(fro),
        "proj_fro_diff_norm": float(fro / max(math.sqrt(2.0 * r_eff), 1e-12)),
    }


class _PerHeadCovAccumulator:
    def __init__(self, num_heads: int, head_dim: int, device_cpu: torch.device):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device_cpu = device_cpu
        z = torch.zeros(head_dim, head_dim, dtype=torch.float32, device=device_cpu)
        self.yty_q = [z.clone() for _ in range(num_heads)]
        self.yty_k = [z.clone() for _ in range(num_heads)]
        self.yty_v = [z.clone() for _ in range(num_heads)]
        self.n_tokens = 0
        # Σ α：各分支分别归一化（Q 与 K/V 可不同权，如 A+C）
        self.alpha_sum_q = 0.0
        self.alpha_sum_k = 0.0
        self.alpha_sum_v = 0.0

    def _add_branch(self, y_list: List[torch.Tensor], y_flat: torch.Tensor):
        # y_flat: [N, hidden] on CPU float32（若加权，行已乘 sqrt(α)）
        N = y_flat.shape[0]
        if N == 0:
            return
        hd = self.head_dim
        for h in range(self.num_heads):
            sl = y_flat[:, h * hd : (h + 1) * hd]
            y_list[h].addmm_(sl.t(), sl)

    def update_from_outputs(
        self,
        yq: torch.Tensor,
        yk: torch.Tensor,
        yv: torch.Tensor,
        alpha_bt: Optional[torch.Tensor] = None,
        alpha_kv_bt: Optional[torch.Tensor] = None,
        alpha_v_bt: Optional[torch.Tensor] = None,
    ):
        """y*: [bs, seq, hidden]。
        alpha_bt: Q（及与 Q 同权的 K）的 [bs, seq]。
        alpha_kv_bt 若给定且未给 alpha_v_bt：Q 用 alpha_bt，K/V 共用 alpha_kv_bt（A+C 等）。
        alpha_v_bt 若给定：Q、K 用 alpha_bt，V 单独用 alpha_v_bt（与 alpha_kv_bt 互斥，勿同时设）。
        """
        yq = yq.detach().float().cpu()
        yk = yk.detach().float().cpu()
        yv = yv.detach().float().cpu()
        bs, seq, h = yq.shape
        assert h == self.num_heads * self.head_dim
        N = bs * seq
        self.n_tokens += N
        fq = yq.reshape(N, h)
        fk = yk.reshape(N, h)
        fv = yv.reshape(N, h)
        if alpha_bt is not None:
            a = alpha_bt.detach().float().cpu().reshape(-1)
            if a.numel() != N:
                raise RuntimeError(f"alpha_q shape {tuple(a.shape)} != flattened {N}")
            a = torch.nan_to_num(a, nan=0.0, posinf=1e12, neginf=0.0).clamp(min=0.0, max=1e12)
            saq = torch.sqrt(a + 1e-20)
            fq = torch.nan_to_num(fq * saq.unsqueeze(1), nan=0.0, posinf=1e6, neginf=-1e6)
            self.alpha_sum_q += float(a.sum().item())
            if alpha_v_bt is not None:
                if alpha_kv_bt is not None:
                    raise RuntimeError("use either alpha_kv_bt or alpha_v_bt, not both")
                fk = torch.nan_to_num(fk * saq.unsqueeze(1), nan=0.0, posinf=1e6, neginf=-1e6)
                self.alpha_sum_k += float(a.sum().item())
                av = alpha_v_bt.detach().float().cpu().reshape(-1)
                if av.numel() != N:
                    raise RuntimeError(f"alpha_v shape {tuple(av.shape)} != flattened {N}")
                av = torch.nan_to_num(av, nan=0.0, posinf=1e12, neginf=0.0).clamp(min=0.0, max=1e12)
                sav = torch.sqrt(av + 1e-20)
                fv = torch.nan_to_num(fv * sav.unsqueeze(1), nan=0.0, posinf=1e6, neginf=-1e6)
                self.alpha_sum_v += float(av.sum().item())
            elif alpha_kv_bt is None:
                self.alpha_sum_k += float(a.sum().item())
                self.alpha_sum_v += float(a.sum().item())
                fk = torch.nan_to_num(fk * saq.unsqueeze(1), nan=0.0, posinf=1e6, neginf=-1e6)
                fv = torch.nan_to_num(fv * saq.unsqueeze(1), nan=0.0, posinf=1e6, neginf=-1e6)
            else:
                ak = alpha_kv_bt.detach().float().cpu().reshape(-1)
                if ak.numel() != N:
                    raise RuntimeError(f"alpha_kv shape {tuple(ak.shape)} != flattened {N}")
                ak = torch.nan_to_num(ak, nan=0.0, posinf=1e12, neginf=0.0).clamp(min=0.0, max=1e12)
                sak = torch.sqrt(ak + 1e-20)
                fk = torch.nan_to_num(fk * sak.unsqueeze(1), nan=0.0, posinf=1e6, neginf=-1e6)
                fv = torch.nan_to_num(fv * sak.unsqueeze(1), nan=0.0, posinf=1e6, neginf=-1e6)
                self.alpha_sum_k += float(ak.sum().item())
                self.alpha_sum_v += float(ak.sum().item())
        else:
            self.alpha_sum_q += float(N)
            self.alpha_sum_k += float(N)
            self.alpha_sum_v += float(N)
        self._add_branch(self.yty_q, fq)
        self._add_branch(self.yty_k, fk)
        self._add_branch(self.yty_v, fv)


def _build_u_from_covs(
    acc: _PerHeadCovAccumulator,
    rank_kq_per_head: int,
    rank_v_per_head: int,
    repair_keqk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = acc.num_heads * acc.head_dim
    rank_kq = rank_kq_per_head * acc.num_heads
    rank_v = rank_v_per_head * acc.num_heads
    u_q = torch.zeros(hidden, rank_kq, dtype=torch.float32)
    u_k = torch.zeros(hidden, rank_kq, dtype=torch.float32)
    u_v = torch.zeros(hidden, rank_v, dtype=torch.float32)
    nq = max(acc.alpha_sum_q, 1e-12)
    nk = max(acc.alpha_sum_k, 1e-12)
    nv = max(acc.alpha_sum_v, 1e-12)
    for h in range(acc.num_heads):
        row = slice(h * acc.head_dim, (h + 1) * acc.head_dim)
        col_kq = slice(h * rank_kq_per_head, (h + 1) * rank_kq_per_head)
        col_v = slice(h * rank_v_per_head, (h + 1) * rank_v_per_head)
        Vq = _pca_cols_from_cov(acc.yty_q[h], nq, rank_kq_per_head)
        Vk = _pca_cols_from_cov(acc.yty_k[h], nk, rank_kq_per_head) if repair_keqk != 1 else Vq
        Vv = _pca_cols_from_cov(acc.yty_v[h], nv, rank_v_per_head)
        u_q[row, col_kq] = Vq
        u_k[row, col_kq] = Vk if repair_keqk != 1 else Vq
        u_v[row, col_v] = Vv
    return u_q, u_k, u_v


def _refine_qk_bases_score_mse(
    model: torch.nn.Module,
    layer_idx: int,
    batch_list: List[torch.Tensor],
    u_q: torch.Tensor,
    u_k: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rank_kq_per_head: int,
    repair_keqk: int,
    input_device: torch.device,
    residual_wrt_ref: bool,
    ref_u_q: Optional[torch.Tensor],
    ref_u_k: Optional[torch.Tensor],
    *,
    steps: int,
    lr: float,
    score_lambda: float,
    score_prox_to_pca: float,
    sample_m: int,
    sample_n: int,
    max_batches: int,
    seed: int,
    center_logits: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    在 PCA 得到的 U_q/U_k 上做一次轻量 score-aware 精调：最小化采样子矩阵上
    || QK^T/sqrt(d) - (P_q Q)(P_k K)^T/sqrt(d) ||_F^2，其中 P=UU^T，Q/K 为 q_proj/k_proj 输出（与协方差统计一致可做 residual）。
    仅优化 Q/K 基，不反传模型权重；每步后对 U 做 QR 收缩保持列正交。
    """
    if score_lambda <= 0.0 or steps <= 0:
        return u_q, u_k, {}

    layer = model.model.decoder.layers[layer_idx]
    attn = layer.self_attn
    hook_out: Dict[str, torch.Tensor] = {}

    def _maybe_resid(yq: torch.Tensor, yk: torch.Tensor):
        if not residual_wrt_ref or ref_u_q is None:
            return yq, yk
        if ref_u_k is None:
            raise RuntimeError("score refine + residual needs ref_u_k")
        return (
            _subtract_ref_subspace_per_head(yq, ref_u_q, num_heads, head_dim, rank_kq_per_head),
            _subtract_ref_subspace_per_head(yk, ref_u_k, num_heads, head_dim, rank_kq_per_head),
        )

    def qh(_m, _inp, out):
        hook_out["q"] = out

    def kh(_m, _inp, out):
        hook_out["k"] = out

    hq = attn.q_proj.register_forward_hook(qh)
    hk = attn.k_proj.register_forward_hook(kh)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
    try:
        model.eval()
        for batch in batch_list[: max(1, max_batches)]:
            hook_out.clear()
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0]
            elif isinstance(batch, dict):
                input_ids = batch.get("input_ids")
            else:
                input_ids = batch
            if not torch.is_tensor(input_ids):
                continue
            input_ids = input_ids.to(input_device)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            with torch.no_grad():
                _ = model(input_ids=input_ids, use_cache=False, return_dict=True)
            if "q" not in hook_out or "k" not in hook_out:
                continue
            rq, rk = _maybe_resid(hook_out["q"], hook_out["k"])
            cache.append((rq.detach().cpu().float(), rk.detach().cpu().float()))
    finally:
        hq.remove()
        hk.remove()
        model.config.use_cache = use_cache

    if not cache:
        return u_q, u_k, {"score_refine_skipped": 1.0}

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U0_q = [
        u_q[h * head_dim : (h + 1) * head_dim, h * rank_kq_per_head : (h + 1) * rank_kq_per_head]
        .float()
        .to(dev)
        .clone()
        for h in range(num_heads)
    ]
    U0_k = [
        u_k[h * head_dim : (h + 1) * head_dim, h * rank_kq_per_head : (h + 1) * rank_kq_per_head]
        .float()
        .to(dev)
        .clone()
        for h in range(num_heads)
    ]
    params: List[torch.nn.Parameter] = []
    if int(repair_keqk) == 1:
        Uq_params = [torch.nn.Parameter(U0_q[h].clone()) for h in range(num_heads)]
        Uk_params = Uq_params
        params.extend(Uq_params)
    else:
        Uq_params = [torch.nn.Parameter(U0_q[h].clone()) for h in range(num_heads)]
        Uk_params = [torch.nn.Parameter(U0_k[h].clone()) for h in range(num_heads)]
        params.extend(Uq_params)
        params.extend(Uk_params)

    opt = torch.optim.Adam(params, lr=float(lr))
    scale = 1.0 / math.sqrt(float(head_dim))

    def _batch_mse(
        Uq_list: List[torch.Tensor], Uk_list: List[torch.Tensor], step_aug: int = 0
    ) -> torch.Tensor:
        tot = torch.zeros((), device=dev, dtype=torch.float32)
        nb = 0
        for bidx, (q_cpu, k_cpu) in enumerate(cache):
            q = q_cpu.to(dev)
            k = k_cpu.to(dev)
            bs, seq, hs = q.shape
            assert hs == num_heads * head_dim
            T = bs * seq
            m = int(min(max(1, sample_m), T))
            n = int(min(max(1, sample_n), T))
            g_cpu = torch.Generator()
            g_cpu.manual_seed(
                int(seed) + int(layer_idx) * 10007 + int(bidx) * 1001 + int(step_aug) * 99991
            )
            idx_i = torch.randperm(T, generator=g_cpu)[:m].to(dev)
            idx_j = torch.randperm(T, generator=g_cpu)[:n].to(dev)
            nb += 1
            for h in range(num_heads):
                sl = slice(h * head_dim, (h + 1) * head_dim)
                qh = q[..., sl].reshape(T, head_dim)
                kh = k[..., sl].reshape(T, head_dim)
                Qi = qh.index_select(0, idx_i)
                Kj = kh.index_select(0, idx_j)
                Uq = Uq_list[h]
                Uk = Uk_list[h]
                Pq = Uq @ Uq.t()
                Pk = Uk @ Uk.t()
                Qi_hat = Qi @ Pq
                Kj_hat = Kj @ Pk
                S = (Qi @ Kj.t()) * scale
                Sh = (Qi_hat @ Kj_hat.t()) * scale
                if center_logits:
                    S = S - S.mean(dim=1, keepdim=True)
                    Sh = Sh - Sh.mean(dim=1, keepdim=True)
                tot = tot + F.mse_loss(Sh, S)
        denom = float(max(1, nb) * num_heads)
        return tot / denom

    @torch.no_grad()
    def _retract():
        if int(repair_keqk) == 1:
            for h in range(num_heads):
                Uq_params[h].copy_(_orthonormalize(Uq_params[h]))
        else:
            for h in range(num_heads):
                Uq_params[h].copy_(_orthonormalize(Uq_params[h]))
                Uk_params[h].copy_(_orthonormalize(Uk_params[h]))

    @torch.no_grad()
    def _pack_u() -> Tuple[torch.Tensor, torch.Tensor]:
        uq_new = u_q.clone()
        uk_new = u_k.clone()
        for h in range(num_heads):
            row = slice(h * head_dim, (h + 1) * head_dim)
            cq = slice(h * rank_kq_per_head, (h + 1) * rank_kq_per_head)
            uq_new[row, cq] = Uq_params[h].float().cpu()
            uk_new[row, cq] = Uk_params[h].float().cpu()
        return uq_new, uk_new

    with torch.no_grad():
        uq0 = [Uq_params[h].detach() for h in range(num_heads)]
        uk0 = [Uk_params[h].detach() for h in range(num_heads)]
        mse0 = _batch_mse(uq0, uk0, step_aug=-1).item()

    for step_i in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        mse = _batch_mse(Uq_params, Uk_params, step_aug=step_i)
        prox = torch.zeros((), device=dev, dtype=torch.float32)
        if score_prox_to_pca > 0.0:
            for h in range(num_heads):
                prox = prox + (Uq_params[h] - U0_q[h]).float().pow(2).mean()
                if int(repair_keqk) != 1:
                    prox = prox + (Uk_params[h] - U0_k[h]).float().pow(2).mean()
        loss = float(score_lambda) * mse + float(score_prox_to_pca) * prox
        loss.backward()
        opt.step()
        _retract()

    with torch.no_grad():
        uq1 = [Uq_params[h].detach() for h in range(num_heads)]
        uk1 = [Uk_params[h].detach() for h in range(num_heads)]
        mse1 = _batch_mse(uq1, uk1, step_aug=int(steps)).item()

    u_out, k_out = _pack_u()
    return u_out, k_out, {
        "score_mse_submatrix_mean_before": float(mse0),
        "score_mse_submatrix_mean_after": float(mse1),
    }


def _subtract_ref_subspace_per_head(
    y: torch.Tensor,
    U_full: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rank_per_head: int,
) -> torch.Tensor:
    """
    y: [bs, seq, hidden], U_full: [hidden, num_heads * rank_per_head]，与 eigen_basis 中按 head 分块的列一致。
    返回 y - P_ref y，其中 P_ref 为各 head 上 ref 列张成的正交投影（假定 ref 列已单位正交，与 PCA 输出一致）。
    """
    dev = y.device
    y = y.detach().float()
    U_full = U_full.float().to(device=dev)
    bs, seq, hs = y.shape
    assert hs == num_heads * head_dim
    assert U_full.shape[0] == hs
    assert U_full.shape[1] == num_heads * rank_per_head
    yf = y.reshape(-1, num_heads, head_dim)
    parts: List[torch.Tensor] = []
    for h in range(num_heads):
        row = slice(h * head_dim, (h + 1) * head_dim)
        cs = slice(h * rank_per_head, (h + 1) * rank_per_head)
        Uh = U_full[row, cs]
        yh = yf[:, h, :]
        coeff = torch.matmul(yh, Uh)
        proj = torch.matmul(coeff, Uh.t())
        parts.append(yh - proj)
    stacked = torch.stack(parts, dim=1)
    return stacked.reshape(bs, seq, hs)


def collect_activation_basis_one_layer(
    model: torch.nn.Module,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    rank_kq_per_head: int,
    rank_v_per_head: int,
    repair_keqk: int,
    data_iter,
    input_device: torch.device,
    weight_mode: str = "uniform",
    nll_power: float = 1.0,
    nll_clip_percentile: float = 100.0,
    nll_mix_lambda: float = 1.0,
    nll_power_qk: Optional[float] = None,
    nll_power_v: Optional[float] = None,
    nll_clip_percentile_qk: Optional[float] = None,
    nll_clip_percentile_v: Optional[float] = None,
    nll_mix_lambda_qk: Optional[float] = None,
    nll_mix_lambda_v: Optional[float] = None,
    fisher_batch_norm: int = 1,
    teacher_model: Optional[torch.nn.Module] = None,
    teacher_distinct: bool = False,
    residual_wrt_ref: bool = False,
    ref_u_q: Optional[torch.Tensor] = None,
    ref_u_k: Optional[torch.Tensor] = None,
    ref_u_v: Optional[torch.Tensor] = None,
    score_lambda: float = 0.0,
    score_refine_steps: int = 24,
    score_refine_lr: float = 0.08,
    score_prox_to_pca: float = 0.0,
    score_sample_m: int = 64,
    score_sample_n: int = 64,
    score_refine_batches: int = 4,
    score_refine_seed: int = 0,
    score_center_logits: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Dict[str, object]]:
    layer = model.model.decoder.layers[layer_idx]
    attn = layer.self_attn
    acc = _PerHeadCovAccumulator(num_heads, head_dim, torch.device("cpu"))
    hook_out: Dict[str, torch.Tensor] = {}
    batches = data_iter if isinstance(data_iter, list) else list(data_iter)

    def _maybe_apply_residual(yq: torch.Tensor, yk: torch.Tensor, yv: torch.Tensor):
        if not residual_wrt_ref or ref_u_q is None:
            return yq, yk, yv
        if ref_u_k is None or ref_u_v is None:
            raise RuntimeError("cov_residual_to_ref requires ref_u_q, ref_u_k, ref_u_v")
        return (
            _subtract_ref_subspace_per_head(yq, ref_u_q, num_heads, head_dim, rank_kq_per_head),
            _subtract_ref_subspace_per_head(yk, ref_u_k, num_heads, head_dim, rank_kq_per_head),
            _subtract_ref_subspace_per_head(yv, ref_u_v, num_heads, head_dim, rank_v_per_head),
        )

    def q_hook(_m, _inp, out):
        if weight_mode == "nll_fisher":
            out.retain_grad()
        hook_out["q"] = out

    def k_hook(_m, _inp, out):
        hook_out["k"] = out

    def v_hook(_m, _inp, out):
        hook_out["v"] = out

    hq = attn.q_proj.register_forward_hook(q_hook)
    hk = attn.k_proj.register_forward_hook(k_hook)
    hv = attn.v_proj.register_forward_hook(v_hook)
    hooks = [hq, hk, hv]

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    def _resolve_nll(opt: Optional[float], fb: float) -> float:
        return float(opt) if opt is not None else float(fb)

    def _one_batch(input_ids: torch.Tensor, grad_mode_nll_fisher: bool):
        """对单个 batch 前向并累加；grad_mode_nll_fisher 时走 A+B。"""
        alpha_bt: Optional[torch.Tensor] = None
        alpha_v_bt: Optional[torch.Tensor] = None
        if grad_mode_nll_fisher:
            # A+B：用可反传的 inputs_embeds 建立到 q_proj 的图；L = mean next-token CE
            emb = model.get_input_embeddings()
            ie = emb(input_ids)
            ie = ie.detach().requires_grad_(True)
            attn_m = torch.ones_like(input_ids, dtype=torch.long)
            out_lm = model(
                inputs_embeds=ie,
                attention_mask=attn_m,
                use_cache=False,
                return_dict=True,
            )
            logits = out_lm.logits
            alpha_bt = _per_position_nll_weights(
                logits,
                input_ids,
                power=nll_power,
                clip_percentile=nll_clip_percentile,
                mix_lambda=nll_mix_lambda,
            )
            v = logits.size(-1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, v),
                shift_labels.view(-1),
                reduction="mean",
            )
            model.zero_grad(set_to_none=True)
            loss.backward()
            gq = hook_out["q"].grad
            if gq is None:
                beta = torch.ones_like(alpha_bt)
            else:
                beta = gq.float().pow(2).sum(dim=-1)
            # device_map 多卡时 logits 与当前层 q_proj 可能在不同 GPU，对齐后再乘
            alpha_bt = alpha_bt.to(device=beta.device, dtype=torch.float32)
            beta = torch.nan_to_num(beta.float(), nan=0.0, posinf=1e12, neginf=0.0)
            gamma = alpha_bt * beta
            gamma = torch.nan_to_num(gamma, nan=0.0, posinf=1e12, neginf=0.0)
            if fisher_batch_norm:
                gm = gamma.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
                gamma = gamma / gm
            alpha_bt = torch.clamp(gamma, min=0.0, max=1e12) + 1e-20
            model.zero_grad(set_to_none=True)
            if "q" in hook_out and "k" in hook_out and "v" in hook_out:
                rq, rk, rv = _maybe_apply_residual(
                    hook_out["q"].detach(),
                    hook_out["k"].detach(),
                    hook_out["v"].detach(),
                )
                acc.update_from_outputs(
                    rq,
                    rk,
                    rv,
                    alpha_bt=alpha_bt.detach(),
                )
            return

        if weight_mode == "nll_teacher_attn":
            out_lm = model(
                input_ids=input_ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
            logits = out_lm.logits
            bs, seq = input_ids.shape
            alpha_q = _per_position_nll_weights(
                logits,
                input_ids,
                power=nll_power,
                clip_percentile=nll_clip_percentile,
                mix_lambda=nll_mix_lambda,
            )
            if teacher_distinct and teacher_model is not None:
                tdev = teacher_model.get_input_embeddings().weight.device
                ids_t = input_ids.to(tdev)
                out_t = teacher_model(
                    input_ids=ids_t,
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
                att_s_layer = (
                    out_lm.attentions[layer_idx]
                    if out_lm.attentions is not None and layer_idx < len(out_lm.attentions)
                    else None
                )
                att_t_layer = (
                    out_t.attentions[layer_idx]
                    if out_t.attentions is not None and layer_idx < len(out_t.attentions)
                    else None
                )
                if att_s_layer is not None and att_t_layer is not None and att_s_layer.shape == att_t_layer.shape:
                    att_s_layer = att_s_layer.to(device=att_t_layer.device, dtype=torch.float32)
                    att_t_layer = att_t_layer.float()
                    d = _attn_student_teacher_discrepancy_per_query(
                        att_s_layer, att_t_layer, (bs, seq), alpha_q.device
                    )
                else:
                    d = torch.ones(bs, seq, device=alpha_q.device, dtype=torch.float32)
            else:
                d = torch.ones(bs, seq, device=alpha_q.device, dtype=torch.float32)
            alpha_kv = alpha_q * d
            alpha_kv = torch.nan_to_num(alpha_kv, nan=0.0, posinf=1e12, neginf=0.0)
            alpha_kv = torch.clamp(alpha_kv, min=0.0, max=1e12)
            if "q" in hook_out and "k" in hook_out and "v" in hook_out:
                rq, rk, rv = _maybe_apply_residual(hook_out["q"], hook_out["k"], hook_out["v"])
                acc.update_from_outputs(
                    rq,
                    rk,
                    rv,
                    alpha_bt=alpha_q,
                    alpha_kv_bt=alpha_kv,
                )
            return

        if weight_mode == "nll_attn_entropy":
            out_lm = model(
                input_ids=input_ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
            logits = out_lm.logits
            bs, seq = input_ids.shape
            alpha_q = _per_position_nll_weights(
                logits,
                input_ids,
                power=nll_power,
                clip_percentile=nll_clip_percentile,
                mix_lambda=nll_mix_lambda,
            )
            Hm = _attention_entropy_factor_per_position(
                out_lm.attentions, layer_idx, (bs, seq), logits.device
            )
            Hm = Hm.to(device=alpha_q.device, dtype=torch.float32)
            alpha_kv = alpha_q * Hm
            alpha_kv = torch.nan_to_num(alpha_kv, nan=0.0, posinf=1e12, neginf=0.0)
            alpha_kv = torch.clamp(alpha_kv, min=0.0, max=1e12)
            if "q" in hook_out and "k" in hook_out and "v" in hook_out:
                rq, rk, rv = _maybe_apply_residual(hook_out["q"], hook_out["k"], hook_out["v"])
                acc.update_from_outputs(
                    rq,
                    rk,
                    rv,
                    alpha_bt=alpha_q,
                    alpha_kv_bt=alpha_kv,
                )
            return

        if weight_mode == "nll":
            out_lm = model(input_ids=input_ids, use_cache=False, return_dict=True)
            logits = out_lm.logits
            pq = _resolve_nll(nll_power_qk, nll_power)
            cq = _resolve_nll(nll_clip_percentile_qk, nll_clip_percentile)
            mq = _resolve_nll(nll_mix_lambda_qk, nll_mix_lambda)
            pv = _resolve_nll(nll_power_v, nll_power)
            cv = _resolve_nll(nll_clip_percentile_v, nll_clip_percentile)
            mv = _resolve_nll(nll_mix_lambda_v, nll_mix_lambda)
            alpha_qk = _per_position_nll_weights(
                logits,
                input_ids,
                power=pq,
                clip_percentile=cq,
                mix_lambda=mq,
            )
            same_qv = (pq == pv) and (cq == cv) and (mq == mv)
            if same_qv:
                alpha_bt = alpha_qk
                alpha_v_bt = None
            else:
                alpha_bt = alpha_qk
                alpha_v_bt = _per_position_nll_weights(
                    logits,
                    input_ids,
                    power=pv,
                    clip_percentile=cv,
                    mix_lambda=mv,
                )
        else:
            _ = model.model.decoder(input_ids)
        if "q" in hook_out and "k" in hook_out and "v" in hook_out:
            rq, rk, rv = _maybe_apply_residual(hook_out["q"], hook_out["k"], hook_out["v"])
            acc.update_from_outputs(
                rq,
                rk,
                rv,
                alpha_bt=alpha_bt,
                alpha_v_bt=alpha_v_bt,
            )

    try:
        for batch in batches:
            hook_out.clear()
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0]
            elif isinstance(batch, dict):
                input_ids = batch.get("input_ids")
            else:
                input_ids = batch
            if not torch.is_tensor(input_ids):
                continue
            input_ids = input_ids.to(input_device)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

            if weight_mode == "nll_fisher":
                with torch.enable_grad():
                    _one_batch(input_ids, grad_mode_nll_fisher=True)
            else:
                with torch.no_grad():
                    _one_batch(input_ids, grad_mode_nll_fisher=False)
    finally:
        for h in hooks:
            h.remove()
        model.config.use_cache = use_cache

    u_q, u_k, u_v = _build_u_from_covs(acc, rank_kq_per_head, rank_v_per_head, repair_keqk)

    score_stats: Dict[str, float] = {}
    if float(score_lambda) > 0.0 and int(score_refine_steps) > 0:
        u_q, u_k, extra = _refine_qk_bases_score_mse(
            model,
            layer_idx,
            batch_list=batches,
            u_q=u_q,
            u_k=u_k,
            num_heads=num_heads,
            head_dim=head_dim,
            rank_kq_per_head=rank_kq_per_head,
            repair_keqk=repair_keqk,
            input_device=input_device,
            residual_wrt_ref=residual_wrt_ref,
            ref_u_q=ref_u_q,
            ref_u_k=ref_u_k,
            steps=int(score_refine_steps),
            lr=float(score_refine_lr),
            score_lambda=float(score_lambda),
            score_prox_to_pca=float(score_prox_to_pca),
            sample_m=int(score_sample_m),
            sample_n=int(score_sample_n),
            max_batches=int(score_refine_batches),
            seed=int(score_refine_seed),
            center_logits=bool(score_center_logits),
        )
        score_stats = {k: float(v) for k, v in extra.items() if isinstance(v, (int, float))}

    nq = max(acc.alpha_sum_q, 1e-12)
    nk = max(acc.alpha_sum_k, 1e-12)
    nv = max(acc.alpha_sum_v, 1e-12)
    eig_q = [_sorted_eigvals_from_cov(acc.yty_q[h], nq).cpu() for h in range(num_heads)]
    eig_k = [_sorted_eigvals_from_cov(acc.yty_k[h], nk).cpu() for h in range(num_heads)]
    eig_v = [_sorted_eigvals_from_cov(acc.yty_v[h], nv).cpu() for h in range(num_heads)]
    spectrum_stats: Dict[str, object] = {
        "eig_q": eig_q,
        "eig_k": eig_k,
        "eig_v": eig_v,
    }
    if score_stats:
        spectrum_stats["score_refine"] = score_stats
    return u_q, u_k, u_v, acc.n_tokens, spectrum_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--ref_basis", type=str, required=True, help="参考 eigen_basis.pt（定 rank / 键名）")
    ap.add_argument("--out_basis", type=str, required=True)
    ap.add_argument("--compare_json", type=str, default="", help="写出新旧 basis 对比指标")
    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="txt/jsonl/json/gz 目录；统计分布≈语料。域对齐（方案 A）时换与下游相近的文本目录。",
    )
    ap.add_argument(
        "--token_budget",
        type=int,
        default=1_200_000,
        help="读入 tokenizer 流的目标 token 数（含 eos）；方案 A 建议 1.2e6～2e6+ 或 --preset large_stats",
    )
    ap.add_argument(
        "--max_data_files",
        type=int,
        default=200,
        help="data_dir 下最多扫描的文本文件数（防止 token_budget 很大但文本行数被截断）",
    )
    ap.add_argument(
        "--max_data_lines",
        type=int,
        default=2_000_000,
        help="从文本中最多读取的行/条数上限（jsonl 行等）；大 token_budget 时同步加大",
    )
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument(
        "--max_batches",
        type=int,
        default=128,
        help="每层前向的 batch 窗口数（× batch_size × seqlen 为重复统计量）；方案 A 建议 128～192 或 --preset large_stats",
    )
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=["none", "large_stats", "aggressive_nll_qkv"],
        help=(
            "large_stats: 只加大 token_budget/max_batches/max_data_*（不改 weight_mode / NLL）；"
            "aggressive_nll_qkv: 更多数据 + 激进 NLL+QK/V（会覆盖 nll/token/max_*）；"
            "完全手写用 none"
        ),
    )
    ap.add_argument("--layers", type=str, default="all", help='e.g. "all" or "46" or "32-47"')
    ap.add_argument(
        "--hetero_nll_layers",
        type=str,
        default="",
        help=(
            '层间异质：非空时，这些层用 --weight_mode 及全套 nll_*（如 "32-47"）；'
            "其余在 --layers 内的层用 --baseline_weight_mode（uniform 或弱 nll）。"
            "留空则全层共用 --weight_mode（与旧行为一致）。"
        ),
    )
    ap.add_argument(
        "--baseline_weight_mode",
        type=str,
        default="uniform",
        choices=["uniform", "nll"],
        help="hetero_nll_layers 非空时，非 hetero 层：uniform 或弱 nll（baseline_nll_*；可选 baseline_nll_*_qk/v 做 QK/V 分权）",
    )
    ap.add_argument("--baseline_nll_power", type=float, default=1.0, help="baseline nll 的 power")
    ap.add_argument(
        "--baseline_nll_clip_percentile",
        type=float,
        default=99.5,
        help="baseline nll 的 clip 百分位",
    )
    ap.add_argument(
        "--baseline_nll_mix_lambda",
        type=float,
        default=0.35,
        help="baseline nll 的 mix（通常低于主 NLL 段，偏稳）",
    )
    ap.add_argument(
        "--baseline_nll_mix_lambda_qk",
        type=float,
        default=None,
        help="hetero 弱层：Q/K 的 mix_lambda；默认 None，见 --hetero-baseline-nll-align-strong",
    )
    ap.add_argument(
        "--baseline_nll_mix_lambda_v",
        type=float,
        default=None,
        help="hetero 弱层：V 的 mix_lambda；默认 None，见 --hetero-baseline-nll-align-strong",
    )
    ap.add_argument("--baseline_nll_power_qk", type=float, default=None, help="hetero 弱层：Q/K 的 NLL pow")
    ap.add_argument("--baseline_nll_power_v", type=float, default=None, help="hetero 弱层：V 的 NLL pow")
    ap.add_argument(
        "--baseline_nll_clip_percentile_qk",
        type=float,
        default=None,
        help="hetero 弱层：Q/K 的 clip 百分位",
    )
    ap.add_argument(
        "--baseline_nll_clip_percentile_v",
        type=float,
        default=None,
        help="hetero 弱层：V 的 clip 百分位",
    )
    ap.add_argument(
        "--hetero-baseline-nll-align-strong",
        dest="hetero_baseline_nll_align_strong",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "hetero 且 baseline=nll、强分支 weight_mode=nll：未显式 baseline_nll_*_qk/v 时，"
            "弱层继承强分支 nll_* 的 qk/v 分权（标量仍用 baseline_nll_power/clip/mix）"
        ),
    )
    ap.add_argument(
        "--hetero_auto_metric",
        type=str,
        default="off",
        choices=["off", "attn_entropy", "q_grad_energy"],
        help=(
            "数据驱动选「强 NLL」层（在 --layers 内排序取 top_fraction）。"
            "off：用手动 --hetero_nll_layers 或全层同权。"
            "attn_entropy：每层平均 attention 熵（轻量，多 batch 前向）。"
            "q_grad_energy：每层 q_proj 上 ||∂L/∂q||² 均值（单 batch backward，重）。"
            "非 off 时忽略手动 --hetero_nll_layers。"
        ),
    )
    ap.add_argument(
        "--hetero_auto_probe_batches",
        type=int,
        default=8,
        help="attn_entropy 探测时最多用多少个 batch（与主流程 batch_list 前缀相同）",
    )
    ap.add_argument(
        "--hetero_auto_top_fraction",
        type=float,
        default=0.33,
        help="在 --layers 集合内，取分数最高(或最低，见 strong_side)的 ceil(|layers|*fraction) 层为强分支",
    )
    ap.add_argument(
        "--hetero_auto_strong_side",
        type=str,
        default="high",
        choices=["high", "low"],
        help="high：分数高者为强（如熵高更平）；low：分数低者为强（如更尖的注意力）",
    )
    ap.add_argument("--repair_keqk", type=int, default=1, choices=[0, 1], help="1: K basis 与 Q 相同（与现有 repair 一致）")
    ap.add_argument(
        "--head_rank_energy_th_kq",
        type=float,
        default=0.995,
        help="per-head 候选 rank：Q/K 谱累计保留能量阈值（0-1）",
    )
    ap.add_argument(
        "--head_rank_energy_th_v",
        type=float,
        default=0.995,
        help="per-head 候选 rank：V 谱累计保留能量阈值（0-1）",
    )
    ap.add_argument(
        "--layer_error_budget_kq",
        type=float,
        default=-1.0,
        help="层级误差预算（KQ）：若 >=0，则按每层聚合谱尾能量比 <= 预算求统一 per-head rank",
    )
    ap.add_argument(
        "--layer_error_budget_v",
        type=float,
        default=-1.0,
        help="层级误差预算（V）：若 >=0，则按每层聚合谱尾能量比 <= 预算求统一 per-head rank",
    )
    ap.add_argument(
        "--global_rank_budget_kq",
        type=int,
        default=-1,
        help="全局 rank 预算（KQ，单位=每层 per-head rank 之和）；<0 关闭",
    )
    ap.add_argument(
        "--global_rank_budget_v",
        type=int,
        default=-1,
        help="全局 rank 预算（V，单位=每层 per-head rank 之和）；<0 关闭",
    )
    ap.add_argument(
        "--cov_residual_to_ref",
        type=int,
        default=0,
        choices=[0, 1],
        help="1: 在累加加权协方差前，对每个 head 用 (I-P_ref)Y（Y 为 Q/K/V 激活，P_ref 为 ref_basis 列投影）；强调参考子空间外的方向",
    )
    ap.add_argument(
        "--weight_mode",
        type=str,
        default="uniform",
        choices=["uniform", "nll", "nll_fisher", "nll_attn_entropy", "nll_teacher_attn"],
        help=(
            "uniform | nll(A) | nll_fisher(A+B) | nll_attn_entropy(A+C) | "
            "nll_teacher_attn(A+D，配合 --teacher_model)"
        ),
    )
    ap.add_argument(
        "--teacher_model",
        type=str,
        default="",
        help="A+D：教师 OPT 权重路径；与 --model 不同则加载第二份模型并算 attention 差异（显存翻倍）。留空或与 --model 相同则 d≡1，等价纯 A。",
    )
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="含 tokenizer 的目录（与 OPT 词表一致即可，如完整 /diskE/opt-30b）。默认未填时优先用 --teacher_model，否则用 --model。",
    )
    ap.add_argument(
        "--nll_power",
        type=float,
        default=1.0,
        help="对 A 的逐位置 NLL 做 pow（nll / nll_fisher 均生效）",
    )
    ap.add_argument(
        "--nll_clip_percentile",
        type=float,
        default=100.0,
        help="A 权重百分位裁剪（0-100，100=不裁剪；推荐 99 或 99.5）",
    )
    ap.add_argument(
        "--nll_mix_lambda",
        type=float,
        default=1.0,
        help="A 与均匀权重混合系数 lambda（0-1，1=纯 A，0=纯均匀）；未单独设 qk/v 时 Q/K/V 共用",
    )
    ap.add_argument(
        "--nll_mix_lambda_qk",
        type=float,
        default=None,
        help="仅 weight_mode=nll：Q/K 分支的 mix_lambda；默认与 --nll_mix_lambda 相同",
    )
    ap.add_argument(
        "--nll_mix_lambda_v",
        type=float,
        default=None,
        help="仅 weight_mode=nll：V 分支的 mix_lambda；可与 qk 不同以拉大 QK vs V 统计",
    )
    ap.add_argument(
        "--nll_clip_percentile_qk",
        type=float,
        default=None,
        help="仅 nll：Q/K 的 NLL 百分位裁剪；默认与 --nll_clip_percentile 相同",
    )
    ap.add_argument(
        "--nll_clip_percentile_v",
        type=float,
        default=None,
        help="仅 nll：V 的 NLL 百分位裁剪",
    )
    ap.add_argument(
        "--nll_power_qk",
        type=float,
        default=None,
        help="仅 nll：Q/K 的 NLL pow；默认与 --nll_power 相同",
    )
    ap.add_argument(
        "--nll_power_v",
        type=float,
        default=None,
        help="仅 nll：V 的 NLL pow",
    )
    ap.add_argument(
        "--fisher_batch_norm",
        type=int,
        default=1,
        choices=[0, 1],
        help="nll_fisher 时对 γ=α·β 按 batch 做 mean 归一化（稳定尺度）",
    )
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--device_map", type=str, default="auto", help='transformers device_map, 或 "cuda:0" 单卡')
    ap.add_argument(
        "--score_lambda",
        type=float,
        default=0.0,
        help="score-aware：在 PCA 后对 Q/K 基做轻量精调，最小化采样子矩阵上 QK^T 与投影后 QK^T 的 MSE；0=关闭",
    )
    ap.add_argument("--score_refine_steps", type=int, default=24, help="score MSE 精调 Adam 步数")
    ap.add_argument("--score_refine_lr", type=float, default=0.08, help="score 精调学习率")
    ap.add_argument(
        "--score_prox_to_pca",
        type=float,
        default=0.0,
        help="score 损失外对 PCA 初值的 L2 均值惩罚系数（防偏离过大；默认 0）",
    )
    ap.add_argument("--score_sample_m", type=int, default=64, help="score 子矩阵采样的 query 行数")
    ap.add_argument("--score_sample_n", type=int, default=64, help="score 子矩阵采样的 key 列数")
    ap.add_argument(
        "--score_refine_batches",
        type=int,
        default=4,
        help="score 精调使用的 batch 数（取与主流程相同的 batch 列表前缀）",
    )
    ap.add_argument(
        "--score_refine_seed",
        type=int,
        default=0,
        help="score 子采样随机种子（与 --seed 独立；每层会叠加 layer_idx）",
    )
    ap.add_argument(
        "--score_center_logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="对采样子矩阵的 logits 做按行去均值（减弱全局平移；默认关）",
    )
    args = ap.parse_args()

    if args.preset == "large_stats":
        args.token_budget = 2_000_000
        args.max_batches = 192
        args.max_data_files = 300
        args.max_data_lines = 3_000_000
        print(
            "[preset large_stats] "
            f"token_budget={args.token_budget}, max_batches={args.max_batches}, "
            f"max_data_files={args.max_data_files}, max_data_lines={args.max_data_lines} "
            "(NLL/weight_mode 仍由命令行决定；域对齐请换 --data_dir)"
        )

    if args.preset == "aggressive_nll_qkv":
        args.weight_mode = "nll"
        args.token_budget = 1_200_000
        args.max_batches = 128
        args.max_data_files = 200
        args.max_data_lines = 2_000_000
        args.nll_power = 1.0
        args.nll_clip_percentile = 99.5
        args.nll_mix_lambda = 0.9
        args.nll_mix_lambda_qk = 1.0
        args.nll_mix_lambda_v = 0.0
        args.nll_clip_percentile_qk = 99.0
        args.nll_clip_percentile_v = 100.0
        args.nll_power_qk = 1.25
        args.nll_power_v = 0.88
        print(
            "[preset aggressive_nll_qkv] weight_mode=nll, "
            f"token_budget={args.token_budget}, max_batches={args.max_batches}, "
            f"lambda_qk={args.nll_mix_lambda_qk}, lambda_v={args.nll_mix_lambda_v}, "
            f"power_qk={args.nll_power_qk}, power_v={args.nll_power_v}"
        )

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ref_path = Path(args.ref_basis)
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    ref_state = torch.load(str(ref_path), map_location="cpu")
    if not isinstance(ref_state, dict):
        raise TypeError("ref basis must be a dict")

    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    n_layers = int(cfg.num_hidden_layers)
    num_heads = int(cfg.num_attention_heads)
    hidden = int(cfg.hidden_size)
    head_dim = hidden // num_heads

    Uq0 = ref_state[_basis_key(0, "q")]
    if Uq0.shape[0] != hidden:
        raise ValueError(f"ref basis hidden {Uq0.shape[0]} != config hidden {hidden}")
    rank_kq = int(Uq0.shape[1])
    rank_v = int(ref_state[_basis_key(0, "v")].shape[1])
    if rank_kq % num_heads or rank_v % num_heads:
        raise ValueError(f"rank_kq={rank_kq} or rank_v={rank_v} not divisible by num_heads={num_heads}")
    rank_kq_per_head = rank_kq // num_heads
    rank_v_per_head = rank_v // num_heads

    layer_ids = _parse_layers(args.layers, n_layers)

    tok_path = _resolve_tokenizer_pretrained_path(args)
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    texts = iter_texts_from_dir(
        args.data_dir, max_files=args.max_data_files, max_lines=args.max_data_lines
    )
    stream = build_token_stream(tok, texts, target_tokens=args.token_budget, seed=args.seed)
    if stream is None:
        raise RuntimeError("no text read from data_dir")
    random.seed(args.seed)
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

    teacher_model: Optional[torch.nn.Module] = None
    teacher_distinct = False
    tm = (args.teacher_model or "").strip()
    if args.weight_mode == "nll_teacher_attn" and tm and tm != args.model:
        if args.device_map != "auto":
            teacher_model = AutoModelForCausalLM.from_pretrained(
                tm,
                torch_dtype=dt,
                device_map={"": args.device_map},
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        else:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                tm,
                torch_dtype=dt,
                device_map="auto",
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        teacher_model.eval()
        teacher_distinct = True

    # 固定数据迭代器：每层复用同一批 batch 序列（可复现）
    def make_batches():
        g = torch.Generator()
        g.manual_seed(args.seed)
        N = stream.numel()
        for _ in range(args.max_batches):
            starts = torch.randint(low=0, high=N - args.seqlen - 1, size=(args.batch_size,), generator=g)
            x = torch.stack([stream[s : s + args.seqlen] for s in starts], dim=0)
            yield x

    out_state: Dict[str, torch.Tensor] = {}
    report: Dict[str, object] = {
        "preset": args.preset,
        "token_budget": int(args.token_budget),
        "max_data_files": int(args.max_data_files),
        "max_data_lines": int(args.max_data_lines),
        "model": args.model,
        "route_legend": {
            "A": "nll",
            "B": "nll_fisher (Fisher factor on top of A)",
            "C": "nll_attn_entropy",
            "D": "nll_teacher_attn (A+D; needs --teacher_model != --model for nontrivial d)",
        },
        "teacher_model": tm if args.weight_mode == "nll_teacher_attn" else "",
        "teacher_distinct": bool(teacher_distinct),
        "tokenizer_path": tok_path,
        "ref_basis": str(ref_path),
        "layers": layer_ids,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "seqlen": args.seqlen,
        "repair_keqk": args.repair_keqk,
        "cov_residual_to_ref": int(args.cov_residual_to_ref),
        "head_rank_energy_th_kq": float(args.head_rank_energy_th_kq),
        "head_rank_energy_th_v": float(args.head_rank_energy_th_v),
        "layer_error_budget_kq": float(args.layer_error_budget_kq),
        "layer_error_budget_v": float(args.layer_error_budget_v),
        "global_rank_budget_kq": int(args.global_rank_budget_kq),
        "global_rank_budget_v": int(args.global_rank_budget_v),
        "hetero_nll_layers": (args.hetero_nll_layers or "").strip() or None,
        "hetero_auto_metric": getattr(args, "hetero_auto_metric", "off"),
        "hetero_auto_top_fraction": float(getattr(args, "hetero_auto_top_fraction", 0.33)),
        "hetero_auto_strong_side": getattr(args, "hetero_auto_strong_side", "high"),
        "baseline_weight_mode": args.baseline_weight_mode,
        "baseline_nll_power": float(args.baseline_nll_power),
        "baseline_nll_clip_percentile": float(args.baseline_nll_clip_percentile),
        "baseline_nll_mix_lambda": float(args.baseline_nll_mix_lambda),
        "hetero_baseline_nll_align_strong": bool(getattr(args, "hetero_baseline_nll_align_strong", True)),
        "baseline_nll_mix_lambda_qk": args.baseline_nll_mix_lambda_qk,
        "baseline_nll_mix_lambda_v": args.baseline_nll_mix_lambda_v,
        "baseline_nll_power_qk": args.baseline_nll_power_qk,
        "baseline_nll_power_v": args.baseline_nll_power_v,
        "baseline_nll_clip_percentile_qk": args.baseline_nll_clip_percentile_qk,
        "baseline_nll_clip_percentile_v": args.baseline_nll_clip_percentile_v,
        "weight_mode": args.weight_mode,
        "nll_power": args.nll_power,
        "nll_clip_percentile": float(args.nll_clip_percentile),
        "nll_mix_lambda": float(args.nll_mix_lambda),
        "nll_mix_lambda_qk": args.nll_mix_lambda_qk,
        "nll_mix_lambda_v": args.nll_mix_lambda_v,
        "nll_clip_percentile_qk": args.nll_clip_percentile_qk,
        "nll_clip_percentile_v": args.nll_clip_percentile_v,
        "nll_power_qk": args.nll_power_qk,
        "nll_power_v": args.nll_power_v,
        "fisher_batch_norm": int(args.fisher_batch_norm),
        "score_lambda": float(args.score_lambda),
        "score_refine_steps": int(args.score_refine_steps),
        "score_refine_lr": float(args.score_refine_lr),
        "score_prox_to_pca": float(args.score_prox_to_pca),
        "score_sample_m": int(args.score_sample_m),
        "score_sample_n": int(args.score_sample_n),
        "score_refine_batches": int(args.score_refine_batches),
        "score_refine_seed": int(args.score_refine_seed),
        "score_center_logits": bool(args.score_center_logits),
        "per_layer": [],
    }

    batch_list = list(make_batches())
    if not batch_list:
        raise RuntimeError("no batches")

    if int(args.cov_residual_to_ref) == 1:
        print(
            "[cov_residual_to_ref] ON: per-head covariance uses (I-P_ref)Y; "
            "ref columns assumed orthonormal (standard PCA basis)."
        )

    hetero_set: Optional[set] = None
    hetero_auto_info: Optional[Dict[str, object]] = None
    manual_spec = (args.hetero_nll_layers or "").strip()

    if args.hetero_auto_metric != "off":
        if manual_spec:
            print("[hetero_auto] ignoring manual --hetero_nll_layers (data-driven selection takes precedence)")
        n_probe = min(len(batch_list), max(1, int(args.hetero_auto_probe_batches)))
        print(
            f"[hetero_auto] probing metric={args.hetero_auto_metric} batches={n_probe} "
            f"(top_fraction={args.hetero_auto_top_fraction}, side={args.hetero_auto_strong_side})..."
        )
        if args.hetero_auto_metric == "attn_entropy":
            scores = _probe_scores_attn_entropy(
                model, batch_list, n_layers, input_device, n_probe
            )
        else:
            print(
                "[hetero_auto] q_grad_energy uses 1 batch + full backward; peak VRAM/time may be high."
            )
            scores = _probe_scores_q_grad_energy(model, batch_list, n_layers, input_device)
            try:
                import torch as _t

                _t.cuda.empty_cache()
            except Exception:
                pass
        hetero_set = _select_hetero_auto_layers(
            layer_ids,
            scores,
            float(args.hetero_auto_top_fraction),
            str(args.hetero_auto_strong_side),
        )
        hetero_auto_info = {
            "metric": args.hetero_auto_metric,
            "scores_all_layers": scores,
            "selected_strong": sorted(hetero_set),
            "top_fraction": float(args.hetero_auto_top_fraction),
            "strong_side": args.hetero_auto_strong_side,
            "probe_batches": n_probe,
        }
        report["hetero_auto"] = hetero_auto_info
        report["hetero_nll_layers_resolved"] = sorted(hetero_set)
        print(
            f"[hetero_auto] strong layers ({len(hetero_set)}/{len(layer_ids)}): {sorted(hetero_set)}; "
            f"baseline={args.baseline_weight_mode!r}"
        )
        if args.weight_mode == "uniform":
            print(
                "[warn] --weight_mode uniform with hetero_auto: strong branch is still uniform; check intent."
            )
    elif manual_spec:
        hetero_set = _parse_hetero_nll_layers(manual_spec, n_layers)
        bad = hetero_set - set(layer_ids)
        if bad:
            raise ValueError(
                f"--hetero_nll_layers contains layers not included in --layers {args.layers!r}: {sorted(bad)}"
            )
        if not hetero_set:
            raise ValueError("--hetero_nll_layers parsed to empty set")
        report["hetero_nll_layers_resolved"] = sorted(hetero_set)
        print(
            f"[hetero] strong branch layers (use --weight_mode={args.weight_mode!r}): "
            f"{sorted(hetero_set)}; other layers in --layers use baseline={args.baseline_weight_mode!r}"
        )
        if args.weight_mode == "uniform":
            print(
                "[warn] --weight_mode uniform with --hetero_nll_layers: strong branch is still uniform; check intent."
            )

    if (
        hetero_set is not None
        and args.baseline_weight_mode == "nll"
        and args.weight_mode == "nll"
    ):
        print(
            "[hetero] weak-layer NLL: hetero_baseline_nll_align_strong="
            f"{bool(getattr(args, 'hetero_baseline_nll_align_strong', True))} "
            "(on: inherit strong qk/v effective nll_* unless baseline_nll_*_qk/v set)"
        )

    rank_alloc_layers: List[int] = []
    rank_alloc_q_candidates: List[List[int]] = []
    rank_alloc_k_candidates: List[List[int]] = []
    rank_alloc_v_candidates: List[List[int]] = []
    rank_alloc_kq_layer_unified: List[int] = []
    rank_alloc_v_layer_unified: List[int] = []
    rank_alloc_kq_eigs_layer: List[torch.Tensor] = []
    rank_alloc_v_eigs_layer: List[torch.Tensor] = []

    for li in layer_ids:
        if hetero_set is None or li in hetero_set:
            wm = args.weight_mode
            nll_power = float(args.nll_power)
            nll_clip_percentile = float(args.nll_clip_percentile)
            nll_mix_lambda = float(args.nll_mix_lambda)
            np_qk, np_v = args.nll_power_qk, args.nll_power_v
            ncp_qk, ncp_v = args.nll_clip_percentile_qk, args.nll_clip_percentile_v
            nml_qk, nml_v = args.nll_mix_lambda_qk, args.nll_mix_lambda_v
            tm = teacher_model if wm == "nll_teacher_attn" else None
            td = teacher_distinct if wm == "nll_teacher_attn" else False
        else:
            wm = args.baseline_weight_mode
            if wm == "uniform":
                wm = "uniform"
                nll_power = 1.0
                nll_clip_percentile = 100.0
                nll_mix_lambda = 1.0
                np_qk = np_v = ncp_qk = ncp_v = nml_qk = nml_v = None
            else:
                wm = "nll"
                nll_power = float(args.baseline_nll_power)
                nll_clip_percentile = float(args.baseline_nll_clip_percentile)
                nll_mix_lambda = float(args.baseline_nll_mix_lambda)
                align = bool(
                    getattr(args, "hetero_baseline_nll_align_strong", True)
                    and args.weight_mode == "nll"
                )
                # 与强分支内 _resolve_nll(optional, fb) 一致后再继承，避免强侧用 nll_mix_lambda 回退、弱侧误用 baseline 标量
                sm_qk = args.nll_mix_lambda_qk if args.nll_mix_lambda_qk is not None else args.nll_mix_lambda
                sm_v = args.nll_mix_lambda_v if args.nll_mix_lambda_v is not None else args.nll_mix_lambda
                sp_qk = args.nll_power_qk if args.nll_power_qk is not None else args.nll_power
                sp_v = args.nll_power_v if args.nll_power_v is not None else args.nll_power
                sc_qk = (
                    args.nll_clip_percentile_qk
                    if args.nll_clip_percentile_qk is not None
                    else args.nll_clip_percentile
                )
                sc_v = (
                    args.nll_clip_percentile_v
                    if args.nll_clip_percentile_v is not None
                    else args.nll_clip_percentile
                )

                def _bl_qk_v(bl_opt: Optional[float], strong_eff: float) -> Optional[float]:
                    if bl_opt is not None:
                        return bl_opt
                    return float(strong_eff) if align else None

                np_qk = _bl_qk_v(args.baseline_nll_power_qk, float(sp_qk))
                np_v = _bl_qk_v(args.baseline_nll_power_v, float(sp_v))
                ncp_qk = _bl_qk_v(args.baseline_nll_clip_percentile_qk, float(sc_qk))
                ncp_v = _bl_qk_v(args.baseline_nll_clip_percentile_v, float(sc_v))
                nml_qk = _bl_qk_v(args.baseline_nll_mix_lambda_qk, float(sm_qk))
                nml_v = _bl_qk_v(args.baseline_nll_mix_lambda_v, float(sm_v))
            tm = None
            td = False

        cr = bool(int(args.cov_residual_to_ref))
        r_q = ref_state[_basis_key(li, "q")].float()
        r_k = ref_state[_basis_key(li, "k")].float()
        r_v = ref_state[_basis_key(li, "v")].float()
        rank_kq_per_head_li = int(r_q.shape[1]) // num_heads
        rank_v_per_head_li = int(r_v.shape[1]) // num_heads
        if int(r_q.shape[1]) % num_heads or int(r_v.shape[1]) % num_heads:
            raise ValueError(
                f"layer {li} ref rank not divisible by num_heads={num_heads}: "
                f"q={int(r_q.shape[1])}, v={int(r_v.shape[1])}"
            )
        if int(r_k.shape[1]) != int(r_q.shape[1]):
            raise ValueError(
                f"layer {li} q/k ref rank mismatch: q={int(r_q.shape[1])}, k={int(r_k.shape[1])}"
            )
        u_q, u_k, u_v, n_tok, sp = collect_activation_basis_one_layer(
            model,
            layer_idx=li,
            num_heads=num_heads,
            head_dim=head_dim,
            rank_kq_per_head=rank_kq_per_head_li,
            rank_v_per_head=rank_v_per_head_li,
            repair_keqk=args.repair_keqk,
            data_iter=batch_list,
            input_device=input_device,
            weight_mode=wm,
            nll_power=nll_power,
            nll_clip_percentile=nll_clip_percentile,
            nll_mix_lambda=nll_mix_lambda,
            nll_power_qk=np_qk,
            nll_power_v=np_v,
            nll_clip_percentile_qk=ncp_qk,
            nll_clip_percentile_v=ncp_v,
            nll_mix_lambda_qk=nml_qk,
            nll_mix_lambda_v=nml_v,
            fisher_batch_norm=int(args.fisher_batch_norm),
            teacher_model=tm,
            teacher_distinct=td,
            residual_wrt_ref=cr,
            ref_u_q=r_q if cr else None,
            ref_u_k=r_k if cr else None,
            ref_u_v=r_v if cr else None,
            score_lambda=float(args.score_lambda),
            score_refine_steps=int(args.score_refine_steps),
            score_refine_lr=float(args.score_refine_lr),
            score_prox_to_pca=float(args.score_prox_to_pca),
            score_sample_m=int(args.score_sample_m),
            score_sample_n=int(args.score_sample_n),
            score_refine_batches=int(args.score_refine_batches),
            score_refine_seed=int(args.score_refine_seed),
            score_center_logits=bool(args.score_center_logits),
        )
        eig_q = sp["eig_q"]
        eig_k = sp["eig_k"]
        eig_v = sp["eig_v"]
        cand_q = [_rank_from_energy_threshold(ev, float(args.head_rank_energy_th_kq)) for ev in eig_q]
        cand_k = [_rank_from_energy_threshold(ev, float(args.head_rank_energy_th_kq)) for ev in eig_k]
        cand_v = [_rank_from_energy_threshold(ev, float(args.head_rank_energy_th_v)) for ev in eig_v]
        rkq_layer = max(cand_q if int(args.repair_keqk) == 1 else [max(a, b) for a, b in zip(cand_q, cand_k)])
        rv_layer = max(cand_v)

        # Layer-level budget: convert per-head spectra to a layer-unified rank using tail-energy proxy.
        if float(args.layer_error_budget_kq) >= 0.0:
            ev_mix_kq = torch.stack(
                [eig_q[h] if int(args.repair_keqk) == 1 else 0.5 * (eig_q[h] + eig_k[h]) for h in range(num_heads)],
                dim=0,
            ).mean(dim=0)
            rkq_layer = _rank_from_tail_budget(ev_mix_kq, float(args.layer_error_budget_kq))
        if float(args.layer_error_budget_v) >= 0.0:
            ev_mix_v = torch.stack(eig_v, dim=0).mean(dim=0)
            rv_layer = _rank_from_tail_budget(ev_mix_v, float(args.layer_error_budget_v))

        rank_alloc_layers.append(int(li))
        rank_alloc_q_candidates.append(cand_q)
        rank_alloc_k_candidates.append(cand_k)
        rank_alloc_v_candidates.append(cand_v)
        rank_alloc_kq_layer_unified.append(int(rkq_layer))
        rank_alloc_v_layer_unified.append(int(rv_layer))
        rank_alloc_kq_eigs_layer.append(
            torch.stack(
                [eig_q[h] if int(args.repair_keqk) == 1 else 0.5 * (eig_q[h] + eig_k[h]) for h in range(num_heads)],
                dim=0,
            ).mean(dim=0)
        )
        rank_alloc_v_eigs_layer.append(torch.stack(eig_v, dim=0).mean(dim=0))

        out_state[_basis_key(li, "q")] = u_q
        out_state[_basis_key(li, "k")] = u_k
        out_state[_basis_key(li, "v")] = u_v

        rec: Dict[str, object] = {
            "layer": li,
            "n_tokens_seen": n_tok,
            "weight_mode_used": wm,
            "score_refine": sp.get("score_refine"),
            "head_rank_candidate_q": cand_q,
            "head_rank_candidate_k": cand_k,
            "head_rank_candidate_v": cand_v,
            "layer_rank_candidate_kq_unified": int(rkq_layer),
            "layer_rank_candidate_v_unified": int(rv_layer),
        }
        for br, Un in [("q", u_q), ("k", u_k), ("v", u_v)]:
            key = _basis_key(li, br)
            if key in ref_state:
                st = principal_angle_stats(ref_state[key].float(), Un)
                rec[f"vs_ref_{br}"] = st
        report["per_layer"].append(rec)
        print(f"[layer {li}] tokens={n_tok} q_mean_angle_deg={rec.get('vs_ref_q', {}).get('mean_deg', 'n/a')}")

    alloc_kq_final = list(rank_alloc_kq_layer_unified)
    alloc_v_final = list(rank_alloc_v_layer_unified)
    if int(args.global_rank_budget_kq) >= 0:
        alloc_kq_final = _allocate_global_budget_greedy(
            rank_alloc_kq_eigs_layer,
            alloc_kq_final,
            int(args.global_rank_budget_kq),
        )
    if int(args.global_rank_budget_v) >= 0:
        alloc_v_final = _allocate_global_budget_greedy(
            rank_alloc_v_eigs_layer,
            alloc_v_final,
            int(args.global_rank_budget_v),
        )
    report["rank_allocation"] = {
        "layers": rank_alloc_layers,
        "kq_per_head_candidate_q": rank_alloc_q_candidates,
        "kq_per_head_candidate_k": rank_alloc_k_candidates,
        "v_per_head_candidate_v": rank_alloc_v_candidates,
        "kq_layer_unified_before_global": rank_alloc_kq_layer_unified,
        "v_layer_unified_before_global": rank_alloc_v_layer_unified,
        "kq_layer_unified_after_global": alloc_kq_final,
        "v_layer_unified_after_global": alloc_v_final,
        "kq_sum_before_global": int(sum(rank_alloc_kq_layer_unified)),
        "v_sum_before_global": int(sum(rank_alloc_v_layer_unified)),
        "kq_sum_after_global": int(sum(alloc_kq_final)),
        "v_sum_after_global": int(sum(alloc_v_final)),
    }
    if rank_alloc_layers:
        print(
            "[rank_alloc] KQ sum "
            f"{sum(rank_alloc_kq_layer_unified)} -> {sum(alloc_kq_final)}, "
            f"V sum {sum(rank_alloc_v_layer_unified)} -> {sum(alloc_v_final)}"
        )

    # 未计算的层：从参考拷贝（保持完整 checkpoint 可加载）
    for li in range(n_layers):
        for br in ("q", "k", "v"):
            key = _basis_key(li, br)
            if key not in out_state:
                out_state[key] = ref_state[key].float().clone()

    out_path = Path(args.out_basis)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, out_path)
    print(f"[saved] {out_path} (activation layers={layer_ids})")

    if args.compare_json:
        cmp_path = Path(args.compare_json)
        cmp_path.parent.mkdir(parents=True, exist_ok=True)
        # 汇总标量
        summary = {"activation_layers": layer_ids}
        for br in ("q", "k", "v"):
            angles = []
            for pl in report["per_layer"]:
                d = pl.get(f"vs_ref_{br}")
                if isinstance(d, dict) and "mean_deg" in d and not math.isnan(d["mean_deg"]):
                    angles.append(d["mean_deg"])
            if angles:
                summary[f"{br}_mean_angle_deg_avg_over_layers"] = float(sum(angles) / len(angles))
        report["summary"] = summary
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[compare] {cmp_path}")


if __name__ == "__main__":
    main()
