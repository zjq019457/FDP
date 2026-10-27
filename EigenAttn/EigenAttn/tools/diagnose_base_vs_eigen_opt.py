#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统化对比 BASE(全秩 OPT) vs CMP(EigenAttn 低秩)：同一 batch、同一 seed、同一 seqlen。

指标（每层 + 全局 logits）：
  - Q/K/V：各侧 RMS norm（基座为 q_proj/k_proj/v_proj 输出；CMP 为 q_proj_low/k_proj_low/v_proj_low 输出，维度不同不可直接比数值，用于查「单层突跳/尺度漂移」）
  - attention 熵：对 softmax 注意力按 key 维平均熵（与 build_activation 思路一致）
  - logits：std、p99、mean |logits|
  - hidden：decoder 每层输出与基座的 cosine、relL2（需 output_hidden_states）
  - [1] lm_head.weight base vs cmp（cosine / max|Δ| / mean|Δ|）；lm_head vs embed_tokens tie（指针是否同一 Tensor + 数值）
  - [2] cmp 最后一层 hidden 过 base.lm_head，与 base logits 最后一行 cosine（定位 head vs decoder）
  - [3] 最后一 token 位置 top-10 token id / logit
  - [4] 首、末 + 随机若干行的 logit row cosine（避免只看均值）

用法（在 EigenAttn 目录下 PYTHONPATH 含当前目录）:
  PYTHONPATH=$(pwd):$PYTHONPATH python tools/diagnose_base_vs_eigen_opt.py \\
    --base /diskE/opt-30b --cmp /path/to/eigen --low_json ./low_rank_config_layerwise_xxx.json \\
    --tok /diskE/opt-30b --seqlen 512 --seed 0 \\
    --base_device cuda:0 --cmp_device cuda:1 \\
    --out_json qkv_outputs/diagnose_base_vs_eigen.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rms(t: torch.Tensor) -> float:
    x = t.detach().float()
    return float(x.pow(2).mean().sqrt().item())


# torch.quantile 对展平后元素个数有上限（大词表 logits 会触发 RuntimeError: too large）
_MAX_QUANTILE_ELEMS = 8_000_000


def _quantile_abs(x: torch.Tensor, q: float) -> float:
    v = x.detach().float().abs().flatten().cpu()
    n = v.numel()
    if n > _MAX_QUANTILE_ELEMS:
        idx = torch.randint(0, n, (_MAX_QUANTILE_ELEMS,), device=v.device)
        v = v[idx]
    return float(torch.quantile(v, q).item())


def _logits_stats(logits: torch.Tensor) -> Dict[str, float]:
    x = logits.detach().float()
    xf = x.flatten()
    return {
        "mean": float(xf.mean().item()),
        "std": float(xf.std().item()),
        "p99_abs": _quantile_abs(x, 0.99),
        "p999_abs": _quantile_abs(x, 0.999),
        "quantile_subsample": int(xf.numel() > _MAX_QUANTILE_ELEMS),
    }


def _align_logits_btv(
    logits: torch.Tensor, vocab_size: int, tag: str
) -> Tuple[torch.Tensor, Optional[str]]:
    """
    因果 LM 标准形状 [B, T, V] 且 V 为最后一维。
    若误为 [B, V, T]（中间维为 vocab），则 transpose 回 [B, T, V]。
    """
    if logits.dim() != 3:
        return logits, f"{tag}: expected 3D logits, got shape={tuple(logits.shape)}"
    b, x, y = logits.shape
    if y == vocab_size:
        return logits, None
    if x == vocab_size and y != vocab_size:
        return logits.transpose(1, 2).contiguous(), f"{tag}: fixed [B,V,T]->[B,T,V] (was {tuple(logits.shape)})"
    return logits, f"{tag}: warn last_dim={y} mid_dim={x} vocab_size={vocab_size}"


def _rowwise_cosine_btv(logits_b: torch.Tensor, logits_c: torch.Tensor) -> torch.Tensor:
    """
    对 [N, V] 逐行余弦；与 F.cosine_similarity(..., dim=-1) 一致。
    显式公式：sum(a*b, dim=-1) / (||a||*||b||)。
    """
    a = logits_b.float().contiguous()
    b = logits_c.float().contiguous()
    if a.device != b.device:
        a = a.to(b.device, non_blocking=True)
    eps = 1e-12
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1).clamp(min=eps) * b.norm(dim=-1).clamp(min=eps)
    return num / den


def _tensor_quantiles_1d(t: torch.Tensor, qs: List[float]) -> Dict[str, float]:
    """t 1D float on CPU；大 tensor 时子采样。"""
    x = t.detach().float().flatten().cpu()
    n = x.numel()
    if n > _MAX_QUANTILE_ELEMS:
        idx = torch.randint(0, n, (_MAX_QUANTILE_ELEMS,), device=x.device)
        x = x[idx]
    out: Dict[str, float] = {}
    for q in qs:
        out[f"p{int(q * 100):02d}"] = float(torch.quantile(x, q).item())
    return out


def _logits_diff_stats(
    logits_b: torch.Tensor,
    logits_c: torch.Tensor,
    vocab_size: int,
) -> Dict[str, Any]:
    """
    同 (b,t,v) 下 d = base - cmp；row cosine 在 [B*T, V] 上逐行计算（与单位置 _row_cosine_1d 一致）。
    另附 global_flat_cosine：把整个 logits 拉成一维再算 cosine（高维下≈0，仅作误用对照）。
    """
    lb, msg_b = _align_logits_btv(logits_b, vocab_size, "base")
    lc, msg_c = _align_logits_btv(logits_c, vocab_size, "cmp")

    out: Dict[str, Any] = {
        "logits_shape_base": list(logits_b.shape),
        "logits_shape_cmp": list(logits_c.shape),
        "aligned_shape_base": list(lb.shape),
        "aligned_shape_cmp": list(lc.shape),
        "align_note_base": msg_b,
        "align_note_cmp": msg_c,
    }

    if lb.shape != lc.shape:
        out["error"] = "shape mismatch after align; skip row cosine / diff tensor stats"
        return out

    device = lc.device
    lb = lb.to(device).float().contiguous()
    lc = lc.float().contiguous()

    d = (lb - lc).flatten()
    da = d.abs()
    out.update(
        {
            "diff_mean_signed": float(d.mean().item()),
            "diff_std_signed": float(d.std().item()),
            "diff_mean_abs": float(da.mean().item()),
            "diff_std_abs": float(da.std().item()),
            "diff_p99_abs": _quantile_abs(d, 0.99),
            "diff_p999_abs": _quantile_abs(d, 0.999),
            "diff_abs_quantiles": _tensor_quantiles_1d(da, [0.5, 0.9, 0.99]),
            "quantile_subsample": int(d.numel() > _MAX_QUANTILE_ELEMS),
        }
    )

    # --- 逐行 [B*T, V]：必须在同一 vocab 维上 reshape ---
    n_pos = lb.size(0) * lb.size(1)
    a = lb.view(n_pos, lb.size(-1))
    b = lc.view(n_pos, lc.size(-1))

    row_cos_manual = _rowwise_cosine_btv(a, b)
    row_cos_f = F.cosine_similarity(a, b, dim=-1, eps=1e-12)
    max_abs_err = float((row_cos_manual - row_cos_f).abs().max().item())
    out["row_cosine_manual_vs_F_cosine_max_abs_err"] = max_abs_err

    rc_cpu = row_cos_manual.detach().float().cpu()
    out["logit_row_cosine_mean"] = float(rc_cpu.mean().item())
    out["logit_row_cosine_std"] = float(rc_cpu.std().item())
    out["logit_row_cosine_min"] = float(rc_cpu.min().item())
    out["logit_row_cosine_max"] = float(rc_cpu.max().item())
    out["logit_row_cosine_quantiles"] = _tensor_quantiles_1d(rc_cpu, [0.01, 0.05, 0.5, 0.95, 0.99])

    # 误导性对照：整向量拉平后 cosine ~0（勿与 row mean 混淆）
    fa = a.flatten()
    fb = b.flatten()
    gcos = float((fa * fb).sum() / (fa.norm() * fb.norm() + 1e-12))
    out["global_flat_cosine_misleading_do_not_use"] = gcos
    out["note"] = (
        "logit_row_cosine_* 为每个 (b,t) 位置一行 vocab 向量与另一模型同行的 cosine；"
        "global_flat_cosine_* 为误把 [B*T*V] 当一根向量，与 row mean 不可比。"
    )

    return out


def _matrix_compare(w1: torch.Tensor, w2: torch.Tensor) -> Dict[str, Any]:
    """cosine / max|a-b| / mean|a-b|，CPU float。"""
    if w1.shape != w2.shape:
        return {
            "shape_a": list(w1.shape),
            "shape_b": list(w2.shape),
            "match": False,
        }
    a = w1.detach().float().cpu().flatten()
    b = w2.detach().float().cpu().flatten()
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    d = (a - b).abs()
    return {
        "match": True,
        "shape": list(w1.shape),
        "cosine": cos,
        "max_abs_diff": float(d.max().item()),
        "mean_abs_diff": float(d.mean().item()),
    }


def _tie_lm_embed_report(model: torch.nn.Module) -> Dict[str, Any]:
    """OPT: lm_head.weight vs model.model.decoder.embed_tokens.weight（tie 是否同一 Tensor / 数值一致）。"""
    if not hasattr(model, "lm_head") or not hasattr(model, "model"):
        return {"error": "expected lm_head + model"}
    dec = getattr(model.model, "decoder", None)
    if dec is None or not hasattr(dec, "embed_tokens"):
        return {"error": "no model.model.decoder.embed_tokens"}
    lm_w = model.lm_head.weight
    emb_w = dec.embed_tokens.weight
    rep = dict(_matrix_compare(lm_w, emb_w))
    rep["weight_ptr_same"] = lm_w is emb_w
    return rep


def _linear_logits_from_hidden(hidden: torch.Tensor, lm_head: torch.nn.Module) -> torch.Tensor:
    w = lm_head.weight
    b = lm_head.bias
    h = hidden.to(device=w.device, dtype=w.dtype)
    return F.linear(h, w, b)


def _row_cosine_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.device != b.device:
        b = b.to(a.device)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def _topk_logits_row(logits_row: torch.Tensor, tokenizer, k: int) -> List[Dict[str, Any]]:
    vals, idx = torch.topk(logits_row.float(), k)
    out: List[Dict[str, Any]] = []
    for r in range(k):
        tid = int(idx[r].item())
        tok_str = tokenizer.convert_ids_to_tokens([tid])[0]
        out.append(
            {
                "rank": r + 1,
                "id": tid,
                "logit": float(vals[r].item()),
                "token": tok_str,
            }
        )
    return out


def _pick_row_cosine_positions(tlen: int, seed: int, n_random: int) -> List[int]:
    """first、last + 若干随机位置（可重复 seed）。"""
    pos: List[int] = [0]
    if tlen > 1:
        pos.append(tlen - 1)
    rng = random.Random(seed + 7)
    pool = [p for p in range(tlen) if p not in (0, tlen - 1)]
    n_extra = min(max(0, n_random), len(pool))
    if n_extra > 0:
        rng.shuffle(pool)
        pos.extend(pool[:n_extra])
    return sorted(set(pos))


def _mean_attn_entropy(attn: torch.Tensor) -> float:
    """attn: [B,H,Tq,Tk] softmax 概率。"""
    if not torch.is_tensor(attn):
        return float("nan")
    a = attn
    if a.dim() == 5:
        a = a.squeeze(2)
    if a.dim() != 4:
        return float("nan")
    p = a.float().clamp(min=1e-12)
    p = p / p.sum(dim=-1, keepdim=True)
    h = -(p * torch.log(p)).sum(dim=-1)
    return float(h.mean().item())


def _hidden_cosine_rel_l2(
    hb: torch.Tensor, hc: torch.Tensor
) -> Tuple[float, float]:
    """hb, hc: [B,T,H] 同形状。"""
    hb = hb.detach().float()
    hc = hc.detach().float()
    a = hb.reshape(-1, hb.size(-1))
    b = hc.reshape(-1, hc.size(-1))
    cos = F.cosine_similarity(a, b, dim=-1).mean().item()
    num = (a - b).pow(2).sum(-1).sqrt()
    den = a.norm(dim=-1).clamp(min=1e-8)
    rel = (num / den).mean().item()
    return float(cos), float(rel)


def _embedding_high(model: torch.nn.Module) -> int:
    emb = model.get_input_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        return 50272
    return max(1, int(emb.weight.shape[0]))


def _max_seqlen(model: torch.nn.Module) -> int:
    c = getattr(model, "config", None)
    m = getattr(c, "max_position_embeddings", None) if c is not None else None
    return int(m) if m is not None else 10**9


def _load_base(path: str, dtype: torch.dtype, device: str):
    return AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )


def _load_eigen(path: str, low_json: str, dtype: torch.dtype, device: str):
    cfg = json.load(open(low_json))
    return OPTForCausalLM_EigenAttn.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device},
        low_rank_config=cfg,
    )


def _attach_qkv_norm_hooks(
    layers: torch.nn.ModuleList,
    tag: str,
    storage: Dict[int, Dict[str, float]],
) -> List[Any]:
    hooks: List[Any] = []

    def _make(li: int, branch: str, proj_name: str):
        def _h(_m, _inp, out):
            if not torch.is_tensor(out):
                return
            storage.setdefault(li, {})[f"{tag}_{branch}_rms"] = _rms(out)

        return _h

    for li, layer in enumerate(layers):
        sa = layer.self_attn
        if tag == "base":
            if hasattr(sa, "q_proj"):
                hooks.append(sa.q_proj.register_forward_hook(_make(li, "q", "q_proj")))
            if hasattr(sa, "k_proj"):
                hooks.append(sa.k_proj.register_forward_hook(_make(li, "k", "k_proj")))
            if hasattr(sa, "v_proj"):
                hooks.append(sa.v_proj.register_forward_hook(_make(li, "v", "v_proj")))
        else:
            if hasattr(sa, "q_proj_low"):
                hooks.append(sa.q_proj_low.register_forward_hook(_make(li, "q", "q_proj_low")))
            if hasattr(sa, "k_proj_low"):
                hooks.append(sa.k_proj_low.register_forward_hook(_make(li, "k", "k_proj_low")))
            if hasattr(sa, "v_proj_low"):
                hooks.append(sa.v_proj_low.register_forward_hook(_make(li, "v", "v_proj_low")))

    return hooks


@torch.inference_mode()
def _forward_collect(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    tag: str,
    layers: torch.nn.ModuleList,
    qkv_store: Dict[int, Dict[str, float]],
) -> Tuple[torch.Tensor, Tuple, Tuple]:
    hooks = _attach_qkv_norm_hooks(layers, tag, qkv_store)
    try:
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = False
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits
        attns = out.attentions
        hids = out.hidden_states
    finally:
        for h in hooks:
            h.remove()
    return logits, attns, hids


def _build_batch_from_c4(
    data_dir: str,
    tok,
    seqlen: int,
    seed: int,
    c4_glob: str,
    scan_lines: int,
    pool_docs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import glob
    import gzip

    def _iter_jsonl_texts_gz(fp: str):
        with gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    if isinstance(j, dict) and isinstance(j.get("text"), str):
                        t = j["text"].strip()
                        if t:
                            yield t
                except Exception:
                    continue

    files = sorted(glob.glob(os.path.join(data_dir, c4_glob)))
    if not files:
        raise FileNotFoundError(f"No files matched {c4_glob} under {data_dir}")
    reservoir: List[str] = []
    rng = random.Random(seed)
    seen = 0
    for fp in files:
        for t in _iter_jsonl_texts_gz(fp):
            seen += 1
            if len(reservoir) < pool_docs:
                reservoir.append(t)
            else:
                j = rng.randint(1, seen)
                if j <= pool_docs:
                    reservoir[j - 1] = t
            if seen >= scan_lines:
                break
        if seen >= scan_lines:
            break
    if not reservoir:
        raise RuntimeError("C4 pool empty")
    text = reservoir[rng.randint(0, len(reservoir) - 1)]
    tmp = tok(text, return_tensors="pt")
    L = tmp.input_ids.shape[1]
    if L <= seqlen:
        raise RuntimeError(f"sampled doc too short L={L} need > seqlen={seqlen}")
    i0 = rng.randint(0, L - seqlen - 1)
    input_ids = tmp.input_ids[:, i0 : i0 + seqlen]
    attn = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attn


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--cmp", type=str, required=True)
    ap.add_argument("--low_json", type=str, required=True)
    ap.add_argument("--tok", type=str, default="")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--base_device", type=str, default="cuda:0")
    ap.add_argument("--cmp_device", type=str, default="cuda:1")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="若提供则从 C4 风格 json.gz 采一条长文档窗口（与 sanity_AB 一致）；否则用随机 input_ids",
    )
    ap.add_argument("--c4_glob", type=str, default="c4-train.*.json.gz")
    ap.add_argument("--c4_scan_lines", type=int, default=200000)
    ap.add_argument("--c4_pool_docs", type=int, default=20000)
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument(
        "--n_random_positions",
        type=int,
        default=4,
        help="除首位置、末位置外，再随机抽几行算 base vs cmp 的 logit row cosine",
    )
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    tok_path = args.tok.strip() or args.base
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    set_seed(args.seed)

    print("[diag] loading BASE ...", flush=True)
    base = _load_base(args.base, dtype, args.base_device)
    print("[diag] loading CMP ...", flush=True)
    cmp_model = _load_eigen(args.cmp, args.low_json, dtype, args.cmp_device)

    n_layers = int(base.config.num_hidden_layers)
    if int(cmp_model.config.num_hidden_layers) != n_layers:
        print("[warn] num_hidden_layers mismatch base vs cmp")

    if args.data_dir:
        input_ids, attn = _build_batch_from_c4(
            args.data_dir,
            tok,
            args.seqlen,
            args.seed,
            args.c4_glob,
            args.c4_scan_lines,
            args.c4_pool_docs,
        )
    else:
        high = min(_embedding_high(base), _embedding_high(cmp_model))
        t = min(args.seqlen, _max_seqlen(base), _max_seqlen(cmp_model))
        g = torch.Generator(device=args.base_device)
        g.manual_seed(args.seed)
        input_ids = torch.randint(
            0, high, (args.batch_size, t), dtype=torch.long, device=args.base_device, generator=g
        )
        attn = torch.ones_like(input_ids, dtype=torch.long)

    bsz, tlen = input_ids.shape
    input_ids_b = input_ids.to(args.base_device)
    attn_b = attn.to(args.base_device)
    input_ids_c = input_ids.to(args.cmp_device)
    attn_c = attn.to(args.cmp_device)

    base_layers = base.model.decoder.layers
    cmp_layers = cmp_model.model.decoder.layers

    qkv_b: Dict[int, Dict[str, float]] = {}
    qkv_c: Dict[int, Dict[str, float]] = {}

    print(f"[diag] forward BASE B={bsz} T={tlen} ...", flush=True)
    logits_b, attns_b, hids_b = _forward_collect(
        base, input_ids_b, attn_b, tag="base", layers=base_layers, qkv_store=qkv_b
    )

    print(f"[diag] forward CMP B={bsz} T={tlen} ...", flush=True)
    logits_c, attns_c, hids_c = _forward_collect(
        cmp_model, input_ids_c, attn_c, tag="cmp", layers=cmp_layers, qkv_store=qkv_c
    )

    vocab_size = int(getattr(base.config, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        vocab_size = int(logits_b.shape[-1])
    logits_b, msg_lb = _align_logits_btv(logits_b, vocab_size, "base")
    logits_c, msg_lc = _align_logits_btv(logits_c, vocab_size, "cmp")
    if msg_lb or msg_lc:
        print("[diag] logits B,T,V align:", msg_lb, msg_lc, flush=True)

    lb = _logits_stats(logits_b)
    lc = _logits_stats(logits_c)
    logits_diff = _logits_diff_stats(logits_b, logits_c, vocab_size)

    # ---- 检查 1：lm_head 权重（base vs cmp）+ tie（lm_head vs embed_tokens）----
    lm_head_cmp = _matrix_compare(base.lm_head.weight.detach(), cmp_model.lm_head.weight.detach())
    tie_base = _tie_lm_embed_report(base)
    tie_cmp = _tie_lm_embed_report(cmp_model)

    # ---- 检查 2/3/4：swap head、top-10、按位置 row cosine ----
    swap_head: Dict[str, Any] = {}
    top10_last: Dict[str, Any] = {}
    pos_cosine: Dict[str, Any] = {}
    h_b = hids_b[-1] if hids_b is not None else None
    h_c = hids_c[-1] if hids_c is not None else None
    if h_b is not None and h_c is not None:
        logits_bb_from_h = _linear_logits_from_hidden(h_b, base.lm_head)
        sanity_linear = _row_cosine_1d(
            logits_bb_from_h[0, -1],
            logits_b[0, -1].to(logits_bb_from_h.device),
        )
        logits_cmp_h_base_head = _linear_logits_from_hidden(h_c, base.lm_head)
        lb_last = logits_b[0, -1]
        lc_last = logits_c[0, -1].to(lb_last.device)
        lswap_last = logits_cmp_h_base_head[0, -1].to(lb_last.device)

        swap_head = {
            "sanity_base_hidden_last_to_base_lm_head_vs_forward_logits_last_row": sanity_linear,
            "cosine_base_logits_vs_cmp_logits_last_row": _row_cosine_1d(lb_last, lc_last),
            "cosine_cmp_hidden_last_plus_base_lm_head_vs_base_logits_last_row": _row_cosine_1d(
                lswap_last, lb_last
            ),
            "cosine_cmp_hidden_last_plus_base_lm_head_vs_cmp_logits_last_row": _row_cosine_1d(
                lswap_last, lc_last
            ),
        }
        top10_last = {
            "base": _topk_logits_row(logits_b[0, -1], tok, 10),
            "cmp": _topk_logits_row(logits_c[0, -1], tok, 10),
        }
        positions = _pick_row_cosine_positions(tlen, args.seed, args.n_random_positions)
        row_cos: Dict[str, float] = {}
        for p in positions:
            row_cos[str(p)] = _row_cosine_1d(
                logits_b[0, p],
                logits_c[0, p].to(logits_b.device),
            )
        pos_cosine = {
            "positions": positions,
            "row_cosine_base_vs_cmp": row_cos,
            "logit_row_cosine_first": row_cos.get("0"),
            "logit_row_cosine_last": row_cos.get(str(tlen - 1)) if tlen > 0 else None,
        }

    per_layer: List[Dict[str, Any]] = []
    for li in range(n_layers):
        row: Dict[str, Any] = {"layer": li}
        row["base_qkv_rms"] = {k: qkv_b.get(li, {}).get(k) for k in ("base_q_rms", "base_k_rms", "base_v_rms")}
        row["cmp_qkv_rms"] = {k: qkv_c.get(li, {}).get(k) for k in ("cmp_q_rms", "cmp_k_rms", "cmp_v_rms")}

        hb = None
        hc = None
        if hids_b is not None and hids_c is not None and li + 1 < len(hids_b) and li + 1 < len(hids_c):
            hb = hids_b[li + 1].detach().float()
            hc = hids_c[li + 1].to(hb.device).detach().float()
        if hb is not None and hc is not None and hb.shape == hc.shape:
            cos, rel = _hidden_cosine_rel_l2(hb, hc)
            row["hidden_cosine"] = cos
            row["hidden_rel_l2"] = rel
        else:
            row["hidden_cosine"] = None
            row["hidden_rel_l2"] = None

        eb = float("nan")
        ec = float("nan")
        if attns_b is not None and li < len(attns_b) and attns_b[li] is not None:
            eb = _mean_attn_entropy(attns_b[li])
        if attns_c is not None and li < len(attns_c) and attns_c[li] is not None:
            ec = _mean_attn_entropy(attns_c[li])
        row["attn_entropy_mean"] = {"base": eb, "cmp": ec, "diff": ec - eb}

        per_layer.append(row)

    hidden_curve: Dict[str, Any] = {}
    hc_vals: List[Tuple[int, float]] = []
    for li in range(n_layers):
        c = per_layer[li].get("hidden_cosine")
        if isinstance(c, (int, float)) and math.isfinite(float(c)):
            hc_vals.append((li, float(c)))
    if hc_vals:
        by_cos = sorted(hc_vals, key=lambda x: x[1])
        hidden_curve = {
            "per_layer_hidden_cosine": [per_layer[i].get("hidden_cosine") for i in range(n_layers)],
            "min_cosine_layer": by_cos[0][0],
            "min_cosine": by_cos[0][1],
            "max_cosine_layer": by_cos[-1][0],
            "max_cosine": by_cos[-1][1],
        }

    embed_row: Dict[str, Any] = {}
    if hids_b is not None and hids_c is not None and len(hids_b) > 0 and len(hids_c) > 0:
        hb0 = hids_b[0].detach().float()
        hc0 = hids_c[0].to(hb0.device).detach().float()
        if hb0.shape == hc0.shape:
            c0, r0 = _hidden_cosine_rel_l2(hb0, hc0)
            embed_row = {"hidden_after_embed_cosine": c0, "hidden_after_embed_rel_l2": r0}

    report: Dict[str, Any] = {
        "base": args.base,
        "cmp": args.cmp,
        "low_json": args.low_json,
        "seqlen": tlen,
        "batch_size": bsz,
        "seed": args.seed,
        "dtype": args.dtype,
        "logits_stats": {"base": lb, "cmp": lc, "diff": logits_diff},
        "lm_head_weight": {
            "base_vs_cmp": lm_head_cmp,
            "tie_base_lm_vs_embed": tie_base,
            "tie_cmp_lm_vs_embed": tie_cmp,
        },
        "swap_head_last_token": swap_head,
        "top10_last_token": top10_last,
        "logit_row_cosine_positions": pos_cosine,
        "hidden_cosine_curve_summary": hidden_curve,
        "embed_compare": embed_row,
        "per_layer": per_layer,
        "note": (
            "base_qkv_rms 与 cmp_qkv_rms 维度不同（全秩 vs 低秩），仅作各自尺度排查。"
            " lm_head_weight：先看 base_vs_cmp；tie_* 中 weight_ptr_same=False 但数值一致可能仍算 tie。"
            " swap_head：若 cmp_hidden+base_head 与 base_logits 最后一行 cosine 明显高于 base_vs_cmp，问题多在 cmp.lm_head。"
            " logits.diff：请看 diff_mean_abs、logit_row_cosine_mean 与各位置 row_cosine。"
        ),
    }

    print("\n=== [1] lm_head.weight: base vs cmp ===")
    print(lm_head_cmp)
    print("=== [1] tie: lm_head vs embed_tokens ===")
    print("BASE tie:", tie_base)
    print("CMP  tie:", tie_cmp)

    print("\n=== [2] swap head (last row) ===")
    print(swap_head)

    print("\n=== [3] top-10 last token ===")
    print("BASE:", top10_last.get("base"))
    print("CMP :", top10_last.get("cmp"))

    print("\n=== [4] logit row cosine by position (base vs cmp) ===")
    print(pos_cosine)

    print("\n=== logits row-cosine / diff (aligned [B,T,V]) ===")
    print(
        "row_cosine_mean/min/max:",
        logits_diff.get("logit_row_cosine_mean"),
        logits_diff.get("logit_row_cosine_min"),
        logits_diff.get("logit_row_cosine_max"),
    )
    print("global_flat_cosine (misleading):", logits_diff.get("global_flat_cosine_misleading_do_not_use"))
    print("align notes:", logits_diff.get("align_note_base"), logits_diff.get("align_note_cmp"))
    if hidden_curve:
        print("hidden_cosine min/max layer:", hidden_curve.get("min_cosine_layer"), hidden_curve.get("max_cosine_layer"))

    print("\n=== logits marginal stats (full tensor) ===")
    print("BASE:", lb)
    print("CMP :", lc)
    _diff_keys = (
        "diff_mean_abs",
        "diff_std_abs",
        "diff_p99_abs",
        "diff_abs_quantiles",
        "logit_row_cosine_quantiles",
        "row_cosine_manual_vs_F_cosine_max_abs_err",
    )
    print("DIFF subset:", {k: logits_diff.get(k) for k in _diff_keys if k in logits_diff})

    print("\n=== per-layer hidden cosine (first/last) ===")
    if per_layer:
        print("layer 0:", per_layer[0].get("hidden_cosine"), per_layer[0].get("hidden_rel_l2"))
        print(f"layer {n_layers-1}:", per_layer[-1].get("hidden_cosine"), per_layer[-1].get("hidden_rel_l2"))

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {p}")

    del base, cmp_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
