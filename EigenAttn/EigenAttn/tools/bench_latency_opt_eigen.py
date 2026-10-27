#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base OPT vs EigenAttn 延迟与显存对比（memory-bound 证据用）。

- Prefill: 单次前向 [B, T]（因果 LM，无 labels）。
- Decode 一步: prefill 后仅对最后一个 token 再前向一步（past_key_values），计时第二步。

用法（在 EigenAttn 目录下 PYTHONPATH 含当前目录）:
  python tools/bench_latency_opt_eigen.py \\
    --base /diskE/opt-30b \\
    --cmp /path/to/eigen_model \\
    --low_json ./low_rank_config_layerwise_5712_opt30b.json \\
    --device cuda:0 \\
    --dtype bf16 \\
    --batch_sizes 1,2,4 \\
    --seqlens 512,1024,2048 \\
    --out_csv bench_latency.csv

顺序加载：先测 base 全流程，释放后再测 cmp，避免单卡双 30B OOM。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# repo root
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"unsupported dtype {s}")


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    m = n // 2
    if n % 2 == 1:
        return xs[m]
    return 0.5 * (xs[m - 1] + xs[m])


def _embedding_vocab_high(model: torch.nn.Module) -> int:
    """embedding 行数；randint 用 [0, high)，故 high=此处返回值。"""
    emb = model.get_input_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        return 50272
    n = int(emb.weight.shape[0])
    return max(1, n)


def _max_seqlen_for_bench(model: torch.nn.Module, modes: List[str]) -> int:
    """
    OPT 可学习位置编码：prefill 长度 T 需 T <= max_position_embeddings；
    decode 一步时总 mask 长 T+1，需 T+1 <= max_position_embeddings（见 HF OPTLearnedPositionalEmbedding）。
    同时跑 prefill+decode 时取更严的上界 max_position_embeddings - 1。
    """
    cfg = getattr(model, "config", None)
    m = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
    if m is None:
        return 10**9
    m = int(m)
    if m < 1:
        return 1
    if "decode_step" in modes:
        return max(1, m - 1)
    return m


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


@torch.inference_mode()
def bench_prefill(
    model: torch.nn.Module,
    device: torch.device,
    batch: int,
    seqlen: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> Tuple[float, int]:
    """返回 (median_latency_ms, peak_bytes_this_segment)。"""
    torch.cuda.reset_peak_memory_stats(device)
    high = _embedding_vocab_high(model)
    input_ids = torch.randint(
        low=0,
        high=high,
        size=(batch, seqlen),
        dtype=torch.long,
        device=device,
    )
    attn = torch.ones(batch, seqlen, dtype=torch.long, device=device)

    for _ in range(warmup):
        _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)

    torch.cuda.synchronize(device)
    times: List[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    peak = torch.cuda.max_memory_allocated(device)
    return _median(times), int(peak)


@torch.inference_mode()
def bench_decode_step(
    model: torch.nn.Module,
    device: torch.device,
    batch: int,
    seqlen: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> Tuple[float, int]:
    """
    prefill [B,T] 后，仅计时第二步 [B,1] + past_key_values。
    attention_mask 总长 T+1（与 HF 因果解码一致）。
    """
    torch.cuda.reset_peak_memory_stats(device)
    high = _embedding_vocab_high(model)
    input_ids = torch.randint(
        low=0,
        high=high,
        size=(batch, seqlen),
        dtype=torch.long,
        device=device,
    )
    attn_prefill = torch.ones(batch, seqlen, dtype=torch.long, device=device)

    # warmup includes prefill+decode
    for _ in range(warmup):
        out = model(input_ids=input_ids, attention_mask=attn_prefill, use_cache=True, return_dict=True)
        past = out.past_key_values
        step_ids = input_ids[:, -1:].contiguous()
        attn_step = torch.ones(batch, seqlen + 1, dtype=torch.long, device=device)
        _ = model(
            input_ids=step_ids,
            past_key_values=past,
            attention_mask=attn_step,
            use_cache=True,
            return_dict=True,
        )

    times: List[float] = []
    for _ in range(repeats):
        out = model(input_ids=input_ids, attention_mask=attn_prefill, use_cache=True, return_dict=True)
        past = out.past_key_values
        step_ids = input_ids[:, -1:].contiguous()
        attn_step = torch.ones(batch, seqlen + 1, dtype=torch.long, device=device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = model(
            input_ids=step_ids,
            past_key_values=past,
            attention_mask=attn_step,
            use_cache=True,
            return_dict=True,
        )
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    peak = torch.cuda.max_memory_allocated(device)
    return _median(times), int(peak)


def _parse_int_list(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def run_grid(
    name: str,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    batch_sizes: List[int],
    seqlens: List[int],
    warmup: int,
    repeats: int,
    modes: List[str],
) -> List[Dict[str, Any]]:
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    ev = _embedding_vocab_high(model)
    cap = _max_seqlen_for_bench(model, modes)
    mpos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    rows: List[Dict[str, Any]] = []
    for B in batch_sizes:
        for T in seqlens:
            T_use = min(T, cap)
            row: Dict[str, Any] = {
                "model": name,
                "batch_size": B,
                "seqlen": T,
                "seqlen_used": T_use,
                "tokens": B * T_use,
                "embed_vocab_high": ev,
                "max_position_embeddings": mpos,
            }
            if T_use < T:
                row["note"] = f"seqlen clamped to {T_use} (OPT position embed / decode limit)"
            try:
                if "prefill" in modes:
                    ms, peak = bench_prefill(
                        model, device, B, T_use, dtype, warmup, repeats
                    )
                    row["prefill_ms_median"] = round(ms, 4)
                    row["prefill_peak_mib"] = round(peak / (1024 * 1024), 2)
                if "decode_step" in modes:
                    ms_d, peak_d = bench_decode_step(
                        model, device, B, T_use, dtype, warmup, repeats
                    )
                    row["decode_step_ms_median"] = round(ms_d, 4)
                    row["decode_step_peak_mib"] = round(peak_d / (1024 * 1024), 2)
            except Exception as e:
                err = str(e)
                if "out of memory" in err.lower():
                    row["error"] = "OOM"
                elif "cuda" in err.lower() or "CUDA" in err or "assert" in err.lower():
                    row["error"] = err[:300]
                else:
                    row["error"] = err[:200]
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True, help="全秩 OPT 权重目录")
    ap.add_argument("--cmp", type=str, required=True, help="EigenAttn 低秩模型目录")
    ap.add_argument("--low_json", type=str, required=True, help="low_rank_config.json")
    ap.add_argument("--tok", type=str, default="", help="tokenizer；默认用 --base")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--batch_sizes", type=str, default="1,2,4", help="逗号分隔，如 1,2,4,8")
    ap.add_argument("--seqlens", type=str, default="512,1024,2048,4096")
    ap.add_argument("--modes", type=str, default="prefill,decode_step", help="prefill / decode_step / 逗号组合")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--out_csv", type=str, default="bench_latency_opt_eigen.csv")
    args = ap.parse_args()

    dtype = _dtype_from_str(args.dtype)
    device = torch.device(args.device)
    tok_path = args.tok.strip() or args.base
    _ = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, use_fast=False)
    batch_sizes = _parse_int_list(args.batch_sizes)
    seqlens = _parse_int_list(args.seqlens)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print("[bench] device=", device, "dtype=", dtype, "modes=", modes)
    print("[bench] grid batch_sizes=", batch_sizes, "seqlens=", seqlens)

    all_rows: List[Dict[str, Any]] = []

    print("[bench] loading BASE ...")
    base = _load_base(args.base, dtype, args.device)
    _cap = _max_seqlen_for_bench(base, modes)
    _mpe = getattr(base.config, "max_position_embeddings", None)
    print(
        "[bench] OPT position embed: max_position_embeddings=",
        _mpe,
        "=> seqlen_used <= ",
        _cap,
        "(with decode_step: min(T, max_pos-1); prefill-only: min(T, max_pos))",
    )
    rows_b = run_grid(
        "base",
        base,
        device,
        dtype,
        batch_sizes,
        seqlens,
        args.warmup,
        args.repeats,
        modes,
    )
    all_rows.extend(rows_b)
    del base
    gc.collect()
    torch.cuda.empty_cache()

    print("[bench] loading CMP (EigenAttn) ...")
    cmp_model = _load_eigen(args.cmp, args.low_json, dtype, args.device)
    rows_c = run_grid(
        "eigen",
        cmp_model,
        device,
        dtype,
        batch_sizes,
        seqlens,
        args.warmup,
        args.repeats,
        modes,
    )
    all_rows.extend(rows_c)
    del cmp_model
    gc.collect()
    torch.cuda.empty_cache()

    # merge ratio per (B,T)
    by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in all_rows:
        key = (int(r["batch_size"]), int(r["seqlen"]))
        if key not in by_key:
            by_key[key] = {"batch_size": key[0], "seqlen": key[1], "tokens": key[0] * key[1]}
        mname = r["model"]
        for k, v in r.items():
            if k in ("model", "batch_size", "seqlen", "tokens"):
                continue
            by_key[key][f"{mname}_{k}"] = v

    ratio_rows: List[Dict[str, Any]] = []
    for key in sorted(by_key.keys()):
        d = by_key[key]
        su = d.get("base_seqlen_used", d.get("eigen_seqlen_used", key[1]))
        try:
            su_i = int(su)
        except (TypeError, ValueError):
            su_i = key[1]
        rr: Dict[str, Any] = {
            "batch_size": key[0],
            "seqlen": key[1],
            "seqlen_used": su_i,
            "tokens": key[0] * su_i,
        }
        if "base_prefill_ms_median" in d and "eigen_prefill_ms_median" in d:
            bb = d["base_prefill_ms_median"]
            ee = d["eigen_prefill_ms_median"]
            if isinstance(bb, (int, float)) and isinstance(ee, (int, float)) and ee > 0:
                rr["prefill_speedup_base_over_eigen"] = round(bb / ee, 4)
        if "base_decode_step_ms_median" in d and "eigen_decode_step_ms_median" in d:
            bb = d["base_decode_step_ms_median"]
            ee = d["eigen_decode_step_ms_median"]
            if isinstance(bb, (int, float)) and isinstance(ee, (int, float)) and ee > 0:
                rr["decode_speedup_base_over_eigen"] = round(bb / ee, 4)
        ratio_rows.append({**d, **rr})

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for r in ratio_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in ratio_rows:
            w.writerow(r)

    print(f"[bench] wrote {out_path} ({len(ratio_rows)} rows)")
    print(
        "[bench] interpret: prefill_speedup_base_over_eigen > 1 means eigen faster (lower ms)."
        " Same for decode_speedup_base_over_eigen."
    )


if __name__ == "__main__":
    main()
