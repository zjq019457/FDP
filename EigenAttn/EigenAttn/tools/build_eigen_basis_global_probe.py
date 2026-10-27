#!/usr/bin/env python
"""
build_eigen_basis_global_probe.py
=================================

Task-aware global error-propagation rank allocation for EigenAttention.

与 build_eigen_basis_opt_error_budget_offline.py 的区别：
- 不再做 per-layer error_budget 阈值搜索；
- 对每一层用同一套 per-head SVD 得到 (basis, 归一化特征谱)；
- 为每一层做一次 "probe cut" —— 用较激进的 rank 替换该层，测该层输出的
  相对激活误差 ε_probe[l]；
- 再在全局 KV 预算约束下做 min-heap greedy:
      marginal_cost(l, kind, r → r-1) = (Σ_h eval[l,kind,h,r-1]) × ε_probe[l]^α
  每次弹出边际代价最小的 (l, kind) 把 per-head rank 减一，直到
      Σ_l (per_head_rank_kq[l] + per_head_rank_v[l]) ≤ kv_target_ratio × 2 × head_dim × n_layers
  同时保留 per_head_rank ≥ min_per_head_rank（对齐原文 rank>64 总约束）。
- 输出 basis .pt 和 low_rank_config.json，格式与 main_eigen_attn.py
  / sanity_AB_opt30b.py 的加载逻辑完全兼容。

Ref 用户 idea:
  Δy ≈ Σ_i J_i · ΔW_i
  probe: 替换一层 → 测 Δactivation（Δlogits 的一阶代理）
  → layer importance score, 取代 per-layer heuristic threshold.
"""
from __future__ import annotations

import argparse
import gc
import heapq
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 复用 offline 脚本里的 I/O + 数据流 helpers
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]  # .../EigenAttn/EigenAttn
sys.path.insert(0, str(_THIS.parent))
from build_eigen_basis_opt_error_budget_offline import (  # type: ignore
    _iter_texts_from_dir,
    _build_token_stream,
    _basis_key,
    _pack_headwise_basis,
    _relative_error_streaming_layers,
)


def _load_py_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cumulative_tail_cost(eval_per_head: torch.Tensor) -> torch.Tensor:
    """
    eval_per_head: [num_heads, head_dim]  归一化谱 (Σ_dim eval[h,:] ≈ 1)
    返回: cost_per_rank_drop[r] = Σ_h eval[h, r]    (r 从 0..head_dim-1)
    即：把 per_head_rank 从 (r+1) 降到 r 的边际能量损失。
    """
    return eval_per_head.float().sum(dim=0).cpu()


def _global_greedy_allocate(
    per_layer_eval_kq: List[torch.Tensor],
    per_layer_eval_v: List[torch.Tensor],
    importance: torch.Tensor,  # [n_layers], layer-wise probe sensitivity
    num_heads: int,
    head_dim: int,
    kv_target_ratio: float,
    min_per_head_rank: int,
    imp_alpha: float,
    verbose: bool = True,
) -> Tuple[List[int], List[int], Dict]:
    """
    min-heap greedy: 每次把边际代价最小的 (l, kind) per-head rank -= 1.
    - marginal_cost(l, kind, r) = tail_cost[l, kind, r-1] × imp[l]^α
    - 边界: per_head_rank >= min_per_head_rank.
    - 停机: 总 per-head-rank 预算 <= target_total.
    """
    n_layers = len(per_layer_eval_kq)
    assert len(per_layer_eval_v) == n_layers

    # Pre-compute tail-energy per rank drop for every layer × kind.
    tail_kq = [_cumulative_tail_cost(e) for e in per_layer_eval_kq]  # each [head_dim]
    tail_v = [_cumulative_tail_cost(e) for e in per_layer_eval_v]

    # 初始: 每层每 kind 的 per-head rank = head_dim (full rank)
    r_kq = [head_dim] * n_layers
    r_v = [head_dim] * n_layers

    # 目标: sum(r_kq) + sum(r_v) <= target_total
    full_total_ph = 2 * head_dim * n_layers
    target_total_ph = int(round(kv_target_ratio * full_total_ph))

    imp_w = importance.pow(float(imp_alpha)).clamp_min(1e-8)

    heap: List[Tuple[float, int, str, int]] = []  # (cost, layer, kind, new_rank_after_cut)
    # Seed heap: 每层每 kind 的下一个可 cut 的 rank (cost at r → r-1)
    for l in range(n_layers):
        # cut r_kq[l] → r_kq[l] - 1 的代价 = tail_kq[l][r_kq[l] - 1]
        if r_kq[l] > min_per_head_rank:
            c = float(tail_kq[l][r_kq[l] - 1]) * float(imp_w[l])
            heapq.heappush(heap, (c, l, "kq", r_kq[l] - 1))
        if r_v[l] > min_per_head_rank:
            c = float(tail_v[l][r_v[l] - 1]) * float(imp_w[l])
            heapq.heappush(heap, (c, l, "v", r_v[l] - 1))

    cur_total = sum(r_kq) + sum(r_v)
    step = 0
    trace = []
    while cur_total > target_total_ph and heap:
        c, l, kind, new_r = heapq.heappop(heap)
        # 检查是否已被更 deep 的 cut 覆盖（heap 里可能有 stale 记录）
        cur_r = r_kq[l] if kind == "kq" else r_v[l]
        if new_r != cur_r - 1:
            # stale (这个候选对应的 cur_r 已经被别的路径减小过); 以当前 r 重新计算
            if cur_r > min_per_head_rank:
                tl = tail_kq[l] if kind == "kq" else tail_v[l]
                c_new = float(tl[cur_r - 1]) * float(imp_w[l])
                heapq.heappush(heap, (c_new, l, kind, cur_r - 1))
            continue

        if kind == "kq":
            r_kq[l] = new_r
        else:
            r_v[l] = new_r
        cur_total -= 1
        step += 1

        # 如还能继续 cut, push 下一档
        if new_r > min_per_head_rank:
            tl = tail_kq[l] if kind == "kq" else tail_v[l]
            c_next = float(tl[new_r - 1]) * float(imp_w[l])
            heapq.heappush(heap, (c_next, l, kind, new_r - 1))

        if verbose and (step % 200 == 0):
            print(
                f"[alloc step={step:05d}] cur_total_ph={cur_total} (target={target_total_ph}) "
                f"last_cut=(L={l},{kind},r={new_r}) cost={c:.3e}",
                flush=True,
            )

    info = {
        "target_ph_total": target_total_ph,
        "final_ph_total": cur_total,
        "full_ph_total": full_total_ph,
        "achieved_ratio": cur_total / full_total_ph,
        "steps": step,
        "imp_alpha": imp_alpha,
        "min_per_head_rank": min_per_head_rank,
    }
    return r_kq, r_v, info


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_prefix", type=str, default="eigen_basis_opt30b_global_probe")

    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--token_budget", type=int, default=1_500_000)
    ap.add_argument("--max_data_files", type=int, default=300)
    ap.add_argument("--max_data_lines", type=int, default=3_000_000)
    ap.add_argument("--avg_dim", type=int, default=32)

    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--device_map", type=str, default="")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tokenizer", type=str, default="")

    # Global allocation parameters
    ap.add_argument(
        "--kv_target_ratio", type=float, default=0.9,
        help="Target KV cache ratio vs. uncompressed baseline (e.g. 0.9 / 0.8 / 0.7).",
    )
    ap.add_argument(
        "--min_per_head_rank", type=int, default=2,
        help="Per-head rank lower bound; OPT-30B=56 heads so 2 ≙ total>=112>64 (paper).",
    )

    # Probe parameters
    ap.add_argument(
        "--probe_per_head_rank", type=int, default=64,
        help="Probe cut target: temporarily compress layer to this per-head rank, "
             "measure activation error → layer importance.",
    )
    ap.add_argument(
        "--probe_batches", type=int, default=32,
        help="Number of cached calibration samples used for probe activation-error eval.",
    )
    ap.add_argument(
        "--imp_alpha", type=float, default=1.0,
        help="Exponent applied to importance score in marginal_cost. α=0 → pure spectrum alloc; "
             "α=1 → probe-weighted; α>1 amplifies importance contrast.",
    )
    ap.add_argument(
        "--no_probe", action="store_true",
        help="Skip probe step; allocate purely by spectrum tail energy (α effectively 0).",
    )

    ap.add_argument("--verbose", type=int, default=1)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    repo_root = _ROOT
    base_root = repo_root.parent / "EigenAttn_base"
    if not base_root.exists():
        raise FileNotFoundError(f"Cannot find EigenAttn_base at: {base_root}")

    mod_utils = _load_py_module("eigen_base_utils", base_root / "decompose" / "eigen_attn_utils.py")
    mod_dm = _load_py_module("eigen_base_decompose_modules", base_root / "models" / "decompose_modules.py")
    decompose_opt_layer = mod_utils.decompose_opt_layer
    OPTEigenAttnDecoderLayer = mod_dm.OPTEigenAttnDecoderLayer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_path = args.tokenizer.strip() if args.tokenizer.strip() else args.model
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    texts = _iter_texts_from_dir(args.data_dir, args.max_data_files, args.max_data_lines)
    stream = _build_token_stream(tok, texts, args.token_budget, args.seed)
    if stream is None:
        raise RuntimeError("empty token stream from data_dir")

    dt = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    dm = (args.device_map or "").strip().lower()
    if dm == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dt, device_map="auto",
            local_files_only=True, low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dt, device_map={"": args.device},
            local_files_only=True, low_cpu_mem_usage=True,
        )
    model.eval()
    model.config.use_cache = False

    layers = model.model.decoder.layers
    n_layers = len(layers)
    num_heads = int(model.config.num_attention_heads)
    hidden = int(model.config.hidden_size)
    head_dim = hidden // num_heads

    if args.probe_per_head_rank > head_dim:
        raise ValueError(f"probe_per_head_rank({args.probe_per_head_rank}) > head_dim({head_dim})")
    if args.min_per_head_rank < 1 or args.min_per_head_rank > head_dim:
        raise ValueError(f"min_per_head_rank out of range: {args.min_per_head_rank}")

    # === Build calibration batches + capture layer-0 inputs ===
    N = stream.numel()
    if N <= args.seqlen + 2:
        raise RuntimeError(f"token stream too short ({N})")

    g = torch.Generator().manual_seed(args.seed)
    batches = []
    for _ in range(args.nsamples):
        s = int(torch.randint(low=0, high=N - args.seqlen - 1, size=(1,), generator=g).item())
        x = stream[s : s + args.seqlen].unsqueeze(0)
        batches.append(x)

    dev = model.get_input_embeddings().weight.device
    inps = torch.zeros((args.nsamples, args.seqlen, hidden), dtype=dt, device=dev)
    cache: Dict[str, object] = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for b in batches:
        if cache["i"] >= args.nsamples:
            break
        try:
            model(b.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    attention_mask = cache.get("attention_mask", None)

    # === Stage 1: per-layer SVD + probe, collect (basis, eval, importance) ===
    per_layer_basis_kq: List[torch.Tensor] = []
    per_layer_basis_v: List[torch.Tensor] = []
    per_layer_eval_kq: List[torch.Tensor] = []
    per_layer_eval_v: List[torch.Tensor] = []
    importance = torch.ones(n_layers, dtype=torch.float32)
    probe_report: List[dict] = []

    print(f"[config] kv_target_ratio={args.kv_target_ratio} "
          f"min_per_head_rank={args.min_per_head_rank} "
          f"probe_per_head_rank={args.probe_per_head_rank} "
          f"probe_batches={args.probe_batches} "
          f"imp_alpha={args.imp_alpha} no_probe={args.no_probe}", flush=True)

    for li in range(n_layers):
        layer = layers[li]
        lay_dev = next(layer.parameters()).device
        if torch.cuda.is_available() and lay_dev.type == "cuda":
            torch.cuda.set_device(lay_dev)
        inps = inps.to(lay_dev)
        if attention_mask is not None and torch.is_tensor(attention_mask):
            attention_mask = attention_mask.to(lay_dev)

        decomp_args = SimpleNamespace(
            nsamples=int(args.nsamples),
            low_rank=False,
            eigen_attn_params={"threshold": 1.0, "avg_dim_features": int(args.avg_dim)},
        )

        basis_kq_h, eval_kq, basis_v_h, eval_v = decompose_opt_layer(
            layer, inps, decomp_args, num_heads, li, attention_mask=attention_mask
        )

        # basis/eval 统一移到 CPU fp32 存储（供后续 pack + alloc）
        basis_kq_h = basis_kq_h.detach().cpu().contiguous().float()
        basis_v_h = basis_v_h.detach().cpu().contiguous().float()
        eval_kq_cpu = eval_kq.detach().cpu().contiguous().float()
        eval_v_cpu = eval_v.detach().cpu().contiguous().float()

        per_layer_basis_kq.append(basis_kq_h)
        per_layer_basis_v.append(basis_v_h)
        per_layer_eval_kq.append(eval_kq_cpu)
        per_layer_eval_v.append(eval_v_cpu)

        # ===== Probe: 用 probe_per_head_rank 做一次激进 cut, 测激活相对误差 =====
        if not args.no_probe:
            probe_rank_kq = int(args.probe_per_head_rank * num_heads)
            probe_rank_v = int(args.probe_per_head_rank * num_heads)
            qlayer = OPTEigenAttnDecoderLayer(
                layer, decomp_args, basis_kq_h, probe_rank_kq, basis_v_h, probe_rank_v, model.config
            ).to(device=lay_dev, dtype=dt)

            probe_n = min(int(args.probe_batches), int(args.nsamples))
            err_probe, _num, _den = _relative_error_streaming_layers(
                layer, qlayer, inps, attention_mask, probe_n
            )
            if torch.cuda.is_available() and lay_dev.type == "cuda":
                torch.cuda.synchronize(lay_dev)
            del qlayer

            importance[li] = float(err_probe)
            if int(args.verbose) == 1:
                print(f"[probe L={li:02d}] probe_per_head_rank={args.probe_per_head_rank} "
                      f"err_probe(activation)={err_probe:.6f}", flush=True)
            probe_report.append({"layer": li, "err_probe": float(err_probe)})

        # Propagate via FULL layer
        for j in range(args.nsamples):
            inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        if torch.cuda.is_available():
            for di in range(torch.cuda.device_count()):
                torch.cuda.synchronize(di)
        layers[li] = layer.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[layer {li:02d}] done", flush=True)

    # === Stage 2: Global greedy allocation ===
    if args.no_probe:
        imp_for_alloc = torch.ones(n_layers, dtype=torch.float32)
    else:
        # 归一到平均=1, 避免不同 layer absolute scale 主导
        imp_for_alloc = importance / importance.mean().clamp_min(1e-8)

    print(f"[importance] raw min={float(importance.min()):.6f} "
          f"max={float(importance.max()):.6f} mean={float(importance.mean()):.6f}", flush=True)
    if int(args.verbose) == 1:
        for li in range(n_layers):
            print(f"  imp[L={li:02d}]={float(importance[li]):.6f} "
                  f"(norm={float(imp_for_alloc[li]):.4f})", flush=True)

    per_head_r_kq, per_head_r_v, alloc_info = _global_greedy_allocate(
        per_layer_eval_kq=per_layer_eval_kq,
        per_layer_eval_v=per_layer_eval_v,
        importance=imp_for_alloc,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_target_ratio=float(args.kv_target_ratio),
        min_per_head_rank=int(args.min_per_head_rank),
        imp_alpha=float(args.imp_alpha),
        verbose=bool(args.verbose),
    )

    # === Stage 3: Save basis + low_rank_config ===
    basis_dict: Dict[str, torch.Tensor] = {}
    low_rank_config: List[List[int]] = []

    for li in range(n_layers):
        rk = int(per_head_r_kq[li] * num_heads)
        rv = int(per_head_r_v[li] * num_heads)
        low_rank_config.append([rk, rv])

        U_kq = _pack_headwise_basis(per_layer_basis_kq[li], rk, num_heads, head_dim).cpu().float()
        U_v = _pack_headwise_basis(per_layer_basis_v[li], rv, num_heads, head_dim).cpu().float()
        basis_dict[_basis_key(li, "q")] = U_kq
        basis_dict[_basis_key(li, "k")] = U_kq
        basis_dict[_basis_key(li, "v")] = U_v

    out_basis = out_dir / f"{args.out_prefix}.pt"
    torch.save(basis_dict, out_basis)
    print(f"[saved] {out_basis}", flush=True)

    out_cfg = out_dir / f"low_rank_config_layerwise_{args.out_prefix}.json"
    with open(out_cfg, "w", encoding="utf-8") as f:
        json.dump(low_rank_config, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_cfg}", flush=True)

    # Detailed report
    report = {
        "config": {
            "model": args.model,
            "data_dir": args.data_dir,
            "nsamples": int(args.nsamples),
            "seqlen": int(args.seqlen),
            "num_heads": num_heads,
            "head_dim": head_dim,
            "n_layers": n_layers,
            "kv_target_ratio": float(args.kv_target_ratio),
            "min_per_head_rank": int(args.min_per_head_rank),
            "probe_per_head_rank": int(args.probe_per_head_rank),
            "probe_batches": int(args.probe_batches),
            "imp_alpha": float(args.imp_alpha),
            "no_probe": bool(args.no_probe),
            "dtype": args.dtype,
        },
        "allocation": alloc_info,
        "per_layer": [
            {
                "layer": li,
                "per_head_rank_kq": int(per_head_r_kq[li]),
                "per_head_rank_v": int(per_head_r_v[li]),
                "rank_kq_total": int(per_head_r_kq[li] * num_heads),
                "rank_v_total": int(per_head_r_v[li] * num_heads),
                "importance_raw": float(importance[li]),
                "importance_norm": float(imp_for_alloc[li]),
            }
            for li in range(n_layers)
        ],
        "probe_report": probe_report,
    }
    out_rep = out_dir / f"{args.out_prefix}_report.json"
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_rep}", flush=True)

    # 快速摘要
    total_ph = alloc_info["final_ph_total"]
    full_ph = alloc_info["full_ph_total"]
    print(f"\n=== SUMMARY ===")
    print(f"  achieved KV ratio = {total_ph}/{full_ph} = {total_ph/full_ph:.4f}")
    print(f"  target KV ratio   = {args.kv_target_ratio}")
    print(f"  alloc steps       = {alloc_info['steps']}")
    print(f"\nlow_rank_config (total ranks per layer):")
    for li in range(n_layers):
        print(f"  L={li:02d}  kq={per_head_r_kq[li]*num_heads:>5d}  "
              f"v={per_head_r_v[li]*num_heads:>5d}  "
              f"per_head=({per_head_r_kq[li]},{per_head_r_v[li]})  "
              f"imp={float(importance[li]):.4f}")


if __name__ == "__main__":
    main()
