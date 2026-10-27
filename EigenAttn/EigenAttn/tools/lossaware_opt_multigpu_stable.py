#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stable multi-GPU / low-VRAM loss-aware basis generator for OPT models.

What this version fixes vs. the original draft:
1. Safer on 4xA100 with device_map=auto/balanced/sequential.
2. No cross-layer candidate module caching in sharded mode.
3. Explicit NaN/Inf handling for candidate NLL.
4. Greedy allocation is the default and the recommended mode for sharded runs.
5. More aggressive memory cleanup and fewer hidden device mismatches.
6. Optional per-GPU max_memory settings for from_pretrained.
7. Safer layer replacement under accelerate AlignDevicesHook.

Notes:
- This script still depends on your local EigenAttn_base repo, exactly like the old one.
- For 4xA100 runs, use alloc_method=greedy unless you intentionally force a single-GPU beam run.
- Beam search remains available only for non-sharded / single-device execution.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
except Exception:  # pragma: no cover
    AlignDevicesHook = None
    add_hook_to_module = None


# ----------------------------- small utilities ----------------------------- #

def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            # If CUDA context is already poisoned (e.g. illegal access), avoid crashing again in cleanup.
            return
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _load_py_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iter_texts_from_dir(data_dir: str, max_files: int, max_lines: int):
    import gzip
    import glob

    exts = ["*.txt", "*.jsonl", "*.json", "*.gz"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(str(Path(data_dir) / "**" / e), recursive=True))
    files = sorted(files)[: max(1, int(max_files))]

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
                                t = it.get("text") if isinstance(it, dict) else it
                                if isinstance(t, str) and t.strip():
                                    yield t.strip()
                                    n += 1
                                    if n >= max_lines:
                                        return
                        elif isinstance(obj, dict):
                            t = obj.get("text") or obj.get("content")
                            if isinstance(t, str) and t.strip():
                                yield t.strip()
                                n += 1
                                if n >= max_lines:
                                    return
                    except Exception:
                        pass
                    continue

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            j = json.loads(line)
                            if isinstance(j, dict):
                                t = j.get("text") or j.get("content")
                            else:
                                t = line
                        except Exception:
                            t = line
                    else:
                        t = line
                    if isinstance(t, str) and t.strip():
                        yield t.strip()
                        n += 1
                        if n >= max_lines:
                            return
        except Exception:
            continue


def _build_token_stream(tok, texts_iter: Iterable[str], target_tokens: int, seed: int):
    random.seed(int(seed))
    ids: List[int] = []
    total = 0
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    for t in texts_iter:
        enc = tok(t, add_special_tokens=False)["input_ids"]
        if not enc:
            continue
        if eos_id is not None:
            ids.extend(enc + [eos_id])
            total += len(enc) + 1
        else:
            ids.extend(enc)
            total += len(enc)
        if total >= int(target_tokens):
            break
    if total == 0:
        return None
    return torch.tensor(ids, dtype=torch.long)


def _basis_key(layer_idx: int, kind: str) -> str:
    return f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{kind}_proj_basis"


def _rank_from_eval(eval_per_head: torch.Tensor, threshold: float, num_heads: int) -> int:
    c = torch.cumsum(eval_per_head.float(), dim=1)
    tgt = torch.full((c.size(0), 1), float(threshold), device=c.device, dtype=c.dtype)
    per_head = torch.searchsorted(c, tgt, right=False).squeeze(1) + 1
    per_head = torch.clamp(per_head, min=1, max=c.size(1))
    return int(num_heads * torch.amax(per_head).item())


def _per_head_rank_from_eval(eval_per_head: torch.Tensor, threshold: float) -> List[int]:
    c = torch.cumsum(eval_per_head.float(), dim=1)
    tgt = torch.full((c.size(0), 1), float(threshold), device=c.device, dtype=c.dtype)
    per_head = torch.searchsorted(c, tgt, right=False).squeeze(1) + 1
    per_head = torch.clamp(per_head, min=1, max=c.size(1))
    return [int(x) for x in per_head.tolist()]


def _finalize_rank(rank: int, num_heads: int, head_dim: int, min_rank: int, strict_original: bool) -> int:
    max_r = num_heads * head_dim
    r = max(0, min(int(rank), max_r))
    k = r // num_heads if num_heads else 0
    if k < 1:
        k = 1
    if k > head_dim:
        k = head_dim
    r = num_heads * k
    if (not strict_original) and r < int(min_rank):
        k = (int(min_rank) + num_heads - 1) // num_heads
        k = max(1, min(k, head_dim))
        r = num_heads * k
    return int(min(r, max_r))


def _pack_headwise_basis(
    basis_headwise: torch.Tensor,
    rank_total: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if rank_total % num_heads != 0:
        raise ValueError(f"rank_total={rank_total} not divisible by num_heads={num_heads}")
    k = int(rank_total // num_heads)
    if k <= 0 or k > head_dim:
        raise ValueError(f"invalid per-head rank {k}, head_dim={head_dim}")
    d = num_heads * head_dim
    U = torch.zeros((d, rank_total), dtype=basis_headwise.dtype, device=basis_headwise.device)
    for h in range(num_heads):
        rs = slice(h * head_dim, (h + 1) * head_dim)
        cs = slice(h * k, (h + 1) * k)
        U[rs, cs] = basis_headwise[h, :, :k]
    return U.contiguous()


def _energy_stats(eval_per_head: torch.Tensor, per_head_ranks: List[int]) -> Dict[str, float]:
    e = eval_per_head.float().cpu()
    c = torch.cumsum(e, dim=1)
    vals = []
    hd = c.size(1)
    for h, r in enumerate(per_head_ranks):
        rr = int(max(1, min(r, hd)))
        vals.append(float(c[h, rr - 1].item()))
    t = torch.tensor(vals, dtype=torch.float32)
    return {
        "mean": float(t.mean().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def _safe_float(x: float, fallback: float = 1e9) -> float:
    if x is None or not math.isfinite(float(x)):
        return float(fallback)
    return float(x)


def _spectral_proxy_score(
    energy_kq: Dict[str, float],
    energy_v: Dict[str, float],
    rank_kq: int,
    rank_v: int,
    hidden: int,
) -> float:
    """
    Pure surrogate score used in multi-shard proxy mode.
    Lower is better, designed to mimic compression-induced risk without any forward pass.
    """
    max_rank = max(1, int(hidden))
    rkq_ratio = float(rank_kq) / float(max_rank)
    rv_ratio = float(rank_v) / float(max_rank)
    # Energy shortfall dominates, rank penalty is a secondary tie-breaker.
    energy_shortfall = max(0.0, 1.0 - float(energy_kq.get("mean", 0.0))) + max(0.0, 1.0 - float(energy_v.get("mean", 0.0)))
    rank_penalty = (1.0 - rkq_ratio) + (1.0 - rv_ratio)
    return _safe_float(energy_shortfall + 0.10 * rank_penalty, fallback=1e9)


def _downsample_candidates(cands: List["Candidate"], keep_n: int) -> List["Candidate"]:
    if len(cands) <= keep_n:
        return cands
    idx = sorted(
        set(
            [0, len(cands) - 1]
            + [int(round(i * (len(cands) - 1) / (keep_n - 1))) for i in range(keep_n)]
        )
    )
    return [cands[i] for i in idx]


# -------------------------- model / scoring helpers ------------------------ #

def _attach_like_accelerate_hook(src_module: torch.nn.Module, dst_module: torch.nn.Module) -> bool:
    if add_hook_to_module is None or AlignDevicesHook is None:
        return False
    src_hook = getattr(src_module, "_hf_hook", None)
    if src_hook is None or src_hook.__class__.__name__ != "AlignDevicesHook":
        return False
    try:
        new_hook = AlignDevicesHook(
            execution_device=getattr(src_hook, "execution_device", None),
            offload=getattr(src_hook, "offload", False),
            io_same_device=getattr(src_hook, "io_same_device", False),
            weights_map=getattr(src_hook, "weights_map", None),
            offload_buffers=getattr(src_hook, "offload_buffers", False),
            place_submodules=getattr(src_hook, "place_submodules", False),
            skip_keys=getattr(src_hook, "skip_keys", None),
            tied_params_map=getattr(src_hook, "tied_params_map", None),
        )
        add_hook_to_module(dst_module, new_hook)
        return True
    except Exception:
        return False


def _maybe_model_input_device(model) -> torch.device:
    emb = model.get_input_embeddings()
    return emb.weight.device


@torch.inference_mode()
def _lm_nll(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> float:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = out.logits[:, :-1, :].float().contiguous()
    labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")
    val = float(loss.item())
    if not math.isfinite(val):
        return float("nan")
    return val


@torch.inference_mode()
def _relative_layer_error_streaming(
    layer,
    qlayer,
    inps: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    nsamples: int,
) -> float:
    num_sq = 0.0
    den_sq = 0.0
    for j in range(int(nsamples)):
        x = inps[j].unsqueeze(0)
        y_hr = layer(x, attention_mask=attention_mask)[0]
        y_lr = qlayer(x, attention_mask=attention_mask)[0]
        diff = y_hr - y_lr
        num_sq += float((diff.float() * diff.float()).sum().item())
        den_sq += float((y_hr.float() * y_hr.float()).sum().item())
        del x, y_hr, y_lr, diff
    if den_sq <= 0.0:
        return 0.0
    val = float(math.sqrt(num_sq / den_sq))
    return _safe_float(val, fallback=1e9)


def _score_single_layer_candidate(
    model,
    layers,
    layer_idx: int,
    replacement_layer: torch.nn.Module,
    nll_batches: List[torch.Tensor],
    input_dev: torch.device,
    baseline_nll: float,
    multi_shard_mode: bool,
) -> float:
    orig = layers[layer_idx]
    try:
        if multi_shard_mode:
            ok = _attach_like_accelerate_hook(orig, replacement_layer)
            if not ok and getattr(orig, "_hf_hook", None) is not None:
                return float("nan")
        layers[layer_idx] = replacement_layer
        vals = []
        for b in nll_batches:
            bdev = b.to(input_dev, non_blocking=True)
            v = _lm_nll(model, bdev, None)
            vals.append(v)
            del bdev
        if len(vals) == 0 or any((not math.isfinite(v)) for v in vals):
            return float("nan")
        cand_nll = float(sum(vals) / len(vals))
        delta = cand_nll - baseline_nll
        return _safe_float(delta, fallback=1e9)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda error" in msg or "device-side assert" in msg:
            cleanup_memory()
            return float("nan")
        raise
    finally:
        layers[layer_idx] = orig
        cleanup_memory()


# ------------------------------- data objects ------------------------------ #

@dataclass
class Candidate:
    layer: int
    threshold: float
    rank_kq: int
    rank_v: int
    per_head_kq: List[int]
    per_head_v: List[int]
    energy_kq: Dict[str, float]
    energy_v: Dict[str, float]
    layer_err: float
    nll_delta: float = 0.0

    @property
    def cost(self) -> int:
        return int(self.rank_kq + self.rank_v)


# ----------------------------- allocation logic ---------------------------- #

def _allocate_greedy(layer_candidates: List[List[Candidate]], n_layers: int, target_total: int, verbose: int):
    cur_idx = [0 for _ in range(n_layers)]
    cur_total = sum(layer_candidates[i][cur_idx[i]].cost for i in range(n_layers))
    alloc_trace = []
    step = 0

    while cur_total > target_total:
        best = None
        for li in range(n_layers):
            i = cur_idx[li]
            if i + 1 >= len(layer_candidates[li]):
                continue
            cur = layer_candidates[li][i]
            nxt = layer_candidates[li][i + 1]
            release = cur.cost - nxt.cost
            if release <= 0:
                continue

            cur_nd = _safe_float(cur.nll_delta, fallback=1e9)
            nxt_nd = _safe_float(nxt.nll_delta, fallback=1e9)
            penalty = max(0.0, nxt_nd - cur_nd)
            score = penalty / release
            cand = (score, penalty, release, li, i + 1)
            if best is None or cand < best:
                best = cand

        if best is None:
            break

        _, penalty, release, li, new_i = best
        old_i = cur_idx[li]
        cur_idx[li] = new_i
        cur_total -= release
        alloc_trace.append(
            {
                "step": step,
                "layer": li,
                "from_candidate": old_i,
                "to_candidate": new_i,
                "released_cost": int(release),
                "nll_penalty": float(penalty),
                "total_cost_after": int(cur_total),
            }
        )
        if int(verbose) == 1:
            print(
                f"[alloc-greedy] step={step} layer={li:02d} {old_i}->{new_i} "
                f"release={release} penalty={penalty:.6f} total={cur_total}",
                flush=True,
            )
        step += 1
    return cur_idx, cur_total, alloc_trace


# ---------------------------------- main ---------------------------------- #

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_prefix", type=str, default="eigen_basis_opt30b_lossaware_stable")

    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=1536)
    ap.add_argument("--token_budget", type=int, default=1_200_000)
    ap.add_argument("--max_data_files", type=int, default=300)
    ap.add_argument("--max_data_lines", type=int, default=3_000_000)
    ap.add_argument("--avg_dim", type=int, default=32)

    ap.add_argument("--svd_threshold_init", type=float, default=1.0)
    ap.add_argument("--svd_threshold_floor", type=float, default=0.30)
    ap.add_argument("--threshold_step", type=float, default=0.04)
    ap.add_argument("--min_rank", type=int, default=64)
    ap.add_argument("--strict_original_rank", type=int, default=1, choices=[0, 1])

    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="'', single, auto, balanced, sequential, or a JSON dict string",
    )
    ap.add_argument("--max_memory_json", type=str, default="")
    ap.add_argument("--per_gpu_max_memory_gib", type=float, default=72.0)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tokenizer", type=str, default="")
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1])

    ap.add_argument("--kv_target_gb", type=float, default=2.10)
    ap.add_argument("--kv_baseline_gb", type=float, default=2.625)
    ap.add_argument("--max_candidates_per_layer", type=int, default=6)
    ap.add_argument("--nll_eval_batches", type=int, default=4)
    ap.add_argument("--nll_eval_seqlen", type=int, default=512)
    ap.add_argument("--alloc_method", type=str, default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam_width", type=int, default=2)
    ap.add_argument("--nan_penalty", type=float, default=1e9)
    ap.add_argument("--skip_nan_candidates", type=int, default=0, choices=[0, 1])
    ap.add_argument(
        "--multi_shard_nll_mode",
        type=str,
        default="proxy",
        choices=["proxy", "real"],
        help="多卡分片时 NLL 打分模式：proxy=用 layer_err 代理（更稳），real=真实单层替换 NLL（更激进）",
    )

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    script_dir = Path(__file__).resolve().parent
    candidate_roots = [script_dir] + list(script_dir.parents)[:4]
    base_root = None
    for r in candidate_roots:
        p = r / "EigenAttn_base"
        if p.exists():
            base_root = p
            break
    if base_root is None:
        raise FileNotFoundError("Cannot find EigenAttn_base next to this script or in parent directory.")

    mod_utils = _load_py_module("eigen_base_utils_lossaware_stable", base_root / "decompose" / "eigen_attn_utils.py")
    mod_dm = _load_py_module("eigen_base_decompose_modules_lossaware_stable", base_root / "models" / "decompose_modules.py")
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
    dm_raw = (args.device_map or "").strip()
    dm = dm_raw.lower()

    max_memory = None
    mj = (args.max_memory_json or "").strip()
    if mj:
        max_memory = json.loads(mj)
        if not isinstance(max_memory, dict):
            raise ValueError("--max_memory_json must parse to a dict")
    elif float(args.per_gpu_max_memory_gib) > 0.0 and torch.cuda.is_available():
        n = int(torch.cuda.device_count())
        g = float(args.per_gpu_max_memory_gib)
        max_memory = {i: f"{g}GiB" for i in range(n)}

    load_kw: Dict[str, object] = dict(
        torch_dtype=dt,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if max_memory is not None:
        load_kw["max_memory"] = max_memory

    if not dm_raw or dm in ("single", "none"):
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": args.device}, **load_kw)
    elif dm in ("auto", "balanced", "sequential"):
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map=dm, **load_kw)
    elif dm_raw.startswith("{"):
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map=json.loads(dm_raw), **load_kw)
    else:
        raise ValueError(f"Unknown --device_map={args.device_map!r}")

    model.eval()
    model.config.use_cache = False

    hf_dm = getattr(model, "hf_device_map", None)
    if isinstance(hf_dm, dict) and hf_dm:
        items = list(hf_dm.items())[:6]
        print(f"[device_map] sharded modules={len(hf_dm)} preview={items}", flush=True)
    else:
        p0 = next(model.parameters())
        print(f"[device_map] single stack device={p0.device} dtype={p0.dtype}", flush=True)

    multi_cuda_devices = set()
    if isinstance(hf_dm, dict) and hf_dm:
        for dv in hf_dm.values():
            if isinstance(dv, int):
                multi_cuda_devices.add(int(dv))
            elif isinstance(dv, str) and dv.startswith("cuda:"):
                try:
                    multi_cuda_devices.add(int(dv.split(":", 1)[1]))
                except Exception:
                    pass
    multi_shard_mode = len(multi_cuda_devices) > 1
    if multi_shard_mode:
        print(f"[info] detected multi-shard mode across GPUs={sorted(multi_cuda_devices)}", flush=True)
        if args.multi_shard_nll_mode != "proxy":
            print("[warn] 多卡分片下强制切换 multi_shard_nll_mode=proxy（禁用真实候选 forward）。", flush=True)
            args.multi_shard_nll_mode = "proxy"
        print(
            "[warn] multi_shard proxy: 完全禁用候选替换 forward，改用纯谱代理分数（energy/rank）以规避 CUDA 非法访问。",
            flush=True,
        )

    if args.alloc_method == "beam" and multi_shard_mode:
        print("[warn] beam on sharded model is unsafe; forcing greedy allocation.", flush=True)
        args.alloc_method = "greedy"

    layers = model.model.decoder.layers
    n_layers = len(layers)
    num_heads = int(model.config.num_attention_heads)
    hidden = int(model.config.hidden_size)
    head_dim = hidden // num_heads

    print(
        f"[config] nsamples={args.nsamples} seqlen={args.seqlen} nll_batches={args.nll_eval_batches} "
        f"nll_seqlen={args.nll_eval_seqlen} target_kv_gb={args.kv_target_gb:.4f} alloc={args.alloc_method}",
        flush=True,
    )

    # token windows
    N = stream.numel()
    if N <= args.seqlen + 2:
        raise RuntimeError(f"token stream too short ({N}) for seqlen={args.seqlen}")
    g = torch.Generator().manual_seed(args.seed)

    calib_batches = []
    for _ in range(args.nsamples):
        s = int(torch.randint(low=0, high=N - args.seqlen - 1, size=(1,), generator=g).item())
        calib_batches.append(stream[s : s + args.seqlen].unsqueeze(0))

    nll_batches = []
    nll_len = int(args.nll_eval_seqlen)
    if N <= nll_len + 2:
        raise RuntimeError(f"token stream too short ({N}) for nll_eval_seqlen={nll_len}")
    for _ in range(int(args.nll_eval_batches)):
        s = int(torch.randint(low=0, high=N - nll_len - 1, size=(1,), generator=g).item())
        nll_batches.append(stream[s : s + nll_len].unsqueeze(0))

    # capture layer0 inputs once
    input_dev = _maybe_model_input_device(model)
    inps = torch.zeros((args.nsamples, args.seqlen, hidden), dtype=dt, device=input_dev)
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
    for b in calib_batches:
        if cache["i"] >= args.nsamples:
            break
        try:
            _ = model(input_ids=b.to(input_dev), use_cache=False, return_dict=True)
        except ValueError:
            pass
    layers[0] = layers[0].module
    attention_mask = cache.get("attention_mask", None)

    # baseline NLL
    use_proxy_nll = multi_shard_mode and args.multi_shard_nll_mode == "proxy"
    if use_proxy_nll:
        baseline_nll = float("nan")
        print("[loss-aware] baseline_nll=nan (proxy mode)", flush=True)
    else:
        baseline_nll_vals = []
        for b in nll_batches:
            bdev = b.to(input_dev, non_blocking=True)
            baseline_nll_vals.append(_lm_nll(model, bdev, None))
            del bdev
        if any((not math.isfinite(v)) for v in baseline_nll_vals):
            raise RuntimeError(f"baseline_nll contains invalid values: {baseline_nll_vals}")
        baseline_nll = float(sum(baseline_nll_vals) / len(baseline_nll_vals))
        print(f"[loss-aware] baseline_nll={baseline_nll:.6f}", flush=True)

    layer_candidates: List[List[Candidate]] = []
    report_layers = []
    basis_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    for li in range(n_layers):
        layer = layers[li]
        lay_dev = next(layer.parameters()).device
        layer_inps = inps.to(lay_dev, non_blocking=True)
        layer_attn_mask = attention_mask.to(lay_dev, non_blocking=True) if torch.is_tensor(attention_mask) else attention_mask

        decomp_args = SimpleNamespace(
            nsamples=int(args.nsamples),
            low_rank=False,
            eigen_attn_params={"threshold": float(args.svd_threshold_init), "avg_dim_features": int(args.avg_dim)},
        )

        w_dtype = layer.self_attn.k_proj.weight.dtype
        basis_kq_h_dev = None
        basis_v_h_dev = None
        try:
            basis_kq_h, eval_kq, basis_v_h, eval_v = decompose_opt_layer(layer, layer_inps, decomp_args, num_heads, li)
            if use_proxy_nll:
                # proxy path keeps basis on CPU to reduce device pressure/sync points
                basis_kq_h_cpu = basis_kq_h.detach().float().cpu()
                basis_v_h_cpu = basis_v_h.detach().float().cpu()
                basis_cache[li] = (basis_kq_h_cpu, basis_v_h_cpu)
            else:
                basis_kq_h_dev = basis_kq_h.to(device=lay_dev, dtype=w_dtype)
                basis_v_h_dev = basis_v_h.to(device=lay_dev, dtype=w_dtype)
                basis_cache[li] = (basis_kq_h_dev.detach().float().cpu(), basis_v_h_dev.detach().float().cpu())
        except RuntimeError as e:
            msg = str(e).lower()
            if use_proxy_nll and ("illegal memory access" in msg or "cuda error" in msg or "out of memory" in msg):
                print(
                    f"[warn][layer {li:02d}] decompose 触发 CUDA 异常，回退到单位基+均匀谱代理以继续运行。",
                    flush=True,
                )
                cleanup_memory()
                eye = torch.eye(head_dim, dtype=torch.float32).unsqueeze(0).repeat(num_heads, 1, 1)
                basis_kq_h_cpu = eye.clone()
                basis_v_h_cpu = eye.clone()
                basis_cache[li] = (basis_kq_h_cpu, basis_v_h_cpu)
                eval_kq = torch.full((num_heads, head_dim), 1.0 / float(head_dim), dtype=torch.float32)
                eval_v = torch.full((num_heads, head_dim), 1.0 / float(head_dim), dtype=torch.float32)
            else:
                raise

        cands: List[Candidate] = []
        seen = set()
        th = float(args.svd_threshold_init)
        while th >= float(args.svd_threshold_floor) - 1e-12:
            raw_kq = _rank_from_eval(eval_kq, th, num_heads)
            raw_v = _rank_from_eval(eval_v, th, num_heads)
            rkq = _finalize_rank(raw_kq, num_heads, head_dim, args.min_rank, bool(args.strict_original_rank))
            rv = _finalize_rank(raw_v, num_heads, head_dim, args.min_rank, bool(args.strict_original_rank))
            key = (rkq, rv)
            if key not in seen:
                seen.add(key)
                per_kq = _per_head_rank_from_eval(eval_kq, th)
                per_v = _per_head_rank_from_eval(eval_v, th)

                if use_proxy_nll:
                    # In multi-shard proxy mode we avoid ANY candidate replacement forward pass.
                    # Use pure spectral surrogate only.
                    ek = _energy_stats(eval_kq, per_kq)
                    ev = _energy_stats(eval_v, per_v)
                    lerr = _spectral_proxy_score(ek, ev, rkq, rv, hidden)
                else:
                    qlayer = OPTEigenAttnDecoderLayer(layer, decomp_args, basis_kq_h_dev, rkq, basis_v_h_dev, rv, model.config).to(device=lay_dev, dtype=dt)
                    try:
                        lerr = _relative_layer_error_streaming(layer, qlayer, layer_inps, layer_attn_mask, int(args.nsamples))
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            lerr = float("nan")
                            cleanup_memory()
                        else:
                            raise
                cand = Candidate(
                    layer=li,
                    threshold=float(th),
                    rank_kq=int(rkq),
                    rank_v=int(rv),
                    per_head_kq=per_kq,
                    per_head_v=per_v,
                    energy_kq=_energy_stats(eval_kq, per_kq),
                    energy_v=_energy_stats(eval_v, per_v),
                    layer_err=_safe_float(lerr, fallback=1e9),
                    nll_delta=0.0,
                )
                cands.append(cand)
                if not use_proxy_nll:
                    del qlayer
                cleanup_memory()
            th -= float(args.threshold_step)

        cands = sorted(cands, key=lambda x: (-x.cost, -x.threshold))
        cands = _downsample_candidates(cands, max(2, int(args.max_candidates_per_layer)))

        if int(args.verbose) == 1:
            print(
                f"[layer {li:02d}] candidates={len(cands)} cost_range=[{min(c.cost for c in cands)}, {max(c.cost for c in cands)}]",
                flush=True,
            )

        # candidate scoring: in multi-shard proxy mode we never run candidate forward.
        if use_proxy_nll:
            for ci, c in enumerate(cands):
                c.nll_delta = float(c.layer_err)
                if int(args.verbose) == 1:
                    print(
                        f"  [layer {li:02d}] cand#{ci} th={c.threshold:.4f} rkq={c.rank_kq} rv={c.rank_v} "
                        f"cost={c.cost} spectral_proxy={c.layer_err:.6f} proxy_nll_delta={c.nll_delta:.6f}",
                        flush=True,
                    )
        else:
            for ci, c in enumerate(cands):
                qlayer = OPTEigenAttnDecoderLayer(layer, decomp_args, basis_kq_h_dev, c.rank_kq, basis_v_h_dev, c.rank_v, model.config).to(device=lay_dev, dtype=dt)
                nd = _score_single_layer_candidate(
                    model=model,
                    layers=layers,
                    layer_idx=li,
                    replacement_layer=qlayer,
                    nll_batches=nll_batches,
                    input_dev=input_dev,
                    baseline_nll=baseline_nll,
                    multi_shard_mode=multi_shard_mode,
                )
                if not math.isfinite(nd):
                    nd = float(args.nan_penalty)
                c.nll_delta = float(nd)
                if int(args.verbose) == 1:
                    print(
                        f"  [layer {li:02d}] cand#{ci} th={c.threshold:.4f} rkq={c.rank_kq} rv={c.rank_v} "
                        f"cost={c.cost} lerr={c.layer_err:.6f} nll_delta={c.nll_delta:.6f}",
                        flush=True,
                    )
                del qlayer
                cleanup_memory()

        if int(args.skip_nan_candidates) == 1:
            valid_cands = [c for c in cands if math.isfinite(c.nll_delta) and c.nll_delta < float(args.nan_penalty)]
            if len(valid_cands) >= 2:
                cands = valid_cands

        layer_candidates.append(cands)
        report_layers.append(
            {
                "layer": li,
                "candidates": [
                    {
                        "threshold": c.threshold,
                        "rank_kq": c.rank_kq,
                        "rank_v": c.rank_v,
                        "cost": c.cost,
                        "layer_error": c.layer_err,
                        "nll_delta": c.nll_delta,
                        "energy_kq": c.energy_kq,
                        "energy_v": c.energy_v,
                    }
                    for c in cands
                ],
            }
        )

        del layer_inps
        cleanup_memory()

    full_cost = n_layers * (2 * hidden)
    target_total = int(round((float(args.kv_target_gb) / float(args.kv_baseline_gb)) * full_cost))
    print(f"[alloc] method={args.alloc_method} full_cost={full_cost} target_total={target_total}", flush=True)

    cur_idx, cur_total, alloc_trace = _allocate_greedy(
        layer_candidates=layer_candidates,
        n_layers=n_layers,
        target_total=target_total,
        verbose=int(args.verbose),
    )

    selected = [layer_candidates[i][cur_idx[i]] for i in range(n_layers)]
    final_total = sum(c.cost for c in selected)
    final_kv_est = float(args.kv_baseline_gb) * (float(final_total) / float(full_cost))
    print(f"[alloc] done total={final_total} est_kv_gb={final_kv_est:.6f}", flush=True)

    # export basis using cached headwise basis to avoid re-decomposition
    basis_dict: Dict[str, torch.Tensor] = {}
    for c in selected:
        bkq_cpu, bv_cpu = basis_cache[c.layer]
        U_kq = _pack_headwise_basis(bkq_cpu, int(c.rank_kq), num_heads, head_dim).float().cpu()
        U_v = _pack_headwise_basis(bv_cpu, int(c.rank_v), num_heads, head_dim).float().cpu()
        basis_dict[_basis_key(c.layer, "q")] = U_kq
        basis_dict[_basis_key(c.layer, "k")] = U_kq
        basis_dict[_basis_key(c.layer, "v")] = U_v

    out_pt = out_dir / f"{args.out_prefix}.pt"
    torch.save(basis_dict, out_pt)
    print(f"[saved] {out_pt}", flush=True)

    report = {
        "model": args.model,
        "mode": f"loss_aware_{args.alloc_method}_stable_single_layer_nll_candidates",
        "config": {
            "nsamples": int(args.nsamples),
            "seqlen": int(args.seqlen),
            "svd_threshold_init": float(args.svd_threshold_init),
            "svd_threshold_floor": float(args.svd_threshold_floor),
            "threshold_step": float(args.threshold_step),
            "strict_original_rank": int(args.strict_original_rank),
            "max_candidates_per_layer": int(args.max_candidates_per_layer),
            "nll_eval_batches": int(args.nll_eval_batches),
            "nll_eval_seqlen": int(args.nll_eval_seqlen),
            "kv_target_gb": float(args.kv_target_gb),
            "kv_baseline_gb": float(args.kv_baseline_gb),
            "alloc_method": str(args.alloc_method),
            "beam_width": int(args.beam_width),
            "nan_penalty": float(args.nan_penalty),
            "multi_shard_mode": bool(multi_shard_mode),
        },
        "baseline_nll": baseline_nll,
        "allocation": {
            "full_cost": int(full_cost),
            "target_total": int(target_total),
            "final_total": int(final_total),
            "final_kv_gb_est": float(final_kv_est),
            "final_combo_nll": None,
            "trace": alloc_trace,
        },
        "layers": report_layers,
        "selected": [
            {
                "layer": c.layer,
                "threshold": c.threshold,
                "rank_kq": c.rank_kq,
                "rank_v": c.rank_v,
                "cost": c.cost,
                "layer_error": c.layer_err,
                "nll_delta": c.nll_delta,
                "energy_kq": c.energy_kq,
                "energy_v": c.energy_v,
            }
            for c in selected
        ],
    }
    out_report = out_dir / f"{args.out_prefix}_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_report}", flush=True)


if __name__ == "__main__":
    main()
