#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss-aware / NLL-aware basis generator for OPT models.

Compared with build_eigen_basis_opt_error_budget_offline.py:
- does NOT overwrite the original script
- keeps headwise SVD basis extraction (K/Q shared, V separate)
- builds threshold-based rank candidates per layer
- scores candidates with REAL end-to-end NLL deltas (single-layer substitution)
- performs global greedy rank allocation under a target KV budget
- exports basis artifact (.pt) + detailed report (.json)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
except Exception:  # pragma: no cover
    AlignDevicesHook = None
    add_hook_to_module = None


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


def _pack_headwise_basis(basis_headwise: torch.Tensor, rank_total: int, num_heads: int, head_dim: int) -> torch.Tensor:
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
    return {"mean": float(t.mean().item()), "min": float(t.min().item()), "max": float(t.max().item())}


def _lm_nll(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> float:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = out.logits[:, :-1, :].float().contiguous()
    labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")
    return float(loss.item())


def _attach_like_accelerate_hook(src_module: torch.nn.Module, dst_module: torch.nn.Module) -> bool:
    """
    For sharded models (device_map=auto/balanced), cloned replacement layers must carry
    an equivalent accelerate AlignDevicesHook, otherwise cross-device transfer logic is lost.
    """
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


def _combo_nll(
    model,
    layers,
    nll_batches: List[torch.Tensor],
    input_dev: torch.device,
    layer_alt_modules: List[List[torch.nn.Module | None]],
    combo_idx: List[int],
) -> float:
    replaced: List[Tuple[int, torch.nn.Module]] = []
    try:
        for li, ci in enumerate(combo_idx):
            if ci <= 0:
                continue
            qlayer = layer_alt_modules[li][ci]
            if qlayer is None:
                continue
            replaced.append((li, layers[li]))
            layers[li] = qlayer

        vals = []
        for b in nll_batches:
            bdev = b.to(input_dev)
            vals.append(_lm_nll(model, bdev, None))
        return float(sum(vals) / max(1, len(vals)))
    finally:
        for li, ori in replaced:
            layers[li] = ori


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
            penalty = max(0.0, nxt.nll_delta - cur.nll_delta)
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
                f"[alloc-greedy] step={step} layer={li:02d} {old_i}->{new_i} release={release} "
                f"penalty={penalty:.6f} total={cur_total}",
                flush=True,
            )
        step += 1
    return cur_idx, cur_total, alloc_trace


def _allocate_beam_search(
    model,
    layers,
    layer_candidates: List[List[Candidate]],
    layer_alt_modules: List[List[torch.nn.Module | None]],
    nll_batches: List[torch.Tensor],
    input_dev: torch.device,
    target_total: int,
    beam_width: int,
    verbose: int,
):
    n_layers = len(layer_candidates)
    min_cost = [min(c.cost for c in cands) for cands in layer_candidates]
    suffix_min = [0] * (n_layers + 1)
    for i in range(n_layers - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + min_cost[i]

    # each state: (combo_idx, used_cost, nll)
    beam: List[Tuple[List[int], int, float]] = [([], 0, 0.0)]
    for li in range(n_layers):
        expanded: List[Tuple[List[int], int, float]] = []
        for prefix_idx, prefix_cost, _ in beam:
            for ci, cand in enumerate(layer_candidates[li]):
                new_cost = prefix_cost + cand.cost
                # prune states that can never satisfy budget
                if new_cost + suffix_min[li + 1] > target_total:
                    continue
                combo = prefix_idx + [ci] + [0] * (n_layers - li - 1)
                nll = _combo_nll(model, layers, nll_batches, input_dev, layer_alt_modules, combo)
                expanded.append((prefix_idx + [ci], new_cost, nll))

        if not expanded:
            # fallback: keep one feasible-by-min-cost route (may exceed target eventually)
            for prefix_idx, prefix_cost, _ in beam:
                ci = len(layer_candidates[li]) - 1
                cand = layer_candidates[li][ci]
                combo = prefix_idx + [ci] + [0] * (n_layers - li - 1)
                nll = _combo_nll(model, layers, nll_batches, input_dev, layer_alt_modules, combo)
                expanded.append((prefix_idx + [ci], prefix_cost + cand.cost, nll))

        expanded.sort(key=lambda x: (x[2], x[1]))  # lower nll then lower cost
        beam = expanded[: max(1, int(beam_width))]
        if int(verbose) == 1:
            print(
                f"[alloc-beam] layer={li:02d} explored={len(expanded)} keep={len(beam)} "
                f"best_nll={beam[0][2]:.6f} best_cost={beam[0][1]}",
                flush=True,
            )

    feasible = [st for st in beam if st[1] <= target_total]
    if feasible:
        feasible.sort(key=lambda x: (x[2], x[1]))
        best = feasible[0]
    else:
        beam.sort(key=lambda x: (x[1], x[2]))
        best = beam[0]

    alloc_trace = [
        {
            "method": "beam",
            "beam_width": int(beam_width),
            "best_nll": float(best[2]),
            "best_cost": int(best[1]),
            "target_total": int(target_total),
            "feasible_found": bool(len(feasible) > 0),
        }
    ]
    return best[0], best[1], alloc_trace, float(best[2])


def _downsample_candidates(cands: List[Candidate], keep_n: int) -> List[Candidate]:
    if len(cands) <= keep_n:
        return cands
    # cands sorted by descending threshold; keep endpoints and roughly uniform interior
    idx = sorted(set([0, len(cands) - 1] + [int(round(i * (len(cands) - 1) / (keep_n - 1))) for i in range(keep_n)]))
    return [cands[i] for i in idx]


@torch.no_grad()
def _relative_layer_error_streaming(
    layer,
    qlayer,
    inps: torch.Tensor,
    attention_mask: torch.Tensor | None,
    nsamples: int,
) -> float:
    """
    Compute ||Y_hr - Y_lr|| / ||Y_hr|| without stacking all sample outputs,
    to avoid peak VRAM spikes on large models.
    """
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
    return float(math.sqrt(num_sq / den_sq))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_prefix", type=str, default="eigen_basis_opt30b_lossaware")
    ap.add_argument("--nsamples", type=int, default=128, help="calibration samples for SVD/error")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--token_budget", type=int, default=1_500_000)
    ap.add_argument("--max_data_files", type=int, default=300)
    ap.add_argument("--max_data_lines", type=int, default=3_000_000)
    ap.add_argument("--avg_dim", type=int, default=32)
    ap.add_argument("--svd_threshold_init", type=float, default=1.0)
    ap.add_argument("--svd_threshold_floor", type=float, default=0.30)
    ap.add_argument("--threshold_step", type=float, default=0.02)
    ap.add_argument("--min_rank", type=int, default=64)
    ap.add_argument("--strict_original_rank", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--device_map",
        type=str,
        default="",
        help=(
            "模型分片方式: 空或 single => 整模在 --device 单卡; "
            "auto / balanced / sequential => HuggingFace accelerate 多卡切分（需配合 CUDA_VISIBLE_DEVICES）; "
            "若以 { 开头则按 JSON 解析为 device_map 字典"
        ),
    )
    ap.add_argument(
        "--max_memory_json",
        type=str,
        default="",
        help='可选，传给 from_pretrained(max_memory=...)，例如 {"0":"18GiB","1":"18GiB","2":"18GiB","3":"18GiB"}',
    )
    ap.add_argument(
        "--per_gpu_max_memory_gib",
        type=float,
        default=0.0,
        help="若 >0 且未设 --max_memory_json，则为当前可见的每张 CUDA 卡设置 max_memory[i]=f'{值}GiB'（4 卡跑 30B 时常用）",
    )
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tokenizer", type=str, default="")
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1])

    # loss-aware controls
    ap.add_argument("--kv_target_gb", type=float, default=2.10, help="target KV cache GB for global allocation")
    ap.add_argument("--kv_baseline_gb", type=float, default=2.625)
    ap.add_argument("--max_candidates_per_layer", type=int, default=6)
    ap.add_argument("--nll_eval_batches", type=int, default=2, help="number of eval batches for NLL scoring")
    ap.add_argument("--nll_eval_seqlen", type=int, default=1024)
    ap.add_argument("--alloc_method", type=str, default="beam", choices=["greedy", "beam"])
    ap.add_argument("--beam_width", type=int, default=4)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    base_root = repo_root.parent / "EigenAttn_base"
    if not base_root.exists():
        raise FileNotFoundError(f"Cannot find EigenAttn_base at: {base_root}")

    mod_utils = _load_py_module("eigen_base_utils_lossaware", base_root / "decompose" / "eigen_attn_utils.py")
    mod_dm = _load_py_module("eigen_base_decompose_modules_lossaware", base_root / "models" / "decompose_modules.py")
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
            raise ValueError("--max_memory_json 必须解析为 JSON 对象（字典）")
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
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": args.device},
            **load_kw,
        )
    elif dm in ("auto", "balanced", "sequential"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=dm,
            **load_kw,
        )
    elif dm_raw.startswith("{"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=json.loads(dm_raw),
            **load_kw,
        )
    else:
        raise ValueError(
            f"未知 --device_map={args.device_map!r}；请用空/single、auto、balanced、sequential 或 JSON 字典"
        )

    model.eval()
    model.config.use_cache = False

    hf_dm = getattr(model, "hf_device_map", None)
    if isinstance(hf_dm, dict) and hf_dm:
        head = list(hf_dm.items())[:6]
        print(
            f"[device_map] accelerate 分片: {len(hf_dm)} 个模块项, 前若干项={head}{' ...' if len(hf_dm) > 6 else ''}",
            flush=True,
        )
    else:
        p0 = next(model.parameters())
        print(f"[device_map] 单栈: 首参数 device={p0.device} dtype={p0.dtype}", flush=True)

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
        print("[info] 检测到多卡分片，候选替换层将尝试复制 accelerate hook。", flush=True)

    layers = model.model.decoder.layers
    n_layers = len(layers)
    num_heads = int(model.config.num_attention_heads)
    hidden = int(model.config.hidden_size)
    head_dim = hidden // num_heads

    print(
        f"[config] nsamples={args.nsamples} seqlen={args.seqlen} th_init={args.svd_threshold_init:.4f} "
        f"floor={args.svd_threshold_floor:.4f} step={args.threshold_step:.4f} strict_original={args.strict_original_rank} "
        f"target_kv_gb={args.kv_target_gb:.4f}",
        flush=True,
    )

    # build calibration batches for layerwise decomposition
    N = stream.numel()
    if N <= args.seqlen + 2:
        raise RuntimeError(f"token stream too short ({N}) for seqlen={args.seqlen}")
    g = torch.Generator().manual_seed(args.seed)
    calib_batches = []
    for _ in range(args.nsamples):
        s = int(torch.randint(low=0, high=N - args.seqlen - 1, size=(1,), generator=g).item())
        calib_batches.append(stream[s : s + args.seqlen].unsqueeze(0))

    # build eval batches for loss-aware scoring
    nll_batches = []
    nll_len = int(args.nll_eval_seqlen)
    if N <= nll_len + 2:
        raise RuntimeError(f"token stream too short ({N}) for nll_eval_seqlen={nll_len}")
    for _ in range(int(args.nll_eval_batches)):
        s = int(torch.randint(low=0, high=N - nll_len - 1, size=(1,), generator=g).item())
        nll_batches.append(stream[s : s + nll_len].unsqueeze(0))

    # capture layer0 inps for decompose workflow
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
    for b in calib_batches:
        if cache["i"] >= args.nsamples:
            break
        try:
            _ = model(input_ids=b.to(dev), use_cache=False, return_dict=True)
        except ValueError:
            pass
    layers[0] = layers[0].module
    attention_mask = cache.get("attention_mask", None)

    # baseline NLL for loss-aware scoring
    baseline_nll_vals = []
    for b in nll_batches:
        bdev = b.to(dev)
        baseline_nll_vals.append(_lm_nll(model, bdev, None))
    baseline_nll = float(sum(baseline_nll_vals) / max(1, len(baseline_nll_vals)))
    print(f"[loss-aware] baseline_nll={baseline_nll:.6f} (batches={len(nll_batches)})", flush=True)

    # collect layer candidates
    layer_candidates: List[List[Candidate]] = []
    layer_alt_modules: List[List[torch.nn.Module | None]] = []
    keep_alt_modules_for_combo = (args.alloc_method == "beam") and (not multi_shard_mode)
    if args.alloc_method == "beam" and multi_shard_mode:
        print(
            "[warn] 多卡分片下为避免候选模块累计导致 CUDA 非法访问，"
            "将禁用跨层组合缓存并回退到 greedy 分配（仍保留单层真实 NLL 打分）。",
            flush=True,
        )
    report_layers = []

    for li in range(n_layers):
        layer = layers[li]
        lay_dev = next(layer.parameters()).device
        inps = inps.to(lay_dev)
        if attention_mask is not None and torch.is_tensor(attention_mask):
            attention_mask = attention_mask.to(lay_dev)

        decomp_args = SimpleNamespace(
            nsamples=int(args.nsamples),
            low_rank=False,
            eigen_attn_params={"threshold": float(args.svd_threshold_init), "avg_dim_features": int(args.avg_dim)},
        )
        basis_kq_h, eval_kq, basis_v_h, eval_v = decompose_opt_layer(layer, inps, decomp_args, num_heads, li)
        w_dtype = layer.self_attn.k_proj.weight.dtype
        basis_kq_h = basis_kq_h.to(device=lay_dev, dtype=w_dtype)
        basis_v_h = basis_v_h.to(device=lay_dev, dtype=w_dtype)

        # enumerate unique candidate ranks over thresholds
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
                qlayer = OPTEigenAttnDecoderLayer(
                    layer, decomp_args, basis_kq_h, rkq, basis_v_h, rv, model.config
                ).to(device=lay_dev, dtype=dt)
                lerr = _relative_layer_error_streaming(
                    layer=layer,
                    qlayer=qlayer,
                    inps=inps,
                    attention_mask=attention_mask,
                    nsamples=int(args.nsamples),
                )
                per_kq = _per_head_rank_from_eval(eval_kq, th)
                per_v = _per_head_rank_from_eval(eval_v, th)
                cands.append(
                    Candidate(
                        layer=li,
                        threshold=float(th),
                        rank_kq=int(rkq),
                        rank_v=int(rv),
                        per_head_kq=per_kq,
                        per_head_v=per_v,
                        energy_kq=_energy_stats(eval_kq, per_kq),
                        energy_v=_energy_stats(eval_v, per_v),
                        layer_err=lerr,
                    )
                )
                del qlayer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            th -= float(args.threshold_step)

        # sort from high-cost to low-cost, keep limited points
        cands = sorted(cands, key=lambda x: (-x.cost, -x.threshold))
        cands = _downsample_candidates(cands, max(2, int(args.max_candidates_per_layer)))

        if int(args.verbose) == 1:
            print(
                f"[layer {li:02d}] candidates={len(cands)} "
                f"cost_range=[{min(c.cost for c in cands)}, {max(c.cost for c in cands)}]",
                flush=True,
            )
            for ci, c in enumerate(cands):
                print(
                    f"  cand#{ci} th={c.threshold:.4f} rkq={c.rank_kq} rv={c.rank_v} "
                    f"lerr={c.layer_err:.6f} energy_kq={c.energy_kq['mean']:.4f} energy_v={c.energy_v['mean']:.4f}",
                    flush=True,
                )

        # NLL scoring per candidate (single-layer substitution)
        layer_dev_for_nll = next(layer.parameters()).device
        alt_modules_for_layer: List[torch.nn.Module | None] = [None for _ in range(len(cands))]
        for ci, c in enumerate(cands):
            qlayer = OPTEigenAttnDecoderLayer(
                layer, decomp_args, basis_kq_h, c.rank_kq, basis_v_h, c.rank_v, model.config
            ).to(device=layer_dev_for_nll, dtype=dt)
            if multi_shard_mode and (not _attach_like_accelerate_hook(layer, qlayer)):
                raise RuntimeError(
                    f"layer {li} 候选模块无法复制 accelerate hook，无法在多卡分片下做真实 NLL 打分。"
                )
            if keep_alt_modules_for_combo:
                alt_modules_for_layer[ci] = qlayer

            orig = layers[li]
            try:
                layers[li] = qlayer
                nll_vals = []
                for b in nll_batches:
                    bdev = b.to(dev)
                    nll_vals.append(_lm_nll(model, bdev, None))
                cand_nll = float(sum(nll_vals) / max(1, len(nll_vals)))
                c.nll_delta = float(cand_nll - baseline_nll)
            finally:
                layers[li] = orig

            if int(args.verbose) == 1:
                print(
                    f"  [layer {li:02d}] NLL cand#{ci} cost={c.cost} nll={cand_nll:.6f} delta={c.nll_delta:.6f}",
                    flush=True,
                )
            if not keep_alt_modules_for_combo:
                del qlayer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # IMPORTANT: keep original device_map placement for end-to-end NLL scoring.
        # Moving processed layers to CPU here will break later full-model forward
        # with device mismatch (some layers on CPU, inputs on CUDA).

        layer_candidates.append(cands)
        layer_alt_modules.append(alt_modules_for_layer)
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

    # global allocation under KV budget
    full_cost = n_layers * (2 * hidden)
    target_total = int(round((float(args.kv_target_gb) / float(args.kv_baseline_gb)) * full_cost))

    print(
        f"[alloc] method={args.alloc_method} full_cost={full_cost} target_total={target_total} "
        f"target_kv_gb={args.kv_target_gb:.4f}",
        flush=True,
    )
    final_combo_nll = None
    use_beam_combo = (args.alloc_method == "beam") and keep_alt_modules_for_combo
    if use_beam_combo:
        cur_idx, cur_total, alloc_trace, final_combo_nll = _allocate_beam_search(
            model=model,
            layers=layers,
            layer_candidates=layer_candidates,
            layer_alt_modules=layer_alt_modules,
            nll_batches=nll_batches,
            input_dev=dev,
            target_total=target_total,
            beam_width=int(args.beam_width),
            verbose=int(args.verbose),
        )
    else:
        if args.alloc_method == "beam" and (not keep_alt_modules_for_combo):
            print("[alloc] 已自动回退到 greedy（多卡稳态模式）。", flush=True)
        cur_idx, cur_total, alloc_trace = _allocate_greedy(
            layer_candidates=layer_candidates,
            n_layers=n_layers,
            target_total=target_total,
            verbose=int(args.verbose),
        )

    selected = [layer_candidates[i][cur_idx[i]] for i in range(n_layers)]
    final_total = sum(c.cost for c in selected)
    final_kv_est = float(args.kv_baseline_gb) * (float(final_total) / float(full_cost))
    print(
        f"[alloc] done total={final_total} est_kv_gb={final_kv_est:.6f} "
        f"(target={args.kv_target_gb:.6f})",
        flush=True,
    )

    # export basis
    basis_dict: Dict[str, torch.Tensor] = {}
    for c in selected:
        li = c.layer
        # find layer-local cached basis tensors from candidates source
        # reuse first candidate carrying same layer (all candidates share same headwise basis matrices)
        # We reconstruct from decompose again to keep code simple and deterministic.
        # This is cheap relative to original run and avoids storing huge tensors in candidate objects.
        layer = model.model.decoder.layers[li]
        lay_dev = next(layer.parameters()).device
        inps = inps.to(lay_dev)
        decomp_args = SimpleNamespace(
            nsamples=int(args.nsamples),
            low_rank=False,
            eigen_attn_params={"threshold": float(args.svd_threshold_init), "avg_dim_features": int(args.avg_dim)},
        )
        basis_kq_h, _, basis_v_h, _ = decompose_opt_layer(layer, inps, decomp_args, num_heads, li)
        basis_kq_h = basis_kq_h.float().cpu()
        basis_v_h = basis_v_h.float().cpu()
        U_kq = _pack_headwise_basis(basis_kq_h, int(c.rank_kq), num_heads, head_dim).cpu().float()
        U_v = _pack_headwise_basis(basis_v_h, int(c.rank_v), num_heads, head_dim).cpu().float()
        basis_dict[_basis_key(li, "q")] = U_kq
        basis_dict[_basis_key(li, "k")] = U_kq
        basis_dict[_basis_key(li, "v")] = U_v

    out_pt = out_dir / f"{args.out_prefix}.pt"
    torch.save(basis_dict, out_pt)
    print(f"[saved] {out_pt}", flush=True)

    report = {
        "model": args.model,
        "mode": f"loss_aware_{args.alloc_method}_single_layer_nll_candidates",
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
        },
        "baseline_nll": baseline_nll,
        "allocation": {
            "full_cost": int(full_cost),
            "target_total": int(target_total),
            "final_total": int(final_total),
            "final_kv_gb_est": float(final_kv_est),
            "final_combo_nll": None if final_combo_nll is None else float(final_combo_nll),
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

