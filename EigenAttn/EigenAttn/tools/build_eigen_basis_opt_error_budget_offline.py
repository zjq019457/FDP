#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 EigenAttn 仓库内复现「原始 EigenAttn 的激活驱动 + error-budget 选秩」并导出 basis artifact（eigen_basis.pt）。

主要特性：
1) 先用 calibration 样本抓 layer0 输入 inps，再逐层串行处理；
2) 每层从 q/k/v 激活做 SVD（Q/K 共享子空间，V 独立）；
3) 按 error_budget 迭代 threshold（每步 -0.02，退出后回退一步）确定 rank；
   支持 ``--per_layer_eb_json``：每层使用独立 EB（见 per_layer_eb_from_step1.py），
   导出单个 ``{out_prefix}_perlayer_eb.pt`` 与 ``{out_prefix}_perlayer_eb_report.json``。
4) 导出可复用的 basis dict（key: ...q_proj_basis / k_proj_basis / v_proj_basis）。

相对你原脚本的关键修正：
- 修正 rank_from_eval 的 off-by-one；
- 去掉 while 条件中由 min_rank 导致的提前停止；
- 默认 threshold_init=1.0，更贴原论文；
- 默认 nsamples=512，更贴原论文；
- 增加 threshold_trace 到 report，方便调试。

说明：
- 不修改 EigenAttn_base 源码；本脚本动态加载其分解函数与 OPT 低秩层实现。
- 输出的是 basis artifact，而不是最终部署后的低秩权重模型。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    # 支持 c4 风格的 .json.gz/.jsonl/.txt 以及 wikitext 风格的 .parquet
    exts = ["*.txt", "*.jsonl", "*.json", "*.gz", "*.parquet"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(str(Path(data_dir) / "**" / e), recursive=True))

    # 跳过 eval split（parquet 目录下会同时存在 train/validation/test 分片）。
    # 约定：文件名以 test-/validation-/val- 开头视为 eval，仅用 train-* 或其它作 calibration。
    _skip_prefixes = ("test-", "validation-", "val-")
    files = [f for f in files if not any(Path(f).name.startswith(p) for p in _skip_prefixes)]
    files = sorted(files)[: max(1, int(max_files))]

    n = 0
    for fp in files:
        # NEW: parquet 支持（wikitext-2/103-raw 的官方 parquet 布局）
        if fp.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq  # type: ignore
                tbl = pq.read_table(fp, columns=["text"])
                for txt in tbl["text"].to_pylist():
                    if isinstance(txt, str) and txt.strip():
                        yield txt.strip()
                        n += 1
                        if n >= max_lines:
                            return
            except Exception:
                pass
            continue

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


# Held so SIGTERM/SIGINT can drop the HF model and release CUDA allocations promptly.
_GPU_MODEL_HOLDER: Dict[str, object] = {}


def _cleanup_cuda_on_signal(signum: int, frame=None) -> None:
    """Release GPU memory when killed via kill/kill -TERM (nohup often logs wrong PID)."""
    print(f"[cleanup] caught signal {signum}; releasing GPU tensors...", flush=True)
    m = _GPU_MODEL_HOLDER.pop("model", None)
    if m is not None:
        try:
            del m
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    raise SystemExit(128 + signum)


def _basis_key(layer_idx: int, kind: str) -> str:
    return f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{kind}_proj_basis"


def _rank_from_eval(eval_per_head: torch.Tensor, threshold: float, num_heads: int) -> int:
    """
    eval_per_head: [num_heads, head_dim], each row sums to ~1
    返回 total rank = num_heads * max(per_head_rank)
    per_head_rank 定义为：最小的 r，使 cumulative_energy >= threshold
    """
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


def _per_head_energy_from_eval_and_ranks(eval_per_head: torch.Tensor, per_head_ranks: List[int]) -> Dict[str, float]:
    """
    根据每头的归一化谱(eval_per_head)与每头rank，计算保留能量统计。
    返回 mean/min/max（范围约在 [0,1]）。
    """
    e = eval_per_head.float().cpu()
    c = torch.cumsum(e, dim=1)
    vals: List[float] = []
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


def _relative_error_fp32(output_hr: torch.Tensor, output_lr: torch.Tensor, eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    Stable relative error:
      err = ||hr-lr||_2 / max(||hr||_2, eps)
    computed in fp32 to avoid fp16 underflow-to-zero artifacts.
    Returns (err, numerator, denominator).
    """
    diff = (output_hr.float() - output_lr.float())
    num = float(torch.norm(diff).item())
    den = float(torch.norm(output_hr.float()).item())
    err = num / max(den, eps)
    return float(err), num, den


def _relative_error_streaming_layers(
    layer_hr: torch.nn.Module,
    layer_lr: torch.nn.Module,
    inps: torch.Tensor,
    attention_mask,
    nsamples: int,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    与 _relative_error_fp32 相同度量，但按样本流式累加，避免 torch.stack 在多卡/长序列下顶爆显存或触发驱动异常。
    err = sqrt(sum_j ||hr_j-lr_j||_F^2) / max(sqrt(sum_j ||hr_j||_F^2), eps)
    """
    diff_sq = 0.0
    hr_sq = 0.0
    p0 = next(layer_hr.parameters())
    for j in range(int(nsamples)):
        x = inps[j].unsqueeze(0).to(
            device=p0.device, dtype=p0.dtype, non_blocking=True
        )
        am = attention_mask
        if am is not None and torch.is_tensor(am):
            am = am.to(device=x.device)
        y_hr = layer_hr(x, attention_mask=am)[0]
        y_lr = layer_lr(x, attention_mask=am)[0]
        dh = y_hr.float() - y_lr.float()
        diff_sq += float(torch.sum(dh * dh).item())
        hr_sq += float(torch.sum(y_hr.float() * y_hr.float()).item())
        del x, y_hr, y_lr, dh, am
    num = float(diff_sq**0.5)
    den = float(max(hr_sq**0.5, eps))
    err = num / den
    return float(err), num, den


def _finalize_rank(
    rank: int,
    num_heads: int,
    head_dim: int,
    min_rank: int,
    *,
    strict_original: bool = True,
) -> int:
    """
    保证总秩为 num_heads 的整数倍，并且每头至少 1 维。
    strict_original=True: 不主动抬到 min_rank，只保留 shape 安全约束。
    strict_original=False: 允许工程化抬秩到 min_rank。
    """
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
    """
    basis_headwise: [num_heads, head_dim, head_dim]
    输出 U: [hidden_size, rank_total]
    每个 head 取前 k=rank_total/num_heads 列，拼成 block-diagonal 形式。
    """
    if rank_total % num_heads != 0:
        raise ValueError(f"rank_total={rank_total} must be divisible by num_heads={num_heads}")

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


def _retarget_budget_ranks(
    layer_rows: List[dict],
    num_heads: int,
    head_dim: int,
    min_rank: int,
    hidden: int,
    target_kv_gb: float,
    baseline_kv_gb: float,
    strict_original: bool,
) -> Tuple[List[Tuple[int, int]], dict]:
    """
    在当前 error_budget 搜索结果基础上，按目标 KV GB 进行全局比例缩放。
    注意：这不是原始 EigenAttn 的论文逻辑，而是你当前 artifact 管线的工程补充。
    """
    full_total = len(layer_rows) * (2 * hidden)  # full K + full V
    cur_total = sum(int(x["rank_kq"]) + int(x["rank_v"]) for x in layer_rows)
    target_ratio = float(target_kv_gb) / float(baseline_kv_gb)
    target_total = int(round(target_ratio * full_total))

    if cur_total <= 0:
        return [(int(x["rank_kq"]), int(x["rank_v"])) for x in layer_rows], {
            "retarget_enabled": True,
            "cur_total": cur_total,
            "target_total": target_total,
            "actual_total": cur_total,
            "scale": 1.0,
            "target_kv_gb": target_kv_gb,
            "baseline_kv_gb": baseline_kv_gb,
            "cur_kv_gb_est": baseline_kv_gb * (cur_total / full_total),
            "actual_kv_gb_est": baseline_kv_gb * (cur_total / full_total),
        }

    scale = max(0.01, float(target_total) / float(cur_total))
    ranks: List[Tuple[int, int]] = []
    for row in layer_rows:
        nk = _finalize_rank(
            int(round(int(row["rank_kq"]) * scale)),
            num_heads,
            head_dim,
            min_rank,
            strict_original=strict_original,
        )
        nv = _finalize_rank(
            int(round(int(row["rank_v"]) * scale)),
            num_heads,
            head_dim,
            min_rank,
            strict_original=strict_original,
        )
        ranks.append((nk, nv))

    actual_total = sum(a + b for a, b in ranks)
    return ranks, {
        "retarget_enabled": True,
        "cur_total": cur_total,
        "target_total": target_total,
        "actual_total": actual_total,
        "scale": scale,
        "target_kv_gb": target_kv_gb,
        "baseline_kv_gb": baseline_kv_gb,
        "cur_kv_gb_est": baseline_kv_gb * (cur_total / full_total),
        "actual_kv_gb_est": baseline_kv_gb * (actual_total / full_total),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF OPT model path, e.g. /diskE/opt-30b")
    ap.add_argument("--data_dir", required=True, help="calibration text directory")
    ap.add_argument("--out_dir", required=True, help="output directory on disk")

    ap.add_argument(
        "--error_budgets",
        type=str,
        default="0.008,0.015,0.024",
        help="comma-separated budgets, e.g. 0.008,0.015,0.024",
    )
    ap.add_argument("--out_prefix", type=str, default="eigen_basis_opt30b")

    # 更贴论文默认值
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)

    ap.add_argument("--token_budget", type=int, default=1_500_000)
    ap.add_argument("--max_data_files", type=int, default=300)
    ap.add_argument("--max_data_lines", type=int, default=3_000_000)

    ap.add_argument("--avg_dim", type=int, default=32, help="avg_dim_features in original utils")
    ap.add_argument("--svd_threshold_init", type=float, default=1.0)
    ap.add_argument("--svd_threshold_floor", type=float, default=0.30)
    ap.add_argument("--threshold_step", type=float, default=0.02)

    # 注意：strict_original_rank=1 时，min_rank 不再主导搜索，只保留 shape 安全作用
    ap.add_argument("--min_rank", type=int, default=64)

    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--device", type=str, default="cuda:0", help="single-device mode")
    ap.add_argument(
        "--device_map",
        type=str,
        default="",
        help="empty: use --device ; auto: HF auto device map",
    )
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tokenizer", type=str, default="")

    ap.add_argument(
        "--kv_target_gb",
        type=float,
        default=0.0,
        help=">0: proportionally retarget layer ranks toward a target KV GB",
    )
    ap.add_argument(
        "--kv_baseline_gb",
        type=float,
        default=2.625,
        help="full-rank KV GB baseline (OPT-30B fp16, seqlen=2048, bs=1 often ~2.625)",
    )

    ap.add_argument(
        "--strict_original_rank",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: do not actively raise rank to min_rank; 0: allow engineering min_rank uplift",
    )
    ap.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1],
        help="1: detailed threshold-search logging",
    )
    ap.add_argument(
        "--report_energy",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: 在 report 和日志中输出每层 KQ/V 光谱能量保留统计",
    )
    ap.add_argument(
        "--paper_defaults",
        action="store_true",
        help="未在命令行显式传入时，使用 nsamples=512、seqlen=2048、error_budgets=0.008,0.015,0.024",
    )
    ap.add_argument(
        "--per_layer_eb_json",
        type=str,
        default="",
        help=(
            "来自 per_layer_eb_from_step1.py 的 JSON（需含 error_budget_by_layer，长度=层数）。"
            "设置后每层使用独立 EB，只导出单个 *_perlayer_eb.pt 与对应 report；忽略多值 --error_budgets。"
        ),
    )
    ap.add_argument(
        "--freeze_layer0_eb",
        action="store_true",
        help="仅在与 --per_layer_eb_json 同用时：第 0 层不用 JSON 里的 EB，改用 --layer0_eb（对齐均匀 EB024）。",
    )
    ap.add_argument(
        "--layer0_eb",
        type=float,
        default=0.024,
        help="与 --freeze_layer0_eb 配合，默认 0.024。",
    )
    ap.add_argument(
        "--no_skip_redundant_eval",
        action="store_true",
        help="禁用「finalize 后秩与当前 qlayer 相同则跳过 streaming 与重建」的加速",
    )
    ap.add_argument(
        "--low_vram",
        action="store_true",
        help=(
            "降低显存：校准激活 inps 常驻 CPU，按样本搬到 GPU；"
            "若未显式传入 --nsamples/--seqlen，则改用 nsamples=256、seqlen=1536（可再手动改更小）。"
        ),
    )
    args = ap.parse_args()

    argv_set = set(sys.argv)
    per_layer_eb_path = (args.per_layer_eb_json or "").strip()
    if args.freeze_layer0_eb and not per_layer_eb_path:
        print(
            "[warn] --freeze_layer0_eb only applies with --per_layer_eb_json; ignoring.",
            flush=True,
        )
    if args.paper_defaults:
        if "--nsamples" not in argv_set:
            args.nsamples = 512
        if "--seqlen" not in argv_set:
            args.seqlen = 2048
        if "--error_budgets" not in argv_set and not per_layer_eb_path:
            args.error_budgets = "0.008,0.015,0.024"

    if args.low_vram:
        if "--nsamples" not in argv_set:
            args.nsamples = 256
        if "--seqlen" not in argv_set:
            args.seqlen = 1536
        print(
            f"[low_vram] inps on CPU; nsamples={args.nsamples} seqlen={args.seqlen} "
            "(pass explicit --nsamples / --seqlen to override)",
            flush=True,
        )
        if args.paper_defaults and "--nsamples" not in argv_set and "--seqlen" not in argv_set:
            print(
                "[warn] --paper_defaults and --low_vram both set without explicit "
                "--nsamples/--seqlen: final size is low_vram default (256×1536), "
                "NOT paper (512×2048). Use e.g. --nsamples 512 --seqlen 2048 to match paper.",
                flush=True,
            )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(
        f"[gpu] Python PID={os.getpid()} — use THIS PID to free VRAM "
        f"(e.g. kill -TERM {os.getpid()}). "
        "If you started with nohup, $! is often the shell, not this process; "
        "prefer the PID shown by: nvidia-smi | grep python",
        flush=True,
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    repo_root = Path(__file__).resolve().parents[1]   # .../EigenAttn/EigenAttn
    base_root = repo_root.parent / "EigenAttn_base"
    if not base_root.exists():
        raise FileNotFoundError(f"Cannot find EigenAttn_base at: {base_root}")

    mod_utils = _load_py_module("eigen_base_utils", base_root / "decompose" / "eigen_attn_utils.py")
    mod_dm = _load_py_module("eigen_base_decompose_modules", base_root / "models" / "decompose_modules.py")
    decompose_opt_layer = mod_utils.decompose_opt_layer
    OPTEigenAttnDecoderLayer = mod_dm.OPTEigenAttnDecoderLayer

    if per_layer_eb_path:
        budgets = [0.0]  # 单槽位；真实 EB 在加载模型后从 JSON 读入
    else:
        budgets = [float(x.strip()) for x in args.error_budgets.split(",") if x.strip()]
        if not budgets:
            raise ValueError("No valid --error_budgets parsed")

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
            args.model,
            torch_dtype=dt,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dt,
            device_map={"": args.device},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

    model.eval()
    model.config.use_cache = False

    _GPU_MODEL_HOLDER["model"] = model
    signal.signal(signal.SIGTERM, _cleanup_cuda_on_signal)
    signal.signal(signal.SIGINT, _cleanup_cuda_on_signal)

    layers = model.model.decoder.layers
    n_layers = len(layers)
    num_heads = int(model.config.num_attention_heads)
    hidden = int(model.config.hidden_size)
    head_dim = hidden // num_heads

    eb_by_layer = None
    if per_layer_eb_path:
        pl_path = Path(per_layer_eb_path)
        if not pl_path.is_file():
            raise FileNotFoundError(f"--per_layer_eb_json not found: {pl_path}")
        with open(pl_path, "r", encoding="utf-8") as f:
            plj = json.load(f)
        if "error_budget_by_layer" not in plj:
            raise KeyError("JSON must contain key 'error_budget_by_layer'")
        eb_by_layer = [float(x) for x in plj["error_budget_by_layer"]]
        if len(eb_by_layer) != n_layers:
            raise ValueError(
                f"per_layer_eb_json has {len(eb_by_layer)} values, model has n_layers={n_layers}"
            )
        print(
            f"[per_layer_eb] loaded {pl_path}  EB min/mean/max="
            f"{min(eb_by_layer):.6f}/{sum(eb_by_layer)/len(eb_by_layer):.6f}/{max(eb_by_layer):.6f}",
            flush=True,
        )
        if args.freeze_layer0_eb:
            prev0 = float(eb_by_layer[0])
            eb_by_layer[0] = float(args.layer0_eb)
            print(
                f"[freeze_layer0_eb] layer0 EB {prev0:.6f} -> {eb_by_layer[0]:.6f}",
                flush=True,
            )

    # build nsamples random token chunks
    N = stream.numel()
    if N <= args.seqlen + 2:
        raise RuntimeError(f"token stream too short ({N}) for seqlen={args.seqlen}")

    g = torch.Generator().manual_seed(args.seed)
    batches = []
    for _ in range(args.nsamples):
        s = int(torch.randint(low=0, high=N - args.seqlen - 1, size=(1,), generator=g).item())
        x = stream[s : s + args.seqlen].unsqueeze(0)
        batches.append(x)

    # capture layer0 inputs
    dev = model.get_input_embeddings().weight.device
    _low_vram_inps = bool(args.low_vram)
    inps_dev = torch.device("cpu") if _low_vram_inps else dev
    inps = torch.zeros((args.nsamples, args.seqlen, hidden), dtype=dt, device=inps_dev)
    cache: Dict[str, object] = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if _low_vram_inps:
                inps[cache["i"]] = inp.detach().cpu().contiguous()
            else:
                inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for b in batches:
        if cache["i"] >= args.nsamples:
            break
        try:
            _ = model(input_ids=b.to(dev), use_cache=False, return_dict=True)
        except ValueError:
            pass
    layers[0] = layers[0].module

    attention_mask = cache.get("attention_mask", None)

    basis_dicts: List[Dict[str, torch.Tensor]] = [dict() for _ in budgets]
    layer_rows_per_budget: List[List[dict]] = [[] for _ in budgets]

    report = {
        "model": args.model,
        "budgets": eb_by_layer if eb_by_layer is not None else budgets,
        "mode": {
            "repo_base_behavior": True,
            "strict_original_rank": bool(args.strict_original_rank),
            "retarget_enabled": float(args.kv_target_gb) > 0,
            "artifact_only_basis_export": True,
            "per_layer_eb": eb_by_layer is not None,
            "low_vram": bool(args.low_vram),
            "freeze_layer0_eb": bool(getattr(args, "freeze_layer0_eb", False)),
            "layer0_eb": float(args.layer0_eb) if eb_by_layer is not None else None,
        },
        "search_config": {
            "svd_threshold_init": float(args.svd_threshold_init),
            "svd_threshold_floor": float(args.svd_threshold_floor),
            "threshold_step": float(args.threshold_step),
            "min_rank": int(args.min_rank),
            "nsamples": int(args.nsamples),
            "seqlen": int(args.seqlen),
        },
        "layers": [],
    }
    if eb_by_layer is not None:
        report["per_layer_eb_json"] = str(Path(per_layer_eb_path).resolve())

    print(
        f"[config] th_init={float(args.svd_threshold_init):.4f} "
        f"th_floor={float(args.svd_threshold_floor):.4f} "
        f"step={float(args.threshold_step):.4f} "
        f"min_rank={int(args.min_rank)} "
        f"strict_original_rank={int(args.strict_original_rank)} "
        f"nsamples={int(args.nsamples)} "
        f"retarget={'on' if float(args.kv_target_gb) > 0 else 'off'} "
        f"low_vram={'on' if args.low_vram else 'off'}",
        flush=True,
    )

    for li in range(n_layers):
        layer = layers[li]
        lay_dev = next(layer.parameters()).device
        if torch.cuda.is_available() and lay_dev.type == "cuda":
            torch.cuda.set_device(lay_dev)
        if not args.low_vram:
            inps = inps.to(lay_dev)

        if attention_mask is not None and torch.is_tensor(attention_mask):
            attention_mask = attention_mask.to(lay_dev)

        decomp_args = SimpleNamespace(
            nsamples=int(args.nsamples),
            low_rank=False,
            eigen_attn_params={
                "threshold": float(args.svd_threshold_init),
                "avg_dim_features": int(args.avg_dim),
            },
        )

        basis_kq_h, eval_kq, basis_v_h, eval_v = decompose_opt_layer(
            layer, inps, decomp_args, num_heads, li, attention_mask=attention_mask
        )

        # basis 保持在 CPU；OPTEigenAttention 在 CPU 上做 basis×权重，并对原投影权重整矩阵一次 D2H 再切片。
        basis_kq_h = basis_kq_h.detach().cpu().contiguous().float()
        basis_v_h = basis_v_h.detach().cpu().contiguous().float()

        layer_info = {"layer": li, "runs": []}

        if eb_by_layer is not None:
            eb_iter = [(0, float(eb_by_layer[li]))]
        else:
            eb_iter = list(enumerate(budgets))
        for bi, eb in eb_iter:
            th = float(args.svd_threshold_init)
            err = 0.0
            iter_idx = 0
            trace = []

            raw_kq = _rank_from_eval(eval_kq, th, num_heads)
            raw_v = _rank_from_eval(eval_v, th, num_heads)

            rank_kq = _finalize_rank(
                raw_kq,
                num_heads,
                head_dim,
                args.min_rank,
                strict_original=bool(args.strict_original_rank),
            )
            rank_v = _finalize_rank(
                raw_v,
                num_heads,
                head_dim,
                args.min_rank,
                strict_original=bool(args.strict_original_rank),
            )

            if int(args.verbose) == 1:
                print(
                    f"[layer {li:02d}][eb={eb}] init th={th:.4f} raw_kq={raw_kq} raw_v={raw_v} "
                    f"rank_kq={rank_kq} rank_v={rank_v}",
                    flush=True,
                )

            # 按当前 threshold 构造低秩层
            qlayer = OPTEigenAttnDecoderLayer(
                layer, decomp_args, basis_kq_h, rank_kq, basis_v_h, rank_v, model.config
            ).to(device=lay_dev, dtype=dt)

            # 显式保存“最后一个满足预算(err <= eb)的配置”
            best_th = float(th)
            best_rank_kq = int(rank_kq)
            best_rank_v = int(rank_v)
            best_err = None

            skip_stream = False
            # 更贴原文：只让 err 和 threshold_floor 决定是否继续压
            while (
                err < float(eb)
                and th > float(args.svd_threshold_floor)
            ):
                reused_err = skip_stream
                if not skip_stream:
                    err, err_num, err_den = _relative_error_streaming_layers(
                        layer, qlayer, inps, attention_mask, int(args.nsamples)
                    )
                    if torch.cuda.is_available() and lay_dev.type == "cuda":
                        torch.cuda.synchronize(lay_dev)
                skip_stream = False

                trace.append(
                    {
                        "iter": int(iter_idx),
                        "threshold": float(th),
                        "raw_kq": int(raw_kq),
                        "raw_v": int(raw_v),
                        "rank_kq": int(rank_kq),
                        "rank_v": int(rank_v),
                        "err": float(err),
                        "reused_err": bool(reused_err),
                    }
                )

                if int(args.verbose) == 1:
                    reuse_note = " (reuse err, same ranks)" if reused_err else ""
                    print(
                        f"[layer {li:02d}][eb={eb}] iter={iter_idx:02d} th={th:.4f} "
                        f"raw_kq={raw_kq} raw_v={raw_v} rank_kq={rank_kq} rank_v={rank_v} "
                        f"err={err:.10f} (num={err_num:.6e}, den={err_den:.6e}){reuse_note}",
                        flush=True,
                    )

                # 记录当前合法最优档位（超过预算的档位不会覆盖 best_*）
                if err <= float(eb):
                    best_th = float(th)
                    best_rank_kq = int(rank_kq)
                    best_rank_v = int(rank_v)
                    best_err = float(err)

                th -= float(args.threshold_step)

                raw_kq = _rank_from_eval(eval_kq, th, num_heads)
                raw_v = _rank_from_eval(eval_v, th, num_heads)

                rk = _finalize_rank(
                    raw_kq,
                    num_heads,
                    head_dim,
                    args.min_rank,
                    strict_original=bool(args.strict_original_rank),
                )
                rv = _finalize_rank(
                    raw_v,
                    num_heads,
                    head_dim,
                    args.min_rank,
                    strict_original=bool(args.strict_original_rank),
                )

                if (not args.no_skip_redundant_eval) and rk == rank_kq and rv == rank_v:
                    skip_stream = True
                    iter_idx += 1
                    if err <= float(eb):
                        best_th = float(th)
                        best_rank_kq = int(rank_kq)
                        best_rank_v = int(rank_v)
                        best_err = float(err)
                    continue

                rank_kq, rank_v = rk, rv
                qlayer_new = OPTEigenAttnDecoderLayer(
                    layer, decomp_args, basis_kq_h, rank_kq, basis_v_h, rank_v, model.config
                ).to(device=lay_dev, dtype=dt)
                del qlayer
                qlayer = qlayer_new
                iter_idx += 1

            # 直接回到最后一个合法档位，避免“退出后 +step”回推带来的错位。
            th = float(best_th)
            rank_kq = int(best_rank_kq)
            rank_v = int(best_rank_v)

            qlayer = OPTEigenAttnDecoderLayer(
                layer, decomp_args, basis_kq_h, rank_kq, basis_v_h, rank_v, model.config
            ).to(device=lay_dev, dtype=dt)

            final_err, final_num, final_den = _relative_error_streaming_layers(
                layer, qlayer, inps, attention_mask, int(args.nsamples)
            )
            del qlayer

            if int(args.verbose) == 1:
                print(
                    f"[layer {li:02d}][eb={eb}] final th={th:.4f} rank_kq={rank_kq} rank_v={rank_v} "
                    f"err={final_err:.10f} (num={final_num:.6e}, den={final_den:.6e}) "
                    f"best_err={(best_err if best_err is not None else final_err):.10f} iters={iter_idx}",
                    flush=True,
                )

            layer_rows_per_budget[bi].append(
                {
                    "layer": li,
                    "basis_kq_h": basis_kq_h.detach().cpu().float(),
                    "basis_v_h": basis_v_h.detach().cpu().float(),
                    "rank_kq": int(rank_kq),
                    "rank_v": int(rank_v),
                }
            )

            layer_info["runs"].append(
                {
                    "error_budget": float(eb),
                    "threshold_final": float(th),
                    "rank_kq_pre": int(rank_kq),
                    "rank_v_pre": int(rank_v),
                    "layer_error": float(final_err),
                    "best_layer_error_within_budget": float(best_err if best_err is not None else final_err),
                    "per_head_rank_kq": _per_head_rank_from_eval(eval_kq, th),
                    "per_head_rank_v": _per_head_rank_from_eval(eval_v, th),
                    "rank_rule": "rank_total = num_heads * max(per_head_rank)",
                    "threshold_trace": trace,
                }
            )

            if int(args.report_energy) == 1:
                per_kq = layer_info["runs"][-1]["per_head_rank_kq"]
                per_v = layer_info["runs"][-1]["per_head_rank_v"]
                e_kq = _per_head_energy_from_eval_and_ranks(eval_kq, per_kq)
                e_v = _per_head_energy_from_eval_and_ranks(eval_v, per_v)
                layer_info["runs"][-1]["energy_kq"] = e_kq
                layer_info["runs"][-1]["energy_v"] = e_v

            print(
                f"[layer {li:02d}][eb={eb}] th={th:.4f} rank_kq={int(rank_kq)} rank_v={int(rank_v)} err={final_err:.6f}",
                flush=True,
            )
            if int(args.report_energy) == 1:
                ekq = layer_info["runs"][-1]["energy_kq"]
                ev = layer_info["runs"][-1]["energy_v"]
                print(
                    f"[layer {li:02d}][eb={eb}] energy_kq(mean/min/max)={ekq['mean']:.6f}/{ekq['min']:.6f}/{ekq['max']:.6f} "
                    f"energy_v(mean/min/max)={ev['mean']:.6f}/{ev['min']:.6f}/{ev['max']:.6f}",
                    flush=True,
                )

        # 用原始 full-rank 层继续传播到下一层输入
        for j in range(args.nsamples):
            if args.low_vram:
                x = inps[j].unsqueeze(0).to(lay_dev, dtype=dt, non_blocking=True)
                y = layer(x, attention_mask=attention_mask)[0]
                inps[j] = y.detach().cpu()
                del x, y
            else:
                inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        # 多卡下把异步 CUDA 错误尽量钉在本层末尾，避免拖到下一层 basis 拷贝才爆栈
        if torch.cuda.is_available():
            for di in range(torch.cuda.device_count()):
                torch.cuda.synchronize(di)
        layers[li] = layer.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        report["layers"].append(layer_info)
        print(f"[layer {li:02d}] done", flush=True)

    # 导出 basis artifact
    def _export_one_budget(bi: int, eb_label_for_log):
        rows = layer_rows_per_budget[bi]

        if float(args.kv_target_gb) > 0:
            new_ranks, info = _retarget_budget_ranks(
                rows,
                num_heads=num_heads,
                head_dim=head_dim,
                min_rank=int(args.min_rank),
                hidden=hidden,
                target_kv_gb=float(args.kv_target_gb),
                baseline_kv_gb=float(args.kv_baseline_gb),
                strict_original=bool(args.strict_original_rank),
            )
            print(
                f"[retarget eb={eb_label_for_log}] est_kv_gb {info['cur_kv_gb_est']:.4f} -> {info['actual_kv_gb_est']:.4f} "
                f"(target {info['target_kv_gb']:.4f}, scale={info['scale']:.4f})",
                flush=True,
            )
            report[f"budget_{eb_label_for_log}_retarget"] = info
        else:
            new_ranks = [(int(x["rank_kq"]), int(x["rank_v"])) for x in rows]

        for row, (rk, rv) in zip(rows, new_ranks):
            li = int(row["layer"])
            report["layers"][li]["runs"][bi]["rank_kq_post"] = int(rk)
            report["layers"][li]["runs"][bi]["rank_v_post"] = int(rv)
            report["layers"][li]["runs"][bi]["post_process"] = "retarget" if float(args.kv_target_gb) > 0 else "none"

            U_kq = _pack_headwise_basis(row["basis_kq_h"], int(rk), num_heads, head_dim).cpu().float()
            U_v = _pack_headwise_basis(row["basis_v_h"], int(rv), num_heads, head_dim).cpu().float()

            basis_dicts[bi][_basis_key(li, "q")] = U_kq
            basis_dicts[bi][_basis_key(li, "k")] = U_kq
            basis_dicts[bi][_basis_key(li, "v")] = U_v

        return None

    if eb_by_layer is not None:
        _export_one_budget(0, "perlayer")
        out_pt = out_dir / f"{args.out_prefix}_perlayer_eb.pt"
        torch.save(basis_dicts[0], out_pt)
        print(f"[saved] {out_pt}", flush=True)
    else:
        for bi, eb in enumerate(budgets):
            _export_one_budget(bi, eb)
            eb_tag = str(eb).replace(".", "p")
            out_pt = out_dir / f"{args.out_prefix}_eb{eb_tag}.pt"
            torch.save(basis_dicts[bi], out_pt)
            print(f"[saved] {out_pt}", flush=True)

    report_path = (
        out_dir / f"{args.out_prefix}_perlayer_eb_report.json"
        if eb_by_layer is not None
        else out_dir / f"{args.out_prefix}_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[saved] {report_path}", flush=True)


if __name__ == "__main__":
    main()