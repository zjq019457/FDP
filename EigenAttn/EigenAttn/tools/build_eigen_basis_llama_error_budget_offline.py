#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama/CausalLM 变体（HF ``LlamaForCausalLM``）离线 Eigen basis：激活驱动分解 + EB 阈值搜索。

与 OPT 离线脚本对齐的思路，但 K 谱为一维、cumsum(rank) 与 EigenAttn_base/decompose/eigen_attn.py 中
``is_llama`` 分支一致；并使用 ``LlamaEigenAttnDecoderLayer`` 做 streaming 重构误差。

数据：复用 ``build_eigen_basis_opt_error_budget_offline.py`` 的 parquet/文本迭代（建议使用 WikiText-2
train parquet 目录作为 ``--data_dir``）。

多卡：``CUDA_VISIBLE_DEVICES=0,1,2,3`` 可见 4 张卡时配合 ``--device_map auto``（需安装 ``accelerate``），
HF 将把各层分到多卡；整模 ``forward`` 的 ``input_ids`` 等设备由 ``embed_tokens`` 权重所在 GPU 推断。

单层 basis 导出 key（与 LlamaEigen 管线常用命名一致）：
  base_model.model.model.layers.{L}.self_attn.{q,k,v}_proj_basis
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
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_TOOLS = Path(__file__).resolve().parent


def _load_sibling_tools():
    """Load shared helpers from the OPT offline builder (text iter, tokenizer stream, CUDA cleanup)."""
    p = _TOOLS / "build_eigen_basis_opt_error_budget_offline.py"
    spec = importlib.util.spec_from_file_location("_ebb_opt_helpers", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_py_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _basis_key(layer_idx: int, kind: str) -> str:
    return (
        f"base_model.model.model.layers.{layer_idx}.self_attn.{kind}_proj_basis"
    )


def _rank_llama_k(eval_kq_1d: torch.Tensor, th: float, num_kv_heads: int) -> int:
    """EigenAttn_base/decompose/eigen_attn.py:is_llama 口径。"""
    e = eval_kq_1d.float().flatten().cpu()
    return int(num_kv_heads * int((torch.cumsum(e, dim=0) < float(th)).sum().item()))


def _rank_llama_v(eval_v: torch.Tensor, th: float, num_kv_heads: int) -> int:
    c = torch.cumsum(eval_v.float(), dim=1)
    mh = torch.amax((c < float(th)).sum(dim=1)).item()
    return int(num_kv_heads * int(mh))


def _finalize_rank_kv(
    rank: int,
    num_kv_heads: int,
    max_rank_per_kv_group: int,
    min_rank: int,
    *,
    strict_original: bool,
) -> int:
    """
    LlamaEigen 里每层 K/V 的谱维按 **HF 语义** 对齐 ``head_dim = hidden_size // num_attention_heads``，
    GQA 下不是 ``hidden // num_kv``；因此这里用 ``max_rank_per_kv_group``（通常为 head_dim_attn）做上界。
    """
    hd = max_rank_per_kv_group
    max_r = num_kv_heads * hd
    r = max(0, min(int(rank), max_r))
    k = r // num_kv_heads if num_kv_heads else 0
    if k < 1:
        k = 1
    if k > hd:
        k = hd
    r = num_kv_heads * k
    if (not strict_original) and r < int(min_rank):
        k = (int(min_rank) + num_kv_heads - 1) // num_kv_heads
        k = max(1, min(k, hd))
        r = num_kv_heads * k
    return int(min(r, max_r))


def _layer_attn_mask_for_forward(am) -> Optional[torch.Tensor]:
    """
    DecoderLayer / SDPA、eager LlamaAttention 都只接受 4D additive causal mask（或 None 走纯 causal）。
    2D pad mask（如 ones）会直接触发 ``[:, :, :, :]`` IndexError。
    校准片段为固定满长、无 padding 时用 None。
    """
    if am is None or (not torch.is_tensor(am)):
        return None
    if am.dim() >= 4:
        return am
    return None


def _relative_error_llama_streams(
    layer_hr,
    layer_lr,
    inps,
    attention_mask,
    position_ids,
    nsamples: int,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    diff_sq = 0.0
    hr_sq = 0.0
    p0 = next(layer_hr.parameters())
    for j in range(int(nsamples)):
        x = inps[j].unsqueeze(0).to(device=p0.device, dtype=p0.dtype, non_blocking=True)
        am = attention_mask
        if am is not None and torch.is_tensor(am):
            am = am.to(device=x.device)
        pos = position_ids
        if pos is not None and torch.is_tensor(pos):
            pos = pos.to(device=x.device)
        y_hr = layer_hr(x, attention_mask=am, position_ids=pos, use_cache=False)[0]
        y_lr = layer_lr(x, attention_mask=am, position_ids=pos, use_cache=False)[0]
        dh = y_hr.float() - y_lr.float()
        diff_sq += float(torch.sum(dh * dh).item())
        hr_sq += float(torch.sum(y_hr.float() * y_hr.float()).item())
        del x, y_hr, y_lr, dh, am, pos
    num = float(diff_sq**0.5)
    den = float(max(hr_sq**0.5, eps))
    err = num / den
    return float(err), num, den


def _pack_v_basis_export(
    basis_v_stack: torch.Tensor,
    rv: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """
    将 ``generate_basis_vectors_per_head`` 的 V stack 拼成块状 U。

    HF Llama GQA 下 ``v_proj`` 输出维度为 ``num_kv * head_dim``，
    ``head_dim = hidden // num_attention_heads``（常为 128），**不是** ``hidden // num_kv``；
    ``basis_v_stack`` 形如 ``[num_kv, head_dim, ...]``，与 ``LlamaEigenAttention`` 中
    ``org_weight_v[i*head_dim:(i+1)*head_dim]`` 一致。
    """
    if rv % num_kv_heads != 0:
        raise ValueError(f"rank_v={rv} not divisible by num_kv_heads={num_kv_heads}")
    if int(basis_v_stack.shape[0]) != int(num_kv_heads):
        raise ValueError(f"basis_v_stack leading dim {basis_v_stack.shape[0]} != num_kv_heads={num_kv_heads}")
    hv = int(basis_v_stack.shape[1])
    kcol = rv // num_kv_heads
    if kcol > hv:
        raise ValueError(f"rank_v per head {kcol} exceeds spectral dim {hv} for V basis slice")
    d_feat = num_kv_heads * hv
    U = torch.zeros((d_feat, rv), dtype=basis_v_stack.dtype, device=basis_v_stack.device)
    for h in range(num_kv_heads):
        rs = slice(h * hv, (h + 1) * hv)
        cs = slice(h * kcol, (h + 1) * kcol)
        U[rs, cs] = basis_v_stack[h, :, :kcol]
    return U.contiguous().float()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF Llama causal LM dir")
    ap.add_argument("--data_dir", required=True, help="含 train-*.parquet 或校准文本的根目录（WikiText-2）")
    ap.add_argument("--out_dir", required=True, help="输出目录（如 /diskD/...）")
    ap.add_argument("--error_budgets", type=str, default="", help="comma-separated e.g. 0.045,0.060（per_layer 时可省略）")
    ap.add_argument("--out_prefix", type=str, default="eigen_basis_llama31_8b_wikitext")
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--token_budget", type=int, default=1_500_000)
    ap.add_argument("--max_data_files", type=int, default=300)
    ap.add_argument("--max_data_lines", type=int, default=3_000_000)
    ap.add_argument("--avg_dim", type=int, default=32)
    ap.add_argument("--svd_threshold_init", type=float, default=1.0)
    ap.add_argument("--svd_threshold_floor", type=float, default=0.30)
    ap.add_argument("--threshold_step", type=float, default=0.02)
    ap.add_argument("--min_rank", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--device_map",
        type=str,
        default="",
        help="默认空：整模叠在 --device。auto/balanced/sequential：多 GPU 分片（需 accelerate；可见 GPU 数由 CUDA_VISIBLE_DEVICES 决定）。",
    )
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tokenizer", type=str, default="", help="为空则用 --model")
    ap.add_argument("--strict_original_rank", type=int, default=1, choices=[0, 1])
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1])
    ap.add_argument("--report_energy", type=int, default=1, choices=[0, 1])
    ap.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa"],
        help="HF 推理实现；单层 forward 若非 4D mask 则用 None causal。推荐 eager 与 EigenAttn 分解 hook 一致",
    )
    ap.add_argument(
        "--low_vram",
        action="store_true",
        help="layer0 activations stay on CPU 直到当前层再上 GPU（与 OPT 脚本类似）",
    )
    ap.add_argument(
        "--per_layer_eb_json",
        type=str,
        default="",
        help="per_layer_eb_from_step1 输出（键 error_budget_by_layer）。"
        "单文件 basis *_perlayer_eb.pt；忽略多值 --error_budgets。",
    )
    ap.add_argument(
        "--freeze_layer0_eb",
        action="store_true",
        help="per_layer JSON 加载后，将第 0 层 EB 设为 --layer0_eb。",
    )
    ap.add_argument(
        "--layer0_eb",
        type=float,
        default=0.045,
        help="与 --freeze_layer0_eb 配合。",
    )
    args = ap.parse_args()

    _h = _load_sibling_tools()
    _GPU_MODEL_HOLDER = _h._GPU_MODEL_HOLDER
    signal.signal(signal.SIGTERM, _h._cleanup_cuda_on_signal)
    signal.signal(signal.SIGINT, _h._cleanup_cuda_on_signal)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    base_root = repo_root.parent / "EigenAttn_base"
    if not base_root.exists():
        raise FileNotFoundError(base_root)

    mod_utils = _load_py_module("eigen_base_utils", base_root / "decompose" / "eigen_attn_utils.py")
    mod_dm = _load_py_module("eigen_base_decompose_modules", base_root / "models" / "decompose_modules.py")
    decompose_llama_layer = mod_utils.decompose_llama_layer
    LlamaEigenAttnDecoderLayer = mod_dm.LlamaEigenAttnDecoderLayer

    tok_path = args.tokenizer.strip() or args.model
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    texts = _h._iter_texts_from_dir(args.data_dir, args.max_data_files, args.max_data_lines)
    stream = _h._build_token_stream(tok, texts, args.token_budget, args.seed)
    if stream is None:
        raise RuntimeError("empty token stream from data_dir")

    dt = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    dm = (args.device_map or "").strip().lower()
    load_kw = dict(
        torch_dtype=dt,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    if dm in ("auto", "balanced", "sequential"):
        try:
            dm_arg = "auto" if dm == "auto" else dm
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map=dm_arg,
                **load_kw,
            )
        except Exception as e:
            raise RuntimeError(
                "Multi-GPU device_map loading failed (try: pip install accelerate). Original error: "
                f"{e}"
            ) from e
        device = model.get_input_embeddings().weight.device
    else:
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": device}, **load_kw)

    model.eval()
    model.config.use_cache = False

    cfg = model.config
    if getattr(cfg, "model_type", "").lower() not in {"llama"}:
        print(
            f"[warn] model_type={getattr(cfg, 'model_type', None)}; still attempting Llama layer path.",
            flush=True,
        )
    # Qwen2Config historically omits attention_bias; EigenAttn path expects it (q/k/v bias=True).
    if not hasattr(cfg, "attention_bias"):
        cfg.attention_bias = True
        print("[info] set config.attention_bias=True for non-Llama / Qwen path", flush=True)
    if not hasattr(cfg, "rope_scaling"):
        cfg.rope_scaling = None
        print("[info] set config.rope_scaling=None for non-Llama / Qwen path", flush=True)
    if not hasattr(cfg, "pretraining_tp"):
        cfg.pretraining_tp = 1
        print("[info] set config.pretraining_tp=1 for non-Llama / Qwen path", flush=True)

    n_attn = int(cfg.num_attention_heads)
    num_kv = int(cfg.num_key_value_heads)
    hidden = int(cfg.hidden_size)
    head_dim_attn = hidden // n_attn
    head_dim_kv = hidden // num_kv if num_kv > 0 else head_dim_attn  # Q/K/V 语义不同；rank 裁剪与 V 块状打包用 head_dim_attn

    _GPU_MODEL_HOLDER["model"] = model

    layers = model.model.layers
    n_layers = len(layers)

    eb_by_layer: Optional[List[float]] = None
    per_layer_eb_path = (args.per_layer_eb_json or "").strip()
    if per_layer_eb_path:
        pl_path = Path(per_layer_eb_path)
        if not pl_path.is_file():
            raise FileNotFoundError(f"--per_layer_eb_json not found: {pl_path}")
        with open(pl_path, "r", encoding="utf-8") as f:
            plj = json.load(f)
        if "error_budget_by_layer" not in plj:
            raise KeyError("JSON must contain 'error_budget_by_layer'")
        eb_by_layer = [float(x) for x in plj["error_budget_by_layer"]]
        if len(eb_by_layer) != n_layers:
            raise ValueError(
                f"per_layer_eb_json length {len(eb_by_layer)} != n_layers {n_layers}"
            )
        print(
            f"[per_layer_eb] loaded {pl_path}  EB min/mean/max="
            f"{min(eb_by_layer):.6f}/{sum(eb_by_layer)/len(eb_by_layer):.6f}/{max(eb_by_layer):.6f}",
            flush=True,
        )
        if args.freeze_layer0_eb:
            prev0 = eb_by_layer[0]
            eb_by_layer[0] = float(args.layer0_eb)
            print(
                f"[freeze_layer0_eb] layer0 EB {prev0:.6f} -> {eb_by_layer[0]:.6f}",
                flush=True,
            )
        budgets = [0.0]
    else:
        budgets = [float(x.strip()) for x in args.error_budgets.split(",") if x.strip()]
        if not budgets:
            raise ValueError("Provide --error_budgets or --per_layer_eb_json")

    Ntok = stream.numel()
    if Ntok <= args.seqlen + 2:
        raise RuntimeError(f"token stream too short ({Ntok})")

    g = torch.Generator().manual_seed(args.seed)
    batches: List[torch.Tensor] = []
    for _ in range(args.nsamples):
        s = int(torch.randint(0, Ntok - args.seqlen - 1, (1,), generator=g).item())
        batches.append(stream[s : s + args.seqlen].unsqueeze(0))

    _low_ram = bool(args.low_vram)
    inps_dev = torch.device("cpu") if _low_ram else device
    inps = torch.zeros((args.nsamples, args.seqlen, hidden), dtype=dt, device=inps_dev)
    cache: Dict[str, object] = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args_inner, **kwargs_inner):
            if _low_ram:
                inp = kwargs_inner["hidden_states"] if "hidden_states" in kwargs_inner else args_inner[0]
                inps[cache["i"]] = inp.detach().cpu().contiguous()
            else:
                inp = kwargs_inner["hidden_states"] if "hidden_states" in kwargs_inner else args_inner[0]
                inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs_inner.get("attention_mask")
            raise ValueError

    layers[0] = Catcher(layers[0])
    attn_mask_big = torch.ones((1, args.seqlen), dtype=torch.long, device=device)
    pos_ids_big = torch.arange(args.seqlen, device=device, dtype=torch.long).unsqueeze(0)
    for b in batches:
        if cache["i"] >= args.nsamples:
            break
        try:
            model(
                input_ids=b.to(device),
                attention_mask=attn_mask_big,
                position_ids=pos_ids_big,
                use_cache=False,
                return_dict=True,
            )
        except ValueError:
            pass
    layers[0] = layers[0].module

    attention_mask = cache.get("attention_mask")
    # 整模型 forward 仍可用 2D ones；单层 / decompose_llama_layer 必须用 4D additive 或 None
    attn_for_layers = _layer_attn_mask_for_forward(attention_mask)
    position_ids = pos_ids_big

    layer_rows_per_budget: List[List[dict]] = [[]] if eb_by_layer is not None else [[] for _ in budgets]

    report: Dict[str, object] = {
        "model": args.model,
        "data_dir": args.data_dir,
        "budgets": eb_by_layer if eb_by_layer is not None else budgets,
        "arch": "llama",
        "mode": {
            "per_layer_eb": eb_by_layer is not None,
            "freeze_layer0_eb": bool(getattr(args, "freeze_layer0_eb", False)),
            "layer0_eb": float(args.layer0_eb) if eb_by_layer is not None else None,
            "per_layer_eb_json": str(Path(per_layer_eb_path).resolve()) if per_layer_eb_path else None,
        },
        "num_attention_heads": n_attn,
        "num_key_value_heads": num_kv,
        "head_dim_attn": head_dim_attn,
        "head_dim_kv": head_dim_kv,
        "attn_implementation": args.attn_implementation,
        "device_map": dm or "single",
        "attention_mask_cached_dim": (
            tuple(attention_mask.shape) if torch.is_tensor(attention_mask) else None
        ),
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

    for li in range(n_layers):
        layer = layers[li]
        lay_dev = next(layer.parameters()).device
        if lay_dev.type == "cuda":
            torch.cuda.set_device(lay_dev)
        # get_kqv_llama / RMSNorm：layer 权重在 GPU 时，激活必须与之一致（仅用逐样本搬到 GPU 不够：
        # 分解里会先跑整层 RMSNorm）。
        inps_work = inps.to(device=lay_dev, dtype=dt, non_blocking=_low_ram)

        am_l = attn_for_layers.to(lay_dev) if torch.is_tensor(attn_for_layers) else None
        pos_l = position_ids.to(lay_dev) if torch.is_tensor(position_ids) else position_ids

        decomp_args = SimpleNamespace(
            nsamples=int(args.nsamples),
            low_rank=False,
            eigen_attn_params={
                "threshold": float(args.svd_threshold_init),
                "avg_dim_features": int(args.avg_dim),
            },
        )

        basis_kq_cpu, eval_kq, basis_v_stack, eval_v = decompose_llama_layer(
            layer, inps_work, decomp_args, num_kv, li, am_l, pos_l
        )
        basis_kq_cpu = basis_kq_cpu.detach().cpu().float().contiguous()
        basis_v_stack = basis_v_stack.detach().cpu().float().contiguous()
        eval_v_cpu = eval_v.detach().cpu().float()

        layer_info = {"layer": li, "runs": []}

        inner_sched = [(0, float(eb_by_layer[li]))] if eb_by_layer is not None else list(enumerate(budgets))
        for bi, eb in inner_sched:
            th = float(args.svd_threshold_init)
            err = 0.0
            iter_idx = 0
            trace: List[Dict[str, object]] = []

            raw_k = _rank_llama_k(eval_kq, th, num_kv)
            raw_v = _rank_llama_v(eval_v_cpu, th, num_kv)
            rank_kq = _finalize_rank_kv(
                raw_k, num_kv, head_dim_attn, args.min_rank, strict_original=bool(args.strict_original_rank)
            )
            rank_v = _finalize_rank_kv(
                raw_v, num_kv, head_dim_attn, args.min_rank, strict_original=bool(args.strict_original_rank)
            )

            qlayer = LlamaEigenAttnDecoderLayer(
                layer, decomp_args, basis_kq_cpu.to(lay_dev, dtype=dt),
                rank_kq, basis_v_stack.to(lay_dev, dtype=dt), rank_v, cfg, li,
            ).to(device=lay_dev, dtype=dt)

            best_th = float(th)
            best_rank_k = int(rank_kq)
            best_rank_v = int(rank_v)
            best_err = None
            skip_stream = False

            while err < float(eb) and th > float(args.svd_threshold_floor):
                reused = skip_stream
                if not skip_stream:
                    err, _, _ = _relative_error_llama_streams(
                        layer, qlayer, inps_work, am_l, pos_l, int(args.nsamples)
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(lay_dev)

                trace.append(
                    {
                        "iter": int(iter_idx),
                        "threshold": float(th),
                        "raw_rank_k": int(raw_k),
                        "raw_rank_v": int(raw_v),
                        "rank_kq": int(rank_kq),
                        "rank_v": int(rank_v),
                        "err": float(err),
                        "reused_err": bool(reused),
                    }
                )

                if int(args.verbose) == 1:
                    print(
                        f"[layer {li:02d}][eb={eb}] iter={iter_idx:02d} th={th:.4f} "
                        f"rank_k={rank_kq} rank_v={rank_v} err={err:.10f} reused={bool(reused)}",
                        flush=True,
                    )

                if err <= float(eb):
                    best_th = float(th)
                    best_rank_k = int(rank_kq)
                    best_rank_v = int(rank_v)
                    best_err = float(err)

                th -= float(args.threshold_step)
                raw_k = _rank_llama_k(eval_kq, th, num_kv)
                raw_v = _rank_llama_v(eval_v_cpu, th, num_kv)
                rk_n = _finalize_rank_kv(
                    raw_k, num_kv, head_dim_attn, args.min_rank, strict_original=bool(args.strict_original_rank)
                )
                rv_n = _finalize_rank_kv(
                    raw_v, num_kv, head_dim_attn, args.min_rank, strict_original=bool(args.strict_original_rank)
                )

                if (not getattr(args, "no_skip_redundant_eval", False)) and rk_n == rank_kq and rv_n == rank_v:
                    skip_stream = True
                    iter_idx += 1
                    continue

                rank_kq, rank_v = rk_n, rv_n
                qlayer_new = LlamaEigenAttnDecoderLayer(
                    layer,
                    decomp_args,
                    basis_kq_cpu.to(lay_dev, dtype=dt),
                    rank_kq,
                    basis_v_stack.to(lay_dev, dtype=dt),
                    rank_v,
                    cfg,
                    li,
                ).to(device=lay_dev, dtype=dt)
                del qlayer
                qlayer = qlayer_new
                skip_stream = False
                iter_idx += 1

            th = float(best_th)
            rank_kq = int(best_rank_k)
            rank_v = int(best_rank_v)
            qlayer = LlamaEigenAttnDecoderLayer(
                layer,
                decomp_args,
                basis_kq_cpu.to(lay_dev, dtype=dt),
                rank_kq,
                basis_v_stack.to(lay_dev, dtype=dt),
                rank_v,
                cfg,
                li,
            ).to(device=lay_dev, dtype=dt)
            final_err, _, _ = _relative_error_llama_streams(
                layer, qlayer, inps_work, am_l, pos_l, int(args.nsamples)
            )
            del qlayer

            k_cols = rank_kq // num_kv if num_kv else rank_kq
            Uqk = basis_kq_cpu[:, :k_cols].contiguous()

            if int(args.report_energy) == 1:
                e_k = _cum_energy(eval_kq, k_cols)
            else:
                e_k = {}

            runs = {
                    "error_budget": float(eb),
                    "threshold_final": float(th),
                    "rank_kq_final": int(rank_kq),
                    "rank_v_final": int(rank_v),
                    "layer_error": float(final_err),
                    "threshold_trace": trace,
                }
            if int(args.report_energy) == 1:
                runs["energy_k_approx"] = e_k

            layer_info["runs"].append(runs)

            layer_rows_per_budget[bi].append(
                {
                    "layer": li,
                    "basis_kq": Uqk,
                    "basis_v_stack": basis_v_stack,
                    "rank_kq": int(rank_kq),
                    "rank_v": int(rank_v),
                }
            )

            print(
                f"[layer {li:02d}][eb={eb}] th={th:.4f} rank_k={rank_kq} rank_v={rank_v} err={final_err:.6f}",
                flush=True,
            )

        next_rows: List[torch.Tensor] = []
        for j in range(args.nsamples):
            x = inps_work[j].unsqueeze(0)
            y = layer(x, attention_mask=am_l, position_ids=pos_l, use_cache=False)[0]
            next_rows.append(y.squeeze(0).detach())
            del x, y
        inps = torch.stack(next_rows, dim=0)
        if _low_ram:
            inps = inps.cpu()
        del inps_work

        layers[li] = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        report["layers"].append(layer_info)
        print(f"[layer {li:02d}] done", flush=True)

    if eb_by_layer is not None:
        rows = layer_rows_per_budget[0]
        d: Dict[str, torch.Tensor] = {}
        for row in rows:
            li = row["layer"]
            rk = int(row["rank_kq"])
            rv = int(row["rank_v"])
            Uqk = row["basis_kq"].float().contiguous()
            Uv = _pack_v_basis_export(row["basis_v_stack"], rv, num_kv)
            d[_basis_key(li, "q")] = Uqk
            d[_basis_key(li, "k")] = Uqk.clone()
            d[_basis_key(li, "v")] = Uv
        outp = out_dir / f"{args.out_prefix}_perlayer_eb.pt"
        torch.save(d, outp)
        print(f"[saved] {outp}", flush=True)
        rep_path = out_dir / f"{args.out_prefix}_perlayer_eb_report.json"
    else:
        for bi, eb in enumerate(budgets):
            rows = layer_rows_per_budget[bi]
            eb_tag = str(eb).replace(".", "p")
            d = {}
            for row in rows:
                li = row["layer"]
                rk = int(row["rank_kq"])
                rv = int(row["rank_v"])
                Uqk = row["basis_kq"].float().contiguous()
                Uv = _pack_v_basis_export(row["basis_v_stack"], rv, num_kv)
                d[_basis_key(li, "q")] = Uqk
                d[_basis_key(li, "k")] = Uqk.clone()
                d[_basis_key(li, "v")] = Uv
            outp = out_dir / f"{args.out_prefix}_eb{eb_tag}.pt"
            torch.save(d, outp)
            print(f"[saved] {outp}", flush=True)
        rep_path = out_dir / f"{args.out_prefix}_report.json"

    def _json_safe(obj):  # minimal: truncate tensors in layers
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return f"<tensor shape={tuple(obj.shape)} dtype={obj.dtype}>"
        return obj

    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, ensure_ascii=False, indent=2)
    print(f"[saved] {rep_path}", flush=True)


def _cum_energy(eval_1d: torch.Tensor, r: int) -> Dict[str, float]:
    ev = torch.cumsum(eval_1d.float().flatten(), dim=0)
    rr = int(max(1, min(r, ev.numel())) - 1)
    return {"retained_to_col": rr + 1, "energy_approx": float(ev[rr].item())}


if __name__ == "__main__":
    main()
