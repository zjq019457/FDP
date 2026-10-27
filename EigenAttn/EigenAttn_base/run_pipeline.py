"""End-to-end EigenAttn-base pipeline (offline) for OPT / MPT / Llama-3.

What it does, in one CLI invocation:

1. Build calibration data (default: c4-train shards, offline).
2. **Stage compress_base**: compress the dense base model with a *uniform*
   ``--error_budget`` and save to ``$OUT/baseline/`` (HF model dir + low_rank_config).
3. **Stage eval_base**: load the baseline checkpoint and run wikitext-2 / c4
   PPL (Utkarsh-aligned) and bench5 0-shot accuracy (offline localjsonl);
   results -> ``$OUT/baseline_eval.json``.
4. **Stage diag**: layerwise fragility diagnosis between the dense base and
   the baseline checkpoint -> ``$OUT/step1.json``.
5. **Stage derive**: per-layer error_budget schedule, with ``freeze_layers=[0]``
   and ``normalize=mean`` so the average EB matches uniform; -> ``$OUT/dynamic_eb.json``.
6. **Stage compress_dyn**: compress with the dynamic schedule -> ``$OUT/dynamic/``.
7. **Stage eval_dyn**: PPL + ACC of the dynamic checkpoint -> ``$OUT/dynamic_eval.json``.
8. **Stage compare**: side-by-side report -> ``$OUT/compare.json``.

Stages can be skipped or restarted via ``--stages compress_base,eval_base,...``.

The whole pipeline is **offline-only**: WikiText-2 reads local parquet, C4 reads
local ``.json.gz``, bench5 reads local jsonl via lm-eval-harness include_path.

Example:

    python run_pipeline.py \\
        --net opt-30b \\
        --base /diskD/jiaqizhang/opt-30b \\
        --tok  /diskD/jiaqizhang/opt-30b \\
        --error_budget 0.024 \\
        --avg_dim 48 --nsamples 64 \\
        --output_dir /diskD/jiaqizhang/opt30b_eb024_pipeline \\
        --multigpu

For Llama-3-8B (paper config: avg_dim=48 nsamples=128) / MPT-7B: change ``--net`` / ``--base``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

# project root -> sys.path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import utils
from datautils import (
    DEFAULT_C4_DIR,
    DEFAULT_C4_TRAIN_GLOB,
    DEFAULT_C4_VAL_GLOB,
    DEFAULT_WIKITEXT2_DIR,
    build_synthetic_calib_dataloader,
    get_loaders,
    set_seed,
)
from decompose.eigen_attn import eigenattn
from diagnose.derive_eb import derive_per_layer_eb, save_dynamic_eb
from diagnose.step1 import run_step1_diagnose, save_step1
from eval.eval_acc import DEFAULT_BENCH5_TASKS, DEFAULT_INCLUDE_PATH, evaluate_bench5, save_acc_results
from eval.eval_ppl import evaluate_ppl
from models.LMClass import LMClass
from parallel_utils import get_lowest_occupied_gpu, map_layers_to_multi_gpus

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Stage names
# ---------------------------------------------------------------------------
ALL_STAGES = (
    "compress_base",
    "eval_base",
    "diag",
    "derive",
    "compress_dyn",
    "eval_dyn",
    "compare",
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model
    p.add_argument("--net", required=True,
                   choices=["opt-30b", "opt-13b", "opt-6.7b", "opt-2.7b", "opt-1.3b", "opt-125m",
                            "mpt-7b", "mpt-30b",
                            "Llama-3-8b", "Llama-3-70b", "Llama-2-7b", "Llama-2-13b", "Llama-2-70b"])
    p.add_argument("--base", required=True, help="path to dense base HF checkpoint dir (offline)")
    p.add_argument("--tok", default=None, help="tokenizer path; defaults to --base")
    p.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])

    # output
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None,
                   help="HF cache dir; defaults to $OUTPUT_DIR/HF_cache to avoid polluting $HOME")

    # stages
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help="comma-separated stages to run; subset of " + ",".join(ALL_STAGES))

    # compression
    p.add_argument("--error_budget", type=float, required=True,
                   help="uniform EB for the baseline; also the eb_base for dynamic mapping")
    p.add_argument("--avg_dim", type=int, default=48,
                   help="EigenAttn avg_dim (paper: 48 for OPT/Llama; must be <= nsamples for the avg-dim averaging in get_kqv_*)")
    p.add_argument("--svd_threshold_init", type=float, default=0.98)
    p.add_argument("--svd_threshold_floor", type=float, default=0.3)
    p.add_argument("--target_attention_rank_ratio", type=float, default=None)
    p.add_argument("--nsamples", type=int, default=128,
                   help="calibration nsamples (paper: 128 for Llama-3, 64 for OPT-30B; must be >= avg_dim)")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--multigpu", action="store_true",
                   help="map layers across all visible GPUs during eval (no impact on compression)")

    # calibration data
    p.add_argument("--calib_dataset", default="c4", choices=["wikitext2", "c4"])
    p.add_argument("--c4_data_dir", default=DEFAULT_C4_DIR)
    p.add_argument("--c4_train_glob", default=DEFAULT_C4_TRAIN_GLOB)
    p.add_argument("--c4_val_glob", default=DEFAULT_C4_VAL_GLOB)
    p.add_argument("--c4_pool_docs", type=int, default=200_000)
    p.add_argument("--c4_scan_lines", type=int, default=200_000)
    p.add_argument("--wikitext2_dir", default=DEFAULT_WIKITEXT2_DIR)

    # PPL eval
    p.add_argument("--ppl_c4_pool_docs", type=int, default=500_000)
    p.add_argument("--ppl_c4_scan_lines", type=int, default=500_000)
    p.add_argument("--ppl_c4_val_chunks", type=int, default=256)
    p.add_argument("--ppl_c4_val_seed", type=int, default=0)
    p.add_argument("--ppl_limit", type=int, default=-1)
    p.add_argument("--skip_ppl", action="store_true")

    # ACC eval
    p.add_argument("--skip_acc", action="store_true")
    p.add_argument("--bench5_tasks", default=",".join(DEFAULT_BENCH5_TASKS))
    p.add_argument("--bench5_include_path", default=DEFAULT_INCLUDE_PATH)
    p.add_argument("--bench5_batch_size", type=int, default=1)
    p.add_argument("--bench5_limit", type=int, default=None,
                   help="lm-eval --limit for ACC stages (None = full)")

    # diagnose
    p.add_argument("--diag_seqlen", type=int, default=512)
    p.add_argument("--diag_prompts_file", default=None,
                   help="optional file with one prompt per line; otherwise uses DEFAULT_PROMPTS")
    p.add_argument("--diag_base_device", default=None)
    p.add_argument("--diag_cmp_device", default=None)

    # derive dynamic EB
    p.add_argument("--lambda_eb", type=float, default=1.2)
    p.add_argument("--eb_normalize", default="mean", choices=["mean", "none"])
    p.add_argument("--freeze_layers", default="0",
                   help="comma-separated layer indices forced to eb_base (default: '0')")
    p.add_argument("--eb_min_ratio", type=float, default=None)

    # synthetic fallback for calib
    p.add_argument("--synthetic_calib", action="store_true",
                   help="use random-token calib (last resort, do not use for paper numbers)")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_compress_args(cli, error_budget: float):
    """Build a Namespace replicating what `eigenattn(...)` expects."""
    a = argparse.Namespace(**vars(cli))
    a.model = cli.base
    a.cache_dir = cli.cache_dir
    a.batch_size = 1
    a.calib_dataset = cli.calib_dataset
    a.error_budget = float(error_budget)
    a.svd_threshold_init = cli.svd_threshold_init
    a.svd_threshold_floor = cli.svd_threshold_floor
    a.target_attention_rank_ratio = cli.target_attention_rank_ratio
    a.avg_dim = cli.avg_dim
    a.nsamples = cli.nsamples
    a.seqlen = cli.seqlen
    a.seed = cli.seed
    a.attn_implementation = cli.attn_implementation
    a.eigen_attn_params = {
        "threshold": 1.0,
        "avg_dim_features": cli.avg_dim,
        "error_budget": float(error_budget),
    }
    a.tasks = ""
    a.eval_ppl = False
    a.evaluate_baseline = False
    a.fine_tune = False
    a.load_low_rank = False
    a.load_peft_model = False
    a.peft_model_path = None
    a.save_dir = None
    a.output_dir = cli.output_dir
    a.multigpu = cli.multigpu
    a.resume = None
    return a


def _make_lmclass(args_compress) -> LMClass:
    lm = LMClass(args_compress)
    lm.seqlen = args_compress.seqlen
    lm.model.eval()
    for p in lm.model.parameters():
        p.requires_grad = False
    return lm


def _model_dtype(cli):
    return torch.bfloat16 if cli.dtype == "bf16" else torch.float16


def _kv_cache_size_gb(model, seqlen: int, arch: str) -> float:
    """Approximate paper-aligned KV cache size in GB for batch=1, fp16."""
    num_elements = 0
    sd = model.state_dict()
    for key, w in sd.items():
        if arch == "mpt":
            if "Wqkv" in key and "weight" in key:
                num_elements += seqlen * (w.shape[0] * 2 / 3)
            elif "Wqkv_low" in key and "weight" in key:
                pass  # rank captured separately via attribute
        else:
            if any(s in key for s in ("k_proj.weight", "v_proj.weight", "k_proj_low.weight", "v_proj_low.weight")):
                num_elements += seqlen * w.shape[0]
    if arch == "mpt":
        # for MPT-low we read rank_kq + rank_v from attention modules instead
        try:
            blocks = model.transformer.blocks
            for b in blocks:
                attn = b.attn
                rkq = getattr(attn, "rank_kq", None)
                rv = getattr(attn, "rank_v", None)
                if rkq is not None and rv is not None:
                    num_elements += seqlen * (int(rkq) + int(rv))
        except AttributeError:
            pass
    return float(num_elements) * 16.0 / (8.0 * 1024 ** 3)


def _model_params_b(model) -> float:
    n = 0
    for k, v in model.state_dict().items():
        if "lm_head" in k:
            continue
        n += v.numel()
    return n / 1e9


def _detect_arch_str(net: str) -> str:
    n = net.lower()
    if "opt" in n:
        return "opt"
    if "mpt" in n:
        return "mpt"
    return "llama"


def _save_compressed(lm: LMClass, save_dir: str, low_rank_config) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    lm.model.save_pretrained(save_dir)
    lm.tokenizer.save_pretrained(save_dir)
    if isinstance(low_rank_config, list):
        torch.save(low_rank_config, os.path.join(save_dir, "low_rank_config.pt"))
        with open(os.path.join(save_dir, "low_rank_config.json"), "w", encoding="utf-8") as f:
            json.dump(low_rank_config, f, indent=2)
    else:
        torch.save(low_rank_config, os.path.join(save_dir, "low_rank_config.pt"))


def _multigpu_map(lm: LMClass, net: str) -> None:
    """Replicates main_eigen_attn.evaluate's multigpu shard mapping."""
    n = net.lower()
    if "opt" in n:
        layers = lm.model.model.decoder.layers
        map_layers_to_multi_gpus(layers)
        input_device = layers[0].device
        output_device = layers[-1].device
        lm._device = input_device
        lm.model.model.decoder.embed_positions.to(input_device)
        lm.model.model.decoder.embed_tokens.to(input_device)
        lm.model.model.decoder.final_layer_norm.to(output_device)
        lm.model.lm_head.to(output_device)
    elif "mpt" in n:
        layers = lm.model.transformer.blocks
        map_layers_to_multi_gpus(layers)
        input_device = layers[0].device
        output_device = layers[-1].device
        lm._device = input_device
        lm.model.transformer.wte.to(input_device)
        lm.model.transformer.norm_f.to(output_device)
        lm.model.lm_head.to(output_device)
    else:
        layers = lm.model.model.layers
        map_layers_to_multi_gpus(layers)
        input_device = layers[0].device
        output_device = layers[-1].device
        lm._device = input_device
        lm.model.model.embed_tokens.to(input_device)
        lm.model.model.norm.to(output_device)
        lm.model.lm_head.to(output_device)


def _move_single_device(lm: LMClass, net: str) -> None:
    n = net.lower()
    if "opt" in n:
        lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
    elif "mpt" in n:
        lm.model.transformer = lm.model.transformer.to(lm.device)
    else:
        lm.model = lm.model.to(lm.device)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------
def _build_calib_loader(cli, args_compress, logger):
    """Build / cache a calibration dataloader on the configured offline source."""
    cache_dir = Path(cli.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"calib_{cli.net}_{cli.calib_dataset}_n{cli.nsamples}_s{cli.seed}.cache"
    if cache_path.is_file():
        logger.info(f"[pipeline] reuse calib cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)
    if cli.synthetic_calib:
        logger.warning("[pipeline] synthetic_calib: NOT recommended for paper numbers")
        loader = build_synthetic_calib_dataloader(cli.base, cli.nsamples, cli.seqlen, cli.seed)
        torch.save(loader, cache_path)
        return loader
    try:
        loader, _ = get_loaders(
            cli.calib_dataset,
            nsamples=cli.nsamples,
            seed=cli.seed,
            seqlen=cli.seqlen,
            model=cli.base,
            wikitext2_dir=cli.wikitext2_dir,
            c4_data_dir=cli.c4_data_dir,
            c4_train_glob=cli.c4_train_glob,
            c4_val_glob=cli.c4_val_glob,
            c4_pool_docs=cli.c4_pool_docs,
            c4_scan_lines=cli.c4_scan_lines,
        )
    except Exception as e:
        logger.warning(f"[pipeline] get_loaders failed ({e}); falling back to synthetic_calib")
        loader = build_synthetic_calib_dataloader(cli.base, cli.nsamples, cli.seqlen, cli.seed)
    torch.save(loader, cache_path)
    return loader


def stage_compress(
    cli,
    *,
    save_dir: str,
    error_budget: float,
    per_layer_eb: Optional[Sequence[float]],
    freeze_layers: Optional[Sequence[int]],
    logger,
) -> Dict[str, object]:
    """Run a single compression run (uniform or dynamic) and save the model."""
    args_compress = _build_compress_args(cli, error_budget)

    if cli.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        logger.info(f"[pipeline] compression placed on cuda:{gpu_id}")
        device_str = f"cuda:{gpu_id}"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    lm = _make_lmclass(args_compress)
    lm._device = device_str
    logger.info(f"[pipeline] error_budget={error_budget:.6f} per_layer={per_layer_eb is not None} freeze={list(freeze_layers or [])}")

    dataloader = _build_calib_loader(cli, args_compress, logger)
    eigenattn(
        lm, args_compress, dataloader, logger,
        per_layer_error_budget=per_layer_eb,
        freeze_layers=freeze_layers,
    )

    arch = _detect_arch_str(cli.net)
    if arch == "opt":
        from transformers import AutoConfig
        from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn
        layers = lm.model.model.decoder.layers
        low_rank_config = torch.zeros(len(layers), 2)
        for i, ly in enumerate(layers):
            low_rank_config[i][0] = ly.self_attn.rank_kq
            low_rank_config[i][1] = ly.self_attn.rank_v
        cfg = AutoConfig.from_pretrained(cli.base, attn_implementation=cli.attn_implementation, cache_dir=cli.cache_dir)
        m2 = OPTForCausalLM_EigenAttn.from_pretrained(
            cli.base, config=cfg, low_rank_config=low_rank_config,
            device_map="cpu", torch_dtype=_model_dtype(cli), cache_dir=cli.cache_dir,
        )
        L2 = m2.model.decoder.layers
        for i in range(len(L2)):
            L2[i].self_attn.k_proj_low.weight.data = layers[i].self_attn.k_proj.weight.data
            L2[i].self_attn.k_proj_low.bias.data = layers[i].self_attn.k_proj.bias.data
            L2[i].self_attn.q_proj_low.weight.data = layers[i].self_attn.q_proj.weight.data
            L2[i].self_attn.q_proj_low.bias.data = layers[i].self_attn.q_proj.bias.data
            L2[i].self_attn.v_proj_low.weight.data = layers[i].self_attn.v_proj.weight.data
            L2[i].self_attn.v_proj_low.bias.data = layers[i].self_attn.v_proj.bias.data
            L2[i].self_attn.out_proj_up.weight.data = layers[i].self_attn.out_proj_up.weight.data
            L2[i].self_attn.out_proj_up.bias.data = layers[i].self_attn.out_proj_up.bias.data
        lm.model = m2
        _save_compressed(lm, save_dir, low_rank_config)
    elif arch == "mpt":
        from transformers import AutoConfig
        from models.modelling_mpt_eigen_attn import MptForCausalLM_EigenAttn
        layers = lm.model.transformer.blocks
        low_rank_config = torch.zeros(len(layers), 2)
        for i, ly in enumerate(layers):
            low_rank_config[i][0] = ly.attn.rank_kq
            low_rank_config[i][1] = ly.attn.rank_v
        cfg = AutoConfig.from_pretrained(cli.base, attn_implementation=cli.attn_implementation, cache_dir=cli.cache_dir)
        m2 = MptForCausalLM_EigenAttn.from_pretrained(
            cli.base, config=cfg, low_rank_config=low_rank_config,
            device_map="cpu", torch_dtype=_model_dtype(cli), cache_dir=cli.cache_dir,
        )
        L2 = m2.transformer.blocks
        for i in range(len(L2)):
            L2[i].attn.Wqkv_low.weight.data = layers[i].attn.Wqkv.weight.data
            L2[i].attn.out_proj_up.weight.data = layers[i].attn.out_proj.weight.data
        lm.model = m2
        _save_compressed(lm, save_dir, low_rank_config)
    else:
        from transformers import AutoConfig
        from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
        layers = lm.model.model.layers
        low_rank_config = []
        for i, ly in enumerate(layers):
            low_rank_config.append([int(ly.self_attn.rank_kq), int(ly.self_attn.rank_v)])
        cfg = AutoConfig.from_pretrained(cli.base, attn_implementation=cli.attn_implementation, cache_dir=cli.cache_dir)
        m2 = LlamaForCausalLM_EigenAttn.from_pretrained(
            cli.base, config=cfg, low_rank_config=low_rank_config,
            device_map="cpu", torch_dtype=_model_dtype(cli), cache_dir=cli.cache_dir,
        )
        L2 = m2.model.layers
        bias = cfg.attention_bias
        for i in range(len(L2)):
            L2[i].self_attn.k_proj_low.weight.data = layers[i].self_attn.k_proj_low.weight.data
            if bias:
                L2[i].self_attn.k_proj_low.bias.data = layers[i].self_attn.k_proj_low.bias.data
            L2[i].self_attn.k_proj_up.weight.data = layers[i].self_attn.k_proj_up.weight.data
            L2[i].self_attn.v_proj_low.weight.data = layers[i].self_attn.v_proj_low.weight.data
            if bias:
                L2[i].self_attn.v_proj_low.bias.data = layers[i].self_attn.v_proj_low.bias.data
            L2[i].self_attn.o_proj_up.weight.data = layers[i].self_attn.o_proj_up.weight.data
            if bias:
                L2[i].self_attn.o_proj_up.bias.data = layers[i].self_attn.o_proj_up.bias.data
        lm.model = m2
        _save_compressed(lm, save_dir, low_rank_config)

    info = {
        "save_dir": str(Path(save_dir).resolve()),
        "arch": arch,
        "n_layers": len(layers),
        "params_b": _model_params_b(lm.model),
        "kv_cache_gb": _kv_cache_size_gb(lm.model, lm.seqlen, arch),
        "low_rank_config": low_rank_config if isinstance(low_rank_config, list) else low_rank_config.tolist(),
    }
    logger.info(
        f"[pipeline] compressed -> {save_dir}: params={info['params_b']:.4f}B kv_cache={info['kv_cache_gb']:.4f}GB"
    )
    del lm
    torch.cuda.empty_cache()
    return info


def stage_eval(
    cli,
    *,
    cmp_dir: str,
    out_json: str,
    label: str,
    logger,
) -> Dict[str, object]:
    """Load an EigenAttn checkpoint and run PPL + ACC; persist results."""
    args_eval = _build_compress_args(cli, cli.error_budget)
    args_eval.load_low_rank = True
    args_eval.save_dir = cmp_dir
    args_eval.model = cli.base  # tokenizer / config init still uses base for fast loaders

    lm = _make_lmclass(args_eval)
    if cli.multigpu:
        _multigpu_map(lm, cli.net)
    else:
        lm._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _move_single_device(lm, cli.net)

    arch = _detect_arch_str(cli.net)
    info: Dict[str, object] = {
        "label": label,
        "cmp_dir": cmp_dir,
        "arch": arch,
        "params_b": _model_params_b(lm.model),
        "kv_cache_gb": _kv_cache_size_gb(lm.model, lm.seqlen, arch),
    }
    logger.info(f"[pipeline] {label} kv_cache={info['kv_cache_gb']:.4f}GB params={info['params_b']:.4f}B")

    if not cli.skip_ppl:
        info["ppl"] = evaluate_ppl(
            lm.model,
            tokenizer_path=cli.base,
            seqlen=lm.seqlen,
            seed=cli.seed,
            limit=cli.ppl_limit,
            device=lm._device if not cli.multigpu else None,
            do_wikitext2=True,
            do_c4=True,
            wikitext2_dir=cli.wikitext2_dir,
            c4_data_dir=cli.c4_data_dir,
            c4_glob=cli.c4_val_glob,
            c4_pool_docs=cli.ppl_c4_pool_docs,
            c4_scan_lines=cli.ppl_c4_scan_lines,
            c4_val_chunks=cli.ppl_c4_val_chunks,
            c4_val_seed=cli.ppl_c4_val_seed,
            label=label,
            log_fn=logger.info,
        )

    if not cli.skip_acc:
        tasks = [t for t in cli.bench5_tasks.split(",") if t.strip()]
        try:
            acc_report = evaluate_bench5(
                lm,
                tasks=tasks,
                include_path=cli.bench5_include_path,
                num_fewshot=0,
                batch_size=cli.bench5_batch_size,
                limit=cli.bench5_limit,
                log_fn=logger.info,
            )
            info["acc"] = acc_report["results"]
            save_acc_results(acc_report, out_json.replace(".json", "_acc_raw.json"))
        except Exception as e:
            logger.exception(f"[pipeline] ACC stage failed for {label}: {e}")
            info["acc"] = {"error": str(e)}

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    logger.info(f"[pipeline] eval saved -> {out_json}")
    del lm
    torch.cuda.empty_cache()
    return info


def stage_diag(
    cli,
    *,
    cmp_dir: str,
    out_json: str,
    logger,
) -> Dict[str, object]:
    """Load BASE and CMP, run forward-hook step1 fragility diagnosis."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
    from models.modelling_mpt_eigen_attn import MptForCausalLM_EigenAttn
    from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

    arch = _detect_arch_str(cli.net)
    dtype = _model_dtype(cli)
    cfg_base = AutoConfig.from_pretrained(cli.base, attn_implementation=cli.attn_implementation, cache_dir=cli.cache_dir)
    cfg_base.use_cache = False
    use_fast = "mpt" in cli.net.lower()
    tok = AutoTokenizer.from_pretrained(cli.tok or cli.base, local_files_only=True, trust_remote_code=True, use_fast=use_fast)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_dev = cli.diag_base_device or "cuda:0"
    cmp_dev = cli.diag_cmp_device or ("cuda:1" if torch.cuda.device_count() >= 2 else base_dev)
    logger.info(f"[diag] arch={arch} base_device={base_dev} cmp_device={cmp_dev}")

    base_model = AutoModelForCausalLM.from_pretrained(
        cli.base, config=cfg_base, torch_dtype=dtype, local_files_only=True,
        device_map={"": base_dev}, low_cpu_mem_usage=True, cache_dir=cli.cache_dir,
    ).eval()

    low_rank_config = torch.load(os.path.join(cmp_dir, "low_rank_config.pt"), weights_only=False)
    cfg_cmp = AutoConfig.from_pretrained(cmp_dir, attn_implementation=cli.attn_implementation, cache_dir=cli.cache_dir)
    cfg_cmp.use_cache = False
    if arch == "opt":
        cmp_model = OPTForCausalLM_EigenAttn.from_pretrained(
            cmp_dir, config=cfg_cmp, low_rank_config=low_rank_config, torch_dtype=dtype,
            local_files_only=True, device_map={"": cmp_dev}, low_cpu_mem_usage=True, cache_dir=cli.cache_dir,
        ).eval()
    elif arch == "mpt":
        cmp_model = MptForCausalLM_EigenAttn.from_pretrained(
            cmp_dir, config=cfg_cmp, low_rank_config=low_rank_config, torch_dtype=dtype,
            local_files_only=True, device_map={"": cmp_dev}, low_cpu_mem_usage=True, cache_dir=cli.cache_dir,
        ).eval()
    else:
        cmp_model = LlamaForCausalLM_EigenAttn.from_pretrained(
            cmp_dir, config=cfg_cmp, low_rank_config=low_rank_config, torch_dtype=dtype,
            local_files_only=True, device_map={"": cmp_dev}, low_cpu_mem_usage=True, cache_dir=cli.cache_dir,
        ).eval()

    prompts = None
    if cli.diag_prompts_file and os.path.isfile(cli.diag_prompts_file):
        with open(cli.diag_prompts_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]

    report = run_step1_diagnose(
        base_model, cmp_model, tok,
        prompts=prompts,
        seqlen=cli.diag_seqlen,
        base_device=base_dev,
        cmp_device=cmp_dev,
        log_fn=logger.info,
    )
    save_step1(report, out_json)
    logger.info(f"[pipeline] step1 saved -> {out_json}")
    del base_model, cmp_model
    torch.cuda.empty_cache()
    return report


def stage_derive(
    cli,
    *,
    step1_json: str,
    out_json: str,
    logger,
) -> Dict[str, object]:
    """Convert step1 fragility into per-layer EB schedule (freeze layer 0)."""
    with open(step1_json, "r", encoding="utf-8") as f:
        step1 = json.load(f)
    frozen = [int(x) for x in (cli.freeze_layers or "").split(",") if x.strip() != ""]
    report = derive_per_layer_eb(
        step1,
        eb_base=cli.error_budget,
        lambda_eb=cli.lambda_eb,
        normalize=cli.eb_normalize,
        freeze_layers=frozen,
        eb_min_ratio=cli.eb_min_ratio,
    )
    save_dynamic_eb(report, out_json)
    m = report["meta"]
    logger.info(
        f"[pipeline] dynamic EB -> {out_json} mean={m['mean_eb_final']:.6f} "
        f"min={m['min_eb_final']:.6f} max={m['max_eb_final']:.6f} freeze={m['freeze_layers']}"
    )
    return report


def stage_compare(
    cli,
    *,
    base_compress_info: Dict[str, object],
    base_eval: Dict[str, object],
    dyn_compress_info: Dict[str, object],
    dyn_eval: Dict[str, object],
    dynamic_eb: Dict[str, object],
    out_json: str,
    logger,
) -> Dict[str, object]:
    def _ppl(d, ds): return d.get("ppl", {}).get(ds, {}).get("ppl", float("nan"))
    def _acc_avg(d):
        accs = []
        for tname, metrics in (d.get("acc") or {}).items():
            if not isinstance(metrics, dict):
                continue
            for k in ("acc,none", "acc"):
                v = metrics.get(k)
                if isinstance(v, (int, float)):
                    accs.append(v)
                    break
        return float(sum(accs) / len(accs)) if accs else float("nan")

    rep = {
        "net": cli.net,
        "error_budget": cli.error_budget,
        "freeze_layers": [int(x) for x in (cli.freeze_layers or "").split(",") if x.strip() != ""],
        "baseline": {
            "save_dir": base_compress_info.get("save_dir"),
            "params_b": base_compress_info.get("params_b"),
            "kv_cache_gb": base_compress_info.get("kv_cache_gb"),
            "ppl_wikitext2": _ppl(base_eval, "wikitext2"),
            "ppl_c4": _ppl(base_eval, "c4"),
            "acc_bench5_avg": _acc_avg(base_eval),
            "acc_per_task": (base_eval.get("acc") or {}),
        },
        "dynamic": {
            "save_dir": dyn_compress_info.get("save_dir"),
            "params_b": dyn_compress_info.get("params_b"),
            "kv_cache_gb": dyn_compress_info.get("kv_cache_gb"),
            "ppl_wikitext2": _ppl(dyn_eval, "wikitext2"),
            "ppl_c4": _ppl(dyn_eval, "c4"),
            "acc_bench5_avg": _acc_avg(dyn_eval),
            "acc_per_task": (dyn_eval.get("acc") or {}),
        },
        "delta": {
            "kv_cache_gb": (dyn_compress_info.get("kv_cache_gb") or 0) - (base_compress_info.get("kv_cache_gb") or 0),
            "ppl_wikitext2": _ppl(dyn_eval, "wikitext2") - _ppl(base_eval, "wikitext2"),
            "ppl_c4": _ppl(dyn_eval, "c4") - _ppl(base_eval, "c4"),
            "acc_bench5_avg": _acc_avg(dyn_eval) - _acc_avg(base_eval),
        },
        "dynamic_eb_meta": dynamic_eb.get("meta", {}),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    logger.info(
        f"[pipeline] compare -> {out_json} | "
        f"PPL wiki: {rep['baseline']['ppl_wikitext2']:.4f} -> {rep['dynamic']['ppl_wikitext2']:.4f} "
        f"(\u0394{rep['delta']['ppl_wikitext2']:+.4f}) | "
        f"PPL c4: {rep['baseline']['ppl_c4']:.4f} -> {rep['dynamic']['ppl_c4']:.4f} "
        f"(\u0394{rep['delta']['ppl_c4']:+.4f}) | "
        f"ACC bench5: {rep['baseline']['acc_bench5_avg']:.4f} -> {rep['dynamic']['acc_bench5_avg']:.4f} "
        f"(\u0394{rep['delta']['acc_bench5_avg']:+.4f})"
    )
    return rep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    cli = parser.parse_args()
    cli.tok = cli.tok or cli.base

    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cli.cache_dir = cli.cache_dir or str(out_dir / "HF_cache")
    Path(cli.cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cli.cache_dir
    os.environ["HF_HOME"] = cli.cache_dir
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    set_seed(cli.seed)

    logger = utils.create_logger(out_dir)
    logger.info(f"[pipeline] argv={sys.argv}")
    logger.info(f"[pipeline] args={vars(cli)}")

    requested_stages = [s.strip() for s in cli.stages.split(",") if s.strip()]
    bad = [s for s in requested_stages if s not in ALL_STAGES]
    if bad:
        raise SystemExit(f"Unknown stages: {bad}; must be subset of {ALL_STAGES}")
    logger.info(f"[pipeline] stages={requested_stages}")

    baseline_dir = out_dir / "baseline"
    dynamic_dir = out_dir / "dynamic"
    baseline_eval_json = out_dir / "baseline_eval.json"
    dynamic_eval_json = out_dir / "dynamic_eval.json"
    step1_json = out_dir / "step1.json"
    dynamic_eb_json = out_dir / "dynamic_eb.json"
    compress_meta_base = out_dir / "baseline_compress.json"
    compress_meta_dyn = out_dir / "dynamic_compress.json"
    compare_json = out_dir / "compare.json"

    base_compress_info: Dict[str, object] = {}
    dyn_compress_info: Dict[str, object] = {}
    base_eval: Dict[str, object] = {}
    dyn_eval: Dict[str, object] = {}
    dynamic_eb: Dict[str, object] = {}

    def _read_or_skip(path: Path):
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # 1) compress_base
    if "compress_base" in requested_stages:
        if (baseline_dir / "low_rank_config.pt").is_file() and not _force_recompress(cli):
            logger.info(f"[pipeline] baseline already exists at {baseline_dir}; skipping compress_base")
            base_compress_info = _read_or_skip(compress_meta_base)
        else:
            logger.info(f"=== STAGE compress_base (uniform EB={cli.error_budget}) ===")
            base_compress_info = stage_compress(
                cli, save_dir=str(baseline_dir),
                error_budget=cli.error_budget,
                per_layer_eb=None, freeze_layers=None,
                logger=logger,
            )
            with open(compress_meta_base, "w", encoding="utf-8") as f:
                json.dump(base_compress_info, f, indent=2, ensure_ascii=False)
    else:
        base_compress_info = _read_or_skip(compress_meta_base)

    # 2) eval_base
    if "eval_base" in requested_stages:
        logger.info("=== STAGE eval_base (PPL + bench5) ===")
        base_eval = stage_eval(cli, cmp_dir=str(baseline_dir), out_json=str(baseline_eval_json), label="BASELINE", logger=logger)
    else:
        base_eval = _read_or_skip(baseline_eval_json)

    # 3) diag
    if "diag" in requested_stages:
        logger.info("=== STAGE diag (Step-1 fragility) ===")
        stage_diag(cli, cmp_dir=str(baseline_dir), out_json=str(step1_json), logger=logger)

    # 4) derive
    if "derive" in requested_stages:
        logger.info("=== STAGE derive (per-layer EB) ===")
        dynamic_eb = stage_derive(cli, step1_json=str(step1_json), out_json=str(dynamic_eb_json), logger=logger)
    else:
        dynamic_eb = _read_or_skip(dynamic_eb_json)

    # 5) compress_dyn
    if "compress_dyn" in requested_stages:
        if not dynamic_eb:
            raise SystemExit("compress_dyn requires dynamic_eb.json (run stage 'derive' first)")
        per_layer_eb = list(dynamic_eb["error_budget_by_layer"])
        frozen = [int(x) for x in (cli.freeze_layers or "").split(",") if x.strip() != ""]
        logger.info(f"=== STAGE compress_dyn (per-layer EB; freeze={frozen}) ===")
        dyn_compress_info = stage_compress(
            cli, save_dir=str(dynamic_dir),
            error_budget=cli.error_budget,
            per_layer_eb=per_layer_eb, freeze_layers=frozen,
            logger=logger,
        )
        with open(compress_meta_dyn, "w", encoding="utf-8") as f:
            json.dump(dyn_compress_info, f, indent=2, ensure_ascii=False)
    else:
        dyn_compress_info = _read_or_skip(compress_meta_dyn)

    # 6) eval_dyn
    if "eval_dyn" in requested_stages:
        logger.info("=== STAGE eval_dyn (PPL + bench5) ===")
        dyn_eval = stage_eval(cli, cmp_dir=str(dynamic_dir), out_json=str(dynamic_eval_json), label="DYNAMIC", logger=logger)
    else:
        dyn_eval = _read_or_skip(dynamic_eval_json)

    # 7) compare
    if "compare" in requested_stages:
        logger.info("=== STAGE compare ===")
        if not (base_compress_info and base_eval and dyn_compress_info and dyn_eval):
            logger.warning("[pipeline] some inputs missing for compare; reading whatever is on disk")
        stage_compare(
            cli,
            base_compress_info=base_compress_info,
            base_eval=base_eval,
            dyn_compress_info=dyn_compress_info,
            dyn_eval=dyn_eval,
            dynamic_eb=dynamic_eb,
            out_json=str(compare_json),
            logger=logger,
        )

    logger.info("[pipeline] done")


def _force_recompress(cli) -> bool:
    return bool(getattr(cli, "force_recompress", False))


if __name__ == "__main__":
    main()
