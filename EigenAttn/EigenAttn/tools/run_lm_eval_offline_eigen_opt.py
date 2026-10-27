#!/usr/bin/env python3
"""
离线 lm-eval：加载 OPT EigenAttn 压缩 checkpoint（可与 checkpoint 同目录的 low_rank_config，或通过 --low_rank_config 指定），并在自定义 YAML 任务上评测。

依赖你在 offline_datasets 下准备好的 *_localjsonl.yaml（dataset_path 指向本地 jsonl）。

示例任务名（与 YAML 内 task: 字段一致）：
  piqa_localjsonl, winogrande_localjsonl, arc_easy_localjsonl,
  arc_challenge_localjsonl, hellaswag_localjsonl

示例（与仓库内 LMClass + vendored lm-eval 对齐）：

  cd /path/to/EigenAttn/EigenAttn && \\
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 HF_DATASETS_OFFLINE=1 \\
  PYTHONPATH=$(pwd):$PYTHONPATH \\
  python -u tools/run_lm_eval_offline_eigen_opt.py \\
    --compressed_model /diskF/jiaqizhang/5712/final_model \\
    --low_rank_config /home/bai-project/jiaqizhang/step1_out/low_rank_from_eigen_basis_diskF.json \\
    --tokenizer_model /diskE/opt-30b \\
    --task_yaml_dir /diskF/jiaqizhang/a1804298693/offline_datasets \\
    --batch_size 1 \\
    --output_json /diskF/jiaqizhang/5712/lm_eval_offline_results.json

首次运行若缺少 HF metric 缓存，可暂时设 HF_DATASETS_OFFLINE=0 以下载 evaluate 依赖。

稠密基座（与 Eigen 压缩无关的 HF 权重）加 --dense_base，仅需 --tokenizer_model 指向模型目录，例如 Llama-3：

  python -u tools/run_lm_eval_offline_eigen_opt.py \\
    --dense_base \\
    --tokenizer_model /diskD/jiaqizhang/Llama-3 \\
    --net llama-3-8b \\
    --dtype bf16 \\
    --task_yaml_dir ... \\
    --tasks piqa_localjsonl \\
    --batch_size 1 \\
    --eval_max_length 512

显存紧张时可：
  - 换空闲卡：CUDA_VISIBLE_DEVICES=3（nvidia-smi 看占用）
  - 限制上下文（显著降激活显存）：--eval_max_length 1024 或 512
  - 多卡切层（需多张可见 GPU）：加 --multigpu（OPT：`decoder.layers`；Llama：`model.layers`）
  - 缓解碎片：PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.LMClass import LMClass  # noqa: E402
from parallel_utils import map_layers_to_multi_gpus  # noqa: E402


def _place_opt_single_gpu(lm, device: torch.device):
    lm.model = lm.model.to(device)
    lm._device = next(lm.model.parameters()).device


def _place_model_multigpu(lm, net: str, logger: logging.Logger):
    """
    Shard transformer blocks across visible GPUs (same helper as OPT training/eval).
    OPT: model.model.decoder.layers; Llama: model.model.layers.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable; skip multigpu.")
        return

    nl = net.lower()
    root = lm.model
    inner = root.model

    if "opt" in nl:
        dec = inner.decoder
        layer_gpu_map = map_layers_to_multi_gpus(dec.layers, logger=logger)
        input_device = dec.layers[0].device
        output_device = dec.layers[-1].device
        lm._device = input_device
        assert input_device == output_device, (input_device, output_device)
        dec.embed_positions.to(input_device)
        dec.embed_tokens.to(input_device)
        dec.final_layer_norm.to(output_device)
        root.lm_head.to(output_device)
        del layer_gpu_map
        return

    if "llama" in nl:
        layers = inner.layers
        layer_gpu_map = map_layers_to_multi_gpus(layers, logger=logger)
        input_device = layers[0].device
        output_device = layers[-1].device
        lm._device = input_device
        assert input_device == output_device, (input_device, output_device)
        inner.embed_tokens.to(input_device)
        rotary = getattr(inner, "rotary_emb", None)
        if rotary is not None:
            rotary.to(input_device)
        inner.norm.to(output_device)
        root.lm_head.to(output_device)
        del layer_gpu_map
        return

    raise NotImplementedError(
        f"--multigpu is only wired for OPT and Llama (got net={net!r}). "
        "Use a single GPU or extend _place_model_multigpu."
    )


def _pick_primary_acc(task_results: dict) -> float | None:
    """Prefer acc,none; for ARC-style tasks fall back to acc_norm,none if missing."""
    if "acc,none" in task_results:
        return float(task_results["acc,none"])
    if "acc_norm,none" in task_results:
        return float(task_results["acc_norm,none"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compressed_model",
        default="",
        help="final_model 目录（Eigen 低秩权重；与 --dense_base 二选一）",
    )
    ap.add_argument(
        "--dense_base",
        action="store_true",
        help="评测原始 HuggingFace 稠密基座：仅用 --tokenizer_model 作为模型目录，忽略 --compressed_model / --low_rank_config。",
    )
    ap.add_argument(
        "--low_rank_config",
        default="",
        help="显式指定 low_rank_config.json 或 .pt（当 final_model 下没有侧车时必传），例如 step1_out/low_rank_from_eigen_basis_diskF.json",
    )
    ap.add_argument("--tokenizer_model", required=True, help="基座 tokenizer/config，例如 /diskE/opt-30b")
    ap.add_argument(
        "--task_yaml_dir",
        required=True,
        help="含 *_localjsonl.yaml 的目录（传给 lm_eval TaskManager include_path）",
    )
    ap.add_argument(
        "--tasks",
        default="piqa_localjsonl,winogrande_localjsonl,arc_easy_localjsonl,arc_challenge_localjsonl,hellaswag_localjsonl",
        help="逗号分隔的任务名，必须与 YAML 内 task: 字段一致",
    )
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument(
        "--eval_max_length",
        type=int,
        default=0,
        help="限制 lm-eval 有效上下文长度（从左侧截断，与 BaseLM 一致）；0 表示使用模型 max_position_embeddings。"
        " 例如 1024/512 可明显降低激活显存，略影响长上下文任务分数。",
    )
    ap.add_argument("--cache_dir", default="./HF_cache")
    ap.add_argument("--net", default="opt-30b")
    ap.add_argument("--attn_implementation", default="eager")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--multigpu", action="store_true")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--output_json", default="", help="若给定，将 lm-eval 原始 results 写入该路径")
    args_cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("offline_lm_eval")

    if args_cli.dense_base:
        if (args_cli.compressed_model or "").strip():
            logger.warning("--dense_base: ignoring --compressed_model %s", args_cli.compressed_model)
    elif not (args_cli.compressed_model or "").strip():
        ap.error("需要提供 --compressed_model，或使用 --dense_base 评测稠密基座")

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", args_cli.cache_dir)
    os.environ.setdefault("HF_HOME", args_cli.cache_dir)

    lr_path = (args_cli.low_rank_config or "").strip()
    eval_cap = args_cli.eval_max_length if args_cli.eval_max_length > 0 else None
    load_lr = not args_cli.dense_base
    lm_ns = SimpleNamespace(
        model=args_cli.tokenizer_model,
        save_dir=(args_cli.compressed_model if load_lr else ""),
        load_low_rank=load_lr,
        low_rank_config_path=(lr_path if load_lr else ""),
        attn_implementation=args_cli.attn_implementation,
        cache_dir=args_cli.cache_dir,
        batch_size=args_cli.batch_size,
        net=args_cli.net,
        load_peft_model=False,
        peft_model_path=None,
        attention_calib_pt=None,
        eval_max_length=eval_cap,
    )

    if eval_cap is not None:
        logger.info("eval_max_length=%s (truncates long prompts from the left)", eval_cap)
    logger.info(
        "Loading %s via LMClass...",
        "dense base model" if args_cli.dense_base else "compressed (low-rank) model",
    )
    lm = LMClass(lm_ns)

    dt = torch.float16 if args_cli.dtype == "fp16" else torch.bfloat16
    try:
        lm.model = lm.model.to(dtype=dt)
    except Exception as e:
        logger.warning("dtype cast failed: %s", e)

    if args_cli.multigpu:
        _place_model_multigpu(lm, args_cli.net, logger)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Need CUDA for single-GPU eval or pass --multigpu with GPUs.")
        _place_opt_single_gpu(lm, torch.device("cuda:0"))

    lm.model.eval()
    for p in lm.model.parameters():
        p.requires_grad = False

    task_list = [t.strip() for t in args_cli.tasks.split(",") if t.strip()]
    yaml_dir = str(Path(args_cli.task_yaml_dir).resolve())

    import lm_eval  # noqa: WPS433

    tm = lm_eval.tasks.TaskManager(include_path=yaml_dir)

    logger.info("Running lm-eval tasks=%s include_path=%s", task_list, yaml_dir)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args_cli.num_fewshot,
        batch_size=args_cli.batch_size,
        limit=(None if args_cli.limit < 0 else args_cli.limit),
        task_manager=tm,
        log_samples=False,
    )

    rdict = results.get("results", {})
    print("\n=== Per-task (primary acc) ===")
    accs: list[float] = []
    for name in task_list:
        row = rdict.get(name, {})
        a = _pick_primary_acc(row)
        metric = "acc,none" if "acc,none" in row else ("acc_norm,none" if "acc_norm,none" in row else "n/a")
        print(f"  {name}: {a}  (used {metric})")
        if a is not None:
            accs.append(a)

    avg = sum(accs) / len(accs) if accs else float("nan")
    print(f"\nAvg-Acc (mean of {len(accs)} tasks): {avg:.6f}\n")

    # Optional extras for ARC
    for arc_name in ("arc_easy_localjsonl", "arc_challenge_localjsonl"):
        row = rdict.get(arc_name, {})
        if not row:
            continue
        if "acc_norm,none" in row:
            print(f"  [{arc_name}] acc_norm = {row['acc_norm,none']:.6f}")

    if args_cli.output_json:
        out_path = Path(args_cli.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, default=str, indent=2)
        logger.info("Wrote full lm-eval payload to %s", out_path)

    pprint(rdict)


if __name__ == "__main__":
    main()
