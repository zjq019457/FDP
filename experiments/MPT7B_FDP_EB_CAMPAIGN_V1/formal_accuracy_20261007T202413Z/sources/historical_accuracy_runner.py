#!/usr/bin/env python3
"""Five-task zero-shot accuracy for a frozen MPT tier arm.

Uses the frozen local task YAMLs, MPT tokenizer, and lm_eval loglikelihood.
Primary metric is acc_norm when that task lists it, otherwise acc.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


PPL_SCRIPT = Path("/home/bai-project/jiaqizhang/EigenAttn/experimental/mpt7b_fdp_eb_pipeline_v1/run_u011_diagnostic_ppl.py")
INCLUDE = Path("/diskF/jiaqizhang/a1804298693/offline_datasets")
TASKS = [
    "arc_challenge_localjsonl",
    "arc_easy_localjsonl",
    "hellaswag_localjsonl",
    "piqa_localjsonl",
    "winogrande_localjsonl",
]
YAML_PRIMARY = {
    "arc_challenge_localjsonl": "acc_norm",
    "arc_easy_localjsonl": "acc_norm",
    "hellaswag_localjsonl": "acc",
    "piqa_localjsonl": "acc",
    "winogrande_localjsonl": "acc",
}


def load_ppl():
    spec = importlib.util.spec_from_file_location("mpt_ppl_for_acc", PPL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def arm_dir(ppl, tier: str, arm: str) -> Path:
    paths = ppl.TIER_PATHS[tier]
    if arm == "fdp":
        base = paths["fdp_out"].parents[1]
    elif arm == "relaxed_uniform":
        base = paths["relaxed_lock"].parents[1]
    else:
        raise SystemExit(arm)
    return base / "11_ACCURACY" / arm


def score_task(lm, tokenizer, task: str, device: str) -> dict:
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=str(INCLUDE))
    res = evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        task_manager=tm,
        num_fewshot=0,
        batch_size=1,
        device=device,
        log_samples=True,
        limit=None,
        cache_requests=False,
        bootstrap_iters=0,
        random_seed=2,
        numpy_random_seed=2,
        torch_random_seed=2,
        fewshot_random_seed=2,
    )
    raw = res["results"][task]
    primary = YAML_PRIMARY[task]
    key = f"{primary},none"
    if key not in raw:
        raise SystemExit(f"{task} missing {key}; keys={sorted(raw)}")
    samples = []
    for row in res["samples"][task]:
        samples.append({
            "doc_id": row.get("doc_id"),
            "acc": row.get("acc"),
            "acc_norm": row.get("acc_norm"),
        })
    if len(samples) == 0:
        raise SystemExit(f"{task} produced no samples")
    return {
        "task": task,
        "primary_metric": primary,
        "primary_score": float(raw[key]),
        "raw": {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))},
        "n": len(samples),
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["U011", "U015", "U019"])
    ap.add_argument("--arm", required=True, choices=["fdp", "relaxed_uniform"])
    ap.add_argument("--memory-fraction", type=float, default=0.40)
    args = ap.parse_args()
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_CACHE": "/dev/shm/mpt_acc_datasets",
        "TOKENIZERS_PARALLELISM": "false",
    })
    Path("/dev/shm/mpt_acc_datasets").mkdir(parents=True, exist_ok=True)
    ppl = load_ppl()
    out = arm_dir(ppl, args.tier, args.arm)
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device("cuda:0")
    free_b, total_b = torch.cuda.mem_get_info(device)
    if free_b < 36 * 1024**3:
        raise SystemExit(f"GPU free {free_b} < 36 GiB")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2)
    model, rank_sha = ppl.realize(args.arm, device, ppl.TIER_PATHS[args.tier])
    from transformers import AutoTokenizer
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(str(ppl.MODEL), local_files_only=True, use_fast=True, trust_remote_code=False)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1, max_length=2048, logits_cache=False, add_bos_token=False)
    done = {}
    for task in TASKS:
        path = out / f"{task}.json"
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("rank_schedule_sha256") == rank_sha and prior.get("status") == "PASS":
                done[task] = prior["primary_score"]
                print(f"[{args.tier} {args.arm}] reuse {task} {prior['primary_score']:.6f}", flush=True)
                continue
        print(f"[{args.tier} {args.arm}] start {task}", flush=True)
        scored = score_task(lm, tok, task, "cuda:0")
        scored.update({
            "status": "PASS",
            "tier": args.tier,
            "arm": args.arm,
            "rank_schedule_sha256": rank_sha,
            "quality_used_to_select_schedule": False,
        })
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(scored) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        done[task] = scored["primary_score"]
        print(f"[{args.tier} {args.arm}] {task} {scored['primary_metric']}={scored['primary_score']:.6f} n={scored['n']}", flush=True)
    macro = sum(done[t] for t in TASKS) / len(TASKS)
    summary = {
        "schema": f"MPT7B_{args.tier}_DIAGNOSTIC_ACCURACY_V1",
        "tier": args.tier,
        "arm": args.arm,
        "status": "ACCURACY_COMPLETE",
        "metric_policy": "acc_norm if listed in the frozen task YAML, otherwise acc; equal-weight macro",
        "macro": macro,
        "per_task": done,
        "rank_schedule_sha256": rank_sha,
        "formal_matched_quality": False,
        "quality_used_to_select_schedule": False,
        "gpu_free_at_start_bytes": int(free_b),
        "gpu_total_bytes": int(total_b),
    }
    (out / "ACC_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"tier": args.tier, "arm": args.arm, "macro": macro, "per_task": done}, indent=2), flush=True)


if __name__ == "__main__":
    main()
