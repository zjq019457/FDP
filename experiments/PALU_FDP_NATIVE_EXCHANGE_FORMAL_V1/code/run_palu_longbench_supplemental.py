#!/usr/bin/env python3
"""Supplemental PaLu LongBench cell using the frozen native schedules.

This lane is deliberately separate from the formal Wiki/C4 and zero-shot
closure artifacts. It reuses the frozen PaLu schedule and official native
conversion path, and writes all outputs to the diskE supplemental directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch


BRANCH = Path("/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1")
CODE = BRANCH / "code"
OUT = BRANCH / "longbench_supplemental_4096"
DATA = Path("/home/bai-project/jiaqizhang/EigenAttn/data/LongBench")
RECALKV = Path("/home/bai-project/jiaqizhang/ReCalKV")
Q_SHA = "6f3b59993abbffbd932825b20ff036197c5be6ff63523ab46e94281bc24eecb1"
MAX_LENGTH = 4096
TASKS = (
    "triviaqa", "qasper", "trec", "samsum",
    "lcc", "repobench-p", "qmsum", "multi_news",
)
METHODS = ("uniform", "fisher", "fdp_prior", "fdp_native")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_obj(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_or_freeze_manifest() -> dict:
    prompt_cfg = json.loads(
        (RECALKV / "longbench_utils/config/dataset2prompt.json").read_text(encoding="utf-8")
    )
    maxlen_cfg = json.loads(
        (RECALKV / "longbench_utils/config/dataset2maxlen.json").read_text(encoding="utf-8")
    )
    task_files = {}
    for task in TASKS:
        path = DATA / f"{task}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        n = sum(1 for _ in path.open(encoding="utf-8"))
        task_files[task] = {
            "path": str(path),
            "n_samples": n,
            "sha256": sha_file(path),
            "prompt": prompt_cfg[task],
            "max_new_tokens": int(maxlen_cfg[task]),
        }
    scorer = RECALKV / "longbench_utils/scorer.py"
    metrics = RECALKV / "longbench_utils/metrics.py"
    manifest = {
        "status": "PALU_LONGBENCH_SUPPLEMENTAL_MANIFEST_FROZEN",
        "campaign": "PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1_LONGBENCH_SUPPLEMENTAL",
        "model": "/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf",
        "ratio": 0.5,
        "tasks": list(TASKS),
        "n_tasks": len(TASKS),
        "n_samples": sum(x["n_samples"] for x in task_files.values()),
        "task_files": task_files,
        "max_length": MAX_LENGTH,
        "generation": {"do_sample": False, "num_beams": 1, "temperature": 1.0, "seed": 42},
        "q_sha": Q_SHA,
        "prompt_sha": sha_obj({t: task_files[t]["prompt"] for t in TASKS}),
        "scorer_sha": sha_file(scorer),
        "metrics_sha": sha_file(metrics),
        "dataset_manifest_sha": sha_obj({t: task_files[t]["sha256"] for t in TASKS}),
    }
    manifest["manifest_sha"] = sha_obj({k: v for k, v in manifest.items() if k != "manifest_sha"})
    path = OUT / "LONGBENCH_MANIFEST.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("manifest_sha") != manifest["manifest_sha"]:
            raise RuntimeError("LongBench manifest changed after freeze")
        return old
    write_json(path, manifest)
    return manifest


def load_rows(task: str) -> list[dict]:
    rows = []
    with (DATA / f"{task}.jsonl").open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            obj = json.loads(line)
            obj["_sample_id"] = obj.get("_id") or f"{task}:{index}"
            obj["_row_index"] = index
            rows.append(obj)
    return rows


def generate_one(model, tokenizer, obj: dict, task: str, prompt_format: str, max_new_tokens: int):
    prompt = prompt_format.format(**obj)
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) > MAX_LENGTH:
        half = MAX_LENGTH // 2
        prompt = tokenizer.decode(tokenized[:half], skip_special_tokens=True) + tokenizer.decode(
            tokenized[-half:], skip_special_tokens=True
        )
    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
    ctx = inputs.input_ids.shape[-1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )[0]
    return tokenizer.decode(output[ctx:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, choices=[0.5], default=0.5)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(RECALKV))

    manifest = load_or_freeze_manifest()
    from run_palu_formal_quality import load_runtime
    from longbench_utils.scorer import dataset2metric

    model, tokenizer, schedule_path, schedule_doc, selection, _wiki, _c4 = load_runtime(
        args.ratio, args.method
    )
    raw_path = OUT / f"ratio_{args.ratio:.1f}" / f"PALU_{args.method.upper()}" / f"{args.task}.raw.jsonl"
    cell_path = raw_path.with_suffix(".json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if raw_path.is_file():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("valid"):
                    done[rec["sample_id"]] = rec

    rows = load_rows(args.task)
    prompt_format = manifest["task_files"][args.task]["prompt"]
    max_new_tokens = manifest["task_files"][args.task]["max_new_tokens"]
    scores = []
    with raw_path.open("a", encoding="utf-8") as append:
        for obj in rows:
            sid = obj["_sample_id"]
            if sid in done:
                scores.append(float(done[sid]["score"]))
                continue
            prediction = generate_one(model, tokenizer, obj, args.task, prompt_format, max_new_tokens)
            score = max(
                float(dataset2metric[args.task](prediction, gt, all_classes=obj.get("all_classes")))
                for gt in obj["answers"]
            )
            rec = {
                "method": f"PALU_{args.method.upper()}",
                "ratio": args.ratio,
                "task": args.task,
                "sample_id": sid,
                "row_index": obj["_row_index"],
                "prediction": prediction,
                "reference": obj["answers"],
                "score": score,
                "valid": True,
                "q_sha": Q_SHA,
                "schedule_sha": schedule_doc["schedule_sha"],
                "native_rank": int(sum(sum(v) for v in selection.values())),
            }
            append.write(json.dumps(rec, ensure_ascii=False) + "\n")
            append.flush()
            scores.append(score)

    if len(scores) != len(rows) or not all(math.isfinite(x) for x in scores):
        write_json(cell_path, {"validity": "INCOMPLETE", "n_scored": len(scores), "n_expected": len(rows)})
        raise SystemExit(3)
    result = {
        "validity": "VALID",
        "campaign": "PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1_LONGBENCH_SUPPLEMENTAL",
        "method": f"PALU_{args.method.upper()}",
        "ratio": args.ratio,
        "task": args.task,
        "score": round(100.0 * sum(scores) / len(scores), 2),
        "n_samples": len(scores),
        "manifest_sha": manifest["manifest_sha"],
        "schedule_sha": schedule_doc["schedule_sha"],
        "q_sha": Q_SHA,
        "native_rank": int(sum(sum(v) for v in selection.values())),
        "formal_cache_bytes": int(sum(sum(v) for v in selection.values()) * 2048 * 2),
        "gpu": args.gpu,
    }
    write_json(cell_path, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
