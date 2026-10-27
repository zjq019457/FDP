#!/usr/bin/env python3
"""Instrumented zero-shot runner (lm_eval localjsonl tasks) for compressed EigenAttn models."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]
_EIGEN = _REPO / "EigenAttn"
sys.path.insert(0, str(_EIGEN))
sys.path.insert(0, str(_REPO / "rebuttal_legacy_reproduction"))

from common import (  # noqa: E402
    atomic_write_json,
    file_hash_or_empty,
    merge_window_dir_to_jsonl,
    now_iso,
    sha256_file,
    sha256_json,
    sha256_text,
    write_window_atomic,
)

RUNNER_PATH = Path(__file__).resolve()
RUNNER_HASH = sha256_file(RUNNER_PATH)
ORIGINAL_TOOL = _EIGEN / "tools" / "run_lm_eval_offline_eigen_opt.py"


def _task_hashes(yaml_dir: Path, tasks: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for t in tasks:
        p = yaml_dir / f"{t}.yaml"
        out[t] = sha256_file(p) if p.is_file() else ""
    return out


def _pick_primary_acc(row: Dict[str, Any]) -> Optional[float]:
    if "acc,none" in row:
        return float(row["acc,none"])
    if "acc_norm,none" in row:
        return float(row["acc_norm,none"])
    return None


def _extract_sample_records(
    task: str,
    raw_samples: List[Dict[str, Any]],
    *,
    method: str,
    model_hash: str,
    basis_hash: str = "",
    rank_plan_hash: str = "",
    task_definition_hash: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in raw_samples:
        doc = s.get("doc") or {}
        doc_id = s.get("doc_id", len(rows))
        args = s.get("arguments") or []
        choices: List[str] = []
        for pair in args:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                choices.append(str(pair[1]).strip())
            else:
                choices.append(str(pair))
        lls: List[float] = []
        for fr in s.get("filtered_resps") or s.get("resps") or []:
            if isinstance(fr, (list, tuple)) and fr:
                v = fr[0]
                lls.append(float(v[0] if isinstance(v, (list, tuple)) else v))
        prediction = int(max(range(len(lls)), key=lambda i: lls[i])) if lls else 0
        gold = s.get("target")
        if gold is None:
            gold = doc.get("label")
        try:
            gold_i = int(gold)
        except Exception:
            gold_i = 0
        acc = s.get("acc")
        correct = bool(acc == 1.0 or acc == 1) if acc is not None else bool(prediction == gold_i)
        prompt = args[0][0] if args and isinstance(args[0], (list, tuple)) else ""
        rows.append(
            {
                "task": task,
                "sample_id": str(doc_id),
                "document_hash": sha256_json(doc),
                "prompt_hash": sha256_text(str(prompt)),
                "gold": gold_i,
                "choices": choices,
                "choice_loglikelihoods": lls,
                "normalized_scores": lls,
                "prediction": prediction,
                "correct": correct,
                "model_hash": model_hash,
                "method": method,
                "basis_hash": basis_hash,
                "rank_plan_hash": rank_plan_hash,
                "task_definition_hash": task_definition_hash,
                "runner_hash": RUNNER_HASH,
                "lm_eval_acc": float(acc) if acc is not None else (1.0 if correct else 0.0),
            }
        )
    return rows


def _load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=False, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model_dir = Path(args.model)
    if args.dense_base:
        model = AutoModelForCausalLM.from_pretrained(
            args.tokenizer_model,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation,
            local_files_only=True,
            device_map={"": "cuda:0"},
        )
        model.eval()
        return model, tok

    lr_path = (args.low_rank_config or "").strip()
    lr_json = Path(lr_path) if lr_path else model_dir / "low_rank_config.json"
    lr_pt = model_dir / "low_rank_config.pt"
    if lr_json.exists():
        low = json.loads(lr_json.read_text(encoding="utf-8"))
        if isinstance(low, dict) and "low_rank_config" in low:
            low = low["low_rank_config"]
    elif lr_pt.exists():
        pt = torch.load(lr_pt, map_location="cpu")
        low = [[int(x[0]), int(x[1])] for x in pt]
    else:
        raise FileNotFoundError(f"missing low_rank_config in {model_dir}")

    cfg = AutoConfig.from_pretrained(
        args.tokenizer_model, attn_implementation=args.attn_implementation, local_files_only=True
    )
    cfg.use_cache = False
    net_l = (args.net or getattr(cfg, "model_type", "") or "").lower()
    load_kwargs = dict(
        config=cfg,
        low_rank_config=low,
        device_map={"": "cuda:0"},
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if "qwen" in net_l:
        from models.modelling_qwen2_eigen_attn import Qwen2ForCausalLM_EigenAttn

        if not hasattr(cfg, "attention_bias"):
            cfg.attention_bias = True
        if not hasattr(cfg, "mlp_bias"):
            cfg.mlp_bias = False
        model = Qwen2ForCausalLM_EigenAttn.from_pretrained(str(model_dir), **load_kwargs)
    elif "opt" in net_l:
        from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

        model = OPTForCausalLM_EigenAttn.from_pretrained(str(model_dir), **load_kwargs)
    else:
        from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn

        if not hasattr(cfg, "attention_bias"):
            cfg.attention_bias = False
        if not hasattr(cfg, "mlp_bias"):
            cfg.mlp_bias = False
        model = LlamaForCausalLM_EigenAttn.from_pretrained(str(model_dir), **load_kwargs)
    model.eval()
    return model, tok


def _run_lm_eval(
    model,
    tok,
    tasks: List[str],
    task_yaml_dir: str,
    *,
    batch_size: int,
    cache_dir: str,
) -> Dict[str, Any]:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
    tm = TaskManager(include_path=task_yaml_dir)
    return evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        task_manager=tm,
        num_fewshot=0,
        batch_size=batch_size,
        device="cuda:0",
        log_samples=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer_model", required=True)
    ap.add_argument("--task_yaml_dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--net", default="")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--method", default="")
    ap.add_argument("--low_rank_config", default="")
    ap.add_argument("--cache_dir", default=str(_EIGEN / "HF_cache"))
    ap.add_argument("--attn_implementation", default="eager")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dense_base", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / "zero_shot_aggregate.json"
    if agg_path.exists():
        print(json.dumps({"status": "SKIP", "reason": "aggregate_exists"}))
        return

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    yaml_dir = Path(args.task_yaml_dir)
    task_hash_map = _task_hashes(yaml_dir, task_list)
    method = args.method or args.net or "dense"
    model_hash = sha256_text(args.model + "|" + file_hash_or_empty(Path(args.model) / "config.json"))

    try:
        from transformers.generation.configuration_utils import GenerationConfig

        _gc_to_json = GenerationConfig.to_json_string

        def _safe_to_json_string(self, *a, **k):  # type: ignore[no-untyped-def]
            try:
                return _gc_to_json(self, *a, **k)
            except TypeError:
                return f"{self.__class__.__name__}({{non_serializable_fields}})\n"

        GenerationConfig.to_json_string = _safe_to_json_string  # type: ignore[method-assign]
    except Exception:
        pass

    model, tok = _load_model(args)
    results = _run_lm_eval(model, tok, task_list, str(yaml_dir.resolve()), batch_size=args.batch_size, cache_dir=args.cache_dir)
    atomic_write_json(out_dir / "lm_eval_full_results.json", results)

    rdict = results.get("results") or {}
    samples_map = results.get("samples") or {}
    samples_root = out_dir / "zero_shot_samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    aggregate: Dict[str, Any] = {"tasks": {}, "method": method, "model_hash": model_hash, "runner_hash": RUNNER_HASH}

    for task in task_list:
        raw_samples = samples_map.get(task) or []
        tdef = task_hash_map.get(task, "")
        rows = _extract_sample_records(
            task,
            raw_samples,
            method=method,
            model_hash=model_hash,
            task_definition_hash=tdef,
        )
        task_out = samples_root / task
        task_out.mkdir(parents=True, exist_ok=True)
        correct_n = 0
        total_n = 0
        for rec in rows:
            write_window_atomic(task_out, rec["sample_id"], rec)
            if rec.get("correct") is not None:
                total_n += 1
                correct_n += int(bool(rec["correct"]))
        merge_window_dir_to_jsonl(task_out, samples_root / f"{task}.jsonl")
        row = rdict.get(task, {})
        lm_acc = _pick_primary_acc(row)
        recomputed = (correct_n / total_n) if total_n else float("nan")
        aggregate["tasks"][task] = {
            "n": total_n,
            "correct": correct_n,
            "accuracy_from_correct": recomputed,
            "lm_eval_primary": lm_acc,
            "lm_eval_row": row,
            "task_definition_hash": tdef,
        }
        atomic_write_json(
            task_out / "TASK.DONE",
            {"task": task, "n": total_n, "correct": correct_n, "runner_hash": RUNNER_HASH, "timestamp": now_iso()},
        )

    atomic_write_json(agg_path, aggregate)
    audit = {
        "runner": str(RUNNER_PATH),
        "runner_hash": RUNNER_HASH,
        "python": sys.executable,
        "original_tool": str(ORIGINAL_TOOL),
        "original_tool_sha256": file_hash_or_empty(ORIGINAL_TOOL),
        "task_yaml_dir": str(yaml_dir),
        "task_hashes": task_hash_map,
        "aggregate": aggregate,
        "notes": [
            "Uses historical *_localjsonl.yaml via lm_eval TaskManager include_path.",
            "log_samples=True; per-sample atomic JSON+DONE; accuracy recomputed from correct flags.",
            "Original EigenAttn sources were not modified.",
        ],
        "timestamp": now_iso(),
    }
    atomic_write_json(out_dir / "zero_shot_raw_audit.json", audit)
    # Formal reported metric must come directly from official lm_eval_primary, not reconstructed correctness
    accs = [v["lm_eval_primary"] for v in aggregate["tasks"].values() if v.get("lm_eval_primary") is not None]
    if not accs:
        accs = [v["accuracy_from_correct"] for v in aggregate["tasks"].values() if v.get("n")]
    macro = sum(accs) / len(accs) if accs else None
    print(json.dumps({"status": "DONE", "zero_shot_macro": macro, "tasks": list(aggregate["tasks"].keys())}))


def regression_test_piqa_no_false_100() -> None:
    """Regression test: wrapper must NOT produce 100% accuracy unless official lm_eval does."""
    raw_samples = [
        {"target": 0, "filtered_resps": [[[-94.0]], [[-102.0]]], "acc": 1.0},
        {"target": 1, "filtered_resps": [[[-38.0]], [[-60.0]]], "acc": 0.0},
    ]
    rows = _extract_sample_records("piqa_localjsonl", raw_samples, method="test", model_hash="test")
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
    assert rows[0]["prediction"] == 0, f"Sample 0 prediction should be 0 (argmax), got {rows[0]['prediction']}"
    assert rows[1]["prediction"] == 0, f"Sample 1 prediction should be 0 (argmax), got {rows[1]['prediction']}"
    assert rows[0]["correct"] is True, "Sample 0 must be True"
    assert rows[1]["correct"] is False, "Sample 1 must be False when acc=0"
    correct_count = sum(1 for r in rows if r["correct"])
    assert correct_count == 1, f"Expected 1 correct, got {correct_count} (must NOT be 2/2 = 100%)"
    print("[REGRESSION TEST PASS] PIQA binary accuracy is 50.0%, NOT false 100%!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        regression_test_piqa_no_false_100()
        sys.exit(0)
    main()
