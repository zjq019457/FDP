"""bench5 (0-shot, localjsonl) accuracy evaluation via lm-eval-harness.

The 5 tasks are the ones the project already ran for the dense Llama-3 baseline
(see ``/diskD/jiaqizhang/Llama-3/lm_eval_bench5_base.json``):

  * ``arc_challenge_localjsonl`` (acc, acc_norm)
  * ``arc_easy_localjsonl``      (acc, acc_norm)
  * ``hellaswag_localjsonl``     (acc)
  * ``piqa_localjsonl``          (acc)
  * ``winogrande_localjsonl``    (acc)

Each task is defined by a YAML in ``include_path`` with ``dataset_path: json`` +
``data_files: { train: ..., validation: ... }`` pointing at offline JSONL shards.
The default ``include_path`` is ``/diskF/.../offline_datasets`` where the YAMLs
already exist. No Hub access is performed (we set ``HF_DATASETS_OFFLINE=1`` and
``HF_HUB_OFFLINE=1`` so any accidental download fails fast).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_BENCH5_TASKS: List[str] = [
    "arc_challenge_localjsonl",
    "arc_easy_localjsonl",
    "hellaswag_localjsonl",
    "piqa_localjsonl",
    "winogrande_localjsonl",
]
DEFAULT_INCLUDE_PATH = os.environ.get(
    "BENCH5_INCLUDE_PATH", "/diskF/jiaqizhang/a1804298693/offline_datasets"
)


def _force_offline_env() -> None:
    """Hard-wire HF/datasets to never go online during ACC eval."""
    for k in (
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        os.environ[k] = "1"


def _flatten_results(t_results: dict) -> Dict[str, Dict[str, float]]:
    """Reduce ``lm_eval.simple_evaluate`` output to ``{task: {metric: value}}``."""
    out: Dict[str, Dict[str, float]] = {}
    for tname, tdict in t_results.get("results", {}).items():
        cleaned: Dict[str, float] = {}
        for k, v in tdict.items():
            if k.startswith("alias"):
                continue
            cleaned[k] = float(v) if isinstance(v, (int, float)) else v
        out[tname] = cleaned
    return out


def evaluate_bench5(
    lm,
    *,
    tasks: Optional[Iterable[str]] = None,
    include_path: str = DEFAULT_INCLUDE_PATH,
    num_fewshot: int = 0,
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
    log_fn=print,
) -> Dict[str, dict]:
    """Run lm-eval-harness ``simple_evaluate`` on the 5 localjsonl tasks.

    ``lm`` must be a configured ``LMClass`` (or any HF lm-eval LM) that already
    holds the model + tokenizer; we don't reconstruct it here so the same
    instance can be reused across PPL / fragility / ACC.

    Returns ``{"results": {task: metrics}, "raw": full_simple_evaluate_dict}``.
    """
    _force_offline_env()
    tasks = list(tasks or DEFAULT_BENCH5_TASKS)

    inc = Path(include_path)
    if not inc.is_dir():
        raise FileNotFoundError(
            f"[acc] BENCH5_INCLUDE_PATH not found: {inc}; pass --bench5_include_path or set env var"
        )
    missing = [t for t in tasks if not (inc / f"{t}.yaml").is_file()]
    if missing:
        raise FileNotFoundError(
            f"[acc] missing localjsonl YAMLs under {inc}: {missing}"
        )

    log_fn(f"[acc] include_path={inc}")
    log_fn(f"[acc] tasks={tasks} num_fewshot={num_fewshot} batch_size={batch_size} limit={limit}")

    import lm_eval

    task_manager = lm_eval.tasks.TaskManager(include_path=str(inc))
    t_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        task_manager=task_manager,
        batch_size=batch_size,
        limit=limit,
    )

    results = _flatten_results(t_results)
    for tname, metrics in results.items():
        flat = " ".join(f"{k}={v}" for k, v in metrics.items())
        log_fn(f"[acc] {tname}: {flat}")
    return {"results": results, "raw": t_results}


def save_acc_results(report: Dict[str, dict], out_json: str) -> None:
    """Persist ACC results in a stable schema (avoid raw lm-eval blob bloat)."""
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    raw = report.get("raw", {})
    serializable = {
        "results": report.get("results", {}),
        "n-shot": raw.get("n-shot", {}),
        "config": raw.get("config", {}),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


__all__ = [
    "DEFAULT_BENCH5_TASKS",
    "DEFAULT_INCLUDE_PATH",
    "evaluate_bench5",
    "save_acc_results",
]
