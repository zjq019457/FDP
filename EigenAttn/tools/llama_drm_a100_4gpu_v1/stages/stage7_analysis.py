#!/usr/bin/env python3
"""Stage 7: offline aggregation and report export for LLAMA_DRM_A100_4GPU_V1.

CPU only, no model weights. Reads each finished job's `metrics.json` (which carries the
official per-task LongBench score computed by the runner's own scorer) and produces the
required tables. Report files are always written: when predictions are missing the rows
are emitted with an explicit `incomplete` marker rather than a silently omitted arm, so
a partially finished run is never mistaken for a complete one.

Macro averaging follows the protocol: score each task independently, then take the
arithmetic mean over tasks. Pooled sample averaging is forbidden.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C

STAGE_DIR_BY_SCOPE: Dict[str, str] = {
    "six_task": "stage2_six_task_main",
    "full_17": "stage3_full_17_task",
    "fullkv": "stage3_full_17_task",  # Full-KV is an arm of the 17-task stage
    "oom_fill": "stage4_oom_accuracy_completion",
    "ablations": "stage5_ablations",
    "efficiency": "stage6_efficiency",
}


def _task_score(run_dir: Path, task: str) -> Tuple[Optional[float], int, str]:
    """(score, prediction_count, status) for one arm/task output directory."""
    pred_path = run_dir / "longbench_predictions" / f"{task}.jsonl"
    count = C.count_jsonl_records(pred_path)
    metrics = C.read_json(run_dir / "metrics.json", default={}) or {}
    rows = (metrics.get("longbench") or {}).get("rows") or []
    for row in rows:
        if row.get("dataset") == task:
            score = row.get("score")
            if isinstance(score, (int, float)):
                return float(score), count, str(row.get("status", "ok"))
    return None, count, "incomplete" if count == 0 else "partial"


def _collect_arm(stage_dir: Path, tasks: List[str]) -> Dict[str, Dict[str, Any]]:
    """arm_name -> {task: {score, predictions, status}} for every arm under stage_dir."""
    arms: Dict[str, Dict[str, Any]] = {}
    if not stage_dir.is_dir():
        return arms
    for arm_dir in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
        per_task: Dict[str, Any] = {}
        for task in tasks:
            run_dir = arm_dir / task
            if not run_dir.exists():
                per_task[task] = {"score": None, "predictions": 0, "status": "missing"}
                continue
            score, count, status = _task_score(run_dir, task)
            per_task[task] = {"score": score, "predictions": count, "status": status}
        arms[arm_dir.name] = per_task
    return arms


def _macro_average(per_task: Dict[str, Any], tasks: List[str]) -> Tuple[Optional[float], int]:
    """Arithmetic mean over per-task scores. None until every task has a score."""
    scores = [per_task.get(task, {}).get("score") for task in tasks]
    present = [value for value in scores if isinstance(value, (int, float))]
    if len(present) != len(tasks):
        return None, len(present)
    return round(sum(present) / len(present), 4), len(present)


def _expected_samples() -> Dict[str, int]:
    manifest = C.read_json(C.PROTOCOL_DIR / "task_manifest.json", default={}) or {}
    return {
        entry["task"]: int(entry.get("sample_count", 0))
        for entry in manifest.get("tasks", [])
        if entry.get("task")
    }


def _quality_table(result_dir: Path, scope: str, tasks: List[str]) -> Dict[str, Any]:
    stage_dir = result_dir / "runs" / STAGE_DIR_BY_SCOPE[scope]
    arms = _collect_arm(stage_dir, tasks)
    expected = _expected_samples()
    rows: List[Dict[str, Any]] = []
    for arm, per_task in sorted(arms.items()):
        macro, complete_tasks = _macro_average(per_task, tasks)
        total_predictions = sum(int(entry["predictions"]) for entry in per_task.values())
        total_expected = sum(expected.get(task, 0) for task in tasks)
        row: Dict[str, Any] = {
            "arm": arm,
            "macro_avg": macro if macro is not None else "",
            "tasks_scored": complete_tasks,
            "tasks_total": len(tasks),
            "predictions": total_predictions,
            "predictions_expected": total_expected,
            "complete": macro is not None and total_predictions >= total_expected,
        }
        for task in tasks:
            score = per_task.get(task, {}).get("score")
            row[task] = score if isinstance(score, (int, float)) else ""
        rows.append(row)
    return {
        "scope": scope,
        "tasks": tasks,
        "macro_rule": "score each task independently, then arithmetic mean over tasks",
        "pooled_sample_average_forbidden": True,
        "rows": rows,
    }


def _efficiency_table(result_dir: Path) -> Dict[str, Any]:
    stage_dir = result_dir / "runs" / STAGE_DIR_BY_SCOPE["efficiency"]
    rows: List[Dict[str, Any]] = []
    if stage_dir.is_dir():
        for arm_dir in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
            metrics = C.read_json(arm_dir / "metrics.json", default={}) or {}
            for latency_row in (metrics.get("latency") or {}).get("rows", []):
                rows.append({
                    "arm": arm_dir.name,
                    "context_length": latency_row.get("context_length", ""),
                    "ttft_s": latency_row.get("ttft_s", ""),
                    "e2e_s": latency_row.get("e2e_s", ""),
                    "peak_gpu_mem_GB": latency_row.get("peak_gpu_mem_GB", ""),
                    "prompt_kv_mem_GB": latency_row.get("prompt_kv_mem_GB", ""),
                })
    return {"scope": "efficiency", "rows": rows}


def _worker_summary(result_dir: Path) -> Dict[str, Any]:
    """OOM / failure accounting straight from the on-disk worker results."""
    counts = {"COMPLETED": 0, "FAILED": 0, "OOM": 0, "RUNNING": 0}
    oom_jobs: List[str] = []
    peaks: List[float] = []
    for path in (result_dir / "runs").rglob("worker_result.json"):
        payload = C.read_json(path, default={}) or {}
        state = str(payload.get("state", ""))
        counts[state] = counts.get(state, 0) + 1
        if payload.get("oom"):
            oom_jobs.append(str(payload.get("job_id", path.parent.name)))
        peak = payload.get("peak_mem_gib")
        if isinstance(peak, (int, float)) and peak > 0:
            peaks.append(float(peak))
    return {
        "state_counts": counts,
        "oom_jobs": sorted(set(oom_jobs)),
        "observed_peak_gib_max": round(max(peaks), 3) if peaks else None,
        "observed_peak_gib_mean": round(sum(peaks) / len(peaks), 3) if peaks else None,
    }


def _write_markdown_table(path: Path, title: str, table: Dict[str, Any],
                          columns: List[str]) -> None:
    lines = [f"# {title}", "", f"- protocol: {C.PROTOCOL_ID}",
             f"- generated: {C.now_iso()}"]
    if table.get("macro_rule"):
        lines.append(f"- macro rule: {table['macro_rule']}")
        lines.append("- pooled sample averaging: FORBIDDEN")
    lines += ["", "| " + " | ".join(columns) + " |",
              "| " + " | ".join("---" for _ in columns) + " |"]
    for row in table.get("rows", []):
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    if not table.get("rows"):
        lines.append("| " + " | ".join("(no data yet)" for _ in columns) + " |")
    C.atomic_write_text(path, "\n".join(lines) + "\n")


def run(result_dir: Path) -> Dict[str, Any]:
    result_dir = Path(result_dir)
    reports = result_dir / "reports"
    analysis = result_dir / "analysis"
    reports.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)

    six = _quality_table(result_dir, "six_task", list(C.SIX_TASKS))
    full = _quality_table(result_dir, "full_17", list(C.SEVENTEEN_TASKS))
    fullkv = _quality_table(result_dir, "fullkv", list(C.SEVENTEEN_TASKS))
    ablations = _quality_table(result_dir, "ablations", list(C.SIX_TASKS))
    efficiency = _efficiency_table(result_dir)
    workers = _worker_summary(result_dir)

    written: List[str] = []

    for name, table, tasks in (
        ("SIX_TASK_MAIN_COMPARISON", six, list(C.SIX_TASKS)),
        ("FULL_17_TASK_COMPARISON", full, list(C.SEVENTEEN_TASKS)),
        ("FULLKV_BASELINE", fullkv, list(C.SEVENTEEN_TASKS)),
        ("ABLATIONS", ablations, list(C.SIX_TASKS)),
    ):
        columns = ["arm", "macro_avg", "tasks_scored", "tasks_total", "predictions",
                   "predictions_expected", "complete"] + tasks
        C.atomic_write_json(analysis / f"{name}.json", table)
        C.atomic_write_csv(analysis / f"{name}.csv", table["rows"], columns)
        _write_markdown_table(reports / f"{name}.md", name.replace("_", " ").title(),
                              table, columns)
        written += [f"{name}.json", f"{name}.csv", f"{name}.md"]

    eff_columns = ["arm", "context_length", "ttft_s", "e2e_s", "peak_gpu_mem_GB",
                   "prompt_kv_mem_GB"]
    C.atomic_write_json(analysis / "EFFICIENCY.json", efficiency)
    C.atomic_write_csv(analysis / "EFFICIENCY.csv", efficiency["rows"], eff_columns)
    _write_markdown_table(reports / "EFFICIENCY.md", "Efficiency", efficiency, eff_columns)
    written += ["EFFICIENCY.json", "EFFICIENCY.csv", "EFFICIENCY.md"]

    summary = {
        "protocol_id": C.PROTOCOL_ID,
        "generated_at": C.now_iso(),
        "result_dir": str(result_dir),
        "frozen_config": C.FROZEN_LLAMA_CONFIG,
        "method_cli_map": C.METHOD_CLI_MAP,
        "six_task_macro": {row["arm"]: row["macro_avg"] for row in six["rows"]},
        "full_17_macro": {row["arm"]: row["macro_avg"] for row in full["rows"]},
        "fullkv_macro": {row["arm"]: row["macro_avg"] for row in fullkv["rows"]},
        "workers": workers,
        "tables_written": written,
    }
    C.atomic_write_json(analysis / "FINAL_SUMMARY.json", summary)

    headline = [
        f"# {C.PROTOCOL_ID} final report", "",
        f"- generated: {C.now_iso()}",
        f"- model: {C.MODEL_NAME} (revision {C.MODEL_REVISION})",
        f"- backend: {C.FROZEN_LLAMA_CONFIG['backend']} "
        f"({C.FROZEN_LLAMA_CONFIG['backend_reason']})",
        f"- prune layers: {C.FROZEN_LLAMA_CONFIG['prune_layers']}",
        f"- retain tokens: {C.FROZEN_LLAMA_CONFIG['retain_tokens']}",
        f"- rescue M per prune layer: {list(C.RESCUE_M_PER_PRUNE_LAYER)} "
        f"at ratio {C.FROZEN_LLAMA_CONFIG['rescue_ratio_main']}",
        "",
        "## Six-task macro averages", "",
    ]
    for row in six["rows"]:
        headline.append(f"- {row['arm']}: {row['macro_avg'] or 'incomplete'} "
                        f"({row['tasks_scored']}/{row['tasks_total']} tasks)")
    headline += ["", "## 17-task macro averages", ""]
    for row in full["rows"] + fullkv["rows"]:
        headline.append(f"- {row['arm']}: {row['macro_avg'] or 'incomplete'} "
                        f"({row['tasks_scored']}/{row['tasks_total']} tasks)")
    headline += ["", "## Worker health", "",
                 f"- states: {workers['state_counts']}",
                 f"- OOM jobs: {workers['oom_jobs'] or 'none'}",
                 f"- observed peak (max): {workers['observed_peak_gib_max']} GiB"]
    C.atomic_write_text(reports / "FINAL_REPORT.md", "\n".join(headline) + "\n")
    written.append("FINAL_REPORT.md")
    summary["tables_written"] = written
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 7 offline analysis")
    parser.add_argument("result_dir")
    args = parser.parse_args()
    out = run(Path(args.result_dir))
    print(f"wrote {len(out['tables_written'])} report files to {args.result_dir}")
