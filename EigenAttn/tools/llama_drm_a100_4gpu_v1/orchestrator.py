#!/usr/bin/env python3
"""Unattended stage orchestrator for LLAMA_DRM_A100_4GPU_V1.

A single instance (enforced by a global flock) drives stages 0 through 7 with no human
approval gate. It is restart-safe: every stage is derived from artefacts on disk, so a
systemd restart resumes exactly where the previous process stopped.

Stage map (protocol LLAMA_DRM_A100_4GPU_V1)
  0  CPU-only audit (no weights on GPU). HARD_BLOCK on failure.
  1  12-sample formal canary (Original x2, no-op, Larger, Random seed0 x2,
     Random seed1, DRM-Add, DRM-Swap) including gov_report; then short
     consistency canaries on the remaining 3 A100s. Canary uses 1 GPU / 1 worker.
  2  Six-task main: Original, Larger, Random seed0/1/2, DRM-Add 5%, DRM-Swap
     = 42 task-workers.
  3  Full 17-task: Original, Larger, Random seed0, DRM-Add, DRM-Swap, Full-KV
     = 102 task-workers.
  4  OOM accuracy completion (exclusive GPU; native results preserved).
  5  Budget/component ablations on six tasks (2%/10% ratio + 4 component arms;
     5% reuses Stage 2). 36 new task-workers.
  6  A100 efficiency (exclusive GPU): 6 methods x 5 lengths.
  7  Offline aggregation and report export.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C
from scheduler.gpu_coloc_scheduler import (
    GpuColocScheduler,
    Job,
    estimated_peak_gib,
    FileLock,
)

STAGES: Tuple[Tuple[int, str], ...] = (
    (0, "cpu_audit"),
    (1, "canary"),
    (2, "six_task_main"),
    (3, "full_17_task"),
    (4, "oom_accuracy_completion"),
    (5, "ablations"),
    (6, "efficiency"),
    (7, "analysis_export"),
)

STAGE_NAMES: Dict[int, str] = dict(STAGES)

#: Conservative first guess until the canary reports a real peak. Llama-3-8B in bf16 is
#: ~16 GiB of weights plus KV/activations for a 7.5k-token prompt.
BOOTSTRAP_PEAK_GIB = 26.0

#: Protocol efficiency lengths. Local Llama-3-8B has max_position_embeddings=8192;
#: lengths above that are still scheduled so failures/OOM are audited honestly.
EFFICIENCY_CONTEXT_LENGTHS: Tuple[int, ...] = (8192, 16384, 24576, 28672, 32768)

EFFICIENCY_METHODS: Tuple[str, ...] = (
    "fullkv", "sliminfer_original", "sliminfer_larger", "random_add", "drm_add", "drm_swap",
)

#: Stage-2 / Stage-3 main arms (Random uses explicit seeds).
STAGE2_ARMS: Tuple[Tuple[str, Optional[int]], ...] = (
    ("sliminfer_original", None),
    ("sliminfer_larger", None),
    ("random_add", 0),
    ("random_add", 1),
    ("random_add", 2),
    ("drm_add", None),
    ("drm_swap", None),
)

STAGE3_ARMS: Tuple[Tuple[str, Optional[int]], ...] = (
    ("sliminfer_original", None),
    ("sliminfer_larger", None),
    ("random_add", 0),
    ("drm_add", None),
    ("drm_swap", None),
    ("fullkv", None),
)

#: Stage-5 component ablations (DRM-Add base) via frozen score-weight CLI knobs.
COMPONENT_ABLATIONS: Tuple[Tuple[str, Dict[str, float]], ...] = (
    ("without_gain", {"alpha_diff": 0.0}),
    ("without_active_core_redundancy", {"lambda_red": 0.0, "eta_active": 0.0}),
    ("without_rescue_rescue_mmr", {"eta_rescue": 0.0}),
    ("base_utility_only", {
        "alpha_diff": 0.0, "lambda_red": 0.0, "eta_rescue": 0.0, "eta_active": 0.0,
    }),
)

ABLATION_RESCUE_RATIOS: Tuple[float, ...] = (0.02, 0.10)

CANARY_SAMPLES = 12
CANARY_SHORT_SAMPLES = 4
CANARY_TASKS_PRIMARY: Tuple[str, ...] = ("qasper", "gov_report")
CANARY_TASKS_SHORT: Tuple[str, ...] = ("qasper", "hotpotqa")


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------


def load_task_manifest() -> Dict[str, Dict[str, Any]]:
    payload = C.read_json(C.PROTOCOL_DIR / "task_manifest.json", default={}) or {}
    return {entry["task"]: entry for entry in payload.get("tasks", []) if entry.get("task")}


TASK_MANIFEST = load_task_manifest()


def expected_samples(task: str) -> int:
    return int(TASK_MANIFEST.get(task, {}).get("sample_count", 0))


def task_max_new_tokens(task: str) -> int:
    return int(TASK_MANIFEST.get(task, {}).get("max_new_tokens", 64))


def is_exclusive_task(task: str) -> bool:
    """gov_report is exclusive (512-token generations dominate memory and runtime)."""
    return task in C.EXCLUSIVE_CATEGORIES


# ---------------------------------------------------------------------------
# Worker command construction
# ---------------------------------------------------------------------------


def base_runner_args(model_path: Path, longbench_dir: Path) -> Dict[str, Any]:
    """Frozen LLaMA DRM configuration shared by every quality arm."""
    return {
        "model_family": C.MODEL_FAMILY,
        "model_path": str(model_path),
        "longbench_dir": str(longbench_dir),
        "protocol_id": C.PROTOCOL_ID,
        "dtype": "bf16",
        "device": "cuda",
        "backend": "sdpa",
        "eval": "longbench",
        "block_size": 64,
        "block_strategy": "fixed",
        "token_unit_size": 8,
        "swap_threshold": 0.9,
        "local_query_window": 4,
        "prune_layers": "10,20,30",
        "retain_tokens": "8192,4096,2048",
        "max_model_len": C.MAX_MODEL_LEN,
        "seed": 0,
        "save_metrics_json": True,
    }


def method_args(method_key: str, *, seed: Optional[int] = None,
                rescue_ratio: Optional[float] = None,
                score_overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Authoritative CLI mapping for one arm.

    Every arm except Full-KV runs `--method sliminfer_original`; arms differ only in
    rescue_policy / rescue_score / rescue_ratio. Keeping this in one place is what makes
    the Random-vs-DRM comparison budget-identical by construction.
    """
    spec = C.METHOD_CLI_MAP[method_key]
    args: Dict[str, Any] = {"method": spec["method"]}
    if method_key == "fullkv":
        return args
    args["rescue_policy"] = spec["rescue_policy"]
    args["rescue_score"] = spec["rescue_score"]
    if "rescue_ratio" in spec:
        args["rescue_ratio"] = spec["rescue_ratio"]
    if rescue_ratio is not None:
        args["rescue_ratio"] = rescue_ratio
    if spec.get("needs_seed"):
        # Random rescue must be seeded explicitly so replicates are reproducible and
        # distinguishable; an unseeded Random arm cannot be audited.
        args["seed"] = int(seed if seed is not None else 0)
    elif seed is not None:
        args["seed"] = int(seed)
    # Frozen DRM weights (component ablations override these without changing equations).
    args.setdefault("alpha_diff", 0.5)
    args.setdefault("lambda_red", 0.25)
    args.setdefault("eta_rescue", 0.25)
    args.setdefault("eta_active", 0.25)
    if score_overrides:
        args.update({k: float(v) for k, v in score_overrides.items()})
    return args


def build_job(
    *,
    result_dir: Path,
    stage: int,
    method_key: str,
    task: str,
    model_path: Path,
    longbench_dir: Path,
    peak_gib: float,
    seed: Optional[int] = None,
    rescue_ratio: Optional[float] = None,
    max_samples: int = 0,
    variant: str = "",
    score_overrides: Optional[Dict[str, float]] = None,
    run_tag: str = "",
    exclusive_override: Optional[bool] = None,
    category_override: Optional[str] = None,
) -> Job:
    stage_name = STAGE_NAMES[stage]
    parts = [method_key]
    if variant:
        parts.append(variant)
    if seed is not None and C.METHOD_CLI_MAP[method_key].get("needs_seed"):
        parts.append(f"seed{seed}")
    if run_tag:
        parts.append(run_tag)
    arm_dir = "__".join(parts)
    output_dir = result_dir / "runs" / f"stage{stage}_{stage_name}" / arm_dir / task

    args = base_runner_args(model_path, longbench_dir)
    args.update(method_args(
        method_key, seed=seed, rescue_ratio=rescue_ratio, score_overrides=score_overrides,
    ))
    args["longbench_tasks"] = task
    args["output_dir"] = str(output_dir)
    if max_samples > 0:
        args["max_samples_per_task"] = max_samples

    job_id = f"s{stage}.{arm_dir}.{task}"
    exclusive = (
        bool(exclusive_override) if exclusive_override is not None
        else (is_exclusive_task(task) or method_key == "fullkv")
    )
    if category_override:
        category = category_override
    elif method_key == "fullkv":
        category = "fullkv"
    elif is_exclusive_task(task):
        category = task
    else:
        category = f"quality_{task}"
    return Job(
        job_id=job_id,
        category=category,
        spec={"args": args, "output_dir": str(output_dir), "kind": "longbench"},
        method_key=method_key,
        task=task,
        estimated_peak_gib=peak_gib,
        exclusive=exclusive,
    )


def build_efficiency_job(
    *, result_dir: Path, method_key: str, model_path: Path, longbench_dir: Path,
    peak_gib: float, context_length: int, seed: Optional[int] = None,
) -> Job:
    output_dir = (
        result_dir / "runs" / "stage6_efficiency" / method_key / f"ctx{context_length}"
    )
    args = base_runner_args(model_path, longbench_dir)
    args.update(method_args(method_key, seed=seed))
    args["eval"] = "latency"
    args["context_lengths"] = str(context_length)
    args["decode_tokens"] = 16
    args["one_context_per_process"] = True
    args["output_dir"] = str(output_dir)
    args["profile_io"] = True
    args["profile_memory"] = True
    args["profile_score_overhead"] = True
    args.pop("longbench_tasks", None)
    exclusive = True
    category = "efficiency"
    if context_length >= C.LONG_CONTEXT_THRESHOLD_TOKENS:
        category = "long_context_32k"
    return Job(
        job_id=f"s6.efficiency.{method_key}.ctx{context_length}",
        category=category,
        spec={"args": args, "output_dir": str(output_dir), "kind": "latency"},
        method_key=method_key,
        task="",
        estimated_peak_gib=peak_gib,
        # Timings measured next to a co-tenant are not comparable.
        exclusive=exclusive,
    )


# ---------------------------------------------------------------------------
# Completion accounting
# ---------------------------------------------------------------------------


def job_is_complete(job: Job) -> bool:
    output_dir = Path(job.spec["output_dir"])
    result = C.read_json(output_dir / "worker_result.json", default={}) or {}
    if result.get("state") == "COMPLETED":
        return True
    if job.spec.get("kind") == "latency":
        metrics = C.read_json(output_dir / "metrics.json", default={}) or {}
        return bool(metrics.get("latency", {}).get("rows"))
    # Quality arms are complete when every expected prediction exists, which is also
    # exactly the condition under which the runner would skip the task on resume.
    target = expected_samples(job.task)
    if target <= 0:
        return False
    args = job.spec.get("args", {})
    limit = int(args.get("max_samples_per_task", 0) or 0)
    if limit > 0:
        target = min(target, limit)
    pred_path = output_dir / "longbench_predictions" / f"{job.task}.jsonl"
    return C.count_jsonl_records(pred_path) >= target


def job_progress(job: Job) -> Tuple[int, int]:
    output_dir = Path(job.spec["output_dir"])
    if job.spec.get("kind") == "latency":
        return (1, 1) if job_is_complete(job) else (0, 1)
    target = expected_samples(job.task)
    args = job.spec.get("args", {})
    limit = int(args.get("max_samples_per_task", 0) or 0)
    if limit > 0:
        target = min(target, limit) if target else limit
    pred_path = output_dir / "longbench_predictions" / f"{job.task}.jsonl"
    return C.count_jsonl_records(pred_path), max(target, 0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    def __init__(self, *, resume: bool = True, poll_interval: float = 20.0,
                 max_workers_per_gpu: int = C.DEFAULT_MAX_WORKERS_PER_GPU,
                 stage_limit: Optional[int] = None, dry_run: bool = False) -> None:
        self.poll_interval = poll_interval
        self.max_workers_per_gpu = max_workers_per_gpu
        self.stage_limit = stage_limit
        self.dry_run = dry_run

        self.results_root = C.results_root()
        self.results_root.mkdir(parents=True, exist_ok=True)

        existing = C.find_resumable_result_dir(self.results_root) if resume else None
        self.result_dir = existing or C.new_result_dir(self.results_root)
        self.is_resumed = existing is not None
        C.ensure_result_layout(self.result_dir)

        self.status_dir = self.result_dir / "status"
        self.model_path: Optional[Path] = C.resolve_model_path()
        self.longbench_dir: Optional[Path] = C.resolve_longbench_dir()
        self.stage_decisions: List[Dict[str, Any]] = (
            C.read_json(self.status_dir / "STAGE_DECISIONS.json", default={}) or {}
        ).get("decisions", [])
        self.measured_peak_gib: float = float(
            (C.read_json(self.status_dir / "PEAK_MEMORY.json", default={}) or {}).get(
                "measured_peak_gib", 0.0
            )
        )
        self._hard_blocked = False

    # -- bookkeeping ------------------------------------------------------

    def decide(self, stage: int, decision: str, detail: str = "", **extra: Any) -> None:
        entry = {
            "at": C.now_iso(), "stage": stage, "stage_name": STAGE_NAMES.get(stage, ""),
            "decision": decision, "detail": detail,
        }
        entry.update(extra)
        self.stage_decisions.append(entry)
        C.atomic_write_json(
            self.status_dir / "STAGE_DECISIONS.json",
            {"protocol_id": C.PROTOCOL_ID, "updated_at": C.now_iso(),
             "decisions": self.stage_decisions},
        )

    def stage_marker(self, stage: int) -> Path:
        return self.status_dir / f"STAGE{stage}_COMPLETE.json"

    def stage_done(self, stage: int) -> bool:
        return self.stage_marker(stage).exists()

    def mark_stage_done(self, stage: int, **extra: Any) -> None:
        payload = {
            "protocol_id": C.PROTOCOL_ID, "stage": stage,
            "stage_name": STAGE_NAMES.get(stage, ""), "completed_at": C.now_iso(),
        }
        payload.update(extra)
        C.atomic_write_json(self.stage_marker(stage), payload)
        self.decide(stage, "STAGE_COMPLETE", **extra)

    def hard_block(self, reason: str, detail: Any = None) -> None:
        payload = {
            "protocol_id": C.PROTOCOL_ID, "state": "HARD_BLOCK", "at": C.now_iso(),
            "reason": reason, "detail": detail,
        }
        C.atomic_write_json(self.status_dir / "HARD_BLOCK.json", payload)
        C.atomic_write_text(
            self.status_dir / "HARD_BLOCK.md",
            f"# HARD BLOCK\n\n- protocol: {C.PROTOCOL_ID}\n- at: {C.now_iso()}\n"
            f"- reason: {reason}\n\n```\n{detail}\n```\n",
        )
        self.decide(-1, "HARD_BLOCK", reason)
        self._hard_blocked = True

    def watchdog(self, scheduler: Optional[GpuColocScheduler], note: str = "") -> None:
        C.atomic_write_json(self.status_dir / "WATCHDOG_STATUS.json", {
            "protocol_id": C.PROTOCOL_ID,
            "updated_at": C.now_iso(),
            "orchestrator_pid": os.getpid(),
            "result_dir": str(self.result_dir),
            "results_root": str(self.results_root),
            "results_root_free_gib": round(C.free_gib(self.results_root), 2),
            "scheduler_state": scheduler.state if scheduler else "NO_SCHEDULER",
            "active_workers": scheduler.active_count() if scheduler else 0,
            "pending_jobs": scheduler.pending_count() if scheduler else 0,
            "measured_peak_gib": self.measured_peak_gib,
            "note": note,
            "heartbeat_epoch": time.time(),
        })

    # -- progress / status ------------------------------------------------

    def write_progress(self, stage: int, jobs: Sequence[Job],
                       scheduler: Optional[GpuColocScheduler], note: str = "") -> Dict[str, Any]:
        per_job = []
        done_samples = total_samples = 0
        complete_jobs = 0
        for job in jobs:
            have, want = job_progress(job)
            done_samples += have
            total_samples += want
            finished = job_is_complete(job)
            complete_jobs += int(finished)
            per_job.append({
                "job_id": job.job_id, "method_key": job.method_key, "task": job.task,
                "category": job.category, "predictions": have, "expected": want,
                "complete": finished,
            })
        progress = {
            "protocol_id": C.PROTOCOL_ID,
            "updated_at": C.now_iso(),
            "result_dir": str(self.result_dir),
            "resumed": self.is_resumed,
            "current_stage": stage,
            "current_stage_name": STAGE_NAMES.get(stage, ""),
            "stages_complete": [s for s, _ in STAGES if self.stage_done(s)],
            "jobs_total": len(jobs),
            "jobs_complete": complete_jobs,
            "samples_done": done_samples,
            "samples_expected": total_samples,
            "percent": round(100.0 * done_samples / total_samples, 2) if total_samples else 0.0,
            "measured_peak_gib": self.measured_peak_gib,
            "scheduler_state": scheduler.state if scheduler else "NO_SCHEDULER",
            "note": note,
            "jobs": per_job,
        }
        C.atomic_write_json(self.status_dir / "PROGRESS.json", progress)
        self.write_status_md(progress, scheduler)
        return progress

    def write_status_md(self, progress: Dict[str, Any],
                        scheduler: Optional[GpuColocScheduler]) -> None:
        gpu_status = C.read_json(self.status_dir / "GPU_STATUS.json", default={}) or {}
        lines = [
            f"# {C.PROTOCOL_ID} pipeline status",
            "",
            f"- updated: {C.now_iso()}",
            f"- result dir: `{self.result_dir}`",
            f"- resumed: {self.is_resumed}",
            f"- stage: {progress['current_stage']} ({progress['current_stage_name']})",
            f"- stages complete: {progress['stages_complete']}",
            f"- jobs: {progress['jobs_complete']}/{progress['jobs_total']} complete",
            f"- samples: {progress['samples_done']}/{progress['samples_expected']} "
            f"({progress['percent']}%)",
            f"- scheduler: {progress['scheduler_state']}",
            f"- measured peak: {self.measured_peak_gib:.2f} GiB "
            f"(estimated for co-location: {estimated_peak_gib(self.measured_peak_gib):.2f} GiB)",
            f"- results volume free: {round(C.free_gib(self.results_root), 1)} GiB",
            "",
            "## GPUs",
            "",
        ]
        for gpu in gpu_status.get("gpus", []):
            lines.append(
                f"- gpu{gpu.get('index')} `{gpu.get('uuid')}`: free {gpu.get('free_gib')} GiB, "
                f"own workers {gpu.get('own_worker_count')}, "
                f"external pids {gpu.get('external_pids')}"
            )
        lines += ["", "## Active workers", ""]
        workers = (C.read_json(self.status_dir / "CURRENT_WORKERS.json", default={}) or {}).get(
            "workers", []
        )
        if not workers:
            lines.append("- none")
        for worker in workers:
            lines.append(
                f"- `{worker.get('worker_id')}` {worker.get('method_key')}/{worker.get('task')} "
                f"gpu{worker.get('gpu_index')} {worker.get('placement')} "
                f"{worker.get('elapsed_s')}s"
            )
        lines += ["", "## Incomplete jobs", ""]
        incomplete = [job for job in progress.get("jobs", []) if not job["complete"]]
        if not incomplete:
            lines.append("- none")
        for job in incomplete[:40]:
            lines.append(
                f"- `{job['job_id']}` {job['predictions']}/{job['expected']}"
            )
        C.atomic_write_text(self.status_dir / "PIPELINE_STATUS.md", "\n".join(lines) + "\n")

    # -- bootstrap --------------------------------------------------------

    def bootstrap_protocol_docs(self) -> None:
        from status_bootstrap import protocol_templates

        protocol_templates.write_all(
            self.result_dir,
            model_path=self.model_path,
            longbench_dir=self.longbench_dir,
        )

    # -- stages -----------------------------------------------------------

    def run_stage0(self) -> bool:
        if self.stage_done(0):
            return True
        audit_dir = self.result_dir / "audit"
        cmd = [C.python_bin(), str(C.STAGE0_AUDIT), "--output-dir", str(audit_dir)]
        self.decide(0, "STAGE_START", " ".join(cmd))
        import subprocess

        proc = subprocess.run(
            cmd, cwd=str(C.TOOL_ROOT), env=C.offline_env({"CUDA_VISIBLE_DEVICES": ""}),
            capture_output=True, text=True, check=False,
        )
        C.atomic_write_text(self.result_dir / "logs" / "stage0_audit.log",
                            proc.stdout + "\n" + proc.stderr)
        if proc.returncode != 0:
            self.hard_block("stage0_cpu_audit failed", proc.stdout[-4000:] + proc.stderr[-4000:])
            return False
        self.mark_stage_done(0, audit_dir=str(audit_dir))
        return True

    def stage_jobs(self, stage: int) -> List[Job]:
        """Job list for a GPU stage. Pure function of stage + resolved paths."""
        assert self.model_path and self.longbench_dir
        peak = estimated_peak_gib(self.measured_peak_gib) if self.measured_peak_gib > 0 else BOOTSTRAP_PEAK_GIB
        kwargs = dict(result_dir=self.result_dir, model_path=self.model_path,
                      longbench_dir=self.longbench_dir, peak_gib=peak)
        jobs: List[Job] = []
        if stage == 1:
            # Formal 12-sample canary on primary tasks (must include gov_report).
            # Arms: Original run1/run2, no-op (==Original CLI), Larger,
            # Random seed0 run1/run2, Random seed1, DRM-Add, DRM-Swap.
            canary_arms: List[Tuple[str, Optional[int], str]] = [
                ("sliminfer_original", None, "run1"),
                ("sliminfer_original", None, "run2"),
                ("sliminfer_original", None, "noop"),
                ("sliminfer_larger", None, "run1"),
                ("random_add", 0, "run1"),
                ("random_add", 0, "run2"),
                ("random_add", 1, "run1"),
                ("drm_add", None, "run1"),
                ("drm_swap", None, "run1"),
            ]
            for method_key, seed, run_tag in canary_arms:
                for task in CANARY_TASKS_PRIMARY:
                    # Split 12 samples across the two tasks (8 short + 4 long).
                    max_samples = 8 if task != "gov_report" else 4
                    jobs.append(build_job(
                        stage=1, method_key=method_key, task=task,
                        seed=seed, run_tag=run_tag, max_samples=max_samples,
                        exclusive_override=(task == "gov_report"),
                        **kwargs,
                    ))
            # Short consistency canaries on the other 3 GPUs (pinned via
            # category so the scheduler spreads them after the primary canary
            # measured peak). They remain canary-mode (1 worker/GPU) via the
            # scheduler canary flag for stage 1.
            for gpu_tag in ("gpu1", "gpu2", "gpu3"):
                for method_key in ("sliminfer_original", "drm_add"):
                    for task in CANARY_TASKS_SHORT:
                        jobs.append(build_job(
                            stage=1, method_key=method_key, task=task,
                            run_tag=f"short_{gpu_tag}",
                            max_samples=CANARY_SHORT_SAMPLES,
                            category_override="canary_short",
                            **kwargs,
                        ))
        elif stage == 2:
            # 7 arms × 6 tasks = 42 task-workers.
            for method_key, seed in STAGE2_ARMS:
                for task in C.SIX_TASKS:
                    jobs.append(build_job(
                        stage=2, method_key=method_key, task=task, seed=seed, **kwargs,
                    ))
        elif stage == 3:
            # 6 arms × 17 tasks = 102 task-workers (Full-KV is exclusive).
            for method_key, seed in STAGE3_ARMS:
                for task in C.SEVENTEEN_TASKS:
                    jobs.append(build_job(
                        stage=3, method_key=method_key, task=task, seed=seed, **kwargs,
                    ))
        elif stage == 4:
            # OOM accuracy completion: scan Stage 2/3 prediction dirs for OOM
            # markers and enqueue exclusive fill jobs. If none, stage is a no-op PASS.
            oom_jobs = self._build_oom_completion_jobs(peak)
            jobs.extend(oom_jobs)
        elif stage == 5:
            # Budget ratios 2%/10% (5% reuses Stage 2 DRM-Add via hash audit in Stage 7).
            for ratio in ABLATION_RESCUE_RATIOS:
                for task in C.SIX_TASKS:
                    jobs.append(build_job(
                        stage=5, method_key="drm_add", task=task,
                        rescue_ratio=ratio,
                        variant=f"ratio{str(ratio).replace('.', 'p')}",
                        **kwargs,
                    ))
            # Component ablations (4 × 6 = 24) + ratios (2 × 6 = 12) = 36 new workers.
            for variant, overrides in COMPONENT_ABLATIONS:
                for task in C.SIX_TASKS:
                    jobs.append(build_job(
                        stage=5, method_key="drm_add", task=task,
                        variant=variant, score_overrides=overrides, **kwargs,
                    ))
        elif stage == 6:
            # 6 methods × 5 lengths; each case exclusive + fresh process.
            for method_key in EFFICIENCY_METHODS:
                seed = 0 if method_key == "random_add" else None
                for ctx in EFFICIENCY_CONTEXT_LENGTHS:
                    jobs.append(build_efficiency_job(
                        result_dir=self.result_dir, method_key=method_key,
                        model_path=self.model_path, longbench_dir=self.longbench_dir,
                        peak_gib=peak, context_length=ctx, seed=seed,
                    ))
        return jobs

    def _build_oom_completion_jobs(self, peak: float) -> List[Job]:
        """Enqueue exclusive OOM fill workers for native OOM prediction rows.

        Native Stage 2/3 results are never overwritten; fill outputs live under
        stage4_oom_accuracy_completion/ and are quality-only (not efficiency).
        """
        jobs: List[Job] = []
        scan_roots = [
            self.result_dir / "runs" / "stage2_six_task_main",
            self.result_dir / "runs" / "stage3_full_17_task",
        ]
        oom_tasks: set = set()
        for root in scan_roots:
            if not root.exists():
                continue
            for pred in root.rglob("longbench_predictions/*.jsonl"):
                try:
                    text = pred.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if "[OOM]" not in text and '"oom": true' not in text.lower():
                    continue
                # path: .../arm_dir/task/longbench_predictions/task.jsonl
                task = pred.stem
                arm_dir = pred.parent.parent.name
                method_key = arm_dir.split("__", 1)[0]
                if method_key not in C.METHOD_CLI_MAP:
                    continue
                key = (method_key, task, arm_dir)
                if key in oom_tasks:
                    continue
                oom_tasks.add(key)
                seed = None
                if "seed" in arm_dir:
                    for part in arm_dir.split("__"):
                        if part.startswith("seed") and part[4:].isdigit():
                            seed = int(part[4:])
                jobs.append(build_job(
                    stage=4, method_key=method_key, task=task, seed=seed,
                    variant=f"oom_fill__{arm_dir}",
                    exclusive_override=True,
                    category_override="oom_fill",
                    result_dir=self.result_dir,
                    model_path=self.model_path,
                    longbench_dir=self.longbench_dir,
                    peak_gib=peak,
                ))
        return jobs

    def record_measured_peak(self, scheduler: GpuColocScheduler) -> None:
        """Adopt the largest observed worker peak; it sizes all later co-location."""
        peaks = [
            float(handle.peak_mem_gib)
            for handle in scheduler.completed
            if handle.peak_mem_gib
        ]
        if not peaks:
            return
        observed = max(peaks)
        if observed <= self.measured_peak_gib:
            return
        self.measured_peak_gib = observed
        C.atomic_write_json(self.status_dir / "PEAK_MEMORY.json", {
            "protocol_id": C.PROTOCOL_ID,
            "updated_at": C.now_iso(),
            "measured_peak_gib": round(observed, 3),
            "estimated_peak_gib": round(estimated_peak_gib(observed), 3),
            "rule": "max(actual_peak * 1.20, actual_peak + 4 GiB)",
            "safety_reserve_gib": C.SAFETY_RESERVE_GIB,
        })

    def run_gpu_stage(self, stage: int) -> bool:
        if self.stage_done(stage):
            return True
        jobs = self.stage_jobs(stage)
        if not jobs:
            self.mark_stage_done(stage, jobs_total=0, note="no jobs defined")
            return True

        pending = [job for job in jobs if not job_is_complete(job)]
        self.decide(stage, "STAGE_START", f"{len(pending)}/{len(jobs)} jobs pending")

        scheduler = GpuColocScheduler(
            self.result_dir,
            max_workers_per_gpu=self.max_workers_per_gpu,
            canary=(stage == 1),
            status_dir=self.status_dir,
        )
        scheduler.submit_all(pending)
        # Publish GPU_STATUS/CURRENT_WORKERS/WORKER_REGISTRY immediately so an external
        # watcher never sees a stage begin with stale status from the previous stage.
        scheduler.write_status()
        self.write_progress(stage, jobs, scheduler, note="stage start")

        if self.dry_run:
            self.decide(stage, "DRY_RUN_SKIP", f"{len(pending)} jobs would be scheduled")
            return True

        idle_ticks = 0
        while True:
            outcome = scheduler.step()
            self.record_measured_peak(scheduler)
            self.write_progress(stage, jobs, scheduler,
                                note=f"state={outcome['state']}")
            self.watchdog(scheduler, note=f"stage {stage}")

            if not scheduler.queue and not scheduler.workers:
                break
            if outcome["state"] == C_WAITING and not outcome["launched"]:
                idle_ticks += 1
            else:
                idle_ticks = 0
            time.sleep(self.poll_interval)

        remaining = [job for job in jobs if not job_is_complete(job)]
        if remaining:
            # Do not mark the stage complete: the next orchestrator run retries the
            # remaining jobs, and repeated failures stay visible in WORKER_REGISTRY.csv.
            self.decide(stage, "STAGE_INCOMPLETE",
                        f"{len(remaining)} jobs still incomplete",
                        incomplete=[job.job_id for job in remaining][:50])
            return False

        # Stage-1 formal canary gate: Original determinism / no-op / Random wiring.
        if stage == 1:
            from stages import canary_audit
            report = canary_audit.audit_canary(self.result_dir)
            if report.get("hard_block"):
                self.hard_block(
                    "stage1 canary audit HARD_BLOCK",
                    report.get("hard_block_reasons", []),
                )
                return False
            self.decide(1, "CANARY_AUDIT_PASS",
                        f"passed={report.get('passed')} failed={report.get('failed')}")

        self.mark_stage_done(stage, jobs_total=len(jobs),
                             failed=len(scheduler.failed))
        return True

    def run_stage7(self) -> bool:
        # Deliberately not gated on its own marker: aggregation is cheap, offline, and
        # must reflect the newest predictions, so it re-runs on every orchestrator pass.
        from stages import stage7_analysis

        self.decide(7, "STAGE_START", "offline aggregation and export")
        summary = stage7_analysis.run(self.result_dir)
        self.mark_stage_done(7, **{"tables": summary.get("tables_written", [])})
        return True

    # -- main loop --------------------------------------------------------

    def preflight(self) -> bool:
        problems: List[str] = []
        if self.model_path is None:
            problems.append(
                f"Llama-3 snapshot not found in {list(C.MODEL_PATH_CANDIDATES)}"
            )
        if self.longbench_dir is None:
            problems.append(
                f"LongBench 17-task directory not found in {list(C.LONGBENCH_PATH_CANDIDATES)}"
            )
        free = C.free_gib(self.results_root)
        if free < C.MIN_FREE_RESULTS_GIB:
            problems.append(
                f"results volume {self.results_root} has {free:.1f} GiB free, "
                f"below the {C.MIN_FREE_RESULTS_GIB} GiB minimum"
            )
        if problems:
            self.hard_block("preflight failed", problems)
            return False
        return True

    def run(self) -> int:
        self._hard_blocked = False
        # Terminal HARD_BLOCK must not restart-loop under systemd Restart=always.
        existing_block = self.status_dir / "HARD_BLOCK.json"
        if existing_block.exists() and not os.environ.get("LLAMA_DRM_CLEAR_HARD_BLOCK"):
            print(
                f"HARD_BLOCK present at {existing_block}; orchestrator idle "
                f"(set LLAMA_DRM_CLEAR_HARD_BLOCK=1 after resolving to resume).",
                flush=True,
            )
            return 0

        C.atomic_write_json(self.status_dir / "RUN_CONTEXT.json", {
            "protocol_id": C.PROTOCOL_ID,
            "started_at": C.now_iso(),
            "result_dir": str(self.result_dir),
            "resumed": self.is_resumed,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_revision": C.MODEL_REVISION,
            "longbench_dir": str(self.longbench_dir) if self.longbench_dir else None,
            "frozen_config": C.FROZEN_LLAMA_CONFIG,
            "environment": C.environment_snapshot(),
        })
        self.bootstrap_protocol_docs()
        self.watchdog(None, note="orchestrator start")

        if not self.preflight():
            return 0 if self._hard_blocked else 3
        if not self.run_stage0():
            return 0 if self._hard_blocked else 4

        for stage, _name in STAGES:
            if stage in (0, 7):
                continue
            if self.stage_limit is not None and stage > self.stage_limit:
                break
            if not self.run_gpu_stage(stage):
                if self._hard_blocked:
                    self.watchdog(None, note=f"stage {stage} HARD_BLOCK; stopping")
                    return 0
                # Incomplete stage: exit non-zero so systemd restarts and resumes.
                self.watchdog(None, note=f"stage {stage} incomplete; will resume")
                return 5

        if self.stage_limit is None or self.stage_limit >= 7:
            self.run_stage7()
            self.finalize()
        return 0

    def finalize(self) -> None:
        required = [stage for stage, _ in STAGES]
        missing = [stage for stage in required if not self.stage_done(stage)]
        if missing:
            self.decide(-1, "FINALIZE_BLOCKED", f"stages incomplete: {missing}")
            return
        payload = {
            "protocol_id": C.PROTOCOL_ID,
            "final_state": C.FINAL_STATE,
            "completed_at": C.now_iso(),
            "result_dir": str(self.result_dir),
            "stages": {str(stage): STAGE_NAMES[stage] for stage, _ in STAGES},
        }
        C.atomic_write_json(self.status_dir / "PIPELINE_COMPLETE.json", payload)
        C.atomic_write_text(
            self.status_dir / "FINAL_STATE.txt", C.FINAL_STATE + "\n"
        )
        self.decide(7, C.FINAL_STATE, "all required experiments complete")


C_WAITING = "WAITING_FOR_GPU_IDLE"


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{C.PROTOCOL_ID} orchestrator")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume the newest incomplete result dir (systemd default)")
    parser.add_argument("--fresh", action="store_true", default=False,
                        help="Force a brand new result dir")
    parser.add_argument("--poll-interval", type=float, default=20.0)
    parser.add_argument("--max-workers-per-gpu", type=int, default=C.DEFAULT_MAX_WORKERS_PER_GPU)
    parser.add_argument("--stage-limit", type=int, default=None,
                        help="Stop after this stage (debugging)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Build and report job lists without launching workers")
    options = parser.parse_args()

    # A second orchestrator would double-schedule every GPU; the flock makes the loop
    # unique across systemd restarts and manual invocations alike.
    lock = FileLock(C.LOCKS_ROOT / "orchestrator.lock", timeout=5.0)
    if not lock.acquire():
        print("Another orchestrator instance holds the lock; exiting.", flush=True)
        return 0
    try:
        orchestrator = Orchestrator(
            resume=not options.fresh,
            poll_interval=options.poll_interval,
            max_workers_per_gpu=options.max_workers_per_gpu,
            stage_limit=options.stage_limit,
            dry_run=options.dry_run,
        )
        try:
            return orchestrator.run()
        except Exception:
            orchestrator.hard_block("orchestrator exception", traceback.format_exc())
            raise
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
