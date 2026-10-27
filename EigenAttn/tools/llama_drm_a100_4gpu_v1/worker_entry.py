#!/usr/bin/env python3
"""Thin worker wrapper for LLAMA_DRM_A100_4GPU_V1.

Responsibilities, deliberately kept minimal so that all experiment semantics stay in
the authoritative runner:

1. Pin the assigned GPU (UUID first, index as fallback) via CUDA_VISIBLE_DEVICES.
2. Materialise the runner argv from the JSON job spec and exec it as a child.
3. Sample peak GPU memory for the child process tree while it runs.
4. Classify CUDA OOM and exit with OOM_EXIT_CODE so the scheduler can retry
   exclusively instead of treating it as a hard failure.
5. Rely on the runner's own per-sample resume: predictions already present in
   `longbench_predictions/<task>.jsonl` are never recomputed.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

OOM_EXIT_CODE = 42

OOM_PATTERNS = (
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.I),
    re.compile(r"CUDA out of memory", re.I),
    re.compile(r"CUBLAS_STATUS_ALLOC_FAILED", re.I),
    re.compile(r"out of memory", re.I),
)


def build_runner_argv(spec: Dict[str, Any]) -> List[str]:
    """Translate the job spec's `args` mapping into runner CLI tokens.

    Values are emitted verbatim; booleans map to argparse store_true flags (emitted
    only when true) so that a spec can never silently invert a flag's meaning.
    """
    argv: List[str] = [C.python_bin(), str(C.RUNNER_SCRIPT)]
    for key, value in spec.get("args", {}).items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        argv.extend([flag, str(value)])
    return argv


class PeakMemorySampler(threading.Thread):
    """Poll nvidia-smi for the memory used by our process tree on the assigned GPU."""

    def __init__(self, gpu_uuid: str, root_pid: int, interval: float = 5.0):
        super().__init__(daemon=True)
        self.gpu_uuid = gpu_uuid
        self.root_pid = root_pid
        self.interval = interval
        self.peak_mib = 0.0
        # Not `_stop`: threading.Thread already defines a private _stop() method and
        # shadowing it breaks Thread.join().
        self._stop_event = threading.Event()

    def _tracked_pids(self) -> set:
        pids = {self.root_pid}
        try:
            out = subprocess.run(
                ["ps", "-o", "pid=", "--ppid", str(self.root_pid)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            for line in out.stdout.split():
                try:
                    pids.add(int(line))
                except ValueError:
                    pass
        except Exception:
            pass
        return pids

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                apps = C.nvidia_smi_query(
                    ["gpu_uuid", "pid", "used_gpu_memory"], query="--query-compute-apps"
                )
            except Exception:
                continue
            tracked = self._tracked_pids()
            total = 0.0
            for app in apps:
                try:
                    pid = int(float(app.get("pid", "-1")))
                    used = float(app.get("used_gpu_memory", "0") or 0)
                except (TypeError, ValueError):
                    continue
                # When CUDA_VISIBLE_DEVICES pins a single GPU the child can only be on
                # that device, so a PID match is sufficient and survives UUID
                # reporting quirks under MIG-less setups.
                if pid in tracked:
                    total += used
            self.peak_mib = max(self.peak_mib, total)

    def stop(self) -> None:
        self._stop_event.set()


def main() -> int:
    parser = argparse.ArgumentParser(description="LLAMA_DRM_A100_4GPU_V1 worker wrapper")
    parser.add_argument("--job-spec", required=True, help="Path to the JSON job spec")
    options = parser.parse_args()

    job = C.read_json(options.job_spec, default=None)
    if not isinstance(job, dict):
        print(f"[worker] unreadable job spec: {options.job_spec}", flush=True)
        return 2

    spec = job.get("spec", {}) or {}
    worker_id = job.get("worker_id", "unknown")
    gpu_uuid = str(job.get("gpu_uuid", "") or "")
    gpu_index = job.get("gpu_index", None)

    output_dir = Path(spec.get("output_dir", "")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = C.offline_env()
    # UUID pinning is index-renumbering proof; fall back to the ordinal only if the
    # UUID is unavailable.
    if gpu_uuid:
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    elif gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["LLAMA_DRM_WORKER_ID"] = str(worker_id)

    argv = build_runner_argv(spec)
    started = time.time()

    resumed_counts = {}
    pred_dir = output_dir / "longbench_predictions"
    if pred_dir.is_dir():
        for path in sorted(pred_dir.glob("*.jsonl")):
            resumed_counts[path.stem] = C.count_jsonl_records(path)

    result: Dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "worker_id": worker_id,
        "job_id": job.get("job_id", ""),
        "category": job.get("category", ""),
        "method_key": job.get("method_key", ""),
        "task": job.get("task", ""),
        "gpu_uuid": gpu_uuid,
        "gpu_index": gpu_index,
        "placement": job.get("placement", ""),
        "attempt": job.get("attempt", 0),
        "started_at": C.now_iso(),
        "cmd": argv,
        "resumed_predictions": resumed_counts,
        "output_dir": str(output_dir),
    }
    C.atomic_write_json(output_dir / "worker_result.json", {**result, "state": "RUNNING"})

    print(f"[worker] {worker_id} gpu_uuid={gpu_uuid} idx={gpu_index}", flush=True)
    print(f"[worker] resume state: {resumed_counts}", flush=True)
    print(f"[worker] cmd: {' '.join(argv)}", flush=True)

    log_path = output_dir / "runner.log"
    oom_detected = False
    tail: List[str] = []

    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
        cwd=str(C.RUNTIME_ROOT), text=True, bufsize=1,
    )
    sampler = PeakMemorySampler(gpu_uuid, process.pid)
    sampler.start()

    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n=== {C.now_iso()} {worker_id} ===\n")
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            sys.stdout.write(line)
            tail.append(line)
            if len(tail) > 200:
                tail.pop(0)
            if not oom_detected and any(pattern.search(line) for pattern in OOM_PATTERNS):
                oom_detected = True
        log_handle.flush()

    exit_code = process.wait()
    sampler.stop()
    sampler.join(timeout=10)

    final_counts = {}
    if pred_dir.is_dir():
        for path in sorted(pred_dir.glob("*.jsonl")):
            final_counts[path.stem] = C.count_jsonl_records(path)

    metrics = C.read_json(output_dir / "metrics.json", default={}) or {}
    metrics_peak = metrics.get("peak_gpu_mem_GB")
    sampled_peak_gib = sampler.peak_mib / 1024.0
    peak_gib = max(
        float(metrics_peak or 0.0),
        sampled_peak_gib,
    )

    is_oom = oom_detected or str(metrics.get("status", "")).upper() == "OOM"
    result.update({
        "state": "OOM" if is_oom else ("COMPLETED" if exit_code == 0 else "FAILED"),
        "finished_at": C.now_iso(),
        "elapsed_s": round(time.time() - started, 1),
        "runner_exit_code": exit_code,
        "oom": is_oom,
        "peak_mem_gib": round(peak_gib, 3),
        "peak_mem_gib_sampled": round(sampled_peak_gib, 3),
        "peak_mem_gib_metrics": metrics_peak,
        # estimated_peak feeds the scheduler's co-location arithmetic on later jobs.
        "estimated_peak_gib": round(
            max(peak_gib * C.PEAK_MULTIPLIER, peak_gib + C.PEAK_HEADROOM_GIB) if peak_gib > 0 else 0.0,
            3,
        ),
        "final_predictions": final_counts,
        "longbench_avg": metrics.get("LongBench_avg"),
        "runner_status": metrics.get("status", ""),
        "note": ("cuda oom" if is_oom else ""),
        "log_tail": tail[-40:],
    })
    C.atomic_write_json(output_dir / "worker_result.json", result)

    if is_oom:
        print(f"[worker] {worker_id} OOM -> exit {OOM_EXIT_CODE} "
              f"(partial predictions kept: {final_counts})", flush=True)
        return OOM_EXIT_CODE
    print(f"[worker] {worker_id} exit={exit_code} peak={peak_gib:.2f} GiB", flush=True)
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
