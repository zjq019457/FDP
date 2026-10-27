#!/usr/bin/env python3
"""Rebuttal completion pipeline orchestrator: single DAG, 4-GPU, tmux-detached."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    HEARTBEAT_SECS,
    PYTHON_BIN,
    SESSION_NAME,
    atomic_write_json,
    now_iso,
    stop_requested,
)
from gpu_scheduler import GPULockManager, write_inventory  # noqa: E402
from stage_materialize import future_stages_remain, materialize_all  # noqa: E402

ROOT = HERE.parent
JOBS_PATH = HERE / "jobs.jsonl"
STATE_PATH = HERE / "orchestrator_state.json"
LOG_DIR = HERE / "background_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Pack ~3-4 Llama-8B jobs per A100 (80GB); refill as soon as any slot frees.
MAX_GPU_PARALLEL = 12
MAX_CPU_PARALLEL = 8
MAX_POOL_WORKERS = 20
JOB_RESERVE_MIB = 18000


def log(msg: str) -> None:
    line = f"{now_iso()} {msg}\n"
    sys.stderr.write(line)
    sys.stderr.flush()
    with (LOG_DIR / "orchestrator.log").open("a", encoding="utf-8") as f:
        f.write(line)


def load_jobs() -> List[Dict[str, Any]]:
    if not JOBS_PATH.exists():
        return []
    return [json.loads(l) for l in JOBS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_jobs(jobs: List[Dict[str, Any]]) -> None:
    JOBS_PATH.write_text("\n".join(json.dumps(j, ensure_ascii=False) for j in jobs) + "\n", encoding="utf-8")


def write_status(state: Dict[str, Any]) -> None:
    state["generated_at"] = now_iso()
    atomic_write_json(STATE_PATH, state)


def job_counts(jobs: List[Dict[str, Any]]) -> Dict[str, int]:
    c = {"completed": 0, "running": 0, "pending": 0, "blocked": 0, "failed": 0, "stopped": 0}
    for j in jobs:
        s = j.get("status", "PENDING").upper()
        if s == "DONE":
            c["completed"] += 1
        elif s == "RUNNING":
            c["running"] += 1
        elif s == "FAILED":
            c["failed"] += 1
        elif s == "STOPPED":
            c["stopped"] += 1
        elif s == "BLOCKED":
            c["blocked"] += 1
        else:
            c["pending"] += 1
    return c


def deps_satisfied(job: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> bool:
    for d in job.get("depends_on") or []:
        dep = by_id.get(d)
        if not dep or dep.get("status") != "DONE":
            return False
        if not (Path(dep["output_dir"]) / "DONE.json").exists():
            return False
    return True


def execute_one_job(job: Dict[str, Any], locks: GPULockManager) -> Dict[str, Any]:
    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "job_id": job["job_id"],
        "cmd": job["cmd"],
        "started_at": now_iso(),
        "protocol_hash": job.get("protocol_hash"),
    }
    atomic_write_json(out_dir / "job_manifest.json", manifest)
    gpu_info = None
    env = os.environ.copy()
    if job.get("kind") == "gpu":
        gpu = job.get("preferred_gpu")
        # None => pack across GPUs by free VRAM; explicit int => soft preference only.
        preferred = int(gpu) if gpu is not None else None
        reserve = int(job.get("meta", {}).get("reserve_mib", JOB_RESERVE_MIB))
        gpu_info = locks.wait_acquire(
            job["job_id"],
            preferred_index=preferred,
            stop_fn=lambda: stop_requested(HERE),
            reserve_mib=reserve,
        )
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_info.get("cuda_visible_devices", gpu if gpu is not None else 0))
    t0 = time.time()
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    rc = 1
    try:
        with stdout_path.open("w", encoding="utf-8") as fo, stderr_path.open("w", encoding="utf-8") as fe:
            cp = subprocess.run(job["cmd"], cwd=str(HERE), env=env, stdout=fo, stderr=fe, text=True)
            rc = cp.returncode
    finally:
        if gpu_info and gpu_info.get("gpu_uuid"):
            locks.release(gpu_info["gpu_uuid"], job["job_id"])
    elapsed = time.time() - t0
    status = "DONE" if rc == 0 else "FAILED"
    result = {
        "job_id": job["job_id"],
        "status": status,
        "returncode": rc,
        "ended_at": now_iso(),
        "elapsed_sec": elapsed,
        "gpu": gpu_info,
    }
    atomic_write_json(out_dir / ("DONE.json" if status == "DONE" else "FAILED.json"), result)
    atomic_write_json(out_dir / "job_state.json", {"job_id": job["job_id"], "status": status})
    job["status"] = status
    return job


def _collect_runnable(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {j["job_id"]: j for j in jobs}
    runnable: List[Dict[str, Any]] = []
    build_running = any(
        j.get("status") == "RUNNING" and (j.get("meta") or {}).get("role") == "build"
        for j in jobs
    )
    for j in jobs:
        if j.get("status") in ("DONE", "STOPPED", "RUNNING"):
            continue
        if j.get("status") == "FAILED":
            out_dir = Path(j["output_dir"])
            if (out_dir / "DONE.json").exists():
                continue
            retries = int(j.get("retries", 0))
            if retries >= 8:
                continue
            j["status"] = "PENDING"
            j["retries"] = retries + 1
            for stale in ("FAILED.json", "job_state.json"):
                p = out_dir / stale
                if p.exists():
                    p.unlink()
            log(f"retrying FAILED job {j['job_id']} (attempt {j['retries']})")
        if not deps_satisfied(j, by_id):
            continue
        # Disk-safe: only one compress/build job at a time; evals may pack freely.
        if build_running and (j.get("meta") or {}).get("role") == "build":
            continue
        runnable.append(j)
    # Prefer launching at most one new build per scheduling round.
    builds = [j for j in runnable if (j.get("meta") or {}).get("role") == "build"]
    evals = [j for j in runnable if (j.get("meta") or {}).get("role") != "build"]
    if builds:
        return [builds[0]] + evals
    return evals


def _update_job(jobs: List[Dict[str, Any]], updated: Dict[str, Any]) -> List[Dict[str, Any]]:
    for i, j in enumerate(jobs):
        if j["job_id"] == updated["job_id"]:
            merged = dict(j)
            merged.update(updated)
            for k in ("cmd", "output_dir", "kind", "depends_on", "meta", "stage", "preferred_gpu"):
                if merged.get(k) in (None, "") and k in j:
                    merged[k] = j[k]
            if not merged.get("output_dir"):
                merged["output_dir"] = str(HERE / "job_outputs" / merged["job_id"])
            jobs[i] = merged
            break
    return jobs

def _adopt_external_zs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mark externally-running granular ZS jobs DONE when their aggregates appear."""
    art = Path("/diskE/jiaqizhang/rebuttal_completion_artifacts")
    for j in jobs:
        if j.get("status") != "RUNNING":
            continue
        jid = j["job_id"]
        if "_zs_" not in jid:
            continue
        body = jid[len("OP_") :]
        point = None
        rem = ""
        for p in ("moderate", "aggressive", "mild"):
            if body.startswith(p + "_zs_"):
                point = p
                rem = body[len(p) + 4 :]
                break
        if point is None:
            continue
        if rem.startswith("uniform_"):
            meth, bundle = "uniform", rem[len("uniform_") :]
        elif rem.startswith("fdp_"):
            meth, bundle = "fdp", rem[len("fdp_") :]
        else:
            continue
        candidates = [
            art / f"OP_{point}_{meth}_{bundle}" / "zero_shot_aggregate.json",
            art / f"OP_{point}_{meth}_zs_{bundle}" / "zero_shot_aggregate.json",
        ]
        if any(c.exists() for c in candidates):
            j["status"] = "DONE"
            out = Path(j["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                out / "DONE.json",
                {"job_id": jid, "status": "DONE", "adopted_external": True, "ended_at": now_iso()},
            )
            atomic_write_json(out / "job_state.json", {"job_id": jid, "status": "DONE"})
            log(f"adopted external DONE {jid}")
    return jobs


def run_loop() -> None:
    """Continuous multi-GPU scheduler with VRAM packing across 4 A100s."""
    state = {
        "phase": "run_loop",
        "master_pid": os.getpid(),
        "python": PYTHON_BIN,
        "gates": {},
        "gpu_jobs": {},
        "scheduler": "continuous_refill_packing",
        "max_gpu_parallel": MAX_GPU_PARALLEL,
        "max_cpu_parallel": MAX_CPU_PARALLEL,
        "max_slots_per_gpu": 4,
        "job_reserve_mib": JOB_RESERVE_MIB,
    }
    write_status(state)
    log(
        f"master run_loop started (packing: gpu_jobs<={MAX_GPU_PARALLEL}, "
        f"slots/gpu<=4, reserve={JOB_RESERVE_MIB}MiB)"
    )
    locks = GPULockManager(
        min_free_mib=0,
        max_slots_per_gpu=4,
        default_reserve_mib=JOB_RESERVE_MIB,
    )
    idle_rounds = 0
    jobs = materialize_all(load_jobs())
    save_jobs(jobs)

    in_flight: Dict[Any, str] = {}  # Future -> job_id
    with ThreadPoolExecutor(max_workers=MAX_POOL_WORKERS) as ex:
        while not stop_requested(HERE):
            jobs = materialize_all(load_jobs())
            jobs = _adopt_external_zs(jobs)
            save_jobs(jobs)
            # Reap finished futures without blocking on the whole batch.
            done_futs = [f for f in list(in_flight) if f.done()]
            for fut in done_futs:
                jid = in_flight.pop(fut)
                try:
                    updated = fut.result()
                except Exception as e:
                    log(f"job {jid} raised: {e!r}")
                    updated = {"job_id": jid, "status": "FAILED", "error": repr(e)}
                    out_dir = Path(next(j["output_dir"] for j in jobs if j["job_id"] == jid))
                    atomic_write_json(out_dir / "FAILED.json", updated)
                    atomic_write_json(out_dir / "job_state.json", {"job_id": jid, "status": "FAILED"})
                jobs = _update_job(jobs, updated)
                save_jobs(jobs)
                log(f"job finished {updated['job_id']} status={updated.get('status')}")
                state["job_counts"] = job_counts(jobs)
                state["last_action"] = f"finished:{updated['job_id']}"
                write_status(state)

            save_jobs(jobs)
            runnable = _collect_runnable(jobs)
            n_gpu_run = sum(1 for j in jobs if j.get("status") == "RUNNING" and j.get("kind") == "gpu")
            n_cpu_run = sum(1 for j in jobs if j.get("status") == "RUNNING" and j.get("kind") == "cpu")
            gpu_slots = max(0, MAX_GPU_PARALLEL - n_gpu_run)
            cpu_slots = max(0, MAX_CPU_PARALLEL - n_cpu_run)
            gpu_batch = [j for j in runnable if j.get("kind") == "gpu"][:gpu_slots]
            cpu_batch = [j for j in runnable if j.get("kind") == "cpu"][:cpu_slots]
            batch = gpu_batch + cpu_batch

            if not batch and not in_flight:
                if future_stages_remain(jobs) or not (HERE / "COMPLETION_PIPELINE_PASS.DONE").exists():
                    # Still have RUNNING jobs recorded but no futures (external/manual) — wait.
                    if any(j.get("status") == "RUNNING" for j in jobs):
                        idle_rounds = 0
                        state["phase"] = "RUNNING_EXTERNAL"
                        state["job_counts"] = job_counts(jobs)
                        write_status(state)
                        time.sleep(5)
                        continue
                    idle_rounds += 1
                    state["phase"] = "WAITING"
                    state["job_counts"] = job_counts(jobs)
                    write_status(state)
                    if idle_rounds == 1 or idle_rounds % 12 == 0:
                        log("WAITING: no runnable jobs but pipeline incomplete")
                    time.sleep(10)
                    continue
                state["phase"] = "all_complete"
                write_status(state)
                log("all jobs complete; exiting run_loop")
                break

            if not batch:
                # Slots full or only waiting on in-flight — poll completions.
                time.sleep(1)
                continue

            idle_rounds = 0
            for b in batch:
                for i, j in enumerate(jobs):
                    if j["job_id"] == b["job_id"]:
                        jobs[i]["status"] = "RUNNING"
            save_jobs(jobs)
            state["phase"] = "RUNNING"
            state["gpu_jobs"] = {
                str(b.get("preferred_gpu")): b["job_id"] for b in gpu_batch
            }
            state["in_flight"] = list(in_flight.values()) + [b["job_id"] for b in batch]
            state["last_action"] = f"launch:{','.join(b['job_id'] for b in batch)}"
            state["job_counts"] = job_counts(jobs)
            write_status(state)
            log(f"launching ({len(batch)} jobs, in_flight→{len(in_flight)+len(batch)}): {[b['job_id'] for b in batch]}")

            for b in batch:
                fut = ex.submit(execute_one_job, b, locks)
                in_flight[fut] = b["job_id"]

        # Drain on stop
        for fut in as_completed(list(in_flight)):
            jid = in_flight.pop(fut, None)
            try:
                updated = fut.result()
                jobs = _update_job(load_jobs(), updated)
                save_jobs(jobs)
                log(f"job finished {updated['job_id']} status={updated.get('status')} (drain)")
            except Exception as e:
                log(f"drain error {jid}: {e!r}")

    log("master run_loop exit")


def bootstrap() -> None:
    (HERE / "background_logs").mkdir(exist_ok=True)
    write_inventory(HERE / "GPU_INVENTORY.json")
    subprocess.run([PYTHON_BIN, str(HERE / "audit_evidence.py")], check=True, cwd=str(HERE))
    subprocess.run([PYTHON_BIN, str(HERE / "runners" / "freeze_protocol.py")], check=True, cwd=str(HERE))
    # Preserve existing DONE/RUNNING statuses; do not wipe the DAG.
    jobs = materialize_all(load_jobs() if (HERE / "jobs.jsonl").exists() else [])
    # Restore status from DONE.json / FAILED.json markers when present.
    for j in jobs:
        out = Path(j.get("output_dir") or (HERE / "job_outputs" / j["job_id"]))
        j["output_dir"] = str(out)
        if (out / "DONE.json").exists():
            j["status"] = "DONE"
        elif j.get("status") == "FAILED" and (out / "FAILED.json").exists():
            # Allow retry by returning to PENDING for repair jobs
            if str(j["job_id"]).startswith(("FIX_", "MCFIX_")):
                j["status"] = "PENDING"
                for name in ("FAILED.json",):
                    p = out / name
                    if p.exists():
                        p.unlink()
    save_jobs(jobs)
    log(f"bootstrap complete: {len(jobs)} jobs materialized")


def start_tmux() -> None:
    has = subprocess.run(["tmux", "has-session", "-t", SESSION_NAME], capture_output=True)
    if has.returncode == 0:
        print(f"session {SESSION_NAME} already running")
        return
    logfile = LOG_DIR / "orchestrator.log"
    cmd = (
        f"{shlex.quote(PYTHON_BIN)} -u {shlex.quote(str(HERE / 'orchestrator.py'))} "
        f"run-loop >> {shlex.quote(str(logfile))} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION_NAME, cmd], check=True)
    print(f"started tmux session {SESSION_NAME}")


def monitor() -> None:
    print(STATE_PATH.read_text(encoding="utf-8") if STATE_PATH.exists() else "{}")
    jobs = load_jobs()
    print(json.dumps(job_counts(jobs), indent=2))
    for j in jobs:
        print(f"{j['job_id']}: {j.get('status')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["bootstrap", "run-loop", "start", "monitor"])
    args = ap.parse_args()
    if args.command == "bootstrap":
        bootstrap()
    elif args.command == "run-loop":
        run_loop()
    elif args.command == "start":
        bootstrap()
        start_tmux()
    elif args.command == "monitor":
        monitor()


if __name__ == "__main__":
    main()
