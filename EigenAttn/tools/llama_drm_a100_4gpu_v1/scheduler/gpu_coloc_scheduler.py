#!/usr/bin/env python3
"""UUID-first multi-worker GPU co-location scheduler for LLAMA_DRM_A100_4GPU_V1.

Design points that matter for correctness of the unattended run:

* **UUID-first.** GPUs are tracked by their immutable `nvidia-smi` UUID, never by an
  ordinal index, because `CUDA_VISIBLE_DEVICES` in a parent process renumbers indices
  and would otherwise let two workers believe they own different devices.
* **Co-location, not one-task-per-GPU.** An A100 80GB fits several Llama-3-8B DRM
  workers. A GPU accepts another worker while
  `free >= estimated_peak + SAFETY_RESERVE` and `workers < cap`.
* **Exclusive categories.** Long-context, gov_report, Full-KV, OOM-fill and efficiency
  jobs get a whole GPU: their memory profile or their timing measurement is invalidated
  by a co-tenant.
* **External processes are sacred.** Foreign PIDs are never signalled. If every GPU is
  occupied by foreign work the scheduler reports WAITING_FOR_GPU_IDLE and idles.
* **OOM is recoverable.** Partial predictions are kept (the runner resumes per sample),
  the job is requeued as exclusive, and the co-location cap for that category drops.

Injectable `gpu_source` / `proc_source` / `launcher` hooks keep the placement algebra
unit-testable on a CPU-only machine, which is what the stage-0 auditor exercises.
"""
from __future__ import annotations

import errno
import fcntl
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C

# ---------------------------------------------------------------------------
# Scheduler states
# ---------------------------------------------------------------------------

STATE_IDLE = "IDLE"
STATE_RUNNING = "RUNNING"
STATE_WAITING_FOR_GPU_IDLE = "WAITING_FOR_GPU_IDLE"
STATE_ALL_PLACED = "ALL_PLACED"
STATE_DRAINED = "DRAINED"

WORKER_REGISTRY_FIELDS: Tuple[str, ...] = (
    "worker_id", "job_id", "category", "method_key", "task", "gpu_uuid", "gpu_index",
    "pid", "placement", "state", "started_at", "finished_at", "exit_code",
    "peak_mem_gib", "estimated_peak_gib", "attempt", "log_path", "note",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuInfo:
    uuid: str
    index: int
    name: str
    total_mib: float
    used_mib: float
    free_mib: float

    @property
    def free_gib(self) -> float:
        return self.free_mib / 1024.0

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024.0


@dataclass
class Job:
    job_id: str
    category: str
    spec: Dict[str, Any]
    method_key: str = ""
    task: str = ""
    estimated_peak_gib: float = 0.0
    exclusive: bool = False
    attempt: int = 0
    max_attempts: int = 3
    note: str = ""

    def is_exclusive(self) -> bool:
        return bool(self.exclusive) or self.category in C.EXCLUSIVE_CATEGORIES


@dataclass
class WorkerHandle:
    worker_id: str
    job: Job
    gpu_uuid: str
    gpu_index: int
    placement: str
    pid: int
    started_at: float
    log_path: str
    process: Any = None
    estimated_peak_gib: float = 0.0
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    peak_mem_gib: Optional[float] = None
    note: str = ""


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


class FileLock:
    """Advisory flock-based lock. Non-reentrant; used for the global + per-UUID locks."""

    def __init__(self, path: os.PathLike | str, timeout: float = 120.0, poll: float = 0.2):
        self.path = Path(path)
        self.timeout = timeout
        self.poll = poll
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.truncate(fd, 0)
                os.write(fd, f"{os.getpid()} {C.now_iso()}\n".encode())
                self._fd = fd
                return True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    os.close(fd)
                    raise
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(self.poll)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "FileLock":
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock {self.path} within {self.timeout}s")
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


# ---------------------------------------------------------------------------
# GPU / process discovery
# ---------------------------------------------------------------------------


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().split()[0])
    except (TypeError, ValueError, IndexError):
        return default


def query_gpus() -> List[GpuInfo]:
    rows = C.nvidia_smi_query(
        ["uuid", "index", "name", "memory.total", "memory.used", "memory.free"]
    )
    gpus: List[GpuInfo] = []
    for row in rows:
        gpus.append(
            GpuInfo(
                uuid=row.get("uuid", ""),
                index=int(_to_float(row.get("index"), -1)),
                name=row.get("name", ""),
                total_mib=_to_float(row.get("memory.total")),
                used_mib=_to_float(row.get("memory.used")),
                free_mib=_to_float(row.get("memory.free")),
            )
        )
    return [gpu for gpu in gpus if gpu.uuid]


def query_compute_apps() -> List[Dict[str, Any]]:
    rows = C.nvidia_smi_query(
        ["gpu_uuid", "pid", "used_gpu_memory"], query="--query-compute-apps"
    )
    apps: List[Dict[str, Any]] = []
    for row in rows:
        try:
            pid = int(_to_float(row.get("pid"), -1))
        except Exception:
            continue
        if pid <= 0:
            continue
        apps.append(
            {
                "gpu_uuid": row.get("gpu_uuid", ""),
                "pid": pid,
                "used_mib": _to_float(row.get("used_gpu_memory")),
            }
        )
    return apps


# ---------------------------------------------------------------------------
# Peak estimation
# ---------------------------------------------------------------------------


def estimated_peak_gib(actual_peak_gib: float) -> float:
    """Protocol rule: max(actual_peak * 1.20, actual_peak + 4 GiB)."""
    if actual_peak_gib <= 0:
        return 0.0
    return max(actual_peak_gib * C.PEAK_MULTIPLIER, actual_peak_gib + C.PEAK_HEADROOM_GIB)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class GpuColocScheduler:
    def __init__(
        self,
        result_dir: os.PathLike | str,
        *,
        gpu_source: Callable[[], List[GpuInfo]] = query_gpus,
        proc_source: Callable[[], List[Dict[str, Any]]] = query_compute_apps,
        launcher: Optional[Callable[[Job, GpuInfo, str], WorkerHandle]] = None,
        max_workers_per_gpu: int = C.DEFAULT_MAX_WORKERS_PER_GPU,
        safety_reserve_gib: float = C.SAFETY_RESERVE_GIB,
        canary: bool = False,
        status_dir: Optional[os.PathLike | str] = None,
        locks_dir: Optional[os.PathLike | str] = None,
    ) -> None:
        self.result_dir = Path(result_dir)
        self.status_dir = Path(status_dir) if status_dir else self.result_dir / "status"
        self.locks_dir = Path(locks_dir) if locks_dir else self.result_dir / "locks"
        self.log_dir = self.result_dir / "logs"
        for directory in (self.status_dir, self.locks_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.gpu_source = gpu_source
        self.proc_source = proc_source
        self.launcher = launcher or self._default_launcher

        # Canary mode deliberately serialises onto a single GPU so the first measured
        # peak is uncontaminated by a co-tenant.
        self.canary = canary
        self.max_workers_per_gpu = 1 if canary else int(max_workers_per_gpu)
        self.max_gpus = 1 if canary else None
        self.safety_reserve_gib = float(safety_reserve_gib)

        self.queue: List[Job] = []
        self.workers: Dict[str, WorkerHandle] = {}
        self.completed: List[WorkerHandle] = []
        self.failed: List[WorkerHandle] = []

        #: Per-category co-location caps, lowered on co-located OOM.
        self.category_caps: Dict[str, int] = {}
        self.state = STATE_IDLE
        self._worker_seq = 0
        self._registry_rows: List[Dict[str, Any]] = []
        self._pinned_gpu_uuids: Optional[List[str]] = None
        self.last_placement_reason = ""

    # -- locks ------------------------------------------------------------

    def global_lock(self, timeout: float = 300.0) -> FileLock:
        return FileLock(self.locks_dir / "scheduler.global.lock", timeout=timeout)

    def gpu_lock(self, gpu_uuid: str, timeout: float = 120.0) -> FileLock:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", gpu_uuid)
        return FileLock(self.locks_dir / f"gpu.{safe}.lock", timeout=timeout)

    # -- queue ------------------------------------------------------------

    def submit(self, job: Job) -> None:
        self.queue.append(job)

    def submit_all(self, jobs: Sequence[Job]) -> None:
        for job in jobs:
            self.submit(job)

    def pending_count(self) -> int:
        return len(self.queue)

    def active_count(self) -> int:
        return len(self.workers)

    # -- GPU view ---------------------------------------------------------

    def visible_gpus(self) -> List[GpuInfo]:
        gpus = self.gpu_source()
        if self._pinned_gpu_uuids is not None:
            allowed = set(self._pinned_gpu_uuids)
            gpus = [gpu for gpu in gpus if gpu.uuid in allowed]
        elif self.max_gpus is not None:
            gpus = sorted(gpus, key=lambda g: g.index)[: self.max_gpus]
            self._pinned_gpu_uuids = [gpu.uuid for gpu in gpus]
        return gpus

    def our_pids(self) -> Dict[int, str]:
        return {handle.pid: handle.worker_id for handle in self.workers.values() if handle.pid > 0}

    def external_procs_by_gpu(self) -> Dict[str, List[Dict[str, Any]]]:
        """Foreign compute apps per GPU UUID. These are never signalled."""
        ours = self.our_pids()
        out: Dict[str, List[Dict[str, Any]]] = {}
        for app in self.proc_source():
            if app["pid"] in ours or app["pid"] == os.getpid():
                continue
            out.setdefault(app["gpu_uuid"], []).append(app)
        return out

    def workers_by_gpu(self) -> Dict[str, List[WorkerHandle]]:
        out: Dict[str, List[WorkerHandle]] = {}
        for handle in self.workers.values():
            out.setdefault(handle.gpu_uuid, []).append(handle)
        return out

    def cap_for(self, category: str) -> int:
        return int(self.category_caps.get(category, self.max_workers_per_gpu))

    # -- placement --------------------------------------------------------

    def can_place(
        self,
        job: Job,
        gpu: GpuInfo,
        *,
        workers_on_gpu: Sequence[WorkerHandle],
        external_on_gpu: Sequence[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        if job.is_exclusive():
            if workers_on_gpu:
                return False, f"exclusive job needs an empty GPU; {len(workers_on_gpu)} own worker(s) present"
            if external_on_gpu:
                return False, "exclusive job needs an empty GPU; external process present"
            return True, "exclusive placement"

        cap = self.cap_for(job.category)
        if len(workers_on_gpu) >= cap:
            return False, f"co-location cap reached ({len(workers_on_gpu)}/{cap})"

        # An exclusive tenant already on this GPU must not be joined.
        if any(handle.placement == "exclusive" for handle in workers_on_gpu):
            return False, "GPU holds an exclusive worker"

        required = job.estimated_peak_gib + self.safety_reserve_gib
        available = self.effective_free_gib(gpu, workers_on_gpu, external_on_gpu)
        if available < required:
            return (
                False,
                f"insufficient free memory: {available:.1f} GiB available < "
                f"{job.estimated_peak_gib:.1f} + {self.safety_reserve_gib:.1f} GiB",
            )
        return True, (
            f"co-located ({len(workers_on_gpu) + 1}/{cap}), available {available:.1f} GiB >= "
            f"{required:.1f} GiB"
        )

    def effective_free_gib(
        self,
        gpu: GpuInfo,
        workers_on_gpu: Sequence[WorkerHandle],
        external_on_gpu: Sequence[Dict[str, Any]],
    ) -> float:
        """Free memory that accounts for workers that have not yet reached their peak.

        `nvidia-smi` reports what is allocated *now*, so a worker launched seconds ago
        still looks nearly free. Trusting that number would let a burst of launches
        over-subscribe a GPU and OOM every tenant. The projected view charges each of our
        own workers its full estimated peak and takes the pessimistic minimum of the two.
        """
        external_used = sum(float(app.get("used_mib", 0.0)) for app in external_on_gpu) / 1024.0
        projected_used = sum(float(handle.estimated_peak_gib or 0.0) for handle in workers_on_gpu)
        projected_free = gpu.total_gib - external_used - projected_used
        return min(gpu.free_gib, projected_free)

    def choose_gpu(self, job: Job) -> Tuple[Optional[GpuInfo], str, str]:
        """Pick a GPU for `job`. Returns (gpu, placement, reason)."""
        gpus = self.visible_gpus()
        if not gpus:
            return None, "", "no GPUs visible"
        workers_map = self.workers_by_gpu()
        external_map = self.external_procs_by_gpu()

        # Prefer emptier GPUs for exclusive jobs and fuller-but-fitting GPUs for
        # co-located ones, so exclusive work keeps a clean device available.
        if job.is_exclusive():
            ordered = sorted(gpus, key=lambda g: (-g.free_gib, g.index))
        else:
            ordered = sorted(gpus, key=lambda g: (len(workers_map.get(g.uuid, [])) == 0, -g.free_gib, g.index))

        reasons: List[str] = []
        for gpu in ordered:
            ok, reason = self.can_place(
                job,
                gpu,
                workers_on_gpu=workers_map.get(gpu.uuid, []),
                external_on_gpu=external_map.get(gpu.uuid, []),
            )
            if ok:
                return gpu, ("exclusive" if job.is_exclusive() else "coloc"), reason
            reasons.append(f"gpu{gpu.index}: {reason}")
        return None, "", "; ".join(reasons)

    def blocked_only_by_external(self) -> bool:
        """True when every visible GPU is held by foreign work (never kill it, just wait)."""
        gpus = self.visible_gpus()
        if not gpus:
            return False
        external_map = self.external_procs_by_gpu()
        workers_map = self.workers_by_gpu()
        if any(workers_map.get(gpu.uuid) for gpu in gpus):
            return False
        return all(external_map.get(gpu.uuid) for gpu in gpus)

    # -- launching --------------------------------------------------------

    def _next_worker_id(self, job: Job) -> str:
        self._worker_seq += 1
        return f"w{self._worker_seq:05d}_{job.job_id}"

    def _default_launcher(self, job: Job, gpu: GpuInfo, worker_id: str) -> WorkerHandle:
        """Spawn worker_entry.py with the GPU pinned by UUID."""
        log_path = self.log_dir / f"{worker_id}.log"
        spec_path = self.result_dir / "runs" / f"{worker_id}.job.json"
        C.atomic_write_json(spec_path, {
            "worker_id": worker_id,
            "job_id": job.job_id,
            "category": job.category,
            "method_key": job.method_key,
            "task": job.task,
            "attempt": job.attempt,
            "placement": "exclusive" if job.is_exclusive() else "coloc",
            "gpu_uuid": gpu.uuid,
            "gpu_index": gpu.index,
            "estimated_peak_gib": job.estimated_peak_gib,
            "spec": job.spec,
        })
        cmd = [C.python_bin(), str(C.WORKER_ENTRY), "--job-spec", str(spec_path)]
        # CUDA_VISIBLE_DEVICES is set by UUID: immune to index renumbering.
        env = C.offline_env({
            "CUDA_VISIBLE_DEVICES": gpu.uuid,
            "LLAMA_DRM_WORKER_ID": worker_id,
            "LLAMA_DRM_GPU_UUID": gpu.uuid,
        })
        handle_log = open(log_path, "a", encoding="utf-8")
        handle_log.write(f"\n=== {C.now_iso()} launch {worker_id} on {gpu.uuid} (idx {gpu.index}) ===\n")
        handle_log.write(f"cmd: {' '.join(shlex.quote(part) for part in cmd)}\n")
        handle_log.flush()
        process = subprocess.Popen(
            cmd, stdout=handle_log, stderr=subprocess.STDOUT, env=env,
            cwd=str(C.TOOL_ROOT), start_new_session=True,
        )
        return WorkerHandle(
            worker_id=worker_id, job=job, gpu_uuid=gpu.uuid, gpu_index=gpu.index,
            placement="exclusive" if job.is_exclusive() else "coloc",
            pid=process.pid, started_at=time.time(), log_path=str(log_path),
            process=process, estimated_peak_gib=job.estimated_peak_gib,
        )

    def try_launch_one(self) -> Optional[WorkerHandle]:
        if not self.queue:
            return None
        for position, job in enumerate(list(self.queue)):
            gpu, placement, reason = self.choose_gpu(job)
            if gpu is None:
                self.last_placement_reason = reason
                continue
            worker_id = self._next_worker_id(job)
            with self.gpu_lock(gpu.uuid):
                job.attempt += 1
                handle = self.launcher(job, gpu, worker_id)
            handle.placement = placement
            self.queue.pop(position)
            self.workers[worker_id] = handle
            self._record(handle, state="RUNNING")
            self.last_placement_reason = reason
            return handle
        return None

    def launch_ready(self, max_launches: int = 8) -> List[WorkerHandle]:
        launched: List[WorkerHandle] = []
        for _ in range(max_launches):
            handle = self.try_launch_one()
            if handle is None:
                break
            launched.append(handle)
        return launched

    # -- reaping ----------------------------------------------------------

    @staticmethod
    def _worker_result(handle: WorkerHandle) -> Dict[str, Any]:
        spec_dir = Path(handle.job.spec.get("output_dir", "")) if handle.job.spec else None
        if spec_dir is None:
            return {}
        return C.read_json(spec_dir / "worker_result.json", default={}) or {}

    def poll(self) -> List[WorkerHandle]:
        """Reap finished workers; classify OOM and requeue exclusively."""
        finished: List[WorkerHandle] = []
        for worker_id, handle in list(self.workers.items()):
            process = handle.process
            if process is None:
                continue
            code = process.poll()
            if code is None:
                continue
            handle.exit_code = int(code)
            handle.finished_at = time.time()
            result = self._worker_result(handle)
            if result:
                handle.peak_mem_gib = result.get("peak_mem_gib")
                handle.note = str(result.get("note", ""))
            del self.workers[worker_id]
            finished.append(handle)

            is_oom = bool(result.get("oom")) or handle.exit_code == C_OOM_EXIT
            if handle.exit_code == 0:
                self.completed.append(handle)
                self._record(handle, state="COMPLETED")
            elif is_oom:
                self._handle_oom(handle)
            else:
                self.failed.append(handle)
                self._record(handle, state="FAILED")
        return finished

    def _handle_oom(self, handle: WorkerHandle) -> None:
        """Keep partial results, drop the category cap, retry the job exclusively."""
        job = handle.job
        if handle.placement == "coloc":
            current = self.cap_for(job.category)
            self.category_caps[job.category] = max(1, current - 1)
        self._record(
            handle,
            state="OOM",
            note=(
                f"OOM under {handle.placement}; partial predictions retained; "
                f"category cap now {self.cap_for(job.category)}"
            ),
        )
        if job.attempt < job.max_attempts:
            retry = Job(
                job_id=job.job_id, category=job.category, spec=job.spec,
                method_key=job.method_key, task=job.task,
                estimated_peak_gib=job.estimated_peak_gib,
                exclusive=True, attempt=job.attempt, max_attempts=job.max_attempts,
                note=f"exclusive retry after OOM (attempt {job.attempt})",
            )
            self.queue.insert(0, retry)
        else:
            self.failed.append(handle)
            self._record(handle, state="FAILED_OOM_EXHAUSTED",
                         note="exclusive retries exhausted")

    def terminate_own_workers(self, grace: float = 30.0) -> None:
        """Stop only our own workers. External PIDs are never signalled."""
        for handle in list(self.workers.values()):
            process = handle.process
            if process is None or process.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(handle.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    process.terminate()
                except Exception:
                    pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and self.workers:
            self.poll()
            time.sleep(0.5)

    # -- status persistence ----------------------------------------------

    def _record(self, handle: WorkerHandle, *, state: str, note: str = "") -> None:
        row = {
            "worker_id": handle.worker_id,
            "job_id": handle.job.job_id,
            "category": handle.job.category,
            "method_key": handle.job.method_key,
            "task": handle.job.task,
            "gpu_uuid": handle.gpu_uuid,
            "gpu_index": handle.gpu_index,
            "pid": handle.pid,
            "placement": handle.placement,
            "state": state,
            "started_at": _iso(handle.started_at),
            "finished_at": _iso(handle.finished_at),
            "exit_code": "" if handle.exit_code is None else handle.exit_code,
            "peak_mem_gib": "" if handle.peak_mem_gib is None else round(float(handle.peak_mem_gib), 3),
            "estimated_peak_gib": round(float(handle.estimated_peak_gib or 0.0), 3),
            "attempt": handle.job.attempt,
            "log_path": handle.log_path,
            "note": note or handle.note or handle.job.note,
        }
        self._registry_rows.append(row)
        C.atomic_write_csv(
            self.status_dir / "WORKER_REGISTRY.csv", self._registry_rows, WORKER_REGISTRY_FIELDS
        )

    def compute_state(self) -> str:
        if self.workers:
            return STATE_RUNNING
        if not self.queue:
            return STATE_DRAINED
        if self.blocked_only_by_external():
            return STATE_WAITING_FOR_GPU_IDLE
        return STATE_IDLE

    def write_status(self) -> Dict[str, Any]:
        gpus = self.visible_gpus()
        workers_map = self.workers_by_gpu()
        external_map = self.external_procs_by_gpu()
        self.state = self.compute_state()

        gpu_status = {
            "protocol_id": C.PROTOCOL_ID,
            "updated_at": C.now_iso(),
            "scheduler_state": self.state,
            "canary": self.canary,
            "max_workers_per_gpu": self.max_workers_per_gpu,
            "safety_reserve_gib": self.safety_reserve_gib,
            "category_caps": dict(self.category_caps),
            "last_placement_reason": self.last_placement_reason,
            "gpus": [
                {
                    "uuid": gpu.uuid,
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_gib": round(gpu.total_gib, 2),
                    "free_gib": round(gpu.free_gib, 2),
                    "available_gib": round(
                        self.effective_free_gib(
                            gpu, workers_map.get(gpu.uuid, []), external_map.get(gpu.uuid, [])
                        ),
                        2,
                    ),
                    "own_workers": [h.worker_id for h in workers_map.get(gpu.uuid, [])],
                    "own_worker_count": len(workers_map.get(gpu.uuid, [])),
                    "external_pids": [app["pid"] for app in external_map.get(gpu.uuid, [])],
                    "external_used_gib": round(
                        sum(app["used_mib"] for app in external_map.get(gpu.uuid, [])) / 1024.0, 2
                    ),
                    "exclusive_held": any(
                        h.placement == "exclusive" for h in workers_map.get(gpu.uuid, [])
                    ),
                }
                for gpu in gpus
            ],
        }
        C.atomic_write_json(self.status_dir / "GPU_STATUS.json", gpu_status)

        current_workers = {
            "protocol_id": C.PROTOCOL_ID,
            "updated_at": C.now_iso(),
            "scheduler_state": self.state,
            "pending_jobs": len(self.queue),
            "active_workers": len(self.workers),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "workers": [
                {
                    "worker_id": handle.worker_id,
                    "job_id": handle.job.job_id,
                    "category": handle.job.category,
                    "method_key": handle.job.method_key,
                    "task": handle.job.task,
                    "gpu_uuid": handle.gpu_uuid,
                    "gpu_index": handle.gpu_index,
                    "pid": handle.pid,
                    "placement": handle.placement,
                    "attempt": handle.job.attempt,
                    "elapsed_s": round(time.time() - handle.started_at, 1),
                    "estimated_peak_gib": round(handle.estimated_peak_gib, 2),
                    "log_path": handle.log_path,
                }
                for handle in self.workers.values()
            ],
            "queue_preview": [
                {
                    "job_id": job.job_id, "category": job.category,
                    "method_key": job.method_key, "task": job.task,
                    "exclusive": job.is_exclusive(), "attempt": job.attempt,
                }
                for job in self.queue[:32]
            ],
        }
        C.atomic_write_json(self.status_dir / "CURRENT_WORKERS.json", current_workers)
        return {"gpu_status": gpu_status, "current_workers": current_workers}

    # -- main step --------------------------------------------------------

    def step(self, *, launch: bool = True, max_launches: int = 8) -> Dict[str, Any]:
        """One scheduling tick: reap, launch what fits, persist status."""
        finished = self.poll()
        launched: List[WorkerHandle] = []
        if launch:
            with self.global_lock():
                launched = self.launch_ready(max_launches=max_launches)
        self.write_status()
        return {
            "finished": [handle.worker_id for handle in finished],
            "launched": [handle.worker_id for handle in launched],
            "state": self.state,
            "pending": len(self.queue),
            "active": len(self.workers),
        }


#: worker_entry.py exits with this code on CUDA OOM so the scheduler can distinguish
#: a memory failure (retry exclusively) from a genuine crash (fail the job).
C_OOM_EXIT = 42


def _iso(epoch: Optional[float]) -> str:
    if not epoch:
        return ""
    import datetime as dt

    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Self-test helpers (used by stage-0 audit; no GPU required)
# ---------------------------------------------------------------------------


def make_mock_gpu_source(specs: Sequence[Dict[str, Any]]) -> Callable[[], List[GpuInfo]]:
    def _source() -> List[GpuInfo]:
        return [
            GpuInfo(
                uuid=spec["uuid"], index=int(spec.get("index", i)),
                name=spec.get("name", "MOCK-A100-80GB"),
                total_mib=float(spec.get("total_mib", 81920)),
                used_mib=float(spec.get("used_mib", 0)),
                free_mib=float(spec.get("free_mib", 81920)),
            )
            for i, spec in enumerate(specs)
        ]

    return _source


def make_mock_launcher(sink: List[WorkerHandle]) -> Callable[[Job, GpuInfo, str], WorkerHandle]:
    """Launcher that fabricates a never-exiting handle, for placement-algebra tests."""
    counter = {"pid": 900000}

    def _launch(job: Job, gpu: GpuInfo, worker_id: str) -> WorkerHandle:
        counter["pid"] += 1
        handle = WorkerHandle(
            worker_id=worker_id, job=job, gpu_uuid=gpu.uuid, gpu_index=gpu.index,
            placement="exclusive" if job.is_exclusive() else "coloc",
            pid=counter["pid"], started_at=time.time(),
            log_path=f"/dev/null/{worker_id}.log", process=None,
            estimated_peak_gib=job.estimated_peak_gib,
        )
        sink.append(handle)
        return handle

    return _launch
