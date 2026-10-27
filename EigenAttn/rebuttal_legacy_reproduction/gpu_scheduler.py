#!/usr/bin/env python3
"""Four-GPU resource inventory and exclusive locks (UUID-first)."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import GPU_RECHECK_SECS, MIN_FREE_MIB_DEFAULT, atomic_write_json, now_iso, state_dir

_LOCK = threading.Lock()


def _run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def query_gpu_inventory() -> Dict[str, Any]:
    q = (
        "index,uuid,name,memory.total,memory.free,memory.used,"
        "utilization.gpu,utilization.memory"
    )
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": int(float(parts[3])),
                "memory_free_mib": int(float(parts[4])),
                "memory_used_mib": int(float(parts[5])),
                "util_gpu": float(parts[6]),
                "util_mem": float(parts[7]),
            }
        )
    # external processes
    try:
        p = _run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        procs = []
        for line in p.strip().splitlines():
            if not line.strip():
                continue
            pp = [x.strip() for x in line.split(",")]
            procs.append(
                {
                    "gpu_uuid": pp[0],
                    "pid": int(pp[1]) if pp[1].isdigit() else pp[1],
                    "process_name": pp[2] if len(pp) > 2 else "",
                    "used_memory_mib": float(pp[3]) if len(pp) > 3 else None,
                }
            )
    except Exception as e:
        procs = [{"error": repr(e)}]
    return {"generated_at": now_iso(), "gpus": gpus, "compute_apps": procs}


def write_inventory(path: Optional[Path] = None) -> Dict[str, Any]:
    inv = query_gpu_inventory()
    path = path or (state_dir() / "GPU_INVENTORY.json")
    atomic_write_json(path, inv)
    return inv


class GPULockManager:
    def __init__(self, lock_path: Optional[Path] = None, min_free_mib: int = MIN_FREE_MIB_DEFAULT):
        self.lock_path = lock_path or (state_dir() / "GPU_RESOURCE_LOCKS.json")
        self.min_free_mib = min_free_mib
        if not self.lock_path.exists():
            self._save({"locks": {}, "updated_at": now_iso()})

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.lock_path.read_text(encoding="utf-8"))

    def _save(self, obj: Dict[str, Any]) -> None:
        obj["updated_at"] = now_iso()
        atomic_write_json(self.lock_path, obj)

    def release(self, uuid: str, job_id: Optional[str] = None) -> None:
        with _LOCK:
            data = self._load()
            locks = data.get("locks", {})
            cur = locks.get(uuid)
            if cur and (job_id is None or cur.get("job_id") == job_id):
                locks.pop(uuid, None)
            data["locks"] = locks
            self._save(data)

    def try_acquire(self, job_id: str, preferred_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        with _LOCK:
            inv = query_gpu_inventory()
            data = self._load()
            locks = data.get("locks", {})
            # drop stale locks whose pid is dead
            alive = {}
            for uuid, info in locks.items():
                pid = info.get("holder_pid")
                if pid and Path(f"/proc/{pid}").exists():
                    alive[uuid] = info
            locks = alive

            candidates = inv["gpus"]
            if preferred_index is not None:
                candidates = sorted(candidates, key=lambda g: 0 if g["index"] == preferred_index else 1)

            for g in candidates:
                uuid = g["uuid"]
                if uuid in locks:
                    continue
                if g["memory_free_mib"] < self.min_free_mib:
                    continue
                info = {
                    "job_id": job_id,
                    "gpu_index": g["index"],
                    "gpu_uuid": uuid,
                    "gpu_name": g["name"],
                    "memory_total_mib": g["memory_total_mib"],
                    "memory_free_mib_at_acquire": g["memory_free_mib"],
                    "holder_pid": os.getpid(),
                    "cuda_visible_devices": str(g["index"]),
                    "acquired_at": now_iso(),
                }
                locks[uuid] = info
                data["locks"] = locks
                self._save(data)
                return info
            self._save({"locks": locks})
            return None

    def wait_acquire(
        self, job_id: str, preferred_index: Optional[int] = None, stop_fn=None
    ) -> Dict[str, Any]:
        while True:
            if stop_fn and stop_fn():
                raise RuntimeError("STOP_REQUESTED while waiting for GPU")
            got = self.try_acquire(job_id, preferred_index=preferred_index)
            if got:
                return got
            time.sleep(GPU_RECHECK_SECS)


def write_scheduler_audit(path: Optional[Path] = None) -> Path:
    path = path or (state_dir() / "MULTIGPU_SCHEDULER_AUDIT.md")
    inv = write_inventory()
    lines = [
        "# Multi-GPU Scheduler Audit",
        "",
        f"- generated_at: `{now_iso()}`",
        f"- n_gpus: `{len(inv['gpus'])}`",
        "- policy: UUID-first locks among OUR jobs only; share physical GPUs with foreign processes; never kill foreign processes.",
        f"- recheck_secs: `{GPU_RECHECK_SECS}`",
        f"- min_free_mib_default: `{MIN_FREE_MIB_DEFAULT}` (0 = no free-VRAM gate; shared GPU allowed)",
        "",
        "## Inventory",
        "",
    ]
    for g in inv["gpus"]:
        lines.append(
            f"- index={g['index']} uuid=`{g['uuid']}` name=`{g['name']}` "
            f"free={g['memory_free_mib']}MiB total={g['memory_total_mib']}MiB"
        )
    lines.extend(["", "## Compute apps", ""])
    for p in inv.get("compute_apps", []):
        lines.append(f"- `{p}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
