#!/usr/bin/env python3
"""Four-GPU resource inventory with VRAM packing (multiple jobs per GPU)."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    GPU_RECHECK_SECS,
    JOB_RESERVE_MIB_DEFAULT,
    MAX_SLOTS_PER_GPU,
    MIN_FREE_MIB_DEFAULT,
    atomic_write_json,
    now_iso,
    here as state_dir,
)

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
    """Pack multiple pipeline jobs onto the same physical GPU when VRAM allows."""

    def __init__(
        self,
        lock_path: Optional[Path] = None,
        min_free_mib: int = MIN_FREE_MIB_DEFAULT,
        *,
        max_slots_per_gpu: int = MAX_SLOTS_PER_GPU,
        default_reserve_mib: int = JOB_RESERVE_MIB_DEFAULT,
    ):
        self.lock_path = lock_path or (state_dir() / "GPU_RESOURCE_LOCKS.json")
        self.min_free_mib = min_free_mib
        self.max_slots_per_gpu = max_slots_per_gpu
        self.default_reserve_mib = default_reserve_mib
        if not self.lock_path.exists():
            self._save({"slots": {}, "updated_at": now_iso()})

    def _load(self) -> Dict[str, Any]:
        raw = json.loads(self.lock_path.read_text(encoding="utf-8"))
        # Migrate legacy exclusive {"locks": {uuid: info}} -> slots lists.
        if "slots" not in raw and "locks" in raw:
            slots: Dict[str, List[Dict[str, Any]]] = {}
            for uuid, info in (raw.get("locks") or {}).items():
                slots[uuid] = [info] if isinstance(info, dict) else list(info)
            raw = {"slots": slots, "updated_at": raw.get("updated_at", now_iso())}
        raw.setdefault("slots", {})
        return raw

    def _save(self, obj: Dict[str, Any]) -> None:
        obj["updated_at"] = now_iso()
        atomic_write_json(self.lock_path, obj)

    def _prune_dead(self, slots: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        alive: Dict[str, List[Dict[str, Any]]] = {}
        for uuid, holders in slots.items():
            keep = []
            for info in holders or []:
                pid = info.get("holder_pid")
                if pid and Path(f"/proc/{pid}").exists():
                    keep.append(info)
            if keep:
                alive[uuid] = keep
        return alive

    def release(self, uuid: str, job_id: Optional[str] = None) -> None:
        with _LOCK:
            data = self._load()
            slots = self._prune_dead(data.get("slots") or {})
            holders = list(slots.get(uuid) or [])
            if job_id is None:
                holders = []
            else:
                holders = [h for h in holders if h.get("job_id") != job_id]
            if holders:
                slots[uuid] = holders
            else:
                slots.pop(uuid, None)
            data["slots"] = slots
            self._save(data)

    def try_acquire(
        self,
        job_id: str,
        preferred_index: Optional[int] = None,
        *,
        reserve_mib: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        need = int(reserve_mib if reserve_mib is not None else self.default_reserve_mib)
        with _LOCK:
            inv = query_gpu_inventory()
            data = self._load()
            slots = self._prune_dead(data.get("slots") or {})

            candidates = list(inv["gpus"])
            if preferred_index is not None:
                candidates = sorted(
                    candidates, key=lambda g: (0 if g["index"] == preferred_index else 1, -g["memory_free_mib"])
                )
            else:
                # Prefer GPUs with more free VRAM and fewer pipeline slots.
                candidates = sorted(
                    candidates,
                    key=lambda g: (len(slots.get(g["uuid"]) or []), -g["memory_free_mib"]),
                )

            for g in candidates:
                uuid = g["uuid"]
                holders = list(slots.get(uuid) or [])
                if len(holders) >= self.max_slots_per_gpu:
                    continue
                # Account for our reserved headroom still not reflected in nvidia-smi.
                reserved = sum(int(h.get("reserved_mib") or 0) for h in holders)
                # Free after current device use, minus reservations from sibling slots
                # that may not have allocated yet.
                effective_free = g["memory_free_mib"] - max(0, reserved - g["memory_used_mib"])
                # Simpler gate: require free VRAM >= need, and slot count under cap.
                if g["memory_free_mib"] < max(need, self.min_free_mib):
                    continue
                if effective_free < need and len(holders) > 0:
                    # Still allow first pack if nvidia shows enough free.
                    if g["memory_free_mib"] < need:
                        continue
                info = {
                    "job_id": job_id,
                    "gpu_index": g["index"],
                    "gpu_uuid": uuid,
                    "gpu_name": g["name"],
                    "memory_total_mib": g["memory_total_mib"],
                    "memory_free_mib_at_acquire": g["memory_free_mib"],
                    "reserved_mib": need,
                    "holder_pid": os.getpid(),
                    "cuda_visible_devices": str(g["index"]),
                    "slot_index": len(holders),
                    "acquired_at": now_iso(),
                }
                holders.append(info)
                slots[uuid] = holders
                data["slots"] = slots
                self._save(data)
                return info
            data["slots"] = slots
            self._save(data)
            return None

    def wait_acquire(
        self,
        job_id: str,
        preferred_index: Optional[int] = None,
        stop_fn=None,
        *,
        reserve_mib: Optional[int] = None,
    ) -> Dict[str, Any]:
        while True:
            if stop_fn and stop_fn():
                raise RuntimeError("STOP_REQUESTED while waiting for GPU")
            got = self.try_acquire(job_id, preferred_index=preferred_index, reserve_mib=reserve_mib)
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
        "- policy: VRAM packing — multiple pipeline jobs per GPU when free memory allows.",
        f"- max_slots_per_gpu: `{MAX_SLOTS_PER_GPU}`",
        f"- default_reserve_mib: `{JOB_RESERVE_MIB_DEFAULT}`",
        f"- recheck_secs: `{GPU_RECHECK_SECS}`",
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
