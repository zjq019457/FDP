#!/usr/bin/env python3
"""Shared atomic I/O, hashing, and path helpers for legacy rebuttal pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SESSION_NAME = "eigenattn_legacy_rebuttal"
HEARTBEAT_SECS = 10
GPU_RECHECK_SECS = 30
# Shared-GPU policy: do not require large free VRAM; co-exist with foreign jobs.
MIN_FREE_MIB_DEFAULT = 0
ALLOW_SHARED_GPU = True
PYTHON_BIN = "/home/bai-project/anaconda3/envs/eigen/bin/python"

# Large artifacts live on /diskF to avoid filling the root filesystem.
ARTIFACT_ROOT = Path("/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts")

PPL_FLOAT_TOL = 1e-5
ACC_FLOAT_TOL = 1e-9


def ensure_artifact_root() -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_ROOT


def artifact_dir(name: str, *, link_under: Optional[Path] = None) -> Path:
    """Return diskF artifact dir; optionally maintain a symlink under state_dir/artifacts."""
    ensure_artifact_root()
    target = ARTIFACT_ROOT / name
    target.mkdir(parents=True, exist_ok=True)
    if link_under is not None:
        link_under = Path(link_under)
        link_under.parent.mkdir(parents=True, exist_ok=True)
        if link_under.is_symlink():
            if os.path.realpath(link_under) != os.path.realpath(target):
                link_under.unlink()
                link_under.symlink_to(target)
        elif not link_under.exists():
            link_under.symlink_to(target)
    return target


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def state_dir() -> Path:
    return Path(__file__).resolve().parent


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


def sha256_tensor_ids(ids) -> str:
    """Stable hash for 1D integer token ids (list/tensor)."""
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().tolist()
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return sha256_text(",".join(str(int(x)) for x in ids))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=indent, default=str) + "\n"
    atomic_write_text(path, payload)


def write_done_marker(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def append_jsonl_atomic_line(path: Path, obj: Any) -> None:
    """Append one JSONL line atomically via exclusive open+fsync of whole rewrite for small files;
    for large streaming, prefer per-window files then merge."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def write_window_atomic(window_dir: Path, window_id: str, record: Dict[str, Any]) -> Path:
    """Write one window/sample as <id>.json then <id>.DONE."""
    window_dir = Path(window_dir)
    window_dir.mkdir(parents=True, exist_ok=True)
    base = window_dir / f"{window_id}"
    json_path = base.with_suffix(".json")
    done_path = Path(str(base) + ".DONE")
    atomic_write_json(json_path, record)
    write_done_marker(
        done_path,
        {
            "window_id": window_id,
            "json_sha256": sha256_file(json_path),
            "runner_hash": record.get("runner_hash"),
            "model_hash": record.get("model_hash"),
            "raw_hash": sha256_json(record),
            "timestamp": now_iso(),
        },
    )
    return json_path


def window_is_complete(window_dir: Path, window_id: str, expected: Dict[str, str]) -> bool:
    base = Path(window_dir) / f"{window_id}"
    json_path = base.with_suffix(".json")
    done_path = Path(str(base) + ".DONE")
    if not json_path.exists() or not done_path.exists():
        return False
    if json_path.stat().st_size == 0:
        return False
    try:
        rec = json.loads(json_path.read_text(encoding="utf-8"))
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if done.get("json_sha256") != sha256_file(json_path):
        return False
    for k, v in expected.items():
        if rec.get(k) != v:
            return False
        if done.get(k) not in (None, v) and done.get(k) != v:
            return False
    return True


def merge_window_dir_to_jsonl(window_dir: Path, out_jsonl: Path) -> int:
    window_dir = Path(window_dir)
    rows = []
    for p in sorted(window_dir.glob("*.json")):
        done = Path(str(p.with_suffix("")) + ".DONE")
        if not done.exists():
            continue
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_jsonl)
    return len(rows)


def stop_requested(root: Optional[Path] = None) -> bool:
    d = root or state_dir()
    return (d / "STOP_REQUESTED").exists()


def gate_payload(
    *,
    name: str,
    passed: bool,
    upstream_hashes: Dict[str, str],
    runner_hash: str,
    protocol_hash: str,
    evaluation_hash: str,
    summary: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    obj = {
        "gate": name,
        "passed": passed,
        "timestamp": now_iso(),
        "upstream_artifact_hashes": upstream_hashes,
        "runner_hash": runner_hash,
        "protocol_hash": protocol_hash,
        "evaluation_hash": evaluation_hash,
        "audit_summary": summary,
    }
    if extra:
        obj["extra"] = extra
    return obj


def write_pass_gate(path: Path, payload: Dict[str, Any]) -> None:
    if not payload.get("passed"):
        raise ValueError("refusing to write PASS gate with passed=false")
    if not payload.get("runner_hash") or not payload.get("protocol_hash"):
        raise ValueError("PASS gate requires runner_hash and protocol_hash")
    atomic_write_json(path, payload)


def file_hash_or_empty(path: Path) -> str:
    p = Path(path)
    return sha256_file(p) if p.is_file() else ""
