#!/usr/bin/env python3
"""Shared utilities for rebuttal completion pipeline. Legacy dirs remain read-only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SESSION_NAME = "fdp_rebuttal_completion"
HEARTBEAT_SECS = 10
GPU_RECHECK_SECS = 10
MIN_FREE_MIB_DEFAULT = 0
ALLOW_SHARED_GPU = True
# Pack multiple Llama-8B evals (~16GB) onto each 80GB A100.
MAX_SLOTS_PER_GPU = 4
JOB_RESERVE_MIB_DEFAULT = 18000
PYTHON_BIN = "/home/bai-project/anaconda3/envs/eigen/bin/python"

# Small eval/stats artifacts on diskE; large OP model checkpoints on diskF (more free space).
ARTIFACT_ROOT = Path("/diskE/jiaqizhang/rebuttal_completion_artifacts")
OP_ARTIFACT_ROOT = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts")
LEGACY_ROOT = Path(__file__).resolve().parents[1] / "rebuttal_legacy_reproduction"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EIGEN_ROOT = PROJECT_ROOT / "EigenAttn"

MODEL = "/diskD/jiaqizhang/Llama-3"
WIKITEXT = "/home/bai-project/jiaqizhang/wikitext2"
C4_TEXTS = (
    "/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/"
    "llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl"
)
TASK_YAML = "/diskF/jiaqizhang/a1804298693/offline_datasets"
OPT30B_MODEL = "/diskE/jiaqizhang/opt-30b"

PPL_FLOAT_TOL = 1e-5
ACC_FLOAT_TOL = 1e-9
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42


def here() -> Path:
    return Path(__file__).resolve().parent


def legacy_artifacts() -> Path:
    return Path("/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def ensure_artifact_root() -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_ROOT


def ensure_op_artifact_root() -> Path:
    OP_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return OP_ARTIFACT_ROOT


def artifact_dir(
    name: str,
    *,
    link_under: Optional[Path] = None,
    root: Optional[Path] = None,
) -> Path:
    base = Path(root) if root is not None else ensure_artifact_root()
    base.mkdir(parents=True, exist_ok=True)
    target = base / name
    target.mkdir(parents=True, exist_ok=True)
    if link_under is not None:
        link_under = Path(link_under)
        link_under.parent.mkdir(parents=True, exist_ok=True)
        want = os.path.realpath(target)
        if link_under.is_symlink() or link_under.exists():
            if os.path.realpath(link_under) != want:
                if link_under.is_symlink() or link_under.is_file():
                    link_under.unlink()
                elif link_under.is_dir() and not any(link_under.iterdir()):
                    link_under.rmdir()
                else:
                    # Non-empty real dir: leave it, but still return canonical target.
                    return target
                link_under.symlink_to(target)
        else:
            link_under.symlink_to(target)
    return target


def op_artifact_dir(name: str, *, link_under: Optional[Path] = None) -> Path:
    """Large operating-point checkpoints (uniform/FDP final_model)."""
    return artifact_dir(name, link_under=link_under, root=ensure_op_artifact_root())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
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


def file_hash_or_empty(path: Path) -> str:
    p = Path(path)
    return sha256_file(p) if p.is_file() else ""


def stop_requested(root: Optional[Path] = None) -> bool:
    d = root or here()
    return (d / "STOP_REQUESTED").exists()


def gate_payload(
    *,
    name: str,
    passed: bool,
    upstream_hashes: Dict[str, str],
    runner_hash: str,
    protocol_hash: str,
    evaluation_hash: str = "",
    summary: str = "",
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
    atomic_write_json(path, payload)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load NDJSON or pretty-printed concatenated JSON objects."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    rows: List[Dict[str, Any]] = []
    dec = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = dec.raw_decode(text, idx)
        except json.JSONDecodeError:
            # fallback: try single line
            nl = text.find("\n", idx)
            if nl < 0:
                break
            idx = nl + 1
            continue
        idx = end
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def load_zs_samples_dir(sample_root: Path, task: str) -> List[Dict[str, Any]]:
    """Load per-sample JSON under zero_shot_samples/<task>/*.json (preferred)."""
    root = Path(sample_root)
    cands = [
        root / "zero_shot_samples" / task,
        root / task,
    ]
    rows: List[Dict[str, Any]] = []
    for d in cands:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.endswith(".DONE.json"):
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        if rows:
            return rows
    # jsonl fallback (pretty or NDJSON)
    for p in [
        root / "zero_shot_samples" / f"{task}.jsonl",
        root / f"{task}.jsonl",
    ]:
        if p.exists():
            return [r for r in load_jsonl(p) if isinstance(r, dict)]
    return rows


def protocol_hash() -> str:
    return file_hash_or_empty(here() / "FIXED_FDP_SCHEDULE_PROTOCOL.json")


def evaluation_hash() -> str:
    p = LEGACY_ROOT / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"
    return file_hash_or_empty(p)
