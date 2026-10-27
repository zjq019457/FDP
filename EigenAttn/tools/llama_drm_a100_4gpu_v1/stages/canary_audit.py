#!/usr/bin/env python3
"""Stage-1 canary audit for LLAMA_DRM_A100_4GPU_V1.

Checks (prediction-level) the formal canary invariants once Stage-1 jobs finish.
HARD_BLOCK on Original determinism / no-op parity / Random wiring failures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C


def _load_preds(run_dir: Path, task: str) -> List[Dict[str, Any]]:
    path = run_dir / "longbench_predictions" / f"{task}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _pred_key(row: Dict[str, Any]) -> str:
    for key in ("sample_id", "id", "_id", "idx"):
        if key in row:
            return str(row[key])
    return json.dumps(row.get("answers", row.get("prediction", "")), sort_keys=True)


def _align(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    mb = {_pred_key(r): r for r in b}
    out = []
    for ra in a:
        rb = mb.get(_pred_key(ra))
        if rb is not None:
            out.append((ra, rb))
    return out


def _prediction_text(row: Dict[str, Any]) -> str:
    for key in ("prediction", "pred", "output", "generated"):
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def audit_canary(result_dir: Path) -> Dict[str, Any]:
    root = result_dir / "runs" / "stage1_canary"
    checks: List[Dict[str, Any]] = []
    hard_blocks: List[str] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            hard_blocks.append(f"{name}: {detail}")

    # Original run1 vs run2
    for task in ("qasper", "gov_report"):
        r1 = _load_preds(root / "sliminfer_original__run1" / task, task)
        r2 = _load_preds(root / "sliminfer_original__run2" / task, task)
        if not r1 or not r2:
            add(f"original_determinism_{task}_present", False, f"rows {len(r1)}/{len(r2)}")
            continue
        pairs = _align(r1, r2)
        same = all(_prediction_text(a) == _prediction_text(b) for a, b in pairs)
        add(f"original_determinism_{task}", same and len(pairs) == len(r1),
            f"aligned={len(pairs)} identical={same}")

    # no-op vs Original run1
    for task in ("qasper", "gov_report"):
        orig = _load_preds(root / "sliminfer_original__run1" / task, task)
        noop = _load_preds(root / "sliminfer_original__noop" / task, task)
        if not orig or not noop:
            add(f"noop_parity_{task}_present", False, f"rows {len(orig)}/{len(noop)}")
            continue
        pairs = _align(orig, noop)
        same = all(_prediction_text(a) == _prediction_text(b) for a, b in pairs)
        add(f"noop_parity_{task}", same and len(pairs) == len(orig),
            f"aligned={len(pairs)} identical={same}")

    # Random seed0 run1 vs run2 (repeatable)
    for task in ("qasper",):
        a = _load_preds(root / "random_add__seed0__run1" / task, task)
        b = _load_preds(root / "random_add__seed0__run2" / task, task)
        if a and b:
            pairs = _align(a, b)
            same = all(_prediction_text(x) == _prediction_text(y) for x, y in pairs)
            add("random_seed0_repeatable", same and len(pairs) == len(a),
                f"aligned={len(pairs)} identical={same}")
        else:
            add("random_seed0_repeatable_present", False, f"rows {len(a)}/{len(b)}")

    # Random seed1 vs seed0 should differ when residual pool is non-empty
    for task in ("qasper",):
        s0 = _load_preds(root / "random_add__seed0__run1" / task, task)
        s1 = _load_preds(root / "random_add__seed1__run1" / task, task)
        larger = _load_preds(root / "sliminfer_larger__run1" / task, task)
        if s0 and s1:
            pairs = _align(s0, s1)
            differ = any(_prediction_text(x) != _prediction_text(y) for x, y in pairs)
            add("random_seed1_differs_seed0", differ or len(pairs) == 0,
                f"aligned={len(pairs)} differ={differ}")
        if s0 and larger:
            pairs = _align(s0, larger)
            differ = any(_prediction_text(x) != _prediction_text(y) for x, y in pairs)
            add("random_not_identical_to_larger", differ or len(pairs) == 0,
                f"aligned={len(pairs)} differ={differ}")

    # DRM-Add / DRM-Swap presence
    for arm, task in (("drm_add__run1", "qasper"), ("drm_swap__run1", "qasper"),
                      ("drm_add__run1", "gov_report")):
        rows = _load_preds(root / arm / task, task)
        add(f"presence_{arm}_{task}", len(rows) > 0, f"rows={len(rows)}")

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "audited_at": C.now_iso(),
        "result_dir": str(result_dir),
        "checks": checks,
        "passed": sum(1 for c in checks if c["ok"]),
        "failed": sum(1 for c in checks if not c["ok"]),
        "hard_block": bool(hard_blocks),
        "hard_block_reasons": hard_blocks,
        "decision": "HARD_BLOCK" if hard_blocks else "PASS",
    }
    out = result_dir / "status" / "LLAMA_DRM_CANARY_REPORT.json"
    C.atomic_write_json(out, payload)
    md = result_dir / "exports" / "LLAMA_DRM_CANARY_REPORT.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# LLAMA_DRM_CANARY_REPORT",
        f"",
        f"- decision: **{payload['decision']}**",
        f"- passed: {payload['passed']} / failed: {payload['failed']}",
        f"",
    ]
    for c in checks:
        mark = "PASS" if c["ok"] else "FAIL"
        lines.append(f"- [{mark}] `{c['name']}` {c['detail']}")
    C.atomic_write_text(md, "\n".join(lines) + "\n")
    return payload


if __name__ == "__main__":
    result = Path(sys.argv[1]) if len(sys.argv) > 1 else C.find_resumable_result_dir()
    if result is None:
        raise SystemExit("no result dir")
    payload = audit_canary(result)
    print(json.dumps(payload, indent=2))
    raise SystemExit(2 if payload["hard_block"] else 0)
