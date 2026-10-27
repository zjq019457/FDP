#!/usr/bin/env python3
"""Aggregate matched control schedules."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import artifact_dir, atomic_write_json, load_json, now_iso  # noqa: E402

CONTROLS = [
    "uniform_matched", "random_seed0", "random_seed1", "random_seed2",
    "spectral_tail", "middle_layer", "depth_front", "depth_back", "reconstruction_slope",
]


def main() -> None:
    rows, budget_rows = [], []
    for ctrl in CONTROLS:
        p = artifact_dir(f"matched_controls/{ctrl}", link_under=HERE / "artifacts" / f"MC_{ctrl}")
        sched = p / "MATCHED_CONTROL_SCHEDULE.json"
        if sched.exists():
            d = load_json(sched)
            rows.append({"control": ctrl, **{k: d.get(k) for k in (
                "target_kv_bytes", "realized_kv_bytes_est", "abs_byte_gap", "rel_gap")}})
            budget_rows.append(d)
    out = HERE / "artifacts" / "matched_controls"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "MATCHED_CONTROL_BUDGET_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            w.writeheader()
            w.writerows(rows)
    atomic_write_json(out / "MATCHED_CONTROL_SUMMARY.json", {"generated_at": now_iso(), "controls": rows})
    print(json.dumps({"ok": True, "controls": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
