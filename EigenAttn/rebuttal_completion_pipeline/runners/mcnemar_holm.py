#!/usr/bin/env python3
"""Merge bootstrap outputs; write McNemar/Holm CSVs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_write_json, now_iso  # noqa: E402


def main() -> None:
    stat_dir = HERE / "artifacts" / "statistics"
    for name in ("MCNEMAR", "HOLM_CORRECTION", "TASK_PAIRED_BOOTSTRAP"):
        src = stat_dir / f"{name}.json"
        if src.exists():
            d = json.loads(src.read_text())
            rows = d.get("rows", [])
            with (stat_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
                    w.writeheader()
                    w.writerows(rows)
    atomic_write_json(stat_dir / "MCNEMAR_HOLM_MANIFEST.json", {"generated_at": now_iso(), "ok": True})
    print(json.dumps({"ok": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
