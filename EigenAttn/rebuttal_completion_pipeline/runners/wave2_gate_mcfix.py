#!/usr/bin/env python3
"""Gate: assert all MCFIX cleanups completed before WAVE2 GPU work."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_write_json, now_iso  # noqa: E402
from materialize_repair_jobs import CONTROLS  # noqa: E402


def main() -> None:
    missing = []
    for c in CONTROLS:
        done = HERE / "job_outputs" / f"MCFIX_{c}_cleanup" / "DONE.json"
        if not done.exists():
            missing.append(str(done))
    out = {
        "generated_at": now_iso(),
        "controls": CONTROLS,
        "missing_cleanups": missing,
        "ok": len(missing) == 0,
    }
    art = HERE / "artifacts" / "wave2"
    art.mkdir(parents=True, exist_ok=True)
    atomic_write_json(art / "WAVE2_GATE_MCFIX.json", out)
    if missing:
        raise SystemExit(f"MCFIX cleanups incomplete: {missing}")
    print(json.dumps({"ok": True, "n_controls": len(CONTROLS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
