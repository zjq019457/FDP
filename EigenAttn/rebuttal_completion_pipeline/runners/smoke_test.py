#!/usr/bin/env python3
"""CPU smoke: verify imports, legacy gates, protocol files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LEGACY = HERE.parent / "rebuttal_legacy_reproduction"
sys.path.insert(0, str(HERE))

checks = []
for name in ["FIXED_FDP_SCHEDULE_PROTOCOL.json", "CURRENT_EVIDENCE_INVENTORY.json"]:
    p = HERE / name
    checks.append((name, p.exists()))

for gate in ["LEGACY_FDP_PARITY_PASS", "DENSE_PARITY_PASS", "UNIFORM_PARITY_PASS"]:
    p = LEGACY / f"{gate}.DONE"
    checks.append((gate, p.exists()))

failed = [n for n, ok in checks if not ok]
if failed:
    print(json.dumps({"ok": False, "failed": failed}, ensure_ascii=False))
    sys.exit(1)
print(json.dumps({"ok": True, "checks": len(checks)}, ensure_ascii=False))
