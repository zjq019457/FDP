#!/usr/bin/env python3
"""External baseline manifest: xKV, ReCalKV, PyramidKV/H2O search."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import atomic_write_json, now_iso  # noqa: E402

CANDIDATES = [
    ("xKV", ROOT / "slim"),
    ("ReCalKV", ROOT / "slim" / "results"),
    ("PyramidKV", None),
    ("H2O", None),
    ("RescueKV", ROOT / "slim" / "results"),
]


def main() -> None:
    found = []
    for name, path in CANDIDATES:
        if path and path.exists():
            hits = list(path.rglob(f"*{name.lower()}*"))[:5] if path.is_dir() else [path]
            found.append({"name": name, "path": str(path), "hits": [str(h) for h in hits]})
    manifest = {"generated_at": now_iso(), "candidates": found, "official_impl_locked": False}
    atomic_write_json(HERE / "EXTERNAL_BASELINE_MANIFEST.json", manifest)
    atomic_write_json(HERE / "EXTERNAL_BASELINE_PROTOCOL.json", {
        "policy": "official implementations only; same Llama3 checkpoint and matched KV bytes",
        "blocked_methods": [n for n, p in CANDIDATES if not p or not Path(p).exists()],
    })
    if not any(c["name"] in ("xKV", "ReCalKV") for c in found):
        (HERE / "EXTERNAL_BASELINE_BLOCKED.md").write_text(
            "# External Baseline Blocked\n\nNo official xKV/ReCalKV implementation found with Llama3 compatibility.\n"
            "- RescueKV results exist in slim/results but require separate validation.\n", encoding="utf-8"
        )
    print(json.dumps({"ok": True, "found": len(found)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
