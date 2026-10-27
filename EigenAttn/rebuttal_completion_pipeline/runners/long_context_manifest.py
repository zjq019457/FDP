#!/usr/bin/env python3
"""Long context asset manifest from local LongBench/needle data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import atomic_write_json, file_hash_or_empty, now_iso, sha256_file  # noqa: E402

ASSETS = [
    ROOT / "data" / "LongBench",
    Path("/diskF/jiaqizhang/combine/data/LongBench"),
    ROOT / "slim" / "longbench_eval.py",
    ROOT / "slim" / "needle_in_haystack.py",
]


def main() -> None:
    hits = []
    for p in ASSETS:
        if p.exists():
            entry = {"path": str(p), "type": "dir" if p.is_dir() else "file"}
            if p.is_dir():
                jsonls = list(p.glob("*.jsonl"))
                entry["jsonl_count"] = len(jsonls)
                entry["sample_files"] = [f.name for f in jsonls[:10]]
            else:
                entry["sha256"] = sha256_file(p)
            hits.append(entry)

    manifest = {
        "generated_at": now_iso(),
        "hits": hits,
        "blocked": len(hits) == 0,
        "license_note": "LongBench data under original dataset license; verify before publication",
    }
    atomic_write_json(HERE / "LONG_CONTEXT_ASSET_MANIFEST.json", manifest)

    hashes = {}
    lb = ROOT / "data" / "LongBench"
    if lb.is_dir():
        for f in sorted(lb.glob("*.jsonl"))[:20]:
            hashes[f.name] = sha256_file(f)
    atomic_write_json(HERE / "LONG_CONTEXT_DATA_HASHES.json", {"generated_at": now_iso(), "files": hashes})

    audit = [
        "# Long Context License Audit",
        "",
        f"- generated_at: `{now_iso()}`",
        f"- Local LongBench found: `{lb.exists()}`",
        "- RULER official repo: not found locally",
        "- Needle-in-a-Haystack scripts: present in slim/",
        "- Policy: use official local copies only; no fabricated datasets",
    ]
    (HERE / "LONG_CONTEXT_LICENSE_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    if not hits:
        (HERE / "LONG_CONTEXT_DATA_BLOCKED.md").write_text(
            "# Long Context Data Blocked\n\nNo official assets found.\n", encoding="utf-8"
        )
    print(json.dumps({"ok": True, "hits": len(hits)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
