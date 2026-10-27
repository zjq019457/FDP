#!/usr/bin/env python3
"""Search for Qwen2.5-7B base; document download path if missing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import atomic_write_json, now_iso  # noqa: E402

SEARCH_ROOTS = [
    Path("/diskD/jiaqizhang"),
    Path("/diskE/jiaqizhang"),
    Path("/diskF/jiaqizhang"),
    ROOT / "external_assets",
]


def main() -> None:
    hits = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*Qwen*"):
            if p.is_dir() and any(p.glob("config.json")):
                hits.append(str(p))
            if len(hits) > 10:
                break
    blocked = len(hits) == 0
    manifest = {
        "generated_at": now_iso(),
        "model_required": "Qwen2.5-7B base (NOT Instruct)",
        "local_hits": hits,
        "blocked": blocked,
        "download_target": str(ROOT / "external_assets" / "Qwen2.5-7B"),
        "hf_repo": "Qwen/Qwen2.5-7B",
    }
    atomic_write_json(HERE / "QWEN_MODEL_LOCK.json", manifest)
    if blocked:
        (HERE / "QWEN_CLAIM_GUARD.md").write_text(
            "# Qwen Third Model — Manual Download Required\n\n"
            "**Official asset**: `Qwen/Qwen2.5-7B` from Hugging Face (base, not Instruct)\n\n"
            "**Windows local path**: download official snapshot locally\n\n"
            "**Server target**: `~/jiaqizhang/EigenAttn/external_assets/Qwen2.5-7B`\n\n"
            "Transfer via interactive scp/rsync after hash verification.\n"
            "Do NOT write HF tokens or passwords to scripts or logs.\n", encoding="utf-8"
        )
    print(json.dumps({"ok": True, "blocked": blocked, "hits": len(hits)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
