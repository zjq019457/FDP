#!/usr/bin/env python3
"""External baseline WAVE2: search official xKV/ReCalKV; document blocked if absent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import atomic_write_json, now_iso  # noqa: E402

NEEDLES = ["xKV", "xkv", "ReCalKV", "recalkv", "PyramidKV", "pyramidkv"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hits = []
    search_roots = [ROOT / "slim", ROOT / "external_assets", Path("/diskE/jiaqizhang/MatryoshkaKV-cache")]
    for root in search_roots:
        if not root.exists():
            continue
        for needle in NEEDLES:
            for p in root.rglob(f"*{needle}*"):
                hits.append(str(p))
                if len(hits) > 40:
                    break
    official = [h for h in hits if "rescuekv" not in h.lower()]
    status = {
        "generated_at": now_iso(),
        "hits": hits[:40],
        "official_xkv_recalkv_found": any("xkv" in h.lower() or "recalkv" in h.lower() for h in official),
        "eval_executed": False,
        "policy": "official implementations only; matched Llama-3 + matched KV bytes",
        "blocked_reason": None,
    }
    if not status["official_xkv_recalkv_found"]:
        status["blocked_reason"] = (
            "No official xKV/ReCalKV codebase located under slim/external_assets/MatryoshkaKV-cache. "
            "RescueKV results exist but are a different method and must not be relabeled."
        )
        (HERE / "EXTERNAL_BASELINE_BLOCKED.md").write_text(
            "# External Baseline — WAVE2\n\n"
            f"- Blocked: {status['blocked_reason']}\n"
            "- Action: clone official repos and adapt to Llama-3-8B + matched KV budget, or keep blocked.\n",
            encoding="utf-8",
        )
    atomic_write_json(out / "EXTERNAL_WAVE2_STATUS.json", status)
    atomic_write_json(HERE / "EXTERNAL_BASELINE_MANIFEST.json", status)
    print(json.dumps({"ok": True, "blocked": status["blocked_reason"] is not None, "hits": len(hits)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
