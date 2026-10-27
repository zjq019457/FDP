#!/usr/bin/env python3
"""Long context eval placeholder: records protocol and blocked/runnable status."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_write_json, now_iso  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["2k", "8k"], required=True)
    args = ap.parse_args()
    manifest = HERE / "LONG_CONTEXT_ASSET_MANIFEST.json"
    blocked = not manifest.exists() or json.loads(manifest.read_text()).get("blocked", True)
    out = HERE / "artifacts" / "long_context"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "profile": args.profile,
        "generated_at": now_iso(),
        "status": "blocked_pending_full_runner" if blocked else "runnable",
        "calibration": {"2k": "512x2048", "8k": "16x8192"}[args.profile] if False else (
            {"nsamples": 512, "seqlen": 2048} if args.profile == "2k" else {"nsamples": 16, "seqlen": 8192}
        ),
        "methods": ["dense", "uniform", f"fdp_{args.profile}", "spectral", "middle"],
        "note": "Full 8K eval runner binds to slim/longbench_eval.py; manifest confirms local data",
    }
    atomic_write_json(out / f"LONG_CONTEXT_{args.profile.upper()}_PROTOCOL.json", result)
    if blocked:
        print(json.dumps({"ok": True, "blocked": True}, ensure_ascii=False))
    else:
        # Mark protocol frozen; GPU eval deferred to next iteration
        atomic_write_json(out / f"LONG_CONTEXT_{args.profile.upper()}_STATUS.json", {
            **result, "eval_executed": False, "reason": "protocol_frozen_first_pass",
        })
        print(json.dumps({"ok": True, "profile": args.profile, "protocol_only": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
