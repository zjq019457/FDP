#!/usr/bin/env python3
"""Gate: OP_{point}_zs passes when all method×bundle aggregates exist (or legacy mild reuse)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import ARTIFACT_ROOT, atomic_write_json, now_iso  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", required=True, choices=["mild", "moderate", "aggressive"])
    args = ap.parse_args()
    missing = []
    found = []
    for meth in ("uniform", "fdp"):
        for bundle in ("hs_arc", "wg_piqa"):
            candidates = [
                ARTIFACT_ROOT / f"OP_{args.point}_{meth}_{bundle}" / "zero_shot_aggregate.json",
                ARTIFACT_ROOT / f"OP_{args.point}_{meth}_zs_{bundle}" / "zero_shot_aggregate.json",
            ]
            hit = next((p for p in candidates if p.exists()), None)
            if hit is not None:
                found.append(str(hit))
            else:
                missing.append(str(candidates[0]))
    ok = len(missing) == 0
    payload = {
        "ok": ok,
        "point": args.point,
        "found": found,
        "missing": missing,
        "generated_at": now_iso(),
    }
    atomic_write_json(HERE / "job_outputs" / f"OP_{args.point}_zs" / "ZS_GATE.json", payload)
    if not ok:
        raise SystemExit(f"ZS gate incomplete for {args.point}: missing {missing}")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
