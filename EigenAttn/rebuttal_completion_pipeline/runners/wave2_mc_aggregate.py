#!/usr/bin/env python3
"""Aggregate real MCFIX matched-control metrics into a single table."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_write_json, now_iso  # noqa: E402
from materialize_repair_jobs import CONTROLS  # noqa: E402

ART = Path("/diskE/jiaqizhang/rebuttal_completion_artifacts/matched_controls")


def _read_ppl(ctrl: str, dataset: str):
    p = ART / ctrl / f"{dataset}_ppl" / "aggregate.json"
    if not p.exists():
        # alternate naming
        alts = list((ART / ctrl).glob(f"*{dataset}*aggregate*.json")) if (ART / ctrl).exists() else []
        if not alts:
            return None
        p = alts[0]
    try:
        return json.loads(p.read_text()).get("ppl")
    except Exception:
        return None


def _read_zs(ctrl: str, bundle: str):
    d = ART / ctrl / bundle
    if not d.exists():
        return None
    # prefer summary json
    for name in ("summary.json", "aggregate.json", "ZS_SUMMARY.json"):
        p = d / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    # collect per-task accuracies from sample dirs
    out = {}
    for sj in d.rglob("*summary*.json"):
        try:
            obj = json.loads(sj.read_text())
            out[sj.parent.name] = obj
        except Exception:
            continue
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for ctrl in CONTROLS:
        row = {
            "control": ctrl,
            "wikitext2_ppl": _read_ppl(ctrl, "wikitext2"),
            "c4_ppl": _read_ppl(ctrl, "c4"),
            "hs_arc": _read_zs(ctrl, "hs_arc"),
            "wg_piqa": _read_zs(ctrl, "wg_piqa"),
            "cleanup_done": (HERE / "job_outputs" / f"MCFIX_{ctrl}_cleanup" / "DONE.json").exists(),
        }
        rows.append(row)

    atomic_write_json(out_root / "MCFIX_AGGREGATE.json", {
        "generated_at": now_iso(),
        "rows": rows,
        "note": "Real MCFIX eval artifacts; not schedule-only.",
    })
    # flat CSV for PPL
    csv_path = out_root / "MCFIX_PPL.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["control", "wikitext2_ppl", "c4_ppl", "cleanup_done"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ["control", "wikitext2_ppl", "c4_ppl", "cleanup_done"]})
    print(json.dumps({"ok": True, "n": len(rows), "csv": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
