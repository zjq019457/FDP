#!/usr/bin/env python3
"""Aggregate operating point results into CSV/audit."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import artifact_dir, atomic_write_json, load_json, now_iso  # noqa: E402


def main() -> None:
    rows = []
    for point in ("mild", "moderate", "aggressive"):
        op = artifact_dir(f"operating_points/{point}", link_under=HERE / "artifacts" / f"OP_{point}")
        manifest = op / "OPERATING_POINT_MANIFEST.json"
        m = load_json(manifest) if manifest.exists() else {}
        for meth in ("uniform", "fdp"):
            for ds in ("wikitext2", "c4"):
                tag = f"OP_{point}_{meth}"
                p = HERE / "artifacts" / f"{tag}_{ds}_ppl" / "ppl_aggregate.json"
                if not p.exists():
                    p = Path(f"/diskE/jiaqizhang/rebuttal_completion_artifacts/{tag}_{ds}_ppl/ppl_aggregate.json")
                if p.exists():
                    d = load_json(p)
                    rows.append({"point": point, "method": meth, "metric": f"{ds}_ppl", "value": d.get("ppl")})
    out = HERE / "artifacts" / "operating_points"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "OPERATING_POINT_RESULTS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            w.writeheader()
            w.writerows(rows)
    atomic_write_json(out / "OPERATING_POINT_SUMMARY.json", {"generated_at": now_iso(), "rows": rows})
    (HERE / "OPERATING_POINT_AUDIT.md").write_text(
        f"# Operating Point Audit\n\n- generated_at: `{now_iso()}`\n- rows: {len(rows)}\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
