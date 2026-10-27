#!/usr/bin/env python3
"""Singular-value / fragility correlation from existing basis reports."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_write_json, load_json, now_iso  # noqa: E402


def main() -> None:
    rows = []
    step1 = Path("/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts/r3_fgeb045/step1_llama3_8b_uniform_eb045.json")
    basis_report = Path("/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json")
    if step1.exists():
        s1 = load_json(step1)
        per = s1.get("per_layer", {})
        for layer_id, metrics in per.items():
            if isinstance(metrics, dict):
                rows.append({
                    "layer": layer_id,
                    "Z_rel_mse": metrics.get("Z_rel_mse"),
                    "score_rel_mse_causal": metrics.get("score_rel_mse_causal"),
                    "value_only_Z_rel_mse": metrics.get("value_only_Z_rel_mse"),
                    "rank_kq": metrics.get("rank_kq"),
                    "rank_v": metrics.get("rank_v"),
                })
    out = HERE / "artifacts" / "spectrum"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "LAYERWISE_SPECTRUM.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            w.writeheader()
            w.writerows(rows)
    (HERE / "SPECTRUM_ANALYSIS.md").write_text(
        f"# Spectrum Analysis\n\n- layers: {len(rows)}\n- source: step1 fragility metrics\n", encoding="utf-8"
    )
    atomic_write_json(out / "SPECTRUM_MANIFEST.json", {"generated_at": now_iso(), "layers": len(rows)})
    print(json.dumps({"ok": True, "layers": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
