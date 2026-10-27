#!/usr/bin/env python3
"""Audit/enrich R8 system metrics; document missing fields."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LEGACY = HERE.parent / "rebuttal_legacy_reproduction"
sys.path.insert(0, str(HERE))
from common import atomic_write_json, load_json, now_iso  # noqa: E402


def main() -> None:
    r8_json = LEGACY / "artifacts" / "r8_system_metrics" / "R8_SYSTEM_METRICS.json"
    if not r8_json.exists():
        r8_json = Path("/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts/r8_system_metrics/R8_SYSTEM_METRICS.json")
    rows = []
    if r8_json.exists():
        d = load_json(r8_json)
        for r in d.get("rows", []):
            rows.append({
                **r,
                "median_prefill_ms": r.get("prefill_median_ms"),
                "median_decode_ms": r.get("decode_median_ms"),
                "tokens_per_sec": r.get("prefill_tokens_per_sec"),
                "peak_allocated_mb": r.get("peak_memory_mb"),
                "p95_prefill_ms": None,
                "p95_decode_ms": None,
                "peak_reserved_mb": None,
                "ttft_ms": r.get("prefill_median_ms"),
                "attn_backend": "eager",
                "flash_sdp": False,
            })
    out = HERE / "artifacts" / "system_metrics"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "SYSTEM_METRICS_FINAL.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            w.writeheader()
            w.writerows(rows)
    guard = [
        "# Deployment Claim Guard",
        "",
        f"- generated_at: `{now_iso()}`",
        "- Metrics collected under **eager** attention implementation.",
        "- Do NOT claim FlashAttention or vLLM compatibility without separate validation.",
        "- Missing from legacy R8: p95, peak reserved, offline Step1/basis timing.",
        "- Reused R8 prefill/decode median and peak allocated only.",
    ]
    (HERE / "DEPLOYMENT_CLAIM_GUARD.md").write_text("\n".join(guard) + "\n", encoding="utf-8")
    (HERE / "SYSTEM_BACKEND_AUDIT.md").write_text("\n".join(guard) + "\n", encoding="utf-8")
    atomic_write_json(out / "SYSTEM_METRICS_AUDIT.json", {"generated_at": now_iso(), "rows": len(rows), "reused_r8": True})
    print(json.dumps({"ok": True, "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
