#!/usr/bin/env python3
"""Component ablation protocol: minus Z/S/V/A/Entropy variants."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_write_json, now_iso  # noqa: E402

VARIANTS = ["full", "minus_Z", "minus_S", "minus_V", "minus_A", "minus_Entropy"]
CAL_DOMAINS = ["wikitext", "c4"]
CAL_SIZES = [16, 32, 64]
LAMBDAS = [0.6, 1.2, 2.0]


def main() -> None:
    out = HERE / "artifacts" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "generated_at": now_iso(),
        "component_variants": VARIANTS,
        "calibration_domains": CAL_DOMAINS,
        "cal_sizes": CAL_SIZES,
        "lambdas": LAMBDAS,
        "confirmatory_lambda": 1.2,
        "policy": "Full FDP at lambda=1.2 is confirmatory; others sensitivity only",
        "status": "protocol_frozen",
        "gpu_build_pending": True,
    }
    atomic_write_json(out / "COMPONENT_ABLATION_PROTOCOL.json", protocol)
    (HERE / "ROBUSTNESS_CLAIM_GUARD.md").write_text(
        "# Robustness Claim Guard\n\n"
        "- lambda=2.0, cal=32 best PPL from R11 is **robustness only**, not main config.\n"
        "- Component ablations must match Full FDP actual KV bytes.\n", encoding="utf-8"
    )
    # Reuse R11 results as lambda/cal sensitivity reference
    r11 = HERE.parent / "rebuttal_legacy_reproduction" / "R11_ROBUSTNESS.md"
    if r11.exists():
        (out / "LAMBDA_SENSITIVITY_FROM_R11.md").write_text(r11.read_text(), encoding="utf-8")
    print(json.dumps({"ok": True, "protocol_only": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
