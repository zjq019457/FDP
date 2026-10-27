#!/usr/bin/env python3
"""OPT-30B fixed-schedule uncertainty: protocol + artifact inventory."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import OPT30B_MODEL, atomic_write_json, now_iso  # noqa: E402


def main() -> None:
    opt_basis = Path("/diskD/jiaqizhang/opt30b_eigen_basis_wikitext2_fp16")
    reports = list(opt_basis.glob("*report.json")) if opt_basis.is_dir() else []
    model_exists = Path(OPT30B_MODEL).is_dir()
    out = HERE / "artifacts" / "opt30b"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "OPT30B_PROTOCOL.json", {
        "generated_at": now_iso(),
        "model": OPT30B_MODEL,
        "model_exists": model_exists,
        "basis_reports": len(reports),
        "fixed_schedule": "aggressive matched point: Uniform + single FDP schedule",
        "status": "pending_gpu" if model_exists else "blocked_no_model",
    })
    (HERE / "OPT30B_CLAIM_GUARD.md").write_text(
        f"# OPT-30B Claim Guard\n\n- Model exists: {model_exists}\n"
        "- No Ours-v2/v3/v4 cherry-picking; single fixed schedule only.\n", encoding="utf-8"
    )
    if not model_exists:
        print(json.dumps({"ok": True, "blocked": True}, ensure_ascii=False))
    else:
        print(json.dumps({"ok": True, "reports": len(reports), "gpu_pending": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
