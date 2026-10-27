#!/usr/bin/env python3
"""Component ablation WAVE2: run confirmatory Wiki PPL for Full FDP vs documented variants.

If adaptive rebuild assets for minus-* are unavailable, reuse moderate FDP as Full
and emit an executable sensitivity grid protocol + R11 lambda reference (no fake numbers).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import PYTHON_BIN, atomic_write_json, now_iso  # noqa: E402

FDP = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/fdp/final_model")
UNI = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/uniform/final_model")
TOK = "/diskD/jiaqizhang/Llama-3"
PPL = ROOT / "rebuttal_legacy_reproduction" / "runners" / "ppl_instrumented.py"


def _run_ppl(method: str, model: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN, str(PPL),
        "--dataset", "wikitext2",
        "--model", str(model),
        "--tok", TOK,
        "--out_dir", str(out_dir),
        "--dtype", "bf16",
        "--method", method,
        "--wikitext2_dir", "/home/bai-project/jiaqizhang/wikitext2",
    ]
    log = out_dir / "run.log"
    with log.open("w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    agg = out_dir / "aggregate.json"
    ppl = None
    if agg.exists():
        ppl = json.loads(agg.read_text()).get("ppl")
    return {"method": method, "returncode": rc.returncode, "ppl": ppl, "log": str(log)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {"generated_at": now_iso(), "runs": [], "eval_executed": False}
    # Confirmatory anchors: Full FDP + Uniform matched at moderate
    for method, model in [("fdp_full", FDP), ("uniform_matched_anchor", UNI)]:
        if not (model / "low_rank_config.json").exists() and not (model / "config.json").exists():
            results["runs"].append({"method": method, "skipped": True, "reason": "missing_model"})
            continue
        r = _run_ppl(method, model, out / method)
        results["runs"].append(r)
        if r["returncode"] == 0:
            results["eval_executed"] = True

    # Document minus-* as pending rebuild (need adaptive basis re-run with ablated objective)
    results["minus_variants_status"] = {
        "variants": ["minus_Z", "minus_S", "minus_V", "minus_A", "minus_Entropy"],
        "status": "pending_adaptive_rebuild",
        "note": (
            "Minus-* require re-running adaptive basis with component disabled then compress "
            "at matched KV bytes. WAVE2 executed Full FDP + Uniform confirmatory PPL anchors."
        ),
    }
    # Attach R11 reference if present
    r11 = ROOT / "rebuttal_legacy_reproduction" / "R11_ROBUSTNESS.md"
    if r11.exists():
        (out / "LAMBDA_SENSITIVITY_FROM_R11.md").write_text(r11.read_text(encoding="utf-8"), encoding="utf-8")

    atomic_write_json(out / "COMPONENT_ABLATION_WAVE2.json", results)
    art = HERE / "artifacts" / "robustness"
    art.mkdir(parents=True, exist_ok=True)
    atomic_write_json(art / "COMPONENT_ABLATION_WAVE2.json", results)
    if not results["eval_executed"]:
        raise SystemExit("No confirmatory PPL runs succeeded")
    print(json.dumps({"ok": True, "runs": len(results["runs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
