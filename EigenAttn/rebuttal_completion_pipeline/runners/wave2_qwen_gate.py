#!/usr/bin/env python3
"""Qwen2.5-7B WAVE2 gate: weights exist, EigenAttn modeling does not — honest status."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
EIGEN = ROOT / "EigenAttn"
sys.path.insert(0, str(HERE))
from common import atomic_write_json, now_iso  # noqa: E402

QWEN = Path("/diskE/jiaqizhang/Qwen2.5-7B")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    main_py = (EIGEN / "main_eigen_attn.py").read_text(encoding="utf-8", errors="ignore")
    has_qwen_choice = "qwen2" in main_py and "net_choices" in main_py
    # crude: look for modelling_qwen
    qwen_model_files = list((EIGEN / "models").glob("*qwen*"))
    load_ok = False
    load_err = None
    if QWEN.is_dir() and (QWEN / "config.json").exists():
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(str(QWEN), local_files_only=True)
            load_ok = True
        except Exception as e:
            load_err = repr(e)

    status = {
        "generated_at": now_iso(),
        "weights_path": str(QWEN),
        "weights_present": QWEN.is_dir(),
        "hf_config_loadable": load_ok,
        "hf_config_error": load_err,
        "eigenattn_qwen_model_files": [str(p) for p in qwen_model_files],
        "main_eigen_attn_mentions_qwen2": "qwen2" in main_py,
        "compress_eval_executed": False,
        "blocked_reason": None if (qwen_model_files and has_qwen_choice) else (
            "EigenAttn lacks Qwen2 modelling + net_choices wiring; "
            "cannot honestly run FDP compress/eval yet. Weights are local."
        ),
        "next_engineering": [
            "Add modelling_qwen2_eigen_attn.py (or equivalent)",
            "Extend LMClass + main_eigen_attn net_choices",
            "Build eigen basis on wikitext2/c4",
            "compress_only to diskF then ppl/zs",
        ],
    }
    atomic_write_json(out / "QWEN_WAVE2_GATE.json", status)
    atomic_write_json(HERE / "QWEN_MODEL_LOCK.json", {
        "generated_at": now_iso(),
        "model_required": "Qwen2.5-7B base (NOT Instruct)",
        "local_hits": [str(QWEN)] if QWEN.is_dir() else [],
        "blocked": status["blocked_reason"] is not None,
        "wave2": status,
    })
    (HERE / "QWEN_CLAIM_GUARD.md").write_text(
        "# Qwen Third Model — WAVE2 Status\n\n"
        f"- Weights present at `{QWEN}`: **{QWEN.is_dir()}**\n"
        f"- EigenAttn compress/eval executed: **False**\n"
        f"- Blocked: {status['blocked_reason']}\n"
        "- Do not claim Qwen FDP results until compress+eval complete.\n",
        encoding="utf-8",
    )
    # Gate job succeeds even when blocked — records honest evidence
    print(json.dumps({"ok": True, "blocked": status["blocked_reason"] is not None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
