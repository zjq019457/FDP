#!/usr/bin/env python3
"""OPT-30B single-point WikiText PPL via sanity_AB_opt30b (real eval)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
EIGEN = ROOT / "EigenAttn"
sys.path.insert(0, str(HERE))
from common import OPT30B_MODEL, PYTHON_BIN, atomic_write_json, now_iso  # noqa: E402

CMP_CANDIDATES = [
    Path("/diskE/jiaqizhang/FGeb026fpv6a/final_model"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cmp", default="")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmp = Path(args.cmp) if args.cmp else None
    if cmp is None:
        for c in CMP_CANDIDATES:
            if (c / "low_rank_config.json").exists():
                cmp = c
                break
    status = {
        "generated_at": now_iso(),
        "model": OPT30B_MODEL,
        "model_exists": Path(OPT30B_MODEL).is_dir(),
        "cmp": str(cmp) if cmp else None,
        "eval_executed": False,
    }
    if not Path(OPT30B_MODEL).is_dir():
        status["blocked"] = "no_opt30b_weights"
        atomic_write_json(out / "OPT30B_WAVE2_STATUS.json", status)
        print(json.dumps({"ok": True, "blocked": True, **status}, ensure_ascii=False))
        return
    if cmp is None or not (cmp / "low_rank_config.json").exists():
        status["blocked"] = "no_compressed_ckpt"
        atomic_write_json(out / "OPT30B_WAVE2_STATUS.json", status)
        raise SystemExit("No OPT compressed final_model with low_rank_config.json")

    sanity = EIGEN / "sanity_AB_opt30b.py"
    cmd = [
        PYTHON_BIN, str(sanity),
        "--cmp_only",
        "--cmp", str(cmp),
        "--low_json", str(cmp / "low_rank_config.json"),
        "--tok", OPT30B_MODEL,
        "--dtype", "fp16",
        "--cmp_device", "cuda:0",
        "--wikitext2",
    ]
    log = out / "sanity_AB_opt30b.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(EIGEN) + os.pathsep + env.get("PYTHONPATH", "")
    env["WIKITEXT2_DATA_DIR"] = "/home/bai-project/jiaqizhang/wikitext2"
    with log.open("w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=str(EIGEN), env=env, stdout=lf, stderr=subprocess.STDOUT)
    status["returncode"] = rc.returncode
    status["cmd"] = cmd
    status["log"] = str(log)
    status["eval_executed"] = rc.returncode == 0
    status["schedule_note"] = (
        "Reused FGeb026fpv6a as single fixed-schedule confirmatory point; "
        "not Ours-v2/v3/v4 cherry-pick sweep."
    )
    atomic_write_json(out / "OPT30B_WAVE2_STATUS.json", status)
    art = HERE / "artifacts" / "opt30b"
    art.mkdir(parents=True, exist_ok=True)
    atomic_write_json(art / "OPT30B_WAVE2_STATUS.json", status)
    if rc.returncode != 0:
        raise SystemExit(f"sanity_AB_opt30b failed rc={rc.returncode}; see {log}")
    print(json.dumps({"ok": True, "cmp": str(cmp)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
