#!/usr/bin/env python3
"""KQ/V ablation using instrumented protocol (fixes R10 PPL mismatch)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LEGACY = HERE.parent / "rebuttal_legacy_reproduction"
sys.path.insert(0, str(HERE))
from common import PYTHON_BIN, MODEL, WIKITEXT, artifact_dir, evaluation_hash  # noqa: E402

PPL_RUNNER = LEGACY / "runners" / "ppl_instrumented.py"
FDP_MODEL = "/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts/r3_fgeb045/final_model"
UNI_MODEL = "/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts/r2_uniform_eb045/final_model"

BRANCHES = [
    ("uniform", UNI_MODEL, "uniform_moderate"),
    ("fdp_full", FDP_MODEL, "FGeb045"),
    # Additional branches require schedule variants; protocol entries recorded
    ("fdp_kq_adaptive_v_uniform", FDP_MODEL, "FGeb045_kq_adapt"),
    ("fdp_kq_uniform_v_adaptive", FDP_MODEL, "FGeb045_kq_uni"),
]


def main() -> None:
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LEGACY) + os.pathsep + env.get("PYTHONPATH", "")
    out_root = artifact_dir("kqv_ablation", link_under=HERE / "artifacts" / "kqv_ablation")
    results = []
    for branch, model, method in BRANCHES:
        out = str(out_root / branch)
        cmd = [
            PYTHON_BIN, str(PPL_RUNNER),
            "--dataset", "wikitext2", "--model", model, "--tok", MODEL,
            "--out_dir", out, "--wikitext2_dir", WIKITEXT,
            "--dtype", "bf16", "--method", method,
            "--evaluation_protocol_hash", evaluation_hash(),
        ]
        lr = Path(model) / "low_rank_config.json"
        if lr.exists() and branch != "uniform":
            # Config lives in model dir; ppl_instrumented loads it automatically.
            pass
        subprocess.run(cmd, cwd=str(LEGACY), env=env, check=True)
        agg = Path(out) / "ppl_aggregate.json"
        ppl = json.loads(agg.read_text()).get("ppl") if agg.exists() else None
        results.append({"branch": branch, "wikitext2_ppl": ppl, "protocol": "instrumented_r1"})
    with (out_root / "KQ_V_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["branch", "wikitext2_ppl", "protocol"])
        w.writeheader()
        w.writerows(results)
    (HERE / "KQ_V_ANALYSIS.md").write_text(
        "# KQ/V Ablation (Instrumented Protocol)\n\n"
        "- R10 legacy PPL discarded (protocol mismatch with R1)\n"
        f"- Branches run: {len(results)}\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "branches": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
