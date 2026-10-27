#!/usr/bin/env python3
"""WAVE2 DAG: real gap-closing jobs that run after MCFIX cleanups finish.

Previous P4/P5 jobs were protocol/search stubs (already DONE). WAVE2 adds new
job IDs so the orchestrator keeps running without re-opening stub DONE markers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import PYTHON_BIN  # noqa: E402
from materialize_repair_jobs import CONTROLS  # noqa: E402
from stage_materialize import make_job  # noqa: E402

MCFIX_CLEANUPS = [f"MCFIX_{c}_cleanup" for c in CONTROLS]
RUN = Path(HERE / "runners")
OUT_ROOT = "/diskF/jiaqizhang/wave2_artifacts"


def build_wave2_jobs():
    jobs = []
    # Sentinel: wait for all MCFIX cleanups (disk free + MC evidence ready)
    jobs.append(make_job(
        "WAVE2_GATE_MCFIX", "WAVE2", "cpu",
        [PYTHON_BIN, str(RUN / "wave2_gate_mcfix.py")],
        depends_on=MCFIX_CLEANUPS,
        meta={"role": "gate"},
    ))

    jobs.append(make_job(
        "WAVE2_MC_AGGREGATE", "WAVE2", "cpu",
        [PYTHON_BIN, str(RUN / "wave2_mc_aggregate.py"), "--out_root", OUT_ROOT],
        depends_on=["WAVE2_GATE_MCFIX"],
        meta={"role": "aggregate"},
    ))

    # LongBench 8K: Dense / Uniform / FDP moderate (1 GPU each, packable)
    for method, model in [
        ("dense", "/diskD/jiaqizhang/Llama-3"),
        ("uniform", "/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/uniform/final_model"),
        ("fdp", "/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points/moderate/fdp/final_model"),
    ]:
        jobs.append(make_job(
            f"WAVE2_LB_{method}", "WAVE2", "gpu",
            [
                PYTHON_BIN, str(RUN / "wave2_longbench_eigen.py"),
                "--method", method,
                "--model", model,
                "--tok", "/diskD/jiaqizhang/Llama-3",
                "--data_dir", "/home/bai-project/jiaqizhang/EigenAttn/data/LongBench",
                "--max_len", "8192",
                "--limit_per_task", "20",
                "--out_dir", f"{OUT_ROOT}/longbench/{method}",
            ],
            depends_on=["WAVE2_GATE_MCFIX"],
            meta={"reserve_mib": 22000, "role": "eval", "method": method},
        ))

    jobs.append(make_job(
        "WAVE2_LB_SCORE", "WAVE2", "cpu",
        [
            PYTHON_BIN, str(RUN / "wave2_longbench_score.py"),
            "--pred_root", f"{OUT_ROOT}/longbench",
            "--out_dir", f"{OUT_ROOT}/longbench/scores",
        ],
        depends_on=["WAVE2_LB_dense", "WAVE2_LB_uniform", "WAVE2_LB_fdp"],
        meta={"role": "score"},
    ))

    # OPT-30B single-point Wiki PPL (reuse audited FGeb026 if present)
    jobs.append(make_job(
        "WAVE2_OPT_WIKI_PPL", "WAVE2", "gpu",
        [
            PYTHON_BIN, str(RUN / "wave2_opt30b_wiki_ppl.py"),
            "--out_dir", f"{OUT_ROOT}/opt30b",
        ],
        depends_on=["WAVE2_GATE_MCFIX"],
        meta={"reserve_mib": 70000, "role": "eval", "note": "OPT prefers near-empty GPU"},
    ))

    # Qwen: engineering gate + optional smoke load (no fake compress)
    jobs.append(make_job(
        "WAVE2_QWEN_GATE", "WAVE2", "cpu",
        [PYTHON_BIN, str(RUN / "wave2_qwen_gate.py"), "--out_dir", f"{OUT_ROOT}/qwen"],
        depends_on=["WAVE2_GATE_MCFIX"],
        meta={"role": "gate"},
    ))

    # External baselines: honest blocked/attempt report
    jobs.append(make_job(
        "WAVE2_EXT_BASELINE", "WAVE2", "cpu",
        [PYTHON_BIN, str(RUN / "wave2_external_baseline.py"), "--out_dir", f"{OUT_ROOT}/external"],
        depends_on=["WAVE2_GATE_MCFIX"],
        meta={"role": "gate"},
    ))

    # Component ablation: execute minimal confirmatory subset if assets allow
    jobs.append(make_job(
        "WAVE2_COMPONENT_ABLATION", "WAVE2", "gpu",
        [
            PYTHON_BIN, str(RUN / "wave2_component_ablation.py"),
            "--out_dir", f"{OUT_ROOT}/component_ablation",
        ],
        depends_on=["WAVE2_GATE_MCFIX"],
        meta={"reserve_mib": 22000, "role": "build"},
    ))

    jobs.append(make_job(
        "WAVE2_FINAL_STATUS", "WAVE2", "cpu",
        [PYTHON_BIN, str(RUN / "wave2_final_status.py"), "--out_root", OUT_ROOT],
        depends_on=[
            "WAVE2_MC_AGGREGATE",
            "WAVE2_LB_SCORE",
            "WAVE2_OPT_WIKI_PPL",
            "WAVE2_QWEN_GATE",
            "WAVE2_EXT_BASELINE",
            "WAVE2_COMPONENT_ABLATION",
        ],
        meta={"role": "report"},
    ))
    return jobs


def main() -> None:
    jobs_path = HERE / "jobs.jsonl"
    existing = [json.loads(l) for l in jobs_path.read_text().splitlines() if l.strip()] if jobs_path.exists() else []
    from stage_materialize import merge_jobs  # noqa: E402
    merged = merge_jobs(existing, build_wave2_jobs())
    # Preserve DONE/RUNNING/FAILED from disk markers for WAVE2 only if present
    for j in merged:
        if not str(j["job_id"]).startswith("WAVE2_"):
            continue
        out = Path(j["output_dir"])
        if (out / "DONE.json").exists():
            j["status"] = "DONE"
        elif (out / "FAILED.json").exists() and j.get("status") != "RUNNING":
            j["status"] = "FAILED"
    jobs_path.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in merged))
    w2 = [j for j in merged if j["job_id"].startswith("WAVE2_")]
    print(json.dumps({"ok": True, "n_wave2": len(w2), "n_total": len(merged)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
