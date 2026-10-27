#!/usr/bin/env python3
"""Job materialization for rebuttal completion pipeline DAG."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import PYTHON_BIN, artifact_dir, evaluation_hash, file_hash_or_empty, here, protocol_hash

HERE = here()
LEGACY = HERE.parent / "rebuttal_legacy_reproduction"
LINK = HERE / "artifacts"


def _job_dir(job_id: str) -> Path:
    d = HERE / "job_outputs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_job(
    job_id: str,
    stage: str,
    kind: str,
    cmd: List[str],
    *,
    depends_on: Optional[List[str]] = None,
    preferred_gpu: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "stage": stage,
        "kind": kind,
        "cmd": cmd,
        "status": "PENDING",
        "depends_on": depends_on or [],
        "preferred_gpu": preferred_gpu,
        "output_dir": str(_job_dir(job_id)),
        "meta": meta or {},
        "protocol_hash": protocol_hash(),
        "evaluation_hash": evaluation_hash(),
    }


def build_phase0_jobs() -> List[Dict[str, Any]]:
    return [
        make_job("P0_audit", "P0", "cpu", [PYTHON_BIN, str(HERE / "audit_evidence.py")]),
        make_job("P0_freeze_protocol", "P0", "cpu", [PYTHON_BIN, str(HERE / "runners" / "freeze_protocol.py")], depends_on=["P0_audit"]),
        make_job("P0_smoke", "P0", "cpu", [PYTHON_BIN, str(HERE / "runners" / "smoke_test.py")], depends_on=["P0_freeze_protocol"]),
    ]


def build_phase1_jobs() -> List[Dict[str, Any]]:
    """Operating points: mild/moderate/aggressive."""
    jobs = []
    for point, eb, gpu in [("mild", "0.045", 0), ("moderate", "0.060", 1), ("aggressive", "0.090", 2)]:
        jobs.append(make_job(
            f"OP_{point}_build",
            "P1",
            "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "operating_point_build.py"), "--point", point, "--eb_base", eb],
            preferred_gpu=gpu,
            depends_on=["P0_smoke"],
            meta={"point": point, "eb_base": eb},
        ))
        for metric, ds, gpu_m in [
            ("wikitext", "wikitext2", 0),
            ("c4", "c4", 1),
        ]:
            jobs.append(make_job(
                f"OP_{point}_{metric}_ppl",
                "P1",
                "gpu",
                [PYTHON_BIN, str(HERE / "runners" / "eval_reuse_instrumented.py"),
                 "--point", point, "--dataset", ds],
                preferred_gpu=gpu_m,
                depends_on=[f"OP_{point}_build"],
            ))
        # Fine-grained zero-shot: method × bundle → pack across GPUs automatically.
        zi = 0
        zs_ids = []
        for meth in ("uniform", "fdp"):
            for bundle in ("hs_arc", "wg_piqa"):
                jid = f"OP_{point}_zs_{meth}_{bundle}"
                zs_ids.append(jid)
                jobs.append(make_job(
                    jid,
                    "P1",
                    "gpu",
                    [
                        PYTHON_BIN, str(HERE / "runners" / "eval_reuse_instrumented.py"),
                        "--point", point, "--dataset", "zero_shot",
                        "--method", meth, "--bundle", bundle,
                    ],
                    preferred_gpu=zi % 4,
                    depends_on=[f"OP_{point}_build"],
                    meta={"point": point, "method": meth, "bundle": bundle, "reserve_mib": 18000},
                ))
                zi += 1
        jobs.append(make_job(
            f"OP_{point}_zs",
            "P1",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "zs_gate.py"), "--point", point],
            depends_on=zs_ids,
        ))
    # Aggregate needs PPL only (tables); zs finishes via OP_*_zs gate in parallel.
    jobs.append(make_job(
        "OP_aggregate",
        "P1",
        "cpu",
        [PYTHON_BIN, str(HERE / "runners" / "operating_point_aggregate.py")],
        depends_on=[
            f"OP_{p}_{m}_ppl"
            for p in ("mild", "moderate", "aggressive")
            for m in ("wikitext", "c4")
        ],
    ))
    return jobs


def build_phase2_jobs() -> List[Dict[str, Any]]:
    """Matched-budget controls at moderate operating point."""
    controls = [
        "uniform_matched", "random_seed0", "random_seed1", "random_seed2",
        "spectral_tail", "middle_layer", "depth_front", "depth_back", "reconstruction_slope",
    ]
    jobs = []
    for i, ctrl in enumerate(controls):
        # Schedule JSON only (CPU). Does not need zs; anchors on moderate FDP build.
        jobs.append(make_job(
            f"MC_{ctrl}",
            "P2",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "matched_control_build.py"), "--control", ctrl, "--anchor", "moderate"],
            preferred_gpu=i % 4,
            depends_on=["OP_moderate_build"],
            meta={"control": ctrl},
        ))
    jobs.append(make_job(
        "MC_aggregate",
        "P2",
        "cpu",
        [PYTHON_BIN, str(HERE / "runners" / "matched_control_aggregate.py")],
        depends_on=[f"MC_{c}" for c in controls],
    ))
    return jobs


def build_phase3_jobs() -> List[Dict[str, Any]]:
    """Bootstrap/McNemar/Holm from existing + new raw outputs."""
    return [
        make_job("P3_bootstrap_legacy", "P3", "cpu",
                 [PYTHON_BIN, str(HERE / "runners" / "paired_bootstrap.py"), "--scope", "legacy_confirmatory"],
                 depends_on=["P0_smoke"]),
        make_job("P3_bootstrap_op", "P3", "cpu",
                 [PYTHON_BIN, str(HERE / "runners" / "paired_bootstrap.py"), "--scope", "operating_points"],
                 depends_on=["OP_aggregate"]),
        make_job("P3_bootstrap_mc", "P3", "cpu",
                 [PYTHON_BIN, str(HERE / "runners" / "paired_bootstrap.py"), "--scope", "matched_controls"],
                 depends_on=["MC_aggregate"]),
        make_job("P3_mcnemar_holm", "P3", "cpu",
                 [PYTHON_BIN, str(HERE / "runners" / "mcnemar_holm.py")],
                 depends_on=["P3_bootstrap_legacy", "P3_bootstrap_op", "P3_bootstrap_mc"]),
    ]


def build_phase4_jobs() -> List[Dict[str, Any]]:
    return [
        make_job("P4_system_audit", "P4", "cpu",
                 [PYTHON_BIN, str(HERE / "runners" / "system_metrics_audit.py")],
                 depends_on=["P0_smoke"]),
        make_job("P4_long_context_manifest", "P4", "cpu",
                 [PYTHON_BIN, str(HERE / "runners" / "long_context_manifest.py")],
                 depends_on=["P0_smoke"]),
        make_job("P4_long_context_8k", "P4", "gpu",
                 [PYTHON_BIN, str(HERE / "runners" / "long_context_eval.py"), "--profile", "8k"],
                 preferred_gpu=0,
                 depends_on=["P4_long_context_manifest", "OP_moderate_build"]),
        make_job("P4_long_context_2k", "P4", "gpu",
                 [PYTHON_BIN, str(HERE / "runners" / "long_context_eval.py"), "--profile", "2k"],
                 preferred_gpu=1,
                 depends_on=["P4_long_context_manifest", "OP_moderate_build"]),
        make_job("P4_spectrum", "P4", "cpu",
                 [PYTHON_BIN, str(HERE / "runners" / "spectrum_analysis.py")],
                 depends_on=["OP_moderate_build"]),
    ]


def build_phase5_jobs() -> List[Dict[str, Any]]:
    return [
        make_job("P5_kqv_ablation", "P5", "gpu",
                 [PYTHON_BIN, str(HERE / "runners" / "kqv_ablation_instrumented.py")],
                 preferred_gpu=0,
                 depends_on=["OP_moderate_build"]),
        make_job("P5_component_ablation", "P5", "gpu",
                 [PYTHON_BIN, str(HERE / "runners" / "component_ablation.py")],
                 preferred_gpu=1,
                 depends_on=["OP_moderate_build"]),
        make_job("P5_external_manifest", "P5", "cpu",
                 [PYTHON_BIN, str(HERE / "runners" / "external_baseline_manifest.py")],
                 depends_on=["P0_smoke"]),
        make_job("P5_opt30b", "P5", "gpu",
                 [PYTHON_BIN, str(HERE / "runners" / "opt30b_fixed_schedule.py")],
                 preferred_gpu=2,
                 depends_on=["P0_smoke"]),
        make_job("P5_qwen_search", "P5", "cpu",
                 [PYTHON_BIN, str(HERE / "runners" / "qwen_model_search.py")],
                 depends_on=["P0_smoke"]),
    ]


def build_phase6_jobs() -> List[Dict[str, Any]]:
    return [
        make_job("P6_final_report", "P6", "cpu",
                 [PYTHON_BIN, str(HERE / "runners" / "final_report.py")],
                 depends_on=[
                     "P3_mcnemar_holm", "P4_system_audit", "P4_spectrum",
                     "P5_kqv_ablation", "P5_component_ablation", "P5_opt30b", "P5_qwen_search",
                 ]),
    ]


def merge_jobs(existing: List[Dict[str, Any]], new_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {j["job_id"]: j for j in existing}
    for nj in new_jobs:
        if nj["job_id"] in by_id:
            old = by_id[nj["job_id"]]
            if old.get("status") in ("DONE", "RUNNING", "FAILED", "STOPPED"):
                continue
            by_id[nj["job_id"]] = nj
        else:
            by_id[nj["job_id"]] = nj
    ordered, seen = [], set()
    for j in existing + new_jobs:
        if j["job_id"] in seen:
            continue
        ordered.append(by_id[j["job_id"]])
        seen.add(j["job_id"])
    return ordered


def materialize_all(existing: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    jobs = list(existing or [])
    for builder in (
        build_phase0_jobs, build_phase1_jobs, build_phase2_jobs,
        build_phase3_jobs, build_phase4_jobs, build_phase5_jobs, build_phase6_jobs,
    ):
        jobs = merge_jobs(jobs, builder())
    # Priority repair DAG (KV fix, MC evals, OP stats, KQV v2)
    try:
        from materialize_repair_jobs import build_repair_jobs  # noqa: WPS433
        jobs = merge_jobs(jobs, build_repair_jobs())
    except Exception:
        pass
    # WAVE2: real gap-closing jobs after MCFIX (LongBench/OPT/Qwen/external/component)
    try:
        from materialize_wave2_jobs import build_wave2_jobs  # noqa: WPS433
        jobs = merge_jobs(jobs, build_wave2_jobs())
    except Exception:
        pass
    return jobs


def future_stages_remain(jobs: List[Dict[str, Any]]) -> bool:
    if not (HERE / "COMPLETION_PIPELINE_PASS.DONE").exists():
        return True
    return any(j.get("status") not in ("DONE", "STOPPED") for j in jobs)
