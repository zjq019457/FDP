#!/usr/bin/env python3
"""Materialize R2/R3/... jobs after upstream parity gates. Never requeues completed R1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import PYTHON_BIN, artifact_dir, file_hash_or_empty, now_iso

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = "/diskD/jiaqizhang/Llama-3"
WIKITEXT = "/home/bai-project/jiaqizhang/wikitext2"
C4_TEXTS = (
    "/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/"
    "llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl"
)
TASK_YAML = "/diskF/jiaqizhang/a1804298693/offline_datasets"
EVAL_HASH = file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json")
LINK_ART = HERE / "artifacts"


def _job_dir(job_id: str) -> Path:
    d = HERE / "job_outputs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_job(
    job_id: str,
    stage: str,
    kind: str,
    command: List[str],
    *,
    preferred_gpu: Optional[int] = None,
    depends_on: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    od = _job_dir(job_id)
    # Preserve DONE jobs
    done = od / "DONE.json"
    status = "DONE" if done.exists() else "PENDING"
    manifest = {
        "job_id": job_id,
        "stage": stage,
        "kind": kind,
        "command": command,
        "cwd": cwd or str(ROOT / "EigenAttn"),
        "preferred_gpu": preferred_gpu,
        "depends_on": depends_on or [],
        "created_at": now_iso(),
        "meta": meta or {},
    }
    (od / "job_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if status == "PENDING":
        (od / "job_state.json").write_text(
            json.dumps({"job_id": job_id, "status": "PENDING", "updated_at": now_iso()}, indent=2) + "\n"
        )
    return {
        "job_id": job_id,
        "stage": stage,
        "kind": kind,
        "status": status,
        "command": command,
        "cwd": manifest["cwd"],
        "preferred_gpu": preferred_gpu,
        "depends_on": depends_on or [],
        "min_free_mib": 0,  # shared-GPU policy: no free-VRAM requirement
        "output_dir": str(od),
        "meta": meta or {},
    }


def build_r2_jobs() -> List[Dict[str, Any]]:
    uni_root = artifact_dir("r2_uniform_eb045", link_under=LINK_ART / "r2_uniform_eb045")
    uni_model = str(uni_root / "final_model")
    jobs = []
    jobs.append(
        make_job(
            "R2_uniform_preflight",
            "R2",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "r2_uniform_preflight.py")],
            meta={"role": "preflight"},
        )
    )
    jobs.append(
        make_job(
            "R2_uniform_compress",
            "R2",
            "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "r2_uniform_compress.py")],
            preferred_gpu=0,
            depends_on=["R2_uniform_preflight"],
            meta={"role": "compress"},
        )
    )
    jobs.append(
        make_job(
            "R2_uniform_smoke",
            "R2",
            "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "r2_uniform_smoke.py")],
            preferred_gpu=0,
            depends_on=["R2_uniform_compress"],
            meta={"role": "smoke"},
        )
    )
    wiki_out = str(artifact_dir("R2_uniform_wikitext_ppl", link_under=LINK_ART / "R2_uniform_wikitext_ppl"))
    c4_out = str(artifact_dir("R2_uniform_c4_ppl", link_under=LINK_ART / "R2_uniform_c4_ppl"))
    zs1_out = str(artifact_dir("R2_uniform_zs_hs_arc", link_under=LINK_ART / "R2_uniform_zs_hs_arc"))
    zs2_out = str(artifact_dir("R2_uniform_zs_wg_piqa", link_under=LINK_ART / "R2_uniform_zs_wg_piqa"))
    jobs.append(
        make_job(
            "R2_uniform_wikitext_ppl",
            "R2",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "ppl_instrumented.py"),
                "--dataset",
                "wikitext2",
                "--model",
                uni_model,
                "--tok",
                MODEL,
                "--out_dir",
                wiki_out,
                "--wikitext2_dir",
                WIKITEXT,
                "--dtype",
                "bf16",
                "--method",
                "uniform_eb045",
                "--evaluation_protocol_hash",
                EVAL_HASH,
            ],
            preferred_gpu=0,
            depends_on=["R2_uniform_smoke"],
        )
    )
    jobs.append(
        make_job(
            "R2_uniform_c4_ppl",
            "R2",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "ppl_instrumented.py"),
                "--dataset",
                "c4",
                "--model",
                uni_model,
                "--tok",
                MODEL,
                "--c4_texts",
                C4_TEXTS,
                "--out_dir",
                c4_out,
                "--dtype",
                "bf16",
                "--method",
                "uniform_eb045",
                "--evaluation_protocol_hash",
                EVAL_HASH,
            ],
            preferred_gpu=1,
            depends_on=["R2_uniform_smoke"],
        )
    )
    jobs.append(
        make_job(
            "R2_uniform_zs_hs_arc",
            "R2",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "zero_shot_instrumented.py"),
                "--model",
                uni_model,
                "--tokenizer_model",
                MODEL,
                "--task_yaml_dir",
                TASK_YAML,
                "--tasks",
                "hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl",
                "--out_dir",
                zs1_out,
                "--dtype",
                "bf16",
                "--net",
                "llama-3-8b",
                "--method",
                "uniform_eb045",
                "--low_rank_config",
                str(Path(uni_model) / "low_rank_config.json"),
            ],
            preferred_gpu=2,
            depends_on=["R2_uniform_smoke"],
        )
    )
    jobs.append(
        make_job(
            "R2_uniform_zs_wg_piqa",
            "R2",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "zero_shot_instrumented.py"),
                "--model",
                uni_model,
                "--tokenizer_model",
                MODEL,
                "--task_yaml_dir",
                TASK_YAML,
                "--tasks",
                "winogrande_localjsonl,piqa_localjsonl",
                "--out_dir",
                zs2_out,
                "--dtype",
                "bf16",
                "--net",
                "llama-3-8b",
                "--method",
                "uniform_eb045",
                "--low_rank_config",
                str(Path(uni_model) / "low_rank_config.json"),
            ],
            preferred_gpu=3,
            depends_on=["R2_uniform_smoke"],
        )
    )
    jobs.append(
        make_job(
            "R2_uniform_aggregate",
            "R2",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "r2_aggregate.py")],
            depends_on=[
                "R2_uniform_wikitext_ppl",
                "R2_uniform_c4_ppl",
                "R2_uniform_zs_hs_arc",
                "R2_uniform_zs_wg_piqa",
            ],
        )
    )
    return jobs


def build_r3_jobs() -> List[Dict[str, Any]]:
    """FDP-EB / FGeb045: Step1 → per-layer EB → adaptive basis → compress → smoke → 4-GPU eval."""
    fdp_root = artifact_dir("r3_fgeb045", link_under=LINK_ART / "r3_fgeb045")
    fdp_model = str(fdp_root / "final_model")
    jobs: List[Dict[str, Any]] = []
    jobs.append(
        make_job(
            "R3_step1",
            "R3",
            "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "r3_step1.py")],
            preferred_gpu=0,
            depends_on=["R2_uniform_aggregate"],
            meta={"role": "step1"},
        )
    )
    jobs.append(
        make_job(
            "R3_per_layer_eb",
            "R3",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "r3_per_layer_eb.py")],
            depends_on=["R3_step1"],
            meta={"role": "per_layer_eb"},
        )
    )
    jobs.append(
        make_job(
            "R3_adaptive_basis",
            "R3",
            "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "r3_adaptive_basis.py")],
            preferred_gpu=0,
            depends_on=["R3_per_layer_eb"],
            meta={"role": "adaptive_basis"},
        )
    )
    jobs.append(
        make_job(
            "R3_compress",
            "R3",
            "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "r3_compress.py")],
            preferred_gpu=0,
            depends_on=["R3_adaptive_basis"],
            meta={"role": "compress"},
        )
    )
    jobs.append(
        make_job(
            "R3_smoke",
            "R3",
            "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "r3_smoke.py")],
            preferred_gpu=0,
            depends_on=["R3_compress"],
            meta={"role": "smoke"},
        )
    )
    wiki_out = str(artifact_dir("R3_fdp_wikitext_ppl", link_under=LINK_ART / "R3_fdp_wikitext_ppl"))
    c4_out = str(artifact_dir("R3_fdp_c4_ppl", link_under=LINK_ART / "R3_fdp_c4_ppl"))
    zs1_out = str(artifact_dir("R3_fdp_zs_hs_arc", link_under=LINK_ART / "R3_fdp_zs_hs_arc"))
    zs2_out = str(artifact_dir("R3_fdp_zs_wg_piqa", link_under=LINK_ART / "R3_fdp_zs_wg_piqa"))
    jobs.append(
        make_job(
            "R3_fdp_wikitext_ppl",
            "R3",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "ppl_instrumented.py"),
                "--dataset",
                "wikitext2",
                "--model",
                fdp_model,
                "--tok",
                MODEL,
                "--out_dir",
                wiki_out,
                "--wikitext2_dir",
                WIKITEXT,
                "--dtype",
                "bf16",
                "--method",
                "FGeb045",
                "--evaluation_protocol_hash",
                EVAL_HASH,
            ],
            preferred_gpu=0,
            depends_on=["R3_smoke"],
        )
    )
    jobs.append(
        make_job(
            "R3_fdp_c4_ppl",
            "R3",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "ppl_instrumented.py"),
                "--dataset",
                "c4",
                "--model",
                fdp_model,
                "--tok",
                MODEL,
                "--c4_texts",
                C4_TEXTS,
                "--out_dir",
                c4_out,
                "--dtype",
                "bf16",
                "--method",
                "FGeb045",
                "--evaluation_protocol_hash",
                EVAL_HASH,
            ],
            preferred_gpu=1,
            depends_on=["R3_smoke"],
        )
    )
    jobs.append(
        make_job(
            "R3_fdp_zs_hs_arc",
            "R3",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "zero_shot_instrumented.py"),
                "--model",
                fdp_model,
                "--tokenizer_model",
                MODEL,
                "--task_yaml_dir",
                TASK_YAML,
                "--tasks",
                "hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl",
                "--out_dir",
                zs1_out,
                "--dtype",
                "bf16",
                "--net",
                "llama-3-8b",
                "--method",
                "FGeb045",
                "--low_rank_config",
                str(Path(fdp_model) / "low_rank_config.json"),
            ],
            preferred_gpu=2,
            depends_on=["R3_smoke"],
        )
    )
    jobs.append(
        make_job(
            "R3_fdp_zs_wg_piqa",
            "R3",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "zero_shot_instrumented.py"),
                "--model",
                fdp_model,
                "--tokenizer_model",
                MODEL,
                "--task_yaml_dir",
                TASK_YAML,
                "--tasks",
                "winogrande_localjsonl,piqa_localjsonl",
                "--out_dir",
                zs2_out,
                "--dtype",
                "bf16",
                "--net",
                "llama-3-8b",
                "--method",
                "FGeb045",
                "--low_rank_config",
                str(Path(fdp_model) / "low_rank_config.json"),
            ],
            preferred_gpu=3,
            depends_on=["R3_smoke"],
        )
    )
    jobs.append(
        make_job(
            "R3_fdp_aggregate",
            "R3",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "r3_aggregate.py")],
            depends_on=[
                "R3_fdp_wikitext_ppl",
                "R3_fdp_c4_ppl",
                "R3_fdp_zs_hs_arc",
                "R3_fdp_zs_wg_piqa",
            ],
            meta={"role": "aggregate"},
        )
    )
    return jobs


def build_supplementary_jobs() -> List[Dict[str, Any]]:
    """R8 system metrics, R10 KQV ablation, R11 robustness, supplementary complete."""
    jobs: List[Dict[str, Any]] = []
    jobs.append(
        make_job(
            "R8_system_metrics",
            "R8",
            "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "r8_system_metrics.py")],
            preferred_gpu=0,
            depends_on=["R6_summary"],
            meta={"role": "system_metrics"},
        )
    )
    jobs.append(
        make_job(
            "R10_kqv_ablation",
            "R10",
            "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "r10_kqv_ablation.py")],
            preferred_gpu=1,
            depends_on=["R6_summary"],
            meta={"role": "kqv_ablation"},
        )
    )
    jobs.append(
        make_job(
            "R11_robustness",
            "R11",
            "gpu",
            [PYTHON_BIN, str(HERE / "runners" / "r11_robustness.py")],
            preferred_gpu=2,
            depends_on=["R6_summary"],
            meta={"role": "robustness"},
        )
    )
    jobs.append(
        make_job(
            "SUPP_complete",
            "SUPP",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "supp_complete.py")],
            depends_on=["R8_system_metrics", "R10_kqv_ablation", "R11_robustness"],
            meta={"role": "supplementary_complete"},
        )
    )
    return jobs


def build_r4_r6_jobs() -> List[Dict[str, Any]]:
    """R4 statistics, R5 historical parity, R6 final summary."""
    jobs: List[Dict[str, Any]] = []
    jobs.append(
        make_job(
            "R4_statistics",
            "R4",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "r4_statistics.py")],
            depends_on=["R3_fdp_aggregate"],
            meta={"role": "statistics"},
        )
    )
    jobs.append(
        make_job(
            "R5_historical_parity",
            "R5",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "r5_historical_parity.py")],
            depends_on=["R4_statistics"],
            meta={"role": "historical_parity"},
        )
    )
    jobs.append(
        make_job(
            "R6_summary",
            "R6",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "r6_summary.py")],
            depends_on=["R5_historical_parity"],
            meta={"role": "summary"},
        )
    )
    return jobs


def merge_jobs(existing: List[Dict[str, Any]], new_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {j["job_id"]: j for j in existing}
    for nj in new_jobs:
        if nj["job_id"] in by_id:
            old = by_id[nj["job_id"]]
            # keep DONE/FAILED/RUNNING statuses; refresh command for pending
            if old.get("status") in ("DONE", "RUNNING", "FAILED", "STOPPED"):
                continue
            by_id[nj["job_id"]] = nj
        else:
            by_id[nj["job_id"]] = nj
    # preserve order: existing first, then new ids
    ordered = []
    seen = set()
    for j in existing + new_jobs:
        jid = j["job_id"]
        if jid in seen:
            continue
        ordered.append(by_id[jid])
        seen.add(jid)
    return ordered


def materialize_after_gates(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jobs = list(existing)
    # Never requeue R1 if DONE
    if (HERE / "DENSE_PARITY_PASS.DONE").exists():
        # Ensure R1 jobs marked DONE if artifacts exist
        for j in jobs:
            if j["job_id"].startswith("R1_") and (Path(j["output_dir"]) / "DONE.json").exists():
                j["status"] = "DONE"
        # Materialize R2 if not already present
        if not any(j["job_id"].startswith("R2_") for j in jobs):
            jobs = merge_jobs(jobs, build_r2_jobs())
        elif not (HERE / "UNIFORM_PARITY_PASS.DONE").exists():
            # refresh pending R2 commands
            jobs = merge_jobs(jobs, build_r2_jobs())

    # R3 full binders after uniform parity
    if (HERE / "UNIFORM_PARITY_PASS.DONE").exists() and not (HERE / "LEGACY_FDP_PARITY_PASS.DONE").exists():
        jobs = merge_jobs(jobs, build_r3_jobs())

    # R4–R6 after FDP parity
    if (HERE / "LEGACY_FDP_PARITY_PASS.DONE").exists():
        if not any(j["job_id"].startswith("R3_") for j in jobs):
            jobs = merge_jobs(jobs, build_r3_jobs())
        jobs = merge_jobs(jobs, build_r4_r6_jobs())

    # Supplementary after R6
    if (HERE / "R6_STATISTICS_PASS.DONE").exists():
        supp_done = (HERE / "supplementary_experiments" / "SUPPLEMENTARY_EXPERIMENTS_COMPLETE.DONE").exists()
        if not supp_done:
            jobs = merge_jobs(jobs, build_supplementary_jobs())
    return jobs


def future_stages_remain(jobs: List[Dict[str, Any]]) -> bool:
    if not (HERE / "DENSE_PARITY_PASS.DONE").exists():
        return True
    if not (HERE / "UNIFORM_PARITY_PASS.DONE").exists():
        return True
    if not (HERE / "LEGACY_FDP_PARITY_PASS.DONE").exists():
        return True
    if not (HERE / "R6_STATISTICS_PASS.DONE").exists():
        return True
    # supplementary incomplete
    supp = HERE / "supplementary_experiments" / "SUPPLEMENTARY_EXPERIMENTS_COMPLETE.DONE"
    if not supp.exists():
        return True
    return False
