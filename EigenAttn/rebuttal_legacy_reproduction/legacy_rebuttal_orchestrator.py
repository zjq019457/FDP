#!/usr/bin/env python3
"""Legacy rebuttal master orchestrator: R1–R6 + supplementary DAG, 4-GPU workers, tmux-detach."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    HEARTBEAT_SECS,
    MIN_FREE_MIB_DEFAULT,
    PYTHON_BIN,
    SESSION_NAME,
    atomic_write_json,
    file_hash_or_empty,
    gate_payload,
    now_iso,
    sha256_file,
    sha256_json,
    sha256_text,
    stop_requested,
    write_pass_gate,
)
from gpu_scheduler import GPULockManager, write_inventory, write_scheduler_audit  # noqa: E402
from stage_materialize import future_stages_remain, materialize_after_gates  # noqa: E402

MODEL = "/diskD/jiaqizhang/Llama-3"
WIKITEXT = "/home/bai-project/jiaqizhang/wikitext2"
C4_TEXTS = (
    "/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/"
    "llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl"
)
TASK_YAML = "/diskF/jiaqizhang/a1804298693/offline_datasets"
UNIFORM_FINAL = "/diskD/jiaqizhang/llama3_8b_eigen_kv_eb045/final_model"
FDP_FINAL = "/diskD/jiaqizhang/FGeb045/final_model"


def ensure_layout() -> None:
    for d in (
        "background_logs",
        "artifacts",
        "job_outputs",
        "heartbeats",
        "raw",
        "gates",
        "runners",
        "reviewed_runners",
        "patches",
        "supplementary_experiments",
        "worker_logs",
    ):
        (HERE / d).mkdir(parents=True, exist_ok=True)


def write_status(state: Dict[str, Any]) -> None:
    atomic_write_json(HERE / "REPRODUCTION_PROGRESS.json", state)
    atomic_write_json(HERE / "orchestrator_state.json", state)
    md = [
        "# Current Status",
        "",
        f"- generated_at: `{now_iso()}`",
        f"- phase: `{state.get('phase')}`",
        f"- blocked: `{state.get('blocked')}`",
        f"- tmux_session: `{SESSION_NAME}`",
        f"- master_pid: `{state.get('master_pid', os.getpid())}`",
        f"- last_action: `{state.get('last_action', '')}`",
        f"- python: `{PYTHON_BIN}`",
        "",
        "## Gates",
    ]
    for k, v in (state.get("gates") or {}).items():
        md.append(f"- `{k}`: `{v}`")
    md.extend(["", "## Jobs", ""])
    jc = state.get("job_counts") or {}
    for k in ("completed", "running", "pending", "blocked", "failed", "stopped"):
        md.append(f"- `{k}`: `{jc.get(k, 0)}`")
    md.extend(["", "## GPU assignments", ""])
    for g, j in (state.get("gpu_jobs") or {}).items():
        md.append(f"- GPU `{g}`: `{j}`")
    (HERE / "CURRENT_STATUS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def freeze_protocol() -> Dict[str, Any]:
    """Freeze recovered legacy candidate protocol with provenance tags."""
    fields = {
        "lambda_eb": {"value": 1.2, "provenance": "recovered_from_artifact", "source": "step1_out/llama3_8b_wikitext2_frag_eb045.json meta.lambda_eb"},
        "normalize": {"value": "mean", "provenance": "recovered_from_artifact", "source": "step1_out/...frag_eb045.json meta.normalize"},
        "weights": {
            "value": {"wz": 1.0, "ws": 1.0, "wv": 1.0, "wa": 1.0, "wh": 1.0},
            "provenance": "recovered_from_artifact",
            "source": "frag_eb045.json meta.weights; also source defaults in per_layer_eb_from_step1.py",
        },
        "freeze_layers": {"value": [0], "provenance": "recovered_from_artifact", "source": "frag_eb045.json meta.freeze_layers"},
        "entropy_penalty": {
            "value": "max(0, -entropy_diff_A)",
            "provenance": "recovered_from_source",
            "source": "per_layer_eb_from_step1.py / realloc_rank_from_fragility.py",
        },
        "fragility_formula": {
            "value": "s_l = wz*Z_rel + ws*Score_rel + wv*Value_rel + wa*AV_rel + wh*max(0,-entropy_diff_A)",
            "provenance": "recovered_from_source",
            "source": "realloc_rank_from_fragility.py",
        },
        "per_layer_eb_map": {
            "value": "eb_l = eb_base / (1 + lambda_eb * s_tilde_l)",
            "provenance": "recovered_from_source",
            "source": "per_layer_eb_from_step1.py",
        },
        "main_operating_point": {
            "value": {"tag": "FGeb045", "uniform": "eb045", "compare": "uniform_vs_FGeb045"},
            "provenance": "recovered_from_artifact",
            "source": "llama3_8b_eb045_uniform_vs_FGeb045_distortion_summary.json",
        },
        "builder_script_defaults_eb_base": {
            "value": 0.024,
            "provenance": "recovered_from_source",
            "source": "per_layer_eb_from_step1.py argparse default (NOT confirmed as submission operating point)",
            "note": "Do not confuse script default with FGeb045 artifact operating point.",
        },
        "calibration_nsamples_seqlen": {
            "value": {"nsamples": "unresolved_exact_submission", "seqlen": 2048},
            "provenance": "inferred_only",
            "source": "builder defaults nsamples=512 seqlen=2048; MANIFEST mentions 128; not hard-coded as truth",
        },
        "score_term_S_definition": {
            "value": "score_rel_mse_causal (Score_rel in fragility sum)",
            "provenance": "recovered_from_source",
            "source": "realloc_rank_from_fragility.py / step1 diagnose fields",
        },
    }
    unresolved = [
        {
            "field": "exact_submission_calibration_nsamples",
            "status": "inferred_only",
            "detail": "builder default 512 vs MANIFEST 128; not frozen as submission truth",
        },
        {
            "field": "optimized_shared_basis_equivalence",
            "status": "pending_R4",
            "detail": "Not yet verified; extensions must use legacy builder until R4 PASS",
        },
    ]
    lock = {
        "generated_at": now_iso(),
        "status": "legacy_candidate_protocol_frozen",
        "policy": "inferred_only fields must not be silently treated as confirmed truth",
        "fields": fields,
        "paths": {
            "model": MODEL,
            "wikitext2": WIKITEXT,
            "c4_selected_texts": C4_TEXTS,
            "task_yaml_dir": TASK_YAML,
            "uniform_final_model_candidate": UNIFORM_FINAL,
            "fdp_final_model_candidate": FDP_FINAL,
        },
        "python_bin": PYTHON_BIN,
        "unresolved": unresolved,
        "v1_2_policy": [
            "simplified_exact_byte_matched_protocol",
            "not_submission_reproduction",
            "negative_result",
        ],
    }
    atomic_write_json(HERE / "LEGACY_PROTOCOL_LOCK.json", lock)
    (HERE / "LEGACY_PROTOCOL_LOCK.sha256").write_text(
        sha256_file(HERE / "LEGACY_PROTOCOL_LOCK.json") + "\n", encoding="utf-8"
    )
    md = [
        "# Legacy Protocol Recovery",
        "",
        f"- generated_at: `{now_iso()}`",
        "- This is a **legacy candidate protocol** frozen from source/artifact evidence.",
        "- Paper numbers were NOT used to reverse-select eb045/058/060.",
        "",
        "## Fields",
        "",
    ]
    for k, v in fields.items():
        md.append(f"### `{k}`")
        md.append(f"- value: `{v['value']}`")
        md.append(f"- provenance: `{v['provenance']}`")
        md.append(f"- source: `{v['source']}`")
        if "note" in v:
            md.append(f"- note: {v['note']}")
        md.append("")
    md.append("## Unresolved")
    md.append("")
    for u in unresolved:
        md.append(f"- `{u['field']}` ({u['status']}): {u['detail']}")
    (HERE / "LEGACY_PROTOCOL_RECOVERY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (HERE / "UNRESOLVED_LEGACY_FIELDS.md").write_text(
        "# Unresolved Legacy Fields\n\n"
        + "\n".join(f"- `{u['field']}` [{u['status']}]: {u['detail']}" for u in unresolved)
        + "\n",
        encoding="utf-8",
    )
    return lock


def freeze_supplementary_protocol() -> None:
    supp = HERE / "supplementary_experiments"
    supp.mkdir(parents=True, exist_ok=True)
    proto = {
        "generated_at": now_iso(),
        "unlock_requires": [
            "RAW_OUTPUT_INTERFACE_PASS",
            "DENSE_PARITY_PASS",
            "UNIFORM_PARITY_PASS",
            "LEGACY_FDP_PARITY_PASS",
            "legacy_fdp_schedule_bytes_frozen",
        ],
        "branches": {
            "R7_long_context": {
                "status": "protocol_defined_pending_data_audit",
                "calibration": {"n": 16, "seqlen": 8192, "total_tokens": 131072},
                "methods": ["dense", "uniform", "fdp_2k", "fdp_8k", "spectral", "middle"],
            },
            "R8_system_metrics": {
                "status": "protocol_defined",
                "exclusive_gpu": True,
                "lengths": [[2048, 128], [8192, 128]],
                "warmup": 10,
                "repeat": 30,
            },
            "R9_operating_points": {
                "status": "protocol_defined_pending_variant_recovery",
                "points": {"mild": 0.820, "moderate": 0.739, "aggressive": 0.589},
            },
            "R10_kq_v_ablation": {"status": "protocol_defined"},
            "R11_robustness": {
                "status": "protocol_defined",
                "cal_sizes": [16, 32, 64],
                "lambdas": [0.6, 1.2, 2.0],
            },
        },
        "note": "Quality branches must not start before legacy FDP parity. Protocols frozen before reading quality results.",
    }
    atomic_write_json(supp / "SUPPLEMENTARY_EXPERIMENT_PROTOCOL.json", proto)
    (supp / "SUPPLEMENTARY_EXPERIMENT_PROTOCOL.sha256").write_text(
        sha256_file(supp / "SUPPLEMENTARY_EXPERIMENT_PROTOCOL.json") + "\n", encoding="utf-8"
    )
    gates = {
        "generated_at": now_iso(),
        "gates": {
            "R7_LONG_CONTEXT_PROTOCOL_FROZEN": False,
            "R8_SYSTEM_BENCHMARK_PROTOCOL_FROZEN": True,
            "R9_OPERATING_POINT_PROTOCOL_FROZEN": False,
            "R10_KQ_V_ABLATION_PROTOCOL_FROZEN": True,
            "R11_ROBUSTNESS_PROTOCOL_FROZEN": True,
            "SUPPLEMENTARY_EXPERIMENTS_COMPLETE": False,
        },
        "blocked_until": "R6_STATISTICS_PASS",
    }
    atomic_write_json(supp / "SUPPLEMENTARY_STAGE_GATES.json", gates)
    (supp / "SUPPLEMENTARY_JOBS.jsonl").write_text("", encoding="utf-8")
    progress = {
        "generated_at": now_iso(),
        "phase": "waiting_mainline_R1_R6",
        "branches": {k: {"status": "pending", "jobs": 0} for k in proto["branches"]},
    }
    atomic_write_json(supp / "SUPPLEMENTARY_PROGRESS.json", progress)
    (supp / "SUPPLEMENTARY_CURRENT_STATUS.md").write_text(
        f"# Supplementary Current Status\n\n- generated_at: `{now_iso()}`\n"
        "- waiting for mainline R1–R6 gates before quality branches.\n"
        "- system metrics may run after dense/uniform/fdp deploy artifacts exist.\n",
        encoding="utf-8",
    )
    # long-context data audit (local only; shallow known roots — no whole-disk crawl)
    blocked = True
    hits = []
    candidates = [
        Path("/home/bai-project/jiaqizhang/RULER"),
        Path("/home/bai-project/jiaqizhang/LongBench"),
        Path("/home/bai-project/jiaqizhang/ruler"),
        Path("/home/bai-project/jiaqizhang/longbench"),
        Path("/diskD/jiaqizhang/RULER"),
        Path("/diskD/jiaqizhang/LongBench"),
        Path("/diskE/jiaqizhang/RULER"),
        Path("/diskE/jiaqizhang/LongBench"),
        Path("/diskF/jiaqizhang/RULER"),
        Path("/diskF/jiaqizhang/LongBench"),
        Path("/home/bai-project/RULER"),
        Path("/home/bai-project/LongBench"),
    ]
    for p in candidates:
        if p.exists():
            hits.append(str(p))
    if not hits:
        (supp / "BLOCKED_NO_OFFICIAL_LONG_CONTEXT_DATA.md").write_text(
            "# BLOCKED_NO_OFFICIAL_LONG_CONTEXT_DATA\n\n"
            f"- generated_at: `{now_iso()}`\n"
            "- No reliable local official RULER/LongBench implementation found.\n"
            "- Will not fabricate tasks. Other supplementary branches continue independently.\n",
            encoding="utf-8",
        )
    else:
        blocked = False
        atomic_write_json(supp / "LONG_CONTEXT_DATA_HITS.json", {"hits": hits, "generated_at": now_iso()})
    gates["gates"]["R7_LONG_CONTEXT_PROTOCOL_FROZEN"] = (not blocked) and True
    # keep False if blocked
    if blocked:
        gates["gates"]["R7_LONG_CONTEXT_PROTOCOL_FROZEN"] = False
    atomic_write_json(supp / "SUPPLEMENTARY_STAGE_GATES.json", gates)


def smoke_tmux() -> Dict[str, Any]:
    name = SESSION_NAME + "_smoke"
    subprocess.run(["tmux", "kill-session", "-t", name], check=False, capture_output=True)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", name, f"{shlex.quote(PYTHON_BIN)} -V && sleep 1"],
        check=True,
    )
    time.sleep(0.5)
    has = subprocess.run(["tmux", "has-session", "-t", name], capture_output=True)
    subprocess.run(["tmux", "kill-session", "-t", name], check=False, capture_output=True)
    return {"has_session_rc": has.returncode, "passed": has.returncode == 0}


def job_dir(job_id: str) -> Path:
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
    min_free_mib: int = MIN_FREE_MIB_DEFAULT,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    od = job_dir(job_id)
    manifest = {
        "job_id": job_id,
        "stage": stage,
        "kind": kind,
        "command": command,
        "cwd": cwd or str(ROOT / "EigenAttn"),
        "preferred_gpu": preferred_gpu,
        "depends_on": depends_on or [],
        "min_free_mib": min_free_mib,
        "created_at": now_iso(),
        "meta": meta or {},
        "input_hashes": {
            "protocol": file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
            "evaluation": file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
            "ppl_runner": file_hash_or_empty(HERE / "runners" / "ppl_instrumented.py"),
            "zs_runner": file_hash_or_empty(HERE / "runners" / "zero_shot_instrumented.py"),
        },
    }
    atomic_write_json(od / "job_manifest.json", manifest)
    atomic_write_json(
        od / "job_state.json",
        {"job_id": job_id, "status": "PENDING", "updated_at": now_iso()},
    )
    return {
        "job_id": job_id,
        "stage": stage,
        "kind": kind,
        "status": "PENDING",
        "command": command,
        "cwd": manifest["cwd"],
        "preferred_gpu": preferred_gpu,
        "depends_on": depends_on or [],
        "min_free_mib": min_free_mib,
        "output_dir": str(od),
        "meta": meta or {},
    }


def build_r1_jobs() -> List[Dict[str, Any]]:
    eval_hash = file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json")
    jobs = []
    # GPU0 WikiText PPL
    jobs.append(
        make_job(
            "R1_dense_wikitext_ppl",
            "R1",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "ppl_instrumented.py"),
                "--dataset",
                "wikitext2",
                "--model",
                MODEL,
                "--tok",
                MODEL,
                "--out_dir",
                str(HERE / "artifacts" / "R1_dense_wikitext_ppl"),
                "--wikitext2_dir",
                WIKITEXT,
                "--dtype",
                "bf16",
                "--method",
                "dense",
                "--evaluation_protocol_hash",
                eval_hash,
            ],
            preferred_gpu=0,
            meta={"dataset": "wikitext2", "method": "dense"},
        )
    )
    # GPU1 C4 PPL
    jobs.append(
        make_job(
            "R1_dense_c4_ppl",
            "R1",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "ppl_instrumented.py"),
                "--dataset",
                "c4",
                "--model",
                MODEL,
                "--tok",
                MODEL,
                "--c4_texts",
                C4_TEXTS,
                "--out_dir",
                str(HERE / "artifacts" / "R1_dense_c4_ppl"),
                "--dtype",
                "bf16",
                "--method",
                "dense",
                "--evaluation_protocol_hash",
                eval_hash,
            ],
            preferred_gpu=1,
            meta={"dataset": "c4", "method": "dense"},
        )
    )
    # GPU2 HS + ARC-C + ARC-E
    jobs.append(
        make_job(
            "R1_dense_zs_hs_arc",
            "R1",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "zero_shot_instrumented.py"),
                "--dense_base",
                "--model",
                MODEL,
                "--tokenizer_model",
                MODEL,
                "--task_yaml_dir",
                TASK_YAML,
                "--tasks",
                "hellaswag_localjsonl,arc_challenge_localjsonl,arc_easy_localjsonl",
                "--out_dir",
                str(HERE / "artifacts" / "R1_dense_zs_hs_arc"),
                "--dtype",
                "bf16",
                "--net",
                "llama-3-8b",
                "--method",
                "dense",
            ],
            preferred_gpu=2,
            meta={"tasks": "hs_arc", "method": "dense"},
        )
    )
    # GPU3 WG + PIQA
    jobs.append(
        make_job(
            "R1_dense_zs_wg_piqa",
            "R1",
            "gpu",
            [
                PYTHON_BIN,
                str(HERE / "runners" / "zero_shot_instrumented.py"),
                "--dense_base",
                "--model",
                MODEL,
                "--tokenizer_model",
                MODEL,
                "--task_yaml_dir",
                TASK_YAML,
                "--tasks",
                "winogrande_localjsonl,piqa_localjsonl",
                "--out_dir",
                str(HERE / "artifacts" / "R1_dense_zs_wg_piqa"),
                "--dtype",
                "bf16",
                "--net",
                "llama-3-8b",
                "--method",
                "dense",
            ],
            preferred_gpu=3,
            meta={"tasks": "wg_piqa", "method": "dense"},
        )
    )
    # Aggregate CPU job after 4 DONE
    jobs.append(
        make_job(
            "R1_dense_aggregate",
            "R1",
            "cpu",
            [PYTHON_BIN, str(HERE / "runners" / "r1_aggregate.py")],
            depends_on=[
                "R1_dense_wikitext_ppl",
                "R1_dense_c4_ppl",
                "R1_dense_zs_hs_arc",
                "R1_dense_zs_wg_piqa",
            ],
            meta={"role": "aggregate"},
        )
    )
    return jobs


def write_jobs(jobs: List[Dict[str, Any]]) -> None:
    path = HERE / "jobs.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")


def load_jobs() -> List[Dict[str, Any]]:
    path = HERE / "jobs.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_jobs(jobs: List[Dict[str, Any]]) -> None:
    write_jobs(jobs)


def job_counts(jobs: List[Dict[str, Any]]) -> Dict[str, int]:
    c = {"completed": 0, "running": 0, "pending": 0, "blocked": 0, "failed": 0, "stopped": 0}
    for j in jobs:
        s = j.get("status", "PENDING").upper()
        if s == "DONE":
            c["completed"] += 1
        elif s == "RUNNING":
            c["running"] += 1
        elif s == "FAILED":
            c["failed"] += 1
        elif s == "STOPPED":
            c["stopped"] += 1
        elif s == "BLOCKED":
            c["blocked"] += 1
        else:
            c["pending"] += 1
    return c


def deps_satisfied(job: Dict[str, Any], jobs_by_id: Dict[str, Dict[str, Any]]) -> bool:
    for d in job.get("depends_on") or []:
        dep = jobs_by_id.get(d)
        if not dep or dep.get("status") != "DONE":
            return False
        # require DONE.json exists
        done = Path(dep["output_dir"]) / "DONE.json"
        if not done.exists():
            return False
    return True


def execute_one_job(job: Dict[str, Any], locks: GPULockManager) -> Dict[str, Any]:
    od = Path(job["output_dir"])
    done = od / "DONE.json"
    failed = od / "FAILED.json"
    # resume skip
    if done.exists() and job.get("status") == "DONE":
        man = json.loads((od / "job_manifest.json").read_text(encoding="utf-8"))
        if man.get("input_hashes") == job.get("meta", {}).get("_skip_check"):
            pass
        # hash-consistent skip if DONE present and manifest hashes match current runners
        cur_hashes = {
            "protocol": file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
            "evaluation": file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
            "ppl_runner": file_hash_or_empty(HERE / "runners" / "ppl_instrumented.py"),
            "zs_runner": file_hash_or_empty(HERE / "runners" / "zero_shot_instrumented.py"),
        }
        if man.get("input_hashes") == cur_hashes:
            job["status"] = "DONE"
            return job

    if stop_requested():
        job["status"] = "STOPPED"
        atomic_write_json(od / "job_state.json", {"status": "STOPPED", "at": now_iso()})
        return job

    gpu_info = None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        if job.get("kind") == "gpu":
            gpu_info = locks.wait_acquire(
                job["job_id"],
                preferred_index=job.get("preferred_gpu"),
                stop_fn=stop_requested,
            )
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_info["gpu_index"])

        job["status"] = "RUNNING"
        atomic_write_json(
            od / "job_state.json",
            {"status": "RUNNING", "started_at": now_iso(), "gpu": gpu_info},
        )
        stdout_p = od / "stdout.log"
        stderr_p = od / "stderr.log"
        hb_p = od / "heartbeat.json"
        with stdout_p.open("a", encoding="utf-8") as fout, stderr_p.open("a", encoding="utf-8") as ferr:
            proc = subprocess.Popen(
                job["command"],
                cwd=job["cwd"],
                stdout=fout,
                stderr=ferr,
                env=env,
                text=True,
            )
            start = time.time()
            while True:
                atomic_write_json(
                    hb_p,
                    {
                        "job_id": job["job_id"],
                        "status": "RUNNING",
                        "pid": proc.pid,
                        "gpu": gpu_info,
                        "running_for_sec": time.time() - start,
                        "updated_at": now_iso(),
                    },
                )
                rc = proc.poll()
                if rc is not None:
                    break
                if stop_requested():
                    # wait for atomic unit; do not kill -9
                    pass
                time.sleep(HEARTBEAT_SECS)
        result = {
            "job_id": job["job_id"],
            "returncode": rc,
            "ended_at": now_iso(),
            "gpu": gpu_info,
            "elapsed_sec": time.time() - start,
        }
        if rc == 0:
            atomic_write_json(done, result)
            job["status"] = "DONE"
            atomic_write_json(od / "job_state.json", {"status": "DONE", **result})
        else:
            atomic_write_json(failed, result)
            job["status"] = "FAILED"
            atomic_write_json(od / "job_state.json", {"status": "FAILED", **result})
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        atomic_write_json(failed, {"error": repr(e), "traceback": tb, "at": now_iso()})
        atomic_write_json(od / "job_state.json", {"status": "FAILED", "error": repr(e), "at": now_iso()})
        job["status"] = "FAILED"
        (od / "stderr.log").open("a", encoding="utf-8").write(tb + "\n")
    finally:
        if gpu_info:
            locks.release(gpu_info["gpu_uuid"], job_id=job["job_id"])
    return job


def bind_runners_gate() -> Dict[str, Any]:
    required = [
        HERE / "runners" / "ppl_instrumented.py",
        HERE / "runners" / "zero_shot_instrumented.py",
        HERE / "runners" / "parity_gate.py",
        HERE / "runners" / "r1_aggregate.py",
        HERE / "gpu_scheduler.py",
        HERE / "common.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    payload = gate_payload(
        name="DENSE_RUNNERS_BOUND",
        passed=len(missing) == 0,
        upstream_hashes={p.name: file_hash_or_empty(p) for p in required},
        runner_hash=sha256_file(HERE / "legacy_rebuttal_orchestrator.py"),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="production runners bound" if not missing else f"missing: {missing}",
        extra={"missing": missing},
    )
    atomic_write_json(HERE / "gates" / "DENSE_RUNNERS_BOUND.json", payload)
    if payload["passed"]:
        write_pass_gate(HERE / "DENSE_RUNNERS_BOUND.DONE", payload)
    (HERE / "LEGACY_RUNNER_BINDING_AUDIT.md").write_text(
        "# Legacy Runner Binding Audit\n\n"
        f"- generated_at: `{now_iso()}`\n"
        f"- passed: `{payload['passed']}`\n"
        + "\n".join(f"- `{p}` sha256=`{file_hash_or_empty(p)}`" for p in required)
        + "\n",
        encoding="utf-8",
    )
    return payload


def write_plan_and_audits() -> None:
    (HERE / "STAGE_R1_R3_EXECUTION_PLAN.md").write_text(
        "\n".join(
            [
                "# Stage R1–R3 Execution Plan",
                "",
                f"- generated_at: `{now_iso()}`",
                "",
                "## R1 Dense (4-GPU parallel)",
                "- GPU0: WikiText-2 PPL",
                "- GPU1: C4 PPL",
                "- GPU2: HellaSwag, ARC-Challenge, ARC-Easy",
                "- GPU3: WinoGrande, PIQA",
                "- Aggregate after all DONE → DENSE_PARITY_GATE",
                "",
                "## R2 Uniform",
                "- Build/audit single GPU (default 0)",
                "- Eval split like R1 across 4 GPUs",
                "",
                "## R3 Legacy FDP",
                "- Step1 single GPU → per-layer EB CPU → adaptive basis single GPU → eval 4-GPU",
                "",
                "## Forbidden until parity",
                "- Qwen/OPT/8K/operating-points/KQ-V/robustness/matched-controls",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (HERE / "ATOMIC_RESUME_AUDIT.md").write_text(
        f"# Atomic Resume Audit\n\n- generated_at: `{now_iso()}`\n"
        "- Per-window/sample: `.tmp`→fsync→rename→`.DONE`\n"
        "- Job skip only if DONE.json and input hashes match.\n"
        "- STOP_REQUESTED: no kill -9; finish atomic unit.\n",
        encoding="utf-8",
    )
    (HERE / "TMUX_DETACH_AUDIT.md").write_text(
        f"# Tmux Detach Audit\n\n- generated_at: `{now_iso()}`\n"
        f"- session: `{SESSION_NAME}`\n"
        f"- python: `{PYTHON_BIN}`\n"
        "- master launched detached; Codex/SSH/VSCode exit must not stop it.\n"
        f"- log: `{HERE / 'background_logs' / 'orchestrator.log'}`\n",
        encoding="utf-8",
    )
    (HERE / "RAW_OUTPUT_INTERFACE_AUDIT.md").write_text(
        f"# RAW Output Interface Audit\n\n- generated_at: `{now_iso()}`\n"
        "- See gates/raw_output_parity and runners/*_instrumented.py\n"
        "- Original EigenAttn sources remain read-only.\n",
        encoding="utf-8",
    )


def engineering_bootstrap() -> Dict[str, Any]:
    ensure_layout()
    freeze_protocol()
    freeze_supplementary_protocol()
    write_plan_and_audits()
    tmux = smoke_tmux()
    inv = write_inventory()
    write_scheduler_audit()
    bind = bind_runners_gate()

    # run parity gate (includes optional llama subset on GPU0)
    parity_cmd = [
        PYTHON_BIN,
        str(HERE / "runners" / "parity_gate.py"),
        "--out_dir",
        str(HERE / "gates" / "raw_output_parity"),
        "--llama_limit",
        "2",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")
    env["PARITY_GPU"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cp = subprocess.run(parity_cmd, cwd=str(ROOT / "EigenAttn"), env=env, capture_output=True, text=True)
    parity_ok = cp.returncode == 0 and (HERE / "RAW_OUTPUT_INTERFACE_PASS.DONE").exists()
    (HERE / "background_logs" / "parity_gate.stdout.log").write_text(cp.stdout, encoding="utf-8")
    (HERE / "background_logs" / "parity_gate.stderr.log").write_text(cp.stderr, encoding="utf-8")

    # SOURCE_AND_PROTOCOL_LOCK_PASS
    proto_hash = file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json")
    source_pass = gate_payload(
        name="SOURCE_AND_PROTOCOL_LOCK_PASS",
        passed=True,
        upstream_hashes={"protocol": proto_hash, "model_config": file_hash_or_empty(Path(MODEL) / "config.json")},
        runner_hash=sha256_file(HERE / "legacy_rebuttal_orchestrator.py"),
        protocol_hash=proto_hash,
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="protocol frozen with provenance; inferred_only not treated as truth",
    )
    write_pass_gate(HERE / "SOURCE_AND_PROTOCOL_LOCK_PASS.DONE", source_pass)

    gates = {
        "SOURCE_AND_PROTOCOL_LOCK_PASS": True,
        "RAW_OUTPUT_INTERFACE_PASS": parity_ok,
        "DENSE_RUNNERS_BOUND": bool(bind.get("passed")),
        "tmux_smoke": bool(tmux.get("passed")),
        "gpu_inventory": len(inv.get("gpus", [])) >= 4,
        "model_exists": Path(MODEL).exists(),
        "c4_exists": Path(C4_TEXTS).exists(),
        "wikitext_exists": Path(WIKITEXT).exists(),
        "task_yaml_exists": Path(TASK_YAML).exists(),
    }
    ready = all(
        gates[k]
        for k in (
            "SOURCE_AND_PROTOCOL_LOCK_PASS",
            "RAW_OUTPUT_INTERFACE_PASS",
            "DENSE_RUNNERS_BOUND",
            "tmux_smoke",
            "gpu_inventory",
            "model_exists",
            "c4_exists",
            "wikitext_exists",
            "task_yaml_exists",
        )
    )
    atomic_write_json(
        HERE / "stage_gates.json",
        {"generated_at": now_iso(), "gates": gates, "ready_for_gpu": ready, "parity_stdout_tail": cp.stdout[-1500:], "parity_stderr_tail": cp.stderr[-1500:]},
    )

    if ready:
        jobs = build_r1_jobs()
        write_jobs(jobs)
        approve = {
            "approved_at": now_iso(),
            "gates": gates,
            "first_stage": "R1_DENSE",
            "jobs": [j["job_id"] for j in jobs],
            "python": PYTHON_BIN,
        }
        atomic_write_json(HERE / "APPROVE_GPU_START", approve)
        if (HERE / "START_BLOCKED_REPORT.md").exists():
            (HERE / "START_BLOCKED_REPORT.md").unlink()
    else:
        # single hardest blocker
        order = [
            ("RAW_OUTPUT_INTERFACE_PASS", "raw-output parity gate failed — see gates/raw_output_parity and background_logs/parity_gate.stderr.log"),
            ("DENSE_RUNNERS_BOUND", "production runners not bound"),
            ("tmux_smoke", "tmux smoke failed"),
            ("gpu_inventory", "need 4 A100 GPUs"),
            ("model_exists", f"missing model {MODEL}"),
            ("c4_exists", f"missing C4 {C4_TEXTS}"),
            ("wikitext_exists", f"missing wikitext {WIKITEXT}"),
            ("task_yaml_exists", f"missing task yaml dir {TASK_YAML}"),
        ]
        blocker = next((msg for k, msg in order if not gates.get(k)), "unknown")
        (HERE / "START_BLOCKED_REPORT.md").write_text(
            f"# START BLOCKED REPORT\n\n- generated_at: `{now_iso()}`\n"
            f"- hard_blocker: `{blocker}`\n\n## Gates\n\n"
            + "\n".join(f"- `{k}`: `{v}`" for k, v in gates.items())
            + "\n",
            encoding="utf-8",
        )
        write_jobs([])

    state = {
        "generated_at": now_iso(),
        "phase": "engineering_bootstrap_complete",
        "blocked": not ready,
        "last_action": "engineering_bootstrap",
        "gates": gates,
        "master_pid": os.getpid(),
        "job_counts": job_counts(load_jobs()),
        "gpu_jobs": {},
        "python": PYTHON_BIN,
    }
    write_status(state)
    return state


def run_loop() -> None:
    ensure_layout()
    log_path = HERE / "background_logs" / "orchestrator.log"
    locks = GPULockManager(min_free_mib=0)  # share GPUs; no free-VRAM gate
    state = {
        "generated_at": now_iso(),
        "phase": "run_loop",
        "blocked": False,
        "last_action": "run_loop_start",
        "master_pid": os.getpid(),
        "gates": json.loads((HERE / "stage_gates.json").read_text(encoding="utf-8")).get("gates", {})
        if (HERE / "stage_gates.json").exists()
        else {},
        "gpu_jobs": {},
        "python": PYTHON_BIN,
    }
    write_status(state)

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{now_iso()} {msg}\n")

    log("master run_loop started")
    idle_rounds = 0
    while True:
        if stop_requested():
            state["phase"] = "stop_requested"
            state["last_action"] = "stop_requested"
            write_status(state)
            log("STOP_REQUESTED observed; exiting after draining")
            break

        jobs = load_jobs()
        # Materialize future stages based on gates (never re-run completed R1)
        before = {j["job_id"] for j in jobs}
        jobs = materialize_after_gates(jobs)
        after = {j["job_id"] for j in jobs}
        if after - before:
            log(f"materialized jobs: {sorted(after - before)}")
            save_jobs(jobs)

        by_id = {j["job_id"]: j for j in jobs}
        runnable = []
        for j in jobs:
            if j.get("status") in ("DONE", "FAILED", "STOPPED", "RUNNING"):
                continue
            if not deps_satisfied(j, by_id):
                # mark blocked if dependency failed
                for d in j.get("depends_on") or []:
                    if by_id.get(d, {}).get("status") == "FAILED":
                        j["status"] = "BLOCKED"
                continue
            blocked = False
            for d in j.get("depends_on") or []:
                if by_id.get(d, {}).get("status") == "FAILED":
                    j["status"] = "BLOCKED"
                    blocked = True
            if blocked:
                continue
            runnable.append(j)

        save_jobs(jobs)

        if not runnable and not any(j.get("status") == "RUNNING" for j in jobs):
            if any(j.get("status") == "FAILED" for j in jobs) and not future_stages_remain(jobs):
                state["phase"] = "failed_closed"
                state["last_action"] = "unrecoverable_failed"
                state["job_counts"] = job_counts(jobs)
                write_status(state)
                log("unrecoverable FAILED with no future stages; exiting")
                break
            if future_stages_remain(jobs):
                idle_rounds += 1
                state["phase"] = "WAITING"
                state["last_action"] = "waiting_for_gates_or_deps"
                state["job_counts"] = job_counts(jobs)
                write_status(state)
                if idle_rounds == 1 or idle_rounds % 12 == 0:
                    log("WAITING: no runnable jobs but future stages remain")
                # Rematerialize periodically in case gates appeared
                time.sleep(10)
                continue
            # all complete
            state["phase"] = "all_current_jobs_finished"
            state["last_action"] = "idle_complete"
            state["job_counts"] = job_counts(jobs)
            write_status(state)
            log("all current jobs finished and no future stages; exiting")
            break

        idle_rounds = 0
        gpu_runnable = [j for j in runnable if j.get("kind") == "gpu"][:4]
        cpu_runnable = [j for j in runnable if j.get("kind") == "cpu"][:2]
        batch = gpu_runnable + cpu_runnable
        if not batch:
            time.sleep(5)
            continue

        for b in batch:
            for i, j in enumerate(jobs):
                if j["job_id"] == b["job_id"]:
                    jobs[i]["status"] = "RUNNING"
        save_jobs(jobs)
        state["job_counts"] = job_counts(jobs)
        state["phase"] = "RUNNING"
        state["gpu_jobs"] = {
            str(b.get("preferred_gpu")): b["job_id"] for b in batch if b.get("kind") == "gpu"
        }
        state["last_action"] = f"launch:{','.join(b['job_id'] for b in batch)}"
        write_status(state)
        log(f"launching batch: {[b['job_id'] for b in batch]}")

        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futs = {ex.submit(execute_one_job, b, locks): b["job_id"] for b in batch}
            for fut in as_completed(futs):
                updated = fut.result()
                jobs = load_jobs()
                for i, j in enumerate(jobs):
                    if j["job_id"] == updated["job_id"]:
                        jobs[i] = updated
                # rematerialize after each completion (e.g. R1 aggregate -> R2)
                jobs = materialize_after_gates(jobs)
                save_jobs(jobs)
                log(f"job finished {updated['job_id']} status={updated['status']}")
                state["job_counts"] = job_counts(jobs)
                state["last_action"] = f"finished:{updated['job_id']}"
                write_status(state)
                if updated["status"] == "FAILED":
                    log(f"FAIL-CLOSED signal from {updated['job_id']}")

    log("master run_loop exit")


def start_tmux() -> None:
    approve = HERE / "APPROVE_GPU_START"
    if not approve.exists():
        raise SystemExit("APPROVE_GPU_START missing; see START_BLOCKED_REPORT.md")
    # do not kill existing healthy session
    has = subprocess.run(["tmux", "has-session", "-t", SESSION_NAME], capture_output=True)
    if has.returncode == 0:
        print(f"session {SESSION_NAME} already running")
        return
    log = HERE / "background_logs" / "orchestrator.log"
    cmd = (
        f"{shlex.quote(PYTHON_BIN)} -u {shlex.quote(str(HERE / 'legacy_rebuttal_orchestrator.py'))} "
        f"run-loop --project-root {shlex.quote(str(ROOT))} >> {shlex.quote(str(log))} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION_NAME, cmd], check=True)
    print(f"started {SESSION_NAME}")


def monitor() -> None:
    state_f = HERE / "orchestrator_state.json"
    print(state_f.read_text(encoding="utf-8") if state_f.exists() else "{}")
    print("--- tmux ---")
    subprocess.run(["tmux", "ls"], check=False)
    print("--- GPU locks ---")
    lf = HERE / "GPU_RESOURCE_LOCKS.json"
    print(lf.read_text(encoding="utf-8") if lf.exists() else "{}")
    print("--- jobs ---")
    jobs = load_jobs()
    print(json.dumps(job_counts(jobs), ensure_ascii=False))
    for j in jobs:
        print(f"{j['job_id']}: {j.get('status')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command",
        choices=["audit", "bootstrap", "start", "run-loop", "monitor", "freeze-protocol"],
    )
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()
    ensure_layout()
    if args.command in ("audit", "bootstrap", "freeze-protocol"):
        # audit now means full engineering bootstrap
        engineering_bootstrap()
    elif args.command == "start":
        start_tmux()
    elif args.command == "run-loop":
        run_loop()
    elif args.command == "monitor":
        monitor()


if __name__ == "__main__":
    main()
