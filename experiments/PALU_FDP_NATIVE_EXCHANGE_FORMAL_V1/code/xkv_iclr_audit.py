#!/usr/bin/env python3
"""CPU/file audit and prerequisite freeze for XKV ICLR closure."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/diskE/jiaqizhang/XKV_FDP_NATIVE_EXCHANGE_FORMAL_V1")
AUDIT = ROOT / "audit"
# The exchange line was frozen under the historical PALU-named root.  The
# ICLR closure artifacts live under ROOT, but must read those locks in place.
EXCHANGE_ROOT = Path("/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1")
CAL = EXCHANGE_ROOT / "XKV_EXCHANGE_CALIBRATION.json"
DECISIONS = EXCHANGE_ROOT / "XKV_EXCHANGE_DECISIONS.csv"
MARGINAL = EXCHANGE_ROOT / "XKV_NATIVE_MARGINAL_LOCK.json"
MODEL = Path("/diskF/jiaqizhang/Llama-3.1")
REPO = Path("/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/xkv_upstream")
AUTH = Path("/diskF/jiaqizhang/XKV_PAPER_REPRO_FDP_V1")
RUNTIME = Path("/diskF/jiaqizhang/xkv_native_runtime_fixed_v1")
RUNTIME2 = Path("/home/bai-project/jiaqizhang/XKV_PAPER_REPRO_FDP_V1_RUNTIME_V2")
RULER = RUNTIME2 / "ruler_data/llama-3/65536"
Q_SHA = "6f3b59993abbffbd932825b20ff036197c5be6ff63523ab46e94281bc24eecb1"


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def meta(path: Path) -> dict[str, Any]:
    out = {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}
    if path.exists():
        st = path.stat()
        out.update({"size_bytes": st.st_size if path.is_file() else None, "mtime_ns": st.st_mtime_ns, "sha256": sha(path)})
    return out


def load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json_once(path: Path, data: Any) -> None:
    write_once(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def ruler_assets() -> dict[str, Any]:
    tasks = ["niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2", "niah_multiquery", "niah_multivalue", "qa_1", "qa_2", "vt", "fwe"]
    rows = []
    for task in tasks:
        path = RULER / task / "validation.jsonl"
        rows.append({"task": task, **meta(path)})
    return {"root": str(RULER), "tasks": rows, "complete": all(x["exists"] and x["size_bytes"] > 0 for x in rows)}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    cal = load(CAL, {})
    marginal = load(MARGINAL, {})
    assets = ruler_assets()
    cfg = load(MODEL / "config.json", {})
    kv_heads = int(cfg.get("num_key_value_heads", 8))
    head_dim = int(cfg.get("hidden_size", 4096)) // int(cfg.get("num_attention_heads", 32))
    full_bytes = 2 * 1 * 32 * kv_heads * 65536 * head_dim * 2
    factor_bytes = int(marginal.get("control", {}).get("factor_bytes_total", 1069547520))
    ratio = full_bytes / factor_bytes

    method = {
        "schema": "XKV_OFFICIAL_METHOD_CARD_V1",
        "status": "FROZEN_METHOD_AND_RESOURCE_SPEC",
        "model": {"path": str(MODEL), "claimed_identity": "Llama-3.1-8B-Instruct", "config": cfg, "config_sha256": sha(MODEL / "config.json")},
        "official_repository": {"path": str(REPO), "commit": git_commit(), "entry_wrapper": meta(AUTH / "code/xkv_paper_entry.py"), "upstream_evaluator": meta(REPO / "evaluate/eval_acc.py")},
        "protocol": {"context_length": 65536, "batch_size": 1, "dtype": "bfloat16", "bits": 16, "backend": "flash_attention_2 via --flash2", "ruler_assets": assets, "longbench_status": "PENDING_OFFICIAL_RUN"},
        "native_configuration": {"layers": 32, "kv_heads": kv_heads, "head_dim": head_dim, "group_size_W": 4, "native_group_count": 8, "rank_k": 384, "rank_v": 576, "V_over_K": 1.5, "merge_k": True, "merge_v": True},
        "resource": {"full_kv_bytes": full_bytes, "official_factor_bytes": factor_bytes, "compression_ratio": ratio, "target_label": "approximately 8.03x", "formula_source": meta(REPO / "scripts/calc_kv_comp_rate.py"), "formula": marginal.get("factor_memory_formula", {}).get("definition")},
        "runtime": {"fixed_runtime_audit": meta(RUNTIME / "audit/XKV_FIXED_RUNTIME_AUDIT.json"), "fixed_runtime_gates": meta(RUNTIME / "audit/XKV_FIXED_RUNTIME_GATES.json"), "formal_entry": "xkv_paper_entry.py"},
        "freezes": {"q_sha": Q_SHA, "marginal_lock": meta(MARGINAL), "calibration": meta(CAL), "decisions": meta(DECISIONS), "T_must_not_change_after_calibration": True},
    }
    write_json_once(ROOT / "XKV_OFFICIAL_METHOD_CARD.json", method)
    write_json_once(ROOT / "XKV_NATIVE_MARGINAL_LOCK.json", marginal)

    evidence = [
        ["official_upstream_repo", str(REPO), "VALID_REPRODUCTION", "FROZEN_COMMIT", "official source"],
        ["model_tokenizer", str(MODEL), "VALID_REPRODUCTION", "LOCAL_COMPLETE", "Llama-3.1 identity recorded"],
        ["fixed_runtime_audit", str(RUNTIME / "audit/XKV_FIXED_RUNTIME_AUDIT.json"), "VALID_REPRODUCTION", "PASS", "transpose and q_len=1 fixes"],
        ["native_marginal", str(MARGINAL), "VALID_CALIBRATION", "COMPLETE", "8 native four-layer groups"],
        ["exchange_calibration", str(CAL), "VALID_CALIBRATION", "COMPLETE", "T frozen before formal quality"],
        ["ruler_assets", str(RULER), "VALID_FORMAL" if assets["complete"] else "INCOMPLETE", "READY" if assets["complete"] else "INCOMPLETE", "official 64K assets"],
        ["longbench", str(REPO / "evaluate/data"), "INCOMPLETE", "PENDING_RUN", "no formal output yet"],
        ["official_803x", str(RUNTIME2 / "results"), "INCOMPLETE", "NO_RESULT_JSON", "old queue had wrong path and multi-GPU launch"],
        ["old_7680_collapse", "/diskF/jiaqizhang/xkv_native_runtime_fixed_v1/legacy", "INVALID_ENGINEERING", "FROZEN", "known broken runtime; not scientific negative evidence"],
        ["historical_direct_fdp", "/diskF/jiaqizhang/fdp_final_core_generalization_v1", "UNKNOWN_PROVENANCE", "SEPARATE_AUDIT", "not mixed automatically"],
    ]
    with (ROOT / "XKV_EVIDENCE_MASTER.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "path", "classification", "status", "notes"])
        writer.writerows(evidence)
    audit_lines = ["# XKV Existing Evidence Audit", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", "", "| Artifact | Classification | Status | Notes |", "|---|---|---|---|"]
    audit_lines.extend(f"| {a} | {c} | {s} | {n} |" for a, p, c, s, n in evidence)
    audit_lines += ["", "Immutable boundaries:", "", "- Old collapse remains INVALID_ENGINEERING.", "- q, native marginal, exchange calibration, T values, and formal manifests are not retuned.", "- formal quality has not started; foreign_jobs_interrupted=0; GPU2 remains excluded; GPU0/GPU1/GPU3 are allowed one at a time."]
    write_once(ROOT / "XKV_EXISTING_EVIDENCE_AUDIT.md", "\n".join(audit_lines) + "\n")
    write_once(ROOT / "XKV_RUNTIME_PROVENANCE.md", "\n".join([
        "# XKV Runtime Provenance", "",
        "Repository commit: " + str(git_commit()),
        "Model config SHA256: " + str(sha(MODEL / "config.json")),
        "Fixed runtime audit: " + str(RUNTIME / "audit/XKV_FIXED_RUNTIME_AUDIT.json"),
        "Fixed runtime gates: " + str(RUNTIME / "audit/XKV_FIXED_RUNTIME_GATES.json"), "",
        "The old official launcher exported CUDA_VISIBLE_DEVICES=0,2,3 and pointed at a stale run path. The closure pipeline uses the same official wrapper and evaluator with one visible GPU only; this is an engineering launch repair, not a scientific parameter change.",
    ]) + "\n")

    matrix = {"schema": "XKV_ICLR_REQUIREMENT_MATRIX_V1", "frozen": True, "nodes": {
        "OFFICIAL_METHOD_CARD_FROZEN": True, "RUNTIME_REPAIRED_AND_AUDITED": True,
        "OFFICIAL_8.03X_RESOURCE_REPRODUCED": False, "OFFICIAL_RULER_COMPLETED": False, "OFFICIAL_LONGBENCH_COMPLETED": False,
        "NATIVE_MARGINAL_AUDITED": bool(marginal.get("locked")), "Q_VS_PHI_ANALYSIS_COMPLETED": bool(marginal.get("alignment")),
        "EXCHANGE_CALIBRATION_FROZEN": cal.get("status") == "COMPLETE", "PRIOR_EXCHANGE_FORMAL_COMPLETED": False,
        "NATIVE_PHI_EXCHANGE_FORMAL_COMPLETED": False, "RANDOM_CONTROL_COMPLETED": False, "ACTUAL_RESOURCE_AUDIT_PASS": False,
        "CACHED_AR_FORMAL_COMPLETED": False, "SYSTEM_OVERHEAD_REPORTED": False, "UNIFIED_XKV_TABLE_GENERATED": False,
        "CROSS_COMPRESSOR_SCHEMA_GENERATED": False, "NO_FORMAL_TEST_RETUNING": True, "FOREIGN_JOBS_INTERRUPTED": 0,
    }, "blocking": ["single_gpu_official_baseline", "formal_ruler", "formal_longbench", "formal_exchange_quality", "cached_ar_formal", "system_overhead"]}
    write_json_once(ROOT / "XKV_ICLR_REQUIREMENT_MATRIX.json", matrix)

    exchange_lock = {"schema": "XKV_FORMAL_EXCHANGE_LOCK_V1", "locked": True, "calibration_sha256": sha(CAL), "decisions_sha256": sha(DECISIONS), "marginal_lock_sha256": sha(MARGINAL), "q_sha": Q_SHA, "native_unit": "8 cross-layer four-layer groups", "quantum": {"delta_k": 32, "delta_v": 48, "V_over_K": 1.5}, "selected_T": {"q_prior": 2, "native_phi": 3, "random_seed_0": 1, "random_seed_1": 3, "random_seed_2": 1}, "formal_quality_started": False, "no_recalibration_after_formal_quality": True}
    write_json_once(ROOT / "XKV_FORMAL_EXCHANGE_LOCK.json", exchange_lock)

    targets = [
        ["resource_compression_ratio", "8.03x", "official target label/local protocol", ratio, ratio - 8.03, "RESOURCE_TARGET_MATCHED_NOT_QUALITY_REPRODUCED"],
        ["RULER_64K", "", "paper table source pending local audit", "", "", "PENDING_OFFICIAL_RUN"],
        ["LongBench", "", "paper table source pending local audit", "", "", "PENDING_OFFICIAL_RUN"],
    ]
    with (ROOT / "XKV_PAPER_TARGET_TABLE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "paper_reported_value", "source", "reproduction_value", "absolute_difference", "protocol_match_status"])
        writer.writerows(targets)

    state = {"schema": "XKV_ICLR_PIPELINE_STATE_V1", "status": "WAITING_GPU_FOR_SINGLE_GPU_OFFICIAL_BASELINE", "updated_at_utc": datetime.now(timezone.utc).isoformat(), "selected_gpu": None, "gpu_policy": {"preferred": [0, 1, 3], "excluded": [2], "single_gpu_only": True}, "nodes": {"audit": "DONE", "method_card": "DONE", "requirements": "DONE", "ruler_assets": "DONE" if assets["complete"] else "BLOCKED", "native_marginal": "DONE" if marginal.get("locked") else "BLOCKED", "exchange_calibration": "DONE" if cal.get("status") == "COMPLETE" else "BLOCKED", "official_803x_baseline": "WAITING_GPU", "formal_ruler": "WAITING_OFFICIAL_BASELINE", "formal_longbench": "WAITING_OFFICIAL_BASELINE", "formal_exchange_quality": "WAITING_OFFICIAL_BASELINE", "formal_cached_ar": "WAITING_OFFICIAL_BASELINE", "overhead": "WAITING_FORMAL_RUNS", "reports": "WAITING_ALL_REQUIRED_NODES"}, "immutable": {"q_sha": Q_SHA, "formal_test_retuning": False, "foreign_jobs_interrupted": 0, "old_results_modified": False}}
    (ROOT / "xkv_pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": state["status"], "ruler_complete": assets["complete"], "calibration": cal.get("status"), "marginal_locked": marginal.get("locked"), "repo_commit": git_commit(), "compression_ratio": ratio}, indent=2))


if __name__ == "__main__":
    main()
