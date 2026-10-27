#!/usr/bin/env python3
"""Merge verified completed uniform diagnostics into the campaign audit JSONs."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/diskE/jiaqizhang/PALU_OFFICIAL_TABLE1_REPRO_V1")
CELLS = (
    "J_LRD_GS32_UNIFORM",
    "G_LRD_GS4_UNIFORM",
    "M_LRD_GS1_UNIFORM",
    "G_LRD_GS16_UNIFORM",
    "G_LRD_GS8_UNIFORM",
    "G_LRD_GS2_UNIFORM",
)
TABLE6 = {
    "DENSE": 5.47,
    "J_LRD_GS32": 5.62,
    "G_LRD_GS16": 5.74,
    "G_LRD_GS8": 5.88,
    "G_LRD_GS4": 6.01,
    "G_LRD_GS2": 6.42,
    "M_LRD_GS1": 6.81,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def result_record(cell):
    path = ROOT / "uniform_diagnostics" / cell / "result.json"
    result = load(path)
    ref = TABLE6.get(cell.replace("_UNIFORM", ""))
    wiki = result["ppl"].get("wikitext2")
    return {
        "cell": cell,
        "status": "DONE",
        "method": result["method"],
        "group_size": result["group_size"],
        "allocator": result["allocator"],
        "classification": result["classification"],
        "ppl": result["ppl"],
        "table6_wiki_reference": ref,
        "wiki_absolute_error_vs_table6": abs(wiki - ref) if wiki is not None and ref is not None else None,
        "wiki_relative_error_vs_table6": abs(wiki - ref) / ref if wiki is not None and ref is not None else None,
        "resource": result["resource"],
        "result_sha256_field": result["result_sha256"],
        "result_file_sha256": sha(path),
        "result_path": str(path),
        "allocation_path": str(ROOT / "uniform_diagnostics" / cell / "allocation.json"),
    }


now = dt.datetime.now(dt.timezone.utc).isoformat()
results = {cell: result_record(cell) for cell in CELLS}
done = load(ROOT / "locks/UNIFORM_SWEEP_DONE/DONE.json")
state_path = ROOT / "REPRO_STATE.json"
state = load(state_path)
state["uniform_diagnostic_status"] = "COMPLETE_USER_REQUESTED_UNIFORM_NO_FISHER"
state["uniform_diagnostic_results"] = results
state["table6_group_size_status"].update({
    "status": "COMPLETE_USER_REQUESTED_UNIFORM_DIAGNOSTIC",
    "observed_wikitext2": {cell: results[cell]["ppl"]["wikitext2"] for cell in CELLS},
    "comparison_classification": "PROTOCOL_MISMATCH_TO_LIKELY_OFFICIAL_TABLE1_FISHER_UNIFORM",
    "sweep_status": done["status"],
    "completed_utc": done["finished_utc"],
})
state["p3_user_uniform_status"].update({
    "status": "COMPLETE_USER_REQUESTED_UNIFORM_DIAGNOSTIC",
    "results": {cell: results[cell] for cell in CELLS if cell in ("J_LRD_GS32_UNIFORM", "G_LRD_GS4_UNIFORM", "M_LRD_GS1_UNIFORM")},
    "fisher_enabled": False,
    "classification": "PROTOCOL_MISMATCH_TO_LIKELY_OFFICIAL_TABLE1_ALLOCATION",
    "completed_utc": done["finished_utc"],
})
state["worker"].update({
    "pid": done["pid"],
    "status": "COMPLETE",
    "current_cell": None,
    "progress": "All six frozen user-requested uniform diagnostics completed; two cache/device-placement failures repaired by retrying only affected cells.",
    "finished_utc": done["finished_utc"],
    "foreign_jobs_interrupted": 0,
})
state.setdefault("history", []).extend([
    {
        "event": "UNIFORM_SWEEP_COMPLETE_USER_REQUESTED_NO_FISHER",
        "utc": done["finished_utc"],
        "cells_done": list(CELLS),
        "fisher_enabled": False,
        "foreign_jobs_interrupted": 0,
        "old_results_valid": True,
    },
    {
        "event": "UNIFORM_RETRY_RUNTIME_REPAIRS",
        "utc": now,
        "failure_behavior": [
            "J-LRD and G-LRD gs4 initially failed at C4 PPL loading a stale incomplete 1194-byte torch archive without data.pkl.",
            "A subsequent retry placed the auto-mapped model across cuda:0/cuda:1 while evaluator inputs were on cuda:0, causing device mismatch for J-LRD and G-LRD gs4.",
        ],
        "repair": [
            "Preserved/quarantined only the corrupt C4 cache; official loader regenerated it and M-LRD/J-LRD/G-LRD gs4 C4 evaluation then succeeded.",
            "Retried J-LRD and G-LRD gs4 with CUDA_VISIBLE_DEVICES=0; this changed device placement only, not model, data, precision, allocation, scoring, or protocol.",
        ],
        "affected_cells": ["J_LRD_GS32_UNIFORM", "G_LRD_GS4_UNIFORM"],
        "old_results_valid": True,
        "foreign_jobs_interrupted": 0,
    },
])
state["foreign_jobs_interrupted"] = 0
atomic_json(state_path, state)

trace_path = ROOT / "audit/TABLE1_OFFICIAL_TRACE.json"
trace = load(trace_path)
trace["user_requested_uniform_sweep"] = {
    "status": "COMPLETE",
    "allocator": "uniform",
    "fisher_enabled": False,
    "finished_utc": done["finished_utc"],
    "cells": results,
    "classification": "USER_REQUESTED_DIAGNOSTIC_NOT_EXACT_TABLE1",
    "foreign_jobs_interrupted": 0,
}
trace["table1_command_recovery"] = {
    "exact_table1_invocation_archived": False,
    "evidence": "Official repo default and README example use fisher_uniform; paper Sec.4.1 defines G-LRD gs4 equal rank within groups and reports Fisher calibration. No table-specific command/config found.",
    "user_constraint": "Fisher/Automatic disabled; resulting cells must remain protocol-mismatch diagnostics.",
}
atomic_json(trace_path, trace)

audit = {
    "schema": "PALU_TABLE1_RUNTIME_REPAIR_AUDIT_V1",
    "captured_utc": now,
    "failure_behavior": [
        {
            "type": "CORRUPT_STALE_TORCH_TESTLOADER_CACHE",
            "reason": "PytorchStreamReader failed locating file data.pkl; inspected archive contained only version/byteorder/serialization_id, 1194 bytes.",
            "affected_cells": ["J_LRD_GS32_UNIFORM", "G_LRD_GS4_UNIFORM"],
            "old_results_valid": True,
        },
        {
            "type": "CUDA_DEVICE_MAP_INPUT_MISMATCH",
            "reason": "device_map=auto placed modules across cuda:0 and cuda:1 while evaluation batches were explicitly moved to cuda:0.",
            "affected_cells": ["J_LRD_GS32_UNIFORM", "G_LRD_GS4_UNIFORM"],
            "old_results_valid": True,
        },
    ],
    "repairs": [
        {
            "action": "Moved the corrupt C4 cache unchanged into audit/quarantine; official loader regenerated a valid cache.",
            "quarantine_path": str(ROOT / "audit/quarantine/corrupt_c4_testloader_20260929.cache"),
            "quarantine_sha256": "657b17891378d24edd8eb0839fc0f0d9cf4dbcfa8de4d1a9f23a65f62c80a89b",
            "old_results_valid": True,
        },
        {
            "action": "Pinned retries to physical GPU0 with CUDA_VISIBLE_DEVICES=0; changed placement only.",
            "affected_cells": ["J_LRD_GS32_UNIFORM", "G_LRD_GS4_UNIFORM"],
            "retry_outcome": "both affected cells DONE with WikiText-2 and C4 PPL",
            "old_results_valid": True,
        },
    ],
    "foreign_jobs_interrupted": 0,
}
atomic_json(ROOT / "audit/UNIFORM_SWEEP_REPAIR_AUDIT.json", audit)
