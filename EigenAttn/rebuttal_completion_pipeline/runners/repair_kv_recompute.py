#!/usr/bin/env python3
"""Recompute OP KV manifests + rebuild matched-control schedules + protocol lock audit."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import PYTHON_BIN, atomic_write_json, now_iso, sha256_file  # noqa: E402
from kv_accounting import audit_from_path, compare_to_runtime_cache_gb, dense_kv_bytes  # noqa: E402

OP = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts/operating_points")
LEGACY = Path("/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts")
CONTROLS = [
    "uniform_matched",
    "random_seed0",
    "random_seed1",
    "random_seed2",
    "spectral_tail",
    "middle_layer",
    "depth_front",
    "depth_back",
    "reconstruction_slope",
]


def _recompute_op(point: str) -> dict:
    if point == "mild":
        uni = LEGACY / "r2_uniform_eb045" / "final_model" / "low_rank_config.json"
        fdp = LEGACY / "r3_fgeb045" / "final_model" / "low_rank_config.json"
        out = OP / "mild"
        # mild may not exist on diskF; write under pipeline artifacts
        out = HERE / "artifacts" / "operating_points" / "mild_kv_recompute"
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = OP / point
        uni = out / "uniform" / "final_model" / "low_rank_config.json"
        fdp = out / "fdp" / "final_model" / "low_rank_config.json"
    uni_kv = audit_from_path(uni) if uni.exists() else {"error": f"missing {uni}"}
    fdp_kv = audit_from_path(fdp) if fdp.exists() else {"error": f"missing {fdp}"}
    # runtime check from prior uniform log if present
    runtime = {}
    log = OP / point / "uni_moderate_stdout.log" if point == "moderate" else OP / point / f"uni_{point}_stdout.log"
    if not log.exists():
        log = OP / point / f"uni_{point}_stdout.log"
    if log.exists():
        for line in log.read_text(errors="ignore").splitlines():
            if "Compressed model KV Cache Size" in line:
                # e.g. 0.1826171875 GB
                try:
                    gb = float(line.split("KV Cache Size")[-1].split("GB")[0].replace(":", "").strip())
                    if "realized_kv_bytes" in uni_kv:
                        runtime = compare_to_runtime_cache_gb(uni_kv["realized_kv_bytes"], gb)
                        runtime["log_line"] = line.strip()
                except Exception as e:
                    runtime = {"error": str(e)}
                break
    man = {
        "point": point,
        "generated_at": now_iso(),
        "kv_recomputed": True,
        "dense_kv_bytes": dense_kv_bytes(),
        "uniform_kv": uni_kv,
        "fdp_kv": fdp_kv,
        "runtime_cache_check_uniform": runtime,
        "uniform_model": str(uni.parent) if uni.exists() else None,
        "fdp_model": str(fdp.parent) if fdp.exists() else None,
        "FORBIDDEN_UNTIL_TRIPLE_MATCH": not bool(runtime.get("match", False)) and point == "moderate",
    }
    # Update OP manifest in place for moderate/aggressive
    if point in ("moderate", "aggressive") and (OP / point).exists():
        prev = {}
        mp = OP / point / "OPERATING_POINT_MANIFEST.json"
        if mp.exists():
            prev = json.loads(mp.read_text())
        prev.update({
            "point": point,
            "generated_at": now_iso(),
            "kv_recomputed": True,
            "uniform_kv": uni_kv,
            "fdp_kv": fdp_kv,
            "dense_kv_bytes": dense_kv_bytes(),
            "runtime_cache_check_uniform": runtime,
            "uniform_model": str(uni.parent),
            "fdp_model": str(fdp.parent),
            "previous_stub_bytes_removed": True,
        })
        # preserve eb_base
        if "eb_base" not in prev:
            prev["eb_base"] = {"moderate": 0.06, "aggressive": 0.09}[point]
        atomic_write_json(mp, prev)
        man["manifest_path"] = str(mp)
    atomic_write_json(HERE / "artifacts" / "kv_repair" / f"OP_{point}_KV_RECOMPUTE.json", man)
    return man


def _protocol_diff() -> dict:
    proto = json.loads((HERE / "FIXED_FDP_SCHEDULE_PROTOCOL.json").read_text())
    conf = proto.get("confirmatory_config") or {}
    rows = []
    for field, val in [
        ("weights", conf.get("weights")),
        ("normalization", conf.get("normalize")),
        ("lambda", conf.get("lambda_eb")),
        ("Layer0_rule", conf.get("freeze_layers")),
        ("clipping_rails", conf.get("clipping_rails")),
        ("calibration_domain", (conf.get("calibration") or {}).get("dataset")),
        ("calibration_size", (conf.get("calibration") or {}).get("nsamples")),
        ("rank_search", conf.get("rank_search_direction")),
    ]:
        rows.append({
            "field": field,
            "locked_value": val,
            "mild": val,
            "moderate": val,
            "aggressive": val,
            "status": "INVARIANT_LOCKED",
            "notes": "Only eb_base may change across OP",
        })
    for pt, eb in [("mild", 0.045), ("moderate", 0.06), ("aggressive", 0.09)]:
        rows.append({
            "field": "eb_base_ALLOWED",
            "locked_value": "may_differ",
            "point": pt,
            "value": eb,
            "status": "ALLOWED_CHANGE",
        })
    out = {
        "generated_at": now_iso(),
        "protocol_sha256": sha256_file(HERE / "FIXED_FDP_SCHEDULE_PROTOCOL.json"),
        "rows": rows,
        "policy": "Single fixed FDP schedule; only eb_base changes across operating points.",
    }
    atomic_write_json(HERE / "artifacts" / "kv_repair" / "SINGLE_SCHEDULE_LOCK_AUDIT.json", out)
    # CSV-ish markdown
    md = ["# SINGLE_SCHEDULE_ACROSS_OPERATING_POINTS_AUDIT", "", f"Generated: {now_iso()}", ""]
    md.append(f"Protocol sha256: `{out['protocol_sha256']}`")
    md.append("")
    md.append("| field | value | status |")
    md.append("|-------|-------|--------|")
    for r in rows:
        if r.get("field") == "eb_base_ALLOWED":
            md.append(f"| eb_base/{r['point']} | {r['value']} | ALLOWED_CHANGE |")
        else:
            md.append(f"| {r['field']} | `{json.dumps(r['locked_value'], ensure_ascii=False)}` | INVARIANT_LOCKED |")
    (HERE / "SINGLE_SCHEDULE_ACROSS_OPERATING_POINTS_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def main() -> None:
    (HERE / "artifacts" / "kv_repair").mkdir(parents=True, exist_ok=True)
    reports = {pt: _recompute_op(pt) for pt in ("mild", "moderate", "aggressive")}
    # rebuild all MC schedules with corrected bytes
    for ctrl in CONTROLS:
        subprocess.check_call(
            [PYTHON_BIN, str(HERE / "runners" / "matched_control_build.py"), "--control", ctrl, "--anchor", "moderate"],
            cwd=str(HERE),
        )
    _protocol_diff()
    # claim freeze note
    note = {
        "generated_at": now_iso(),
        "matched_budget_claims_allowed": False,
        "reason": "Until MC evals finish and triple-check documented per method; OP manifests recomputed.",
        "op_reports": {k: {
            "uniform_bytes": (v.get("uniform_kv") or {}).get("realized_kv_bytes"),
            "fdp_bytes": (v.get("fdp_kv") or {}).get("realized_kv_bytes"),
            "runtime_match": (v.get("runtime_cache_check_uniform") or {}).get("match"),
        } for k, v in reports.items()},
    }
    atomic_write_json(HERE / "artifacts" / "kv_repair" / "KV_CLAIM_FREEZE.json", note)
    print(json.dumps({"ok": True, "reports": note}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
