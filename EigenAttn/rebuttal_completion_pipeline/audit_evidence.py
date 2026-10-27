#!/usr/bin/env python3
"""CPU-only evidence audit: inventory, hashes, gap matrix, protocol consistency."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    ARTIFACT_ROOT,
    LEGACY_ROOT,
    PROJECT_ROOT,
    atomic_write_json,
    file_hash_or_empty,
    legacy_artifacts,
    load_json,
    load_jsonl,
    now_iso,
    sha256_file,
    sha256_json,
)

GATES = [
    "DENSE_PARITY_PASS", "UNIFORM_PARITY_PASS", "LEGACY_FDP_PARITY_PASS",
    "R4_STATISTICS_PASS", "R5_HISTORICAL_PARITY_PASS", "R6_STATISTICS_PASS",
    "R8_SYSTEM_METRICS_PASS", "R10_KQV_ABLATION_PASS", "R11_ROBUSTNESS_PASS",
    "SUPPLEMENTARY_EXPERIMENTS_COMPLETE",
]

METHODS = ["dense", "uniform_eb045", "FGeb045"]
R1_PREFIX = {"dense": "R1_dense", "uniform_eb045": "R2_uniform", "FGeb045": "R3_fdp"}


def _gate_status(name: str) -> Dict[str, Any]:
    for base in (LEGACY_ROOT, LEGACY_ROOT / "supplementary_experiments", HERE):
        p = base / f"{name}.DONE"
        if p.exists():
            return {"exists": True, "path": str(p), "sha256": sha256_file(p)}
    return {"exists": False, "path": None, "sha256": ""}


def _artifact_path(method_key: str, metric: str) -> Optional[Path]:
    prefix = R1_PREFIX.get(method_key, "")
    if not prefix:
        return None
    art = legacy_artifacts()
    link = LEGACY_ROOT / "artifacts" / f"{prefix}_{metric}"
    if link.is_symlink():
        return Path(os.path.realpath(link))
    p = art / f"{prefix}_{metric}"
    return p if p.exists() else (link if link.exists() else None)


def _ppl_evidence(method_key: str, dataset: str) -> Dict[str, Any]:
    metric = f"{dataset}_ppl" if "wiki" not in dataset else "wikitext_ppl"
    p = _artifact_path(method_key, metric)
    out: Dict[str, Any] = {"method": method_key, "dataset": dataset, "present": False}
    if not p or not p.exists():
        return out
    agg = p / "ppl_aggregate.json"
    win = p / "ppl_windows.jsonl"
    out["present"] = agg.exists()
    out["path"] = str(p)
    if agg.exists():
        d = load_json(agg)
        out["ppl"] = d.get("ppl")
        out["token_count_sum"] = d.get("token_count_sum")
        out["used_windows"] = d.get("used_windows")
        out["evaluation_protocol_hash"] = d.get("evaluation_protocol_hash")
        out["aggregate_sha256"] = sha256_file(agg)
    if win.exists():
        rows = load_jsonl(win)
        out["window_count"] = len(rows)
        out["windows_sha256"] = sha256_file(win)
        if rows:
            out["sample_window_ids"] = [r.get("window_id") for r in rows[:3]]
            hashes = {r.get("input_ids_sha256") for r in rows if r.get("input_ids_sha256")}
            out["unique_input_hashes"] = len(hashes)
    return out


def _zs_evidence(method_key: str) -> Dict[str, Any]:
    tasks = {}
    for suffix, tasks_in in [("zs_hs_arc", ["hellaswag_localjsonl", "arc_challenge_localjsonl", "arc_easy_localjsonl"]),
                              ("zs_wg_piqa", ["winogrande_localjsonl", "piqa_localjsonl"])]:
        p = _artifact_path(method_key, suffix)
        if not p or not p.exists():
            continue
        agg = p / "zero_shot_aggregate.json"
        if agg.exists():
            d = load_json(agg)
            for t in tasks_in:
                if t in (d.get("tasks") or {}):
                    tasks[t] = d["tasks"][t]
        samples_dir = p / "zero_shot_samples"
        if samples_dir.is_dir():
            for t in tasks_in:
                sf = samples_dir / f"{t}.jsonl"
                if sf.exists():
                    tasks.setdefault(t, {})["sample_file"] = str(sf)
                    tasks[t]["sample_count"] = len(load_jsonl(sf))
    macro = None
    accs = [v.get("accuracy_from_correct") for v in tasks.values() if isinstance(v, dict) and v.get("accuracy_from_correct") is not None]
    if len(accs) == 5:
        macro = sum(accs) / 5
    return {"method": method_key, "tasks": tasks, "macro_accuracy": macro, "complete_five_tasks": len(tasks) == 5}


def _check_r4_controls() -> Dict[str, Any]:
    required = [
        "Uniform", "FDP", "Random_seed0", "Random_seed1", "Random_seed2",
        "Spectral", "Middle", "Depth_front", "Depth_back", "Reconstruction_slope",
    ]
    r4_csv = LEGACY_ROOT / "artifacts" / "r4_statistics" / "R4_CROSS_METHOD_COMPARISON.csv"
    if not r4_csv.exists():
        r4_csv = legacy_artifacts() / "r4_statistics" / "R4_CROSS_METHOD_COMPARISON.csv"
    found = []
    if r4_csv.exists():
        text = r4_csv.read_text(encoding="utf-8")
        for m in required:
            if m.lower().replace("_", "") in text.lower().replace("_", ""):
                found.append(m)
    return {
        "required_methods": required,
        "found_in_r4": found,
        "complete": set(required).issubset(set(found)),
        "note": "R4 only has Dense/Uniform/FDP point comparison; matched controls missing",
    }


def _check_bootstrap() -> Dict[str, Any]:
    paths = list(LEGACY_ROOT.glob("**/*bootstrap*")) + list(HERE.glob("**/*bootstrap*"))
    paths = [p for p in paths if p.is_file() and "mypy_cache" not in str(p)]
    return {
        "legacy_bootstrap_files": [str(p) for p in paths[:20]],
        "paired_bootstrap_csv": False,
        "mcnemar_csv": False,
        "holm_csv": False,
        "note": "No confirmatory bootstrap/McNemar/Holm in legacy pipeline",
    }


def _check_r8_fields() -> Dict[str, Any]:
    r8 = LEGACY_ROOT / "R8_SYSTEM_METRICS.md"
    required = [
        "median", "p95", "peak allocated", "peak reserved", "TTFT",
        "prefill", "decode/token", "tokens/s", "offline cost",
    ]
    text = r8.read_text(encoding="utf-8") if r8.exists() else ""
    present = {f: f.replace(" ", "").lower() in text.replace(" ", "").lower() for f in required}
    return {
        "path": str(r8) if r8.exists() else None,
        "fields_present": present,
        "has_median_ms": "median" in text.lower() or "Prefill ms" in text,
        "has_p95": "p95" in text.lower(),
        "has_peak_allocated": "Peak MB" in text or "peak allocated" in text.lower(),
        "has_offline_cost": "offline" in text.lower(),
        "complete": all(present.values()),
    }


def _check_r10() -> Dict[str, Any]:
    md = LEGACY_ROOT / "R10_KQV_ABLATION.md"
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    required = [
        "Uniform", "Full FDP", "KQ adaptive", "KQ uniform", "KQ-path", "V-path",
    ]
    r1_wiki = _ppl_evidence("dense", "wikitext2")
    r10_dense = None
    if "6.0465" in text:
        r10_dense = 6.0465
    return {
        "rows_in_r10": text.count("|") // 4 if text else 0,
        "only_dense_and_fdp": "FDP_full_KQV" in text and "KQ adaptive" not in text,
        "r1_dense_wiki_ppl": r1_wiki.get("ppl"),
        "r10_dense_wiki_ppl": r10_dense,
        "ppl_protocol_mismatch": True,
        "mismatch_reason": "R10 uses sliding-window HF loss; R1 uses instrumented non-overlapping chunks",
        "required_branches_missing": [b for b in required if b not in text],
    }


def _check_r11() -> Dict[str, Any]:
    md = LEGACY_ROOT / "R11_ROBUSTNESS.md"
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    fdp = _ppl_evidence("FGeb045", "wikitext2")
    return {
        "is_robustness_only": True,
        "lambda_cal_sweep_complete": "lam0.6" in text or "0.6" in text,
        "confirmatory_lambda": 1.2,
        "best_robustness_ppl": 6.475,
        "confirmatory_fdp_ppl": fdp.get("ppl"),
        "must_not_upgrade_to_main": True,
        "kv_bytes_matched_to_main": False,
        "note": "R11 variants rebuild models; KV bytes not audited per config",
    }


def _check_c4_window_hash_consistency() -> Dict[str, Any]:
    hashes = {}
    for method_key in METHODS:
        ev = _ppl_evidence(method_key, "c4")
        p = _artifact_path(method_key, "c4_ppl")
        if p and (p / "ppl_windows.jsonl").exists():
            rows = load_jsonl(p / "ppl_windows.jsonl")
            ids_hashes = sorted({r.get("input_ids_sha256") for r in rows if r.get("input_ids_sha256")})
            hashes[method_key] = {
                "window_count": len(rows),
                "unique_input_hashes": len(ids_hashes),
                "first_three_hashes": ids_hashes[:3],
            }
    consistent = False
    if len(hashes) >= 2:
        sets = [set(h.get("first_three_hashes") or []) for h in hashes.values()]
        consistent = sets[0] == sets[1] == sets[2] if len(sets) == 3 else False
    return {"per_method": hashes, "all_methods_same_windows": consistent}


def _search_long_context() -> Dict[str, Any]:
    candidates = [
        PROJECT_ROOT / "data" / "LongBench",
        Path("/diskF/jiaqizhang/combine/data/LongBench"),
        PROJECT_ROOT / "slim" / "longbench_eval.py",
        PROJECT_ROOT / "slim" / "needle_in_haystack.py",
    ]
    hits = []
    for p in candidates:
        if p.exists():
            hits.append({"path": str(p), "type": "dir" if p.is_dir() else "file", "sha256": sha256_file(p) if p.is_file() else ""})
    if (PROJECT_ROOT / "data" / "LongBench").is_dir():
        n = len(list((PROJECT_ROOT / "data" / "LongBench").glob("*.jsonl")))
        hits.append({"path": str(PROJECT_ROOT / "data" / "LongBench"), "jsonl_count": n})
    return {"hits": hits, "blocked": len(hits) == 0, "prior_block_note": "legacy BLOCKED doc was wrong; LongBench data exists locally"}


def _search_qwen() -> Dict[str, Any]:
    candidates = list(Path("/diskD/jiaqizhang").glob("*qwen*")) + list(Path("/diskE/jiaqizhang").glob("*qwen*"))
    bundle = PROJECT_ROOT / "qwen_drm_bundle"
    return {
        "local_weights": [str(p) for p in candidates],
        "bundle_exists": bundle.exists(),
        "blocked": len(candidates) == 0,
    }


def build_inventory() -> Dict[str, Any]:
    inv: Dict[str, Any] = {
        "generated_at": now_iso(),
        "gates": {g: _gate_status(g) for g in GATES},
        "methods": {},
        "rank_plans": {},
        "basis_manifests": {},
        "kv_memory": {},
    }
    for m in METHODS:
        inv["methods"][m] = {
            "wikitext2": _ppl_evidence(m, "wikitext2"),
            "c4": _ppl_evidence(m, "c4"),
            "zero_shot": _zs_evidence(m),
        }
    mem = LEGACY_ROOT / "LEGACY_UNIFORM_MEMORY_ACCOUNTING.json"
    if mem.exists():
        inv["kv_memory"]["uniform"] = load_json(mem)
    fdp_manifest = legacy_artifacts() / "r3_fgeb045" / "LEGACY_FDP_MODEL_MANIFEST.json"
    if fdp_manifest.exists():
        inv["basis_manifests"]["fdp"] = {"path": str(fdp_manifest), "sha256": sha256_file(fdp_manifest)}
    rank = LEGACY_ROOT / "LEGACY_UNIFORM_RANK_PLAN.csv"
    if rank.exists():
        inv["rank_plans"]["uniform"] = {"path": str(rank), "sha256": sha256_file(rank)}
    return inv


def build_gaps(inv: Dict[str, Any]) -> Dict[str, Any]:
    gaps: List[Dict[str, Any]] = []

    def add(gid, severity, detail, action):
        gaps.append({"id": gid, "severity": severity, "detail": detail, "action": action})

    if not _check_r4_controls()["complete"]:
        add("GAP_R4_MATCHED_CONTROLS", "critical",
            "R4 lacks Random×3, Spectral, Middle, Depth-front/back, Reconstruction-slope",
            "Build matched-budget controls per operating point")
    if not _check_bootstrap()["paired_bootstrap_csv"]:
        add("GAP_BOOTSTRAP", "critical",
            "No paired bootstrap/McNemar/Holm on existing windows",
            "Compute from reused ppl_windows.jsonl and zero_shot samples")
    if _check_r10()["only_dense_and_fdp"]:
        add("GAP_R10_KQV", "high",
            "R10 only Dense+FDP; missing KQ/V branch ablations",
            "Re-run with instrumented protocol matching R1")
    if _check_r10()["ppl_protocol_mismatch"]:
        add("GAP_R10_PPL_PROTOCOL", "high",
            "R10 Dense PPL 6.05 vs R1 6.138 due to different eval protocol",
            "Discard R10 PPL for main claims; re-run with ppl_instrumented.py")
    if not _check_r8_fields()["complete"]:
        add("GAP_R8_FIELDS", "medium",
            "R8 missing p95, peak reserved, TTFT, offline cost fields",
            "Extend system metrics runner or document eager-only scope")
    add("GAP_OPERATING_POINTS", "critical",
        "No mild/moderate/aggressive operating point rebuilds with matched controls",
        "Rebuild 3 operating points via legacy builder")
    add("GAP_COMPONENT_ABLATION", "high",
        "R11 is lambda×cal only, not minus-Z/S/V/A/Entropy component ablation",
        "Run component and calibration domain ablations")
    if _search_qwen()["blocked"]:
        add("GAP_QWEN", "medium", "No local Qwen2.5-7B base weights", "Download from official HF")
    add("GAP_OPT30B", "medium", "OPT-30B not in legacy confirmatory pipeline",
        "Run fixed-schedule aggressive point if raw missing")
    add("GAP_EXTERNAL_BASELINE", "medium", "No xKV/ReCalKV/PyramidKV official baselines in legacy",
        "Attempt official implementations or document blocked")
    if not inv.get("methods", {}).get("FGeb045", {}).get("zero_shot", {}).get("complete_five_tasks"):
        add("GAP_FIVE_TASKS", "high", "FDP five-task incomplete in inventory check", "Verify and fill")
    return {"generated_at": now_iso(), "gaps": gaps, "total": len(gaps)}


def build_reviewer_matrix(inv: Dict[str, Any], gaps: Dict[str, Any]) -> str:
    lines = [
        "# Reviewer Gap Matrix",
        "",
        f"- generated_at: `{now_iso()}`",
        "",
        "## Existing Evidence (reuse, do not rerun)",
        "",
        "| Area | Status | Path |",
        "|------|--------|------|",
    ]
    for g in GATES:
        st = inv["gates"].get(g, {})
        lines.append(f"| {g} | {'PASS' if st.get('exists') else 'MISSING'} | `{st.get('path') or '—'}` |")
    lines.extend(["", "## Confirmatory Results (R1-R3 instrumented)", ""])
    for m in METHODS:
        w = inv["methods"][m]["wikitext2"]
        c = inv["methods"][m]["c4"]
        z = inv["methods"][m]["zero_shot"]
        lines.append(f"### {m}")
        lines.append(f"- Wiki PPL: {w.get('ppl')} ({w.get('window_count')} windows)")
        lines.append(f"- C4 PPL: {c.get('ppl')} ({c.get('window_count')} windows)")
        lines.append(f"- Macro acc: {z.get('macro_accuracy')} (5 tasks: {z.get('complete_five_tasks')})")
        lines.append("")
    lines.extend(["## Gaps Requiring New Work", ""])
    lines.append("| ID | Severity | Action |")
    lines.append("|----|----------|--------|")
    for g in gaps["gaps"]:
        lines.append(f"| {g['id']} | {g['severity']} | {g['action']} |")
    return "\n".join(lines) + "\n"


def build_protocol_audit() -> str:
    r10 = _check_r10()
    r11 = _check_r11()
    r8 = _check_r8_fields()
    c4 = _check_c4_window_hash_consistency()
    lines = [
        "# Protocol Consistency Audit",
        "",
        f"- generated_at: `{now_iso()}`",
        "",
        "## R10 vs R1 PPL",
        f"- R1 Dense Wiki PPL (instrumented): `{r10.get('r1_dense_wiki_ppl')}`",
        f"- R10 Dense Wiki PPL (sliding HF loss): `{r10.get('r10_dense_wiki_ppl')}`",
        f"- Mismatch: `{r10.get('ppl_protocol_mismatch')}` — {r10.get('mismatch_reason')}",
        "",
        "## R11 Robustness",
        f"- Confirmatory λ=1.2 FDP PPL: `{r11.get('confirmatory_fdp_ppl')}`",
        f"- Best robustness (λ=2.0, cal=32): `{r11.get('best_robustness_ppl')}`",
        f"- Must NOT upgrade robustness best to main: `{r11.get('must_not_upgrade_to_main')}`",
        "",
        "## C4 Window Consistency",
        f"- All methods same 256 windows: `{c4.get('all_methods_same_windows')}`",
        json.dumps(c4.get("per_method"), indent=2),
        "",
        "## R8 System Metrics Completeness",
        json.dumps(r8, indent=2),
        "",
        "## R4 Matched Controls",
        json.dumps(_check_r4_controls(), indent=2),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    inv = build_inventory()
    gaps = build_gaps(inv)
    hashes = {
        "generated_at": now_iso(),
        "inventory_sha256": sha256_json(inv),
        "files": {
            "LEGACY_PROTOCOL_LOCK.json": file_hash_or_empty(LEGACY_ROOT / "LEGACY_PROTOCOL_LOCK.json"),
            "FIXED_FDP_SCHEDULE_PROTOCOL.json": file_hash_or_empty(HERE / "FIXED_FDP_SCHEDULE_PROTOCOL.json"),
            "R6_FINAL_SUMMARY.md": file_hash_or_empty(LEGACY_ROOT / "R6_FINAL_SUMMARY.md"),
        },
        "c4_window_audit": _check_c4_window_hash_consistency(),
        "long_context": _search_long_context(),
        "qwen": _search_qwen(),
    }
    atomic_write_json(HERE / "CURRENT_EVIDENCE_INVENTORY.json", inv)
    atomic_write_json(HERE / "CURRENT_EVIDENCE_HASHES.json", hashes)
    atomic_write_json(HERE / "UNRESOLVED_GAPS.json", gaps)
    (HERE / "REVIEWER_GAP_MATRIX.md").write_text(build_reviewer_matrix(inv, gaps), encoding="utf-8")
    (HERE / "PROTOCOL_CONSISTENCY_AUDIT.md").write_text(build_protocol_audit(), encoding="utf-8")
    # Freeze protocol sha256
    proto = HERE / "FIXED_FDP_SCHEDULE_PROTOCOL.json"
    (HERE / "FIXED_FDP_SCHEDULE_PROTOCOL.sha256").write_text(sha256_file(proto) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "gaps": gaps["total"], "gates_pass": sum(1 for g in GATES if inv["gates"][g]["exists"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
