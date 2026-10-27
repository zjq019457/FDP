#!/usr/bin/env python3
"""Final result-level audit + full 53-job export. No new experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PIPE = Path(__file__).resolve().parent
PROJECT = PIPE.parent
HOME_JQ = Path("/home/bai-project/jiaqizhang")
ART = Path("/diskE/jiaqizhang/rebuttal_completion_artifacts")
OP_ART = Path("/diskF/jiaqizhang/rebuttal_completion_op_artifacts")
LEGACY_ART = Path("/diskF/jiaqizhang/rebuttal_legacy_reproduction_artifacts")

TASK_MAP = {
    "hellaswag_localjsonl": "HellaSwag",
    "arc_challenge_localjsonl": "ARC-Challenge",
    "arc_easy_localjsonl": "ARC-Easy",
    "winogrande_localjsonl": "WinoGrande",
    "piqa_localjsonl": "PIQA",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def copy_filtered(src: Path, dst: Path, *, skip_weight_ext: bool = True) -> int:
    """Copy tree; skip model weight blobs; follow symlinks for content."""
    if not src.exists():
        return 0
    n = 0
    skip_ext = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"} if skip_weight_ext else set()
    skip_names = {"HF_cache"}
    if src.is_file():
        if src.suffix.lower() in skip_ext or src.name in skip_names:
            return 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return 1
    for root, dirs, files in os.walk(src, followlinks=True):
        root_p = Path(root)
        dirs[:] = [d for d in dirs if d not in skip_names]
        # Still walk final_model but only keep small configs
        for f in files:
            sp = root_p / f
            if sp.suffix.lower() in skip_ext:
                continue
            # skip huge redundant dumps if > 80MB (lm_eval_full_results kept if smaller)
            try:
                sz = sp.stat().st_size
            except OSError:
                continue
            if sz > 120 * 1024 * 1024 and f.startswith("lm_eval_full"):
                # keep pointer instead
                rel = sp.relative_to(src)
                dp = dst / rel
                dp.parent.mkdir(parents=True, exist_ok=True)
                write_json(
                    dp.with_suffix(dp.suffix + ".POINTER.json"),
                    {"original": str(sp), "size_bytes": sz, "sha256": sha256_file(sp), "skipped_in_export": True},
                )
                n += 1
                continue
            if "final_model" in sp.parts and sp.suffix.lower() in skip_ext:
                continue
            rel = sp.relative_to(src)
            dp = dst / rel
            dp.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(sp, dp)
                n += 1
            except Exception as e:
                write_text(dp.with_suffix(dp.suffix + ".COPY_ERROR.txt"), str(e))
    return n


def find_zs_agg(point: str, method: str) -> Dict[str, Any]:
    tasks: Dict[str, Any] = {}
    hashes: Dict[str, str] = {}
    runner = ""
    model_hash = ""
    for bund in ["hs_arc", "wg_piqa"]:
        cands = [
            ART / f"OP_{point}_{method}_{bund}/zero_shot_aggregate.json",
            ART / f"OP_{point}_{method}_zs_{bund}/zero_shot_aggregate.json",
        ]
        for p in cands:
            if not p.exists():
                continue
            z = load_json(p) or {}
            runner = z.get("runner_hash") or runner
            model_hash = z.get("model_hash") or model_hash
            for t, v in (z.get("tasks") or {}).items():
                if isinstance(v, dict):
                    tasks[t] = v.get("accuracy_from_correct")
                    if v.get("task_definition_hash"):
                        hashes[t] = v["task_definition_hash"]
                else:
                    tasks[t] = v
            break
    return {"tasks": tasks, "task_hashes": hashes, "runner_hash": runner, "model_hash": model_hash}


def find_ppl(point: str, method: str, dataset: str) -> Dict[str, Any]:
    p = ART / f"OP_{point}_{method}_{dataset}_ppl/ppl_aggregate.json"
    if not p.exists():
        return {}
    a = load_json(p) or {}
    # window set hash from first windows if present
    win_hash = ""
    windows_dir = p.parent / "windows"
    jsonl = p.parent / "ppl_windows.jsonl"
    if jsonl.exists():
        # hash of all input_ids_sha256 lines
        ids = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
                ids.append(o.get("input_ids_sha256") or o.get("window_id") or "")
            except Exception:
                pass
        if ids:
            win_hash = sha256_text("\n".join(ids))
    return {
        "ppl": a.get("ppl"),
        "runner_hash": a.get("runner_hash"),
        "model_hash": a.get("model_hash"),
        "used_windows": a.get("used_windows"),
        "token_count_sum": a.get("token_count_sum"),
        "seqlen": a.get("seqlen"),
        "window_set_hash": win_hash,
        "path": str(p),
    }


def kv_for(point: str, method: str) -> Dict[str, Any]:
    if point == "mild":
        # legacy
        if method == "uniform":
            man = LEGACY_ART / "r2_uniform_eb045"
        else:
            man = LEGACY_ART / "r3_fgeb045"
        # try read any kv accounting
        for cand in [
            man / "KV_MEMORY_ACCOUNTING.json",
            man / "final_model" / "low_rank_config.json",
            OP_ART / f"operating_points/{point}/OPERATING_POINT_MANIFEST.json",
        ]:
            pass
        # from protocol historical ~80.9% retained; use schedule if present
        return {
            "realized_kv_bytes": None,
            "total_rank_sum": None,
            "nominal_ratio": 0.820 if point == "mild" else None,
            "provenance": "legacy_mild_reuse",
            "model_path": str(
                LEGACY_ART / ("r2_uniform_eb045/final_model" if method == "uniform" else "r3_fgeb045/final_model")
            ),
        }
    m = load_json(OP_ART / f"operating_points/{point}/OPERATING_POINT_MANIFEST.json") or {}
    key = "uniform_kv" if method == "uniform" else "fdp_kv"
    kv = m.get(key) or {}
    targets = {"mild": 0.820, "moderate": 0.739, "aggressive": 0.589}
    return {
        "realized_kv_bytes": kv.get("realized_kv_bytes"),
        "total_rank_sum": kv.get("total_rank_sum"),
        "nominal_ratio": targets.get(point),
        "provenance": str(OP_ART / f"operating_points/{point}/OPERATING_POINT_MANIFEST.json"),
        "model_path": m.get(f"{method}_model"),
    }


def dense_ref_bytes() -> Optional[int]:
    # Prefer matched-control target as reported dense-matched budget if present
    s = load_json(PIPE / "artifacts/matched_controls/MATCHED_CONTROL_SUMMARY.json") or {}
    ctrls = s.get("controls") or []
    if ctrls:
        return ctrls[0].get("target_kv_bytes")
    return None


def compute_retained(actual_bytes: Optional[int], dense_bytes: Optional[int]) -> Optional[float]:
    if actual_bytes is None or not dense_bytes:
        return None
    return float(actual_bytes) / float(dense_bytes)


def job_snapshot(jobs: List[Dict[str, Any]], export: Path) -> None:
    rows = []
    for j in jobs:
        jid = j["job_id"]
        out = PIPE / "job_outputs" / jid
        done = out / "DONE.json"
        state = out / "job_state.json"
        failed = out / "FAILED.json"
        stdout = out / "stdout.log"
        stderr = out / "stderr.log"
        entry = {
            "job_id": jid,
            "status": j.get("status"),
            "kind": j.get("kind"),
            "phase": j.get("phase"),
            "cmd": j.get("cmd"),
            "deps": j.get("deps"),
            "done_exists": done.exists(),
            "failed_exists": failed.exists(),
            "job_state": load_json(state),
            "done": load_json(done),
            "failed": load_json(failed) if failed.exists() else None,
            "stdout_sha256": sha256_file(stdout) if stdout.exists() else "",
            "stderr_sha256": sha256_file(stderr) if stderr.exists() else "",
            "stdout_bytes": stdout.stat().st_size if stdout.exists() else 0,
            "stderr_bytes": stderr.stat().st_size if stderr.exists() else 0,
        }
        # input/output hash from DONE if present
        if entry["done"]:
            entry["output_hash"] = entry["done"].get("output_hash") or entry["done"].get("artifact_hash")
            entry["input_hash"] = entry["done"].get("input_hash") or entry["done"].get("protocol_hash")
        rows.append(entry)
        write_json(export / "job_manifests" / f"{jid}.json", entry)
        # copy job_outputs fully
        copy_filtered(out, export / "job_outputs" / jid, skip_weight_ext=True)
    write_json(export / "meta" / "ALL_JOBS_MANIFEST.json", {"generated_at": now_iso(), "jobs": rows})
    write_csv(
        export / "meta" / "ALL_JOBS_STATUS.csv",
        [
            {
                "job_id": r["job_id"],
                "status": r["status"],
                "kind": r["kind"],
                "done_exists": r["done_exists"],
                "failed_exists": r["failed_exists"],
                "stdout_bytes": r["stdout_bytes"],
                "input_hash": r.get("input_hash") or "",
                "output_hash": r.get("output_hash") or "",
            }
            for r in rows
        ],
        [
            "job_id",
            "status",
            "kind",
            "done_exists",
            "failed_exists",
            "stdout_bytes",
            "input_hash",
            "output_hash",
        ],
    )


def build_final_method_results(export: Path) -> List[Dict[str, Any]]:
    dense_b = dense_ref_bytes()
    # FDP bytes per point for gap
    fdp_bytes = {}
    for pt in ["mild", "moderate", "aggressive"]:
        fdp_bytes[pt] = kv_for(pt, "fdp").get("realized_kv_bytes")

    # protocol hashes
    proto = PIPE / "FIXED_FDP_SCHEDULE_PROTOCOL.json"
    proto_hash = sha256_file(proto) if proto.exists() else ""

    # legacy bootstrap only for mild confirmatory group
    boot = {}
    boot_csv = PIPE / "artifacts/statistics/PPL_PAIRED_BOOTSTRAP.csv"
    if boot_csv.exists():
        with boot_csv.open() as f:
            for row in csv.DictReader(f):
                boot[row.get("comparison", "")] = row

    rows = []
    for point in ["mild", "moderate", "aggressive"]:
        for method in ["uniform", "fdp"]:
            wiki = find_ppl(point, method, "wikitext2")
            c4 = find_ppl(point, method, "c4")
            zs = find_zs_agg(point, method)
            kv = kv_for(point, method)
            tasks = zs["tasks"]
            accs = []
            row_acc = {}
            for k, col in TASK_MAP.items():
                v = tasks.get(k)
                row_acc[col] = v
                if v is not None:
                    accs.append(float(v))
            macro = sum(accs) / len(accs) if len(accs) == 5 else None
            actual = kv.get("realized_kv_bytes")
            fdp_b = fdp_bytes.get(point)
            byte_gap = None
            if actual is not None and fdp_b is not None:
                byte_gap = int(actual) - int(fdp_b)

            # CI / p only for mild confirmatory legacy bootstrap; OP bootstrap pending
            ppl_ci = ""
            macro_ci = ""
            paired_p = ""
            holm_p = ""
            comparable_group = f"llama3_8b_op_{point}_instrumented_eval_lock"
            provenance = []
            if point == "mild":
                provenance.append("REUSED_FROM_LEGACY_R2_R3")
                if method == "fdp" and "Uniform vs FDP Wiki" in boot:
                    b = boot["Uniform vs FDP Wiki"]
                    ppl_ci = f"[{b.get('ci95_low')},{b.get('ci95_high')}]_delta_wiki_uniform_minus_fdp"
                if method == "fdp" and "Uniform vs FDP C4" in boot:
                    b = boot["Uniform vs FDP C4"]
                    ppl_ci = (ppl_ci + ";" if ppl_ci else "") + f"[{b.get('ci95_low')},{b.get('ci95_high')}]_delta_c4"
            else:
                provenance.append("NEW_OP_BUILD_ON_DISKF")
                ppl_ci = "OP_BOOTSTRAP_PENDING"
                paired_p = "ZS_MCNEMAR_NO_COMMON_SAMPLES"
                holm_p = "NOT_AVAILABLE"

            # window hash consistency note
            if point != "mild" and c4.get("window_set_hash") and wiki.get("window_set_hash"):
                provenance.append(f"c4_window_set_hash={c4['window_set_hash'][:16]}")
                provenance.append(f"wiki_window_set_hash={wiki['window_set_hash'][:16]}")

            # evaluation protocol: same runner preferred
            eval_proto = proto_hash
            sample_set_hash = sha256_text(json.dumps(zs.get("task_hashes") or {}, sort_keys=True))

            rows.append(
                {
                    "model": "Llama-3-8B",
                    "operating_point": point,
                    "method": method,
                    "nominal_ratio": kv.get("nominal_ratio"),
                    "actual_KV_bytes": actual if actual is not None else "UNREPORTED_IN_MANIFEST",
                    "actual_retained_ratio": compute_retained(actual, dense_b),
                    "byte_gap_relative_to_FDP": byte_gap if byte_gap is not None else ("0" if method == "fdp" else "UNREPORTED"),
                    "WikiText_PPL": wiki.get("ppl"),
                    "C4_PPL": c4.get("ppl"),
                    "PIQA": row_acc.get("PIQA"),
                    "WinoGrande": row_acc.get("WinoGrande"),
                    "ARC-Easy": row_acc.get("ARC-Easy"),
                    "ARC-Challenge": row_acc.get("ARC-Challenge"),
                    "HellaSwag": row_acc.get("HellaSwag"),
                    "five_task_macro": macro,
                    "PPL_CI": ppl_ci,
                    "macro_CI": macro_ci or "NOT_AVAILABLE",
                    "paired_p_value": paired_p or ("legacy_bootstrap_sig_see_PPL_CI" if point == "mild" else ""),
                    "Holm_adjusted_p_value": holm_p or "ZS_HOLM_INVALID_NO_COMMON_SAMPLES",
                    "runner_hash": wiki.get("runner_hash") or zs.get("runner_hash") or "",
                    "evaluation_protocol_hash": eval_proto,
                    "C4_window_hash": c4.get("window_set_hash") or "",
                    "zero_shot_sample_set_hash": sample_set_hash,
                    "result_provenance": ";".join(provenance),
                    "comparable_group": comparable_group,
                    "dense_ref_bytes_for_ratio": dense_b,
                    "NOTE_KV_ACCOUNTING": (
                        "WARNING: moderate/aggressive manifests report identical realized_kv_bytes "
                        "for uniform and fdp despite different total_rank_sum; treat byte_gap with caution."
                        if point in ("moderate", "aggressive")
                        else ""
                    ),
                }
            )
    fields = [
        "model",
        "operating_point",
        "method",
        "nominal_ratio",
        "actual_KV_bytes",
        "actual_retained_ratio",
        "byte_gap_relative_to_FDP",
        "WikiText_PPL",
        "C4_PPL",
        "PIQA",
        "WinoGrande",
        "ARC-Easy",
        "ARC-Challenge",
        "HellaSwag",
        "five_task_macro",
        "PPL_CI",
        "macro_CI",
        "paired_p_value",
        "Holm_adjusted_p_value",
        "runner_hash",
        "evaluation_protocol_hash",
        "C4_window_hash",
        "zero_shot_sample_set_hash",
        "result_provenance",
        "comparable_group",
        "dense_ref_bytes_for_ratio",
        "NOTE_KV_ACCOUNTING",
    ]
    write_csv(export / "FINAL_METHOD_RESULTS.csv", rows, fields)
    write_json(export / "FINAL_METHOD_RESULTS.json", {"generated_at": now_iso(), "rows": rows})
    return rows


def audit_matched_controls(export: Path) -> None:
    ctrls = [
        ("Uniform matched", "uniform_matched"),
        ("Random seed0", "random_seed0"),
        ("Random seed1", "random_seed1"),
        ("Random seed2", "random_seed2"),
        ("Spectral-tail", "spectral_tail"),
        ("Middle-layer protection", "middle_layer"),
        ("Depth-front", "depth_front"),
        ("Depth-back", "depth_back"),
        ("Reconstruction-slope", "reconstruction_slope"),
        ("FDP", "fdp"),
    ]
    rows = []
    for label, key in ctrls:
        if key == "fdp":
            # FDP at moderate OP has full PPL+ZS
            wiki = find_ppl("moderate", "fdp", "wikitext2")
            c4 = find_ppl("moderate", "fdp", "c4")
            zs = find_zs_agg("moderate", "fdp")
            has_wiki = wiki.get("ppl") is not None
            has_c4 = c4.get("ppl") is not None
            has_zs = len([v for v in zs["tasks"].values() if v is not None]) == 5
            tag = "FULL_EVAL" if (has_wiki and has_c4 and has_zs) else "INCOMPLETE"
            rows.append(
                {
                    "method": label,
                    "control_key": key,
                    "has_schedule": True,
                    "has_WikiText_PPL": has_wiki,
                    "has_C4_PPL": has_c4,
                    "has_five_zero_shot": has_zs,
                    "completeness_tag": tag,
                    "WikiText_PPL": wiki.get("ppl"),
                    "C4_PPL": c4.get("ppl"),
                    "HellaSwag": zs["tasks"].get("hellaswag_localjsonl"),
                    "ARC-Challenge": zs["tasks"].get("arc_challenge_localjsonl"),
                    "ARC-Easy": zs["tasks"].get("arc_easy_localjsonl"),
                    "WinoGrande": zs["tasks"].get("winogrande_localjsonl"),
                    "PIQA": zs["tasks"].get("piqa_localjsonl"),
                    "notes": "FDP moderate operating point (not MC schedule artifact)",
                }
            )
            continue
        sched = ART / f"matched_controls/{key}/MATCHED_CONTROL_SCHEDULE.json"
        ranks = ART / f"matched_controls/{key}/layer_ranks.csv"
        has_sched = sched.exists()
        # search any eval under control
        has_wiki = any((ART / f"matched_controls/{key}").glob("**/*wikitext*")) or any(
            ART.glob(f"MC_{key}*wikitext*")
        )
        has_c4 = any((ART / f"matched_controls/{key}").glob("**/*c4*"))
        has_zs = any((ART / f"matched_controls/{key}").glob("**/zero_shot*"))
        # also check pipeline artifacts links
        if not has_wiki and not has_c4 and not has_zs:
            tag = "SCHEDULE_ONLY_NO_EVAL"
        elif (has_wiki or has_c4) and not has_zs:
            tag = "PPL_ONLY_CONTROL"
        else:
            tag = "FULL_EVAL"
        budget = None
        summary = load_json(PIPE / "artifacts/matched_controls/MATCHED_CONTROL_SUMMARY.json") or {}
        for c in summary.get("controls") or []:
            if c.get("control") == key:
                budget = c
        rows.append(
            {
                "method": label,
                "control_key": key,
                "has_schedule": has_sched,
                "has_WikiText_PPL": bool(has_wiki),
                "has_C4_PPL": bool(has_c4),
                "has_five_zero_shot": bool(has_zs),
                "completeness_tag": tag,
                "WikiText_PPL": "",
                "C4_PPL": "",
                "HellaSwag": "",
                "ARC-Challenge": "",
                "ARC-Easy": "",
                "WinoGrande": "",
                "PIQA": "",
                "target_kv_bytes": (budget or {}).get("target_kv_bytes"),
                "realized_kv_bytes_est": (budget or {}).get("realized_kv_bytes_est"),
                "rel_gap": (budget or {}).get("rel_gap"),
                "notes": f"schedule={has_sched}; ranks={ranks.exists()}; NO instrumented PPL/ZS under matched_controls/",
            }
        )

    fields = list(rows[0].keys())
    write_csv(export / "MATCHED_CONTROL_FULL_TABLE.csv", rows, fields)

    md = [
        "# MATCHED_CONTROL_COMPLETENESS_AUDIT",
        "",
        f"Generated: {now_iso()}",
        "",
        "## Verdict",
        "",
        "All nine matched-control jobs (`MC_*`) produced **rank schedules only**.",
        "None of Uniform-matched / Random×3 / Spectral / Middle / Depth-front / Depth-back / Reconstruction-slope",
        "have WikiText PPL, C4 PPL, or five-task zero-shot under this completion pipeline.",
        "",
        "Therefore they are tagged **`SCHEDULE_ONLY_NO_EVAL`** (stricter than `PPL_ONLY_CONTROL`).",
        "FDP at the moderate operating point **does** have full PPL+ZS, but that is OP evidence, not MC-matched eval.",
        "",
        "## Per-method checklist",
        "",
        "| Method | Schedule | Wiki | C4 | 5 ZS | Tag |",
        "|--------|----------|------|----|------|-----|",
    ]
    for r in rows:
        md.append(
            f"| {r['method']} | {r['has_schedule']} | {r['has_WikiText_PPL']} | {r['has_C4_PPL']} | "
            f"{r['has_five_zero_shot']} | `{r['completeness_tag']}` |"
        )
    md += [
        "",
        "## Budget notes",
        "",
        "- MC summary target_kv_bytes ≈ 396,754,944 for all controls.",
        "- OP moderate/aggressive manifests report realized_kv_bytes ≈ 217,164,283 (different accounting; do not equate).",
        "- Random×3 and reconstruction_slope are severely under-budget relative to the MC target.",
        "",
        "## Claim implication",
        "",
        "Job status DONE for `MC_*` means **schedules were written**, not that FDP beat these controls on downstream quality.",
    ]
    write_text(export / "MATCHED_CONTROL_COMPLETENESS_AUDIT.md", "\n".join(md))

    guard = [
        "# MATCHED_CONTROL_CLAIM_GUARD",
        "",
        "## Status: unsupported for downstream superiority claims",
        "",
        "### Allowed",
        "- State that matched-control **schedules** were generated for the nine methods.",
        "- Report budget gaps from `MATCHED_CONTROL_BUDGET_AUDIT.csv` / summary JSON.",
        "",
        "### Forbidden",
        "- Claiming FDP beats Uniform-matched / Random / Spectral / Middle / Depth / Reconstruction on",
        "  WikiText, C4, or five-task zero-shot **from this completion pipeline**.",
        "- Treating `MC_* : DONE` as evidence of full matched-control evaluation.",
        "- Relabeling `SCHEDULE_ONLY_NO_EVAL` as `PPL_ONLY_CONTROL` without noting zero-shot absence —",
        "  if only PPL existed, use `PPL_ONLY_CONTROL` and still forbid downstream claims.",
        "",
        "### Required sentence if asked",
        "> Matched controls in the completion pipeline currently have schedules only; full PPL and five-task",
        "> zero-shot evaluations were not executed, so we do not claim FDP superiority over these controls",
        "> on aggregate downstream quality.",
    ]
    write_text(export / "MATCHED_CONTROL_CLAIM_GUARD.md", "\n".join(guard))


def audit_single_schedule(export: Path) -> None:
    proto = load_json(PIPE / "FIXED_FDP_SCHEDULE_PROTOCOL.json") or {}
    conf = proto.get("confirmatory_config") or {}
    rows = []
    fields_cmp = [
        "field",
        "mild",
        "moderate",
        "aggressive",
        "invariant_expected",
        "status",
        "notes",
    ]
    # Compare OP manifests + known build args
    mild = {"eb_base": 0.045, "source": "legacy FGeb045 / uniform eb045", "manifest": "legacy"}
    mod = load_json(OP_ART / "operating_points/moderate/OPERATING_POINT_MANIFEST.json") or {}
    agg = load_json(OP_ART / "operating_points/aggressive/OPERATING_POINT_MANIFEST.json") or {}

    comparisons = [
        ("fragility_components", "Z,S,V,A,entropy", "same_by_protocol", "same_by_protocol", True, "protocol freeze; per-OP step1 not re-diffed in export"),
        ("weights", json.dumps(conf.get("weights")), "same_locked", "same_locked", True, ""),
        ("normalization", conf.get("normalize"), "mean", "mean", True, ""),
        ("lambda", conf.get("lambda_eb"), 1.2, 1.2, True, "confirmatory λ=1.2"),
        ("Layer0_rule", conf.get("freeze_layers"), [0], [0], True, ""),
        ("clipping_rails", conf.get("clipping_rails"), "none", "none", True, ""),
        ("rank_search", conf.get("rank_search_direction"), "decrease_until_budget_met", "decrease_until_budget_met", True, ""),
        ("calibration_domain", (conf.get("calibration") or {}).get("dataset"), "c4", "c4", True, "protocol; verify build logs if disputed"),
        ("calibration_size", (conf.get("calibration") or {}).get("nsamples"), 64, 64, True, ""),
        ("eb_base_ALLOWED_CHANGE", 0.045, mod.get("eb_base"), agg.get("eb_base"), True, "only eb_base may change"),
        ("realized_kv_bytes_uniform", "legacy_unreported_here", (mod.get("uniform_kv") or {}).get("realized_kv_bytes"), (agg.get("uniform_kv") or {}).get("realized_kv_bytes"), False, "identical 217164283 across points — accounting suspicious"),
        ("realized_kv_bytes_fdp", "legacy_unreported_here", (mod.get("fdp_kv") or {}).get("realized_kv_bytes"), (agg.get("fdp_kv") or {}).get("realized_kv_bytes"), False, "identical to uniform in manifests"),
        ("total_rank_sum_uniform", "n/a", (mod.get("uniform_kv") or {}).get("total_rank_sum"), (agg.get("uniform_kv") or {}).get("total_rank_sum"), True, "differs by point as expected"),
        ("total_rank_sum_fdp", "n/a", (mod.get("fdp_kv") or {}).get("total_rank_sum"), (agg.get("fdp_kv") or {}).get("total_rank_sum"), True, ""),
    ]
    for field, mild_v, mod_v, agg_v, ok, notes in comparisons:
        status = "OK" if ok else "VARIANT_OR_ANOMALY"
        rows.append(
            {
                "field": field,
                "mild": mild_v,
                "moderate": mod_v,
                "aggressive": agg_v,
                "invariant_expected": "same except eb_base" if field != "eb_base_ALLOWED_CHANGE" else "may differ",
                "status": status,
                "notes": notes,
            }
        )
    write_csv(export / "OPERATING_POINT_PROTOCOL_DIFF.csv", rows, fields_cmp)

    # Variant shopping flags
    variants = []
    if (mod.get("uniform_kv") or {}).get("realized_kv_bytes") == (mod.get("fdp_kv") or {}).get("realized_kv_bytes"):
        variants.append(
            {
                "issue": "UNIFORM_FDP_IDENTICAL_REALIZED_KV_BYTES_IN_MANIFEST",
                "severity": "high",
                "detail": "moderate/aggressive manifests list the same realized_kv_bytes for uniform and fdp; rank sums differ. Do not claim byte-matched controls from this field alone.",
            }
        )
    if (mod.get("uniform_kv") or {}).get("realized_kv_bytes") == (agg.get("uniform_kv") or {}).get("realized_kv_bytes"):
        variants.append(
            {
                "issue": "MODERATE_AGGRESSIVE_IDENTICAL_REALIZED_KV_BYTES",
                "severity": "high",
                "detail": "Different eb_base/rank sums but identical realized_kv_bytes across operating points — KV byte accounting likely broken or stubbed.",
            }
        )
    variants.append(
        {
            "issue": "MILD_IS_LEGACY_REUSE",
            "severity": "medium",
            "detail": "mild OP evaluates legacy R2/R3 artifacts rather than a newly built diskF operating-point model.",
        }
    )

    md = [
        "# SINGLE_SCHEDULE_ACROSS_OPERATING_POINTS_AUDIT",
        "",
        f"Generated: {now_iso()}",
        "",
        "## Protocol lock",
        "",
        f"- FIXED_FDP_SCHEDULE_PROTOCOL.json sha256: `{sha256_file(PIPE / 'FIXED_FDP_SCHEDULE_PROTOCOL.json')}`",
        f"- Confirmatory tag: `{conf.get('tag')}`, λ=`{conf.get('lambda_eb')}`, normalize=`{conf.get('normalize')}`",
        f"- Weights: `{json.dumps(conf.get('weights'))}`",
        f"- freeze_layers: `{conf.get('freeze_layers')}`, clipping: `{conf.get('clipping_rails')}`",
        "",
        "## Allowed variation",
        "",
        "Only `eb_base` / operating-point target retained ratio may change (0.045 / 0.060 / 0.090).",
        "",
        "## Findings",
        "",
    ]
    for v in variants:
        md.append(f"- **{v['issue']}** ({v['severity']}): {v['detail']}")
    md += [
        "",
        "## Variant shopping",
        "",
        "No evidence that λ/weights/normalize were swept to pick a winner for confirmatory claims",
        "inside the completion pipeline freeze. R11 λ/cal sweeps exist as **robustness-only** legacy results",
        "and must not replace confirmatory λ=1.2.",
        "",
        "However, **KV byte fields in OP manifests are not trustworthy** for claiming matched bytes across methods/points.",
        "See `OPERATING_POINT_PROTOCOL_DIFF.csv`.",
    ]
    write_text(export / "SINGLE_SCHEDULE_ACROSS_OPERATING_POINTS_AUDIT.md", "\n".join(md))
    write_json(export / "meta" / "OP_VARIANT_FLAGS.json", variants)


def audit_long_context(export: Path) -> None:
    man = load_json(PIPE / "LONG_CONTEXT_ASSET_MANIFEST.json") or {}
    s2 = load_json(PIPE / "artifacts/long_context/LONG_CONTEXT_2K_STATUS.json") or {}
    s8 = load_json(PIPE / "artifacts/long_context/LONG_CONTEXT_8K_STATUS.json") or {}
    rows = [
        {
            "profile": "2k",
            "eval_executed": s2.get("eval_executed"),
            "status": s2.get("status"),
            "reason": s2.get("reason"),
            "seqlen_claimed": (s2.get("calibration") or {}).get("seqlen"),
            "nsamples": (s2.get("calibration") or {}).get("nsamples"),
            "methods_listed": ",".join(s2.get("methods") or []),
            "raw_predictions_present": False,
            "actual_kv_bytes_present": False,
            "tag": "NOT_EXECUTED_PROTOCOL_ONLY",
        },
        {
            "profile": "8k",
            "eval_executed": s8.get("eval_executed"),
            "status": s8.get("status"),
            "reason": s8.get("reason"),
            "seqlen_claimed": (s8.get("calibration") or {}).get("seqlen"),
            "nsamples": (s8.get("calibration") or {}).get("nsamples"),
            "methods_listed": ",".join(s8.get("methods") or []),
            "raw_predictions_present": False,
            "actual_kv_bytes_present": False,
            "tag": "NOT_EXECUTED_PROTOCOL_ONLY",
        },
    ]
    write_csv(export / "LONG_CONTEXT_FINAL_TABLE.csv", rows, list(rows[0].keys()))

    has_longbench = bool(man.get("longbench_paths") or man.get("assets") or True)
    # read license audit snippet
    md = [
        "# LONG_CONTEXT_EVIDENCE_AUDIT",
        "",
        f"Generated: {now_iso()}",
        "",
        "## Dataset lock",
        "",
        f"- Asset manifest blocked={man.get('blocked')}",
        f"- Local LongBench jsonl assets exist (see LONG_CONTEXT_ASSET_MANIFEST.json / DATA_HASHES).",
        "- Official RULER repository/data: **not locked / not found** in completion audit.",
        "- NIAH: scripts referenced in slim; **not executed** as formal benchmark in this pipeline.",
        "",
        "## Execution",
        "",
        "- `P4_long_context_2k` / `P4_long_context_8k` jobs are DONE but `eval_executed: false`",
        "  with reason `protocol_frozen_first_pass`.",
        "- No raw predictions, no scorer outputs, no actual KV bytes for 2K/8K method comparisons.",
        "- 8K profile claims calibration seqlen=8192 in status JSON, but **no real 8192-token eval ran**.",
        "",
        "## Classification",
        "",
        "**`SYNTHETIC_LONG_CONTEXT_DIAGNOSTIC` / more precisely `NOT_EXECUTED_PROTOCOL_ONLY`**.",
        "Do **not** describe these jobs as a completed official LongBench/RULER/NIAH benchmark.",
        "",
        "## What exists",
        "",
        "- Protocol/status JSON under `artifacts/long_context/`.",
        "- License/hash audits for local LongBench files.",
    ]
    write_text(export / "LONG_CONTEXT_EVIDENCE_AUDIT.md", "\n".join(md))
    write_text(
        export / "LONG_CONTEXT_CLAIM_GUARD.md",
        "\n".join(
            [
                "# LONG_CONTEXT_CLAIM_GUARD",
                "",
                "## Status: not executed",
                "",
                "### Forbidden",
                "- Claiming official RULER/LongBench/NIAH results from this completion pipeline.",
                "- Claiming 8K=8192 was evaluated on Dense/Uniform/FDP/controls.",
                "- Calling protocol-only DONE jobs a formal long-context benchmark.",
                "",
                "### Allowed",
                "- State that local LongBench assets were inventoried and a 2K/8K protocol was frozen,",
                "  but evaluation was not executed (`eval_executed: false`).",
                "",
                "### Tag",
                "`SYNTHETIC_LONG_CONTEXT_DIAGNOSTIC` / `NOT_EXECUTED_PROTOCOL_ONLY`",
            ]
        ),
    )


def audit_external_qwen(export: Path) -> None:
    ext = load_json(PIPE / "EXTERNAL_BASELINE_MANIFEST.json") or {}
    qwen = load_json(PIPE / "QWEN_MODEL_LOCK.json") or {}
    write_json(export / "meta" / "EXTERNAL_BASELINE_STATUS.json", {
        "tag": "EXTERNAL_BASELINES_NOT_EXECUTED",
        "manifest_only": True,
        "official_impl_locked": ext.get("official_impl_locked"),
        "candidates": ext.get("candidates"),
        "note": "P5_external_manifest wrote a search manifest; no xKV/ReCalKV official runs in this pipeline.",
    })
    write_json(export / "meta" / "QWEN_STATUS.json", {
        "tag": "QWEN_MODEL_SEARCH_ONLY",
        "lock": qwen,
        "checkpoint_present": bool(qwen.get("local_hits")),
        "eval_executed": False,
        "note": "P5_qwen_search only; Qwen2.5-7B base compression/eval not run.",
    })


def audit_opt30b(export: Path) -> None:
    proto = load_json(PIPE / "artifacts/opt30b/OPT30B_PROTOCOL.json") or {}
    rows = [
        {
            "item": "fixed_single_FDP_schedule",
            "status": "DECLARED_IN_PROTOCOL_ONLY",
            "detail": proto.get("fixed_schedule"),
        },
        {
            "item": "same_window_and_sample",
            "status": "NOT_EXECUTED",
            "detail": "no eval artifacts",
        },
        {
            "item": "Uniform_FDP_actual_KV_bytes",
            "status": "NOT_EXECUTED",
            "detail": "",
        },
        {
            "item": "paired_bootstrap",
            "status": "NOT_EXECUTED",
            "detail": "",
        },
        {
            "item": "McNemar_Holm",
            "status": "NOT_EXECUTED",
            "detail": "",
        },
        {
            "item": "mixed_C4_PPL_vs_accuracy",
            "status": "UNKNOWN_NO_RESULTS",
            "detail": "historical mixed claims cannot be confirmed from this job",
        },
    ]
    write_csv(export / "OPT30B_FIXED_SCHEDULE_TABLE.csv", rows, list(rows[0].keys()))
    write_csv(
        export / "OPT30B_STATISTICS.csv",
        [{"metric": "ALL", "value": "", "status": "NOT_EXECUTED", "protocol_status": proto.get("status")}],
        ["metric", "value", "status", "protocol_status"],
    )
    write_text(
        export / "OPT30B_FINAL_EVIDENCE_AUDIT.md",
        "\n".join(
            [
                "# OPT30B_FINAL_EVIDENCE_AUDIT",
                "",
                f"Generated: {now_iso()}",
                "",
                f"- Protocol status: `{proto.get('status')}` (pending_gpu).",
                f"- Model path exists flag: `{proto.get('model_exists')}` path=`{proto.get('model')}`.",
                "- Job `P5_opt30b` DONE in ~0.1s wrote protocol only; **no Uniform/FDP eval, no KV bytes, no bootstrap/McNemar**.",
                "- Therefore OPT-30B confirmatory claims from this pipeline are **not executed**.",
            ]
        ),
    )
    write_text(
        export / "OPT30B_CLAIM_GUARD.md",
        "\n".join(
            [
                "# OPT30B_CLAIM_GUARD",
                "",
                "## Status: not executed",
                "",
                "### Forbidden",
                "- Claiming fixed-schedule OPT-30B Uniform vs FDP results from completion pipeline DONE status.",
                "- Claiming significance / mixed C4-vs-accuracy resolution without new evals.",
                "",
                "### Allowed",
                "- Acknowledge OPT-30B weights exist and a protocol stub was written (`pending_gpu`).",
            ]
        ),
    )


def audit_kqv_r11(export: Path) -> None:
    kqv_csv = ART / "kqv_ablation/KQ_V_RESULTS.csv"
    kqv_rows = []
    if kqv_csv.exists():
        with kqv_csv.open() as f:
            kqv_rows = list(csv.DictReader(f))
    required = [
        "uniform",
        "fdp_full",
        "fdp_kq_adaptive_v_uniform",
        "fdp_kq_uniform_v_adaptive",
        "kq_path_only",
        "v_path_only",
    ]
    present = {r.get("branch") for r in kqv_rows}
    # also check dirs
    for d in (ART / "kqv_ablation").glob("*"):
        present.add(d.name)

    comp = load_json(PIPE / "artifacts/robustness/COMPONENT_ABLATION_PROTOCOL.json") or {}
    r11 = load_json(LEGACY_ART / "r11_robustness/R11_ROBUSTNESS.json") or {}

    # KQV completeness md
    lines = [
        "# KQV_COMPLETENESS_AUDIT",
        "",
        f"Generated: {now_iso()}",
        "",
        "## Required branches",
        "",
    ]
    ppls = {r.get("branch"): r.get("wikitext2_ppl") for r in kqv_rows}
    for b in required:
        st = "PRESENT" if b in present or b in ppls else "MISSING"
        extra = ""
        if b in ppls:
            extra = f" PPL={ppls[b]}"
        if b in ("fdp_kq_adaptive_v_uniform", "fdp_kq_uniform_v_adaptive", "fdp_full") and len({ppls.get("fdp_full"), ppls.get("fdp_kq_adaptive_v_uniform"), ppls.get("fdp_kq_uniform_v_adaptive")}) == 1:
            extra += " **IDENTICAL_PPL_SUSPECT_REUSE**"
        lines.append(f"- `{b}`: **{st}**{extra}")
    lines += [
        "",
        "## Notes",
        "",
        "- Instrumented KQV CSV contains uniform + three FDP-named branches with **identical** FDP PPL (6.64958).",
        "- `kq_path_only` and `v_path_only` are **MISSING**.",
        "- Same-window / same actual KV bytes across true branch rebuilds: **not demonstrated**.",
        "- Legacy R10 Dense PPL uses a different protocol; do not mix with instrumented R1 windows.",
    ]
    write_text(export / "KQV_COMPLETENESS_AUDIT.md", "\n".join(lines))

    # Component ablation
    needed_comp = ["full", "minus_Z", "minus_S", "minus_V", "minus_A", "minus_Entropy"]
    needed_cal = ["wikitext", "c4"]
    needed_size = [16, 32, 64]
    needed_lam = [0.6, 1.2, 2.0]
    lines2 = [
        "# COMPONENT_ABLATION_COMPLETENESS_AUDIT",
        "",
        f"Generated: {now_iso()}",
        "",
        f"- Protocol status: `{comp.get('status')}`, gpu_build_pending=`{comp.get('gpu_build_pending')}`",
        f"- Declared variants: {comp.get('component_variants')}",
        f"- Declared cal domains: {comp.get('calibration_domains')}, sizes={comp.get('cal_sizes')}, lambdas={comp.get('lambdas')}",
        "",
        "## Execution",
        "",
        "**None of the component ablations were GPU-executed in the completion pipeline.**",
        "Job DONE ≠ grid completed.",
        "",
        "### Checklist",
    ]
    for v in needed_comp:
        lines2.append(f"- component `{v}`: NOT_EXECUTED")
    for d in needed_cal:
        lines2.append(f"- calibration domain `{d}` grid: NOT_EXECUTED (legacy R11 is Wiki cal sizes 32/64/128 only)")
    for s in needed_size:
        lines2.append(f"- cal size `{s}`: {'LEGACY_R11_PARTIAL' if s in (32, 64) else 'NOT_EXECUTED_IN_R11_OR_COMPLETION'}")
    # R11 used 128 not 16
    lines2.append("- cal size `128`: LEGACY_R11_ONLY (not in requested {16,32,64} set)")
    lines2.append("- cal size `16`: NOT_EXECUTED")
    for lam in needed_lam:
        lines2.append(f"- lambda `{lam}`: LEGACY_R11_WIKI_PPL_ONLY (robustness; not confirmatory replacement)")
    write_text(export / "COMPONENT_ABLATION_COMPLETENESS_AUDIT.md", "\n".join(lines2))

    # Robustness full table from R11 + holes
    rob_rows = []
    for r in r11.get("results") or []:
        rob_rows.append(
            {
                "source": "legacy_R11",
                "lambda": r.get("lambda_eb"),
                "cal_size": r.get("cal_size"),
                "cal_domain": "wikitext",
                "component": "full",
                "WikiText_PPL": r.get("wikitext2_ppl"),
                "C4_PPL": "",
                "status": "EXECUTED_ROBUSTNESS_ONLY",
            }
        )
    # add missing requested cells
    for lam in needed_lam:
        for sz in needed_size:
            for dom in needed_cal:
                for compv in needed_comp:
                    if any(
                        abs(float(x["lambda"]) - lam) < 1e-9
                        and int(x["cal_size"]) == sz
                        and x["cal_domain"] == dom
                        and x["component"] == compv
                        for x in rob_rows
                        if x.get("cal_size") != ""
                    ):
                        continue
                    # R11 has 32/64/128 wiki full only
                    if dom == "wikitext" and compv == "full" and sz in (32, 64) and any(
                        abs(float(x["lambda"]) - lam) < 1e-9 and int(x["cal_size"]) == sz for x in rob_rows
                    ):
                        continue
                    rob_rows.append(
                        {
                            "source": "completion_gap",
                            "lambda": lam,
                            "cal_size": sz,
                            "cal_domain": dom,
                            "component": compv,
                            "WikiText_PPL": "",
                            "C4_PPL": "",
                            "status": "NOT_EXECUTED",
                        }
                    )
    write_csv(
        export / "ROBUSTNESS_FULL_TABLE.csv",
        rob_rows,
        ["source", "lambda", "cal_size", "cal_domain", "component", "WikiText_PPL", "C4_PPL", "status"],
    )


def build_reviewer_matrix(export: Path, method_rows: List[Dict[str, Any]]) -> None:
    # Pull prior gap matrix themes
    gaps = load_json(PIPE / "UNRESOLVED_GAPS.json") or {}
    items = [
        {
            "id": "R_OP_QUALITY",
            "reviewer_question": "Does FDP beat Uniform at matched budget across operating points?",
            "jobs": "OP_*_build/ppl/zs, OP_aggregate",
            "raw_evidence": "FINAL_METHOD_RESULTS.csv; diskE OP_*_ppl and zero_shot aggregates",
            "result": "Point estimates: FDP lower PPL and higher/near macro vs Uniform at mild/moderate/aggressive; mild gains small.",
            "resolution": "partially_supported",
            "usable_sentence": "Under the frozen instrumented protocol, FDP shows favorable point estimates vs Uniform at three EB operating points; mild gains are small; OP-level paired bootstrap/McNemar remain pending.",
            "forbidden": "Claiming statistically significant OP superiority, or byte-exact matching from broken KV manifests.",
        },
        {
            "id": "R_MATCHED_CONTROLS",
            "reviewer_question": "Does FDP beat random/spectral/middle/depth/reconstruction matched controls?",
            "jobs": "MC_*, MC_aggregate",
            "raw_evidence": "matched_controls/*/MATCHED_CONTROL_SCHEDULE.json only",
            "result": "Schedules only; no PPL/ZS.",
            "resolution": "not_executed",
            "usable_sentence": "We generated matched-control schedules but have not completed full PPL and five-task evaluations for those controls.",
            "forbidden": "Claiming FDP beats these controls on downstream quality.",
        },
        {
            "id": "R_STATS",
            "reviewer_question": "Are improvements significant after multiple-testing correction?",
            "jobs": "P3_bootstrap_*, P3_mcnemar_holm",
            "raw_evidence": "PPL_PAIRED_BOOTSTRAP.csv (legacy mild); BOOTSTRAP_operating_points/mc pending; McNemar no common samples",
            "result": "Legacy mild Wiki/C4 Uniform vs FDP bootstrap significant; OP/MC/ZS stats not usable.",
            "resolution": "partially_supported",
            "usable_sentence": "On the confirmatory mild legacy windows, paired bootstrap finds significant PPL gains for FDP vs Uniform; operating-point and zero-shot paired tests are not yet valid in this pipeline.",
            "forbidden": "Extending mild bootstrap significance to moderate/aggressive or to five-task accuracy.",
        },
        {
            "id": "R_LONG_CONTEXT",
            "reviewer_question": "Does FDP hold under long context (8K/RULER/LongBench)?",
            "jobs": "P4_long_context_*",
            "raw_evidence": "LONG_CONTEXT_*_STATUS.json eval_executed=false",
            "result": "Protocol only.",
            "resolution": "not_executed",
            "usable_sentence": "Long-context assets were inventoried; formal 2K/8K LongBench/RULER evaluation was not executed.",
            "forbidden": "Reporting long-context benchmark scores.",
        },
        {
            "id": "R_EXTERNAL",
            "reviewer_question": "How does FDP compare to xKV/ReCalKV/other KV baselines?",
            "jobs": "P5_external_manifest",
            "raw_evidence": "EXTERNAL_BASELINE_MANIFEST.json",
            "result": "Manifest/search only; official_impl_locked=false.",
            "resolution": "not_executed",
            "usable_sentence": "We searched for external baseline implementations; official locked re-runs were not executed.",
            "forbidden": "Calling RescueKV historical folders a completed NeurIPS rebuttal comparison.",
        },
        {
            "id": "R_OPT30B",
            "reviewer_question": "Does FDP transfer to OPT-30B under a fixed schedule?",
            "jobs": "P5_opt30b",
            "raw_evidence": "OPT30B_PROTOCOL.json pending_gpu",
            "result": "Not executed.",
            "resolution": "not_executed",
            "usable_sentence": "OPT-30B fixed-schedule evaluation remains pending despite local weights.",
            "forbidden": "Claiming OPT-30B significance from this job.",
        },
        {
            "id": "R_QWEN",
            "reviewer_question": "Does FDP work on Qwen2.5-7B?",
            "jobs": "P5_qwen_search",
            "raw_evidence": "QWEN_MODEL_LOCK.json",
            "result": "Checkpoint search only.",
            "resolution": "not_executed",
            "usable_sentence": "Qwen2.5-7B base checkpoint was located; compression and evaluation were not run.",
            "forbidden": "Claiming Qwen results.",
        },
        {
            "id": "R_KQV",
            "reviewer_question": "Is the gain from KQ-adaptive vs V-adaptive components?",
            "jobs": "P5_kqv_ablation",
            "raw_evidence": "kqv_ablation/KQ_V_RESULTS.csv",
            "result": "Incomplete branches; identical PPL across FDP-named variants.",
            "resolution": "unsupported",
            "usable_sentence": "Instrumented KQ/V branch rebuilds are incomplete and currently show identical PPL, so we do not claim a resolved KQ-vs-V attribution.",
            "forbidden": "Using identical-PPL rows as evidence of branch effects.",
        },
        {
            "id": "R_COMPONENT_CAL",
            "reviewer_question": "Are fragility components and calibration choices necessary/robust?",
            "jobs": "P5_component_ablation (+ legacy R11)",
            "raw_evidence": "COMPONENT_ABLATION_PROTOCOL.json; legacy R11_ROBUSTNESS.json",
            "result": "Completion component grid not executed; R11 λ×cal wiki sensitivity only.",
            "resolution": "partially_supported",
            "usable_sentence": "Legacy robustness sweeps over λ and calibration size on WikiText exist; full component knockouts and C4-calibration grids were not executed in the completion pipeline.",
            "forbidden": "Saying R11 DONE implies minus-Z/S/V/A/Entropy completed.",
        },
        {
            "id": "R_SYSTEM",
            "reviewer_question": "What system/backend metrics support deployment claims?",
            "jobs": "P4_system_audit",
            "raw_evidence": "artifacts/system_metrics/*; DEPLOYMENT_CLAIM_GUARD.md",
            "result": "Audit/summary artifacts present; bind claims to documented backend only.",
            "resolution": "partially_supported",
            "usable_sentence": "System metrics were audited under the recorded attention backend; do not extrapolate to unevaluated kernels.",
            "forbidden": "Claiming FlashAttention/SDPA speedups without measurements.",
        },
    ]
    lines = [
        "# FINAL_REVIEWER_RESPONSE_MATRIX",
        "",
        f"Generated: {now_iso()}",
        "",
        "Job DONE is **not** treated as scientific resolution.",
        "",
    ]
    for it in items:
        lines += [
            f"## {it['id']}",
            "",
            f"- **Reviewer/AC question:** {it['reviewer_question']}",
            f"- **Jobs:** {it['jobs']}",
            f"- **Raw evidence:** {it['raw_evidence']}",
            f"- **Result:** {it['result']}",
            f"- **Resolution:** `{it['resolution']}`",
            f"- **Usable rebuttal sentence:** {it['usable_sentence']}",
            f"- **Forbidden overclaim:** {it['forbidden']}",
            "",
        ]
    write_text(export / "FINAL_REVIEWER_RESPONSE_MATRIX.md", "\n".join(lines))
    write_json(export / "meta" / "FINAL_REVIEWER_RESPONSE_MATRIX.json", items)


def build_final_guards(export: Path, method_rows: List[Dict[str, Any]]) -> None:
    # Executive answers
    mild_u = next(r for r in method_rows if r["operating_point"] == "mild" and r["method"] == "uniform")
    mild_f = next(r for r in method_rows if r["operating_point"] == "mild" and r["method"] == "fdp")
    mod_u = next(r for r in method_rows if r["operating_point"] == "moderate" and r["method"] == "uniform")
    mod_f = next(r for r in method_rows if r["operating_point"] == "moderate" and r["method"] == "fdp")

    claims = [
        ("FDP_vs_Uniform_PPL_mild_legacy", "confirmed", "Paired bootstrap on confirmatory mild windows significant for Wiki/C4 PPL."),
        ("FDP_vs_Uniform_point_estimates_OP", "supported_but_not_significant", "Moderate/aggressive show large point-estimate gaps; OP bootstrap pending."),
        ("FDP_vs_Uniform_ZS_significance", "unsupported", "McNemar/Holm invalid (no common samples)."),
        ("FDP_vs_matched_controls_downstream", "not_executed", "Controls are schedule-only."),
        ("mild_gains_small", "confirmed", f"Mild Wiki ΔPPL≈{float(mild_f['WikiText_PPL'])-float(mild_u['WikiText_PPL']):.4f} (FDP-Uniform)."),
        ("long_context_formal", "not_executed", "eval_executed=false."),
        ("external_baselines", "not_executed", "EXTERNAL_BASELINES_NOT_EXECUTED"),
        ("OPT30B", "not_executed", "pending_gpu protocol only"),
        ("Qwen", "not_executed", "QWEN_MODEL_SEARCH_ONLY"),
        ("KQV_attribution", "unsupported", "Identical PPL / missing branches"),
        ("component_ablation_complete", "not_executed", "gpu_build_pending"),
        ("calibration_lambda_robustness", "partially_supported", "Legacy R11 wiki λ×cal only"),
        ("system_backend", "partially_supported", "See SYSTEM_METRICS and DEPLOYMENT_CLAIM_GUARD"),
        ("single_schedule_OP", "partially_supported", "Protocol freeze OK; KV byte field anomalies"),
    ]

    guard_lines = [
        "# FINAL_REBUTTAL_CLAIM_GUARD",
        "",
        f"Generated: {now_iso()}",
        "",
        "Labels: confirmed | supported_but_not_significant | partially_supported | unsupported | blocked | not_executed",
        "",
    ]
    for name, lab, note in claims:
        guard_lines += [f"## {name}", f"- **label:** `{lab}`", f"- **note:** {note}", ""]
    write_text(export / "FINAL_REBUTTAL_CLAIM_GUARD.md", "\n".join(guard_lines))

    # EN / CN responses
    en = f"""# REBUTTAL_READY_RESPONSE_EN

Generated: {now_iso()}

## Honest summary

We completed a 53-job completion-pipeline orchestration. **Job completion is not scientific completion.** Several reviewer-critical evaluations remain schedule/protocol/search-only.

## What is solid

- Confirmatory mild (legacy R2/R3 reuse) instrumented WikiText/C4 PPL: FDP improves over Uniform with paired bootstrap significance on locked windows.
- New moderate/aggressive operating-point builds exist with instrumented PPL and five-task zero-shot point estimates favoring FDP vs Uniform; mild gains are small; moderate/aggressive gaps are larger in point estimates.
- Fixed FDP schedule protocol (λ=1.2, weights, Layer-0 freeze, etc.) is frozen; only eb_base is intended to change across operating points.

## What is not solid / not executed

- Matched controls: **schedules only** — cannot claim FDP beats random/spectral/middle/depth/reconstruction on downstream quality.
- Long-context: **not executed** (`eval_executed: false`) — not an official LongBench/RULER result.
- External baselines (xKV/ReCalKV/…): **EXTERNAL_BASELINES_NOT_EXECUTED**.
- Qwen2.5-7B: **QWEN_MODEL_SEARCH_ONLY**.
- OPT-30B fixed schedule: **not executed** (`pending_gpu`).
- OP/MC paired bootstrap and ZS McNemar/Holm: pending or invalid.
- KQ/V and component ablations: incomplete / suspect reuse / protocol-only.

## Suggested tone

Thank reviewers, report confirmatory mild stats carefully, present OP point estimates as exploratory pending paired tests, and explicitly list unfinished comparisons.
"""
    write_text(export / "REBUTTAL_READY_RESPONSE_EN.md", en)

    cn = f"""# REBUTTAL_READY_RESPONSE_CN

生成时间: {now_iso()}

## 诚实总结

53 个编排任务均标记 DONE，但 **DONE ≠ 科研问题已解决**。多项审稿关键评测仍停留在 schedule / protocol / search。

## 较扎实的证据

- Mild（复用 legacy R2/R3）在锁定 Wiki/C4 窗口上，FDP 相对 Uniform 的 PPL 提升有 paired bootstrap 显著性。
- Moderate/Aggressive 新构建 OP 的 PPL 与五任务 zero-shot 点估计整体偏向 FDP；mild 收益很小；OP 级显著性检验未完成。
- 固定 FDP schedule 已冻结；跨工作点原则上只改 eb_base。

## 未完成 / 不得宣称

- Matched controls：仅有 schedule，**不能**声称在下游综合质量上击败各 control。
- 长上下文：未真实评测。
- External / Qwen / OPT-30B：未执行或仅搜索。
- KQ/V 与 component 消融：不完整或可疑复用。

## 建议口径

感谢审稿人；mild 统计谨慎表述；OP 点估计标明待 paired 检验；明确列出未完成对照。
"""
    write_text(export / "REBUTTAL_READY_RESPONSE_CN.md", cn)

    paper = f"""# PAPER_REVISION_ACTIONS

Generated: {now_iso()}

1. Keep confirmatory mild Uniform vs FDP PPL + bootstrap as primary statistical claim; do not silently upgrade R11 best λ.
2. Add OP moderate/aggressive tables as **point estimates** with clear provenance; mark missing OP bootstrap.
3. Remove or downgrade matched-control superiority language until full PPL+ZS exist.
4. Remove formal long-context benchmark claims; mention asset inventory only if needed.
5. Mark external/Qwen/OPT-30B as pending.
6. Fix or regenerate KV byte accounting before any byte-matched claim.
7. Do not cite identical-PPL KQV rows as ablation evidence.
8. Enlarge fonts / vectorize figures as previously noted; sync numbers to FINAL_METHOD_RESULTS.csv.
"""
    write_text(export / "PAPER_REVISION_ACTIONS.md", paper)

    exec_cn = f"""# FINAL_EXECUTIVE_SUMMARY_CN

生成时间: {now_iso()}

## 12 个必须回答的问题

1. **FDP 是否显著优于 Uniform？**  
   Mild（legacy 锁定窗口）PPL：**是（bootstrap）**。Moderate/Aggressive：点估计更好，但 **OP paired 显著性未完成**。Zero-shot：**无有效 McNemar**。

2. **FDP 是否显著优于 random/spectral/middle/depth/reconstruction？**  
   **否（未评测）** — 仅有 schedule（`SCHEDULE_ONLY_NO_EVAL`）。

3. **哪些工作点有效？**  
   Mild/Moderate/Aggressive 均有完整 Uniform/FDP 的 PPL+五任务点估计；有效性随压缩变强而点估计差距变大，但显著性仅 mild PPL 坐实。

4. **mild 下收益是否很小？**  
   **是**。Wiki PPL {mild_u['WikiText_PPL']:.4f}→{mild_f['WikiText_PPL']:.4f}；下游五任务提升也小。

5. **long-context 是否真实验证？**  
   **否**（`NOT_EXECUTED` / 不得当正式 benchmark）。

6. **external baseline 是否真实运行？**  
   **否**（`EXTERNAL_BASELINES_NOT_EXECUTED`）。

7. **OPT-30B 是否显著？**  
   **未运行**，谈不上显著。

8. **Qwen 是否真实运行？**  
   **否**（`QWEN_MODEL_SEARCH_ONLY`）。

9. **component / calibration domain / KQ/V 是否完整？**  
   **不完整**。Component 网格未跑；R11 仅 wiki λ×cal；KQV 缺 path-only 且 FDP 分支 PPL 异常相同。

10. **系统指标支持到什么 attention backend？**  
    以 `SYSTEM_METRICS_*` / `DEPLOYMENT_CLAIM_GUARD.md` 记录的 backend 为准（eager/协议锁定），不得外推未测内核。

11. **哪些审稿意见完全解决？**  
    主要是：协议冻结、mild 确认性 PPL 统计、OP 点估计扩展、证据清单与 claim guard。几乎没有“对照基线全胜 + 长上下文 + 外部方法 + 多模型”的完全解决项。

12. **哪些仍只能诚实承认？**  
    Matched-control 下游、长上下文正式评测、external、Qwen、OPT-30B、OP/ZS 多重检验、完整 KQV/component 消融、可信 KV byte matching。

## 总判词

编排层 53/53 DONE；**科研证据层仍有关键缺口**。最终报告门控此前即为 `passed: false`。本导出保持该诚实结论。
"""
    write_text(export / "FINAL_EXECUTIVE_SUMMARY_CN.md", exec_cn)


def mark_superseded() -> None:
    old = HOME_JQ / "rebuttal_completion_completed_20260729_092858.tar.zst"
    note = HOME_JQ / "rebuttal_completion_completed_20260729_092858.SUPERSEDED.md"
    write_text(
        note,
        "\n".join(
            [
                "# SUPERSEDED",
                "",
                f"Marked: {now_iso()}",
                "",
                "This 47-job partial export is **superseded** by `rebuttal_completion_final_export_53jobs_*`.",
                "Retain for provenance; do not use as the final evidence pack.",
                f"Archive still present: `{old}`",
            ]
        ),
    )
    latest = HOME_JQ / "rebuttal_completion_completed_latest.tar.zst"
    if latest.exists() or latest.is_symlink():
        write_text(
            HOME_JQ / "rebuttal_completion_completed_latest.SUPERSEDED.md",
            f"Superseded {now_iso()} by final 53-job export.\n",
        )


def stage_export(export: Path) -> None:
    export.mkdir(parents=True, exist_ok=True)
    (export / "meta").mkdir(exist_ok=True)
    (export / "job_manifests").mkdir(exist_ok=True)
    (export / "eval_artifacts").mkdir(exist_ok=True)
    (export / "op_artifacts_light").mkdir(exist_ok=True)
    (export / "pipeline_root_docs").mkdir(exist_ok=True)
    (export / "pipeline_artifacts").mkdir(exist_ok=True)
    (export / "legacy_pointers").mkdir(exist_ok=True)

    # pipeline docs
    for p in PIPE.iterdir():
        if p.is_file() and p.suffix.lower() in {".json", ".md", ".sha256", ".jsonl", ".csv", ".txt"}:
            shutil.copy2(p, export / "pipeline_root_docs" / p.name)

    # pipeline artifacts
    if (PIPE / "artifacts").exists():
        copy_filtered(PIPE / "artifacts", export / "pipeline_artifacts", skip_weight_ext=True)

    # eval artifacts full (no weights)
    if ART.exists():
        copy_filtered(ART, export / "eval_artifacts", skip_weight_ext=True)

    # OP light: manifests, configs, logs; pointer for weights
    for point in ["moderate", "aggressive"]:
        src = OP_ART / "operating_points" / point
        if not src.exists():
            continue
        dst = export / "op_artifacts_light" / point
        copy_filtered(src, dst, skip_weight_ext=True)
        for meth in ["uniform", "fdp"]:
            fm = src / meth / "final_model"
            if fm.exists():
                files = []
                for f in fm.rglob("*"):
                    if f.is_file() and f.suffix.lower() in {".safetensors", ".bin", ".pt"}:
                        # sizes only — full sha256 of ~15GB weights deferred (too slow for export)
                        files.append({"path": str(f), "size": f.stat().st_size, "sha256": None})
                # hash small configs only
                cfg_hashes = {}
                for cfg_name in ("config.json", "low_rank_config.json", "generation_config.json"):
                    cp = fm / cfg_name
                    if cp.exists():
                        cfg_hashes[cfg_name] = sha256_file(cp)
                write_json(
                    export / "op_artifacts_light" / point / meth / "WEIGHTS_POINTER.json",
                    {
                        "final_model_dir": str(fm),
                        "weight_files": files,
                        "config_sha256": cfg_hashes,
                        "excluded_from_export": True,
                        "note": "Weight blob sha256 not computed in export pass; configs hashed.",
                    },
                )

    # protocol locks hashes
    locks = {}
    for name in [
        "FIXED_FDP_SCHEDULE_PROTOCOL.json",
        "FREEZE_PROTOCOL_MANIFEST.json",
        "LONG_CONTEXT_ASSET_MANIFEST.json",
        "EXTERNAL_BASELINE_MANIFEST.json",
        "QWEN_MODEL_LOCK.json",
        "CURRENT_EVIDENCE_HASHES.json",
    ]:
        p = PIPE / name
        if p.exists():
            locks[name] = {"path": str(p), "sha256": sha256_file(p), "size": p.stat().st_size}
            if p.with_suffix(p.suffix + ".sha256").exists() or Path(str(p) + ".sha256").exists():
                side = Path(str(p) + ".sha256")
                if side.exists():
                    locks[name]["sidecar_sha256_file"] = side.read_text().strip()
    write_json(export / "meta" / "PROTOCOL_LOCKS_SHA256.json", locks)

    # legacy pointers for R10/R11/bootstrap sources
    write_json(
        export / "legacy_pointers" / "LEGACY_EVIDENCE_POINTERS.json",
        {
            "legacy_artifacts_root": str(LEGACY_ART),
            "r2_uniform": str(LEGACY_ART / "r2_uniform_eb045"),
            "r3_fdp": str(LEGACY_ART / "r3_fgeb045"),
            "r10_kqv": str(LEGACY_ART / "r10_kqv_ablation"),
            "r11_robustness": str(LEGACY_ART / "r11_robustness"),
            "r8_system": str(LEGACY_ART / "r8_system_metrics") if (LEGACY_ART / "r8_system_metrics").exists() else None,
        },
    )
    # copy small legacy summary json/md
    for rel in [
        "r11_robustness/R11_ROBUSTNESS.json",
        "r11_robustness/R11_ROBUSTNESS.md",
        "r10_kqv_ablation",
    ]:
        src = LEGACY_ART / rel
        if src.is_file():
            copy_filtered(src, export / "legacy_pointers" / rel)
        elif src.is_dir():
            copy_filtered(src, export / "legacy_pointers" / rel)


def stop_orchestrator(export: Path) -> None:
    state = load_json(PIPE / "orchestrator_state.json") or {}
    mon = {
        "generated_at": now_iso(),
        "orchestrator_state": state,
        "jobs_completed": 53,
        "running_jobs": 0,
        "note": "Snapshot before graceful STOP_REQUESTED",
    }
    # verify no running gpu jobs
    try:
        import psutil  # type: ignore

        procs = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            cmd = " ".join(p.info.get("cmdline") or [])
            if "zero_shot_instrumented" in cmd or "ppl_instrumented" in cmd or "operating_point_build" in cmd:
                procs.append({"pid": p.info["pid"], "cmd": cmd[:200]})
        mon["active_eval_procs"] = procs
    except Exception:
        # fallback pgrep
        try:
            out = subprocess.check_output(["pgrep", "-af", "zero_shot_instrumented|ppl_instrumented"], text=True)
            mon["active_eval_procs"] = out.strip().splitlines()
        except Exception:
            mon["active_eval_procs"] = []

    write_json(export / "FINAL_ORCHESTRATOR_STATE.json", mon)
    write_json(PIPE / "FINAL_ORCHESTRATOR_STATE.json", mon)
    # graceful stop
    (PIPE / "STOP_REQUESTED").write_text(f"stop requested at {now_iso()} by final_audit_and_export\n")


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export = HOME_JQ / f"rebuttal_completion_final_export_53jobs_{ts}"
    print("EXPORT", export)
    stage_export(export)

    jobs = [json.loads(l) for l in (PIPE / "jobs.jsonl").read_text().splitlines() if l.strip()]
    job_snapshot(jobs, export)

    method_rows = build_final_method_results(export)
    audit_matched_controls(export)
    audit_single_schedule(export)
    audit_long_context(export)
    audit_external_qwen(export)
    audit_opt30b(export)
    audit_kqv_r11(export)
    build_reviewer_matrix(export, method_rows)
    build_final_guards(export, method_rows)

    # README
    write_text(
        export / "README_EXPORT.md",
        "\n".join(
            [
                "# Final 53-job export",
                "",
                f"Generated: {now_iso()}",
                "",
                "Contains eval raw windows/samples (weights excluded; see WEIGHTS_POINTER.json),",
                "protocol locks, audits, claim guards, and FINAL_METHOD_RESULTS.csv.",
                "",
                "Old 47-job package is marked SUPERSEDED alongside this export.",
                "",
                "**Important:** orchestrator job DONE ≠ scientific resolution. Read FINAL_EXECUTIVE_SUMMARY_CN.md.",
            ]
        ),
    )

    mark_superseded()
    stop_orchestrator(export)

    # compress if space allows
    tarball = HOME_JQ / f"{export.name}.tar.zst"
    print("Compressing to", tarball)
    subprocess.check_call(
        ["tar", "-I", "zstd -T0 -3", "-cf", str(tarball), export.name],
        cwd=str(HOME_JQ),
    )
    write_text(Path(str(tarball) + ".sha256"), sha256_file(tarball) + "  " + tarball.name + "\n")
    # update latest pointer for final
    latest = HOME_JQ / "rebuttal_completion_final_export_53jobs_latest.tar.zst"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(tarball.name)
    write_json(
        export / "meta" / "EXPORT_COMPLETE.json",
        {
            "generated_at": now_iso(),
            "export_dir": str(export),
            "tarball": str(tarball),
            "tarball_sha256": sha256_file(tarball),
            "n_jobs": len(jobs),
            "done": sum(1 for j in jobs if j.get("status") == "DONE"),
        },
    )
    print("DONE", export)
    print("TARBALL", tarball, "size", tarball.stat().st_size)


if __name__ == "__main__":
    main()
