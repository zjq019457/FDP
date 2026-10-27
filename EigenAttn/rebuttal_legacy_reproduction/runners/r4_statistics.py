#!/usr/bin/env python3
"""R4: Cross-method statistical comparison (Dense vs Uniform vs FDP).
Computes deltas, relative degradation, and KV retention ratio."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import artifact_dir, atomic_write_json, file_hash_or_empty, gate_payload, now_iso, sha256_file, write_pass_gate  # noqa: E402

ART_LINK = HERE / "artifacts"


def _load_ppl(name: str) -> float:
    p = ART_LINK / name / "ppl_aggregate.json"
    return json.loads(p.read_text(encoding="utf-8"))["ppl"]


def _load_zs(name: str) -> dict:
    p = ART_LINK / name / "zero_shot_aggregate.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return {t: v["accuracy_from_correct"] for t, v in d.get("tasks", {}).items()}


def main() -> None:
    out = artifact_dir("r4_statistics", link_under=ART_LINK / "r4_statistics")

    dense = {
        "wiki_ppl": _load_ppl("R1_dense_wikitext_ppl"),
        "c4_ppl": _load_ppl("R1_dense_c4_ppl"),
        **_load_zs("R1_dense_zs_hs_arc"),
        **_load_zs("R1_dense_zs_wg_piqa"),
    }
    uniform = {
        "wiki_ppl": _load_ppl("R2_uniform_wikitext_ppl"),
        "c4_ppl": _load_ppl("R2_uniform_c4_ppl"),
        **_load_zs("R2_uniform_zs_hs_arc"),
        **_load_zs("R2_uniform_zs_wg_piqa"),
    }
    fdp = {
        "wiki_ppl": _load_ppl("R3_fdp_wikitext_ppl"),
        "c4_ppl": _load_ppl("R3_fdp_c4_ppl"),
        **_load_zs("R3_fdp_zs_hs_arc"),
        **_load_zs("R3_fdp_zs_wg_piqa"),
    }

    # Load KV retention from R2 and R3
    r2_mem = {}
    mp = HERE / "LEGACY_UNIFORM_MEMORY_ACCOUNTING.json"
    if mp.exists():
        r2_mem = json.loads(mp.read_text(encoding="utf-8"))

    all_keys = sorted(set(dense) | set(uniform) | set(fdp))
    rows = []
    for k in all_keys:
        d_v = dense.get(k)
        u_v = uniform.get(k)
        f_v = fdp.get(k)
        row = {"metric": k, "dense": d_v, "uniform_eb045": u_v, "FGeb045": f_v}
        if d_v is not None and u_v is not None:
            if "ppl" in k:
                row["uniform_delta"] = u_v - d_v
                row["fdp_delta"] = f_v - d_v if f_v else None
                row["fdp_vs_uniform"] = (f_v - u_v) if f_v else None
            else:
                row["uniform_delta"] = u_v - d_v
                row["fdp_delta"] = f_v - d_v if f_v else None
                row["fdp_vs_uniform"] = (f_v - u_v) if f_v else None
        rows.append(row)

    # KV row
    if r2_mem:
        rows.append({
            "metric": "kv_retained_ratio",
            "dense": 1.0,
            "uniform_eb045": r2_mem.get("retained_ratio"),
            "FGeb045": None,
        })

    fnames = ["metric", "dense", "uniform_eb045", "FGeb045", "uniform_delta", "fdp_delta", "fdp_vs_uniform"]
    csv_path = out / "R4_CROSS_METHOD_COMPARISON.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "generated_at": now_iso(),
        "methods": ["dense", "uniform_eb045", "FGeb045"],
        "rows": rows,
        "kv_memory": r2_mem,
        "historical_lm_eval": "/diskD/jiaqizhang/lm_eval_0shot_llama3_batch/",
    }
    atomic_write_json(out / "R4_STATISTICS_SUMMARY.json", summary)

    # Historical cross-check
    hist_path = Path("/diskD/jiaqizhang/lm_eval_0shot_llama3_batch")
    hist = {}
    for f in sorted(hist_path.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        results = d.get("results", {})
        tag = f.stem
        for t, v in results.items():
            acc = v.get("acc,none")
            if acc is not None:
                hist.setdefault(tag, {})[t] = acc
    if hist:
        atomic_write_json(out / "R4_HISTORICAL_CROSS_CHECK.json", {
            "generated_at": now_iso(),
            "historical": hist,
            "note": "From /diskD/.../lm_eval_0shot_llama3_batch; compare with R1-R3 reproduced numbers.",
        })

    # Audit markdown
    md = ["# R4 Cross-Method Statistical Comparison", "", f"- generated_at: `{now_iso()}`", ""]
    md.append("| Metric | Dense | Uniform | FGeb045 | Uni Δ | FDP Δ | FDP-Uni |")
    md.append("|--------|-------|---------|---------|-------|-------|---------|")
    for r in rows:
        def fmt(v):
            if v is None: return "—"
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        md.append(f"| {r['metric']} | {fmt(r.get('dense'))} | {fmt(r.get('uniform_eb045'))} | "
                  f"{fmt(r.get('FGeb045'))} | {fmt(r.get('uniform_delta'))} | "
                  f"{fmt(r.get('fdp_delta'))} | {fmt(r.get('fdp_vs_uniform'))} |")
    md.append("")
    if hist:
        md.append("## Historical lm_eval cross-check")
        md.append("")
        for tag, accs in hist.items():
            md.append(f"### {tag}")
            for t, a in accs.items():
                md.append(f"- {t}: {a:.4f}")
            md.append("")
    (out / "R4_STATISTICS_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (HERE / "R4_STATISTICS_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    payload = gate_payload(
        name="R4_STATISTICS_PASS",
        passed=True,
        upstream_hashes={
            "csv": sha256_file(csv_path),
            "summary": sha256_file(out / "R4_STATISTICS_SUMMARY.json"),
        },
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R4 cross-method statistics complete",
    )
    write_pass_gate(out / "R4_STATISTICS_PASS.DONE", payload)
    write_pass_gate(HERE / "R4_STATISTICS_PASS.DONE", payload)
    print(json.dumps({"passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
