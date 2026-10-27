#!/usr/bin/env python3
"""R6: Final summary and R6_STATISTICS_PASS gate. Collects all R1-R5 results."""
from __future__ import annotations

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
    out = artifact_dir("r6_summary", link_under=ART_LINK / "r6_summary")

    methods = {
        "Dense (R1)": {
            "wiki_ppl": _load_ppl("R1_dense_wikitext_ppl"),
            "c4_ppl": _load_ppl("R1_dense_c4_ppl"),
            **_load_zs("R1_dense_zs_hs_arc"),
            **_load_zs("R1_dense_zs_wg_piqa"),
        },
        "Uniform eb045 (R2)": {
            "wiki_ppl": _load_ppl("R2_uniform_wikitext_ppl"),
            "c4_ppl": _load_ppl("R2_uniform_c4_ppl"),
            **_load_zs("R2_uniform_zs_hs_arc"),
            **_load_zs("R2_uniform_zs_wg_piqa"),
        },
        "FGeb045 FDP (R3)": {
            "wiki_ppl": _load_ppl("R3_fdp_wikitext_ppl"),
            "c4_ppl": _load_ppl("R3_fdp_c4_ppl"),
            **_load_zs("R3_fdp_zs_hs_arc"),
            **_load_zs("R3_fdp_zs_wg_piqa"),
        },
    }

    # Load KV memory
    kv = {}
    mp = HERE / "LEGACY_UNIFORM_MEMORY_ACCOUNTING.json"
    if mp.exists():
        kv = json.loads(mp.read_text(encoding="utf-8"))

    # Collect gates
    gates = {}
    for g in ["DENSE_PARITY_PASS", "UNIFORM_PARITY_PASS", "LEGACY_FDP_PARITY_PASS",
              "R4_STATISTICS_PASS", "R5_HISTORICAL_PARITY_PASS"]:
        gates[g] = (HERE / f"{g}.DONE").exists()

    summary = {
        "generated_at": now_iso(),
        "methods": methods,
        "kv_memory": kv,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    atomic_write_json(out / "R6_FINAL_SUMMARY.json", summary)

    # Build comprehensive markdown
    md = [
        "# R6 Legacy Rebuttal Final Summary",
        "",
        f"- generated_at: `{now_iso()}`",
        f"- all_gates_passed: `{summary['all_gates_passed']}`",
        "",
        "## Results Table",
        "",
    ]
    all_metrics = sorted({k for m in methods.values() for k in m})
    header = "| Metric | " + " | ".join(methods.keys()) + " |"
    sep = "|--------|" + "|".join(["-------"] * len(methods)) + "|"
    md.extend([header, sep])
    for metric in all_metrics:
        vals = []
        for m_name in methods:
            v = methods[m_name].get(metric)
            vals.append(f"{v:.4f}" if isinstance(v, float) else "—")
        md.append(f"| {metric} | " + " | ".join(vals) + " |")

    if kv:
        md.extend([
            "",
            "## KV Cache Memory",
            "",
            f"- Dense KV bytes: `{kv.get('dense_kv_bytes')}`",
            f"- Uniform retained bytes: `{kv.get('realized_kv_bytes')}`",
            f"- Retained ratio: `{kv.get('retained_ratio')}`",
        ])

    md.extend(["", "## Gates", ""])
    for g, v in gates.items():
        md.append(f"- `{g}`: {'✓' if v else '✗'}")

    (out / "R6_FINAL_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (HERE / "R6_FINAL_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    payload = gate_payload(
        name="R6_STATISTICS_PASS",
        passed=summary["all_gates_passed"],
        upstream_hashes={
            "summary": sha256_file(out / "R6_FINAL_SUMMARY.json"),
            "r4": file_hash_or_empty(HERE / "R4_STATISTICS_PASS.DONE"),
            "r5": file_hash_or_empty(HERE / "R5_HISTORICAL_PARITY_PASS.DONE"),
        },
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R6 final statistics gate",
        extra={"gates": gates},
    )
    if summary["all_gates_passed"]:
        write_pass_gate(out / "R6_STATISTICS_PASS.DONE", payload)
        write_pass_gate(HERE / "R6_STATISTICS_PASS.DONE", payload)
    else:
        atomic_write_json(out / "R6_STATISTICS_FAIL.json", payload)
    print(json.dumps({"passed": summary["all_gates_passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
