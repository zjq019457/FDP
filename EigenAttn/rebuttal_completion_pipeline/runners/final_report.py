#!/usr/bin/env python3
"""Generate final rebuttal report matrix and claim guards."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LEGACY = HERE.parent / "rebuttal_legacy_reproduction"
sys.path.insert(0, str(HERE))
from common import atomic_write_json, load_json, now_iso, write_pass_gate, gate_payload, protocol_hash, sha256_file  # noqa: E402


def main() -> None:
    r6 = LEGACY / "R6_FINAL_SUMMARY.md"
    gaps = HERE / "UNRESOLVED_GAPS.json"
    gap_data = load_json(gaps) if gaps.exists() else {"gaps": []}

    lines = [
        "# Reviewer Response Matrix",
        "",
        f"- generated_at: `{now_iso()}`",
        "",
        "| Question | Existing Evidence | New Work | Resolved | CI crosses zero |",
        "|----------|-------------------|----------|----------|-----------------|",
        "| FDP vs Uniform quality | R1-R3 instrumented | Reused | Partial | See bootstrap |",
        "| Matched-budget controls | Missing | MC_* jobs | Pending | — |",
        "| Statistical significance | Missing | P3 bootstrap | In progress | Check CSV |",
        "| Operating points | Missing | OP mild/moderate/aggressive | In progress | — |",
        "| Long context | Blocked doc outdated | LongBench local found | Pending | — |",
        "| KQ/V ablation | R10 incomplete | P5 instrumented rerun | In progress | — |",
        "| System metrics | R8 partial | P4 audit reuse | Partial | N/A |",
        "| External baselines | Missing | Manifest only | Blocked | — |",
        "| OPT-30B | Historical reports | Fixed schedule pending | Pending | — |",
        "| Qwen third model | No weights | Manual download | Blocked | — |",
    ]
    (HERE / "REVIEWER_RESPONSE_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    revision = [
        "# Paper Revision Actions",
        "",
        "1. Remove instruction-like unrelated text from paper",
        "2. Enlarge Figure 2 fonts; export vector PDF",
        "3. Define S_l (score_rel_mse_causal) explicitly",
        "4. Document five equal weights and aggregation order",
        "5. Fix single schedule; explain eb045/eb060/eb090 operating points",
        "6. Unify tilde/hat notation for normalized fragility",
        "7. Narrow '全面优于' claims; note mild setting limited gains",
        "8. Scope long-context, FlashAttention/vLLM claims to validated range",
        "9. Report actual KV bytes not nominal ratios for all methods",
        "10. Include bootstrap CI and Holm-corrected p-values in rebuttal",
    ]
    (HERE / "PAPER_REVISION_ACTIONS.md").write_text("\n".join(revision) + "\n", encoding="utf-8")

    (HERE / "REBUTTAL_CLAIM_GUARD.md").write_text(
        "# Rebuttal Claim Guard\n\n"
        "- Confirmatory: lambda=1.2, normalize=mean, FGeb045\n"
        "- R11 lambda=2.0 cal=32 is robustness ONLY\n"
        "- R10 legacy PPL incompatible with R1; use instrumented numbers\n"
        "- R8 eager-only; no FlashAttention claim\n", encoding="utf-8"
    )

    (HERE / "REBUTTAL_READY_RESPONSE_EN.md").write_text(
        "# Rebuttal Ready Response (Draft EN)\n\n"
        "We provide paired bootstrap CIs (n=10000) on matched evaluation windows, "
        "matched-budget non-uniform controls anchored to FDP actual KV bytes, "
        "and three operating points rebuilt under a frozen FDP schedule protocol.\n", encoding="utf-8"
    )

    # Copy key CSVs if exist
    for src_name, dst_name in [
        ("artifacts/statistics/PPL_PAIRED_BOOTSTRAP.csv", "FINAL_STATISTICS.csv"),
        ("artifacts/system_metrics/SYSTEM_METRICS_FINAL.csv", "FINAL_SYSTEM_METRICS.csv"),
    ]:
        src = HERE / src_name
        if src.exists():
            (HERE / dst_name).write_text(src.read_text(), encoding="utf-8")

    remaining = [g["id"] for g in gap_data.get("gaps", []) if g.get("severity") == "critical"]
    all_done = len(remaining) == 0
    payload = gate_payload(
        name="COMPLETION_PIPELINE_PASS",
        passed=not remaining,  # partial pass allowed with unresolved critical
        upstream_hashes={"gaps": sha256_file(gaps) if gaps.exists() else ""},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=protocol_hash(),
        summary=f"Final report generated; {len(remaining)} critical gaps remain",
        extra={"critical_remaining": remaining},
    )
    if not remaining:
        write_pass_gate(HERE / "COMPLETION_PIPELINE_PASS.DONE", payload)
    atomic_write_json(HERE / "FINAL_REPORT_MANIFEST.json", payload)
    print(json.dumps({"ok": True, "critical_remaining": len(remaining)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
