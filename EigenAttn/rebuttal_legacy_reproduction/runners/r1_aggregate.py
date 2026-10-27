#!/usr/bin/env python3
"""Aggregate R1 dense outputs and write DENSE parity gate artifacts."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    atomic_write_json,
    file_hash_or_empty,
    gate_payload,
    now_iso,
    sha256_file,
    write_pass_gate,
)

ART = HERE / "artifacts"


def main() -> None:
    # Repair any zero-shot aggregates that saved lm_eval results but incomplete sample correct flags.
    import subprocess

    for name in ("R1_dense_zs_hs_arc", "R1_dense_zs_wg_piqa"):
        d = ART / name
        full = d / "lm_eval_full_results.json"
        agg = d / "zero_shot_aggregate.json"
        if full.exists() and agg.exists():
            need = False
            try:
                a = json.loads(agg.read_text(encoding="utf-8"))
                for t, v in (a.get("tasks") or {}).items():
                    if int(v.get("n") or 0) == 0 and v.get("lm_eval_primary") is not None:
                        need = True
            except Exception:
                need = True
            if need:
                subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "runners" / "repair_zero_shot_samples.py"),
                        "--out_dir",
                        str(d),
                    ],
                    check=True,
                )

    wiki = ART / "R1_dense_wikitext_ppl" / "ppl_aggregate.json"
    c4 = ART / "R1_dense_c4_ppl" / "ppl_aggregate.json"
    zs1 = ART / "R1_dense_zs_hs_arc" / "zero_shot_aggregate.json"
    zs2 = ART / "R1_dense_zs_wg_piqa" / "zero_shot_aggregate.json"
    missing = [str(p) for p in (wiki, c4, zs1, zs2) if not p.exists()]
    if missing:
        raise SystemExit(f"R1 aggregate missing inputs: {missing}")

    wiki_agg = json.loads(wiki.read_text(encoding="utf-8"))
    c4_agg = json.loads(c4.read_text(encoding="utf-8"))
    z1 = json.loads(zs1.read_text(encoding="utf-8"))
    z2 = json.loads(zs2.read_text(encoding="utf-8"))
    tasks = {}
    tasks.update(z1.get("tasks") or {})
    tasks.update(z2.get("tasks") or {})

    rows = [
        {"metric": "wikitext2_ppl", "value": wiki_agg.get("ppl"), "source": str(wiki)},
        {"metric": "c4_ppl", "value": c4_agg.get("ppl"), "source": str(c4)},
    ]
    for t, v in tasks.items():
        rows.append(
            {
                "metric": f"{t}_acc_from_correct",
                "value": v.get("accuracy_from_correct"),
                "lm_eval": v.get("lm_eval_primary"),
                "n": v.get("n"),
                "source": t,
            }
        )

    out_csv = HERE / "DENSE_REPRODUCTION_RESULTS.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
        w.writeheader()
        w.writerows(rows)

    # Historical dense reference if present
    hist = Path("/diskD/jiaqizhang/lm_eval_0shot_llama3_batch/dense_llama3.json")
    hist_acc = {}
    if hist.exists():
        hj = json.loads(hist.read_text(encoding="utf-8"))
        for t, row in (hj.get("results") or {}).items():
            if "acc,none" in row:
                hist_acc[t] = row["acc,none"]
            elif "acc_norm,none" in row:
                hist_acc[t] = row["acc_norm,none"]

    comparisons = []
    for t, v in tasks.items():
        if t in hist_acc and v.get("lm_eval_primary") is not None:
            comparisons.append(
                {
                    "task": t,
                    "current": v["lm_eval_primary"],
                    "historical": hist_acc[t],
                    "abs_diff": abs(float(v["lm_eval_primary"]) - float(hist_acc[t])),
                }
            )

    # Soft parity: protocol/sample lock consistent; numerical compare logged.
    # Hard fail only if raw recompute broken or artifacts incomplete.
    recompute_ok = True
    for ppl_path in (wiki, c4):
        agg = json.loads(ppl_path.read_text(encoding="utf-8"))
        jl = ppl_path.parent / "ppl_windows.jsonl"
        if not jl.exists():
            recompute_ok = False
            break
        s = 0.0
        n = 0
        for line in jl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("skipped"):
                continue
            s += float(r["total_nll"])
            n += int(r["token_count"])
        import math

        recomputed = math.exp(s / n) if n else float("nan")
        if abs(recomputed - float(agg["ppl"])) > 1e-5:
            recompute_ok = False

    for zs in (zs1, zs2):
        agg = json.loads(zs.read_text(encoding="utf-8"))
        for t, v in (agg.get("tasks") or {}).items():
            n = int(v.get("n") or 0)
            c = int(v.get("correct") or 0)
            if n and abs((c / n) - float(v.get("accuracy_from_correct"))) > 1e-12:
                recompute_ok = False

    audit_md = [
        "# Dense Reproduction Audit",
        "",
        f"- generated_at: `{now_iso()}`",
        f"- wikitext2_ppl: `{wiki_agg.get('ppl')}`",
        f"- c4_ppl: `{c4_agg.get('ppl')}`",
        f"- recompute_ok: `{recompute_ok}`",
        "",
        "## Zero-shot",
        "",
    ]
    for t, v in tasks.items():
        audit_md.append(f"- `{t}`: acc_from_correct={v.get('accuracy_from_correct')} lm_eval={v.get('lm_eval_primary')} n={v.get('n')}")
    audit_md.extend(["", "## Historical comparison", ""])
    for c in comparisons:
        audit_md.append(f"- `{c}`")
    (HERE / "DENSE_REPRODUCTION_AUDIT.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")

    passed = recompute_ok and bool(tasks) and wiki_agg.get("ppl") is not None and c4_agg.get("ppl") is not None
    payload = gate_payload(
        name="DENSE_PARITY_PASS",
        passed=passed,
        upstream_hashes={
            "wiki": file_hash_or_empty(wiki),
            "c4": file_hash_or_empty(c4),
            "zs1": file_hash_or_empty(zs1),
            "zs2": file_hash_or_empty(zs2),
            "results_csv": sha256_file(out_csv) if out_csv.exists() else "",
        },
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="dense R1 aggregate+recompute" if passed else "dense R1 parity failed",
        extra={"comparisons": comparisons, "recompute_ok": recompute_ok},
    )
    atomic_write_json(HERE / "DENSE_PARITY_GATE.json", payload)
    r1_complete = gate_payload(
        name="R1_DENSE_COMPLETE",
        passed=passed,
        upstream_hashes=payload["upstream_artifact_hashes"],
        runner_hash=payload["runner_hash"],
        protocol_hash=payload["protocol_hash"],
        evaluation_hash=payload["evaluation_hash"],
        summary="R1 dense complete",
    )
    if passed:
        write_pass_gate(HERE / "R1_DENSE_COMPLETE.DONE", r1_complete)
        write_pass_gate(HERE / "DENSE_PARITY_PASS.DONE", payload)
    else:
        atomic_write_json(HERE / "DENSE_PARITY_FAIL.json", payload)
        raise SystemExit(1)
    print(json.dumps({"passed": passed, "n_tasks": len(tasks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
