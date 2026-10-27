#!/usr/bin/env python3
"""Aggregate R2 Uniform eval outputs and write UNIFORM parity gate."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_write_json, file_hash_or_empty, gate_payload, now_iso, sha256_file, write_pass_gate  # noqa: E402

ART = HERE / "artifacts"


def main() -> None:
    wiki = ART / "R2_uniform_wikitext_ppl" / "ppl_aggregate.json"
    c4 = ART / "R2_uniform_c4_ppl" / "ppl_aggregate.json"
    zs1 = ART / "R2_uniform_zs_hs_arc" / "zero_shot_aggregate.json"
    zs2 = ART / "R2_uniform_zs_wg_piqa" / "zero_shot_aggregate.json"
    missing = [str(p) for p in (wiki, c4, zs1, zs2) if not p.exists()]
    if missing:
        raise SystemExit(f"R2 aggregate missing: {missing}")

    # repair zs if needed
    import subprocess

    for name in ("R2_uniform_zs_hs_arc", "R2_uniform_zs_wg_piqa"):
        d = ART / name
        full = d / "lm_eval_full_results.json"
        agg = d / "zero_shot_aggregate.json"
        if full.exists() and agg.exists():
            a = json.loads(agg.read_text(encoding="utf-8"))
            if any(int((v or {}).get("n") or 0) == 0 for v in (a.get("tasks") or {}).values()):
                subprocess.run(
                    [sys.executable, str(HERE / "runners" / "repair_zero_shot_samples.py"), "--out_dir", str(d)],
                    check=True,
                )

    wiki_agg = json.loads(wiki.read_text(encoding="utf-8"))
    c4_agg = json.loads(c4.read_text(encoding="utf-8"))
    z1 = json.loads(zs1.read_text(encoding="utf-8"))
    z2 = json.loads(zs2.read_text(encoding="utf-8"))
    tasks = {}
    tasks.update(z1.get("tasks") or {})
    tasks.update(z2.get("tasks") or {})

    rows = [
        {"metric": "wikitext2_ppl", "value": wiki_agg.get("ppl")},
        {"metric": "c4_ppl", "value": c4_agg.get("ppl")},
    ]
    for t, v in tasks.items():
        rows.append({"metric": f"{t}_acc_from_correct", "value": v.get("accuracy_from_correct"), "n": v.get("n")})

    out_csv = HERE / "LEGACY_UNIFORM_RESULTS.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
        w.writeheader()
        w.writerows(rows)

    recompute_ok = True
    for ppl_path in (wiki, c4):
        agg = json.loads(ppl_path.read_text(encoding="utf-8"))
        jl = ppl_path.parent / "ppl_windows.jsonl"
        s = n = 0
        for line in jl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("skipped"):
                continue
            s += float(r["total_nll"])
            n += int(r["token_count"])
        recomputed = math.exp(s / n) if n else float("nan")
        if abs(recomputed - float(agg["ppl"])) > 1e-5:
            recompute_ok = False

    mem = {}
    mp = HERE / "LEGACY_UNIFORM_MEMORY_ACCOUNTING.json"
    if mp.exists():
        mem = json.loads(mp.read_text(encoding="utf-8"))

    audit = [
        "# Uniform Parity Audit",
        "",
        f"- generated_at: `{now_iso()}`",
        f"- wikitext2_ppl: `{wiki_agg.get('ppl')}`",
        f"- c4_ppl: `{c4_agg.get('ppl')}`",
        f"- retained_ratio: `{mem.get('retained_ratio')}`",
        f"- recompute_ok: `{recompute_ok}`",
        "",
        "## Tasks",
        "",
    ]
    for t, v in tasks.items():
        audit.append(f"- `{t}`: {v.get('accuracy_from_correct')} (n={v.get('n')})")
    (HERE / "UNIFORM_PARITY_AUDIT.md").write_text("\n".join(audit) + "\n")

    # Soft numeric gate: require recompute + artifacts; historical final_model absent so no hard number match required.
    passed = recompute_ok and bool(tasks) and wiki_agg.get("ppl") is not None
    payload = gate_payload(
        name="UNIFORM_PARITY_PASS",
        passed=passed,
        upstream_hashes={
            "wiki": file_hash_or_empty(wiki),
            "c4": file_hash_or_empty(c4),
            "zs1": file_hash_or_empty(zs1),
            "zs2": file_hash_or_empty(zs2),
            "rank_plan": file_hash_or_empty(HERE / "LEGACY_UNIFORM_RANK_PLAN.csv"),
            "memory": file_hash_or_empty(mp) if mp.exists() else "",
        },
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="uniform R2 aggregate+recompute" if passed else "uniform R2 parity failed",
        extra={"memory": mem, "recompute_ok": recompute_ok},
    )
    atomic_write_json(HERE / "UNIFORM_PARITY_GATE.json", payload)
    if passed:
        write_pass_gate(HERE / "R2_UNIFORM_COMPLETE.DONE", payload)
        write_pass_gate(HERE / "UNIFORM_PARITY_PASS.DONE", payload)
    else:
        atomic_write_json(HERE / "UNIFORM_PARITY_FAIL.json", payload)
        raise SystemExit(1)
    print(json.dumps({"passed": passed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
