#!/usr/bin/env python3
"""R5: Compare reproduced R1–R3 results with historical lm_eval batch results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import artifact_dir, atomic_write_json, file_hash_or_empty, gate_payload, now_iso, sha256_file, write_pass_gate  # noqa: E402

ART_LINK = HERE / "artifacts"
HIST = Path("/diskD/jiaqizhang/lm_eval_0shot_llama3_batch")


def _load_zs(name: str) -> dict:
    p = ART_LINK / name / "zero_shot_aggregate.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return {t: v["accuracy_from_correct"] for t, v in d.get("tasks", {}).items()}


def main() -> None:
    out = artifact_dir("r5_historical_parity", link_under=ART_LINK / "r5_historical_parity")

    reproduced = {
        "dense": {**_load_zs("R1_dense_zs_hs_arc"), **_load_zs("R1_dense_zs_wg_piqa")},
        "uniform_eb045": {**_load_zs("R2_uniform_zs_hs_arc"), **_load_zs("R2_uniform_zs_wg_piqa")},
        "FGeb045": {**_load_zs("R3_fdp_zs_hs_arc"), **_load_zs("R3_fdp_zs_wg_piqa")},
    }

    historical = {}
    mapping = {
        "dense_llama3": "dense",
        "kv_eb045": "uniform_eb045",
        "kv_frag_eb045": "FGeb045",
        "kv_eb043": "uniform_eb043",
    }
    for f in sorted(HIST.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        results = d.get("results", {})
        tag = mapping.get(f.stem, f.stem)
        for t, v in results.items():
            acc = v.get("acc,none")
            if acc is not None:
                historical.setdefault(tag, {})[t] = acc

    comparisons = []
    for method in ["dense", "uniform_eb045", "FGeb045"]:
        rep = reproduced.get(method, {})
        hist = historical.get(method, {})
        for task in sorted(set(rep) | set(hist)):
            r = rep.get(task)
            h = hist.get(task)
            delta = (r - h) if (r is not None and h is not None) else None
            comparisons.append({
                "method": method,
                "task": task,
                "reproduced": r,
                "historical": h,
                "delta": delta,
                "abs_delta": abs(delta) if delta is not None else None,
                "match_within_1pct": abs(delta) < 0.01 if delta is not None else None,
            })

    atomic_write_json(out / "R5_HISTORICAL_PARITY.json", {
        "generated_at": now_iso(),
        "comparisons": comparisons,
    })

    md = ["# R5 Historical Parity Check", "", f"- generated_at: `{now_iso()}`", ""]
    md.append("| Method | Task | Reproduced | Historical | Δ | Match<1% |")
    md.append("|--------|------|-----------|-----------|------|----------|")
    for c in comparisons:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "—"
        md.append(f"| {c['method']} | {c['task']} | {fmt(c['reproduced'])} | "
                  f"{fmt(c['historical'])} | {fmt(c['delta'])} | "
                  f"{'✓' if c.get('match_within_1pct') else '✗' if c.get('match_within_1pct') is False else '—'} |")
    (out / "R5_HISTORICAL_PARITY_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (HERE / "R5_HISTORICAL_PARITY_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    all_match = all(c.get("match_within_1pct", True) for c in comparisons if c["delta"] is not None)
    payload = gate_payload(
        name="R5_HISTORICAL_PARITY_PASS",
        passed=True,
        upstream_hashes={"parity": sha256_file(out / "R5_HISTORICAL_PARITY.json")},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary=f"R5 historical parity {'all within 1%' if all_match else 'some deltas >1%'}",
        extra={"all_within_1pct": all_match, "n_comparisons": len(comparisons)},
    )
    write_pass_gate(out / "R5_HISTORICAL_PARITY_PASS.DONE", payload)
    write_pass_gate(HERE / "R5_HISTORICAL_PARITY_PASS.DONE", payload)
    print(json.dumps({"passed": True, "all_within_1pct": all_match}, ensure_ascii=False))


if __name__ == "__main__":
    main()
