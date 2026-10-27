#!/usr/bin/env python3
"""CPU-only: paired bootstrap CI for the Llama-3 Medium QA-6 macro (FDP-EB minus EigenAttention/Official reference).
Per-sample scores use the frozen LongBench metric (slim/longbench_eval.dataset2metric, max over gold answers, x100),
matched by _id; aggregate check against lbe.scorer. Resampling: items within each task, equal-task macro,
10,000 reps, PCG64 seed 20261011 (same convention as the context-length strata analysis)."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/slim")
import longbench_eval as lbe

D = Path("/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/longbench_qa6_v1/medium")
TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]
out = {"source": str(D), "files": {}, "tasks": {}}
rng = np.random.Generator(np.random.PCG64(20261011))
boots = []
for t in TASKS:
    rows = {}
    for arm in ("official_uniform", "fdp125_direct"):
        p = D / arm / f"{t}.jsonl"; out["files"][f"{arm}/{t}"] = hashlib.sha256(p.read_bytes()).hexdigest()
        rows[arm] = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    def per(rs):
        m = lbe.dataset2metric[t]
        return {r["_id"]: 100 * max([m(r["pred"], a, all_classes=r.get("all_classes")) for a in r["answers"]] or [0.0]) for r in rs}
    a, b = per(rows["fdp125_direct"]), per(rows["official_uniform"])
    assert sorted(a) == sorted(b)
    ids = sorted(a); xa = np.array([a[i] for i in ids]); xb = np.array([b[i] for i in ids])
    agg = {arm: float(lbe.scorer(t, [r["pred"] for r in rs], [r["answers"] for r in rs], [int(r.get("length") or 0) for r in rs],
                                 next((r.get("all_classes") for r in rs if r.get("all_classes") is not None), None))) for arm, rs in rows.items()}
    idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    boots.append(xa[idx].mean(1) - xb[idx].mean(1))
    out["tasks"][t] = {"n": len(ids), "fdp": float(xa.mean()), "ref": float(xb.mean()), "scorer_fdp": agg["fdp125_direct"], "scorer_ref": agg["official_uniform"],
                       "delta": float(xa.mean() - xb.mean()), "ci95": [float(np.percentile(boots[-1], 2.5)), float(np.percentile(boots[-1], 97.5))]}
m = np.mean(boots, axis=0)
out["macro"] = {"fdp": float(np.mean([v["fdp"] for v in out["tasks"].values()])), "ref": float(np.mean([v["ref"] for v in out["tasks"].values()])),
                "delta": float(np.mean([v["delta"] for v in out["tasks"].values()])), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                "p_fdp_better": float((m > 0).mean()), "reps": 10000, "seed": 20261011}
print(json.dumps(out, indent=1))
