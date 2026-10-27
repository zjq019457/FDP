#!/usr/bin/env python3
"""Paired zero-shot CI with 10,000 within-task bootstrap replicates (PCG64 20261014), matching the Llama-3 analyses.
usage: l2x_zs_ci.py A_SAMPLES_DIR B_SAMPLES_DIR OUT_JSON   (delta = B - A)"""
import json, pathlib, sys
import numpy as np

TASKS = ["arc_challenge_localjsonl", "arc_easy_localjsonl", "hellaswag_localjsonl", "piqa_localjsonl", "winogrande_localjsonl"]


def load(d, t):
    rows = [json.loads(x) for x in (pathlib.Path(d) / f"{t}.jsonl").read_text().splitlines() if x.strip()]
    return {(r["sample_id"], r["document_hash"]): int(bool(r["correct"])) for r in rows}


def main(a_dir, b_dir, out):
    rng = np.random.Generator(np.random.PCG64(20261014)); boots, per = [], {}
    for t in TASKS:
        a, b = load(a_dir, t), load(b_dir, t)
        if set(a) != set(b): raise SystemExit(f"key mismatch {t}")
        k = sorted(a); xa = np.array([a[i] for i in k], float); xb = np.array([b[i] for i in k], float)
        idx = rng.integers(0, len(k), size=(10000, len(k))); bs = 100 * (xb[idx].mean(1) - xa[idx].mean(1)); boots.append(bs)
        per[t] = {"n": len(k), "A": 100 * xa.mean(), "B": 100 * xb.mean(), "delta": 100 * (xb.mean() - xa.mean()),
                  "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    m = np.mean(boots, 0)
    res = {"A": str(a_dir), "B": str(b_dir), "reps": 10000, "seed": 20261014, "tasks": per,
           "macro": {"A": float(np.mean([v["A"] for v in per.values()])), "B": float(np.mean([v["B"] for v in per.values()])),
                     "delta": float(np.mean([v["delta"] for v in per.values()])), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}}
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["macro"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
