#!/usr/bin/env python3
"""Paired window bootstrap for the 10/3 extension (same convention as Severe V2 / 10/3: 10,000 reps, PCG64,
seeds wiki 20261012 / c4 20261013, percentile 95%). delta = logPPL(A) - logPPL(B). dirs contain wiki/ and c4/."""
import json, sys
import numpy as np
A, B = sys.argv[1:3]
def load(d):
    rows = [json.loads(x) for x in open(d + "/ppl_windows.jsonl") if x.strip()]
    return {r["window_id"]: (r["total_nll"], r["token_count"], r["input_ids_sha256"]) for r in rows}
out, reps = {}, {}
for ds, seed in [("wiki", 20261012), ("c4", 20261013)]:
    a, b = load(f"{A}/{ds}"), load(f"{B}/{ds}")
    ids = sorted(a); assert ids == sorted(b) and all(a[i][2] == b[i][2] for i in ids)
    na = np.array([a[i][0] for i in ids]); nb = np.array([b[i][0] for i in ids]); tk = np.array([a[i][1] for i in ids], float)
    rng = np.random.Generator(np.random.PCG64(seed)); idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    bs = (na[idx].sum(1) - nb[idx].sum(1)) / tk[idx].sum(1); reps[ds] = bs
    out[ds] = {"n": len(ids), "ppl_A": float(np.exp(na.sum() / tk.sum())), "ppl_B": float(np.exp(nb.sum() / tk.sum())),
               "delta_logppl": float((na.sum() - nb.sum()) / tk.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
j = (reps["wiki"] + reps["c4"]) / 2
out["J"] = {"delta": (out["wiki"]["delta_logppl"] + out["c4"]["delta_logppl"]) / 2, "ci95": [float(np.percentile(j, 2.5)), float(np.percentile(j, 97.5))], "p_A_better": float((j < 0).mean())}
print(json.dumps(out, indent=1))
