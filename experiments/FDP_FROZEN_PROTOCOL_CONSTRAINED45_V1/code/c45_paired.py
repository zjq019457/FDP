#!/usr/bin/env python3
"""FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1: paired FDP-EB minus Uniform (same bootstrap as the paper: PCG64 20261012/13 windows, 20261014 items)."""
import json, math
from pathlib import Path
import numpy as np

R = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1")
TASKS = ["arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande"]


def windows(d): return {r["window_id"]: (r["total_nll"], r["token_count"], r["input_ids_sha256"]) for r in map(json.loads, filter(str.strip, open(d / "ppl_windows.jsonl")))}
def items(d, t):
    p = list(d.glob(f"{t}_localjsonl*.jsonl")); assert len(p) == 1, (d, t)
    return {r["sample_id"]: (r["acc"] if "acc" in r else float(r["correct"]), r["document_hash"]) for r in map(json.loads, filter(str.strip, open(p[0])))}


def load(t, m):
    d = R / "tiers" / t / m
    return {"W": windows(d / "ppl_wikitext2_bf16"), "C": windows(d / "ppl_c4_bf16"), "Z": {k: items(d / "zs_bf16/output/zero_shot_samples", k) for k in TASKS}}


def paired(A, B):
    out = {}; reps = {}
    for key, ds, seed in (("W", "wikitext2", 20261012), ("C", "c4", 20261013)):
        a, b = A[key], B[key]; ids = sorted(a)
        assert ids == sorted(b) and all(a[i][2] == b[i][2] for i in ids)
        na = np.array([a[i][0] for i in ids]); nb = np.array([b[i][0] for i in ids]); tk = np.array([a[i][1] for i in ids], float)
        idx = np.random.Generator(np.random.PCG64(seed)).integers(0, len(ids), size=(10000, len(ids)))
        bs = (na[idx].sum(1) - nb[idx].sum(1)) / tk[idx].sum(1); reps[ds] = bs
        out[ds] = {"delta": float((na.sum() - nb.sum()) / tk.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    j = (reps["wikitext2"] + reps["c4"]) / 2
    out["J"] = {"delta": (out["wikitext2"]["delta"] + out["c4"]["delta"]) / 2, "ci95": [float(np.percentile(j, 2.5)), float(np.percentile(j, 97.5))]}
    rng = np.random.Generator(np.random.PCG64(20261014)); boots = []; per = {}
    for t in TASKS:
        a, b = A["Z"][t], B["Z"][t]; ids = sorted(a)
        assert ids == sorted(b) and all(a[i][1] == b[i][1] for i in ids)
        xa = np.array([a[i][0] for i in ids], float); xb = np.array([b[i][0] for i in ids], float)
        idx = rng.integers(0, len(ids), size=(10000, len(ids))); bs = 100 * (xa[idx].mean(1) - xb[idx].mean(1)); boots.append(bs)
        per[t] = {"delta": float(100 * (xa.mean() - xb.mean())), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    m = np.mean(boots, axis=0)
    out["acc_tasks"] = per; out["avg"] = {"delta": float(np.mean([v["delta"] for v in per.values()])), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
    return out


res = {}
for t in ("light", "medium", "aggressive"):
    res[t] = paired(load(t, "fdp"), load(t, "uniform"))
    p = res[t]
    print(t, "J %+.4f [%+.4f,%+.4f]" % (p["J"]["delta"], *p["J"]["ci95"]), "wiki %+.4f [%+.4f,%+.4f]" % (p["wikitext2"]["delta"], *p["wikitext2"]["ci95"]),
          "c4 %+.4f [%+.4f,%+.4f]" % (p["c4"]["delta"], *p["c4"]["ci95"]), "avg %+.2f [%+.2f,%+.2f]" % (p["avg"]["delta"], *p["avg"]["ci95"]),
          {k[:5]: round(v["delta"], 2) for k, v in p["acc_tasks"].items()})
(R / "PAIRED_FDP_MINUS_UNIFORM_PPL_ZS.json").write_text(json.dumps(res, indent=1))
