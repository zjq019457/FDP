#!/usr/bin/env python3
"""FLOOR45_CONTROLS_V1 paired comparisons (same bootstrap as the paper: PCG64 20261012/13 windows, 20261014 items).
delta = first - second; negative dlogPPL / positive dAcc favor the first arm."""
import json, sys
from pathlib import Path
import numpy as np

SRC = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/tiers")
NEW = Path("/diskF/jiaqizhang/FLOOR45_CONTROLS_V1/tiers")
TASKS = ["arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande"]


def windows(d): return {r["window_id"]: (r["total_nll"], r["token_count"], r["input_ids_sha256"]) for r in map(json.loads, filter(str.strip, open(d / "ppl_windows.jsonl")))}
def items(d, t):
    p = list(d.glob(f"{t}_localjsonl*.jsonl")); assert len(p) == 1, (d, t)
    return {r["sample_id"]: (r["acc"] if "acc" in r else float(r["correct"]), r["document_hash"]) for r in map(json.loads, filter(str.strip, open(p[0])))}


def load(root, t, m):
    d = root / t / m
    o = {"W": windows(d / "ppl_wikitext2_bf16"), "C": windows(d / "ppl_c4_bf16")}
    z = d / "zs_bf16/output/zero_shot_samples"
    if z.exists() and all(list(z.glob(f"{k}_localjsonl*.jsonl")) for k in TASKS):
        o["Z"] = {k: items(z, k) for k in TASKS}
    return o


def ppl(A, key):
    a = A[key]; n = sum(v[0] for v in a.values()); t = sum(v[1] for v in a.values())
    return float(np.exp(n / t))


def paired(A, B):
    out = {}; reps = {}
    for key, ds, seed in (("W", "wikitext2", 20261012), ("C", "c4", 20261013)):
        a, b = A[key], B[key]; ids = sorted(a)
        assert ids == sorted(b) and all(a[i][2] == b[i][2] for i in ids)
        na = np.array([a[i][0] for i in ids]); nb = np.array([b[i][0] for i in ids]); tk = np.array([a[i][1] for i in ids], float)
        idx = np.random.Generator(np.random.PCG64(seed)).integers(0, len(ids), size=(10000, len(ids)))
        bs = (na[idx].sum(1) - nb[idx].sum(1)) / tk[idx].sum(1); reps[ds] = bs
        out[ds] = {"delta": float((na.sum() - nb.sum()) / tk.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   "ppl_first": ppl(A, key), "ppl_second": ppl(B, key)}
    j = (reps["wikitext2"] + reps["c4"]) / 2
    out["J"] = {"delta": (out["wikitext2"]["delta"] + out["c4"]["delta"]) / 2, "ci95": [float(np.percentile(j, 2.5)), float(np.percentile(j, 97.5))]}
    if "Z" in A and "Z" in B:
        rng = np.random.Generator(np.random.PCG64(20261014)); boots = []; per = {}
        for t in TASKS:
            a, b = A["Z"][t], B["Z"][t]; ids = sorted(a)
            assert ids == sorted(b) and all(a[i][1] == b[i][1] for i in ids)
            xa = np.array([a[i][0] for i in ids], float); xb = np.array([b[i][0] for i in ids], float)
            idx = rng.integers(0, len(ids), size=(10000, len(ids))); bs = 100 * (xa[idx].mean(1) - xb[idx].mean(1)); boots.append(bs)
            per[t] = {"delta": float(100 * (xa.mean() - xb.mean())), "acc_first": float(100 * xa.mean()), "acc_second": float(100 * xb.mean())}
        m = np.mean(boots, axis=0)
        out["acc_tasks"] = per
        out["avg"] = {"delta": float(np.mean([v["delta"] for v in per.values()])), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                      "avg_first": float(np.mean([v["acc_first"] for v in per.values()])), "avg_second": float(np.mean([v["acc_second"] for v in per.values()]))}
    return out


def line(name, p):
    s = f"{name:40s} J {p['J']['delta']:+.4f} [{p['J']['ci95'][0]:+.4f},{p['J']['ci95'][1]:+.4f}]"
    for ds in ("wikitext2", "c4"):
        q = p[ds]; s += f" | {ds[:4]} {q['ppl_first']:.3f} vs {q['ppl_second']:.3f} d {q['delta']:+.4f} [{q['ci95'][0]:+.4f},{q['ci95'][1]:+.4f}]"
    if "avg" in p:
        a = p["avg"]; s += f" | avg {a['avg_first']:.2f} vs {a['avg_second']:.2f} d {a['delta']:+.2f} [{a['ci95'][0]:+.2f},{a['ci95'][1]:+.2f}]"
    print(s)


res = {}
for t in ("light", "medium", "aggressive"):
    U45, F45 = load(SRC, t, "uniform"), load(SRC, t, "fdp")
    UNF, T45 = load(NEW, t, "uniform_nofloor"), load(NEW, t, "spectral_tail45")
    for name, A, B in ((f"{t}: uniform_nofloor - uniform45", UNF, U45), (f"{t}: fdp45 - tail45", F45, T45), (f"{t}: tail45 - uniform45", T45, U45)):
        res[name] = paired(A, B); line(name, res[name])
Path("/diskF/jiaqizhang/FLOOR45_CONTROLS_V1/PAIRED_PARTIAL.json").write_text(json.dumps(res, indent=1))
