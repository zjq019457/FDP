#!/usr/bin/env python3
"""Paired B-minus-A on Llama-2 (windows PCG64 20261012/13; items within task 20261014; 10,000 replicates).
usage: l2x_paired.py A_LABEL B_LABEL [A_LABEL B_LABEL ...]"""
import glob, json, math, pathlib, sys
import numpy as np
Q = pathlib.Path("/diskF/jiaqizhang/LLAMA2_EXCHANGE_EXPANSION_V1/quality")
TASKS = ["arc_challenge_localjsonl", "arc_easy_localjsonl", "hellaswag_localjsonl", "piqa_localjsonl", "winogrande_localjsonl"]


def wins(lab, ds):
    out = {}
    for f in glob.glob(str(Q / lab / "ppl" / ds / "windows" / "*.json")):
        d = json.load(open(f))
        if not d.get("skipped"): out[d["window_id"]] = (d["total_nll"], d["token_count"] - 1 if False else d["token_count"], d["input_ids_sha256"])
    return out


def items(lab, t):
    p = Q / lab / "zs" / lab / "zero_shot_samples" / f"{t}.jsonl"
    return {(r["sample_id"], r["document_hash"]): int(bool(r["correct"])) for r in map(json.loads, filter(str.strip, open(p)))}


def paired(a, b):
    res, reps = {}, {}
    for ds, seed in (("wikitext2", 20261012), ("c4", 20261013)):
        A, B = wins(a, ds), wins(b, ds); ids = sorted(set(A) & set(B))
        assert len(ids) == len(A) == len(B) and all(A[i][2] == B[i][2] for i in ids)
        na = np.array([A[i][0] for i in ids]); nb = np.array([B[i][0] for i in ids]); tk = np.array([A[i][1] for i in ids], float)
        idx = np.random.Generator(np.random.PCG64(seed)).integers(0, len(ids), (10000, len(ids)))
        bs = (nb[idx].sum(1) - na[idx].sum(1)) / tk[idx].sum(1); reps[ds] = bs
        res[ds] = ((nb.sum() - na.sum()) / tk.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    j = (reps["wikitext2"] + reps["c4"]) / 2
    res["J"] = ((res["wikitext2"][0] + res["c4"][0]) / 2, np.percentile(j, 2.5), np.percentile(j, 97.5))
    rng = np.random.Generator(np.random.PCG64(20261014)); boots, d = [], []
    for t in TASKS:
        A, B = items(a, t), items(b, t); assert set(A) == set(B); k = sorted(A)
        xa = np.array([A[i] for i in k], float); xb = np.array([B[i] for i in k], float)
        idx = rng.integers(0, len(k), (10000, len(k))); boots.append(100 * (xb[idx].mean(1) - xa[idx].mean(1))); d.append(100 * (xb.mean() - xa.mean()))
    m = np.mean(boots, 0); res["acc"] = (float(np.mean(d)), np.percentile(m, 2.5), np.percentile(m, 97.5))
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    for i in range(0, len(a), 2):
        r = paired(a[i], a[i + 1])
        print(f"{a[i+1]} - {a[i]}: dJ {r['J'][0]:+.4f} [{r['J'][1]:+.4f},{r['J'][2]:+.4f}]  wiki {r['wikitext2'][0]:+.4f}  c4 {r['c4'][0]:+.4f}  dAcc {r['acc'][0]:+.2f} [{r['acc'][1]:+.2f},{r['acc'][2]:+.2f}]")
