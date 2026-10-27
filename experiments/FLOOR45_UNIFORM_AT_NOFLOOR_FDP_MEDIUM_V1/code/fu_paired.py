#!/usr/bin/env python3
"""Paired: paper no-floor FDP-EB Medium (199,753,728 B) minus floored Uniform at eb 0.06 (199,589,888 B).
Same bootstrap as the paper (PCG64 20261012/13 windows, 20261014 items). Negative dJ / positive dAvg favor FDP-EB."""
import glob, json, math, os
import numpy as np
E = "/diskE/jiaqizhang"; F = "/diskF/jiaqizhang"
X = E + "/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_SPECTRAL_CROSS_TIER_V1"
MB = F + "/TACL_LLAMA3_MEDIUM_NONFUNCTIONAL_CONTROLS_COMMON_BASIS_V1"
FDP = (MB + "/FDP-EB", X + "/06_ZERO_SHOT_EXECUTION/MEDIUM_FDP_EB_20261006T073324Z/output/zero_shot_samples")
NEWR = F + "/FLOOR45_UNIFORM_AT_NOFLOOR_FDP_MEDIUM_V1/tiers/medium/uniform45_at_199p75M"
U45 = (NEWR, NEWR + "/zs_bf16/output/zero_shot_samples")
TASKS = ["arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande"]


def ppl_dir(base, ds):
    c = [os.path.dirname(p) for p in glob.glob(base + "/**/ppl_aggregate.json", recursive=True)
         if json.load(open(p)).get("dataset") == ds and json.load(open(p)).get("used_windows") in (141, 256) and "canary" not in p]
    if not c:
        c = [base + f"/ppl_{ds}_bf16"]
    assert len(c) == 1, (base, ds, c); return c[0]
def windows(d): return {r["window_id"]: (r["total_nll"], r["token_count"], r["input_ids_sha256"]) for r in map(json.loads, filter(str.strip, open(d + "/ppl_windows.jsonl")))}
def items(d, t):
    p = glob.glob(f"{d}/{t}_localjsonl*.jsonl"); assert len(p) == 1, (d, t, p)
    return {r["sample_id"]: (r["acc"] if "acc" in r else float(r["correct"]), r["document_hash"]) for r in map(json.loads, filter(str.strip, open(p[0])))}


def load(arm):
    pb, zd = arm
    w = {ds: windows(ppl_dir(pb, ds)) for ds in ("wikitext2", "c4")}
    return w, {t: items(zd, t) for t in TASKS}


wa, za = load(FDP); wb, zb = load(U45)
out = {}; reps = {}
for ds, seed in (("wikitext2", 20261012), ("c4", 20261013)):
    A, B = wa[ds], wb[ds]; ids = sorted(A); assert ids == sorted(B) and all(A[i][2] == B[i][2] for i in ids)
    na = np.array([A[i][0] for i in ids]); nb = np.array([B[i][0] for i in ids]); tk = np.array([A[i][1] for i in ids], float)
    idx = np.random.Generator(np.random.PCG64(seed)).integers(0, len(ids), size=(10000, len(ids)))
    bs = (na[idx].sum(1) - nb[idx].sum(1)) / tk[idx].sum(1); reps[ds] = bs
    out[ds] = {"ppl_fdp": math.exp(na.sum() / tk.sum()), "ppl_uni45": math.exp(nb.sum() / tk.sum()), "delta": float((na.sum() - nb.sum()) / tk.sum()),
               "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
j = (reps["wikitext2"] + reps["c4"]) / 2
out["J"] = {"fdp": 0.5 * (math.log(out["wikitext2"]["ppl_fdp"]) + math.log(out["c4"]["ppl_fdp"])),
            "uni45": 0.5 * (math.log(out["wikitext2"]["ppl_uni45"]) + math.log(out["c4"]["ppl_uni45"])),
            "delta": (out["wikitext2"]["delta"] + out["c4"]["delta"]) / 2, "ci95": [float(np.percentile(j, 2.5)), float(np.percentile(j, 97.5))]}
rng = np.random.Generator(np.random.PCG64(20261014)); boots = []; per = {}
for t in TASKS:
    A, B = za[t], zb[t]; ids = sorted(A); assert ids == sorted(B) and all(A[i][1] == B[i][1] for i in ids), t
    xa = np.array([A[i][0] for i in ids], float); xb = np.array([B[i][0] for i in ids], float)
    idx = rng.integers(0, len(ids), size=(10000, len(ids))); bs = 100 * (xa[idx].mean(1) - xb[idx].mean(1)); boots.append(bs)
    per[t] = {"fdp": 100 * xa.mean(), "uni45": 100 * xb.mean(), "delta": float(100 * (xa.mean() - xb.mean())),
              "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
m = np.mean(boots, axis=0)
out["acc_tasks"] = per
out["avg"] = {"fdp": float(np.mean([v["fdp"] for v in per.values()])), "uni45": float(np.mean([v["uni45"] for v in per.values()])),
              "delta": float(np.mean([v["delta"] for v in per.values()])), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
json.dump(out, open(F + "/FLOOR45_UNIFORM_AT_NOFLOOR_FDP_MEDIUM_V1/PAIRED_FDP_NOFLOOR_MINUS_UNI45.json", "w"), indent=1)
for ds in ("wikitext2", "c4"):
    o = out[ds]; print(ds, "FDP %.3f  U45 %.3f  dlogPPL %+.4f [%+.4f,%+.4f]" % (o["ppl_fdp"], o["ppl_uni45"], o["delta"], *o["ci95"]))
print("J FDP %.4f U45 %.4f dJ %+.4f [%+.4f,%+.4f]" % (out["J"]["fdp"], out["J"]["uni45"], out["J"]["delta"], *out["J"]["ci95"]))
for t, v in per.items(): print(t, "%.2f vs %.2f d %+.2f [%+.2f,%+.2f]" % (v["fdp"], v["uni45"], v["delta"], *v["ci95"]))
print("avg %.2f vs %.2f d %+.2f [%+.2f,%+.2f]" % (out["avg"]["fdp"], out["avg"]["uni45"], out["avg"]["delta"], *out["avg"]["ci95"]))
