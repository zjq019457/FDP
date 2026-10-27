"""Medium: w/o Causal (resource-matched) minus Full (resource-matched), paired over identical windows (PCG64 20261012/13, 10,000)."""
import json, math, pathlib
import numpy as np
F = pathlib.Path("/diskF/jiaqizhang/llama3_full_resource_robustness_v1/final/medium/quality")
W = pathlib.Path("/diskF/jiaqizhang/ABL_WOC_MEDIUM_RM_V1/tiers/medium/woc_rm")
def win(p): return {r["window_id"]: (r["total_nll"], r["token_count"], r["input_ids_sha256"]) for r in map(json.loads, filter(str.strip, open(p)))}
res, reps = {}, {}
for ds, fdir, wdir, seed in (("wikitext2", "wiki", "ppl_wikitext2_bf16", 20261012), ("c4", "c4", "ppl_c4_bf16", 20261013)):
    a, b = win(F / fdir / "ppl_windows.jsonl"), win(W / wdir / "ppl_windows.jsonl"); ids = sorted(a)
    assert ids == sorted(b) and all(a[i][2] == b[i][2] for i in ids), ds
    na = np.array([a[i][0] for i in ids]); nb = np.array([b[i][0] for i in ids]); tk = np.array([a[i][1] for i in ids], float)
    idx = np.random.Generator(np.random.PCG64(seed)).integers(0, len(ids), (10000, len(ids)))
    bs = (nb[idx].sum(1) - na[idx].sum(1)) / tk[idx].sum(1); reps[ds] = bs
    res[ds] = {"full_ppl": math.exp(na.sum() / tk.sum()), "woc_ppl": math.exp(nb.sum() / tk.sum()), "delta": float((nb.sum() - na.sum()) / tk.sum()),
               "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": len(ids)}
j = (reps["wikitext2"] + reps["c4"]) / 2
res["J"] = {"full": 0.5 * (math.log(res["wikitext2"]["full_ppl"]) + math.log(res["c4"]["full_ppl"])),
            "woc": 0.5 * (math.log(res["wikitext2"]["woc_ppl"]) + math.log(res["c4"]["woc_ppl"])),
            "delta": (res["wikitext2"]["delta"] + res["c4"]["delta"]) / 2, "ci95": [float(np.percentile(j, 2.5)), float(np.percentile(j, 97.5))]}
print(json.dumps(res, indent=1))
json.dump(res, open("/diskF/jiaqizhang/ABL_WOC_MEDIUM_RM_V1/PAIRED_WOC_MINUS_FULL_RM.json", "w"), indent=1)
