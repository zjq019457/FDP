"""PaLu LongBench (official protocol, ratio 0.5): re-score every prediction with the official PaLu scorer rule
(first output line for trec/triviaqa/samsum; max over references; official dataset2metric), gate against AGGREGATE.json,
then paired bootstrap PaLu-FDP initial minus Fisher (within-task resampling, equal-weight 8-task macro, 10,000 reps)."""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, "/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu")
from longbench_utils.scorer import dataset2metric
O = pathlib.Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1/PALU_OFFICIAL_LONGBENCH_V1")
D = pathlib.Path("/home/bai-project/jiaqizhang/EigenAttn/data/LongBench")
TASKS = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"]
agg = json.load(open(O / "AGGREGATE.json"))
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
def scores(method, t):
    data = jl(D / f"{t}.jsonl"); out = {}
    for r in jl(O / "runs/ratio_0p5" / method / t / "predictions.jsonl"):
        p = r["prediction"]
        if t in ("trec", "triviaqa", "samsum", "lsht"): p = p.lstrip("\n").split("\n")[0]
        ac = data[r["row_index"]].get("all_classes")
        out[r["sample_id"]] = max(dataset2metric[t](p, g, all_classes=ac) for g in r["reference"])
    return out
rng = np.random.Generator(np.random.PCG64(20261016)); boots = []; res = {}
for t in TASKS:
    f, n = scores("fisher", t), scores("fdp_native", t); k = sorted(f); assert sorted(n) == k
    a = np.array([f[i] for i in k]); b = np.array([n[i] for i in k])
    idx = rng.integers(0, len(k), (10000, len(k))); bs = 100 * (b[idx].mean(1) - a[idx].mean(1)); boots.append(bs)
    res[t] = {"fisher": round(100 * a.mean(), 2), "initial": round(100 * b.mean(), 2), "delta": 100 * (b.mean() - a.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    print(t, res[t]["fisher"], res[t]["initial"], f"{res[t]['delta']:+.2f} [{res[t]['ci95'][0]:+.2f},{res[t]['ci95'][1]:+.2f}]")
m = np.mean(boots, 0); mf = np.mean([v["fisher"] for v in res.values()]); mi = np.mean([v["initial"] for v in res.values()])
print("MACRO", round(mf, 4), round(mi, 4), f"{mi-mf:+.3f} [{np.percentile(m,2.5):+.3f},{np.percentile(m,97.5):+.3f}]")
print("AGGREGATE.md says Fisher 39.3475 / FDP Native 39.4425")
json.dump({"tasks": res, "macro": {"fisher": mf, "initial": mi, "delta": mi - mf, "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}},
          open("/diskF/jiaqizhang/NAACL_PAPER_SUPPLEMENT_V1/PALU_LONGBENCH_FISHER_VS_INITIAL_PAIRED.json", "w"), indent=1)
