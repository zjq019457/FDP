"""PaLu ratio 0.5 zero-shot (official protocol, six tasks): gate the logged rerun against the existing aggregates, then paired
bootstrap PaLu-FDP initial minus Fisher (items within task, equal-weight six-task macro, 10,000 replicates, PCG64 20261017)."""
import json, pathlib
import numpy as np
R = pathlib.Path("/diskF/jiaqizhang/NAACL_PAPER_SUPPLEMENT_V1/palu_zs_logged/ratio_0p5")
O = pathlib.Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1/official_zero_shot/ratio_0p5")
TASKS = ["arc_challenge_localjsonl", "arc_easy_localjsonl", "hellaswag_localjsonl", "openbookqa_localjsonl", "piqa_localjsonl", "winogrande_localjsonl"]
def items(m, t): return {r["doc_id"]: float(r["acc"]) for r in map(json.loads, open(R / m / "samples" / f"{t}.jsonl"))}
gate = {}
for m in ("fisher", "fdp_native"):
    old = json.load(open(O / m / "zeroshot.json"))["lm_eval_results"]
    gate[m] = {t: (round(100 * np.mean(list(items(m, t).values())), 4), round(100 * old[t]["acc,none"], 4)) for t in TASKS}
ok = all(a == b for g in gate.values() for a, b in g.values())
print("GATE", "PASS" if ok else "FAIL", gate)
rng = np.random.Generator(np.random.PCG64(20261017)); boots, per = [], {}
for t in TASKS:
    a, b = items("fisher", t), items("fdp_native", t); k = sorted(a); assert sorted(b) == k
    xa = np.array([a[i] for i in k]); xb = np.array([b[i] for i in k])
    idx = rng.integers(0, len(k), (10000, len(k))); bs = 100 * (xb[idx].mean(1) - xa[idx].mean(1)); boots.append(bs)
    per[t] = {"n": len(k), "fisher": 100 * xa.mean(), "initial": 100 * xb.mean(), "delta": 100 * (xb.mean() - xa.mean()),
              "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
m = np.mean(boots, 0)
macro = {"fisher": float(np.mean([v["fisher"] for v in per.values()])), "initial": float(np.mean([v["initial"] for v in per.values()])),
         "delta": float(np.mean([v["delta"] for v in per.values()])), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
for t, v in per.items(): print(t, round(v["fisher"], 2), round(v["initial"], 2), f"{v['delta']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]")
print("MACRO", macro)
json.dump({"gate_pass": ok, "gate": gate, "tasks": per, "macro": macro}, open("/diskF/jiaqizhang/NAACL_PAPER_SUPPLEMENT_V1/PALU_ZS_FISHER_VS_INITIAL_PAIRED.json", "w"), indent=1)
