#!/usr/bin/env python3
"""CPU-only. Unified Llama-3 main table for Light/Medium/Aggressive: Uniform (Official control), Spectral-tail and
FDP-EB both on the common basis (one FDP-EB materialization per tier, shared by every comparison).
Paired window bootstrap (10,000, PCG64 20261012 wiki / 20261013 c4) and item bootstrap (10,000, PCG64 20261014,
within-task) - same conventions as paired_bootstrap.py / zs_paired_bootstrap.py. Identity of windows/items is asserted."""
import glob, json, math, os
import numpy as np
E = "/diskE/jiaqizhang"; F = "/diskF/jiaqizhang"; D = "/diskD/jiaqizhang"
X = E + "/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_SPECTRAL_CROSS_TIER_V1"
CT = X + "/CROSS_TIER_DIAGNOSTIC_20261006T000542Z/06_ZERO_SHOT_EXECUTION"
MB = F + "/TACL_LLAMA3_MEDIUM_NONFUNCTIONAL_CONTROLS_COMMON_BASIS_V1"
MAT = F + "/TACL_LLAMA3_SPECTRAL_CROSS_TIER_V1/materialized"
def ppl_dir(base, ds):
    c = [os.path.dirname(p) for p in glob.glob(base + "/**/ppl_aggregate.json", recursive=True)
         if json.load(open(p)).get("dataset") == ds and json.load(open(p)).get("used_windows") in (141, 256) and "canary" not in p]
    assert len(c) == 1, (base, ds, c); return c[0]
ARMS = {
 "light": {"Uniform": (E + "/FDP125_STRICT_FORMAL_CLOSURE_V1/official_control/light", D + "/FDP125_OFFICIAL_LIGHT_AGG_ZS_20261001_V1/light/accuracy/zero_shot_samples"),
           "Spectral-tail": (MAT + "/light_spectral_tail_common_basis", X + "/CROSS_TIER_DIAGNOSTIC_20261006T000542Z/06_ZERO_SHOT_EXECUTION/LIGHT_SPECTRAL_TAIL_20261006T204128Z/output/zero_shot_samples"),
           "FDP-EB": (MAT + "/light_fdp_eb_common_basis", X + "/CROSS_TIER_DIAGNOSTIC_20261006T000542Z/06_ZERO_SHOT_EXECUTION/LIGHT_FDP_EB_20261006T204128Z/output/zero_shot_samples")},
 "medium": {"Uniform": (E + "/FDP125_STRICT_FORMAL_CLOSURE_V1/official_control/medium", E + "/ICLR_FDP125_PAPER_EVIDENCE_V1/13_MODEL_MATRIX_4X3/cells/llama3_8b/medium/accuracy/official/zero_shot_samples"),
            "Spectral-tail": (MB + "/spectral_tail", X + "/06_ZERO_SHOT_EXECUTION/MEDIUM_SPECTRAL_TAIL_20261006T001103Z/output/zero_shot_samples"),
            "FDP-EB": (MB + "/FDP-EB", X + "/06_ZERO_SHOT_EXECUTION/MEDIUM_FDP_EB_20261006T073324Z/output/zero_shot_samples")},
 "aggressive": {"Uniform": (E + "/FDP125_STRICT_FORMAL_CLOSURE_V1/official_control/aggressive", D + "/FDP125_OFFICIAL_LIGHT_AGG_ZS_20261001_V1/aggressive/accuracy/zero_shot_samples"),
                "Spectral-tail": (MAT + "/aggressive_spectral_tail_common_basis", X + "/06_ZERO_SHOT_EXECUTION/AGGRESSIVE_SPECTRAL_TAIL_20261006T094703Z/output/zero_shot_samples"),
                "FDP-EB": (MAT + "/aggressive_fdp_eb_common_basis", X + "/06_ZERO_SHOT_EXECUTION/AGGRESSIVE_FDP_EB_20261006T094703Z/output/zero_shot_samples")}}
TASKS = ["arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande"]
def windows(d):
    return {r["window_id"]: (r["total_nll"], r["token_count"], r["input_ids_sha256"]) for r in map(json.loads, filter(str.strip, open(d + "/ppl_windows.jsonl")))}
def items(d, t):
    p = glob.glob(f"{d}/{t}_localjsonl*.jsonl"); assert len(p) == 1, (d, t, p)
    rows = [json.loads(x) for x in open(p[0]) if x.strip()]
    return {r["sample_id"]: (r["acc"] if "acc" in r else float(r["correct"]), r["document_hash"]) for r in rows}
def ppl_boot(a, b):
    out = {}; reps = {}
    for ds, seed in (("wikitext2", 20261012), ("c4", 20261013)):
        A, B = a[ds], b[ds]; ids = sorted(A); assert ids == sorted(B) and all(A[i][2] == B[i][2] for i in ids)
        na = np.array([A[i][0] for i in ids]); nb = np.array([B[i][0] for i in ids]); tk = np.array([A[i][1] for i in ids], float)
        rng = np.random.Generator(np.random.PCG64(seed)); idx = rng.integers(0, len(ids), size=(10000, len(ids)))
        bs = (na[idx].sum(1) - nb[idx].sum(1)) / tk[idx].sum(1); reps[ds] = bs
        out[ds] = {"delta": float((na.sum() - nb.sum()) / tk.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    j = (reps["wikitext2"] + reps["c4"]) / 2
    out["J"] = {"delta": (out["wikitext2"]["delta"] + out["c4"]["delta"]) / 2, "ci95": [float(np.percentile(j, 2.5)), float(np.percentile(j, 97.5))], "p_A_better": float((j < 0).mean())}
    return out
def zs_boot(a, b):
    rng = np.random.Generator(np.random.PCG64(20261014)); per = {}; boots = []
    for t in TASKS:
        A, B = a[t], b[t]; ids = sorted(A); assert ids == sorted(B) and all(A[i][1] == B[i][1] for i in ids), t
        xa = np.array([A[i][0] for i in ids], float); xb = np.array([B[i][0] for i in ids], float)
        idx = rng.integers(0, len(ids), size=(10000, len(ids))); bs = 100 * (xa[idx].mean(1) - xb[idx].mean(1)); boots.append(bs)
        per[t] = {"delta": float(100 * (xa.mean() - xb.mean())), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    m = np.mean(boots, axis=0)
    return {"tasks": per, "avg": {"delta": float(np.mean([v["delta"] for v in per.values()])), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "p_A_better": float((m > 0).mean())}}
res = {}
for tier, arms in ARMS.items():
    R = {}
    for arm, (pb, zd) in arms.items():
        w = {ds: windows(ppl_dir(pb, ds)) for ds in ("wikitext2", "c4")}
        ppl = {ds: math.exp(sum(v[0] for v in w[ds].values()) / sum(v[1] for v in w[ds].values())) for ds in w}
        z = {t: items(zd, t) for t in TASKS}
        acc = {t: 100 * float(np.mean([v[0] for v in z[t].values()])) for t in TASKS}
        R[arm] = {"w": w, "z": z, "ppl_dir": {ds: ppl_dir(pb, ds) for ds in w}, "zs_dir": zd, "wiki": ppl["wikitext2"], "c4": ppl["c4"],
                  "J": 0.5 * (math.log(ppl["wikitext2"]) + math.log(ppl["c4"])), "acc": acc, "avg": float(np.mean(list(acc.values())))}
    res[tier] = {arm: {k: v for k, v in R[arm].items() if k not in ("w", "z")} for arm in R}
    res[tier]["FDP_minus_Uniform"] = {"ppl": ppl_boot(R["FDP-EB"]["w"], R["Uniform"]["w"]), "zs": zs_boot(R["FDP-EB"]["z"], R["Uniform"]["z"])}
    res[tier]["Spectral_minus_FDP"] = {"ppl": ppl_boot(R["Spectral-tail"]["w"], R["FDP-EB"]["w"]), "zs": zs_boot(R["Spectral-tail"]["z"], R["FDP-EB"]["z"])}
    res[tier]["Spectral_minus_Uniform"] = {"ppl": ppl_boot(R["Spectral-tail"]["w"], R["Uniform"]["w"]), "zs": zs_boot(R["Spectral-tail"]["z"], R["Uniform"]["z"])}
os.makedirs(F + "/NAACL_PAPER_SUPPLEMENT_V1", exist_ok=True)
json.dump(res, open(F + "/NAACL_PAPER_SUPPLEMENT_V1/LLAMA3_UNIFIED_MAIN_TABLE.json", "w"), indent=1)
for tier, r in res.items():
    for arm in ("Uniform", "Spectral-tail", "FDP-EB"):
        a = r[arm]; print(tier, arm, round(a["wiki"], 4), round(a["c4"], 4), round(a["J"], 6), {t: round(v, 2) for t, v in a["acc"].items()}, round(a["avg"], 2))
    for k in ("FDP_minus_Uniform", "Spectral_minus_FDP", "Spectral_minus_Uniform"):
        p = r[k]["ppl"]; z = r[k]["zs"]["avg"]
        print("  ", k, "J", round(p["J"]["delta"], 4), [round(x, 4) for x in p["J"]["ci95"]], "wiki", round(p["wikitext2"]["delta"], 4), [round(x, 4) for x in p["wikitext2"]["ci95"]],
              "c4", round(p["c4"]["delta"], 4), [round(x, 4) for x in p["c4"]["ci95"]], "| avg", round(z["delta"], 2), [round(x, 2) for x in z["ci95"]])
