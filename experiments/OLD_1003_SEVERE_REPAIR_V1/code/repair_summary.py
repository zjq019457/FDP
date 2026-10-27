import json, math, glob
X = "/diskF/jiaqizhang/OLD_1003_RANKFLOOR_EXTENSION_V1/quality"
rows = {}
for a, d in (("FULL", X + "/FULL"), ("UNI_HIST", X + "/UNIFORM"), ("NOZ_HIST", X + "/NO_Z"), ("NOENT", X + "/NO_ENTROPY"),
             ("SPEC_FROZEN", "/diskF/jiaqizhang/OLD_1003_SPECTRAL_TAIL_V1/quality"), ("UNIFORM_FULLTRACE", "UNIFORM_FULLTRACE/quality"),
             ("NO_Z_FULLTRACE", "NO_Z_FULLTRACE/quality"), ("SPECTRAL_REANCHORED", "SPECTRAL_REANCHORED/quality")):
    w = json.load(open(d + "/wiki/ppl_aggregate.json")); c = json.load(open(d + "/c4/ppl_aggregate.json"))
    r = {"wiki": w["ppl"], "c4": c["ppl"], "J": (math.log(w["ppl"]) + math.log(c["ppl"])) / 2}
    try:
        t = json.load(open(d + "/zs_bf16/output/zero_shot_aggregate.json"))["tasks"]
        r["acc"] = [100 * v["accuracy_from_correct"] for v in t.values()]; r["macro"] = sum(r["acc"]) / 5
    except FileNotFoundError:
        pass
    rows[a] = r
    print(a, round(r["wiki"], 4), round(r["c4"], 4), round(r["J"], 4), [round(x, 2) for x in r.get("acc", [])], round(r.get("macro", 0), 2))
for f in sorted(glob.glob("BOOTSTRAP_*.json")):
    d = json.load(open(f)); print(f, {k: (round(v.get("delta", v.get("delta_logppl")), 4), [round(x, 4) for x in v["ci95"]], round(v["p_A_better"], 4)) for k, v in d.items()})
for f in sorted(glob.glob("ZS_BOOTSTRAP_*.json")):
    z = json.load(open(f)); print(f, {k: (round(v["delta"], 2), [round(x, 2) for x in v["ci95"]]) for k, v in z["tasks"].items()}, round(z["macro"]["delta"], 2), [round(x, 2) for x in z["macro"]["ci95"]], z["macro"]["p_A_better"])
for a in ("UNIFORM_FULLTRACE", "NO_Z_FULLTRACE", "SPECTRAL_REANCHORED"):
    v = json.load(open(a + "/VERIFY_" + a + ".json")); o = v["operator_parity_v3"]
    print(a, v["gates"], v["ALL_PASS"], v["ranks"]["B_live_from_tensors"], v["ranks"]["rank_sha"][:12], o["gate3B_worst_max_ulp"], o["gate3C_worst_relL2_K"], o["gate3C_worst_relL2_VO"], v["dual_process_canary"]["match"])
json.dump(rows, open("REPAIR_QUALITY_SUMMARY.json", "w"), indent=1)
