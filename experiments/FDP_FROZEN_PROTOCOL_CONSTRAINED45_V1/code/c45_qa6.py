#!/usr/bin/env python3
"""CONSTRAINED45_V1 Aggressive QA-6: FDP-EB minus Uniform, same metric and bootstrap as NAACL_PAPER_SUPPLEMENT_V1/code/qa6_macro_ci.py."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/slim")
import longbench_eval as lbe
D = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/tiers/aggressive")
TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]
rng = np.random.Generator(np.random.PCG64(20261011)); boots = []; out = {"tasks": {}}
for t in TASKS:
    part = "lbnqa" if t == "narrativeqa" else "lb5"
    rows = {arm: [json.loads(x) for x in (D / arm / "longbench_qa6" / part / f"{t}.jsonl").read_text().splitlines() if x.strip()] for arm in ("fdp", "uniform")}
    m = lbe.dataset2metric[t]
    per = lambda rs: {r["_id"]: 100 * max([m(r["pred"], a, all_classes=r.get("all_classes")) for a in r["answers"]] or [0.0]) for r in rs}
    a, b = per(rows["fdp"]), per(rows["uniform"]); assert sorted(a) == sorted(b)
    ids = sorted(a); xa = np.array([a[i] for i in ids]); xb = np.array([b[i] for i in ids])
    idx = rng.integers(0, len(ids), size=(10000, len(ids))); boots.append(xa[idx].mean(1) - xb[idx].mean(1))
    out["tasks"][t] = {"n": len(ids), "fdp": float(xa.mean()), "uniform": float(xb.mean()), "delta": float(xa.mean() - xb.mean()),
                       "ci95": [float(np.percentile(boots[-1], 2.5)), float(np.percentile(boots[-1], 97.5))]}
mm = np.mean(boots, axis=0)
out["macro"] = {"fdp": float(np.mean([v["fdp"] for v in out["tasks"].values()])), "uniform": float(np.mean([v["uniform"] for v in out["tasks"].values()])),
                "delta": float(np.mean([v["delta"] for v in out["tasks"].values()])), "ci95": [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]}
Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/PAIRED_QA6_AGGRESSIVE.json").write_text(json.dumps(out, indent=1))
for t, v in out["tasks"].items(): print(t, round(v["fdp"], 2), round(v["uniform"], 2), "%+.2f [%+.2f,%+.2f]" % (v["delta"], *v["ci95"]))
v = out["macro"]; print("MACRO", round(v["fdp"], 2), round(v["uniform"], 2), "%+.2f [%+.2f,%+.2f]" % (v["delta"], *v["ci95"]))
