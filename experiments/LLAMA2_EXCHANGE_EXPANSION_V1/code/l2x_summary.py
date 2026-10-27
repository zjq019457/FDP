#!/usr/bin/env python3
"""Summary of all evaluated Llama-2 arms: paired vs Uniform at each operating point, random-seed distribution."""
import glob, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l2x_paired as PP
Q = PP.Q
U = {"e060": "uniform_e060", "e068": "gate_uniform068", "e090": "uniform_e090", "f060": "uniform_f060", "f068": "uniform_f068", "f090": "uniform_f090"}
res = {}
for op, u in U.items():
    if not (Q / u / "RESULT.json").exists(): continue
    arms = [a for a in (f"func_{op}_std", f"func_{op}_dd", f"fdp_{op}") if (Q / a / "RESULT.json").exists()]
    rnd = sorted(pathlib.Path(p).parent.name for p in glob.glob(str(Q / f"rand_{op}_*" / "RESULT.json")))
    res[op] = {"uniform": json.load(open(Q / u / "RESULT.json"))["J"]}
    for a in arms + rnd:
        r = PP.paired(u, a)
        res[op][a] = {"dJ": r["J"][0], "dJ_ci": [r["J"][1], r["J"][2]], "dAcc": r["acc"][0], "dAcc_ci": [r["acc"][1], r["acc"][2]]}
    for a in arms:
        x = res[op][a]; print(f"{op} {a:18s} dJ {x['dJ']:+.4f} [{x['dJ_ci'][0]:+.4f},{x['dJ_ci'][1]:+.4f}]  dAcc {x['dAcc']:+.2f} [{x['dAcc_ci'][0]:+.2f},{x['dAcc_ci'][1]:+.2f}]")
    if rnd:
        dj = np.array([res[op][a]["dJ"] for a in rnd]); da = np.array([res[op][a]["dAcc"] for a in rnd])
        sig_better = sum(res[op][a]["dJ_ci"][1] < 0 for a in rnd); sig_worse = sum(res[op][a]["dJ_ci"][0] > 0 for a in rnd)
        print(f"{op} random n={len(rnd)} dJ mean {dj.mean():+.4f} sd {dj.std(ddof=1) if len(rnd) > 1 else 0:.4f} range [{dj.min():+.4f},{dj.max():+.4f}] sig_better {sig_better} sig_worse {sig_worse}; dAcc mean {da.mean():+.2f} range [{da.min():+.2f},{da.max():+.2f}]")
        for a in arms:
            print(f"   {a}: better than {sum(res[op][a]['dJ'] < res[op][r]['dJ'] for r in rnd)}/{len(rnd)} random seeds on J")
json.dump(res, open("/diskF/jiaqizhang/LLAMA2_EXCHANGE_EXPANSION_V1/results/SUMMARY_PAIRED.json", "w"), indent=1)
