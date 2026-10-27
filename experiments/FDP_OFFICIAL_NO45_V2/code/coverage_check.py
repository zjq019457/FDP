"""Can Light/Medium FDP-EB (new formula) be planned from the V1 recorded curves already? (CPU, quality-blind, no files written)"""
import json, sys
sys.path.insert(0, "/diskF/jiaqizhang/FDP_OFFICIAL_NO45_V2/code")
from plan_lib import *  # noqa

curves = union_v1_curves()
print("V1 union curve lengths:", [len(L["loop"]) for L in curves])
for t in ("light", "medium", "aggressive"):
    par = json.loads((V1 / "records" / f"parent_{t}" / "result.json").read_text())
    r = replay(curves, [OFFICIAL_EB[t]] * 32, "official")
    print(t, "official replay == parent:", r == par["ranks"], "B", par["B_live"])
    q = fdp_q(V1 / "tiers" / t / "step1.json")
    m = match(curves, q, par["B_live"], OFFICIAL_EB[t], complete=False, lo=OFFICIAL_EB[t] * 0.6, hi=OFFICIAL_EB[t] * 1.6, step=2e-5)
    p = m["pick"]
    print("  FDP new formula:", m["class"], "distinct", m["n_distinct"], "eligible", m["n_eligible"], "uncovered_bases", m["uncovered_bases"],
          None if p is None else (round(p["base"], 6), p["B"], round(p["gap_pct"], 4), p["rank_sha"][:12], min(min(x) for x in p["ranks"])))
