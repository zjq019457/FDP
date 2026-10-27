#!/usr/bin/env python3
"""No-floor plan (user decision 2026-10-09: drop the rank>=64 rule, a transcription error per
RANK_FLOOR_64_PROTOCOL_AUDIT_V1). Everything else identical to the 10/3 protocol. Quality-blind.
Rule vs fixed old Full (151,289,856 B): |gap| <= 0.15% of min(B); minimise max(B, B_full), then gap, then |base-0.09|.
Uniform: single EB, same window rule, minimise gap."""
import json, sys
src = open("code/old1003_extension_plan.py").read().split("# ---- gate")[0]
exec(src)
BF = 151289856
plan = {"schema": "OLD_1003_NOFLOOR_EXTENSION_PLAN_V1", "quality_scores_read": False, "rank_floor": "none (frozen builder lattice, min 8)",
        "full_B": BF, "full_rank_sha": "0bafdf9cd1dbfa3d2a4a4080f5b8c4ab75a34260b3e1c163f7b65d50131e20a7", "arms": {}}
for a in ("NO_Z", "NO_ENTROPY"):
    seen = {}
    for k in range(0, 40001):
        base = 0.07 + k * 1e-6
        eps, q = schedule(ARMS[a], base); r = select(eps); key = jsha(r)
        if key not in seen: seen[key] = (base, B_of(r), r, eps)
    el = [v for v in seen.values() if abs(v[1] - BF) / min(v[1], BF) <= 0.0015]
    el.sort(key=lambda v: (max(v[1], BF), abs(v[1] - BF) / min(v[1], BF), abs(v[0] - 0.09)))
    base, b, r, eps = el[0]
    plan["arms"][a] = {"components": ARMS[a], "epsilon_base": base, "B": b, "gap_pct": 100 * (b - BF) / BF, "ranks": r, "rank_sha": jsha(r),
                       "min_rank": min(min(x) for x in r), "eps_by_layer": eps, "n_eligible": len(el)}
errs = sorted({p[3] for c in CURVES.values() for p in c})
uni = {}
for e in [errs[0] / 2] + [(x + y) / 2 for x, y in zip(errs, errs[1:])]:
    r = select([e] * 32); uni.setdefault(B_of(r), (e, r))
el = sorted([(b, e, r) for b, (e, r) in uni.items() if abs(b - BF) / min(b, BF) <= 0.0015], key=lambda x: abs(x[0] - BF))
b, e, r = el[0]
plan["arms"]["UNIFORM"] = {"eb": e, "B": b, "gap_pct": 100 * (b - BF) / BF, "ranks": r, "rank_sha": jsha(r), "min_rank": min(min(x) for x in r)}
plan["arms"]["FULL_REBUILD"] = {"ranks": select(schedule(ARMS["FULL"], 0.09355136167585255)[0]), "B": BF, "purpose": "accuracy (checkpoint deleted on 10/3); must reproduce old weights"}
json.dump(plan, open("PLAN_NOFLOOR.json", "w"), indent=1)
for k, v in plan["arms"].items(): print(k, v["B"], round(v.get("gap_pct", 0.0), 4), v.get("min_rank"), v.get("epsilon_base", v.get("eb")), v.get("rank_sha", "")[:12])
