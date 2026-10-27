#!/usr/bin/env python3
"""FLOOR45_UNIFORM_AT_NOFLOOR_FDP_MEDIUM_V1 - CPU-only, quality-blind plan (user request 2026-10-12).
Floored Uniform EigenAttention (official 45% rule) at the cache of the no-floor FDP-EB Medium schedule of the paper
(199,753,728 B). Single budget = floored lattice point nearest that B (ranks/bytes only)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/code")
import c45lib as L

TARGET = 199753728
OUT = Path("/diskF/jiaqizhang/FLOOR45_UNIFORM_AT_NOFLOOR_FDP_MEDIUM_V1")
lat = L.uniform45_lattice()
near = sorted(lat, key=lambda kv: abs(kv[0] - TARGET))[:5]
b, (emid, r, s) = near[0]
errs = sorted({x[3] for c in L.CURVES.values() for x in c})
lo = max([x for x in errs if x < emid], default=0.0); hi = min([x for x in errs if x > emid], default=10.0)
e = emid
for nd in range(2, 12):
    c = round((lo + hi) / 2, nd)
    if lo < c < hi and L.select([c] * 32, True)[0] == r:
        e = c; break
assert L.select([e] * 32, True)[0] == r
gap = 100 * (b - TARGET) / TARGET
plan = {"schema": "FLOOR45_UNIFORM_AT_NOFLOOR_FDP_MEDIUM_PLAN_V1", "quality_scores_read": False, "target_B": TARGET,
        "target": "paper Table 2(a) Medium FDP-EB (no floor)", "uni45": {"eb": e, "B": b, "gap_pct": gap, "match_class": L.match_class(gap),
        "ranks": r, "stops": L.stopdist(s), "rank_sha": L.json_sha(r)},
        "nearest_points": [{"B": x, "gap_pct": 100 * (x - TARGET) / TARGET} for x, _ in near]}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "PLAN.json").write_text(json.dumps(plan, indent=1))
print("eb", e, "B", b, "gap%", round(gap, 4), plan["uni45"]["match_class"], plan["uni45"]["stops"], [round(p["gap_pct"], 3) for p in plan["nearest_points"]])
