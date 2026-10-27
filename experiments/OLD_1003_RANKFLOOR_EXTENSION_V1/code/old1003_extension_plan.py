#!/usr/bin/env python3
"""OLD_1003_RANKFLOOR_EXTENSION_V1 - CPU-only, quality-blind resource plan.

Extends the 2026-10-03 rank-floor prospective Severe ablation (Full=d1+d2+d5 at 151,289,856 B,
No-S=d1+d5 at 151,257,088 B) with the two missing arms under the SAME protocol:
  No-Z = d2+d5, No-Entropy = d1+d2 (d1=Z, d2=causal S, d5=entropy H),
plus a Uniform arm at the Full resource. Old protocol reproduced exactly:
  profile per-layer components from the old schedule CSVs (profile sha 864cf983...), q = MinMax(sum),
  eps_l = base/(1+1.2 q_l) mean-normalised to base, layer0 rho=0.75 geometric then frozen to base,
  ranks by exact replay of the old frozen-builder curve report (envelope_target_attempt3), no 45% rule,
  hard filter: every QK and V rank >= 64 (the old constraint; see RANK_FLOOR_64_PROTOCOL_AUDIT_V1),
  pair rule vs fixed Full: |gap| <= 0.15% of min(B_full, B_arm); minimise max(B), then gap, then |eps-0.09|.
Gate: the replay must reproduce old Full and No-S ranks/B exactly before anything else is used.
"""
import csv, hashlib, json, math
from pathlib import Path

OLD = Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_3COMP_5COMP_AUDIT/ULTRA_AGGRESSIVE_DS_DIAGNOSTIC_V1/CAUSAL_SCORE_COMPONENT_RANKFLOOR_PROSPECTIVE_V1_20261003")
OUT = Path("/diskF/jiaqizhang/OLD_1003_RANKFLOOR_EXTENSION_V1")
lock = json.loads((OLD / "RESOURCE_SCAN_LOCK.json").read_text())
REP = Path(lock["rank_trace_report_path"])
assert hashlib.sha256(REP.read_bytes()).hexdigest() == lock["rank_trace_report_sha256"]
rep = json.loads(REP.read_text())
CURVES = {l["layer"]: [(t["threshold"], t["rank_kq"], t["rank_v"], t["err"]) for t in l["runs"][-1]["threshold_trace"]] for l in rep["layers"]}


def jsha(o): return hashlib.sha256(json.dumps(o, separators=(",", ":")).encode()).hexdigest()
def B_of(r): return sum((a + b) * 4096 for a, b in r)


def select(ebv):
    out = []
    for li in range(32):
        pts = CURVES[li]; best = (pts[0][1], pts[0][2]); e = ebv[li]
        for th, rk, rv, er in pts:
            if er <= e: best = (rk, rv)
            if not er < e: break
        out.append([int(best[0]), int(best[1])])
    return out


comp = {}
with open(OLD / "U_FULL_per_layer_schedule.csv") as f:
    for row in csv.DictReader(f):
        comp[int(row["layer"])] = {k: float(row[k]) for k in ("d1", "d2", "d5")}
D = {k: [comp[i][k] for i in range(32)] for k in ("d1", "d2", "d5")}


def minmax(x):
    lo, hi = min(x), max(x)
    return [0.5] * len(x) if hi - lo < 1e-12 else [(v - lo) / (hi - lo) for v in x]


def schedule(parts, base):
    q = minmax([sum(D[p][i] for p in parts) for i in range(32)])
    eps = [base / (1.0 + 1.2 * x) for x in q]
    s = base / (sum(eps) / 32); eps = [x * s for x in eps]
    eps[0] = math.exp(0.75 * math.log(base) + 0.25 * math.log(eps[0]))
    eps[0] = base                                   # layer-0 frozen to base (builder --freeze_layer0_eb)
    return eps, q


ARMS = {"FULL": ["d1", "d2", "d5"], "NO_S": ["d1", "d5"], "NO_Z": ["d2", "d5"], "NO_ENTROPY": ["d1", "d2"]}
# ---- gate: reproduce the old frozen pair exactly
gate = {}
for a in ("FULL", "NO_S"):
    old = lock["arms"][a]
    eps, _ = schedule(ARMS[a], old["epsilon_base"])
    r = select(eps)
    gate[a] = {"B": B_of(r), "old_B": old["B_live_candidate_bytes"], "rank_sha": jsha(r), "old_rank_sha": old["rank_sha256"],
               "eps_max_abs_diff_vs_old": max(abs(x - y) for x, y in zip(eps, old["epsilon_by_layer"])),
               "ranks_equal_old": r == [list(x) for x in old["rank_vector_kq_v"]]}
gate_pass = all(v["ranks_equal_old"] and v["B"] == v["old_B"] for v in gate.values())
plan = {"schema": "OLD_1003_RANKFLOOR_EXTENSION_PLAN_V1", "quality_scores_read": False, "replay_gate": gate, "replay_gate_PASS": gate_pass}
if gate_pass:
    BF = lock["arms"]["FULL"]["B_live_candidate_bytes"]
    errs = sorted({p[3] for c in CURVES.values() for p in c})
    for a in ("NO_Z", "NO_ENTROPY"):
        seen = {}
        # scalar plateaus: scan base over a dense grid around the old base; ranks change only at breakpoints
        for k in range(0, 40001):
            base = 0.07 + k * 0.000001 * 1.0
            eps, q = schedule(ARMS[a], base)
            r = select(eps); b = B_of(r)
            key = jsha(r)
            if key not in seen: seen[key] = {"base": base, "B": b, "ranks": r, "min_rank": min(min(x) for x in r)}
        elig = [v for v in seen.values() if v["min_rank"] >= 64 and abs(v["B"] - BF) / min(v["B"], BF) <= 0.0015]
        elig.sort(key=lambda v: (max(v["B"], BF), abs(v["B"] - BF) / min(v["B"], BF), abs(v["base"] - 0.09)))
        plan[a] = {"n_plateaus": len(seen), "n_eligible": len(elig),
                   "selected": ({k: elig[0][k] for k in ("base", "B", "min_rank")} | {"rank_sha": jsha(elig[0]["ranks"]), "ranks": elig[0]["ranks"],
                                "gap_pct_vs_full": 100 * (elig[0]["B"] - BF) / BF,
                                "eps_by_layer": schedule(ARMS[a], elig[0]["base"])[0]}) if elig else None}
    # Uniform at the Full resource: all distinct uniform points, with and without the old floor
    uni = {}
    for e in [errs[0] / 2] + [(x + y) / 2 for x, y in zip(errs, errs[1:])]:
        r = select([e] * 32); uni.setdefault(B_of(r), (e, r))
    pts = sorted(uni.items())
    near = sorted(pts, key=lambda kv: abs(kv[0] - BF))[:6]
    plan["UNIFORM_candidates_near_full"] = [{"B": b, "eb": e, "gap_pct": 100 * (b - BF) / BF, "min_rank": min(min(x) for x in r), "floor64_ok": min(min(x) for x in r) >= 64} for b, (e, r) in near]
    leg = [(b, e, r) for b, (e, r) in pts if min(min(x) for x in r) >= 64]
    lb = min(leg, key=lambda x: abs(x[0] - BF))
    above = min([x for x in leg if x[0] >= BF], key=lambda x: x[0])
    plan["UNIFORM_floor64_nearest"] = {"B": lb[0], "eb": lb[1], "gap_pct": 100 * (lb[0] - BF) / BF, "ranks": lb[2]}
    plan["UNIFORM_floor64_smallest_ge_full"] = {"B": above[0], "eb": above[1], "gap_pct": 100 * (above[0] - BF) / BF, "ranks": above[2]}
    plan["full_B"] = BF
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "RESOURCE_PLAN.json").write_text(json.dumps(plan, indent=1))
show = {k: v for k, v in plan.items() if k not in ("NO_Z", "NO_ENTROPY", "UNIFORM_floor64_nearest", "UNIFORM_floor64_smallest_ge_full")}
print(json.dumps(show, indent=1)[:3000])
for a in ("NO_Z", "NO_ENTROPY"):
    if a in plan:
        s = plan[a]["selected"]
        print(a, plan[a]["n_plateaus"], plan[a]["n_eligible"], None if not s else {k: s[k] for k in ("base", "B", "min_rank", "gap_pct_vs_full", "rank_sha")})
for k in ("UNIFORM_floor64_nearest", "UNIFORM_floor64_smallest_ge_full"):
    if k in plan: print(k, {x: plan[k][x] for x in ("B", "eb", "gap_pct")})
