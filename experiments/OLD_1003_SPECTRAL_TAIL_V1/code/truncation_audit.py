#!/usr/bin/env python3
"""OLD_1003 truncated-curve audit (CPU only, quality-blind; reads no PPL/accuracy).
Reuses the frozen definitions of code/old1003_spectral_plan.py (curves, select, schedule, components) without
re-running its selection and without touching PLAN.json. Writes TRUNCATED_CURVE_AUDIT.json."""
import json, math
from pathlib import Path
import torch

O3S = Path("/diskF/jiaqizhang/OLD_1003_SPECTRAL_TAIL_V1")
src = (O3S / "code/old1003_spectral_plan.py").read_text().split("gate_pass = all")[0]
G = {}; exec(src, G)
CUR, old_cur, select, schedule_q, minmax, D, ARMS, pn, lock, B_of, jsha, SPEC = (G[k] for k in (
    "CUR", "old_cur", "select", "schedule_q", "minmax", "D", "ARMS", "pn", "lock", "B_of", "jsha", "SPEC"))
env = json.loads(Path("/diskF/jiaqizhang/llama3_full_resource_robustness_v1/ultra_aggressive_v1/envelope_target_attempt3/ultra_curve_envelope_epsilon.json").read_text())["error_budget_by_layer"]
BF = pn["full_B"]
frozen = json.loads((O3S / "PLAN.json").read_text())
A = {"schema": "OLD_1003_TRUNCATED_CURVE_AUDIT_V1", "quality_scores_read": False, "full_B": BF}

# coverage: the truncated trace ends at the first point whose err exceeded the envelope EB, so it determines the
# rank exactly iff eps < err of that last point; for eps >= last err the true rank is unknown (replay caps it there).
cov = []
for li in range(32):
    last = old_cur[li][-1]
    cov.append({"layer": li, "envelope_eb": env[li], "old_points": len(old_cur[li]), "full_points": len(CUR[li]),
                "last_point": {"threshold": last[0], "rank_kq": last[1], "rank_v": last[2], "err": last[3]},
                "last_point_is_failing_point": last[3] > env[li], "valid_for_eps_below": last[3]})
A["coverage"] = cov


def arm_eps(a, base=None):
    if a == "UNIFORM": return [pn["arms"]["UNIFORM"]["eb"]] * 32
    if a in ("FULL", "NO_S"): base = lock["arms"][a]["epsilon_base"]
    elif base is None: base = pn["arms"][a]["epsilon_base"]
    return schedule_q(minmax([sum(D[p][i] for p in ARMS[a]) for i in range(32)]), base)


locked = {"FULL": [list(x) for x in lock["arms"]["FULL"]["rank_vector_kq_v"]], "NO_S": [list(x) for x in lock["arms"]["NO_S"]["rank_vector_kq_v"]],
          "NO_Z": pn["arms"]["NO_Z"]["ranks"], "NO_ENTROPY": pn["arms"]["NO_ENTROPY"]["ranks"], "UNIFORM": pn["arms"]["UNIFORM"]["ranks"]}
A["arms"] = {}
for a in ("FULL", "NO_S", "NO_ENTROPY", "NO_Z", "UNIFORM"):
    eps = arm_eps(a); ro, rf = select(eps, old_cur), select(eps)
    layers = []
    for li in range(32):
        layers.append({"layer": li, "eps": eps[li], "trace_valid_below": cov[li]["valid_for_eps_below"], "in_coverage": eps[li] < cov[li]["valid_for_eps_below"],
                       "locked": locked[a][li], "old_trace": ro[li], "full_trace": rf[li], "differs": rf[li] != locked[a][li],
                       "dB_bytes": (sum(locked[a][li]) - sum(rf[li])) * 4096})
    A["arms"][a] = {"locked_rank_sha": jsha(locked[a]), "old_trace_replay_eq_locked": ro == locked[a], "full_trace_replay_eq_locked": rf == locked[a],
                    "full_trace_rank_sha": jsha(rf), "B_locked": B_of(locked[a]), "B_full_trace": B_of(rf), "gap_full_trace_vs_fullB_pct": 100 * (B_of(rf) - BF) / BF,
                    "n_layers_out_of_coverage": sum(not l["in_coverage"] for l in layers),
                    "min_coverage_margin": min(l["trace_valid_below"] - l["eps"] for l in layers), "layers": layers}

# were the original 10/3 scans (Full, No-S; and the 10/9 No-Entropy / No-Z scans) affected anywhere on their grid?
grid = [0.07 + k * 1e-6 for k in range(0, 40001, 10)]
scan = {}
for a in ("FULL", "NO_S", "NO_Z", "NO_ENTROPY"):
    q = minmax([sum(D[p][i] for p in ARMS[a]) for i in range(32)])
    mism = [b for b in grid if select(schedule_q(q, b), old_cur) != select(schedule_q(q, b))]
    scan[a] = {"grid_step": 1e-5, "n_grid": len(grid), "n_bases_where_truncated_trace_differs": len(mism),
               "first_differing_base": mism[0] if mism else None}
A["scan_consistency"] = scan


def rescan(a):
    q = minmax([sum(D[p][i] for p in ARMS[a]) for i in range(32)]); seen = {}
    for k in range(0, 40001):
        base = 0.07 + k * 1e-6; r = select(schedule_q(q, base)); h = jsha(r)
        if h not in seen: seen[h] = (base, B_of(r), r)
    el = sorted([v for v in seen.values() if abs(v[1] - BF) / min(v[1], BF) <= 0.0015], key=lambda v: (max(v[1], BF), abs(v[1] - BF) / min(v[1], BF), abs(v[0] - 0.09)))
    near = sorted(seen.values(), key=lambda v: abs(v[1] - BF))[:6]
    return {"n_distinct": len(seen), "n_eligible": len(el),
            "eligible": [{"base": v[0], "B": v[1], "gap_pct": 100 * (v[1] - BF) / BF, "rank_sha": jsha(v[2]), "min_rank": min(min(x) for x in v[2])} for v in el],
            "selected": None if not el else {"epsilon_base": el[0][0], "B": el[0][1], "gap_pct": 100 * (el[0][1] - BF) / BF, "ranks": el[0][2],
                                             "rank_sha256": jsha(el[0][2]), "min_rank": min(min(x) for x in el[0][2]), "eps_by_layer": schedule_q(q, el[0][0])},
            "nearest": [{"base": v[0], "B": v[1], "gap_pct": 100 * (v[1] - BF) / BF} for v in near]}


A["NO_ENTROPY_full_trace_rescan"] = rescan("NO_ENTROPY")
A["NO_ENTROPY_full_trace_rescan"]["same_as_locked"] = (A["NO_ENTROPY_full_trace_rescan"]["selected"] or {}).get("rank_sha256") == pn["arms"]["NO_ENTROPY"]["rank_sha"]
A["NO_Z_full_trace_rescan"] = rescan("NO_Z")

# Uniform: original EB on the full trace, and the full-trace single-EB candidates around the Full resource (10/3 rule)
ue = pn["arms"]["UNIFORM"]["eb"]; ru = select([ue] * 32)
errs = sorted({p[3] for c in CUR.values() for p in c})
uni = {}
for e in [errs[0] / 2] + [(x + y) / 2 for x, y in zip(errs, errs[1:])]:
    r = select([e] * 32); uni.setdefault(B_of(r), (e, r))
el = sorted([(b, e, r) for b, (e, r) in uni.items() if abs(b - BF) / min(b, BF) <= 0.0015], key=lambda x: abs(x[0] - BF))
near = sorted(uni.items(), key=lambda kv: abs(kv[0] - BF))[:6]
A["UNIFORM_full_trace"] = {"eb": ue, "ranks_at_original_eb": ru, "rank_sha_at_original_eb": jsha(ru), "B_at_original_eb": B_of(ru),
                           "gap_vs_fullB_pct": 100 * (B_of(ru) - BF) / BF, "B_locked": B_of(locked["UNIFORM"]),
                           "n_eligible_matched": len(el),
                           "matched_selected": None if not el else {"eb": el[0][1], "B": el[0][0], "gap_pct": 100 * (el[0][0] - BF) / BF, "ranks": el[0][2],
                                                                    "rank_sha256": jsha(el[0][2]), "min_rank": min(min(x) for x in el[0][2]),
                                                                    "layers_differ_from_locked": [i for i in range(32) if el[0][2][i] != locked["UNIFORM"][i]]},
                           "nearest": [{"eb": e, "B": b, "gap_pct": 100 * (b - BF) / BF, "rank_sha": jsha(r)} for b, (e, r) in near]}

# Spectral anchor sensitivity (diagnostic only; the frozen PLAN is not changed)
spec = []
for i in range(32):
    d = torch.load(SPEC / f"layer_{i:02d}.pt", map_location="cpu", weights_only=True)
    spec.append((d["eval_kq"].float().flatten(), d["eval_v"].float()))


def spectral(anchor):
    raw = []; tq = []; tv = []
    for i, (rk, rv) in enumerate(anchor):
        k, v = spec[i]; a = float(k[rk // 8:].sum() / k.sum()); b = float(v[:, rv // 8:].sum() / v.sum())
        tq.append(a); tv.append(b); raw.append(.5 * a + .5 * b)
    q = minmax(raw); seen = {}
    for kk in range(0, 40001):
        base = 0.07 + kk * 1e-6; r = select(schedule_q(q, base)); h = jsha(r)
        if h not in seen: seen[h] = (base, B_of(r), r)
    el = sorted([v for v in seen.values() if abs(v[1] - BF) / min(v[1], BF) <= 0.0015], key=lambda v: (max(v[1], BF), abs(v[1] - BF) / min(v[1], BF), abs(v[0] - 0.09)))
    out = {"anchor_rank_sha": jsha(anchor), "tail_qk": tq, "tail_v": tv, "q": q, "q_sha": jsha(q), "n_distinct": len(seen), "n_eligible": len(el)}
    if el:
        out["selected"] = {"epsilon_base": el[0][0], "B": el[0][1], "gap_pct": 100 * (el[0][1] - BF) / BF, "ranks": el[0][2], "rank_sha256": jsha(el[0][2]),
                           "min_rank": min(min(x) for x in el[0][2])}
    return out


anchors = {"historical_uniform": locked["UNIFORM"], "uniform_full_trace_same_eb": ru}
if A["UNIFORM_full_trace"]["matched_selected"]: anchors["uniform_full_trace_matched"] = A["UNIFORM_full_trace"]["matched_selected"]["ranks"]
anchors["uniform_full_trace_nearest_plus_0p217"] = select([0.09305876924047846] * 32)          # = Severe V2 Uniform ranks (823e850e)
S = {k: spectral(v) for k, v in anchors.items()}
fz = frozen["selected"]
for k, v in S.items():
    s = v.get("selected", {})
    v["vs_frozen"] = {"q_sha_equal": v["q_sha"] == frozen["q_sha"], "max_abs_q_diff": max(abs(x - y) for x, y in zip(v["q"], frozen["q_spectral"])),
                      "rank_sha_equal": s.get("rank_sha256") == fz["rank_sha256"], "B_equal": s.get("B") == fz["B"],
                      "layers_rank_differ": [i for i in range(32) if s and s["ranks"][i] != fz["ranks"][i]]}
A["spectral_anchor_sensitivity"] = S
(O3S / "TRUNCATED_CURVE_AUDIT.json").write_text(json.dumps(A, indent=1))

for a, v in A["arms"].items():
    print(a, "fullEq", v["full_trace_replay_eq_locked"], "Blocked", v["B_locked"], "Bfull", v["B_full_trace"], "outcov", v["n_layers_out_of_coverage"], "margin", round(v["min_coverage_margin"], 5),
          [(l["layer"], round(l["eps"], 5), round(l["trace_valid_below"], 5), l["locked"], l["full_trace"]) for l in v["layers"] if not l["in_coverage"]])
print("scan", scan)
for k in ("NO_ENTROPY_full_trace_rescan", "NO_Z_full_trace_rescan"):
    v = A[k]; print(k, v["n_distinct"], v["n_eligible"], v["eligible"][:4], v.get("same_as_locked"), v["nearest"][:4])
    if v["selected"]: print("   ranks", v["selected"]["ranks"])
u = A["UNIFORM_full_trace"]; print("UNI", u["B_at_original_eb"], round(u["gap_vs_fullB_pct"], 4), u["n_eligible_matched"], u["matched_selected"] and {k: u["matched_selected"][k] for k in ("eb", "B", "gap_pct", "rank_sha256", "min_rank", "layers_differ_from_locked")}, u["nearest"])
for k, v in S.items(): print("SPEC", k, v["q_sha"][:12], v["n_eligible"], v.get("selected") and {x: v["selected"][x] for x in ("epsilon_base", "B", "gap_pct", "rank_sha256")}, v["vs_frozen"])
