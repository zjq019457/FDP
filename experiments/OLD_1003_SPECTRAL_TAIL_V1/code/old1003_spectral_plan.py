#!/usr/bin/env python3
"""OLD_1003_SPECTRAL_TAIL_V1 - CPU-only, quality-blind resource plan (user request 2026-10-10).

Spectral-tail control under the 10/3 Severe protocol (no rank floor), matched to the 10/3 Full (151,289,856 B).
Only the per-layer score differs from the 10/3 arms:
  Spectral-tail (frozen cross-tier definition): at the 10/3 Uniform ranks (PLAN_NOFLOOR UNIFORM),
  QK tail = sum(eval_kq[rQK/8:])/sum(eval_kq), V tail = sum(eval_v[:, rV/8:])/sum(eval_v), raw = 0.5/0.5, MinMax.
  10/3 schedule: eps_l = base/(1+1.2 q_l), mean-normalised to base, layer 0 frozen to base.
  10/3 selection: base grid 0.07 + k*1e-6 (k<=40000); |gap| <= 0.15% of min(B, B_full);
  minimise max(B, B_full), then gap, then |base-0.09|.
Curves: the 10/3 trace (envelope_target_attempt3) stops at each layer's envelope EB, so it cannot rank schedules
that give a layer a larger EB. The Severe full trace (same frozen builder, calibration and threshold lattice, run
with eb=1.0) is used after two gates: (1) every 10/3 trace point equals the full-trace prefix; (2) the full trace
replays the locked 10/3 Full / No-S / No-Entropy ranks exactly. Audit finding (reported, not fixed): the locked 10/3
No-Z and Uniform ranks are capped by the truncated trace (Uniform: layer 3; No-Z: layers 3, 25, 29); the full trace
gives lower ranks there. The Spectral point is selected on the full trace (primary) and, as a diagnostic, on the
truncated 10/3 trace with the same rule.
"""
import csv, hashlib, json, math
from pathlib import Path
import torch

OUT = Path("/diskF/jiaqizhang/OLD_1003_SPECTRAL_TAIL_V1")
OLDX = Path("/diskF/jiaqizhang/OLD_1003_RANKFLOOR_EXTENSION_V1")
OLD = Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_3COMP_5COMP_AUDIT/ULTRA_AGGRESSIVE_DS_DIAGNOSTIC_V1/CAUSAL_SCORE_COMPONENT_RANKFLOOR_PROSPECTIVE_V1_20261003")
FULLTRACE = Path("/diskF/jiaqizhang/fdp_v2_fulltrace_scratch/severe/severe_fulltrace_v2_perlayer_eb_report.json")
SPEC = Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/05_controls_ablation/LLAMA3_MEDIUM_MATCHED_CONTROLS_V1_20261002T202000Z/RETRY_SPECTRA_CAPTURE_V2_20261002T142500Z/native_search/fdp125_native_spectra")
MASTER = Path("/diskF/jiaqizhang/FDP_OFFICIAL_PROTOCOL_V1/records/old_master/result.json")
LAMBDA = 1.2


def jsha(o): return hashlib.sha256(json.dumps(o, separators=(",", ":")).encode()).hexdigest()
def fsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def B_of(r): return sum((a + b) * 4096 for a, b in r)


lock = json.loads((OLD / "RESOURCE_SCAN_LOCK.json").read_text())
REP = Path(lock["rank_trace_report_path"]); assert fsha(REP) == lock["rank_trace_report_sha256"]
old_cur = {l["layer"]: [(t["threshold"], t["rank_kq"], t["rank_v"], t["err"]) for t in l["runs"][-1]["threshold_trace"]] for l in json.loads(REP.read_text())["layers"]}
ft = json.loads(FULLTRACE.read_text())
CUR = {l["layer"]: [(t["threshold"], t["rank_kq"], t["rank_v"], t["err"]) for t in l["runs"][0]["threshold_trace"]] for l in ft["layers"]}
old_cfg = json.loads(REP.read_text())["search_config"]
gate_prefix = {li: old_cur[li] == CUR[li][:len(old_cur[li])] for li in range(32)}
gate_cfg = old_cfg == ft["search_config"]


def select(ebv, cur=None):
    cur = cur or CUR
    out = []
    for li in range(32):
        pts = cur[li]; best = (pts[0][1], pts[0][2]); e = ebv[li]
        for th, rk, rv, er in pts:
            if er <= e: best = (rk, rv)
            if not er < e: break
        out.append([int(best[0]), int(best[1])])
    return out


def schedule_q(q, base):
    eps = [base / (1.0 + LAMBDA * x) for x in q]
    s = base / (sum(eps) / 32); eps = [x * s for x in eps]
    eps[0] = base                                   # layer-0 frozen to base (10/3 rule; the rho=0.75 step is overwritten)
    return eps


def minmax(x):
    lo, hi = min(x), max(x)
    return [0.5] * len(x) if hi - lo < 1e-12 else [(v - lo) / (hi - lo) for v in x]


comp = {}
with open(OLD / "U_FULL_per_layer_schedule.csv") as f:
    for row in csv.DictReader(f): comp[int(row["layer"])] = {k: float(row[k]) for k in ("d1", "d2", "d5")}
D = {k: [comp[i][k] for i in range(32)] for k in ("d1", "d2", "d5")}
ARMS = {"FULL": ["d1", "d2", "d5"], "NO_S": ["d1", "d5"], "NO_Z": ["d2", "d5"], "NO_ENTROPY": ["d1", "d2"]}
pn = json.loads((OLDX / "PLAN_NOFLOOR.json").read_text())
replay = {}
for a in ("FULL", "NO_S"):
    o = lock["arms"][a]; r = select(schedule_q(minmax([sum(D[p][i] for p in ARMS[a]) for i in range(32)]), o["epsilon_base"]))
    replay[a] = {"rank_sha": jsha(r), "locked": o["rank_sha256"], "PASS": jsha(r) == o["rank_sha256"]}
for a in ("NO_Z", "NO_ENTROPY"):
    o = pn["arms"][a]; r = select(schedule_q(minmax([sum(D[p][i] for p in ARMS[a]) for i in range(32)]), o["epsilon_base"]))
    replay[a] = {"rank_sha": jsha(r), "locked": o["rank_sha"], "PASS": jsha(r) == o["rank_sha"]}
u = pn["arms"]["UNIFORM"]; r = select([u["eb"]] * 32)
replay["UNIFORM"] = {"rank_sha": jsha(r), "locked": u["rank_sha"], "PASS": jsha(r) == u["rank_sha"]}
for a in ("NO_Z", "UNIFORM"):
    eps = [u["eb"]] * 32 if a == "UNIFORM" else schedule_q(minmax([sum(D[p][i] for p in ARMS[a]) for i in range(32)]), pn["arms"][a]["epsilon_base"])
    ro, rn, lk = select(eps, old_cur), select(eps), pn["arms"][a]["ranks"]
    replay[a].update({"old_trace_replay_equals_locked": ro == lk, "B_locked": B_of(lk), "B_full_trace": B_of(rn),
                      "capped_layers": [{"layer": i, "eps": eps[i], "locked": lk[i], "full_trace": rn[i], "old_trace_last_point": list(old_cur[i][-1])}
                                        for i in range(32) if rn[i] != lk[i]]})
gate_pass = all(gate_prefix.values()) and gate_cfg and all(replay[a]["PASS"] for a in ("FULL", "NO_S", "NO_ENTROPY"))
plan = {"schema": "OLD_1003_SPECTRAL_TAIL_PLAN_V1", "quality_scores_read": False, "rank_floor": "none (frozen builder lattice, min 8)",
        "old_trace": str(REP), "old_trace_sha256": lock["rank_trace_report_sha256"], "full_trace": str(FULLTRACE), "full_trace_sha256": fsha(FULLTRACE),
        "gate_old_trace_is_prefix_of_full_trace": all(gate_prefix.values()), "gate_search_config_equal": gate_cfg,
        "old_trace_points_per_layer": [len(old_cur[i]) for i in range(32)], "replay_gate": replay, "gate_PASS": gate_pass}
if gate_pass:
    BF = pn["full_B"]; uni = u["ranks"]
    raw, qk, vv, sh = [], [], [], {}
    for i in range(32):
        p = SPEC / f"layer_{i:02d}.pt"; sh[p.name] = fsha(p)
        d = torch.load(p, map_location="cpu", weights_only=True); k = d["eval_kq"].float().flatten(); v = d["eval_v"].float()
        assert d["layer"] == i and tuple(k.shape) == (128,) and tuple(v.shape) == (8, 128)
        rk, rv = uni[i]
        a = float(k[rk // 8:].sum() / k.sum()); b = float(v[:, rv // 8:].sum() / v.sum())
        qk.append(a); vv.append(b); raw.append(.5 * a + .5 * b)
    q = minmax(raw)
    seen = {}
    for kk in range(0, 40001):
        base = 0.07 + kk * 1e-6
        eps = schedule_q(q, base); r = select(eps); key = jsha(r)
        if key not in seen: seen[key] = (base, B_of(r), r, eps)
    el = [v for v in seen.values() if abs(v[1] - BF) / min(v[1], BF) <= 0.0015]
    el.sort(key=lambda v: (max(v[1], BF), abs(v[1] - BF) / min(v[1], BF), abs(v[0] - 0.09)))
    seen_o = {}
    for kk in range(0, 40001):
        base = 0.07 + kk * 1e-6
        r = select(schedule_q(q, base), old_cur); key = jsha(r)
        if key not in seen_o: seen_o[key] = (base, B_of(r), r)
    el_o = sorted([v for v in seen_o.values() if abs(v[1] - BF) / min(v[1], BF) <= 0.0015],
                  key=lambda v: (max(v[1], BF), abs(v[1] - BF) / min(v[1], BF), abs(v[0] - 0.09)))
    plan["diagnostic_truncated_trace_selection"] = None if not el_o else {"epsilon_base": el_o[0][0], "B": el_o[0][1], "ranks": el_o[0][2],
        "rank_sha256": jsha(el_o[0][2]), "n_eligible": len(el_o), "full_trace_ranks_at_same_base": select(schedule_q(q, el_o[0][0]))}
    cap = json.loads(MASTER.read_text())["capacity"]
    plan.update({"spectral_definition": "tails at 10/3 Uniform ranks (PLAN_NOFLOOR UNIFORM), QK/V 0.5/0.5, MinMax, lambda 1.2",
                 "uniform_rank_sha": jsha(uni), "spectra_dir": str(SPEC), "spectra_sha256": sh, "tail_qk": qk, "tail_v": vv, "raw": raw,
                 "q_spectral": q, "q_sha": jsha(q), "full_B": BF, "n_distinct_schedules": len(seen), "n_eligible": len(el),
                 "eligible": [{"base": v[0], "B": v[1], "gap_pct": 100 * (v[1] - BF) / BF, "rank_sha": jsha(v[2])} for v in el],
                 "nearest_outside_window": sorted([{"base": v[0], "B": v[1], "gap_pct": 100 * (v[1] - BF) / BF} for v in seen.values()],
                                                  key=lambda x: abs(x["gap_pct"]))[:6]})
    if el:
        base, b, r, eps = el[0]
        plan["selected"] = {"epsilon_base": base, "B": b, "gap_pct_vs_full": 100 * (b - BF) / BF, "ranks": r, "rank_sha256": jsha(r),
                            "min_rank": min(min(x) for x in r), "eps_by_layer": eps,
                            "master_capacity_ok": all(r[i][0] <= cap[i][0] and r[i][1] <= cap[i][1] for i in range(32))}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "PLAN.json").write_text(json.dumps(plan, indent=1))
print("prefix", plan["gate_old_trace_is_prefix_of_full_trace"], "cfg", gate_cfg, json.dumps(replay), "PASS", gate_pass)
if gate_pass:
    print("distinct", plan["n_distinct_schedules"], "eligible", plan["n_eligible"], plan["eligible"][:5])
    print("nearest", plan["nearest_outside_window"])
    if "selected" in plan:
        s = plan["selected"]; print("SELECTED base", s["epsilon_base"], "B", s["B"], "gap%", round(s["gap_pct_vs_full"], 4), "min", s["min_rank"], s["rank_sha256"][:12], "cap", s["master_capacity_ok"])
        print("q", [round(x, 3) for x in q]); print("ranks", s["ranks"])
    d = plan["diagnostic_truncated_trace_selection"]
    if d: print("TRUNC-TRACE selection base", d["epsilon_base"], "B", d["B"], d["rank_sha256"][:12], "same as primary:", "selected" in plan and d["rank_sha256"] == plan["selected"]["rank_sha256"], "ranks", d["ranks"])
