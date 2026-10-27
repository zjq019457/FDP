#!/usr/bin/env python3
"""SEVERE_V2_SPECTRAL_TAIL_V1 - CPU-only, quality-blind resource plan.

Same pipeline as the Severe V2 Full / ablations; only the per-layer score changes:
  Spectral-tail score (frozen cross-tier definition): at the Severe V2 Uniform ranks,
  QK tail = sum(eval_kq[rQK/8:])/sum(eval_kq), V tail = sum(eval_v[:, rV/8:])/sum(eval_v), raw = 0.5/0.5,
  MinMax over 32 layers; g = 1/(1+1.2 q).
  V2 schedule: eps_l = base * s * g_l / mean(g) for l>=1, eps_0 = base (base = 0.09305876924047846).
  Ranks: exact replay of the V2 Severe full-trace builder report (frozen builder semantics).
  Selection (V2 rule): |B - B_full| <= 0.15% of B_full; minimise |log s|, then |gap|, then smaller s.
Gate: the replay must reproduce V2 Full and No-Z rank SHA exactly before the Spectral point is used.
"""
import hashlib, json, math
from pathlib import Path
import torch

OUT = Path("/diskF/jiaqizhang/SEVERE_V2_SPECTRAL_TAIL_V1")
S2 = Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_SPECTRAL_CROSS_TIER_V1/FDP_CANONICAL_RANK_REPRODUCTION_AUDIT_V1/SEVERE_V2")
TRACE = Path("/diskF/jiaqizhang/fdp_v2_fulltrace_scratch/severe/severe_fulltrace_v2_perlayer_eb_report.json")
SPEC = Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/05_controls_ablation/LLAMA3_MEDIUM_MATCHED_CONTROLS_V1_20261002T202000Z/RETRY_SPECTRA_CAPTURE_V2_20261002T142500Z/native_search/fdp125_native_spectra")
BASE = 0.09305876924047846
LAMBDA = 1.2


def jsha(o): return hashlib.sha256(json.dumps(o, separators=(",", ":")).encode()).hexdigest()
def B_of(r): return sum((a + b) * 4096 for a, b in r)
def fsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rep = json.loads(TRACE.read_text())
CUR = [[(t["rank_kq"], t["rank_v"], t["err"]) for t in l["runs"][0]["threshold_trace"]] for l in rep["layers"]]


def select(eps):
    out = []
    for li in range(32):
        pts = CUR[li]; best = (pts[0][0], pts[0][1])
        for rk, rv, er in pts:
            if er <= eps[li]: best = (rk, rv)
            if not er < eps[li]: break
        out.append([int(best[0]), int(best[1])])
    return out


def schedule(g, s):
    m = sum(g) / len(g)
    eps = [BASE * s * x / m for x in g]; eps[0] = BASE
    return eps


prof = json.loads((S2 / "SEVERE_ABLATION_PROFILES_Q_G.json").read_text())
lock = json.loads((S2 / "SEVERE_MATCHED_ABLATION_LOCK_V1.json").read_text())
gate = {}
for arm in ("Full", "No_Z", "No_Causal"):
    r = select(schedule(prof[arm]["g"], lock[arm]["selected_s"]))
    gate[arm] = {"replay_rank_sha": jsha(r), "locked_rank_sha": lock[arm]["rank_sha256"], "B": B_of(r), "locked_B": lock[arm]["B"], "PASS": jsha(r) == lock[arm]["rank_sha256"]}
plan = {"schema": "SEVERE_V2_SPECTRAL_TAIL_PLAN_V1", "quality_scores_read": False, "trace": str(TRACE), "trace_sha256": fsha(TRACE),
        "replay_gate": gate, "replay_gate_PASS": all(v["PASS"] for v in gate.values())}
if plan["replay_gate_PASS"]:
    uni = json.loads((S2 / "SEVERE_UNIFORM_LOCK.json").read_text())["ranks"]
    raw, qk, vv, sh = [], [], [], {}
    for i in range(32):
        p = SPEC / f"layer_{i:02d}.pt"; sh[p.name] = fsha(p)
        d = torch.load(p, map_location="cpu", weights_only=True); k = d["eval_kq"].float().flatten(); v = d["eval_v"].float()
        assert d["layer"] == i and tuple(k.shape) == (128,) and tuple(v.shape) == (8, 128)
        rk, rv = uni[i]
        a = float(k[rk // 8:].sum() / k.sum()); b = float(v[:, rv // 8:].sum() / v.sum())
        qk.append(a); vv.append(b); raw.append(.5 * a + .5 * b)
    lo, hi = min(raw), max(raw); q = [(x - lo) / (hi - lo) for x in raw]
    g = [1 / (1 + LAMBDA * x) for x in q]
    BF = lock["Full"]["B"]
    m = sum(g) / len(g)
    pts = {1.0}
    for li in range(1, 32):
        for _, _, er in CUR[li]:
            s = er * m / (BASE * g[li])
            if 0.3 < s < 3.0:
                pts.update((s, math.nextafter(s, math.inf), math.nextafter(s, -math.inf)))
    by = {}
    for s in sorted(pts):
        r = select(schedule(g, s)); h = jsha(r)
        if h not in by or abs(math.log(s)) < abs(math.log(by[h][0])): by[h] = (s, r)
    cands = [(s, r, B_of(r)) for s, r in by.values() if abs(B_of(r) - BF) / BF <= 0.0015]
    cands.sort(key=lambda x: (abs(math.log(x[0])), abs(x[2] - BF), x[0]))
    plan.update({"spectral_definition": "tails at Severe V2 Uniform ranks, QK/V 0.5/0.5, MinMax, lambda 1.2 (frozen cross-tier definition)",
                 "uniform_rank_sha": jsha(uni), "spectra_sha256": sh, "tail_qk": qk, "tail_v": vv, "raw": raw, "q_spectral": q,
                 "q_sha": jsha(q), "full_B": BF, "n_distinct_schedules": len(by), "n_eligible": len(cands)})
    if cands:
        s, r, b = cands[0]
        plan["selected"] = {"selected_s": s, "B": b, "gap_pct_vs_full": 100 * (b - BF) / BF, "ranks": r, "rank_sha256": jsha(r),
                            "min_rank": min(min(x) for x in r), "eps_by_layer": schedule(g, s),
                            "layers_differ_from_full": None}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "PLAN.json").write_text(json.dumps(plan, indent=1))
print(json.dumps(plan["replay_gate"], indent=1), plan["replay_gate_PASS"])
if "selected" in plan:
    s = plan["selected"]; print("SELECTED s", s["selected_s"], "B", s["B"], "gap%", round(s["gap_pct_vs_full"], 4), "min_rank", s["min_rank"], s["rank_sha256"][:12], "eligible", plan["n_eligible"])
