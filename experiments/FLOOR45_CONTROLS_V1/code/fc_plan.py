#!/usr/bin/env python3
"""FLOOR45_CONTROLS_V1 - CPU-only, quality-blind resource plan (user request 2026-10-12).

Two controls for the 45% rank-floor results of FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1 (read-only source):
  A. uni_nofloor_<tier>: Uniform EigenAttention WITHOUT the floor (Algorithm 1, official_min_thresh 0), single budget
     chosen as the no-floor lattice point nearest in B_live to the floored Uniform of the same tier.
  B. tail45_<tier>: Spectral-tail WITH the floor. Score = main-table definition (tails at this tier's floored Uniform
     ranks, QK/V 0.5/0.5, MinMax, frozen spectra), schedule = FDP-EB mapping (lambda 1.2, mean-normalised, layer 0 = base),
     selected under the floor; base scanned so B_live matches floored FDP-EB within 0.15%
     (rank: max(B, B_fdp), then |gap|, then |base - base_e|).
All selections use ranks/bytes only (exact CPU replay on the frozen curves, c45lib).
"""
import hashlib, json, sys
from pathlib import Path
import torch

SRC = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1")
OUT = Path("/diskF/jiaqizhang/FLOOR45_CONTROLS_V1")
SPEC = Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/05_controls_ablation/LLAMA3_MEDIUM_MATCHED_CONTROLS_V1_20261002T202000Z/RETRY_SPECTRA_CAPTURE_V2_20261002T142500Z/native_search/fdp125_native_spectra")
sys.path.insert(0, str(SRC / "code"))
import c45lib as L


def fsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def minmax(x):
    lo, hi = min(x), max(x)
    return [0.5] * len(x) if hi - lo < 1e-12 else [(v - lo) / (hi - lo) for v in x]
def schedule(q, base, lam=1.2):
    eps = [base / (1.0 + lam * x) for x in q]
    s = base / (sum(eps) / 32); eps = [x * s for x in eps]
    eps[0] = base
    return eps


errs = sorted({x[3] for c in L.CURVES.values() for x in c})
probe = [errs[0] / 2] + [(a + b) / 2 for a, b in zip(errs, errs[1:])] + [10.0]
lat_nf = {}
for e in probe:
    r, s = L.select([e] * 32, False)
    lat_nf.setdefault(L.B_of(r), (e, r, s))

spec = {}
for i in range(32):
    p = SPEC / f"layer_{i:02d}.pt"
    d = torch.load(p, map_location="cpu", weights_only=True)
    assert d["layer"] == i
    spec[i] = (d["eval_kq"].float().flatten(), d["eval_v"].float(), fsha(p))

plan = {"schema": "FLOOR45_CONTROLS_PLAN_V1", "quality_scores_read": False, "source": str(SRC), "spectra_dir": str(SPEC),
        "spectra_sha256": {f"layer_{i:02d}.pt": spec[i][2] for i in range(32)}, "tiers": {}}
for t in ["light", "medium", "aggressive"]:
    rp = json.loads((SRC / "tiers" / t / "RESOURCE_PLAN.json").read_text())
    base_e, BF, BU, RU = rp["base_e"], rp["fdp_predicted_B"], rp["uniform_predicted_B"], rp["uniform_predicted_ranks"]
    # realized floored arms must equal the plan (frozen records)
    real_u = json.loads((SRC / "records/basis" / f"uni_basis_{t}" / "result.json").read_text())["ranks"]
    real_f = json.loads((SRC / "records/basis" / f"fdp_basis_{t}" / "result.json").read_text())["ranks"]
    assert real_u == RU and real_f == rp["fdp_predicted_ranks"], t
    # A. no-floor Uniform nearest to floored Uniform
    bn, (emid, rn, sn) = min(lat_nf.items(), key=lambda kv: (abs(kv[0] - BU), kv[0]))
    lo = max([x for x in errs if x < emid], default=0.0); hi = min([x for x in errs if x > emid], default=10.0)
    en = emid
    for nd in range(2, 12):
        c = round((lo + hi) / 2, nd)
        if lo < c < hi and L.select([c] * 32, False)[0] == rn:
            en = c; break
    assert L.select([en] * 32, False)[0] == rn
    near = sorted(lat_nf.items(), key=lambda kv: abs(kv[0] - BU))[:4]
    # B. Spectral-tail under the floor
    raw, tq, tv = [], [], []
    for i in range(32):
        k, v, _ = spec[i]; rk, rv = RU[i]
        a = float(k[rk // 8:].sum() / k.sum()); b = float(v[:, rv // 8:].sum() / v.sum())
        tq.append(a); tv.append(b); raw.append(0.5 * a + 0.5 * b)
    q = minmax(raw)
    seen = {}
    for kk in range(0, 60001):
        base = base_e * 0.5 + kk * (base_e * 1.5) / 60000
        eps = schedule(q, base); r, s = L.select(eps, True); key = L.json_sha(r)
        if key not in seen: seen[key] = (base, L.B_of(r), r, s, eps)
    el = [v for v in seen.values() if abs(v[1] - BF) / min(v[1], BF) <= 0.0015]
    el.sort(key=lambda v: (max(v[1], BF), abs(v[1] - BF) / min(v[1], BF), abs(v[0] - base_e)))
    tp = {"base_e": base_e, "fdp45_B": BF, "uni45_B": BU, "uni45_eb": rp["uniform_eb"],
          "uni_nofloor": {"eb": en, "B": bn, "gap_pct_vs_uni45": 100 * (bn - BU) / BU, "gap_pct_vs_fdp45": 100 * (bn - BF) / BF,
                          "match_class_vs_uni45": L.match_class(100 * (bn - BU) / BU), "ranks": rn, "stops": L.stopdist(sn),
                          "rank_sha": L.json_sha(rn), "nearest_points": [{"B": b, "gap_pct": 100 * (b - BU) / BU} for b, _ in near],
                          "layers_differing_from_uni45": sum(a != b for a, b in zip(rn, RU))},
          "tail45": {"tail_qk": tq, "tail_v": tv, "raw": raw, "q": q, "q_sha": L.json_sha([round(x, 10) for x in q]),
                     "n_distinct": len(seen), "n_eligible": len(el),
                     "nearest_outside": sorted([{"base": v[0], "B": v[1], "gap_pct": 100 * (v[1] - BF) / BF} for v in seen.values()],
                                               key=lambda x: abs(x["gap_pct"]))[:6]}}
    if el:
        base, b, r, s, eps = el[0]
        tp["tail45"].update({"selected": True, "epsilon_base": base, "B": b, "gap_pct_vs_fdp45": 100 * (b - BF) / BF,
                             "match_class": L.match_class(100 * (b - BF) / BF), "eps_by_layer": eps, "ranks": r, "stops": L.stopdist(s),
                             "rank_sha": L.json_sha(r), "layers_differing_from_fdp45": sum(a != c for a, c in zip(r, real_f)),
                             "layers_differing_from_uni45": sum(a != c for a, c in zip(r, RU))})
    else:
        tp["tail45"]["selected"] = False
    plan["tiers"][t] = tp
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "PLAN.json").write_text(json.dumps(plan, indent=1))
for t, tp in plan["tiers"].items():
    u, s = tp["uni_nofloor"], tp["tail45"]
    print(t, "| uni45 B", tp["uni45_B"], "| nofloor eb", u["eb"], "B", u["B"], "gap% %.4f" % u["gap_pct_vs_uni45"], u["match_class_vs_uni45"],
          "diff", u["layers_differing_from_uni45"], u["stops"])
    print("   tail45 eligible", s["n_eligible"], (("base %.6f B %d gap%% %.4f %s diff_fdp %d diff_uni %d %s" % (s["epsilon_base"], s["B"], s["gap_pct_vs_fdp45"],
          s["match_class"], s["layers_differing_from_fdp45"], s["layers_differing_from_uni45"], s["stops"])) if s["selected"] else s["nearest_outside"]))
