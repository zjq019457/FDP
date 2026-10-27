#!/usr/bin/env python3
"""FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1 - CPU-only resource feasibility audit (no quality scores used).

The frozen offline builder propagates DENSE layer outputs to the next layer, so the candidate
sequence (threshold, rank_kq, rank_v, err) of every layer is independent of the error budget.
A full curve trace (run to the 0.30 floor) therefore lets any EB vector be replayed exactly on CPU.

Selection rules replayed:
  no45  : frozen builder - evaluate candidate; accept if err <= eb; continue while err < eb and th > floor.
  plus45: frozen builder + official EigenAttn Llama min_thresh=0.45 loop condition (commit 8bd0d3fa,
          decompose/eigen_attn.py): the candidate is evaluated, then the loop continues only while
          err < eb AND rank_kq > 0.45*1024 AND rank_v > 0.45*1024; final = last accepted candidate
          (official step-back). A candidate violating the 45% condition is never accepted.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1")
ENV = Path("/diskF/jiaqizhang/llama3_full_resource_robustness_v1/ultra_aggressive_v1/envelope_attempt2/ultra_aggressive_curve_full_q_attempt2_perlayer_eb_report.json")
FULL = 268435456; MAXR = 1024; MT = 0.45 * MAXR
CAMP = Path("/diskF/jiaqizhang/fdp125_final_campaign_v1/results/llama")
RS = Path("/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1")
UCTRL = {"light": (0.044, RS / "resource_search_retry_gpu2_v2/light/eb_0p0440"),
         "medium": (0.0567, RS / "resource_search_retry_gpu2_v2/medium/eb_0p0567"),
         "aggressive": (0.0875, RS / "official_control/aggressive")}
UCTRL_SHA = {"light": "30b9f17d90f858b667c39c192abab405aff6093152083f06033d28a703d84a57",
             "aggressive": "baaee591e54fbb41e13119e1bf31e15bb26156216dde4f55674ce0454050d618"}
BASE_E = {"light": 0.045, "medium": 0.060, "aggressive": 0.090}
PREF = {"light": 219_000_000, "medium": 200_000_000, "aggressive": 184_549_376}


def fsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def json_sha(o): return hashlib.sha256(json.dumps(o, separators=(",", ":")).encode()).hexdigest()
def Bof(r): return sum((a + b) * 4096 for a, b in r)


rep = json.loads(ENV.read_text())
C = {l["layer"]: [(t["threshold"], t["rank_kq"], t["rank_v"], t["err"]) for t in l["runs"][-1]["threshold_trace"]] for l in rep["layers"]}
assert sorted(C) == list(range(32))


def select(ebv, rule45):
    ranks, stops = [], []
    for li in range(32):
        e = ebv[li]; pts = C[li]; best = (pts[0][1], pts[0][2]); stop = "floor"
        for th, rk, rv, er in pts:
            ok45 = (rk > MT and rv > MT) if rule45 else True
            if er <= e and ok45: best = (rk, rv)
            if not (er < e and ok45):
                stop = ("both" if (not er < e and not ok45) else ("error_budget" if not er < e else "min_thresh_45"))
                break
        ranks.append(list(best)); stops.append(stop)
    return ranks, stops


def stopdist(s): return {k: s.count(k) for k in ["error_budget", "min_thresh_45", "both", "floor"]}


out = {"envelope": str(ENV), "envelope_sha256": fsha(ENV), "points_per_layer": {li: len(C[li]) for li in C}, "full_kv_bytes": FULL}

# A1 curve consistency vs every frozen Uniform-control builder log
pat = re.compile(r"\[layer (\d+)\]\[eb=([0-9.]+)\] iter=(\d+) th=([0-9.]+) rank_k=(\d+) rank_v=(\d+) err=([0-9.]+)")
fin = re.compile(r"\[layer (\d+)\]\[eb=([0-9.]+)\] th=([0-9.]+) rank_k=(\d+) rank_v=(\d+) err=")
a1 = {}; logranks = {}
for t, (eb, d) in UCTRL.items():
    n = mism = 0; fr = {}
    for line in (d / "run.log").read_text().splitlines():
        m = pat.search(line)
        if m:
            li, it = int(m.group(1)), int(m.group(3))
            th, rk, rv, er = C[li][it]
            n += 1
            if not (abs(th - float(m.group(4))) < 5e-5 and rk == int(m.group(5)) and rv == int(m.group(6)) and abs(er - float(m.group(7))) < 5e-10):
                mism += 1
            continue
        m = fin.search(line)
        if m: fr[int(m.group(1))] = [int(m.group(4)), int(m.group(5))]
    logranks[t] = [fr[i] for i in range(32)] if len(fr) == 32 else None
    a1[t] = {"log_points": n, "mismatches": mism}
out["A1_curve_consistency_vs_frozen_uniform_logs"] = a1

# A2 identity gate (replay, 45% disabled)
a2 = {}
for t, (eb, d) in UCTRL.items():
    r, s = select([eb] * 32, False)
    ref = json.loads((d / "low_rank_config.json").read_text()) if (d / "low_rank_config.json").is_file() else logranks[t]
    meta = d / ("resource.json" if (d / "resource.json").is_file() else "provenance.json")
    row = {"eb": eb, "replay_B": Bof(r), "frozen_B": json.loads(meta.read_text())["B_live"],
           "ranks_equal_frozen": r == ref, "replay_rank_json_sha": json_sha(r)}
    if t in UCTRL_SHA:
        cands = {"json_compact": json_sha(r), "json_indent2": hashlib.sha256(json.dumps(r, indent=2).encode()).hexdigest(),
                 "json_indent2_nl": hashlib.sha256((json.dumps(r, indent=2) + "\n").encode()).hexdigest(),
                 "json_default": hashlib.sha256(json.dumps(r).encode()).hexdigest()}
        if (d / "low_rank_config.json").is_file(): cands["frozen_file_bytes"] = fsha(d / "low_rank_config.json")
        row["frozen_rank_sha256"] = UCTRL_SHA[t]; row["rank_sha_match_methods"] = [k for k, v in cands.items() if v == UCTRL_SHA[t]]
    a2["uniform_control_" + t] = row
for t in BASE_E:
    p = json.loads((CAMP / t / "fdp125/provenance.json").read_text())
    v = list(p["eb_vector"]); v[0] = p["base_e"]   # builder --freeze_layer0_eb --layer0_eb base_e
    r, s = select(v, False)
    a2["fdp_" + t] = {"replay_B": Bof(r), "frozen_B": p["B_live"], "ranks_equal_frozen": r == p["ranks"],
                      "replay_rank_sha": json_sha(r), "frozen_rank_sha": p["rank_sha"], "rank_sha_equal": json_sha(r) == p["rank_sha"]}
    pu = json.loads((CAMP / t / "uniform/provenance.json").read_text())
    r2, _ = select([p["base_e"]] * 32, False)
    a2["fdp_parent_uniform_" + t] = {"replay_B": Bof(r2), "frozen_B": pu["B_live"], "ranks_equal_frozen": r2 == pu["ranks"],
                                     "rank_sha_equal": json_sha(r2) == pu["rank_sha"]}
out["A2_identity_gate_replay_no45"] = a2
out["A2_identity_gate_replay_PASS"] = all(v.get("ranks_equal_frozen") and v.get("replay_B") == v.get("frozen_B") for v in a2.values())

# A3 Uniform+45 resource lattice (all distinct reachable B over EB)
errs = sorted({x[3] for c in C.values() for x in c})
probe = [errs[0] / 2] + [(a + b) / 2 for a, b in zip(errs, errs[1:])] + [10.0]
lat = {}
for e in probe:
    r, s = select([e] * 32, True)
    b = Bof(r)
    if b not in lat: lat[b] = {"eb": e, "rank_sha": json_sha(r), "stops": stopdist(s)}
L = sorted(lat.items())
out["A3_uniform45"] = {"min_B": L[0][0], "min_ratio": L[0][0] / FULL, "max_B": L[-1][0], "n_distinct_B": len(L),
                       "eb_at_min": L[0][1]["eb"], "stops_at_min": L[0][1]["stops"]}
out["A3_uniform45_lattice"] = [{"B": k, **d} for k, d in L]

# A4 per tier: FDP-parent Uniform+45 at base_e, FDP+45 proxy (historical eb vector), Uniform+45 matching
tiers = {}
for t, e in BASE_E.items():
    p = json.loads((CAMP / t / "fdp125/provenance.json").read_text())
    v = list(p["eb_vector"]); v[0] = e
    rp, sp = select([e] * 32, True)
    rf, sf = select(v, True)
    bf = Bof(rf)
    near = sorted(L, key=lambda kv: abs(kv[0] - bf))[:3]
    pref = sorted(L, key=lambda kv: abs(kv[0] - PREF[t]))[:3]
    tiers[t] = {"base_e": e,
        "parent_uniform45_at_base_e": {"B": Bof(rp), "ratio": Bof(rp) / FULL, "stops": stopdist(sp), "rank_sha": json_sha(rp)},
        "fdp45_PROXY_historical_eb_vector": {"B": bf, "ratio": bf / FULL, "stops": stopdist(sf), "rank_sha": json_sha(rf),
            "layers_at_45_floor": sum(1 for a, b in rf if a <= 471 or b <= 471),
            "note": "proxy only: the real FDP eb vector must come from profiling the new Uniform+45 parent"},
        "uniform45_nearest_to_fdp_proxy": [{"B": k, "gap_pct": 100 * (k - bf) / bf, "eb": d["eb"], "stops": d["stops"]} for k, d in near],
        "uniform45_nearest_to_preferred": {"preferred_B": PREF[t], "candidates": [{"B": k, "gap_pct": 100 * (k - PREF[t]) / PREF[t], "eb": d["eb"]} for k, d in pref]},
        "frozen_no45_fdp_B": p["B_live"],
        "frozen_no45_fdp_reachable_under_uniform45": p["B_live"] >= L[0][0]}
out["A4_tiers"] = tiers
(ROOT / "cpu_audit/CPU_FEASIBILITY_AUDIT.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ["A1_curve_consistency_vs_frozen_uniform_logs", "A2_identity_gate_replay_PASS", "A3_uniform45"]}, indent=1))
for k, v in out["A2_identity_gate_replay_no45"].items(): print(k, json.dumps(v)[:300])
for t, v in tiers.items(): print(t, json.dumps(v)[:1500])
