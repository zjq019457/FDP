#!/usr/bin/env python3
"""Llama-2 Uniform schedules WITH the released-code 45% rank floor, by exact CPU replay of the historical builder traces
(per-layer curves are budget-independent: dense outputs propagate). The floor only stops a search earlier, so each historical
trace (which runs until the error budget is exceeded) covers the floored search at the same budget.
Rule (upstream 8bd0d3fa Llama branch): continue while err < eb and rank_qk > 0.45*4096 and rank_v > 0.45*4096 for the candidate
just evaluated; keep the last candidate that passed. Gate: the same replay without the floor must reproduce the historical
Uniform schedules exactly."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l2x_lib as X

MT = 0.45 * 4096
SRC = {"068": ("eb070_eb090_shared_decomp/llama2_7b_uniform_pass1_eb070_eb090_report.json", 0, 0.068),
       "060": ("llama2_7b_uniform_pass1_e060_report.json", 0, 0.060),
       "090": ("eb070_eb090_shared_decomp/llama2_7b_uniform_pass1_eb070_eb090_report.json", 1, 0.090)}


def select(trace, eb, floor):
    best = (trace[0]["rank_kq"], trace[0]["rank_v"]); stop = "trace_end"
    for t in trace:
        ok = (t["rank_kq"] > MT and t["rank_v"] > MT) if floor else True
        if t["err"] <= eb and ok: best = (t["rank_kq"], t["rank_v"])
        if not (t["err"] < eb and ok):
            stop = "both" if (t["err"] >= eb and not ok) else ("error_budget" if t["err"] >= eb else "floor"); break
    return [int(best[0]), int(best[1])], stop


out = {}
for k, (f, i, eb) in SRC.items():
    r = X.rjson(X.REP / f)
    tr = [l["runs"][i]["threshold_trace"] for l in r["layers"]]
    no = [select(t, eb, False)[0] for t in tr]
    ref = X.uniform_pairs("e" + k)
    gate = no == ref
    sel = [select(t, eb, True) for t in tr]
    fl = [s[0] for s in sel]; stops = [s[1] for s in sel]
    if any(s == "trace_end" for s in stops): raise SystemExit(f"f{k}: trace ended before a stop")
    out[k] = {"eb": eb, "gate_no_floor_reproduces_historical": gate, "layers_differing_from_no_floor": sum(a != b for a, b in zip(fl, no)),
              "floor_stops": stops.count("floor") + stops.count("both"), "B_live": X.B_of(fl), "ratio": X.B_of(fl) / X.FULL}
    print(k, out[k])
    if not gate: raise SystemExit(f"GATE FAIL e{k}: no-floor replay != historical Uniform")
    p = X.ROOT / "schedules" / f"uniform_f{k}" / "SCHEDULE.json"
    if not p.exists():
        X.wjson(p, {"method": "Uniform with the 45% rank floor (exact replay of historical builder trace)", "op": f"f{k}", "eb": eb, "rank_pairs": fl,
                    "B_live": X.B_of(fl), "rank_sha": X.sha_json(fl), "stops": stops, "source_report": str(X.REP / f), "run_index": i, "at": X.now()})
X.wjson(X.ROOT / "results" / "FLOOR_UNIFORM_REPLAY.json", out)
