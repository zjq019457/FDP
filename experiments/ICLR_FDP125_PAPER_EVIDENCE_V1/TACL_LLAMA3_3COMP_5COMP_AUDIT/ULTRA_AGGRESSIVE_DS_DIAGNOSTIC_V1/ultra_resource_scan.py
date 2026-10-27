#!/usr/bin/env python3
"""Quality-blind resource schedule search for the FDP-EB Ultra diagnostic.

This tool reads only the frozen component/profile input, the protocol lock,
and canonical builder threshold traces. It never reads evaluation outputs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

E = Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_3COMP_5COMP_AUDIT/FULL_PROFILE_RESOURCE_ROBUSTNESS_V1")
F = Path("/diskF/jiaqizhang/llama3_full_resource_robustness_v1")
PROFILE_ROOT = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles")
OUT_E = Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_3COMP_5COMP_AUDIT/ULTRA_AGGRESSIVE_DS_DIAGNOSTIC_V1")
OUT_F = F / "ultra_aggressive_v1" / "envelope_attempt2"
METHOD = Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_3COMP_5COMP_AUDIT/LLAMA3_3COMP_TRUE_TWO_PASS_METHOD_LOCK.json")
TARGET_ANCHOR = 159_416_320
TARGET = 143_474_688
TOL = 0.0015
START = 0.09
LAMBDA = 1.2
RHO = 0.75
BYTES_PER_RANK = 4096


def jbytes(x):
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def profile_and_q():
    lock = json.loads((E / "FULL_PROFILE_RESOURCE_ROBUSTNESS_PROTOCOL_LOCK_V1.1.json").read_text())
    inp = lock["tier_inputs"]["AGGRESSIVE"]
    profile = Path(inp["profile_path"])
    if sha_file(profile) != inp["profile_sha256"]:
        raise RuntimeError("frozen Aggressive profile hash mismatch")
    raw = json.loads(profile.read_text())["per_layer"]
    rows = [raw[str(i)] if isinstance(raw, dict) else raw[i] for i in range(32)]
    sums = {"U_FULL": [], "U_NO_S": []}
    comp = []
    for r in rows:
        d1 = float(r["Z_rel_mse"])
        d2 = float(r["score_rel_mse_causal"])
        d5 = max(0.0, -float(r["entropy_diff_A"]))
        comp.append({"d1": d1, "d2": d2, "d5": d5})
        sums["U_FULL"].append(d1 + d2 + d5)
        sums["U_NO_S"].append(d1 + d5)
    qs = {}
    for arm, vals in sums.items():
        lo, hi = min(vals), max(vals)
        if hi == lo:
            raise RuntimeError(f"constant component sum for {arm}")
        qs[arm] = [(v - lo) / (hi - lo) for v in vals]
    max_err = max(abs(a - b) for a, b in zip(qs["U_FULL"], inp["q_minmax"]))
    if max_err > 1e-12:
        raise RuntimeError(f"computed U-FULL q does not replay frozen q: max_abs={max_err}")
    return lock, inp, profile, rows, comp, qs


def effective_eps(base, q):
    raw = [float(base) / (1 + LAMBDA * float(x)) for x in q]
    mean = sum(raw) / len(raw)
    eps = [x * float(base) / mean for x in raw]
    # Reproduce the existing geometric layer-0 blend and canonical L0 freeze.
    eps[0] = math.exp(RHO * math.log(float(base)) + (1 - RHO) * math.log(eps[0]))
    eps[0] = float(base)
    return eps


def ranks_for(base, q, traces):
    eps = effective_eps(base, q)
    ranks = []
    for i, tr in enumerate(traces):
        best = [1024, 1024]
        for row in tr:
            if float(row["err"]) <= eps[i] and float(row["threshold"]) >= 0.30:
                best = [int(row["rank_kq"]), int(row["rank_v"])]
            else:
                break
        ranks.append(best)
    return ranks


def b_live(ranks):
    return sum(k + v for k, v in ranks) * BYTES_PER_RANK


def rank_sha(ranks):
    return sha_bytes(jbytes(ranks))


def q_sha_round8(q):
    return sha_bytes(jbytes([round(float(x), 8) for x in q]))


def cmd_prepare(args):
    lock, inp, profile, rows, comp, qs = profile_and_q()
    out_f = Path(args.output_dir) if args.output_dir else OUT_F
    out_f.mkdir(parents=True, exist_ok=True)
    base = args.envelope_base
    eps = effective_eps(base, qs["U_FULL"])
    ep = out_f / "ultra_curve_envelope_epsilon.json"
    payload = {
        "schema": "FDP125_ULTRA_RESOURCE_CURVE_ENVELOPE_INPUT_V1",
        "purpose": "resource-only native-rank threshold trace coverage; not a quality arm",
        "quality_blind": True,
        "envelope_base": base,
        "lambda": LAMBDA,
        "rho": RHO,
        "layer0_frozen": True,
        "error_budget_by_layer": eps,
        "profile_path": str(profile),
        "profile_sha256": sha_file(profile),
        "frozen_full_q_sha256": inp["q_sha256"],
        "frozen_full_q_sha256_round8": inp.get("q_sha256_round8"),
        "q_full_max_abs_replay_error": max(abs(a-b) for a,b in zip(qs["U_FULL"], inp["q_minmax"])),
        "rank_trace_source_method_lock_sha256": sha_file(METHOD),
        "rank_trace_builder_sha256": lock["sources"]["rank_builder"]["sha256"],
        "acceptance_target_bytes": TARGET,
    }
    ep.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "ENVELOPE_INPUT_READY", "path": str(ep), "sha256": sha_file(ep), "epsilon_min": min(eps), "epsilon_max": max(eps), "profile_sha256": sha_file(profile), "full_q_replay_max_abs": payload["q_full_max_abs_replay_error"]}, indent=2))


def plateaus(q, traces):
    fac = effective_eps(1.0, q)
    events = {START}
    for i, tr in enumerate(traces):
        for row in tr:
            v = float(row["err"]) / fac[i]
            if v > START and math.isfinite(v):
                events.add(v)
    events = sorted(events)
    if len(events) < 2:
        raise RuntimeError("insufficient rank-transition events")
    points = [(a+b)/2 for a,b in zip(events[:-1], events[1:])]
    points.insert(0, START)
    points.append(events[-1] + max(1e-9, abs(events[-1]) * 1e-9))
    segs = []
    for base in points:
        r = ranks_for(base, q, traces)
        sig = tuple(tuple(x) for x in r)
        if segs and segs[-1]["sig"] == sig:
            segs[-1]["hi"] = base
            segs[-1]["mid"] = (segs[-1]["lo"] + base) / 2
        else:
            segs.append({"lo": base, "hi": base, "mid": base, "sig": sig, "ranks": r, "B": b_live(r), "sha": rank_sha(r)})
    segs.sort(key=lambda s: (s["mid"], s["sha"]))
    # Ensure the response is monotone over the resource search.
    for a, b in zip(segs, segs[1:]):
        if b["B"] > a["B"]:
            raise RuntimeError("B_live is not monotone with increasing epsilon_base")
    return segs


def select_candidate(segs, arm, q, traces, inp, coverage_base):
    # Frozen three-component method lock specifies min rank 64. The canonical
    # builder traces can include its optional strict-original sub-64 states;
    # those are excluded from this formal resource search.
    legal = [s for s in segs if s["mid"] <= coverage_base and all(k >= 64 and v >= 64 for k, v in s["ranks"])]
    if not legal:
        raise RuntimeError(f"no schedule in the documented >=64 legal rank lattice within verified curve coverage for {arm}")
    above = [s for s in legal if s["B"] >= TARGET]
    below = [s for s in legal if s["B"] <= TARGET]
    if not above or not below:
        raise RuntimeError(f"verified curve coverage does not bracket frozen target for {arm}; schedule not frozen")
    if above and below:
        a = max(above, key=lambda s: (s["B"], -s["mid"]))
        b = max(below, key=lambda s: (s["B"], -s["mid"]))
        adjacent = [a] if a["sig"] == b["sig"] else [a, b]
    else:
        # User's frozen rule: if no legal schedule is within tolerance, report
        # the nearest legal schedule; do not move the target.
        adjacent = legal
    chosen = min(adjacent, key=lambda s: (abs(s["B"]-TARGET), 0 if s["B"] >= TARGET else 1, abs(s["mid"]-START), s["sha"]))
    gap = chosen["B"] - TARGET
    eps = effective_eps(chosen["mid"], q)
    return {
        "arm": arm,
        "method": "TRUE_TWO_PASS_FDP_EB",
        "component_set": "d1+d2+d5" if arm == "U_FULL" else "d1+d5",
        "epsilon_base": chosen["mid"],
        "scalar_plateau": [chosen["lo"], chosen["hi"]],
        "B_target_bytes": TARGET,
        "B_live_bytes": chosen["B"],
        "target_gap_bytes": gap,
        "target_gap_pct": 100 * gap / TARGET,
        "within_0p15_percent": abs(gap) / TARGET <= TOL,
        "rank_floor_min": 64,
        "all_ranks_legal": all(k >= 64 and v >= 64 for k, v in chosen["ranks"]),
        "verified_scalar_curve_coverage_max": coverage_base,
        "rank_vector_kq_v": chosen["ranks"],
        "rank_sha256": chosen["sha"],
        "q_sha256_round8": q_sha_round8(q),
        "q_full_sha256_from_frozen_lock": inp["q_sha256"] if arm == "U_FULL" else None,
        "profile_path": inp["profile_path"],
        "profile_sha256": inp["profile_sha256"],
        "lambda": LAMBDA,
        "rho": RHO,
        "layer0_frozen": True,
        "epsilon_by_layer": eps,
        "quality_blind": True,
    }


def cmd_scan(args):
    lock, inp, profile, rows, comp, qs = profile_and_q()
    env_report = Path(args.envelope_report)
    env_basis = Path(args.envelope_basis)
    if not env_report.is_file() or not env_basis.is_file():
        raise RuntimeError("envelope report/basis missing")
    rep = json.loads(env_report.read_text())
    traces = [x["runs"][0]["threshold_trace"] for x in rep["layers"]]
    if len(traces) != 32 or any(not t for t in traces):
        raise RuntimeError("threshold trace layer count/empties")
    # Validate frozen Full and historical anchor at epsilon_base=.09.
    full_anchor = ranks_for(START, qs["U_FULL"], traces)
    historical_root = Path("/diskF/jiaqizhang/fdp_component_subset_search_v2/results/aggressive")
    historical_full = json.loads((historical_root / "11001/provenance.json").read_text())
    if full_anchor != historical_full["final_rank_vector"] or rank_sha(full_anchor) != historical_full["final_rank_sha"] or b_live(full_anchor) != TARGET_ANCHOR:
        raise RuntimeError("Full anchor does not reproduce the frozen Aggressive Full point")
    no_s_anchor = ranks_for(START, qs["U_NO_S"], traces)
    historical_no_s = json.loads((historical_root / "10001/provenance.json").read_text())
    no_s_anchor_match = no_s_anchor == historical_no_s["final_rank_vector"] and rank_sha(no_s_anchor) == historical_no_s["final_rank_sha"]

    scan_by_arm = {}
    candidates = {}
    coverage = {}
    for arm in ("U_FULL", "U_NO_S"):
        segs = plateaus(qs[arm], traces)
        scan_by_arm[arm] = segs
        fac = effective_eps(1.0, qs[arm])
        cover = float("inf")
        for i, tr in enumerate(traces):
            if float(tr[-1]["threshold"]) <= 0.30 + 1e-9:
                continue
            cover = min(cover, float(tr[-1]["err"]) / fac[i])
        coverage[arm] = cover
        candidates[arm] = select_candidate(segs, arm, qs[arm], traces, inp, cover)

    full_base = candidates["U_FULL"]["epsilon_base"]
    native_no_s = ranks_for(full_base, qs["U_NO_S"], traces)
    native_full = candidates["U_FULL"]["rank_vector_kq_v"]
    deltas = [{"layer": i, "qk_delta": native_no_s[i][0]-native_full[i][0], "v_delta": native_no_s[i][1]-native_full[i][1]} for i in range(32)]
    changed_layers = [x for x in deltas if x["qk_delta"] or x["v_delta"]]
    native = {
        "epsilon_base": full_base,
        "U_FULL_B_live_bytes": b_live(native_full),
        "U_NO_S_same_epsilon_B_live_bytes": b_live(native_no_s),
        "B_live_delta_NO_S_minus_FULL_bytes": b_live(native_no_s)-b_live(native_full),
        "changed_layer_count": len(changed_layers),
        "total_qk_rank_delta_NO_S_minus_FULL": sum(x["qk_delta"] for x in deltas),
        "total_v_rank_delta_NO_S_minus_FULL": sum(x["v_delta"] for x in deltas),
        "max_abs_per_layer_qk_rank_delta": max(abs(x["qk_delta"]) for x in deltas),
        "max_abs_per_layer_v_rank_delta": max(abs(x["v_delta"]) for x in deltas),
        "rank_sha256": rank_sha(native_no_s),
        "allocation_equivalent": native_no_s == native_full,
        "documented_min_rank_64_pass": all(k >= 64 and v >= 64 for k, v in native_no_s),
        "rank_lattice_note": "native builder trace values are stored unchanged; formal selection filters out any schedule below the frozen >=64 per-projection floor",
        "per_layer_rank_deltas": deltas,
        "quality_evaluation": "NOT_RUN_NOT_AUTHORIZED_FOR_NATIVE_DIAGNOSTIC",
    }
    out_e = Path(args.output_dir) if args.output_dir else OUT_E
    out_e.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "FDP125_ULTRA_AGGRESSIVE_DS_RESOURCE_CANDIDATES_V1",
        "status": "RESOURCE_CANDIDATES_FROZEN_PENDING_OFFICIAL_BUILDER_REPLAY",
        "quality_blind": True,
        "quality_blind_note": "An initial broad read-only directory inventory incidentally displayed existing quality fields; no quality values are consumed by this resource-only deterministic selector. Deviation disclosed in task log.",
        "model": lock["model"],
        "target_anchor_bytes": TARGET_ANCHOR,
        "target_ultra_bytes": TARGET,
        "target_definition": "0.90 * 159416320",
        "acceptance_abs_gap_fraction": TOL,
        "method_lock_path": str(METHOD),
        "method_lock_sha256": sha_file(METHOD),
        "protocol_lock_path": str(E / "FULL_PROFILE_RESOURCE_ROBUSTNESS_PROTOCOL_LOCK_V1.1.json"),
        "protocol_lock_sha256": sha_file(E / "FULL_PROFILE_RESOURCE_ROBUSTNESS_PROTOCOL_LOCK_V1.1.json"),
        "profile_path": str(profile),
        "profile_sha256": sha_file(profile),
        "rank_trace_report_path": str(env_report),
        "rank_trace_report_sha256": sha_file(env_report),
        "rank_trace_basis_path": str(env_basis),
        "rank_trace_basis_sha256": sha_file(env_basis),
        "verified_scalar_curve_coverage_max_by_arm": coverage,
        "full_q_sha256_frozen": inp["q_sha256"],
        "full_q_replay_max_abs_error": max(abs(a-b) for a,b in zip(qs["U_FULL"], inp["q_minmax"])),
        "no_s_q_sha256_round8": q_sha_round8(qs["U_NO_S"]),
        "anchor_replay": {"U_FULL_matches_historical": True, "U_FULL_B_live_bytes": b_live(full_anchor), "U_NO_S_matches_historical_component_search": no_s_anchor_match, "U_NO_S_historical_rank_sha": historical_no_s["final_rank_sha"], "U_NO_S_current_trace_rank_sha": rank_sha(no_s_anchor)},
        "resource_candidates": candidates,
        "same_epsilon_native_allocation_effect": native,
        "selection_rule": "resource-only; compare the two adjacent legal rank plateaus bracketing frozen target; minimize absolute B_live gap; exact tie prefers at/above target; then scalar nearest .09; then lexicographic rank SHA",
    }
    out = out_e / "ULTRA_AGGRESSIVE_RESOURCE_CANDIDATES.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for arm in ("U_FULL", "U_NO_S"):
        cand = candidates[arm]
        rows_out = []
        q = qs[arm]
        eps = cand["epsilon_by_layer"]
        ranks = cand["rank_vector_kq_v"]
        for i in range(32):
            rows_out.append({"layer": i, "d1": comp[i]["d1"], "d2": comp[i]["d2"], "d5": comp[i]["d5"], "q": q[i], "epsilon": eps[i], "qk_rank": ranks[i][0], "v_rank": ranks[i][1]})
        with (out_e / f"{arm}_per_layer_schedule.csv").open("w", newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
        with (out_e / f"{arm}_resource_plateaus.csv").open("w", newline="") as f:
            w=csv.DictWriter(f,fieldnames=["scalar_lo","scalar_hi","scalar_mid","B_live_bytes","gap_bytes","gap_pct","rank_floor_64_pass","rank_sha256"]); w.writeheader()
            for s in scan_by_arm[arm]: w.writerow({"scalar_lo":s["lo"],"scalar_hi":s["hi"],"scalar_mid":s["mid"],"B_live_bytes":s["B"],"gap_bytes":s["B"]-TARGET,"gap_pct":100*(s["B"]-TARGET)/TARGET,"rank_floor_64_pass":all(k>=64 and v>=64 for k,v in s["ranks"]),"rank_sha256":s["sha"]})
        (out_e / f"{arm}_schedule_candidate.json").write_text(json.dumps(cand, indent=2) + "\n")
        # Canonical builder input schema: preserve the selected per-layer EB
        # vector verbatim and bind it to this resource-only candidate.
        eps_input = {
            "error_budget_by_layer": eps,
            "source_profile_path": str(profile),
            "source_profile_sha256": sha_file(profile),
            "component_set": cand["component_set"],
            "q_sha256_round8": cand["q_sha256_round8"],
            "candidate_rank_sha256": cand["rank_sha256"],
            "epsilon_base": cand["epsilon_base"],
            "lambda": LAMBDA,
            "rho": RHO,
            "layer0_frozen": True,
            "source_resource_candidate_sha256": sha_file(out),
        }
        (out_e / f"{arm}_epsilon_builder_input.json").write_text(json.dumps(eps_input, indent=2) + "\n")
    (out_e / "U_NO_S_same_epsilon_native_effect.json").write_text(json.dumps(native, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "anchor_replay": result["anchor_replay"], "candidates": {k: {"epsilon_base":v["epsilon_base"],"B_live_bytes":v["B_live_bytes"],"target_gap_bytes":v["target_gap_bytes"],"target_gap_pct":v["target_gap_pct"],"within_0p15_percent":v["within_0p15_percent"],"rank_sha256":v["rank_sha256"]} for k,v in candidates.items()}, "native_effect": native, "output":str(out), "output_sha256":sha_file(out)}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--envelope-base", type=float, default=2.0)
    p.add_argument("--output-dir", default="")
    p.set_defaults(func=cmd_prepare)
    s = sub.add_parser("scan")
    s.add_argument("--envelope-report", required=True)
    s.add_argument("--envelope-basis", required=True)
    s.add_argument("--output-dir", default="")
    s.set_defaults(func=cmd_scan)
    a = ap.parse_args()
    a.func(a)
