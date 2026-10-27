#!/usr/bin/env python3
"""Quality-blind resource diagnostic for a lower tier; never authorizes quality."""
import json
from pathlib import Path
import ultra_resource_scan as u

REPORT = Path("/diskF/jiaqizhang/llama3_full_resource_robustness_v1/ultra_aggressive_v1/envelope_attempt2/ultra_aggressive_curve_full_q_attempt2_perlayer_eb_report.json")
BASIS = Path("/diskF/jiaqizhang/llama3_full_resource_robustness_v1/ultra_aggressive_v1/envelope_attempt2/ultra_aggressive_curve_full_q_attempt2_perlayer_eb.pt")
OUT = Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_3COMP_5COMP_AUDIT/ULTRA_AGGRESSIVE_DS_DIAGNOSTIC_V1/next_lower_tier_protocol_audit")
OUT.mkdir(parents=True, exist_ok=True)
lock, inp, profile, rows, comp, qs = u.profile_and_q()
rep = json.loads(REPORT.read_text())
traces = [x["runs"][0]["threshold_trace"] for x in rep["layers"]]
if len(traces) != 32 or any(not t for t in traces):
    raise RuntimeError("rank curve incomplete")
result = {
    "schema": "FDP125_ULTRA_NEXT_LOWER_TIER_RESOURCE_ONLY_AUDIT_V1",
    "status": "PROTOCOL_DECISION_REQUIRED_BEFORE_QUALITY",
    "quality_evaluation": "NOT_RUN",
    "frozen_target_bytes": u.TARGET,
    "target_tolerance_fraction": u.TOL,
    "frozen_min_rank_per_projection": 64,
    "full_curve_report_sha256": u.sha_file(REPORT),
    "full_curve_basis_sha256": u.sha_file(BASIS),
    "profile_sha256": u.sha_file(profile),
    "method_lock_sha256": u.sha_file(u.METHOD),
    "arms": {}
}
for arm in ("U_FULL", "U_NO_S"):
    fac = u.effective_eps(1.0, qs[arm])
    coverage = min(float(t[-1]["err"])/fac[i] for i,t in enumerate(traces)
                   if float(t[-1]["threshold"]) > 0.30)
    all_states = [s for s in u.plateaus(qs[arm], traces) if s["mid"] <= coverage]
    above = [s for s in all_states if s["B"] >= u.TARGET]
    below = [s for s in all_states if s["B"] <= u.TARGET]
    if not above or not below:
        chosen = min(all_states, key=lambda s: (abs(s["B"]-u.TARGET), 0 if s["B"]>=u.TARGET else 1, abs(s["mid"]-u.START), s["sha"]))
        bracket = False
    else:
        lo = max(below, key=lambda s: s["B"])
        hi = min(above, key=lambda s: s["B"])
        choices = [lo] if lo["sig"] == hi["sig"] else [lo,hi]
        chosen = min(choices, key=lambda s: (abs(s["B"]-u.TARGET), 0 if s["B"]>=u.TARGET else 1, abs(s["mid"]-u.START), s["sha"]))
        bracket = True
    ranks = chosen["ranks"]
    eps = u.effective_eps(chosen["mid"], qs[arm])
    result["arms"][arm] = {
        "component_set": "d1+d2+d5" if arm == "U_FULL" else "d1+d5",
        "q_sha256_round8": u.q_sha_round8(qs[arm]),
        "epsilon_base": chosen["mid"], "epsilon_by_layer": eps,
        "rank_vector_kq_v": ranks, "rank_sha256": chosen["sha"],
        "B_live_bytes": chosen["B"], "gap_bytes": chosen["B"]-u.TARGET,
        "gap_pct": 100*(chosen["B"]-u.TARGET)/u.TARGET,
        "within_target_tolerance": abs(chosen["B"]-u.TARGET)/u.TARGET <= u.TOL,
        "min_rank": min(min(x) for x in ranks),
        "passes_frozen_rank_floor_64": all(k>=64 and v>=64 for k,v in ranks),
        "scalar_plateau": [chosen["lo"], chosen["hi"]],
        "curve_coverage_max": coverage,
        "target_bracketed": bracket,
    }
    cand = result["arms"][arm]
    cand["method"] = "TRUE_TWO_PASS_FDP_EB"
    cand["lambda"] = u.LAMBDA
    cand["rho"] = u.RHO
    cand["rank_floor_exception"] = cand["min_rank"] < 64
    cand["status"] = "PROTOCOL_EXCEPTION_DIAGNOSTIC_NOT_FORMAL_LOCK"
    cand_path = OUT/f"{arm}_next_lower_rank56_candidate.json"
    cand_path.write_text(json.dumps(cand, indent=2)+"\n")
    eps_input = {
        "error_budget_by_layer": eps,
        "source_profile_path": str(profile),
        "source_profile_sha256": u.sha_file(profile),
        "q_sha256": cand["q_sha256_round8"],
        "q_sha256_round8": cand["q_sha256_round8"],
        "candidate_rank_sha256": cand["rank_sha256"],
        "base_scalar": cand["epsilon_base"],
        "epsilon_base": cand["epsilon_base"],
        "lambda": u.LAMBDA,
        "rho": u.RHO,
        "layer0_freeze": True,
        "layer0_frozen": True,
        "rank_floor_exception": cand["rank_floor_exception"],
        "source_candidate_sha256": u.sha_file(cand_path),
    }
    eps_path = OUT/f"{arm}_next_lower_epsilon_builder_input.json"
    eps_path.write_text(json.dumps(eps_input,indent=2)+"\n")
    cand["candidate_json_sha256"] = u.sha_file(cand_path)
    cand["builder_input_sha256"] = u.sha_file(eps_path)
path = OUT/"ULTRA_NEXT_LOWER_TIER_RESOURCE_CANDIDATES_V2.json"
path.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":result["status"],"target_bytes":u.TARGET,"arms":{k:{x:v[x] for x in ("B_live_bytes","gap_bytes","gap_pct","min_rank","passes_frozen_rank_floor_64","rank_sha256","within_target_tolerance")} for k,v in result["arms"].items()},"path":str(path),"sha256":u.sha_file(path)},indent=2))
