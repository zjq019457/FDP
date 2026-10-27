#!/usr/bin/env python3
"""Quality-blind diagnosis of verified curve coverage versus frozen target."""
import json
import ultra_resource_scan as u
from pathlib import Path

report = "/diskF/jiaqizhang/llama3_full_resource_robustness_v1/ultra_aggressive_v1/envelope_target_attempt3/ultra_aggressive_curve_full_q_target_attempt3_perlayer_eb_report.json"
rep = json.load(open(report))
traces = [x["runs"][0]["threshold_trace"] for x in rep["layers"]]
_, _, _, _, _, qs = u.profile_and_q()
out = {"target_bytes": u.TARGET, "arms": {}}
for arm in ("U_FULL", "U_NO_S"):
    fac = u.effective_eps(1.0, qs[arm])
    coverage = min(float(t[-1]["err"]) / fac[i] for i, t in enumerate(traces)
                   if float(t[-1]["threshold"]) > 0.30)
    ranks = u.ranks_for(coverage, qs[arm], traces)
    segs = u.plateaus(qs[arm], traces)
    legal = [s for s in segs if s["mid"] <= coverage and
             all(k >= 64 and v >= 64 for k, v in s["ranks"])]
    legal_min = min(legal, key=lambda s: s["B"]) if legal else None
    legal_max = max(legal, key=lambda s: s["B"]) if legal else None
    out["arms"][arm] = {
        "verified_base_coverage_max": coverage,
        "B_live_at_coverage": u.b_live(ranks),
        "target_gap_bytes_at_coverage": u.b_live(ranks) - u.TARGET,
        "min_qk_or_v_rank": min(min(x) for x in ranks),
        "all_ranks_ge_64": all(k >= 64 and v >= 64 for k, v in ranks),
        "legal_rank_floor_min_B_live": legal_min["B"] if legal_min else None,
        "legal_rank_floor_min_gap": legal_min["B"] - u.TARGET if legal_min else None,
        "legal_rank_floor_min_schedule_min_rank": min(min(x) for x in legal_min["ranks"]) if legal_min else None,
        "legal_rank_floor_min_rank_sha256": legal_min["sha"] if legal_min else None,
        "legal_rank_floor_min_scalar_plateau": [legal_min["lo"], legal_min["hi"]] if legal_min else None,
        "legal_rank_floor_min_rank_vector_kq_v": legal_min["ranks"] if legal_min else None,
        "legal_max_B_live": legal_max["B"] if legal_max else None,
        "target_bracketed": u.b_live(ranks) <= u.TARGET,
    }
out["source_profile_path"] = str(u.profile_and_q()[2])
out["source_profile_sha256"] = u.sha_file(u.profile_and_q()[2])
out["rank_trace_report_sha256"] = u.sha_file(report)
out["rank_trace_basis_sha256"] = u.sha_file("/diskF/jiaqizhang/llama3_full_resource_robustness_v1/ultra_aggressive_v1/envelope_target_attempt3/ultra_aggressive_curve_full_q_target_attempt3_perlayer_eb.pt")
out["method_lock_sha256"] = u.sha_file(u.METHOD)
out["status"] = "PRELIMINARY_TARGET_NOT_REACHED_ON_VERIFIED_CURVE_WITH_RANK_FLOOR_64"
out["quality_evaluation"] = "NOT_RUN"
Path("ULTRA_TARGET_FEASIBILITY_DIAGNOSTIC_attempt3.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
