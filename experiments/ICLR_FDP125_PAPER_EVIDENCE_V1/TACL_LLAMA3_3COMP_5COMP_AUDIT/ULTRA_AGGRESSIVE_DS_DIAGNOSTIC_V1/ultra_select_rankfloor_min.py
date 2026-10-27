#!/usr/bin/env python3
"""Freeze the rank-floor minimum-resource Ultra diagnostic arms, without quality."""
import argparse, csv, json
from pathlib import Path
import ultra_resource_scan as u

ap = argparse.ArgumentParser()
ap.add_argument("--envelope-report", required=True)
ap.add_argument("--envelope-basis", required=True)
ap.add_argument("--output-dir", required=True)
a = ap.parse_args()
outdir = Path(a.output_dir); outdir.mkdir(parents=True, exist_ok=True)
lock, inp, profile, rows, comp, qs = u.profile_and_q()
rep = json.loads(Path(a.envelope_report).read_text())
traces = [x["runs"][0]["threshold_trace"] for x in rep["layers"]]
if len(traces) != 32 or any(not x for x in traces):
    raise RuntimeError("incomplete rank-response traces")
# Reconfirm the Full Aggressive anchor from the same basis trace.
anchor = u.ranks_for(u.START, qs["U_FULL"], traces)
hist = json.loads(Path("/diskF/jiaqizhang/fdp_component_subset_search_v2/results/aggressive/11001/provenance.json").read_text())
if anchor != hist["final_rank_vector"] or u.rank_sha(anchor) != hist["final_rank_sha"] or u.b_live(anchor) != u.TARGET_ANCHOR:
    raise RuntimeError("frozen Aggressive Full anchor replay failed")
summary = {
    "schema": "FDP125_ULTRA_RANKFLOOR_MIN_RESOURCE_DIAGNOSTIC_V1",
    "status": "RESOURCE_CANDIDATES_FROZEN_PENDING_CANONICAL_REPLAY",
    "quality_blind": True,
    "quality_evaluation": "NOT_RUN",
    "quality_authorization": "User authorized PPL at these rank-floor minimum resource points; execution remains gated on canonical replay, realization, parity, and canary.",
    "model": lock["model"],
    "target_bytes_unchanged": u.TARGET,
    "target_tolerance_fraction": u.TOL,
    "rank_floor": 64,
    "anchor_bytes": u.TARGET_ANCHOR,
    "anchor_replay": {"pass": True, "rank_sha256": u.rank_sha(anchor), "B_live_bytes": u.b_live(anchor)},
    "method_lock_sha256": u.sha_file(u.METHOD),
    "protocol_lock_sha256": u.sha_file(u.E / "FULL_PROFILE_RESOURCE_ROBUSTNESS_PROTOCOL_LOCK_V1.1.json"),
    "profile_path": str(profile), "profile_sha256": u.sha_file(profile),
    "rank_trace_report_path": str(Path(a.envelope_report)),
    "rank_trace_report_sha256": u.sha_file(a.envelope_report),
    "rank_trace_basis_path": str(Path(a.envelope_basis)),
    "rank_trace_basis_sha256": u.sha_file(a.envelope_basis),
    "full_q_sha256_frozen": inp["q_sha256"],
    "full_q_replay_max_abs_error": max(abs(x-y) for x,y in zip(qs["U_FULL"], inp["q_minmax"])),
    "arms": {}
}
for arm in ("U_FULL", "U_NO_S"):
    fac = u.effective_eps(1.0, qs[arm])
    coverage = min(float(t[-1]["err"]) / fac[i] for i,t in enumerate(traces)
                   if float(t[-1]["threshold"]) > 0.30)
    segs = u.plateaus(qs[arm], traces)
    legal = [s for s in segs if s["mid"] <= coverage and
             all(k >= 64 and v >= 64 for k,v in s["ranks"])]
    if not legal:
        raise RuntimeError(f"no verified legal rank-floor schedule for {arm}")
    chosen = min(legal, key=lambda s: (s["B"], abs(s["mid"]-u.START), s["sha"]))
    eps = u.effective_eps(chosen["mid"], qs[arm])
    gap = chosen["B"] - u.TARGET
    cand = {
        "arm": arm, "component_set": "d1+d2+d5" if arm == "U_FULL" else "d1+d5",
        "method": "TRUE_TWO_PASS_FDP_EB", "lambda": u.LAMBDA, "rho": u.RHO,
        "epsilon_base": chosen["mid"], "scalar_plateau": [chosen["lo"], chosen["hi"]],
        "epsilon_by_layer": eps, "rank_vector_kq_v": chosen["ranks"],
        "rank_sha256": chosen["sha"], "q_sha256_round8": u.q_sha_round8(qs[arm]),
        "B_live_bytes": chosen["B"], "B_target_bytes_unchanged": u.TARGET,
        "target_gap_bytes": gap, "target_gap_pct": 100*gap/u.TARGET,
        "within_original_target_tolerance": abs(gap)/u.TARGET <= u.TOL,
        "rank_floor_min": 64, "all_ranks_legal": True,
        "verified_scalar_curve_coverage_max": coverage,
        "selection_rule": "minimum B_live among all verified rank-response plateaus with every QK/V rank >=64; ties by scalar distance to 0.09 then rank SHA; quality blind",
        "quality_status": "NOT_RUN_PENDING_CANONICAL_REPLAY_PARITY_CANARY"
    }
    cand_path = outdir / f"{arm}_rankfloor_min_candidate.json"
    cand_path.write_text(json.dumps(cand, indent=2)+"\n")
    eps_input = {
        "error_budget_by_layer": eps, "source_profile_path": str(profile),
        "source_profile_sha256": u.sha_file(profile), "component_set": cand["component_set"],
        "q_sha256": cand["q_sha256_round8"], "q_sha256_round8": cand["q_sha256_round8"],
        "candidate_rank_sha256": cand["rank_sha256"],
        "base_scalar": cand["epsilon_base"], "epsilon_base": cand["epsilon_base"],
        "lambda": u.LAMBDA, "rho": u.RHO,
        "layer0_freeze": True, "layer0_frozen": True,
        "source_candidate_sha256": u.sha_file(cand_path)
    }
    eps_path = outdir / f"{arm}_epsilon_builder_input.json"
    eps_path.write_text(json.dumps(eps_input, indent=2)+"\n")
    csv_path = outdir / f"{arm}_per_layer_schedule.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer","d1","d2","d5","q","epsilon","qk_rank","v_rank"])
        w.writeheader()
        for i in range(32):
            w.writerow({"layer":i,"d1":comp[i]["d1"],"d2":comp[i]["d2"],"d5":comp[i]["d5"],
                        "q":qs[arm][i],"epsilon":eps[i],"qk_rank":chosen["ranks"][i][0],"v_rank":chosen["ranks"][i][1]})
    cand["candidate_json_sha256"] = u.sha_file(cand_path)
    cand["builder_input_sha256"] = u.sha_file(eps_path)
    cand["per_layer_csv_sha256"] = u.sha_file(csv_path)
    summary["arms"][arm] = cand
summary["status"] = "RESOURCE_CANDIDATES_FROZEN_PENDING_CANONICAL_REPLAY"
path = outdir / "ULTRA_RANKFLOOR_MIN_RESOURCE_CANDIDATES.json"
path.write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps({k:{a:{x:v[x] for x in ("epsilon_base","B_live_bytes","target_gap_bytes","target_gap_pct","rank_sha256","within_original_target_tolerance")} for a,v in summary[k].items()} for k in ("arms",)}, indent=2))
print("summary", path, u.sha_file(path))
