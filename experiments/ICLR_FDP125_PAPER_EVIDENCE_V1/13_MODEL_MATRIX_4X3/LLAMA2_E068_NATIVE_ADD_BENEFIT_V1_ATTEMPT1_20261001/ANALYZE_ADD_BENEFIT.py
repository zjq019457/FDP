#!/usr/bin/env python3
import csv
import hashlib
import json
import pathlib
from datetime import datetime, timezone

BASE = pathlib.Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/13_MODEL_MATRIX_4X3")
OUT = BASE / "LLAMA2_E068_NATIVE_ADD_BENEFIT_V1_ATTEMPT1_20261001"
REMOVE = BASE / "LLAMA2_E068_NATIVE_FRAGILITY_V1_ATTEMPT7_a06651aa"
RANK_CSV = BASE / "LLAMA2_OPERATING_POINT_E068_FDP070_CAPPED_V1/LLAMA2_E068_UNIFORM_FDP070_CAPPED_PER_LAYER.csv"
Q_JSON = BASE / "LLAMA2_CLEAN_REDESIGN_UNIFORM068_PROFILE_V1_20261001T1516_ATTEMPT7/LLAMA2_CLEAN_REDESIGN_D1_D2_D5_PROFILE_V1.json"
QUANTUM = 32
BYTES_PER_RANK = 4096
DELTA_BYTES = QUANTUM * BYTES_PER_RANK
BASELINE_BYTES = 746_323_968
EXPECTED = {
    "add_protocol_lock_sha256": "b85c02610e1f6a1e19df1ca069d5d5bb1c837583938e725f3f0429435f71bb5e",
    "basis_sha256": "196863f8ac17c17b276a8dc74a9dee9e30ee8e540f960c0638dccf86095f01d0",
    "profile_q_sha256": "720bbbe44103c95e1a2aa04ec4f0ad5cb85f343bab27ca98b7bc81f5ea066cae",
    "rank_schedule_sha256": "590129c5a0401474266d7652601461b52f7be452ced772dd489b25f3a744913c",
    "tuning_manifest_sha256": "53532566c8fc635e66a556c51db96ef423ce321adf50b18a8dc8ee4ad2e72194",
    "tuning_windows_sha256": "ce6e67bccf7f142a6d24da2185a7e7a22f9ed9371902dbfa567e23c08005a115",
    "remove_csv_sha256": "ef6e089799bfb6abe761d666435fe8e384bd0062e24ac424b17dda09ad0d63a3",
}


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_json(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def main():
    lock_path = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_PROTOCOL_LOCK.json"
    rule_path = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_STABILITY_RULE_ADDENDUM.json"
    canary_path = OUT / "MATERIALIZATION_CANARY_GPU0.json"
    assert sha(lock_path) == EXPECTED["add_protocol_lock_sha256"]
    lock = json.loads(lock_path.read_text())
    assert lock["reference"]["basis_sha256"] == EXPECTED["basis_sha256"]
    assert lock["reference"]["q_sha256"] == EXPECTED["profile_q_sha256"]
    assert lock["reference"]["rank_schedule_sha256"] == EXPECTED["rank_schedule_sha256"]
    assert lock["tuning_split"]["manifest_sha256"] == EXPECTED["tuning_manifest_sha256"]
    assert lock["tuning_split"]["window_data_sha256"] == EXPECTED["tuning_windows_sha256"]
    assert sha(REMOVE / "LLAMA2_E068_NATIVE_FRAGILITY.csv") == EXPECTED["remove_csv_sha256"]
    assert json.loads(canary_path.read_text())["status"] == "PASS_BASELINE_BUILD_AND_FORWARD_CANARY"
    rule = json.loads(rule_path.read_text())

    add_raw = []
    worker_summaries = []
    for gpu, expected_rows in ((0, 22), (1, 21), (2, 21)):
        probe = OUT / f"worker_gpu{gpu}_probes.jsonl"
        summary = OUT / f"worker_gpu{gpu}_summary.json"
        assert probe.exists() and summary.exists(), f"missing GPU{gpu} worker output"
        rows = read_jsonl(probe)
        assert len(rows) == expected_rows, (gpu, len(rows), expected_rows)
        worker_summaries.append(json.loads(summary.read_text()))
        add_raw.extend(rows)
    add_by_id = {x["unit_id"]: x for x in add_raw}
    expected_ids = {f"L{li:02d}_{path}" for li in range(32) for path in ("QK", "V")}
    assert len(add_by_id) == 64 and set(add_by_id) == expected_ids
    assert all(x["status"] in ("MEASURED", "SATURATED_NO_RECEIVER_STEP") for x in add_raw)
    assert sum(x["status"] == "MEASURED" for x in add_raw) == 59
    assert sum(x["status"] == "SATURATED_NO_RECEIVER_STEP" for x in add_raw) == 5

    removal_rows = list(csv.DictReader((REMOVE / "LLAMA2_E068_NATIVE_FRAGILITY.csv").open(newline="")))
    removal = {}
    for row in removal_rows:
        li = int(row["layer"])
        for p, pre in (("QK", "qk"), ("V", "v")):
            removal[f"L{li:02d}_{p}"] = {
                "delta": float(row[f"{pre}_delta_loss"]) if row[f"{pre}_delta_loss"] not in ("", "None") else None,
                "half_a": float(row[f"{pre}_delta_loss_half_a"]) if row[f"{pre}_delta_loss_half_a"] not in ("", "None") else None,
                "half_b": float(row[f"{pre}_delta_loss_half_b"]) if row[f"{pre}_delta_loss_half_b"] not in ("", "None") else None,
                "bytes": int(row[f"{pre}_delta_bytes"] or 0),
            }
    q = json.loads(Q_JSON.read_text())["q"]
    rank_rows = list(csv.DictReader(RANK_CSV.open(newline="")))
    ranks = [[int(r["uniform068_QK_rank"]), int(r["uniform068_V_rank"])] for r in rank_rows]
    assert len(ranks) == 32
    assert sum((k + v) * BYTES_PER_RANK for k, v in ranks) == BASELINE_BYTES

    units = []
    for li in range(32):
        for path, col in (("QK", 0), ("V", 1)):
            uid = f"L{li:02d}_{path}"
            a = add_by_id[uid]
            rm = removal[uid]
            current_rank = ranks[li][col]
            assert a["current_qk_rank"] == ranks[li][0] and a["current_v_rank"] == ranks[li][1]
            assert abs(float(a["current_q"]) - float(q[li])) < 1e-12
            measured = a["status"] == "MEASURED"
            if measured:
                assert int(a["delta_bytes"]) == DELTA_BYTES
                add_d = float(a["delta_loss"])
                add_a, add_b = float(a["delta_loss_half_a"]), float(a["delta_loss_half_b"])
                add_benefit = -add_d / DELTA_BYTES
                add_benefit_a, add_benefit_b = -add_a / DELTA_BYTES, -add_b / DELTA_BYTES
            else:
                add_d = add_a = add_b = add_benefit = add_benefit_a = add_benefit_b = None
            donor_cost = rm["delta"] / rm["bytes"] if rm["delta"] is not None and rm["bytes"] else None
            donor_a = rm["half_a"] / rm["bytes"] if rm["half_a"] is not None and rm["bytes"] else None
            donor_b = rm["half_b"] / rm["bytes"] if rm["half_b"] is not None and rm["bytes"] else None
            donor_eligible = (
                current_rank > 64 and donor_cost is not None and
                rm["delta"] <= 0 and rm["half_a"] <= 0 and rm["half_b"] <= 0
            )
            receiver_eligible = (
                current_rank < 4096 and measured and add_d < 0 and add_a < 0 and add_b < 0
            )
            units.append({
                "unit_id": uid, "layer": li, "path": path, "q_diagnostic_only": float(q[li]),
                "current_qk_rank": ranks[li][0], "current_v_rank": ranks[li][1],
                "current_path_rank": current_rank,
                "next_add_rank": current_rank + QUANTUM if current_rank < 4096 else None,
                "add_delta_bytes": DELTA_BYTES if measured else 0,
                "add_delta_loss": add_d, "receiver_benefit_per_byte": add_benefit,
                "add_delta_loss_half_a": add_a, "add_delta_loss_half_b": add_b,
                "receiver_benefit_half_a": add_benefit_a, "receiver_benefit_half_b": add_benefit_b,
                "remove_delta_loss": rm["delta"], "donor_cost_per_byte": donor_cost,
                "remove_delta_loss_half_a": rm["half_a"], "remove_delta_loss_half_b": rm["half_b"],
                "donor_cost_half_a": donor_a, "donor_cost_half_b": donor_b,
                "status": a["status"],
                "donor_stable_eligible": donor_eligible,
                "receiver_stable_eligible": receiver_eligible,
            })

    per_unit_path = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT.csv"
    fields = list(units[0])
    with per_unit_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(units)

    donors = [u for u in units if u["donor_stable_eligible"]]
    receivers = [u for u in units if u["receiver_stable_eligible"]]
    all_pairs = []
    stable_pairs = []
    for d in donors:
        for r in receivers:
            if d["unit_id"] == r["unit_id"] or d["layer"] == r["layer"]:
                continue
            gain = (r["receiver_benefit_per_byte"] - d["donor_cost_per_byte"]) * DELTA_BYTES
            gain_a = (r["receiver_benefit_half_a"] - d["donor_cost_half_a"]) * DELTA_BYTES
            gain_b = (r["receiver_benefit_half_b"] - d["donor_cost_half_b"]) * DELTA_BYTES
            item = {
                "donor_unit": d["unit_id"], "receiver_unit": r["unit_id"],
                "donor_q_diagnostic": d["q_diagnostic_only"], "receiver_q_diagnostic": r["q_diagnostic_only"],
                "donor_remove_delta_loss": d["remove_delta_loss"], "receiver_add_delta_loss": r["add_delta_loss"],
                "predicted_exchange_delta_loss": d["remove_delta_loss"] + r["add_delta_loss"],
                "predicted_exchange_gain": gain, "predicted_gain_half_a": gain_a,
                "predicted_gain_half_b": gain_b, "delta_total_bytes": 0,
                "donor_rank_before": d["current_path_rank"], "donor_rank_after": d["current_path_rank"] - QUANTUM,
                "receiver_rank_before": r["current_path_rank"], "receiver_rank_after": r["current_path_rank"] + QUANTUM,
            }
            all_pairs.append(item)
            if gain > 0 and gain_a > 0 and gain_b > 0:
                stable_pairs.append(item)
    all_pairs.sort(key=lambda x: (-x["predicted_exchange_gain"], x["donor_unit"], x["receiver_unit"]))
    stable_pairs.sort(key=lambda x: (-x["predicted_exchange_gain"], x["donor_unit"], x["receiver_unit"]))
    for name, data in (("LLAMA2_E068_NATIVE_EXCHANGE_PAIRS_ALL.csv", all_pairs),
                       ("LLAMA2_E068_NATIVE_EXCHANGE_PAIRS_STABLE.csv", stable_pairs)):
        if data:
            with (OUT / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    if stable_pairs:
        best = stable_pairs[0]
        candidate = [p[:] for p in ranks]
        for item, delta in ((best["donor_unit"], -QUANTUM), (best["receiver_unit"], QUANTUM)):
            li, path = int(item[1:3]), item.split("_")[1]
            candidate[li][0 if path == "QK" else 1] += delta
        assert sum((k + v) * BYTES_PER_RANK for k, v in candidate) == BASELINE_BYTES
        candidate_sha = sha_json(candidate)
        with (OUT / "LLAMA2_E068_NATIVE_FIRST_EXCHANGE_CANDIDATE.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "uniform068_qk_rank", "uniform068_v_rank", "candidate_qk_rank", "candidate_v_rank", "delta_qk_rank", "delta_v_rank"])
            for i, (old, new) in enumerate(zip(ranks, candidate)):
                w.writerow([i, old[0], old[1], new[0], new[1], new[0]-old[0], new[1]-old[1]])
    else:
        best, candidate, candidate_sha = None, ranks, sha_json(ranks)

    refs = [x["reference_loss"] for x in worker_summaries]
    ref_spread = {}
    for half in ("A", "B"):
        vals = [float(x[half]["mean_nll"]) for x in refs]
        ref_spread[half] = max(vals) - min(vals)
    summary = {
        "artifact_id": "LLAMA2_E068_NATIVE_ADD_BENEFIT_ANALYSIS_V1",
        "status": "ADD_ONE_PROBES_COMPLETE_GEOMETRY_READY_FOR_JOINT_CANARY" if best else "NO_STABLE_EXCHANGE_CANDIDATE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_lock_sha256": EXPECTED["add_protocol_lock_sha256"],
        "stability_rule_path": str(rule_path), "stability_rule_sha256": sha(rule_path),
        "baseline": {"B_live_bytes": BASELINE_BYTES, "rank_schedule_sha256": EXPECTED["rank_schedule_sha256"],
                     "basis_sha256": EXPECTED["basis_sha256"], "q_sha256": EXPECTED["profile_q_sha256"]},
        "add_probe_counts": {"measured": 59, "saturated_no_receiver_step": 5},
        "stable_donor_count": len(donors), "stable_receiver_count": len(receivers),
        "legal_pair_count_stable": len(stable_pairs), "legal_pair_count_all_stable_eligible_units": len(all_pairs),
        "top10_receiver_benefit": sorted((u for u in units if u["receiver_benefit_per_byte"] is not None),
                                           key=lambda x: (-x["receiver_benefit_per_byte"], x["unit_id"]))[:10],
        "top10_safest_donor_cost": sorted((u for u in units if u["donor_cost_per_byte"] is not None),
                                          key=lambda x: (x["donor_cost_per_byte"], x["unit_id"]))[:10],
        "top20_stable_exchange_pairs": stable_pairs[:20],
        "first_exchange": best,
        "first_exchange_candidate_rank_sha256": candidate_sha,
        "first_exchange_candidate_B_live_bytes": BASELINE_BYTES if best else None,
        "probe_reference_split_spread_by_worker": ref_spread,
        "add_csv_path": str(per_unit_path), "add_csv_sha256": sha(per_unit_path),
        "final_quality_started": False,
        "q_usage": "diagnostic column only; not used for donor/receiver selection",
    }
    summary_path = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
