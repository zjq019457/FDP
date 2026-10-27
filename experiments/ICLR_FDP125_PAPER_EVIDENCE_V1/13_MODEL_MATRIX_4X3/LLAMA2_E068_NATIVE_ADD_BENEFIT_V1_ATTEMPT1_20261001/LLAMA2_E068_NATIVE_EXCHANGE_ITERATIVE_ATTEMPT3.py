#!/usr/bin/env python3
"""Greedy resource-neutral Llama-2 E068 native-unit exchange tuning."""
import argparse
import builtins
import copy
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

BASE = pathlib.Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/13_MODEL_MATRIX_4X3")
OUT = BASE / "LLAMA2_E068_NATIVE_ADD_BENEFIT_V1_ATTEMPT1_20261001"
ADD_RUNNER = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_RUNNER_V1.py"
ADD_LOCK = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_PROTOCOL_LOCK.json"
ADD_SUMMARY = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_SUMMARY.json"
ADD_CSV = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT.csv"
RULE = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_STABILITY_RULE_ADDENDUM.json"
ITER_LOCK = OUT / "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT3_PROTOCOL_LOCK.json"
ITER_STATUS = OUT / "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT3_STATUS.json"
ITER_LOG = OUT / "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT3_ATTEMPTS.jsonl"
FINAL_JSON = OUT / "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT3_FROZEN_SCHEDULE.json"
FINAL_CSV = OUT / "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT3_FROZEN_SCHEDULE.csv"
INVALIDATION = OUT / "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT2_ENGINEERING_INVALIDATION.json"
FAILED_STATUS = OUT / "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT2_STATUS.json"
FAILED_LOG = OUT / "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT2.log"
FAILED_EXECUTOR = OUT / "LLAMA2_E068_NATIVE_EXCHANGE_ITERATIVE_ATTEMPT2.py"
QUANTUM = 32
BYTES_PER_RANK = 4096
DELTA_BYTES = QUANTUM * BYTES_PER_RANK
RANK_MIN = 64
RANK_MAX = 4096
GPU_GATE_MIB = 36864
GPU_PHYSICAL_ID = int(os.environ.get("FDP_GPU_PHYSICAL_ID", "0"))
MAX_ACCEPTED = 12
EXPECTED_ADD_LOCK_SHA = "b85c02610e1f6a1e19df1ca069d5d5bb1c837583938e725f3f0429435f71bb5e"
EXPECTED_BASIS_SHA = "196863f8ac17c17b276a8dc74a9dee9e30ee8e540f960c0638dccf86095f01d0"
EXPECTED_SCHEDULE_SHA = "590129c5a0401474266d7652601461b52f7be452ced772dd489b25f3a744913c"
EXPECTED_RANK_PAIRS_SHA = "c566c93a731e1ba3241513d6544ff6eedf007aa989e81d01a1fcceb82d686c6c"
SCRIPT_PATH = pathlib.Path(__file__).resolve()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def gpu_snapshot():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"
    ], text=True)
    rows = []
    for line in raw.strip().splitlines():
        idx, total, used, free, util = [int(x.strip()) for x in line.split(",")]
        rows.append({"index": idx, "total_mib": total, "used_mib": used, "free_mib": free, "util_pct": util})
    return rows


def selected_gpu_snapshot():
    rows = gpu_snapshot()
    match = [r for r in rows if r["index"] == GPU_PHYSICAL_ID]
    if len(match) != 1:
        raise RuntimeError(f"selected physical GPU {GPU_PHYSICAL_ID} is not visible: {rows}")
    return match[0]


def load_mod(path):
    spec = importlib.util.spec_from_file_location("llama2_add_runner", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preflight():
    if ITER_LOCK.exists() or ITER_STATUS.exists():
        raise FileExistsError("iteration lock/status already exists; refusing overwrite")
    if INVALIDATION.exists():
        raise FileExistsError("attempt-1 invalidation record already exists; refusing overwrite")
    prior_status = json.loads(FAILED_STATUS.read_text())
    prior_log = FAILED_LOG.read_text(errors="replace")
    assert "AssertionError" in prior_log and prior_status.get("status") == "RUNNING_JOINT_TUNING_EXCHANGE"
    invalidation = {
        "artifact_id": "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT2_ENGINEERING_INVALIDATION_V1",
        "status": "INVALID_ENGINEERING_PRE_SCORE",
        "failed_stage": "baseline rank-schedule hash binding in setup_worker",
        "failure_class": "HASH_DOMAIN_MISMATCH",
        "diagnosis": "attempt 2 added exact frozen rank-pair value and pair-list digest checks but accidentally retained the original incompatible hash-domain assertion; attempt 3 removes only that redundant assertion",
        "final_quality_started": False,
        "prior_status_sha256": sha_file(FAILED_STATUS),
        "prior_log_sha256": sha_file(FAILED_LOG),
        "prior_executor_sha256": sha_file(FAILED_EXECUTOR),
        "created_utc": now(),
    }
    assert sha_file(ADD_LOCK) == EXPECTED_ADD_LOCK_SHA
    add_lock = json.loads(ADD_LOCK.read_text())
    add_summary = json.loads(ADD_SUMMARY.read_text())
    stability = json.loads(RULE.read_text())
    assert add_lock["reference"]["basis_sha256"] == EXPECTED_BASIS_SHA
    assert add_lock["reference"]["rank_schedule_sha256"] == EXPECTED_SCHEDULE_SHA
    assert add_summary["status"] == "ADD_ONE_PROBES_COMPLETE_GEOMETRY_READY_FOR_JOINT_CANARY"
    assert add_summary["final_quality_started"] is False
    assert stability["status"] == "FROZEN_PRE_ANALYSIS"
    snap = gpu_snapshot()
    selected = selected_gpu_snapshot()
    if selected["free_mib"] < GPU_GATE_MIB:
        raise RuntimeError(f"selected GPU {GPU_PHYSICAL_ID} does not meet 36-GiB tuning gate: {selected}")
    atomic_json(INVALIDATION, invalidation)
    lock = {
        "artifact_id": "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT3_PROTOCOL_LOCK_V1",
        "status": "FROZEN_BEFORE_JOINT_EXCHANGE_EVALUATION",
        "created_utc": now(),
        "method": "resource-neutral one-rank-quantum path exchange using measured add/remove marginal NLL; profile q is diagnostic only",
        "acceptance": "accept a selected exchange only if joint 61-window full token-mean NLL decreases and both frozen halves are non-worsening; otherwise stop without accepting",
        "stability_rule_path": str(RULE), "stability_rule_sha256": sha_file(RULE),
        "add_protocol_lock_sha256": sha_file(ADD_LOCK),
        "add_summary_sha256": sha_file(ADD_SUMMARY),
        "add_csv_sha256": sha_file(ADD_CSV),
        "add_runner_sha256": sha_file(ADD_RUNNER),
        "basis_sha256": EXPECTED_BASIS_SHA,
        "baseline_rank_schedule_sha256": EXPECTED_SCHEDULE_SHA,
        "baseline_rank_pairs_sha256": EXPECTED_RANK_PAIRS_SHA,
        "executor_sha256": sha_file(SCRIPT_PATH),
        "prior_attempt_invalidation_sha256": sha_file(INVALIDATION),
        "baseline_B_live_bytes": 746323968,
        "rank_quantum": QUANTUM, "delta_bytes_per_quantum": DELTA_BYTES,
        "max_accepted_exchanges": MAX_ACCEPTED,
        "selected_physical_gpu": GPU_PHYSICAL_ID,
        "gpu_preflight": snap,
        "final_wiki_c4_or_accuracy_started": False,
    }
    atomic_json(ITER_LOCK, lock)
    atomic_json(ITER_STATUS, {"status": "FROZEN_PRELAUNCH", "iteration_lock_sha256": sha_file(ITER_LOCK), "updated_utc": now()})
    print(json.dumps({"status": "FROZEN_PRELAUNCH", "iteration_lock_sha256": sha_file(ITER_LOCK), "gpu": selected}, indent=2))


def loss_metrics(res):
    total_tokens = res["A"]["token_count"] + res["B"]["token_count"]
    mean = (res["A"]["nll_sum"] + res["B"]["nll_sum"]) / total_tokens
    return {"full": mean, "A": res["A"]["mean_nll"], "B": res["B"]["mean_nll"]}


def to_float(value):
    return None if value in (None, "", "None") else float(value)


def setup_worker():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    free_bytes = torch.cuda.mem_get_info(0)[0]
    if free_bytes < GPU_GATE_MIB * 1024 * 1024:
        raise RuntimeError(f"visible GPU0 free memory under gate: {free_bytes}")
    mod = load_mod(ADD_RUNNER)
    lock_sha = sha_file(ADD_LOCK)
    assert lock_sha == EXPECTED_ADD_LOCK_SHA
    profile, resource, tune_manifest, rank_rows, pairs, windows, halves, hashes = mod.load_sources()
    lock = json.loads(ADD_LOCK.read_text())
    id_to_half = {wid: half for half, ids in lock["tuning_split"]["half_ids"].items() for wid in ids}
    for w in windows:
        w["_half"] = id_to_half[w["window_id"]]
    assert len(windows) == 61 and set(id_to_half) == {w["window_id"] for w in windows}

    from transformers.cache_utils import Cache
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.llama import modeling_llama as hf_llama
    cfg = AutoConfig.from_pretrained(str(mod.MODEL), local_files_only=True)
    assert cfg.model_type == "llama" and cfg.rope_scaling is None
    HFLLamaRotaryEmbedding = hf_llama.LlamaRotaryEmbedding
    def legacy_rope_constructor(dim, max_position_embeddings, base):
        assert int(dim) == cfg.hidden_size // cfg.num_attention_heads
        assert int(max_position_embeddings) == int(cfg.max_position_embeddings)
        assert float(base) == float(cfg.rope_theta)
        cc = copy.deepcopy(cfg)
        cc.max_position_embeddings = int(max_position_embeddings)
        cc.rope_theta = float(base)
        return HFLLamaRotaryEmbedding(config=cc)
    builtins.Cache = Cache
    builtins.LlamaRotaryEmbedding = legacy_rope_constructor
    builtins.apply_rotary_pos_emb = hf_llama.apply_rotary_pos_emb
    builtins.repeat_kv = hf_llama.repeat_kv

    cm = mod.load_module("canonical_exchange_materializer", mod.CANONICAL)
    base_pairs = [[int(k), int(v)] for k, v in pairs]
    per_layer = [{"layer": i, "rank_kq": k, "rank_v": v} for i, (k, v) in enumerate(base_pairs)]
    census = {"official": {"per_layer": per_layer, "rank_pairs": base_pairs}}
    cm.common_preflight = lambda: ({}, census, {}, {}, {"identity": "llama2_7b"})
    cm.basis_complete = lambda: True
    cm.E_MIN_BYTES = 0
    cm.GPU_GATE_MIB = GPU_GATE_MIB
    cm.gpu_free_mib = lambda: int(torch.cuda.mem_get_info(0)[0] // (1024 * 1024))
    cm.disk_free = lambda: shutil.disk_usage("/diskE").free
    cm.MODEL = mod.MODEL
    cm.BASIS_FILE = mod.BASIS
    basis_meta = {"basis_sha256": EXPECTED_BASIS_SHA, "basis_bytes": mod.BASIS.stat().st_size}
    model, factor_calc = cm.build_model("official", basis_meta)
    runtime_pairs = [[int(x["rank_kq"]), int(x["rank_v"])] for x in factor_calc]
    assert runtime_pairs == base_pairs
    frozen_pairs = [[int(k), int(v)] for k, v in lock["reference"]["rank_pairs_qk_v"]]
    assert base_pairs == frozen_pairs, "runtime rank pairs differ from frozen reference values"
    assert mod.sha_json(base_pairs) == EXPECTED_RANK_PAIRS_SHA, "canonical rank-pair-list digest mismatch"
    device = torch.device("cuda:0")

    base_model = AutoModelForCausalLM.from_pretrained(
        str(mod.MODEL), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    base_model.eval()
    base_model.config.use_cache = False
    assert base_model.config.num_hidden_layers == 32 and base_model.config.num_attention_heads == 32
    raw_basis = torch.load(mod.BASIS, map_location="cpu", weights_only=False, mmap=True)
    from types import SimpleNamespace
    from models.decompose_modules import LlamaEigenAttnDecoderLayer
    args = SimpleNamespace()
    head_dim = 128

    def build_layer(layer_idx, k_rank, v_rank):
        base_layer = base_model.model.layers[layer_idx].to(device)
        prefix = f"base_model.model.model.layers.{layer_idx}.self_attn."
        basis_k = raw_basis[prefix + "k_proj_basis"][:, :k_rank // 32].to(device=device, dtype=torch.bfloat16)
        vp = raw_basis[prefix + "v_proj_basis"]
        vstack = torch.stack(
            [vp[h * head_dim:(h + 1) * head_dim, h * head_dim:(h + 1) * head_dim] for h in range(32)],
            dim=0,
        )[:, :, :v_rank // 32].to(device=device, dtype=torch.bfloat16)
        layer = LlamaEigenAttnDecoderLayer(base_layer, args, basis_k, k_rank, vstack, v_rank, model.config, layer_idx).to(device=device, dtype=torch.bfloat16)
        base_layer.to("cpu")
        del basis_k, vstack, vp, base_layer
        return layer

    baseline_probe = json.loads(ADD_SUMMARY.read_text())["probe_reference_split_spread_by_worker"]
    assert baseline_probe["A"] == 0 and baseline_probe["B"] == 0
    reference = mod.score_windows(model, windows, device, torch)
    refs = [x["reference_loss"] for x in [
        json.loads((OUT / f"worker_gpu{g}_summary.json").read_text()) for g in range(3)
    ]]
    for half in ("A", "B"):
        old_vals = [r[half]["mean_nll"] for r in refs]
        assert max(abs(reference[half]["mean_nll"] - x) for x in old_vals) <= 1e-7, (half, reference[half], old_vals)
    return mod, torch, model, base_model, raw_basis, build_layer, windows, base_pairs, reference, device


def initial_cache():
    cache = {}
    with ADD_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            uid = r["unit_id"]
            cache[uid] = {
                "add": {
                    "delta": to_float(r["add_delta_loss"]),
                    "A": to_float(r["add_delta_loss_half_a"]),
                    "B": to_float(r["add_delta_loss_half_b"]),
                },
                "remove": {
                    "delta": to_float(r["remove_delta_loss"]),
                    "A": to_float(r["remove_delta_loss_half_a"]),
                    "B": to_float(r["remove_delta_loss_half_b"]),
                },
            }
    return cache


def unit_layer_path(uid):
    return int(uid[1:3]), uid.split("_")[1]


def update_rank(pairs, uid, delta):
    li, path = unit_layer_path(uid)
    col = 0 if path == "QK" else 1
    pairs[li][col] += delta


def rank_of(pairs, uid):
    li, path = unit_layer_path(uid)
    return pairs[li][0 if path == "QK" else 1]


def probe_one(mod, torch, model, build_layer, windows, pairs, reference, device, uid, direction):
    li, path = unit_layer_path(uid)
    rank = rank_of(pairs, uid)
    new_rank = rank + direction * QUANTUM
    if not (RANK_MIN <= new_rank <= RANK_MAX):
        return {"delta": None, "A": None, "B": None}
    candidate = [x[:] for x in pairs]
    update_rank(candidate, uid, direction * QUANTUM)
    old_layer = model.model.layers[li]
    replacement = build_layer(li, candidate[li][0], candidate[li][1])
    model.model.layers[li] = replacement
    try:
        measured = mod.score_windows(model, windows, device, torch)
    finally:
        model.model.layers[li] = old_layer
        del replacement
        torch.cuda.empty_cache()
    before, after = loss_metrics(reference), loss_metrics(measured)
    return {h: after[h] - before[h] for h in ("full", "A", "B")} | {"delta": after["full"], "A": after["A"] - before["A"], "B": after["B"] - before["B"]}


def choose_pair(pairs, cache, q):
    donors, receivers = [], []
    for li in range(32):
        for path in ("QK", "V"):
            uid = f"L{li:02d}_{path}"
            rank = rank_of(pairs, uid)
            entry = cache[uid]
            rm, ad = entry["remove"], entry["add"]
            if rank > RANK_MIN and rm["delta"] is not None and rm["delta"] <= 0 and rm["A"] <= 0 and rm["B"] <= 0:
                donors.append(uid)
            if rank < RANK_MAX and ad["delta"] is not None and ad["delta"] < 0 and ad["A"] < 0 and ad["B"] < 0:
                receivers.append(uid)
    choices = []
    for d in donors:
        for r in receivers:
            dl, _ = unit_layer_path(d)
            rl, _ = unit_layer_path(r)
            if d == r or dl == rl:
                continue
            dm, ra = cache[d]["remove"], cache[r]["add"]
            gain = -(dm["delta"] + ra["delta"])
            gain_a = -(dm["A"] + ra["A"])
            gain_b = -(dm["B"] + ra["B"])
            choices.append({
                "donor_unit": d, "receiver_unit": r,
                "donor_q_diagnostic_only": float(q[dl]), "receiver_q_diagnostic_only": float(q[rl]),
                "predicted_exchange_gain": gain, "predicted_gain_half_a": gain_a, "predicted_gain_half_b": gain_b,
                "predicted_delta_loss": -gain,
                "donor_remove_delta_loss": dm["delta"], "receiver_add_delta_loss": ra["delta"],
                "donor_rank_before": rank_of(pairs, d), "donor_rank_after": rank_of(pairs, d) - QUANTUM,
                "receiver_rank_before": rank_of(pairs, r), "receiver_rank_after": rank_of(pairs, r) + QUANTUM,
                "delta_total_bytes": 0,
            })
    choices = [x for x in choices if x["predicted_exchange_gain"] > 0 and x["predicted_gain_half_a"] > 0 and x["predicted_gain_half_b"] > 0]
    choices.sort(key=lambda x: (-x["predicted_exchange_gain"], x["donor_unit"], x["receiver_unit"]))
    return choices


def launch():
    if not ITER_LOCK.exists() or ITER_STATUS.exists() is False:
        raise RuntimeError("run --preflight first")
    lock_sha = sha_file(ITER_LOCK)
    lock = json.loads(ITER_LOCK.read_text())
    assert lock["status"] == "FROZEN_BEFORE_JOINT_EXCHANGE_EVALUATION"
    assert sha_file(SCRIPT_PATH) == lock["executor_sha256"]
    assert sha_file(INVALIDATION) == lock["prior_attempt_invalidation_sha256"]
    assert sha_file(ADD_SUMMARY) == lock["add_summary_sha256"]
    assert sha_file(ADD_CSV) == lock["add_csv_sha256"]
    if selected_gpu_snapshot()["free_mib"] < GPU_GATE_MIB:
        raise RuntimeError(f"selected GPU {GPU_PHYSICAL_ID} no longer meets 36-GiB gate")
    atomic_json(ITER_STATUS, {"status": "RUNNING_JOINT_TUNING_EXCHANGE", "iteration_lock_sha256": lock_sha, "started_utc": now(), "accepted_count": 0})

    mod, torch, model, base_model, raw_basis, build_layer, windows, pairs, reference, device = setup_worker()
    profile, *_ = mod.load_sources()
    q = [float(x) for x in profile["q"]]
    cache = initial_cache()
    baseline = loss_metrics(reference)
    accepted, attempts = [], []
    stop_reason = None

    for iteration in range(1, MAX_ACCEPTED + 1):
        choices = choose_pair(pairs, cache, q)
        if not choices:
            stop_reason = "NO_POSITIVE_SPLIT_STABLE_PREDICTED_EXCHANGE"
            break
        best = choices[0]
        candidate = [x[:] for x in pairs]
        update_rank(candidate, best["donor_unit"], -QUANTUM)
        update_rank(candidate, best["receiver_unit"], QUANTUM)
        assert sum((k + v) * BYTES_PER_RANK for k, v in candidate) == 746323968
        dl, _ = unit_layer_path(best["donor_unit"])
        rl, _ = unit_layer_path(best["receiver_unit"])
        old_layers = {dl: model.model.layers[dl], rl: model.model.layers[rl]}
        model.model.layers[dl] = build_layer(dl, candidate[dl][0], candidate[dl][1])
        model.model.layers[rl] = build_layer(rl, candidate[rl][0], candidate[rl][1])
        measured = mod.score_windows(model, windows, device, torch)
        actual = loss_metrics(measured)
        before = loss_metrics(reference)
        delta = {h: actual[h] - before[h] for h in ("full", "A", "B")}
        accepted_now = delta["full"] < 0 and delta["A"] <= 0 and delta["B"] <= 0
        record = {
            "iteration": iteration, "candidate": best, "actual_delta_loss": delta,
            "accepted": accepted_now, "candidate_rank_pairs_sha256": mod.sha_json(candidate),
            "candidate_B_live_bytes": 746323968,
            "comparison": "same 61-window pooled token-mean NLL and frozen A/B split",
            "created_utc": now(),
        }
        attempts.append(record)
        with ITER_LOG.open("a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if not accepted_now:
            model.model.layers[dl] = old_layers[dl]
            model.model.layers[rl] = old_layers[rl]
            del measured
            torch.cuda.empty_cache()
            stop_reason = "JOINT_EXCHANGE_FAILED_FULL_OR_SPLIT_ACCEPTANCE"
            break

        accepted.append(record)
        pairs = candidate
        reference = measured
        for uid in (best["donor_unit"], best["receiver_unit"]):
            cache[uid]["remove"] = probe_one(mod, torch, model, build_layer, windows, pairs, reference, device, uid, -1)
            cache[uid]["add"] = probe_one(mod, torch, model, build_layer, windows, pairs, reference, device, uid, +1)
        atomic_json(ITER_STATUS, {
            "status": "RUNNING_JOINT_TUNING_EXCHANGE", "iteration_lock_sha256": lock_sha,
            "accepted_count": len(accepted), "latest_accepted": record, "updated_utc": now(),
        })
    if stop_reason is None:
        stop_reason = "MAXIMUM_ACCEPTED_EXCHANGES_REACHED"

    final_loss = loss_metrics(reference)
    rank_sha = mod.sha_json(pairs)
    assert sum((k + v) * BYTES_PER_RANK for k, v in pairs) == 746323968
    with FINAL_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "baseline_qk_rank", "baseline_v_rank", "final_qk_rank", "final_v_rank", "delta_qk_rank", "delta_v_rank", "q_diagnostic_only"])
        baseline_pairs = json.loads(ADD_LOCK.read_text())["reference"]["rank_pairs_qk_v"]
        for li, (base_pair, final_pair) in enumerate(zip(baseline_pairs, pairs)):
            w.writerow([li, base_pair[0], base_pair[1], final_pair[0], final_pair[1], final_pair[0]-base_pair[0], final_pair[1]-base_pair[1], q[li]])
    summary = {
        "artifact_id": "LLAMA2_E068_NATIVE_EXCHANGE_ITERATION_ATTEMPT3_FROZEN_SCHEDULE_V1",
        "status": "FROZEN_AFTER_TUNING" if accepted else "NO_EXCHANGE_ACCEPTED_BASELINE_RETAINED",
        "created_utc": now(), "iteration_lock_sha256": lock_sha,
        "baseline": {"B_live_bytes": 746323968, "rank_schedule_sha256": EXPECTED_SCHEDULE_SHA, "tuning_nll": baseline},
        "final": {"B_live_bytes": 746323968, "rank_schedule": pairs, "rank_pairs_sha256": rank_sha, "tuning_nll": final_loss},
        "tuning_nll_improvement_full": baseline["full"] - final_loss["full"],
        "accepted_exchange_count": len(accepted), "attempted_exchange_count": len(attempts),
        "stop_reason": stop_reason, "accepted_exchanges": accepted, "attempted_exchanges": attempts,
        "final_csv_path": str(FINAL_CSV), "final_csv_sha256": sha_file(FINAL_CSV),
        "q_usage": "diagnostic only; not used to choose exchanges",
        "basis_rebuilt": False, "final_wiki_c4_or_accuracy_started": False,
    }
    atomic_json(FINAL_JSON, summary)
    atomic_json(ITER_STATUS, {"status": summary["status"], "iteration_lock_sha256": lock_sha, "summary_sha256": sha_file(FINAL_JSON), "accepted_count": len(accepted), "stop_reason": stop_reason, "updated_utc": now()})
    print(json.dumps({"status": summary["status"], "accepted": len(accepted), "attempted": len(attempts),
                      "stop_reason": stop_reason, "baseline_nll": baseline, "final_nll": final_loss,
                      "improvement": baseline["full"]-final_loss["full"], "B_live": 746323968,
                      "final_rank_pairs_sha256": rank_sha, "summary_path": str(FINAL_JSON)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()
    if args.preflight:
        preflight()
    elif args.launch:
        launch()
    else:
        ap.error("choose --preflight or --launch")


if __name__ == "__main__":
    main()
