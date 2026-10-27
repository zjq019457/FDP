#!/usr/bin/env python3
"""Measure one legal rank-quantum addition at each Llama-2 E068 path."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE = pathlib.Path("/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/13_MODEL_MATRIX_4X3")
PROFILE = BASE / "LLAMA2_CLEAN_REDESIGN_UNIFORM068_PROFILE_V1_20261001T1516_ATTEMPT7/LLAMA2_CLEAN_REDESIGN_D1_D2_D5_PROFILE_V1.json"
PROFILE_LOCK = PROFILE.parent / "PROFILE_PROTOCOL_LOCK.json"
RANK_CSV = BASE / "LLAMA2_OPERATING_POINT_E068_FDP070_CAPPED_V1/LLAMA2_E068_UNIFORM_FDP070_CAPPED_PER_LAYER.csv"
RESOURCE = BASE / "LLAMA2_OPERATING_POINT_E068_FDP070_CAPPED_V1/LLAMA2_E068_UNIFORM_FDP070_CAPPED_RESOURCE_ONLY_MANIFEST_V2.json"
TUNE_MANIFEST = BASE / "LLAMA2_ALLOCATOR_TUNING_STUDY_V1/LLAMA2_FDP_STAGE3_HELDOUT_TUNING_SPLIT_V2_UNIQUE_IDS.json"
TUNE_WINDOWS = pathlib.Path("/diskF/jiaqizhang/fdp125_tuning_stage2_llama2_20261001/LLAMA2_FDP_STAGE3_TUNING_WINDOWS_V2_UNIQUE_IDS.jsonl")
BASIS = pathlib.Path("/diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/llama2_e060_full_capacity_basis_only_v1/llama2_e060_full_basis_capacity_only_eb0p06.pt")
MODEL = pathlib.Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf")
CANONICAL = BASE / "cells/llama2_7b/medium/canonical_materialization_canary_v4_20260925T094600Z/l2_medium_canonical_materialize_canary_v4.py"
OLD_FDP060 = BASE / "LLAMA2_ALLOCATOR_TUNING_STUDY_V1/LLAMA2_FDP060_ORIGINAL_3COMP_PROFILE_AUDIT.csv"
AUTH = BASE / "cells/llama2_7b/medium/provenance_binding_v1_20260925T061200Z/LLAMA2_MEDIUM_CELL_LAUNCH_AUTHORIZATION_V1.json"
PROVENANCE_AUDIT = BASE / "cells/llama2_7b/medium/provenance_binding_v1_20260925T061200Z/LLAMA2_MEDIUM_FINAL_PROVENANCE_HASH_AUDIT_V3.json"
EXPECTED = {
    "profile_sha256": "a06651aa00aaa4a1f2584aed506d3c5e358a50c6a8bec5a10531449c971e8fc3",
    "profile_q_sha256": "720bbbe44103c95e1a2aa04ec4f0ad5cb85f343bab27ca98b7bc81f5ea066cae",
    "profile_lock_sha256": "6a590bbb1f983a84598745227238973af668ff1b3349725bad61811b5a2fbe0b",
    "rank_csv_sha256": "9a5289ac6f2bf386ccc5cdb1d7ece76d7a965679c39088e79a67150d732b8487",
    "resource_manifest_sha256": "7a9d3ec68b2aa590e79fb6f853e1e5155dc1785f3c392e9527f80b382a983cfd",
    "tuning_manifest_sha256": "53532566c8fc635e66a556c51db96ef423ce321adf50b18a8dc8ee4ad2e72194",
    "tuning_manifest_declared_model_fingerprint": "a627fb4d2200089590b109ad82c35098778aa5856660cc691f6626d25c2cf25197",
    "tuning_windows_sha256": "ce6e67bccf7f142a6d24da2185a7e7a22f9ed9371902dbfa567e23c08005a115",
    "basis_sha256": "196863f8ac17c17b276a8dc74a9dee9e30ee8e540f960c0638dccf86095f01d0",
    "model_fingerprint_sha256": "a627fb4d2200089590b109ad82c35098778aa585660c691f6626d25c2cf25197",
    "tokenizer_fingerprint_sha256": "74d06083412d9c11688da0dd9e8e22d7d794028a34c61e00884901668a8ad300",
    "authorization_sha256": "5bd25896cb172564fba5b684483809c0db39b604185bf36645d5b548471c0d6b",
    "provenance_audit_sha256": "e67de18706ce1e1cfd68aaa7892f95712fea54f711eb733c4b1664c705b525a7",
    "baseline_B_live": 746323968,
    "rank_schedule_sha256": "590129c5a0401474266d7652601461b52f7be452ced772dd489b25f3a744913c",
}
TOKENIZER_FILE_HASHES = {
    "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
    "tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
    "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    "tokenizer_config.json": "f514e7c3008881b6ba7e6a0cdb44c71ce47dc335920dac143ae7bc788197e53a",
}
RANK_QUANTUM = 32
RANK_MIN = 64
RANK_MAX = 4096
BYTES_PER_RANK_UNIT = 4096
GPU_GATE_MIB = 36864
WORKER_COUNT = 3
OUT = BASE / "LLAMA2_E068_NATIVE_ADD_BENEFIT_V1_ATTEMPT1_20261001"
PREVIOUS_ATTEMPT = BASE / "LLAMA2_E068_NATIVE_FRAGILITY_V1_ATTEMPT7_a06651aa"
RUNNER_PATH = pathlib.Path(__file__).resolve()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def int64_ids_sha(ids):
    raw = struct.pack("<" + "q" * len(ids), *(int(x) for x in ids))
    return hashlib.sha256(raw).hexdigest()


def gpu_snapshot():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"
    ], text=True)
    result = []
    for line in raw.strip().splitlines():
        i, total, used, free, util = [int(x.strip()) for x in line.split(",")]
        result.append({"index": i, "total_mib": total, "used_mib": used, "free_mib": free, "utilization_pct": util})
    return result


def load_sources():
    for p in (PROFILE, PROFILE_LOCK, RANK_CSV, RESOURCE, TUNE_MANIFEST, TUNE_WINDOWS, BASIS, MODEL, CANONICAL, OLD_FDP060, AUTH, PROVENANCE_AUDIT):
        if not p.exists():
            raise FileNotFoundError(str(p))
    expected_paths = {
        PROFILE: EXPECTED["profile_sha256"],
        PROFILE_LOCK: EXPECTED["profile_lock_sha256"],
        RANK_CSV: EXPECTED["rank_csv_sha256"],
        RESOURCE: EXPECTED["resource_manifest_sha256"],
        TUNE_MANIFEST: EXPECTED["tuning_manifest_sha256"],
        TUNE_WINDOWS: EXPECTED["tuning_windows_sha256"],
        BASIS: EXPECTED["basis_sha256"],
    }
    hashes = {str(p): sha_file(p) for p in expected_paths}
    for p, expected in expected_paths.items():
        if hashes[str(p)] != expected:
            raise RuntimeError(f"SHA mismatch: {p}: {hashes[str(p)]} != {expected}")

    if sha_file(AUTH) != EXPECTED["authorization_sha256"]:
        raise RuntimeError("Llama-2 authorization SHA mismatch")
    if sha_file(PROVENANCE_AUDIT) != EXPECTED["provenance_audit_sha256"]:
        raise RuntimeError("Llama-2 final provenance audit SHA mismatch")
    auth = json.loads(AUTH.read_text())
    provenance_audit = json.loads(PROVENANCE_AUDIT.read_text())
    if auth.get("model_content_fingerprint_sha256") != EXPECTED["model_fingerprint_sha256"]:
        raise RuntimeError("frozen model identity differs from cell authorization")
    if not auth.get("cell_launch_authorized") or not auth.get("provenance_binding_pass"):
        raise RuntimeError("frozen Llama-2 model authorization/provenance is not PASS")
    if provenance_audit.get("model", {}).get("content_fingerprint_sha256") != EXPECTED["model_fingerprint_sha256"]:
        raise RuntimeError("model fingerprint differs from final provenance audit")
    if provenance_audit.get("tokenizer", {}).get("fingerprint_sha256") != EXPECTED["tokenizer_fingerprint_sha256"]:
        raise RuntimeError("tokenizer fingerprint differs from final provenance audit")
    for name, expected in TOKENIZER_FILE_HASHES.items():
        if sha_file(MODEL / name) != expected:
            raise RuntimeError(f"tokenizer file SHA mismatch: {name}")

    profile = json.loads(PROFILE.read_text())
    resource = json.loads(RESOURCE.read_text())
    tune_manifest = json.loads(TUNE_MANIFEST.read_text())
    if profile.get("status") != "PROFILE_COMPLETE_NOT_RESOURCE_LOCK":
        raise RuntimeError(f"unexpected profile status {profile.get('status')}")
    if profile.get("q_sha256") != EXPECTED["profile_q_sha256"] or len(profile.get("q", [])) != 32:
        raise RuntimeError("profile q SHA or layer count mismatch")
    if profile.get("checkpoint_content_fingerprint_sha256") != EXPECTED["model_fingerprint_sha256"]:
        raise RuntimeError("profile checkpoint identity mismatch")
    if profile.get("sample_manifest_file_sha256") != "82cf192e5cd24aca50093b2d11a33ae352ca4cbe50b13eb84887d455c37e6293":
        raise RuntimeError("profile sample manifest binding mismatch")
    if profile.get("compressed_comparator", {}).get("B_live_rank_implied_bytes") != EXPECTED["baseline_B_live"]:
        raise RuntimeError("Uniform068 B_live mismatch")
    if profile.get("compressed_comparator", {}).get("rank_schedule_sha256") != EXPECTED["rank_schedule_sha256"]:
        raise RuntimeError("Uniform068 rank SHA mismatch")
    if resource.get("schedule_sha256", {}).get("uniform068_rank_schedule") != EXPECTED["rank_schedule_sha256"]:
        raise RuntimeError("resource manifest rank SHA mismatch")
    if tune_manifest.get("status") != "FROZEN_MECHANICAL_DEDUP_ERRATUM_PRE_RESCORING":
        raise RuntimeError("held-out tuning split is not the deduplicated frozen version")
    # The frozen tuning manifest contains a 66-character malformed model fingerprint.
    # Do not edit that historical file; bind a metadata-only erratum to the already
    # frozen profile + formal authorization + provenance audit before any scoring.
    if tune_manifest.get("model_content_fingerprint_sha256") != EXPECTED["tuning_manifest_declared_model_fingerprint"]:
        raise RuntimeError("unexpected held-out tuning manifest model-identity field; protocol audit required")
    if len(tune_manifest["model_content_fingerprint_sha256"]) != 66:
        raise RuntimeError("the known tuning manifest fingerprint discrepancy changed")
    if tune_manifest.get("tokenizer_fingerprint_sha256") != EXPECTED["tokenizer_fingerprint_sha256"]:
        raise RuntimeError("tuning tokenizer identity mismatch")

    with RANK_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 32:
        raise RuntimeError(f"expected 32 rank rows, got {len(rows)}")
    pairs = [[int(r["uniform068_QK_rank"]), int(r["uniform068_V_rank"])] for r in rows]
    if pairs != profile["compressed_comparator"]["rank_schedule"]:
        raise RuntimeError("rank CSV and frozen profile comparator schedule differ")
    byte_count = sum((k + v) * BYTES_PER_RANK_UNIT for k, v in pairs)
    if byte_count != EXPECTED["baseline_B_live"]:
        raise RuntimeError(f"rank-implied B_live mismatch {byte_count}")
    for li, (rk, rv) in enumerate(pairs):
        if any(x < RANK_MIN or x > RANK_MAX or x % RANK_QUANTUM for x in (rk, rv)):
            raise RuntimeError(f"illegal baseline rank at layer {li}: {rk}/{rv}")

    windows = []
    with TUNE_WINDOWS.open() as f:
        for line in f:
            if line.strip():
                w = json.loads(line)
                if len(w.get("input_ids", [])) != 2048 or w.get("token_count") != 2048:
                    raise RuntimeError(f"bad tuning sequence length in {w.get('window_id')}")
                if int64_ids_sha(w["input_ids"]) != w.get("input_ids_sha256_int64le"):
                    raise RuntimeError(f"token ID hash mismatch in {w.get('window_id')}")
                windows.append(w)
    counts = {ds: sum(w["dataset"] == ds for w in windows) for ds in ("wikitext2", "c4")}
    if len(windows) != 61 or counts != {"wikitext2": 32, "c4": 29}:
        raise RuntimeError(f"held-out window count mismatch: {len(windows)} {counts}")
    manifest_ids = {
        "wikitext2": set(tune_manifest["wikitext2_window_ids"]),
        "c4": set(tune_manifest["c4_window_ids"]),
    }
    for ds in manifest_ids:
        got = [w["window_id"] for w in windows if w["dataset"] == ds]
        if len(got) != len(set(got)) or set(got) != manifest_ids[ds]:
            raise RuntimeError(f"held-out IDs do not match frozen manifest for {ds}")

    # Fixed, dataset-stratified alternating split; IDs and assignment are frozen before scoring.
    halves = {"A": [], "B": []}
    for ds in ("wikitext2", "c4"):
        subset = [w for w in windows if w["dataset"] == ds]
        for i, w in enumerate(subset):
            halves["A" if i % 2 == 0 else "B"].append(w)
    return profile, resource, tune_manifest, rows, pairs, windows, halves, hashes


def freeze_protocol():
    lock_path = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_PROTOCOL_LOCK.json"
    status_path = OUT / "RUN_STATUS.json"
    if lock_path.exists() or status_path.exists():
        raise FileExistsError(f"refusing to overwrite existing protocol/status in {OUT}")
    profile, resource, tune_manifest, rows, pairs, windows, halves, hashes = load_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    assignment = {h: [w["window_id"] for w in ws] for h, ws in halves.items()}
    identity_addendum = {
        "artifact_id": "LLAMA2_E068_ADD_BENEFIT_TUNING_MODEL_IDENTITY_METADATA_ERRATUM_V1",
        "status": "FROZEN_PRE_SCORING_METADATA_ONLY_BINDING",
        "created_utc": utc_now(),
        "source_manifest_path": str(TUNE_MANIFEST),
        "source_manifest_sha256": hashes[str(TUNE_MANIFEST)],
        "source_declared_model_fingerprint": tune_manifest["model_content_fingerprint_sha256"],
        "source_declared_fingerprint_length": len(tune_manifest["model_content_fingerprint_sha256"]),
        "canonical_model_fingerprint": EXPECTED["model_fingerprint_sha256"],
        "independent_identity_authorities": {
            "uniform068_profile_path": str(PROFILE),
            "uniform068_profile_sha256": hashes[str(PROFILE)],
            "cell_authorization_path": str(AUTH),
            "cell_authorization_sha256": EXPECTED["authorization_sha256"],
            "final_provenance_audit_path": str(PROVENANCE_AUDIT),
            "final_provenance_audit_sha256": EXPECTED["provenance_audit_sha256"],
        },
        "decision": "Bind this tuning split to the 64-character model identity already fixed by the profile and formal cell authorization; preserve the malformed 66-character field and source manifest unchanged.",
        "unchanged": ["window_data_bytes", "window_data_sha256", "all_unique_window_ids", "token_id_hashes", "split_assignment", "selection_rule", "tokenizer", "scorer", "loss definition"],
        "scoring_started": False,
    }
    addendum_path = OUT / "LLAMA2_E068_ADD_BENEFIT_TUNING_MODEL_IDENTITY_METADATA_ERRATUM.json"
    atomic_json(addendum_path, identity_addendum)
    lock = {
        "artifact_id": "LLAMA2_E068_NATIVE_ADD_BENEFIT_PROTOCOL_LOCK_V1",
        "status": "FROZEN_BEFORE_SCORING",
        "created_utc": utc_now(),
        "model_repo_id": "meta-llama/Llama-2-7b-hf",
        "model_fingerprint_sha256": EXPECTED["model_fingerprint_sha256"],
        "authorization_path": str(AUTH),
        "authorization_sha256": hashes.get(str(AUTH), sha_file(AUTH)),
        "provenance_audit_path": str(PROVENANCE_AUDIT),
        "provenance_audit_sha256": sha_file(PROVENANCE_AUDIT),
        "tokenizer_fingerprint_sha256": EXPECTED["tokenizer_fingerprint_sha256"],
        "reference": {
            "name": "Uniform EB=.068",
            "q_sha256": EXPECTED["profile_q_sha256"],
            "profile_sha256": EXPECTED["profile_sha256"],
            "basis_sha256": EXPECTED["basis_sha256"],
            "rank_schedule_sha256": EXPECTED["rank_schedule_sha256"],
            "rank_pairs_qk_v": pairs,
            "B_live_bytes": EXPECTED["baseline_B_live"],
            "accounting": "sum_l((QK_rank_l+V_rank_l)*4096 bytes)",
        },
        "legal_perturbation": {
            "rank_quantum": RANK_QUANTUM,
            "interpretation": "one total-rank increment of 32, i.e. one rank per each of 32 heads",
            "minimum_rank": RANK_MIN,
            "maximum_rank": RANK_MAX,
            "delta_B_live_bytes_per_unsaturated_path": RANK_QUANTUM * BYTES_PER_RANK_UNIT,
            "paths": ["QK", "V"],
            "all_other_paths_and_layers_unchanged": True,
        },
        "loss_protocol": {
            "metric": "token_mean_next_token_cross_entropy",
            "delta_loss": "mean NLL of one-rank-quantum-added realization minus Uniform068 reference",
            "logits_for_loss": "float32 conversion before cross_entropy for both WikiText-2 and C4 tuning windows",
            "aggregation": "pool token-level NLL sums and target-token counts across all 61 windows; each window has 2047 next-token targets",
            "same_loss_definition_for_all_64_path_probes": True,
            "not_final_quality_evaluation": True,
        },
        "tuning_split": {
            "manifest_path": str(TUNE_MANIFEST),
            "manifest_sha256": hashes[str(TUNE_MANIFEST)],
            "source_manifest_declared_model_fingerprint": tune_manifest["model_content_fingerprint_sha256"],
            "canonical_model_fingerprint_bound_by_metadata_erratum": EXPECTED["model_fingerprint_sha256"],
            "model_identity_metadata_erratum_path": str(addendum_path),
            "model_identity_metadata_erratum_sha256": sha_file(addendum_path),
            "window_data_path": str(TUNE_WINDOWS),
            "window_data_sha256": hashes[str(TUNE_WINDOWS)],
            "unique_windows": len(windows),
            "dataset_counts": {"wikitext2_train": 32, "c4_nonoverlap_continuation": 29},
            "disjointness": tune_manifest["disjointness"],
            "half_assignment_rule": "within each dataset's frozen manifest order, even index -> A and odd index -> B",
            "half_ids": assignment,
        },
        "input_hashes": hashes,
        "runner_sha256": sha_file(RUNNER_PATH),
        "attempt": 7,
        "supersedes_attempt": {
            "path": str(PREVIOUS_ATTEMPT),
            "status_path": str(PREVIOUS_ATTEMPT / "RUN_STATUS.json"),
            "status_sha256": sha_file(PREVIOUS_ATTEMPT / "RUN_STATUS.json"),
            "canary_status_path": str(PREVIOUS_ATTEMPT / "MATERIALIZATION_CANARY_GPU0_STATUS.json"),
            "canary_status_sha256": sha_file(PREVIOUS_ATTEMPT / "MATERIALIZATION_CANARY_GPU0_STATUS.json"),
            "canary_log_path": str(PREVIOUS_ATTEMPT / "MATERIALIZATION_CANARY_GPU0.log"),
            "canary_log_sha256": sha_file(PREVIOUS_ATTEMPT / "MATERIALIZATION_CANARY_GPU0.log"),
            "worker_log_sha256": {f"gpu{g}": sha_file(PREVIOUS_ATTEMPT / f"worker_gpu{g}.log") for g in range(WORKER_COUNT)},
            "classification": "ENGINEERING_ADAPTER_SCOPE_RESTORED_BEFORE_SINGLE_LAYER_RECONSTRUCTION_ATTEMPT6_NO_PROBES_COMPLETED",
        },
        "import_compatibility": {
            "bindings": [
                {"name": "Cache", "source": "transformers.cache_utils.Cache", "reason": "type annotation export compatibility"},
                {"name": "LlamaRotaryEmbedding", "source": "transformers.models.llama.modeling_llama.LlamaRotaryEmbedding", "reason": "HF 4.53 adapter for the historical three-argument constructor call"},
                {"name": "apply_rotary_pos_emb", "source": "transformers.models.llama.modeling_llama.apply_rotary_pos_emb", "reason": "exact upstream HF 4.53 rotary operator omitted by modeling_llama.__all__"},
                {"name": "repeat_kv", "source": "transformers.models.llama.modeling_llama.repeat_kv", "reason": "exact upstream HF 4.53 KV replication operator omitted by modeling_llama.__all__"}
            ],
            "scope": "worker-subprocess-lifetime name bindings for legacy operator references missing from HF 4.53 wildcard exports; each worker runs in an isolated process; all functions delegate to exact upstream HF 4.53 symbols; rotary adapter delegates to upstream LlamaRotaryEmbedding configured from the exact local Llama-2 config after checking dim/base/max-position; no attention/operator math changes",
        },
        "target_output_directory": str(OUT),
        "gpu_preflight": gpu_snapshot(),
        "disk_free_bytes_preflight": {p: shutil.disk_usage(p).free for p in ("/diskE", "/diskF", "/diskD", "/")},
        "disk_write_policy": "small protocol/results only on E; model, basis and inputs read-only; no D or F writes",
    }
    lock_path = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_PROTOCOL_LOCK.json"
    atomic_json(lock_path, lock)
    status = {"artifact_id": "LLAMA2_E068_NATIVE_ADD_BENEFIT_RUN_STATUS_V1", "status": "FROZEN_PRELAUNCH", "protocol_lock": str(lock_path), "protocol_lock_sha256": sha_file(lock_path), "updated_utc": utc_now()}
    atomic_json(OUT / "RUN_STATUS.json", status)
    print(json.dumps(status, indent=2))


def check_gpu_preflight(min_free=GPU_GATE_MIB):
    snap = gpu_snapshot()
    if len(snap) < WORKER_COUNT:
        raise RuntimeError(f"need {WORKER_COUNT} visible GPUs, found {len(snap)}")
    bad = [g for g in snap[:WORKER_COUNT] if g["free_mib"] < min_free]
    if bad:
        raise RuntimeError(f"GPU preflight below {min_free} MiB free: {bad}")
    # Do not share with an already allocated CUDA process for this three-model run.
    occupied = [g for g in snap[:WORKER_COUNT] if g["used_mib"] > 512]
    if occupied:
        raise RuntimeError(f"GPU is not clean for three-way probe: {occupied}")
    return snap


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_windows(model, windows, device, torch):
    import torch.nn.functional as F
    model.eval()
    sums = {"A": 0.0, "B": 0.0}
    tokens = {"A": 0, "B": 0}
    for w in windows:
        ids = torch.tensor(w["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits
            shifted_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1)).float()
            shifted_labels = ids[:, 1:].reshape(-1)
            nll = F.cross_entropy(shifted_logits, shifted_labels, reduction="sum")
        if not bool(torch.isfinite(nll).item()):
            raise RuntimeError(f"nonfinite NLL at tuning window {w['window_id']}")
        half = w["_half"]
        sums[half] += float(nll.item())
        tokens[half] += int(shifted_labels.numel())
        del ids, logits, shifted_logits, shifted_labels, nll
    return {h: {"nll_sum": sums[h], "token_count": tokens[h], "mean_nll": sums[h] / tokens[h]} for h in ("A", "B")}


def worker(gpu_index, shard_index, shard_count, lock_sha, build_only=False):
    import torch
    if torch.cuda.is_available() is False:
        raise RuntimeError("CUDA is unavailable in worker")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info(0)[0] < GPU_GATE_MIB * 1024 * 1024:
        raise RuntimeError("worker visible GPU below 36 GiB free gate")
    lock_path = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_PROTOCOL_LOCK.json"
    if sha_file(lock_path) != lock_sha:
        raise RuntimeError("frozen protocol lock SHA changed")
    lock = json.loads(lock_path.read_text())
    if lock.get("runner_sha256") != sha_file(RUNNER_PATH):
        raise RuntimeError("worker runner SHA differs from pre-scoring protocol lock")
    profile = json.loads(PROFILE.read_text())
    with RANK_CSV.open(newline="") as f:
        rank_rows = list(csv.DictReader(f))
    pairs = [[int(r["uniform068_QK_rank"]), int(r["uniform068_V_rank"])] for r in rank_rows]
    q = [float(x) for x in profile["q"]]
    with TUNE_WINDOWS.open() as f:
        windows = [json.loads(line) for line in f if line.strip()]
    half_ids = lock["tuning_split"]["half_ids"]
    id_to_half = {wid: half for half, ids in half_ids.items() for wid in ids}
    for w in windows:
        w["_half"] = "A" if w["window_id"] in id_to_half and w["window_id"] in half_ids["A"] else "B"
    if len(id_to_half) != len(windows):
        raise RuntimeError("frozen split assignment does not cover all tuning windows")

    # transformers 4.53 omits Cache/RoPE names from the old operator's wildcard exports,
    # and changed LlamaRotaryEmbedding's constructor API. Bind exact upstream objects;
    # adapt only the constructor signature after checking against this checkpoint config.
    import builtins
    from transformers.cache_utils import Cache
    from transformers import AutoConfig
    from transformers.models.llama import modeling_llama as hf_llama
    HFLLamaRotaryEmbedding = hf_llama.LlamaRotaryEmbedding
    rope_config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    if rope_config.model_type != "llama" or rope_config.rope_scaling is not None:
        raise RuntimeError("compatibility adapter only accepts the frozen unscaled Llama-2 RoPE config")
    def legacy_rope_constructor(dim, max_position_embeddings, base):
        expected_dim = rope_config.hidden_size // rope_config.num_attention_heads
        if int(dim) != expected_dim or int(max_position_embeddings) != int(rope_config.max_position_embeddings) or float(base) != float(rope_config.rope_theta):
            raise RuntimeError("legacy RoPE constructor arguments differ from frozen Llama-2 config")
        cfg = copy.deepcopy(rope_config)
        cfg.max_position_embeddings = int(max_position_embeddings)
        cfg.rope_theta = float(base)
        return HFLLamaRotaryEmbedding(config=cfg)
    builtins.Cache = Cache
    builtins.LlamaRotaryEmbedding = legacy_rope_constructor
    builtins.apply_rotary_pos_emb = hf_llama.apply_rotary_pos_emb
    builtins.repeat_kv = hf_llama.repeat_kv
    # Keep exact upstream bindings for the whole isolated worker process: the
    # canonical materializer and every one-layer perturbation both construct
    # LlamaEigenAttention modules, and forward resolves the omitted helpers.
    cm = load_module(f"canonical_llama2_gpu{gpu_index}", CANONICAL)
    baseline_rows = [{"layer": i, "rank_kq": rk, "rank_v": rv} for i, (rk, rv) in enumerate(pairs)]
    census = {"official": {"per_layer": baseline_rows, "rank_pairs": pairs}}
    cm.common_preflight = lambda: ({}, census, {}, {}, {"identity": "llama2_7b"})
    cm.basis_complete = lambda: True
    cm.E_MIN_BYTES = 0  # no model/checkpoint is written; all model realization stays in GPU memory
    cm.GPU_GATE_MIB = GPU_GATE_MIB
    cm.gpu_free_mib = lambda: int(torch.cuda.mem_get_info(0)[0] // (1024 * 1024))
    cm.disk_free = lambda: shutil.disk_usage("/diskE").free
    cm.MODEL = MODEL
    cm.BASIS_FILE = BASIS
    basis_meta = {"basis_sha256": EXPECTED["basis_sha256"], "basis_bytes": BASIS.stat().st_size}
    model, factor_calc = cm.build_model("official", basis_meta)
    runtime_pairs = [[int(x["rank_kq"]), int(x["rank_v"])] for x in factor_calc]
    if runtime_pairs != pairs:
        raise RuntimeError("canonical runtime schedule differs from frozen Uniform068 ranks")
    device = torch.device("cuda:0")

    if build_only:
        test_rope = model.model.layers[0].self_attn.rotary_emb
        ref_rope = HFLLamaRotaryEmbedding(config=copy.deepcopy(rope_config)).to(device=device, dtype=test_rope.inv_freq.dtype)
        pos = torch.arange(4, device=device, dtype=torch.long).unsqueeze(0)
        x = torch.zeros((1, 32, 4, 128), device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            cos_a, sin_a = test_rope(x, pos)
            cos_b, sin_b = ref_rope(x, pos)
            max_rope_error = max(float((cos_a.float() - cos_b.float()).abs().max().item()), float((sin_a.float() - sin_b.float()).abs().max().item()))
            ids = torch.tensor(windows[0]["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids=ids, use_cache=False).logits
        if max_rope_error != 0.0 or not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"materialization/RoPE canary failed: rope_max_abs={max_rope_error}, finite={bool(torch.isfinite(logits).all())}")
        canary = {
            "artifact_id": "LLAMA2_E068_NATIVE_ADD_BENEFIT_MATERIALIZATION_CANARY_V1",
            "status": "PASS_BASELINE_BUILD_AND_FORWARD_CANARY",
            "gpu_index": gpu_index,
            "runner_sha256": sha_file(RUNNER_PATH),
            "protocol_lock_sha256": lock_sha,
            "basis_sha256": EXPECTED["basis_sha256"], "rank_schedule_sha256": EXPECTED["rank_schedule_sha256"],
            "B_live_bytes": EXPECTED["baseline_B_live"], "runtime_layer_count": len(factor_calc),
            "runtime_rank_schedule_matches_frozen": runtime_pairs == pairs,
            "rope_compatibility": "HF 4.53 upstream LlamaRotaryEmbedding instantiated with frozen local Llama-2 config; exact cos/sin parity on positions 0..3",
            "runtime_rope_buffer_dtype": str(test_rope.inv_freq.dtype),
            "reference_rope_buffer_dtype": str(ref_rope.inv_freq.dtype),
            "rope_max_abs_error": max_rope_error,
            "finite_logits": bool(torch.isfinite(logits).all()), "logits_shape": list(logits.shape),
            "token_window_id": windows[0]["window_id"],
            "final_quality_started": False, "probe_count_started": 0,
            "created_utc": utc_now(),
        }
        atomic_json(OUT / f"MATERIALIZATION_CANARY_GPU{gpu_index}.json", canary)
        del ref_rope, test_rope, x, pos, logits, ids, model
        torch.cuda.empty_cache()
        print(json.dumps(canary, indent=2), flush=True)
        return

    # Keep frozen original per-layer weights on CPU for exact one-layer reconstruction.
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    base_model.eval()
    base_model.config.use_cache = False
    if base_model.config.num_hidden_layers != 32 or base_model.config.num_attention_heads != 32:
        raise RuntimeError("base checkpoint architecture does not match frozen Llama-2 binding")

    reference = score_windows(model, windows, device, torch)
    completed_path = OUT / f"worker_gpu{gpu_index}_probes.jsonl"
    completed = set()
    if completed_path.exists():
        for line in completed_path.read_text().splitlines():
            if line.strip():
                completed.add(json.loads(line)["unit_id"])

    from types import SimpleNamespace
    from models.decompose_modules import LlamaEigenAttnDecoderLayer
    raw_basis = torch.load(BASIS, map_location="cpu", weights_only=False, mmap=True)
    work = []
    for li in range(32):
        for path in ("qk", "v"):
            unit_id = f"L{li:02d}_{path.upper()}"
            if len(work) % shard_count == shard_index and unit_id not in completed:
                work.append((li, path, unit_id))
    # Assignment is by stable global ordinal, not local append order.
    all_units = [(li, p, f"L{li:02d}_{p.upper()}") for li in range(32) for p in ("qk", "v")]
    work = [(li, p, uid) for ordinal, (li, p, uid) in enumerate(all_units) if ordinal % shard_count == shard_index and uid not in completed]
    args = SimpleNamespace()
    head_dim = 128
    for n, (li, path, unit_id) in enumerate(work, 1):
        rk, rv = pairs[li]
        next_rk = rk + RANK_QUANTUM if path == "qk" and rk < RANK_MAX else rk
        next_rv = rv + RANK_QUANTUM if path == "v" and rv < RANK_MAX else rv
        if next_rk == rk and next_rv == rv:
            record = {"unit_id": unit_id, "layer": li, "path": path, "status": "SATURATED_NO_RECEIVER_STEP", "current_q": q[li], "current_qk_rank": rk, "current_v_rank": rv, "next_qk_rank": None, "next_v_rank": None, "delta_bytes": 0}
        else:
            base_layer = base_model.model.layers[li].to(device)
            prefix = f"base_model.model.model.layers.{li}.self_attn."
            basis_k = raw_basis[prefix + "k_proj_basis"][:, :next_rk // 32].to(device=device, dtype=torch.bfloat16)
            vp = raw_basis[prefix + "v_proj_basis"]
            vstack = torch.stack([vp[h * head_dim:(h + 1) * head_dim, h * head_dim:(h + 1) * head_dim] for h in range(32)], dim=0)[:, :, :next_rv // 32].to(device=device, dtype=torch.bfloat16)
            perturbed = LlamaEigenAttnDecoderLayer(base_layer, args, basis_k, next_rk, vstack, next_rv, model.config, li).to(device=device, dtype=torch.bfloat16)
            for name, param in perturbed.named_parameters():
                if any(k in name for k in ("k_proj_low.weight", "k_proj_up.weight", "v_proj_low.weight", "o_proj_up.weight")) and not torch.isfinite(param).all():
                    raise RuntimeError(f"nonfinite factor {name} in {unit_id}")
            reference_layer = model.model.layers[li]
            model.model.layers[li] = perturbed
            measured = score_windows(model, windows, device, torch)
            model.model.layers[li] = reference_layer
            base_layer.to("cpu")
            delta_bytes = (RANK_QUANTUM * BYTES_PER_RANK_UNIT)
            delta_loss = ((measured["A"]["nll_sum"] + measured["B"]["nll_sum"]) - (reference["A"]["nll_sum"] + reference["B"]["nll_sum"])) / (reference["A"]["token_count"] + reference["B"]["token_count"])
            da = measured["A"]["mean_nll"] - reference["A"]["mean_nll"]
            db = measured["B"]["mean_nll"] - reference["B"]["mean_nll"]
            record = {
                "unit_id": unit_id, "layer": li, "path": path, "status": "MEASURED",
                "current_q": q[li], "current_qk_rank": rk, "current_v_rank": rv,
                "next_qk_rank": next_rk, "next_v_rank": next_rv,
                "delta_bytes": delta_bytes, "delta_loss": delta_loss,
                "fragility_per_byte": delta_loss / delta_bytes,
                "delta_loss_half_a": da, "delta_loss_half_b": db,
                "fragility_per_byte_half_a": da / delta_bytes, "fragility_per_byte_half_b": db / delta_bytes,
                "split_sign_agreement": bool((da > 0 and db > 0) or (da < 0 and db < 0) or (da == 0 and db == 0)),
                "reference_loss": reference, "perturbed_loss": measured,
            }
            del perturbed, basis_k, vstack, vp, base_layer
            torch.cuda.empty_cache()
        with completed_path.open("a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"gpu{gpu_index} probe {n}/{len(work)} {unit_id} {record['status']}", flush=True)
        atomic_json(OUT / f"worker_gpu{gpu_index}_status.json", {
            "status": "RUNNING", "gpu_index": gpu_index, "shard_index": shard_index,
            "completed_this_attempt": n, "assigned_this_attempt": len(work),
            "completed_total": len(completed) + n, "reference_loss": reference,
            "updated_utc": utc_now(),
        })
    atomic_json(OUT / f"worker_gpu{gpu_index}_summary.json", {
        "status": "WORKER_COMPLETE", "gpu_index": gpu_index, "shard_index": shard_index,
        "reference_loss": reference, "probe_file": str(completed_path),
        "probe_file_sha256": sha_file(completed_path), "updated_utc": utc_now(),
    })


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / math.sqrt(vx * vy) if vx and vy else None


def finalize():
    lock_path = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_PROTOCOL_LOCK.json"
    lock_sha = sha_file(lock_path)
    raw = []
    for gpu in range(WORKER_COUNT):
        summary_path = OUT / f"worker_gpu{gpu}_summary.json"
        probe_path = OUT / f"worker_gpu{gpu}_probes.jsonl"
        if not summary_path.exists() or not probe_path.exists():
            raise RuntimeError(f"missing worker completion: GPU{gpu}")
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "WORKER_COMPLETE":
            raise RuntimeError(f"worker not complete: GPU{gpu}")
        raw.extend(json.loads(line) for line in probe_path.read_text().splitlines() if line.strip())
    by_id = {x["unit_id"]: x for x in raw}
    expected_ids = {f"L{li:02d}_{path.upper()}" for li in range(32) for path in ("qk", "v")}
    if set(by_id) != expected_ids:
        raise RuntimeError(f"probe coverage mismatch: {len(by_id)}/64")
    if any(x["status"] not in ("MEASURED", "SATURATED_NO_RECEIVER_STEP") for x in raw):
        raise RuntimeError("probe contains unexpected status")

    # Check reference NLL reproduction across workers; local deltas remain paired to each worker's reference.
    refs = [json.loads((OUT / f"worker_gpu{g}_summary.json").read_text())["reference_loss"] for g in range(WORKER_COUNT)]
    ref_mean = {}
    ref_spread = {}
    for h in ("A", "B"):
        vals = [r[h]["mean_nll"] for r in refs]
        ref_mean[h] = sum(vals) / len(vals)
        ref_spread[h] = max(vals) - min(vals)

    layer_rows = []
    for li in range(32):
        qk = by_id[f"L{li:02d}_QK"]
        vv = by_id[f"L{li:02d}_V"]
        row = {
            "layer": li, "current_q": qk["current_q"],
            "current_qk_rank": qk["current_qk_rank"], "current_v_rank": qk["current_v_rank"],
            "qk_next_rank": qk.get("next_qk_rank"), "qk_delta_bytes": qk["delta_bytes"],
            "qk_delta_loss": qk.get("delta_loss"), "qk_fragility_per_byte": qk.get("fragility_per_byte"),
            "qk_delta_loss_half_a": qk.get("delta_loss_half_a"), "qk_delta_loss_half_b": qk.get("delta_loss_half_b"),
            "qk_fragility_per_byte_half_a": qk.get("fragility_per_byte_half_a"), "qk_fragility_per_byte_half_b": qk.get("fragility_per_byte_half_b"),
            "qk_split_sign_agreement": qk.get("split_sign_agreement"), "qk_status": qk["status"],
            "v_next_rank": vv.get("next_v_rank"), "v_delta_bytes": vv["delta_bytes"],
            "v_delta_loss": vv.get("delta_loss"), "v_fragility_per_byte": vv.get("fragility_per_byte"),
            "v_delta_loss_half_a": vv.get("delta_loss_half_a"), "v_delta_loss_half_b": vv.get("delta_loss_half_b"),
            "v_fragility_per_byte_half_a": vv.get("fragility_per_byte_half_a"), "v_fragility_per_byte_half_b": vv.get("fragility_per_byte_half_b"),
            "v_split_sign_agreement": vv.get("split_sign_agreement"), "v_status": vv["status"],
        }
        vals = [x for x in (qk.get("fragility_per_byte"), vv.get("fragility_per_byte")) if x is not None]
        deltas = [x.get("delta_loss") for x in (qk, vv) if x.get("delta_loss") is not None]
        bytes_ = [x.get("delta_bytes", 0) for x in (qk, vv) if x.get("delta_loss") is not None]
        row["max_path_fragility"] = max(vals) if vals else None
        row["mean_path_fragility"] = sum(vals) / len(vals) if vals else None
        row["byte_weighted_fragility"] = sum(deltas) / sum(bytes_) if bytes_ else None
        layer_rows.append(row)

    out_csv = OUT / "LLAMA2_E068_NATIVE_FRAGILITY.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
        w.writeheader(); w.writerows(layer_rows)

    path_units = [x for x in raw if x["status"] == "MEASURED"]
    layer_valid = [r for r in layer_rows if r["byte_weighted_fragility"] is not None]
    q_by_layer = {i: float(json.loads(PROFILE.read_text())["q"][i]) for i in range(32)}
    q_order = sorted(range(32), key=lambda i: (-q_by_layer[i], i))
    frag_order = sorted((r["layer"] for r in layer_valid), key=lambda i: (-layer_rows[i]["byte_weighted_fragility"], i))
    q_bottom = sorted(range(32), key=lambda i: (q_by_layer[i], i))
    frag_bottom = sorted((r["layer"] for r in layer_valid), key=lambda i: (layer_rows[i]["byte_weighted_fragility"], i))
    layer_spear = spearman([q_by_layer[i] for i in range(32) if layer_rows[i]["byte_weighted_fragility"] is not None], [layer_rows[i]["byte_weighted_fragility"] for i in range(32) if layer_rows[i]["byte_weighted_fragility"] is not None])

    # Split-half stability is computed on the 64 distinct native path units.
    stable_a = [x for x in path_units if x.get("fragility_per_byte_half_a") is not None and x.get("fragility_per_byte_half_b") is not None]
    split_spear = spearman([x["fragility_per_byte_half_a"] for x in stable_a], [x["fragility_per_byte_half_b"] for x in stable_a])
    order_a = sorted(stable_a, key=lambda x: (-x["fragility_per_byte_half_a"], x["unit_id"]))
    order_b = sorted(stable_a, key=lambda x: (-x["fragility_per_byte_half_b"], x["unit_id"]))
    top8_a, top8_b = {x["unit_id"] for x in order_a[:8]}, {x["unit_id"] for x in order_b[:8]}
    sign_ok = sum(1 for x in stable_a if x["split_sign_agreement"])

    receivers = sorted([x for x in path_units if x["current_" + ("qk_rank" if x["path"] == "qk" else "v_rank")] < RANK_MAX], key=lambda x: (-x["fragility_per_byte"], x["unit_id"]))
    donors = sorted([x for x in path_units if x["current_" + ("qk_rank" if x["path"] == "qk" else "v_rank")] > RANK_MIN], key=lambda x: (x["fragility_per_byte"], x["unit_id"]))
    robust_receivers = [x for x in receivers if x["delta_loss_half_a"] > 0 and x["delta_loss_half_b"] > 0]
    stable_donors = [x for x in donors if x["delta_loss_half_a"] <= 0 and x["delta_loss_half_b"] <= 0]
    chosen = None
    for recv in robust_receivers:
        for donor in stable_donors:
            if recv["unit_id"] != donor["unit_id"] and recv["layer"] != donor["layer"] and recv["fragility_per_byte"] > donor["fragility_per_byte"]:
                chosen = (donor, recv); break
        if chosen: break
    baseline_pairs = [list(x) for x in json.loads(PROFILE.read_text())["compressed_comparator"]["rank_schedule"]]
    candidate_pairs = [x[:] for x in baseline_pairs]
    exchanges = []
    if chosen:
        donor, recv = chosen
        dcol = 0 if donor["path"] == "qk" else 1
        rcol = 0 if recv["path"] == "qk" else 1
        candidate_pairs[donor["layer"]][dcol] -= RANK_QUANTUM
        candidate_pairs[recv["layer"]][rcol] += RANK_QUANTUM
        exchanges.append({"donor_unit": donor["unit_id"], "receiver_unit": recv["unit_id"], "rank_quantum": RANK_QUANTUM,
                          "donor_fragility_per_byte": donor["fragility_per_byte"], "receiver_fragility_per_byte": recv["fragility_per_byte"],
                          "donor_delta_loss_half_a": donor["delta_loss_half_a"], "donor_delta_loss_half_b": donor["delta_loss_half_b"],
                          "receiver_delta_loss_half_a": recv["delta_loss_half_a"], "receiver_delta_loss_half_b": recv["delta_loss_half_b"]})
    candidate_B = sum((k + v) * BYTES_PER_RANK_UNIT for k, v in candidate_pairs)
    if candidate_B != EXPECTED["baseline_B_live"]:
        raise RuntimeError("proposed one-quantum exchange does not exactly preserve target B_live")

    candidate_path = OUT / "LLAMA2_E068_NATIVE_FRAGILITY_PROPOSED_EXCHANGE.csv"
    proposed_rows = []
    for i, (base_pair, cand_pair) in enumerate(zip(baseline_pairs, candidate_pairs)):
        proposed_rows.append({"layer": i, "uniform068_qk_rank": base_pair[0], "uniform068_v_rank": base_pair[1],
                              "proposed_qk_rank": cand_pair[0], "proposed_v_rank": cand_pair[1],
                              "delta_qk_rank": cand_pair[0] - base_pair[0], "delta_v_rank": cand_pair[1] - base_pair[1],
                              "delta_B_live_bytes": ((cand_pair[0] + cand_pair[1]) - sum(base_pair)) * BYTES_PER_RANK_UNIT})
    with candidate_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(proposed_rows[0])); w.writeheader(); w.writerows(proposed_rows)

    old = {}
    with OLD_FDP060.open(newline="") as f:
        for x in csv.DictReader(f):
            li = int(x["layer"])
            old[li] = {"role": x["role"], "delta_qk_rank": int(x["delta_QK_rank"]), "delta_v_rank": int(x["delta_V_rank"])}
    old_comparison = []
    old_role_results = {"receiver": [], "donor": []}
    for li in range(32):
        for path, delta_key, unit_key in (("qk", "delta_qk_rank", f"L{li:02d}_QK"), ("v", "delta_v_rank", f"L{li:02d}_V")):
            delta_rank = old[li][delta_key]
            unit = by_id[unit_key]
            if delta_rank:
                role = "receiver" if delta_rank > 0 else "donor"
                agrees_full = unit.get("fragility_per_byte", 0.0) > 0 if role == "receiver" else unit.get("fragility_per_byte", 0.0) < 0
                agrees_halves = (unit.get("delta_loss_half_a", 0.0) > 0 and unit.get("delta_loss_half_b", 0.0) > 0) if role == "receiver" else (unit.get("delta_loss_half_a", 0.0) < 0 and unit.get("delta_loss_half_b", 0.0) < 0)
                item = {"unit_id": unit_key, "old_role": role.upper(), "old_delta_rank": delta_rank,
                        "measured_fragility_per_byte": unit.get("fragility_per_byte"),
                        "delta_loss_half_a": unit.get("delta_loss_half_a"), "delta_loss_half_b": unit.get("delta_loss_half_b"),
                        "direction_agrees_full_sample": agrees_full, "direction_agrees_both_halves": agrees_halves}
                old_comparison.append(item)
                old_role_results[role].append(item)

    unit_sort_desc = sorted(path_units, key=lambda x: (-x["fragility_per_byte"], x["unit_id"]))
    unit_sort_asc = sorted(path_units, key=lambda x: (x["fragility_per_byte"], x["unit_id"]))
    split_stable_safe_units = sorted([x for x in path_units if x["delta_loss_half_a"] < 0 and x["delta_loss_half_b"] < 0],
                                     key=lambda x: (x["fragility_per_byte"], x["unit_id"]))
    high_q_low_measured = sorted(set(q_order[:8]) & set(frag_bottom[:8]))
    low_q_high_measured = sorted(set(q_bottom[:8]) & set(frag_order[:8]))
    summary = {
        "artifact_id": "LLAMA2_E068_NATIVE_FRAGILITY_SUMMARY_V1",
        "status": "PASS_PROBES_COMPLETE_NO_FINAL_QUALITY",
        "created_utc": utc_now(),
        "protocol_lock_path": str(lock_path), "protocol_lock_sha256": lock_sha,
        "profile_q_sha256": EXPECTED["profile_q_sha256"], "basis_sha256": EXPECTED["basis_sha256"],
        "rank_schedule_sha256": EXPECTED["rank_schedule_sha256"], "baseline_B_live_bytes": EXPECTED["baseline_B_live"],
        "rank_quantum": RANK_QUANTUM, "delta_B_live_bytes_per_probe": RANK_QUANTUM * BYTES_PER_RANK_UNIT,
        "probe_count_measured": len(path_units), "saturated_count": sum(x["status"] == "SATURATED_NO_RECEIVER_STEP" for x in raw),
        "calibration_split": {"manifest_sha256": EXPECTED["tuning_manifest_sha256"], "window_data_sha256": EXPECTED["tuning_windows_sha256"],
                              "unique_windows": 61, "half_A_count": len(lock["tuning_split"]["half_ids"]["A"]), "half_B_count": len(lock["tuning_split"]["half_ids"]["B"]),
                              "C4_is_token_window_disjoint_but_same_source_documents": True, "final_accuracy_used": False},
        "loss_definition": lock["loss_protocol"],
        "layer_fragility_spearman_profile_q": layer_spear,
        "top8_q_layers": q_order[:8], "top8_measured_fragility_layers": frag_order[:8],
        "top8_layer_overlap": sorted(set(q_order[:8]) & set(frag_order[:8])),
        "bottom8_q_layers": q_bottom[:8], "bottom8_measured_fragility_layers": frag_bottom[:8],
        "bottom8_layer_overlap": sorted(set(q_bottom[:8]) & set(frag_bottom[:8])),
        "high_q_bottom_measured_fragility_layers": high_q_low_measured,
        "low_q_top_measured_fragility_layers": low_q_high_measured,
        "split_half_stability": {"path_unit_count": len(stable_a), "spearman_half_a_vs_b": split_spear,
                                  "top8_half_a": [x["unit_id"] for x in order_a[:8]], "top8_half_b": [x["unit_id"] for x in order_b[:8]],
                                  "top8_overlap": sorted(top8_a & top8_b), "sign_agreement_count": sign_ok,
                                  "sign_agreement_fraction": sign_ok / len(stable_a) if stable_a else None},
        "top10_most_fragile_path_units": unit_sort_desc[:10], "top10_safest_donor_path_units": unit_sort_asc[:10],
        "old_fdp060_changed_path_comparison": old_comparison,
        "old_fdp060_direction_agreement": {
            "receiver_count": len(old_role_results["receiver"]),
            "receiver_agrees_full_sample": sum(x["direction_agrees_full_sample"] for x in old_role_results["receiver"]),
            "receiver_agrees_both_halves": sum(x["direction_agrees_both_halves"] for x in old_role_results["receiver"]),
            "donor_count": len(old_role_results["donor"]),
            "donor_agrees_full_sample": sum(x["direction_agrees_full_sample"] for x in old_role_results["donor"]),
            "donor_agrees_both_halves": sum(x["direction_agrees_both_halves"] for x in old_role_results["donor"]),
        },
        "split_stable_negative_cost_donor_units": split_stable_safe_units,
        "proposed_exchange": {"status": "ONE_QUANTUM_PROPOSAL_NOT_QUALITY_EVALUATED" if chosen else "NO_SPLIT_STABLE_CROSS_LAYER_EXCHANGE_FOUND",
                              "exchange_count": len(exchanges), "exchanges": exchanges,
                              "candidate_B_live_bytes": candidate_B, "target_gap_bytes": candidate_B - EXPECTED["baseline_B_live"],
                              "affected_layers": sorted({int(x[k][1:3]) for x in exchanges for k in ("donor_unit", "receiver_unit")}) if exchanges else [],
                              "schedule_csv": str(candidate_path),
                              "global_allocator_run": False, "final_quality_run": False},
        "baseline_reference_loss_by_worker": refs, "reference_loss_half_spread": ref_spread,
        "csv_path": str(out_csv), "csv_sha256": sha_file(out_csv),
        "proposed_schedule_csv_sha256": sha_file(candidate_path),
        "worker_result_files": {str(p): sha_file(p) for p in sorted(OUT.glob("worker_gpu*_probes.jsonl"))},
        "final_quality_started": False,
    }
    atomic_json(OUT / "LLAMA2_E068_NATIVE_FRAGILITY_SUMMARY.json", summary)
    status = {"artifact_id": "LLAMA2_E068_NATIVE_ADD_BENEFIT_RUN_STATUS_V1", "status": summary["status"],
              "completed_probe_count": len(path_units), "updated_utc": utc_now(),
              "summary_path": str(OUT / "LLAMA2_E068_NATIVE_FRAGILITY_SUMMARY.json"),
              "summary_sha256": sha_file(OUT / "LLAMA2_E068_NATIVE_FRAGILITY_SUMMARY.json")}
    atomic_json(OUT / "RUN_STATUS.json", status)
    print(json.dumps({"status": summary["status"], "probes": len(path_units), "B_live": candidate_B,
                      "q_fragility_spearman": layer_spear, "split_spearman": split_spear,
                      "summary": str(OUT / "LLAMA2_E068_NATIVE_FRAGILITY_SUMMARY.json")}, indent=2))


def launch():
    lock_path = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_PROTOCOL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("run --preflight first")
    lock_sha = sha_file(lock_path)
    gpu_before = check_gpu_preflight()
    status = {"artifact_id": "LLAMA2_E068_NATIVE_ADD_BENEFIT_RUN_STATUS_V1", "status": "RUNNING",
              "protocol_lock_sha256": lock_sha, "gpu_preflight": gpu_before, "worker_count": WORKER_COUNT,
              "started_utc": utc_now(), "final_quality_started": False}
    atomic_json(OUT / "RUN_STATUS.json", status)
    procs = []
    logs = []
    for gpu in range(WORKER_COUNT):
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1", "PYTHONPATH": "/home/bai-project/jiaqizhang/EigenAttn:/home/bai-project/jiaqizhang/EigenAttn/EigenAttn"})
        log_path = OUT / f"worker_gpu{gpu}.log"
        stream = log_path.open("a")
        proc = subprocess.Popen([sys.executable, str(RUNNER_PATH), "--worker", str(gpu), str(gpu), str(WORKER_COUNT), lock_sha],
                                env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((gpu, proc, stream)); logs.append(str(log_path))
    status["worker_pids"] = {str(g): p.pid for g, p, _ in procs}
    status["worker_logs"] = logs
    atomic_json(OUT / "RUN_STATUS.json", status)
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(60)
        status["workers"] = {str(g): {"pid": p.pid, "returncode": p.poll()} for g, p, _ in procs}
        status["latest_gpu_snapshot"] = gpu_snapshot()
        status["updated_utc"] = utc_now()
        atomic_json(OUT / "RUN_STATUS.json", status)
    for _, _, stream in procs:
        stream.close()
    failures = {str(g): p.returncode for g, p, _ in procs if p.returncode != 0}
    if failures:
        status.update({"status": "WORKER_FAILURE_PRESERVED", "failures": failures, "updated_utc": utc_now()})
        atomic_json(OUT / "RUN_STATUS.json", status)
        raise RuntimeError(f"probe workers failed: {failures}")
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", nargs=4, metavar=("GPU", "SHARD", "SHARDS", "LOCK_SHA"))
    parser.add_argument("--materialization-canary", type=int, metavar="GPU")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        freeze_protocol()
    elif args.launch:
        launch()
    elif args.worker:
        worker(int(args.worker[0]), int(args.worker[1]), int(args.worker[2]), args.worker[3])
    elif args.materialization_canary is not None:
        lock_path = OUT / "LLAMA2_E068_NATIVE_ADD_BENEFIT_PROTOCOL_LOCK.json"
        worker(args.materialization_canary, args.materialization_canary, WORKER_COUNT, sha_file(lock_path), build_only=True)
    elif args.finalize:
        finalize()
    else:
        parser.error("choose --preflight, --launch, --worker or --finalize")


if __name__ == "__main__":
    main()
