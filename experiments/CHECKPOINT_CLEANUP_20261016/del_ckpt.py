#!/usr/bin/env python3
"""Delete compressed-model weight files after writing a per-file manifest (bytes + sha256).
User request 2026-10-16: delete all saved compressed models, keep original models.
Keeps config.json, low_rank_config.*, tokenizer files and index files in place; removes only weight shards."""
import datetime, hashlib, json, os, subprocess, sys
from pathlib import Path

DIRS = """/diskD/jiaqizhang/fdp_eb_orth_paper_v1_diskf_spill/qwen_semantic_hold_spill/qwen_Old_FDP_Exact_MEDIUM/final_model
/diskD/jiaqizhang/fdp_eb_orth_paper_v1_diskf_spill/reviewer_full_completion_v2_20260731/external_official/recalkv_models/llama3_ratio-0.5_gs-4-moderate
/diskD/jiaqizhang/llama3_full_resource_robustness_v1/ultra_aggressive_v1/causal_component_loo151m_v1_20261005/drop_d1_materialized_quality_diag_v1_20261008
/diskD/jiaqizhang/fdp125_failed_materialization_quarantine_20260930/llama2_e090_uniform_materialization_partial
/diskD/jiaqizhang/llama2_eb090_true_two_pass_lane_20261001/materialized/UNIFORM_EB090
/diskD/jiaqizhang/LLAMA2_UNIFORM_EB059_STRICT_MATCH_V1/attempt6/final_model
/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/materialized_formal_checkpoints/medium/fdp125_direct/final_model
/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/materialized_formal_checkpoints/medium/official_uniform/final_model
/diskE/jiaqizhang/FLOOR45_CONTROLS_V1_SCRATCH/models/tail45_mat_aggressive/final_model
/diskF/jiaqizhang/fdp125_aggressive_sibling_replay_20261007_gpu1/final_model
/diskF/jiaqizhang/fdp125_aggressive_sibling_replay_20261007_gpu1/realized_model
/diskF/jiaqizhang/FDP_OFFICIAL_FLOOR45_V3_SCRATCH/models/build_fdp_aggressive/final_model
/diskF/jiaqizhang/TACL_LLAMA3_SPECTRAL_CROSS_TIER_V1/materialized_v2/severe_no_causal/final_model
/diskF/jiaqizhang/TACL_LLAMA3_SPECTRAL_CROSS_TIER_V1/materialized_v2/severe_no_z/final_model""".split("\n")
LOG = Path("/diskF/jiaqizhang/CHECKPOINT_CLEANUP_20261016/cleanup.log")
LOG.parent.mkdir(parents=True, exist_ok=True)
ORIGINALS = ["/diskD/jiaqizhang/Llama-3", "/diskF/jiaqizhang/Llama-3.1", "/diskE/jiaqizhang/Qwen2.5-7B", "/diskF/jiaqizhang/models",
             "/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf", "/diskF/jiaqizhang/stepkv/models",
             "/diskD/jiaqizhang/MISTRAL7B_FDP_CANONICAL/model_binding/hf_base"]


def log(m):
    with open(LOG, "a") as f: f.write(f"{datetime.datetime.now().astimezone().isoformat()} {m}\n")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


summary = []
for d in DIRS:
    d = Path(d)
    if not d.is_dir(): log(f"SKIP missing {d}"); continue
    if any(str(d).startswith(o) for o in ORIGINALS): log(f"SKIP original {d}"); continue
    cfg = json.loads((d / "config.json").read_text())
    arch = " ".join(cfg.get("architectures", []))
    lrc = any((d / n).exists() for n in ("low_rank_config.json", "low_rank_config.pt"))
    if not ("EigenAttn" in arch or lrc or "recalkv" in str(d).lower()):
        log(f"SKIP not recognized as compressed {d} arch={arch}"); continue
    weights = sorted([p for p in d.iterdir() if p.is_file() and (p.suffix == ".safetensors" or p.name.startswith("pytorch_model") and p.suffix == ".bin")])
    if not weights: log(f"SKIP no weights {d}"); continue
    busy = subprocess.run(["fuser"] + [str(w) for w in weights], capture_output=True, text=True).stdout.strip()
    if busy: log(f"SKIP in use {d}: {busy}"); continue
    man = {"checkpoint": str(d), "architectures": cfg.get("architectures"), "deleted_at": None,
           "reason": "user request 2026-10-16: delete saved compressed models after all evaluations; originals kept",
           "kept_in_place": sorted(p.name for p in d.iterdir() if p.is_file() and p not in weights), "files": []}
    for p in sorted(d.iterdir()):
        if p.is_file(): man["files"].append({"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p), "deleted": p in weights})
    man["deleted_at"] = datetime.datetime.now().astimezone().isoformat()
    (d / "CHECKPOINT_DELETION_MANIFEST.json").write_text(json.dumps(man, indent=1))
    freed = 0
    for w in weights:
        freed += w.stat().st_size; w.unlink()
    log(f"DELETED {len(weights)} weight files ({freed/1e9:.1f} GB) in {d}; manifest written")
    summary.append({"dir": str(d), "freed_gb": round(freed / 1e9, 2), "n_weights": len(weights)})
(LOG.parent / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
log(f"DONE {len(summary)} checkpoints, {sum(s['freed_gb'] for s in summary):.1f} GB freed")
