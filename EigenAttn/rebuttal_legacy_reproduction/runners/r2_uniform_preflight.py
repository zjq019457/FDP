#!/usr/bin/env python3
"""R2 Uniform eb045 preflight: lock params, materialize surviving basis into isolation, extract tentative ranks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    PYTHON_BIN,
    artifact_dir,
    atomic_write_json,
    file_hash_or_empty,
    gate_payload,
    now_iso,
    sha256_file,
    write_pass_gate,
)

MODEL = "/diskD/jiaqizhang/Llama-3"
WIKITEXT = "/home/bai-project/jiaqizhang/wikitext2"
HIST_BASIS = Path("/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_eb/eigen_basis_llama3_8b_wikitext2_eb0p045.pt")
BUILDER = ROOT / "EigenAttn" / "tools" / "build_eigen_basis_llama_error_budget_offline.py"
MAIN = ROOT / "EigenAttn" / "main_eigen_attn.py"
OUT = artifact_dir("r2_uniform_eb045", link_under=HERE / "artifacts" / "r2_uniform_eb045")
CACHE_DIR = artifact_dir("HF_cache", link_under=ROOT / "EigenAttn" / "HF_cache_rebuttal")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not HIST_BASIS.is_file():
        raise SystemExit(f"historical uniform basis missing unexpectedly: {HIST_BASIS}")

    # Copy basis into isolation (do not modify historical path)
    basis_dst = OUT / "eigen_basis_llama3_8b_wikitext2_eb0p045.pt"
    if not basis_dst.exists() or sha256_file(basis_dst) != sha256_file(HIST_BASIS):
        shutil.copy2(HIST_BASIS, basis_dst)

    import torch

    basis = torch.load(basis_dst, map_location="cpu")
    rows = []
    for L in range(32):
        k = basis[f"base_model.model.model.layers.{L}.self_attn.k_proj_basis"]
        v = basis[f"base_model.model.model.layers.{L}.self_attn.v_proj_basis"]
        # Stored basis second-dim is candidate capacity; realized ranks come from compress low_rank_config.
        rows.append(
            {
                "layer": L,
                "k_basis_shape0": int(k.shape[0]),
                "k_basis_shape1": int(k.shape[1]),
                "v_basis_shape0": int(v.shape[0]),
                "v_basis_shape1": int(v.shape[1]),
                "tentative_kq_rank_from_basis_dim": int(k.shape[1]),
                "tentative_v_rank_from_basis_dim": int(v.shape[1]),
            }
        )

    rank_csv = OUT / "LEGACY_UNIFORM_RANK_PLAN.csv"
    with rank_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Historical compress command evidence (from log Namespace)
    compress_args = {
        "model": MODEL,
        "net": "Llama-3-8b",
        "dtype": "bf16",
        "error_budget": 0.045,
        "avg_dim": 4096,
        "seqlen": 2048,
        "nsamples": 64,
        "seed": 2,
        "calib_dataset": "c4",
        "compress_only": True,
        "attn_implementation": "eager",
        "eigen_basis_path": str(basis_dst),
        "save_dir": str(OUT / "final_model"),
        "output_dir": str(OUT / "compress_run"),
        "cache_dir": str(CACHE_DIR),
        "multigpu": False,
        "single_gpu_policy": "CUDA_VISIBLE_DEVICES=<preferred>; no --multigpu",
        "provenance": {
            "basis": "recovered_from_artifact",
            "basis_source": str(HIST_BASIS),
            "compress_flags": "recovered_from_log",
            "log_evidence": "/diskD/jiaqizhang/llama3_8b_eigen_kv_eb045/log_rank0_1777656418.txt",
            "builder_defaults_nsamples_seqlen": "recovered_from_source (offline builder 512x2048 for basis creation)",
            "note": "Surviving historical basis reused (hashed+copied). Compressed final_model will be rebuilt via main_eigen_attn --compress_only into isolation.",
        },
    }

    builder_lock = {
        "generated_at": now_iso(),
        "stage": "R2_UNIFORM_PREFLIGHT",
        "python": PYTHON_BIN,
        "paths": {
            "model": MODEL,
            "wikitext2": WIKITEXT,
            "historical_basis": str(HIST_BASIS),
            "isolated_basis": str(basis_dst),
            "builder_script": str(BUILDER),
            "main_script": str(MAIN),
            "out_dir": str(OUT),
        },
        "hashes": {
            "historical_basis": sha256_file(HIST_BASIS),
            "isolated_basis": sha256_file(basis_dst),
            "builder_script": file_hash_or_empty(BUILDER),
            "main_script": file_hash_or_empty(MAIN),
            "model_config": file_hash_or_empty(Path(MODEL) / "config.json"),
            "protocol": file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        },
        "uniform_operating_point": {"tag": "eb045", "error_budget": 0.045},
        "compress_args": compress_args,
        "expected_artifacts": [
            "eigen_basis_llama3_8b_wikitext2_eb0p045.pt",
            "LEGACY_UNIFORM_RANK_PLAN.csv",
            "final_model/",
            "final_model/low_rank_config.json",
            "LEGACY_UNIFORM_BASIS_MANIFEST.json",
            "LEGACY_UNIFORM_MODEL_MANIFEST.json",
        ],
    }
    atomic_write_json(OUT / "R2_UNIFORM_BUILD_LOCK.json", builder_lock)
    (OUT / "R2_UNIFORM_BUILD_LOCK.sha256").write_text(sha256_file(OUT / "R2_UNIFORM_BUILD_LOCK.json") + "\n")
    # also top-level convenience copies
    atomic_write_json(HERE / "R2_UNIFORM_BUILD_LOCK.json", builder_lock)
    (HERE / "R2_UNIFORM_BUILD_LOCK.sha256").write_text(sha256_file(HERE / "R2_UNIFORM_BUILD_LOCK.json") + "\n")

    manifest = {
        "generated_at": now_iso(),
        "basis_path": str(basis_dst),
        "basis_sha256": sha256_file(basis_dst),
        "source_historical_basis": str(HIST_BASIS),
        "source_sha256": sha256_file(HIST_BASIS),
        "n_layers": 32,
        "error_budget": 0.045,
        "materialization": "copy_historical_surviving_basis_into_isolation",
    }
    atomic_write_json(OUT / "LEGACY_UNIFORM_BASIS_MANIFEST.json", manifest)
    atomic_write_json(HERE / "LEGACY_UNIFORM_BASIS_MANIFEST.json", manifest)

    plan_md = [
        "# R2 Uniform Reconstruction Plan",
        "",
        f"- generated_at: `{now_iso()}`",
        "- Historical `final_model` missing → rebuild via legacy compress; basis still present → hash+copy into isolation.",
        "",
        "## Basis",
        f"- source: `{HIST_BASIS}`",
        f"- isolated: `{basis_dst}`",
        f"- sha256: `{manifest['basis_sha256']}`",
        "",
        "## Compress (main_eigen_attn.py)",
        "",
        "```bash",
        f"CUDA_VISIBLE_DEVICES=0 {PYTHON_BIN} -u {MAIN} \\",
        f"  --model {MODEL} --net Llama-3-8b --dtype bf16 \\",
        "  --error_budget 0.045 --avg_dim 4096 --seqlen 2048 --nsamples 64 --seed 2 \\",
        "  --calib_dataset c4 --compress_only --attn_implementation eager \\",
        f"  --eigen_basis_path {basis_dst} \\",
        f"  --save_dir {OUT / 'final_model'} \\",
        f"  --output_dir {OUT / 'compress_run'} \\",
        f"  --cache_dir {CACHE_DIR}",
        "```",
        "",
        "## Eval",
        "- After runtime smoke: 4-GPU PPL/zero-shot using instrumented runners with --cmp final_model.",
        "",
    ]
    (OUT / "R2_UNIFORM_RECONSTRUCTION_PLAN.md").write_text("\n".join(plan_md) + "\n")
    (HERE / "R2_UNIFORM_RECONSTRUCTION_PLAN.md").write_text("\n".join(plan_md) + "\n")

    payload = gate_payload(
        name="R2_UNIFORM_PREFLIGHT_PASS",
        passed=True,
        upstream_hashes={
            "basis": manifest["basis_sha256"],
            "dense_parity": file_hash_or_empty(HERE / "DENSE_PARITY_PASS.DONE"),
            "build_lock": sha256_file(OUT / "R2_UNIFORM_BUILD_LOCK.json"),
        },
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R2 preflight: historical basis materialized; compress args locked from logs",
    )
    write_pass_gate(OUT / "R2_UNIFORM_PREFLIGHT_PASS.DONE", payload)
    write_pass_gate(HERE / "R2_UNIFORM_PREFLIGHT_PASS.DONE", payload)
    print(json.dumps({"ok": True, "out": str(OUT), "basis_sha256": manifest["basis_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
