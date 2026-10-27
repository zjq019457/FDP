#!/usr/bin/env python3
"""R2 Uniform compress: rebuild final_model via legacy main_eigen_attn --compress_only."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import PYTHON_BIN, atomic_write_json, file_hash_or_empty, now_iso, sha256_file, write_pass_gate, gate_payload, artifact_dir  # noqa: E402

OUT = artifact_dir("r2_uniform_eb045", link_under=HERE / "artifacts" / "r2_uniform_eb045")
MAIN = ROOT / "EigenAttn" / "main_eigen_attn.py"
MODEL = "/diskD/jiaqizhang/Llama-3"


def kv_bytes_from_ranks(ranks_kq, ranks_v, seqlen: int = 2048, bytes_per: int = 2) -> dict:
    # Llama3-8B GQA: store K and V cache low-rank dims roughly proportional to ranks.
    # Report both dense baseline and realized using rank sums (bf16=2 bytes).
    # Dense KV: 32 layers * 2 (K,V) * num_kv_heads * head_dim * seqlen * bytes
    n_layers = 32
    n_kv = 8
    head_dim = 128
    dense = n_layers * 2 * n_kv * head_dim * seqlen * bytes_per
    # Realized: use rank_kq and rank_v as effective stored dims for K and V respectively.
    real = 0
    for rk, rv in zip(ranks_kq, ranks_v):
        real += int(rk) * seqlen * bytes_per  # K
        real += int(rv) * seqlen * bytes_per  # V
    return {
        "seqlen": seqlen,
        "bytes_per_elem": bytes_per,
        "dense_kv_bytes": dense,
        "realized_kv_bytes": real,
        "retained_ratio": (real / dense) if dense else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_if_done", action="store_true")
    args = ap.parse_args()

    lock_path = OUT / "R2_UNIFORM_BUILD_LOCK.json"
    if not lock_path.exists():
        raise SystemExit("run r2_uniform_preflight.py first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    ca = lock["compress_args"]
    save_dir = Path(ca["save_dir"])
    out_dir = Path(ca["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    lr_json = save_dir / "low_rank_config.json"
    if args.skip_if_done and lr_json.exists() and (OUT / "R2_UNIFORM_BUILD_PASS.DONE").exists():
        print(json.dumps({"skipped": True, "save_dir": str(save_dir)}))
        return

    cmd = [
        PYTHON_BIN,
        "-u",
        str(MAIN),
        "--model",
        ca["model"],
        "--net",
        ca["net"],
        "--dtype",
        ca["dtype"],
        "--error_budget",
        str(ca["error_budget"]),
        "--avg_dim",
        str(ca["avg_dim"]),
        "--seqlen",
        str(ca["seqlen"]),
        "--nsamples",
        str(ca["nsamples"]),
        "--seed",
        str(ca["seed"]),
        "--calib_dataset",
        ca["calib_dataset"],
        "--compress_only",
        "--attn_implementation",
        ca["attn_implementation"],
        "--eigen_basis_path",
        ca["eigen_basis_path"],
        "--save_dir",
        str(save_dir),
        "--output_dir",
        str(out_dir),
        "--cache_dir",
        ca["cache_dir"],
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")
    # single GPU policy enforced by orchestrator CUDA_VISIBLE_DEVICES
    log_out = OUT / "compress_stdout.log"
    log_err = OUT / "compress_stderr.log"
    with log_out.open("w", encoding="utf-8") as fo, log_err.open("w", encoding="utf-8") as fe:
        cp = subprocess.run(cmd, cwd=str(ROOT / "EigenAttn"), env=env, stdout=fo, stderr=fe, text=True)
    if cp.returncode != 0:
        raise SystemExit(f"compress failed rc={cp.returncode}; see {log_err}")

    # Prefer low_rank_config.json; also accept .pt
    ranks_kq, ranks_v = [], []
    if lr_json.exists():
        cfg = json.loads(lr_json.read_text(encoding="utf-8"))
        # expected list of [kq,v] or dict
        if isinstance(cfg, list):
            for row in cfg:
                ranks_kq.append(int(row[0]))
                ranks_v.append(int(row[1]))
        elif isinstance(cfg, dict) and "low_rank_config" in cfg:
            for row in cfg["low_rank_config"]:
                ranks_kq.append(int(row[0]))
                ranks_v.append(int(row[1]))
    elif (save_dir / "low_rank_config.pt").exists():
        import torch

        pt = torch.load(save_dir / "low_rank_config.pt", map_location="cpu")
        for row in pt:
            ranks_kq.append(int(row[0]))
            ranks_v.append(int(row[1]))
    else:
        raise SystemExit(f"no low_rank_config under {save_dir}")

    # Update rank plan with realized ranks
    import csv

    rank_csv = OUT / "LEGACY_UNIFORM_RANK_PLAN.csv"
    rows = []
    for i, (kq, v) in enumerate(zip(ranks_kq, ranks_v)):
        rows.append({"layer": i, "rank_kq": kq, "rank_v": v})
    with rank_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "rank_kq", "rank_v"])
        w.writeheader()
        w.writerows(rows)
    # top-level copy
    import shutil

    shutil.copy2(rank_csv, HERE / "LEGACY_UNIFORM_RANK_PLAN.csv")

    mem = kv_bytes_from_ranks(ranks_kq, ranks_v, seqlen=2048, bytes_per=2)
    atomic_write_json(OUT / "LEGACY_UNIFORM_MEMORY_ACCOUNTING.json", mem)
    atomic_write_json(HERE / "LEGACY_UNIFORM_MEMORY_ACCOUNTING.json", mem)

    model_manifest = {
        "generated_at": now_iso(),
        "save_dir": str(save_dir),
        "low_rank_config_sha256": file_hash_or_empty(lr_json if lr_json.exists() else save_dir / "low_rank_config.pt"),
        "basis_sha256": file_hash_or_empty(Path(ca["eigen_basis_path"])),
        "n_layers": len(ranks_kq),
        "ranks_kq": ranks_kq,
        "ranks_v": ranks_v,
        "memory": mem,
        "compress_command": cmd,
        "returncode": 0,
    }
    atomic_write_json(OUT / "LEGACY_UNIFORM_MODEL_MANIFEST.json", model_manifest)
    atomic_write_json(HERE / "LEGACY_UNIFORM_MODEL_MANIFEST.json", model_manifest)

    audit = [
        "# R2 Uniform Build Audit",
        "",
        f"- generated_at: `{now_iso()}`",
        f"- save_dir: `{save_dir}`",
        f"- realized_kv_bytes: `{mem['realized_kv_bytes']}`",
        f"- dense_kv_bytes: `{mem['dense_kv_bytes']}`",
        f"- retained_ratio: `{mem['retained_ratio']}`",
        f"- stdout: `{log_out}`",
        f"- stderr: `{log_err}`",
        "",
    ]
    (OUT / "R2_UNIFORM_BUILD_AUDIT.md").write_text("\n".join(audit) + "\n")
    (HERE / "R2_UNIFORM_BUILD_AUDIT.md").write_text("\n".join(audit) + "\n")

    payload = gate_payload(
        name="R2_UNIFORM_BUILD_PASS",
        passed=True,
        upstream_hashes={
            "model_manifest": sha256_file(OUT / "LEGACY_UNIFORM_MODEL_MANIFEST.json"),
            "basis": file_hash_or_empty(Path(ca["eigen_basis_path"])),
            "build_lock": sha256_file(lock_path),
        },
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary=f"uniform compress done retained_ratio={mem['retained_ratio']}",
        extra=mem,
    )
    write_pass_gate(OUT / "R2_UNIFORM_BUILD_PASS.DONE", payload)
    write_pass_gate(HERE / "R2_UNIFORM_BUILD_PASS.DONE", payload)
    print(json.dumps({"ok": True, "memory": mem, "save_dir": str(save_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
