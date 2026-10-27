#!/usr/bin/env python3
"""R3 FGeb045 compress from adaptive per-layer basis."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
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

OUT = artifact_dir("r3_fgeb045", link_under=HERE / "artifacts" / "r3_fgeb045")
MAIN = ROOT / "EigenAttn" / "main_eigen_attn.py"
MODEL = "/diskD/jiaqizhang/Llama-3"
CACHE = artifact_dir("HF_cache", link_under=ROOT / "EigenAttn" / "HF_cache_rebuttal")
BASIS = OUT / "eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt"


def _sync_final_model(save_dir: Path, run_dir: Path) -> Path:
    """Main may write under output_dir/final_model; ensure save_dir is complete."""
    candidates = [save_dir, run_dir / "final_model"]
    src = None
    for c in candidates:
        if (c / "low_rank_config.json").is_file() or (c / "low_rank_config.pt").is_file():
            src = c
            break
    if src is None:
        raise SystemExit(f"no low_rank_config under {candidates}")
    save_dir.mkdir(parents=True, exist_ok=True)
    if src.resolve() != save_dir.resolve():
        for p in src.iterdir():
            dst = save_dir / p.name
            if p.is_file():
                shutil.copy2(p, dst)
    # Prefer json from the richest source
    for c in candidates:
        j = c / "low_rank_config.json"
        if j.is_file():
            shutil.copy2(j, save_dir / "low_rank_config.json")
            pt = c / "low_rank_config.pt"
            if pt.is_file():
                shutil.copy2(pt, save_dir / "low_rank_config.pt")
            break
    return save_dir


def main() -> None:
    if not BASIS.is_file():
        raise SystemExit(f"missing adaptive basis: {BASIS}")
    save_dir = OUT / "final_model"
    run_dir = OUT / "compress_run"
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Ensure offline C4 calib cache present
    cache_file = CACHE / "dataloader_Llama_c4_64.cache"
    if not cache_file.is_file():
        hist = ROOT / "EigenAttn" / "HF_cache" / "dataloader_Llama_c4_64.cache"
        if hist.is_file():
            CACHE.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hist, cache_file)

    cmd = [
        PYTHON_BIN,
        "-u",
        str(MAIN),
        "--model",
        MODEL,
        "--net",
        "Llama-3-8b",
        "--dtype",
        "bf16",
        "--error_budget",
        "0.045",
        "--avg_dim",
        "4096",
        "--seqlen",
        "2048",
        "--nsamples",
        "64",
        "--seed",
        "2",
        "--calib_dataset",
        "c4",
        "--compress_only",
        "--attn_implementation",
        "eager",
        "--eigen_basis_path",
        str(BASIS),
        "--save_dir",
        str(save_dir),
        "--output_dir",
        str(run_dir),
        "--cache_dir",
        str(CACHE),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")
    log_out = OUT / "compress_stdout.log"
    log_err = OUT / "compress_stderr.log"
    with log_out.open("w", encoding="utf-8") as fo, log_err.open("w", encoding="utf-8") as fe:
        cp = subprocess.run(cmd, cwd=str(ROOT / "EigenAttn"), env=env, stdout=fo, stderr=fe, text=True)
    if cp.returncode != 0:
        raise SystemExit(f"R3 compress failed rc={cp.returncode}; see {log_err}")

    save_dir = _sync_final_model(save_dir, run_dir)
    lr_json = save_dir / "low_rank_config.json"
    manifest = {
        "generated_at": now_iso(),
        "save_dir": str(save_dir),
        "basis_sha256": sha256_file(BASIS),
        "low_rank_config_sha256": file_hash_or_empty(lr_json),
        "method": "FGeb045",
    }
    atomic_write_json(OUT / "LEGACY_FDP_MODEL_MANIFEST.json", manifest)
    gate = gate_payload(
        name="R3_FDP_BUILD_PASS",
        passed=True,
        upstream_hashes={"basis": sha256_file(BASIS), "model": file_hash_or_empty(lr_json)},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R3 FGeb045 compress complete",
    )
    write_pass_gate(OUT / "R3_FDP_BUILD_PASS.DONE", gate)
    print(json.dumps({"ok": True, "save_dir": str(save_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
