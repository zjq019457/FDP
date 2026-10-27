#!/usr/bin/env python3
"""R3 Step1 fragility diagnose: dense base vs rebuilt Uniform eb045."""
from __future__ import annotations

import json
import os
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
MODEL = "/diskD/jiaqizhang/Llama-3"
UNI = artifact_dir("r2_uniform_eb045", link_under=HERE / "artifacts" / "r2_uniform_eb045") / "final_model"
STEP1_SCRIPT = ROOT / "EigenAttn" / "step1_diagnose_llama.py"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    low_json = UNI / "low_rank_config.json"
    if not low_json.is_file():
        # recover from compress_run if needed
        alt = UNI.parent / "compress_run" / "final_model" / "low_rank_config.json"
        if alt.is_file():
            import shutil

            shutil.copy2(alt, low_json)
            pt = alt.with_suffix(".pt")
            if pt.is_file():
                shutil.copy2(pt, UNI / "low_rank_config.pt")
        else:
            raise SystemExit(f"uniform low_rank_config missing: {low_json}")

    out_json = OUT / "step1_llama3_8b_uniform_eb045.json"
    # Historical: seqlen=512, num_seqs=2, dtype=bf16, eager, single GPU.
    cmd = [
        PYTHON_BIN,
        "-u",
        str(STEP1_SCRIPT),
        "--base",
        MODEL,
        "--cmp",
        str(UNI),
        "--tok",
        MODEL,
        "--low_json",
        str(low_json),
        "--dtype",
        "bf16",
        "--seqlen",
        "512",
        "--num_seqs",
        "2",
        "--out",
        str(out_json),
        "--attn_implementation",
        "eager",
        "--gpu",
        "0",
        "--device",
        "cuda",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")
    log_out = OUT / "step1_stdout.log"
    log_err = OUT / "step1_stderr.log"
    with log_out.open("w", encoding="utf-8") as fo, log_err.open("w", encoding="utf-8") as fe:
        cp = subprocess.run(cmd, cwd=str(ROOT / "EigenAttn"), env=env, stdout=fo, stderr=fe, text=True)
    if cp.returncode != 0 or not out_json.is_file():
        raise SystemExit(f"step1 failed rc={cp.returncode}; see {log_err}")

    payload = {
        "generated_at": now_iso(),
        "out": str(out_json),
        "sha256": sha256_file(out_json),
        "uniform_model": str(UNI),
        "provenance": {
            "seqlen": "recovered_from_artifact",
            "num_seqs": "recovered_from_artifact",
            "source": "/home/bai-project/jiaqizhang/step1_out/step1_llama3_8b_eb045.json args",
        },
    }
    atomic_write_json(OUT / "R3_STEP1_MANIFEST.json", payload)
    gate = gate_payload(
        name="R3_STEP1_PASS",
        passed=True,
        upstream_hashes={"step1": payload["sha256"], "uniform_lr": file_hash_or_empty(low_json)},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R3 step1 fragility on rebuilt uniform",
    )
    write_pass_gate(OUT / "R3_STEP1_PASS.DONE", gate)
    print(json.dumps({"ok": True, "out": str(out_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
