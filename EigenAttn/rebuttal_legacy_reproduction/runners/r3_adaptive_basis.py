#!/usr/bin/env python3
"""R3 adaptive / per-layer eigen basis from per-layer EB JSON (WikiText-2, 512x2048)."""
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
BUILDER = ROOT / "EigenAttn" / "tools" / "build_eigen_basis_llama_error_budget_offline.py"
MODEL = "/diskD/jiaqizhang/Llama-3"
WIKITEXT = "/home/bai-project/jiaqizhang/wikitext2"
PREFIX = "eigen_basis_llama3_8b_wikitext2_frag"


def main() -> None:
    pl = OUT / "llama3_8b_wikitext2_frag_eb045_from_uniform.json"
    if not pl.is_file():
        raise SystemExit(f"missing per_layer json: {pl}")
    basis_dir = OUT / "adaptive_basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    expected = basis_dir / f"{PREFIX}_perlayer_eb.pt"
    cmd = [
        PYTHON_BIN,
        "-u",
        str(BUILDER),
        "--model",
        MODEL,
        "--data_dir",
        WIKITEXT,
        "--out_dir",
        str(basis_dir),
        "--out_prefix",
        PREFIX,
        "--per_layer_eb_json",
        str(pl),
        "--freeze_layer0_eb",
        "--layer0_eb",
        "0.045",
        "--nsamples",
        "512",
        "--seqlen",
        "2048",
        "--dtype",
        "bf16",
        "--attn_implementation",
        "eager",
        "--device",
        "cuda:0",
        "--seed",
        "2",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")
    log_out = OUT / "adaptive_basis_stdout.log"
    log_err = OUT / "adaptive_basis_stderr.log"
    with log_out.open("w", encoding="utf-8") as fo, log_err.open("w", encoding="utf-8") as fe:
        cp = subprocess.run(cmd, cwd=str(ROOT / "EigenAttn"), env=env, stdout=fo, stderr=fe, text=True)
    if cp.returncode != 0 or not expected.is_file():
        raise SystemExit(f"adaptive basis failed rc={cp.returncode}; see {log_err}")

    # Isolate copy at OUT root for compress path stability
    isolated = OUT / f"{PREFIX}_perlayer_eb.pt"
    import shutil

    shutil.copy2(expected, isolated)
    payload = {
        "generated_at": now_iso(),
        "basis": str(isolated),
        "sha256": sha256_file(isolated),
        "builder": str(BUILDER),
        "builder_sha256": sha256_file(BUILDER),
        "per_layer_eb_json": str(pl),
        "nsamples": 512,
        "seqlen": 2048,
        "freeze_layer0_eb": True,
        "layer0_eb": 0.045,
        "provenance": {
            "builder_flags": "recovered_from_artifact",
            "source_report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        },
    }
    atomic_write_json(OUT / "R3_ADAPTIVE_BASIS_MANIFEST.json", payload)
    gate = gate_payload(
        name="R3_ADAPTIVE_BASIS_PASS",
        passed=True,
        upstream_hashes={"basis": payload["sha256"], "per_layer": file_hash_or_empty(pl)},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R3 adaptive per-layer eigen basis built",
    )
    write_pass_gate(OUT / "R3_ADAPTIVE_BASIS_PASS.DONE", gate)
    print(json.dumps({"ok": True, "basis": str(isolated)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
