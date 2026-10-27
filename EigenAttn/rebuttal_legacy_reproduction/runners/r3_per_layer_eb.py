#!/usr/bin/env python3
"""R3 per-layer EB map from Step1 (lambda_eb=1.2, normalize=mean, freeze L0)."""
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
SCRIPT = ROOT / "EigenAttn" / "per_layer_eb_from_step1.py"


def main() -> None:
    step1 = OUT / "step1_llama3_8b_uniform_eb045.json"
    if not step1.is_file():
        raise SystemExit(f"missing step1: {step1}")
    out_json = OUT / "llama3_8b_wikitext2_frag_eb045_from_uniform.json"
    cmd = [
        PYTHON_BIN,
        "-u",
        str(SCRIPT),
        "--step1",
        str(step1),
        "--out",
        str(out_json),
        "--eb_base",
        "0.045",
        "--lambda_eb",
        "1.2",
        "--normalize",
        "mean",
        "--freeze_layers",
        "0",
        "--wz",
        "1.0",
        "--ws",
        "1.0",
        "--wv",
        "1.0",
        "--wa",
        "1.0",
        "--wh",
        "1.0",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "EigenAttn") + os.pathsep + env.get("PYTHONPATH", "")
    cp = subprocess.run(cmd, cwd=str(ROOT / "EigenAttn"), env=env, capture_output=True, text=True)
    (OUT / "per_layer_stdout.log").write_text(cp.stdout or "", encoding="utf-8")
    (OUT / "per_layer_stderr.log").write_text(cp.stderr or "", encoding="utf-8")
    if cp.returncode != 0 or not out_json.is_file():
        raise SystemExit(f"per_layer_eb failed rc={cp.returncode}")

    meta = json.loads(out_json.read_text(encoding="utf-8")).get("meta", {})
    payload = {
        "generated_at": now_iso(),
        "out": str(out_json),
        "sha256": sha256_file(out_json),
        "meta": meta,
        "protocol": {
            "eb_base": 0.045,
            "lambda_eb": 1.2,
            "normalize": "mean",
            "freeze_layers": [0],
            "weights": {"wz": 1.0, "ws": 1.0, "wv": 1.0, "wa": 1.0, "wh": 1.0},
            "provenance": "recovered_from_artifact",
        },
    }
    atomic_write_json(OUT / "R3_PER_LAYER_EB_MANIFEST.json", payload)
    gate = gate_payload(
        name="R3_PER_LAYER_EB_PASS",
        passed=True,
        upstream_hashes={"per_layer": payload["sha256"], "step1": file_hash_or_empty(step1)},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="R3 per-layer EB λ=1.2 mean-normalize freeze L0",
    )
    write_pass_gate(OUT / "R3_PER_LAYER_EB_PASS.DONE", gate)
    print(json.dumps({"ok": True, "out": str(out_json), "meta": meta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
