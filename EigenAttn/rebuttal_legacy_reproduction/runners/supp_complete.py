#!/usr/bin/env python3
"""Mark supplementary experiments complete after R8, R10, R11 pass."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import atomic_write_json, file_hash_or_empty, gate_payload, now_iso, sha256_file, write_pass_gate  # noqa: E402


def main() -> None:
    gates = {}
    for g in ["R8_SYSTEM_METRICS_PASS", "R10_KQV_ABLATION_PASS", "R11_ROBUSTNESS_PASS"]:
        gates[g] = (HERE / f"{g}.DONE").exists()

    all_ok = all(gates.values())
    supp = HERE / "supplementary_experiments"
    supp.mkdir(exist_ok=True)

    payload = gate_payload(
        name="SUPPLEMENTARY_EXPERIMENTS_COMPLETE",
        passed=all_ok,
        upstream_hashes={k: file_hash_or_empty(HERE / f"{k}.DONE") for k in gates},
        runner_hash=sha256_file(Path(__file__)),
        protocol_hash=file_hash_or_empty(HERE / "LEGACY_PROTOCOL_LOCK.json"),
        evaluation_hash=file_hash_or_empty(HERE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"),
        summary="All supplementary experiments complete" if all_ok else "Some supplementary experiments missing",
        extra={"gates": gates},
    )
    if all_ok:
        write_pass_gate(supp / "SUPPLEMENTARY_EXPERIMENTS_COMPLETE.DONE", payload)
        write_pass_gate(HERE / "SUPPLEMENTARY_EXPERIMENTS_COMPLETE.DONE", payload)
    else:
        atomic_write_json(supp / "SUPPLEMENTARY_INCOMPLETE.json", payload)
    print(json.dumps({"passed": all_ok, "gates": gates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
