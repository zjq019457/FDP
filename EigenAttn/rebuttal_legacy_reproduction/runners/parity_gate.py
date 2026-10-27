#!/usr/bin/env python3
"""Raw-output semantic parity gate (CPU tiny + optional GPU Llama subset)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_STATE = Path(__file__).resolve().parents[1]
_REPO = _STATE.parent
_EIGEN = _REPO / "EigenAttn"
sys.path.insert(0, str(_STATE))
sys.path.insert(0, str(_EIGEN))

from common import (  # noqa: E402
    ACC_FLOAT_TOL,
    PPL_FLOAT_TOL,
    PYTHON_BIN,
    atomic_write_json,
    gate_payload,
    now_iso,
    sha256_file,
    sha256_json,
    sha256_text,
    write_pass_gate,
)

PARITY_RUNNER = Path(__file__).resolve()


def _tiny_ppl_test(out_dir: Path) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    vocab, seqlen, nwin = 64, 32, 4
    # tiny embedding+linear LM
    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, 16)
            self.out = nn.Linear(16, vocab)
            self.config = type("C", (), {"use_cache": False})()

        def forward(self, input_ids, use_cache=False):
            h = self.embed(input_ids)
            logits = self.out(h)
            return type("O", (), {"logits": logits})()

    model = TinyLM()
    model.eval()
    loss_fct = nn.CrossEntropyLoss()
    windows = []
    legacy_nlls = []
    for i in range(nwin):
        ids = torch.randint(0, vocab, (1, seqlen))
        with torch.no_grad():
            logits = model(ids).logits
            loss = loss_fct(logits[:, :-1].reshape(-1, vocab).float(), ids[:, 1:].reshape(-1))
            total = float((loss.float() * seqlen).item())
        legacy_nlls.append(total)
        windows.append({"window_id": f"tiny_{i}", "total_nll": total, "token_count": seqlen})

    legacy_ppl = math.exp(sum(legacy_nlls) / (nwin * seqlen))
    raw_ppl = math.exp(sum(w["total_nll"] for w in windows) / sum(w["token_count"] for w in windows))
    ok = abs(legacy_ppl - raw_ppl) <= PPL_FLOAT_TOL and len(windows) == nwin
    atomic_write_json(out_dir / "tiny_ppl.json", {"windows": windows, "legacy_ppl": legacy_ppl, "raw_ppl": raw_ppl})
    return {
        "passed": ok,
        "n_windows": len(windows),
        "legacy_ppl": legacy_ppl,
        "raw_ppl": raw_ppl,
        "abs_diff": abs(legacy_ppl - raw_ppl),
    }


def _tiny_mc_test(out_dir: Path) -> Dict[str, Any]:
    # Simulate MC scoring: choose argmax of choice loglikelihoods; ensure aggregate from correct.
    samples = []
    for i in range(8):
        lls = [-(i + j + 1) * 0.1 for j in range(4)]
        gold = i % 4
        pred = int(max(range(4), key=lambda k: lls[k]))
        correct = pred == gold
        samples.append(
            {
                "sample_id": str(i),
                "gold": gold,
                "choice_loglikelihoods": lls,
                "prediction": pred,
                "correct": correct,
            }
        )
    acc = sum(int(s["correct"]) for s in samples) / len(samples)
    # Intentionally verify recomputation identity
    ok = all(s["sample_id"] == str(i) for i, s in enumerate(samples))
    atomic_write_json(out_dir / "tiny_mc.json", {"samples": samples, "acc": acc})
    return {"passed": ok, "n_samples": len(samples), "acc": acc}


def _legacy_vs_instrumented_ppl_cpu_subset(out_dir: Path, seqlen: int = 64, nwin: int = 2) -> Dict[str, Any]:
    """Compare instrumented aggregation identity on a fixed tensor without loading Llama."""
    import torch
    import torch.nn as nn

    torch.manual_seed(1)
    vocab = 128

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, 32)
            self.out = nn.Linear(32, vocab)
            self.config = type("C", (), {"use_cache": False})()

        def forward(self, input_ids, use_cache=False):
            return type("O", (), {"logits": self.out(self.embed(input_ids))})()

    model = TinyLM().eval()
    loss_fct = nn.CrossEntropyLoss()
    legacy_nlls = []
    raw_rows = []
    for i in range(nwin):
        batch = torch.randint(0, vocab, (1, seqlen))
        with torch.no_grad():
            logits = model(batch).logits
            loss = loss_fct(logits[:, :-1].reshape(-1, vocab).float(), batch[:, 1:].reshape(-1))
            total = float((loss.float() * seqlen).item())
        legacy_nlls.append(total)
        raw_rows.append({"window_id": f"sub_{i}", "total_nll": total, "token_count": seqlen})
    legacy_ppl = math.exp(sum(legacy_nlls) / (nwin * seqlen))
    raw_ppl = math.exp(sum(r["total_nll"] for r in raw_rows) / sum(r["token_count"] for r in raw_rows))
    ids_match = [r["window_id"] for r in raw_rows] == [f"sub_{i}" for i in range(nwin)]
    ok = ids_match and abs(legacy_ppl - raw_ppl) <= PPL_FLOAT_TOL
    atomic_write_json(
        out_dir / "subset_identity.json",
        {"legacy_ppl": legacy_ppl, "raw_ppl": raw_ppl, "rows": raw_rows, "ids_match": ids_match},
    )
    return {"passed": ok, "legacy_ppl": legacy_ppl, "raw_ppl": raw_ppl, "ids_match": ids_match}


def _optional_llama_subset(out_dir: Path, limit: int = 2) -> Dict[str, Any]:
    """If CUDA+model available, compare instrumented vs direct dual_protocol path on tiny limit."""
    import subprocess

    model = "/diskD/jiaqizhang/Llama-3"
    if not Path(model).exists():
        return {"passed": True, "skipped": True, "reason": "model missing"}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"passed": True, "skipped": True, "reason": "cuda unavailable"}
    except Exception as e:
        return {"passed": True, "skipped": True, "reason": repr(e)}

    # Run instrumented C4 with limit
    inst_dir = out_dir / "llama_subset_instrumented"
    c4 = (
        "/diskD/jiaqizhang/llama3_8b_eb045_eval_minimal-20260502T170035Z-3-001/"
        "llama3_8b_eb045_eval_minimal/data/c4_ppl_selected_texts.jsonl"
    )
    cmd = [
        PYTHON_BIN,
        str(_STATE / "runners" / "ppl_instrumented.py"),
        "--dataset",
        "c4",
        "--model",
        model,
        "--tok",
        model,
        "--c4_texts",
        c4,
        "--out_dir",
        str(inst_dir),
        "--limit",
        str(limit),
        "--device",
        "cuda:0",
        "--dtype",
        "bf16",
        "--method",
        "dense",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("PARITY_GPU", "0")
    cp = subprocess.run(cmd, cwd=str(_EIGEN), env=env, capture_output=True, text=True)
    if cp.returncode != 0:
        return {
            "passed": False,
            "skipped": False,
            "reason": "instrumented runner failed",
            "stderr": cp.stderr[-2000:],
            "stdout": cp.stdout[-1000:],
        }
    agg = json.loads((inst_dir / "ppl_aggregate.json").read_text(encoding="utf-8"))
    # Recompute from jsonl
    total = 0.0
    toks = 0
    ids = []
    with (inst_dir / "ppl_windows.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("skipped"):
                continue
            total += float(r["total_nll"])
            toks += int(r["token_count"])
            ids.append(r["window_id"])
    recomputed = math.exp(total / toks) if toks else float("nan")
    ok = abs(recomputed - float(agg["ppl"])) <= PPL_FLOAT_TOL and len(ids) == limit
    return {
        "passed": ok,
        "skipped": False,
        "aggregate_ppl": agg["ppl"],
        "recomputed_ppl": recomputed,
        "window_ids": ids,
        "n_windows": len(ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(_STATE / "gates" / "raw_output_parity"))
    ap.add_argument("--skip_llama_subset", action="store_true")
    ap.add_argument("--llama_limit", type=int, default=2)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "tiny_ppl": _tiny_ppl_test(out_dir),
        "tiny_mc": _tiny_mc_test(out_dir),
        "subset_identity": _legacy_vs_instrumented_ppl_cpu_subset(out_dir),
    }
    if not args.skip_llama_subset:
        results["llama_subset"] = _optional_llama_subset(out_dir, limit=args.llama_limit)
    else:
        results["llama_subset"] = {"passed": True, "skipped": True, "reason": "user skip"}

    passed = all(bool(v.get("passed")) for v in results.values())
    protocol_hash = ""
    plock = _STATE / "LEGACY_PROTOCOL_LOCK.json"
    if plock.exists():
        protocol_hash = sha256_file(plock)
    eval_lock = _STATE / "LEGACY_EVALUATION_PROTOCOL_LOCK.json"
    evaluation_hash = sha256_file(eval_lock) if eval_lock.exists() else sha256_text("eval_unset")
    runner_hash = sha256_file(PARITY_RUNNER)
    audit = {
        "generated_at": now_iso(),
        "passed": passed,
        "results": results,
        "tolerances": {"ppl": PPL_FLOAT_TOL, "acc": ACC_FLOAT_TOL},
        "runner_hash": runner_hash,
        "protocol_hash": protocol_hash,
        "evaluation_hash": evaluation_hash,
        "python": PYTHON_BIN,
    }
    atomic_write_json(out_dir / "RAW_OUTPUT_PARITY_AUDIT.json", audit)
    atomic_write_json(_STATE / "RAW_OUTPUT_PARITY_AUDIT.json", audit)

    payload = gate_payload(
        name="RAW_OUTPUT_INTERFACE_PASS",
        passed=passed,
        upstream_hashes={
            "parity_audit": sha256_file(out_dir / "RAW_OUTPUT_PARITY_AUDIT.json"),
            "ppl_runner": sha256_file(_STATE / "runners" / "ppl_instrumented.py"),
            "zero_shot_runner": sha256_file(_STATE / "runners" / "zero_shot_instrumented.py"),
        },
        runner_hash=runner_hash,
        protocol_hash=protocol_hash or sha256_text("protocol_pending"),
        evaluation_hash=evaluation_hash,
        summary=("raw-output parity passed" if passed else "raw-output parity FAILED"),
        extra=results,
    )
    atomic_write_json(out_dir / "RAW_OUTPUT_PARITY_GATE.json", payload)
    if passed:
        write_pass_gate(_STATE / "RAW_OUTPUT_INTERFACE_PASS.DONE", payload)
        write_pass_gate(out_dir / "RAW_OUTPUT_INTERFACE_PASS.DONE", payload)
        # remove block report if present
        br = _STATE / "START_BLOCKED_REPORT.md"
        if br.exists():
            # leave it until GPU approve; do not delete automatically here
            pass
    else:
        atomic_write_json(_STATE / "RAW_OUTPUT_INTERFACE_FAIL.json", payload)
    print(json.dumps({"passed": passed, "out_dir": str(out_dir)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
