#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path

REPO = Path("/home/bai-project/jiaqizhang/EigenAttn")
PY = "/home/bai-project/anaconda3/envs/eigen/bin/python"
MODEL = Path("/diskE/jiaqizhang/Qwen2.5-7B")
WIKI = Path("/home/bai-project/jiaqizhang/wikitext2")
BUILDER = REPO / "EigenAttn/tools/build_eigen_basis_llama_error_budget_offline.py"
SCRATCH = Path("/diskE/jiaqizhang/fdp125_final_campaign_v1_scratch/qwen_light_resource")
OUT = Path("/diskF/jiaqizhang/fdp125_final_campaign_v1/authority/QWEN_LIGHT_RESOURCE_SEARCH.json")
FULL_BYTES = 117_440_512
TARGET_RATIO = 0.816
TARGET_BYTES = round(FULL_BYTES * TARGET_RATIO)
STRICT_PCT = 0.15
MIN_FREE_BYTES = 32 * 1024**3


def now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def build_candidate(e: float, gpu: int) -> dict:
    from experimental.fdp_eb_orth_paper_v1.qwen_resource_engine import (
        realize_qwen_rank_schedule_from_basis_pt,
    )

    tag = f"e_{e:.8f}".replace(".", "p")
    work = SCRATCH / tag
    if work.exists():
        shutil.rmtree(work)
    basis_dir = work / "basis"
    basis_dir.mkdir(parents=True)
    eb_path = work / "per_layer_eb.json"
    atomic_json(
        eb_path,
        {
            "meta": {"resource_only": True, "base_e": e, "target_ratio": TARGET_RATIO},
            "error_budget_by_layer": [e] * 28,
        },
    )
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": f"{REPO}:{REPO / 'EigenAttn'}:{env.get('PYTHONPATH', '')}",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    cmd = [
        PY,
        "-u",
        str(BUILDER),
        "--model",
        str(MODEL),
        "--data_dir",
        str(WIKI),
        "--out_dir",
        str(basis_dir),
        "--out_prefix",
        f"fdp125_qwen_light_{tag}",
        "--per_layer_eb_json",
        str(eb_path),
        "--freeze_layer0_eb",
        "--layer0_eb",
        "0.06",
        "--nsamples",
        "64",
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
    log = work / "basis.log"
    started = now()
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"basis failed e={e} rc={proc.returncode}; log={log}")
    pts = sorted(basis_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise RuntimeError(f"basis missing e={e}")
    basis = pts[0]
    realized = realize_qwen_rank_schedule_from_basis_pt(basis)
    row = {
        "base_e": e,
        "started_at": started,
        "completed_at": now(),
        "B_live": int(realized["B_live"]),
        "full_resource_bytes": FULL_BYTES,
        "retained_ratio": float(realized["B_live"]) / FULL_BYTES,
        "target_bytes": TARGET_BYTES,
        "resource_gap_pct": (float(realized["B_live"]) - TARGET_BYTES) / TARGET_BYTES * 100.0,
        "rank_sha": realized["rank_sha"],
        "ranks": realized["ranks"],
        "basis_sha": sha(basis),
        "builder": str(BUILDER),
        "builder_sha": sha(BUILDER),
        "command": cmd,
    }
    # Streaming cleanup: keep compact rank/provenance only.
    shutil.rmtree(work)
    return row


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    if shutil.disk_usage("/diskE").free < MIN_FREE_BYTES:
        raise RuntimeError("DISPATCH_PAUSED_LOW_DISK: diskE free below 32 GiB")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "RUNNING",
        "started_at": now(),
        "selection_policy": "RESOURCE_ONLY_NO_QUALITY",
        "model": str(MODEL),
        "regime": "LIGHT",
        "target_ratio": TARGET_RATIO,
        "target_bytes": TARGET_BYTES,
        "full_resource_bytes": FULL_BYTES,
        "tested": [],
    }
    atomic_json(OUT, state)
    lo, hi = 0.015, 0.018
    candidates = [lo, hi]
    for _ in range(7):
        if not candidates:
            candidates.append((lo + hi) / 2.0)
        e = candidates.pop(0)
        if any(abs(e - x["base_e"]) < 1e-10 for x in state["tested"]):
            continue
        try:
            row = build_candidate(e, args.gpu)
        except Exception as exc:
            state.update(
                {
                    "status": "FAILED_ENGINEERING_NEEDS_RETRY",
                    "failed_at": now(),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            atomic_json(OUT, state)
            shutil.rmtree(SCRATCH, ignore_errors=True)
            raise
        state["tested"].append(row)
        state["updated_at"] = now()
        atomic_json(OUT, state)
        if abs(row["resource_gap_pct"]) <= STRICT_PCT:
            break
        if row["retained_ratio"] > TARGET_RATIO:
            lo = e
        else:
            hi = e
        candidates.append((lo + hi) / 2.0)
    best = min(state["tested"], key=lambda x: abs(x["resource_gap_pct"]))
    state.update(
        {
            "status": "FROZEN_STRICT" if abs(best["resource_gap_pct"]) <= STRICT_PCT else "FROZEN_CLOSEST_FRONTIER",
            "completed_at": now(),
            "selected": best,
            "no_quality_observed": True,
            "scratch_cleaned": True,
        }
    )
    atomic_json(OUT, state)
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
