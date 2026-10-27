#!/usr/bin/env python3
"""
MASTER PARALLEL MANAGER: LOO & CONTROLS (CLEAN PROJECTION)
==========================================================
Distributes unique states across GPU 0, 1, 2, 3.
Uses clean_eval_worker.py to guarantee distinct weights projection per config.
Safe scratch on /diskE with max concurrency = 3 (keeping >50GB headroom at all times).
"""
import os, sys, json, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures

WORKSPACE   = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR  = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"
WORKER_PY   = FORMAL_DIR / "clean_eval_worker.py"
SCRATCH_DIR = Path("/diskE/jiaqizhang/core_models_scratch")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

with open(FORMAL_DIR / "LOO_CONTROLS_AUTHORITY.json") as f:
    all_records = json.load(f)

# Priority: Light LOO -> Medium LOO -> Aggressive LOO -> Controls
def sort_key(r):
    tier_order = {"light": 0, "medium": 1, "aggressive": 2}
    group_order = {"LOO": 0, "CONTROL": 1}
    return (tier_order.get(r["tier"], 9), group_order.get(r["group"], 9), r["config_name"])

all_records = sorted(all_records, key=sort_key)

states_to_run = {}
for r in all_records:
    sha = r["sha"]
    if (PER_WIN_DIR / sha / "STATE_EVAL_DONE.json").exists():
        continue
    if sha not in states_to_run:
        states_to_run[sha] = r

items = list(states_to_run.items())
print(f"[{utc()}] Master clean queue: {len(items)} unique states to evaluate.")

if not items:
    print(f"[{utc()}] All states already done!")
    sys.exit(0)

def worker_job(sha, info, gpu_id):
    cmd = [
        sys.executable, "-u", str(WORKER_PY),
        "--sha", sha,
        "--gpu", str(gpu_id),
        "--scratch", str(SCRATCH_DIR)
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", errors="ignore")
    for line in out.splitlines():
        if "SUCCESS" in line or "FAILED" in line or "Building" in line:
            print(f"[{utc()}][GPU {gpu_id}] {line}")
    return p.returncode == 0

# Concurrency = 3 across GPU 3, 2, 0 (GPU 1 excluded or reserved for safety)
# 3 concurrent models * 28GB = 84GB < 135GB free on /diskE (super safe)
gpu_pool = [3, 2, 0]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futs = {}
    for i, (sha, info) in enumerate(items):
        gpu = gpu_pool[i % len(gpu_pool)]
        futs[executor.submit(worker_job, sha, info, gpu)] = (sha, info)

    for fut in concurrent.futures.as_completed(futs):
        sha, info = futs[fut]
        try:
            ok = fut.result()
            if not ok:
                print(f"[{utc()}] WARNING: Task failed for {info['tier']} {info['config_name']} ({sha[:12]})")
        except Exception as e:
            print(f"[{utc()}] EXCEPTION {sha[:12]}: {e}")

print(f"[{utc()}] ALL CLEAN LOO AND CONTROLS JOBS COMPLETED.")
