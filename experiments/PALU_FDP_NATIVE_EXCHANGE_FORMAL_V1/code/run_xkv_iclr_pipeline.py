#!/usr/bin/env python3
"""Resumable single-GPU watchdog for XKV ICLR closure.

The only live stage implemented here is the official baseline launcher.  It
waits for GPU0, GPU1, or GPU3, never uses GPU2, and leaves formal exchange
quality pending until the baseline is complete and audited.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/diskE/jiaqizhang/XKV_FDP_NATIVE_EXCHANGE_FORMAL_V1")
STATE = ROOT / "xkv_pipeline_state.json"
LOG = ROOT / "xkv_pipeline.log"
RUNNER = ROOT / "code/run_xkv_single_gpu_official.sh"
FORMAL_RUNNER = ROOT / "code/run_xkv_formal_quality_single_gpu.sh"
EXP = Path("/home/bai-project/jiaqizhang/XKV_PAPER_REPRO_FDP_V1_RUNTIME_V2")
RULER = EXP / "ruler_data/llama-3/65536"
TASKS = ["niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2", "niah_multiquery", "niah_multivalue", "qa_1", "qa_2", "vt", "fwe"]


def log(message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def state_update(**changes: object) -> None:
    data = json.loads(STATE.read_text(encoding="utf-8"))
    data.update(changes)
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE)


def ruler_ready() -> bool:
    return all((RULER / task / "validation.jsonl").is_file() and (RULER / task / "validation.jsonl").stat().st_size > 0 for task in TASKS)


def active_official() -> bool:
    result = subprocess.run(["pgrep", "-af", "xkv_paper_entry.py"], text=True, capture_output=True)
    return bool(result.stdout.strip())


def gpu_rows() -> list[tuple[int, int, int]]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) == 3:
            rows.append((int(fields[0]), int(fields[1]), int(fields[2])))
    return rows


def select_gpu() -> int | None:
    values = {idx: (used, util) for idx, used, util in gpu_rows()}
    for idx, max_used in ((0, 2000), (1, 2000), (3, 25000)):
        used, util = values.get(idx, (999999, 100))
        if used <= max_used and util <= 5:
            return idx
    return None


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        state_update(gpu_policy={"preferred": [0, 1, 3], "excluded": [2], "single_gpu_only": True})
    while True:
        baseline_done = (EXP / "results/OFFICIAL_XKV_RULER_SINGLE_GPU_DONE").exists() and (EXP / "results/OFFICIAL_XKV_LONGBENCH_SINGLE_GPU_DONE").exists()
        formal_done = (ROOT / "results/formal/XKV_FORMAL_QUALITY_COMPLETE").exists()
        if formal_done:
            data = json.loads(STATE.read_text(encoding="utf-8"))
            nodes = data.get("nodes", {})
            nodes.update({"official_803x_baseline": "COMPLETE", "formal_ruler": "COMPLETE", "formal_longbench": "COMPLETE", "formal_exchange_quality": "COMPLETE", "formal_cached_ar": "PENDING_SEPARATE_NODE"})
            state_update(status="FORMAL_QUALITY_COMPLETE_WAITING_REPORTS", selected_gpu=None, official_baseline_pid=None, nodes=nodes)
            log("FORMAL_QUALITY_COMPLETE_WAITING_REPORTS")
            return
        if not ruler_ready():
            state_update(status="WAITING_RULER_DATA")
            log("WAITING_RULER_DATA")
            time.sleep(120)
            continue
        if active_official():
            data = json.loads(STATE.read_text(encoding="utf-8"))
            nodes = data.get("nodes", {})
            nodes["official_803x_baseline"] = "RUNNING"
            state_update(status="OFFICIAL_STAGE_ACTIVE", selected_gpu=data.get("selected_gpu"), failure_code=None, nodes=nodes)
            log("OFFICIAL_STAGE_ALREADY_ACTIVE")
            time.sleep(120)
            continue
        if baseline_done and not FORMAL_RUNNER.exists():
            state_update(status="FAILED_MISSING_FORMAL_RUNNER")
            log(f"FAILED_MISSING_FORMAL_RUNNER path={FORMAL_RUNNER}")
            return
        gpu = select_gpu()
        if gpu is None:
            state_update(status="WAITING_GPU", selected_gpu=None)
            log("WAITING_GPU policy=GPU0_or_GPU1_or_GPU3_only;GPU2_excluded")
            time.sleep(120)
            continue
        env = os.environ.copy()
        env["XKV_SELECTED_GPU"] = str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        data = json.loads(STATE.read_text(encoding="utf-8"))
        nodes = data.get("nodes", {})
        if baseline_done:
            nodes.update({"official_803x_baseline": "COMPLETE", "formal_ruler": "RUNNING", "formal_longbench": "RUNNING", "formal_exchange_quality": "RUNNING"})
            state_update(status="FORMAL_QUALITY_RUNNING", selected_gpu=gpu, failure_code=None, nodes=nodes)
            log(f"LAUNCH_FORMAL_QUALITY_SINGLE_GPU physical_gpu={gpu}")
            launch = FORMAL_RUNNER
        else:
            nodes.update({"official_803x_baseline": "RUNNING"})
            state_update(status="OFFICIAL_BASELINE_RUNNING", selected_gpu=gpu, failure_code=None, nodes=nodes)
            log(f"LAUNCH_OFFICIAL_SINGLE_GPU physical_gpu={gpu}")
            launch = RUNNER
        with LOG.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(["bash", str(launch)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        state_update(official_baseline_pid=process.pid)
        code = process.wait()
        if code == 0:
            log(f"PIPELINE_STAGE_EXITED code={code}")
        else:
            data = json.loads(STATE.read_text(encoding="utf-8"))
            nodes = data.get("nodes", {})
            nodes["formal_exchange_quality" if baseline_done else "official_803x_baseline"] = "FAILED"
            state_update(status="FORMAL_QUALITY_FAILED" if baseline_done else "OFFICIAL_BASELINE_FAILED", failure_code=code, nodes=nodes)
            log(f"PIPELINE_STAGE_FAILED code={code}")
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
