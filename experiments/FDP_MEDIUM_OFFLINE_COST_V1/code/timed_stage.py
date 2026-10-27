#!/usr/bin/env python3
"""Run one pipeline stage as a subprocess and measure it.
- wall time (monotonic), every stdout line timestamped (for sub-phase markers)
- peak CPU RSS of the whole process tree (psutil, 0.5 s sampling)
- peak per-process GPU memory from NVML, restricted to PIDs of this process tree (all GPUs)
- torch peak allocated/reserved (from cost_launch.py, written after CUDA synchronize)
- disk usage of the given output paths after the stage
"""
import json, os, re, subprocess, sys, threading, time
from pathlib import Path
import psutil, pynvml


def run_stage(name, cmd, env, cwd, outdir, markers=(), disk_paths=(), torch_peak=None):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    logp = outdir / f"{name}.log"
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    peak = {"rss": 0, "gpu": 0, "gpu_per_device": {}}
    t0w = time.time(); t0 = time.monotonic()
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, errors="replace")
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                root = psutil.Process(p.pid); procs = [root] + root.children(recursive=True)
                pids = {x.pid for x in procs}
                rss = 0
                for x in procs:
                    try: rss += x.memory_info().rss
                    except psutil.Error: pass
                peak["rss"] = max(peak["rss"], rss)
                tot = 0
                for i, h in enumerate(handles):
                    used = 0
                    for gp in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                        if gp.pid in pids and gp.usedGpuMemory: used += gp.usedGpuMemory
                    if used:
                        peak["gpu_per_device"][i] = max(peak["gpu_per_device"].get(i, 0), used)
                    tot += used
                peak["gpu"] = max(peak["gpu"], tot)
            except psutil.Error:
                pass
            time.sleep(0.5)

    th = threading.Thread(target=sampler, daemon=True); th.start()
    marks = {m: None for m in markers}
    with open(logp, "w") as lf:
        for line in p.stdout:
            t = time.monotonic() - t0
            lf.write(f"[{t:10.3f}] {line}")
            for m in markers:
                if marks[m] is None and re.search(m, line): marks[m] = t
    rc = p.wait(); wall = time.monotonic() - t0
    stop.set(); th.join(timeout=5)
    rec = {"stage": name, "cmd": cmd, "cwd": str(cwd), "rc": rc, "start_epoch": t0w, "wall_s": wall,
           "peak_cpu_rss_bytes": peak["rss"], "peak_nvml_gpu_bytes_process_tree": peak["gpu"],
           "peak_nvml_gpu_bytes_per_device": peak["gpu_per_device"], "marker_offsets_s": marks, "log": str(logp)}
    if torch_peak and Path(torch_peak).exists():
        rec["torch"] = json.loads(Path(torch_peak).read_text())
    du = {}
    for d in disk_paths:
        d = Path(d)
        du[str(d)] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
    rec["disk_bytes_after_stage"] = du
    (outdir / f"{name}.timing.json").write_text(json.dumps(rec, indent=1))
    if rc != 0:
        raise RuntimeError(f"stage {name} failed rc={rc}; see {logp}")
    return rec
