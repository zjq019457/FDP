#!/usr/bin/env python3
"""Instrumentation-only launcher: runs a frozen script unchanged via runpy (same argv, __file__, cwd, sys.path[0])
and, at exit, CUDA-synchronizes and records torch peak allocated/reserved GPU memory. No algorithm code is touched.
usage: cost_launch.py <torch_peak_out.json> <script.py> [script args...]"""
import atexit, json, os, runpy, sys, time

out, script = sys.argv[1], os.path.abspath(sys.argv[2])
T0 = time.time()


def _dump():
    rec = {"script": script, "t_start": T0, "t_end": time.time()}
    try:
        import torch
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                torch.cuda.synchronize(d)
            rec["torch_peak_allocated_bytes"] = max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count()))
            rec["torch_peak_reserved_bytes"] = max(torch.cuda.max_memory_reserved(d) for d in range(torch.cuda.device_count()))
    except Exception as e:  # never fail the measured run because of instrumentation
        rec["torch_peak_error"] = repr(e)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)


atexit.register(_dump)
sys.argv = [script] + sys.argv[3:]
sys.path[0] = os.path.dirname(script)
runpy.run_path(script, run_name="__main__")
