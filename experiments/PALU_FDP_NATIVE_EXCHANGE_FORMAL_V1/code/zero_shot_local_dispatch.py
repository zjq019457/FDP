import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1")
PYTHON = "/home/bai-project/jiaqizhang/.venv/bin/python"
SCRIPT = ROOT / "code" / "run_palu_formal_zeroshot.py"
LOGDIR = ROOT / "logs"
STATE = ROOT / "PALU_ZERO_SHOT_LOCAL_DISPATCH_STATE.json"
METHODS = ["uniform", "fisher", "fdp_prior", "fdp_native"]
WORKER_PEAK_MIB = 28672
RESERVED_MARGIN_MIB = 12288
GPU_EXCLUDE = {1}

def gpu_snapshot():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    out = []
    for line in raw.splitlines():
        try:
            idx, free, util = [int(x.strip()) for x in line.split(",")]
        except Exception:
            continue
        out.append((idx, free, util))
    return out

def safe_gpu():
    required = WORKER_PEAK_MIB + RESERVED_MARGIN_MIB
    candidates = [(free, idx) for idx, free, _ in gpu_snapshot() if idx not in GPU_EXCLUDE and free >= required]
    return max(candidates)[1] if candidates else None

def active(fragment):
    try:
        raw = subprocess.check_output(["ps", "-eo", "args"], text=True)
    except Exception:
        return False
    return fragment in raw

def official_ready():
    base = Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1")
    required = [
        base / "official_exchange_calibration/ratio_0p7/fdp_prior/status.json",
        base / "official_exchange_calibration/ratio_0p7/fdp_native/status.json",
        base / "official_formal_quality/ratio_0p7/fdp_prior/status.json",
        base / "official_formal_quality/ratio_0p7/fdp_native/status.json",
    ]
    try:
        return all(json.loads(p.read_text()).get("status", "").endswith("COMPLETE") for p in required)
    except Exception:
        return False

def result_path(method):
    return ROOT / "quality" / "llama2" / "ratio_0.5" / ("PALU_" + method.upper()) / "zeroshot.json"

def write_state(status, **extra):
    doc = {
        "status": status,
        "campaign": "PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1",
        "methods": METHODS,
        "worker_peak_mib": WORKER_PEAK_MIB,
        "reserved_margin_mib": RESERVED_MARGIN_MIB,
        "gpu1_excluded": True,
        "foreign_jobs_interrupted": 0,
    }
    doc.update(extra)
    STATE.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

def main():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    while True:
        missing = [m for m in METHODS if not result_path(m).exists()]
        if not missing:
            write_state("COMPLETE", completed_methods=METHODS, pending_methods=[])
            return
        if not official_ready():
            write_state("WAIT_OFFICIAL_0P7_FORMAL", completed_methods=[m for m in METHODS if result_path(m).exists()], pending_methods=missing)
            time.sleep(60)
            continue
        if active("run_palu_formal_zeroshot.py"):
            write_state("WAIT_EXISTING_ZERO_SHOT_WORKER", completed_methods=[m for m in METHODS if result_path(m).exists()], pending_methods=missing)
            time.sleep(60)
            continue
        gpu = safe_gpu()
        if gpu is None:
            write_state("WAIT_MEMORY_SAFE_GPU", completed_methods=[m for m in METHODS if result_path(m).exists()], pending_methods=missing)
            time.sleep(60)
            continue
        method = missing[0]
        log = LOGDIR / ("FORMAL_ZS_0P5_" + method.upper() + "_LOCAL.log")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["HF_DATASETS_OFFLINE"] = "1"
        env["HF_HOME"] = "/diskE/jiaqizhang/PALU_ZERO_SHOT_CACHE/hf_home"
        env["HF_DATASETS_CACHE"] = "/diskE/jiaqizhang/PALU_ZERO_SHOT_CACHE/datasets"
        env['TRANSFORMERS_CACHE'] = '/diskE/jiaqizhang/PALU_ZERO_SHOT_CACHE/transformers'
        env['TMPDIR'] = '/diskE/jiaqizhang/PALU_ZERO_SHOT_CACHE/tmp'
        env['TEMP'] = env['TMPDIR']
        env['TMP'] = env['TMPDIR']
        with log.open("ab", buffering=0) as stream:
            p = subprocess.Popen([PYTHON, str(SCRIPT), "--ratio", "0.5", "--method", method], stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        write_state("RUNNING", method=method, gpu=gpu, pid=p.pid, completed_methods=[m for m in METHODS if result_path(m).exists()], pending_methods=missing)
        while p.poll() is None:
            time.sleep(30)
        if p.returncode != 0:
            write_state("FAILED_ENGINEERING_OR_RUNTIME", method=method, returncode=p.returncode, completed_methods=[m for m in METHODS if result_path(m).exists()], pending_methods=missing)
            time.sleep(60)
        else:
            write_state("METHOD_COMPLETE", method=method, completed_methods=[m for m in METHODS if result_path(m).exists()], pending_methods=[m for m in METHODS if not result_path(m).exists()])

if __name__ == "__main__":
    main()
