#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1')
PYTHON = '/home/bai-project/anaconda3/envs/eigen/bin/python'
RATIOS = [0.7, 0.5, 0.3]
BASE_METHODS = ['fisher', 'fdp_prior', 'fdp_native']
GPU_ORDER = [2, 0, 3]
WORKER_PEAK_MIB = 28672
RESERVED_MARGIN_MIB = 4096


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def tag(ratio):
    return ('%.1f' % ratio).replace('.', 'p')


def result_path(ratio, method):
    return ROOT / 'official_zero_shot' / ('ratio_' + tag(ratio)) / method / 'zeroshot.json'


def phi_ready(ratio):
    d = ROOT / 'official_native_marginal' / ('ratio_' + tag(ratio))
    status = load(d / 'status.json', {}) or {}
    return status.get('status') == 'MARGINAL_PROFILE_COMPLETE' and (d / 'PALU_OFFICIAL_NATIVE_PHI_LOCK.json').exists()


def calibration_ready(ratio, method):
    if method == 'fisher':
        return (ROOT / 'results' / ('fisher_' + tag(ratio) + '_ratio_' + ('%.1f' % ratio) + '_fisher_uniform.json')).exists()
    return (ROOT / 'official_exchange_calibration' / ('ratio_' + tag(ratio)) / method / 'final_schedule.json').exists()


def random_ready(ratio, seed):
    return (ROOT / 'official_exchange_calibration' / ('ratio_' + tag(ratio)) / ('random_' + str(seed)) / 'final_schedule.json').exists()


def methods_for(ratio):
    if not phi_ready(ratio):
        return []
    methods = [m for m in BASE_METHODS if calibration_ready(ratio, m)]
    if abs(ratio - 0.5) < 1e-9:
        methods.extend('random_' + str(i) for i in range(3) if random_ready(ratio, i))
    return methods


def free_memory():
    try:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    except Exception:
        return {}
    out = {}
    for line in raw.splitlines():
        try:
            idx, free = [int(x.strip()) for x in line.split(',')]
            if idx != 1:
                out[idx] = free
        except Exception:
            pass
    return out


def active_worker():
    try:
        ps = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    except Exception:
        return False
    return 'run_official_zero_shot.py' in ps


def initial_state():
    return {'status': 'STARTING', 'ratios': {}, 'completed': [], 'pending': [], 'active': None, 'gpu1_excluded': True, 'foreign_jobs_interrupted': 0, 'worker_peak_mib': WORKER_PEAK_MIB, 'reserved_margin_mib': RESERVED_MARGIN_MIB, 'old_anchor_schema_mixed': False}


def main():
    state_path = ROOT / 'OFFICIAL_ZERO_SHOT_DISPATCH_STATE.json'
    state = load(state_path, initial_state()) or initial_state()
    while True:
        state['ratios'] = {str(ratio): {'phi': 'READY' if phi_ready(ratio) else 'WAIT_PHI', 'methods': methods_for(ratio)} for ratio in RATIOS}
        state['completed'] = []
        state['pending'] = []
        launched = False
        for ratio in RATIOS:
            ready = methods_for(ratio)
            for method in ready:
                path = result_path(ratio, method)
                if path.exists():
                    state['completed'].append('%.1f/%s' % (ratio, method))
                    continue
                state['pending'].append('%.1f/%s' % (ratio, method))
                if active_worker():
                    continue
                chosen = None
                for gpu in GPU_ORDER:
                    if free_memory().get(gpu, 0) >= WORKER_PEAK_MIB + RESERVED_MARGIN_MIB:
                        chosen = gpu
                        break
                if chosen is None:
                    state['status'] = 'WAIT_SAFE_GPU'
                    continue
                log = ROOT / 'logs' / ('official_zero_shot_' + tag(ratio) + '_' + method + '_gpu' + str(chosen) + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(chosen)
                env['PYTHONUNBUFFERED'] = '1'
                stream = log.open('ab', buffering=0)
                p = subprocess.Popen([PYTHON, str(ROOT / 'code' / 'run_official_zero_shot.py'), '--ratio', '%.1f' % ratio, '--method', method, '--device', 'cuda'], stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                state['active'] = {'ratio': ratio, 'method': method, 'gpu': chosen, 'pid': p.pid, 'log': str(log)}
                state['status'] = 'RUNNING'
                state['foreign_jobs_interrupted'] = 0
                state['gpu1_excluded'] = True
                write(state_path, state)
                launched = True
                break
            if launched:
                break
        if not launched:
            state['active'] = None
            if any(methods_for(r) for r in RATIOS) and all(result_path(r, m).exists() for r in RATIOS for m in methods_for(r)):
                state['status'] = 'ALL_READY_OFFICIAL_ZERO_SHOT_COMPLETE'
            else:
                state['status'] = 'WAIT_PHI_OR_SAFE_GPU'
            state['foreign_jobs_interrupted'] = 0
            state['gpu1_excluded'] = True
            write(state_path, state)
        time.sleep(60)


if __name__ == '__main__':
    main()
