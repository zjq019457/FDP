#!/usr/bin/env python3
"""Calibration-only runner for XKV_FDP_NATIVE_EXCHANGE_FORMAL_V1.

This evaluates the shared native exchange schedules on the frozen train
manifest.  It never touches RULER, LongBench, or the formal cached-AR test.
The T choice is frozen from this calibration output only.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any


FORMAL = Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1')
MANIFEST = FORMAL / 'exchange_calibration/XKV_EXCHANGE_VALIDATION_MANIFEST.json'
MARGINAL_LOCK = FORMAL / 'XKV_NATIVE_MARGINAL_LOCK.json'
OUT_JSON = FORMAL / 'XKV_EXCHANGE_CALIBRATION.json'
OUT_CSV = FORMAL / 'XKV_EXCHANGE_DECISIONS.csv'
LOG = FORMAL / 'XKV_EXCHANGE_CALIBRATION.log'
MODULE_ROOT = Path('/home/bai-project/jiaqizhang')
ENGINE_ROOT = FORMAL / 'code'
BOOTSTRAPS = 10_000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float('nan')
    position = (len(ordered) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def paired_bootstrap(delta_rows: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    by_dataset: dict[str, list[float]] = {'WikiText-2': [], 'C4': []}
    for row in delta_rows:
        by_dataset[row['dataset']].append(float(row['delta_mean_nll']))
    if any(len(v) != 8 for v in by_dataset.values()):
        raise RuntimeError('calibration bootstrap requires exactly 8 rows per dataset')
    rng = random.Random(seed)
    samples = []
    for _ in range(BOOTSTRAPS):
        wiki = [rng.choice(by_dataset['WikiText-2']) for _ in range(8)]
        c4 = [rng.choice(by_dataset['C4']) for _ in range(8)]
        samples.append(0.5 * (statistics.fmean(wiki) + statistics.fmean(c4)))
    return {
        'n_bootstrap': BOOTSTRAPS,
        'seed': seed,
        'mean_delta_J': statistics.fmean(samples),
        'ci95_delta_J': [percentile(samples, 0.025), percentile(samples, 0.975)],
        'fraction_FDP_better': sum(x < 0 for x in [r['delta_mean_nll'] for r in delta_rows]) / len(delta_rows),
        'median_paired_delta': statistics.median(r['delta_mean_nll'] for r in delta_rows),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = {}
    for dataset in ('WikiText-2', 'C4'):
        vals = [float(row['mean_nll']) for row in rows if row['dataset'] == dataset]
        by_dataset[dataset] = {
            'windows': len(vals),
            'mean_token_nll': statistics.fmean(vals),
            'ppl': math.exp(statistics.fmean(vals)),
        }
    j = 0.5 * (by_dataset['WikiText-2']['mean_token_nll'] + by_dataset['C4']['mean_token_nll'])
    return {'dataset': by_dataset, 'J': j, 'ppl_from_J': math.exp(j)}


def main() -> None:
    if OUT_JSON.exists() or OUT_CSV.exists():
        raise SystemExit('refusing to overwrite existing exchange calibration outputs')
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '1':
        raise SystemExit('GPU1 is forbidden')
    manifest = load(MANIFEST)
    lock = load(MARGINAL_LOCK)
    if manifest.get('frozen') is not True:
        raise SystemExit('exchange validation manifest is not frozen')
    if manifest.get('sequence_length') != 4160 or manifest.get('prefix_tokens') != 4096 or manifest.get('continuation_tokens') != 64:
        raise SystemExit('unexpected exchange manifest geometry')
    if manifest.get('counts') != {'WikiText-2': 8, 'C4': 8}:
        raise SystemExit('unexpected exchange manifest counts')
    if lock.get('locked') is not True or lock.get('q_sha') != '6f3b59993abbffbd932825b20ff036197c5be6ff63523ab46e94281bc24eecb1':
        raise SystemExit('native marginal lock/q gate failed')

    sys.path.insert(0, str(ENGINE_ROOT))
    sys.path.insert(0, str(MODULE_ROOT))
    import xkv_native_exchange_engine as engine
    import xkv_native_marginal as runtime

    q_group = [float(row['q_group']) for row in lock['per_group']]
    phi_group = [float(row['phi_xkv']) for row in lock['per_group']]
    signals: dict[str, tuple[list[float], int | None]] = {
        'q_prior': (q_group, None),
        'native_phi': (phi_group, None),
        'random_seed_0': (engine.random_signal(0), 0),
        'random_seed_1': (engine.random_signal(1), 1),
        'random_seed_2': (engine.random_signal(2), 2),
    }
    schedules = []
    for name, (signal, seed) in signals.items():
        for schedule in engine.nested_schedules(name, signal, seed=seed):
            schedules.append(schedule)

    # T=0 is identical across all signals and is evaluated once, then linked
    # into each signal arm.  Every nonzero signal/T schedule is independent.
    unique = {}
    for schedule in schedules:
        key = (schedule.T, schedule.rank_k, schedule.rank_v)
        unique.setdefault(key, schedule)

    runtime.setup_runtime()
    import torch
    rows = manifest['rows']
    model, _ = runtime.load_model([384] * 8, [576] * 8)
    case_by_key = {}
    started = time.time()
    for index, (key, schedule) in enumerate(unique.items(), start=1):
        label = f'{schedule.signal_name}_T{schedule.T}'
        rk = list(schedule.rank_k)
        rv = list(schedule.rank_v)
        if engine.total_factor_bytes(rk, rv) != engine.total_factor_bytes([384] * 8, [576] * 8):
            raise RuntimeError(f'factor-memory mismatch in {label}')
        model.kv_compress_config = runtime.build_cfg(rk, rv)
        config = runtime.runtime_config(model, rk, rv)
        if config['requested_effective_config_equal'] is not True:
            raise RuntimeError(f'rank realization mismatch in {label}')
        print(json.dumps({'event': 'case_start', 'index': index, 'total_unique_cases': len(unique), 'case': label, 'T': schedule.T, 'donors': list(schedule.donors), 'receivers': list(schedule.receivers)}), flush=True)
        case_rows = []
        case_start = time.time()
        for row_index, row in enumerate(rows, start=1):
            scored = runtime.score_window(model, row)
            case_rows.append(scored)
            print(json.dumps({'event': 'window_done', 'case': label, 'index': row_index, 'document_id': row['document_id'], 'mean_nll': scored['mean_nll']}), flush=True)
        case_by_key[key] = {
            'signal': schedule.signal_name,
            'seed': schedule.seed,
            'T': schedule.T,
            'donors': list(schedule.donors),
            'receivers': list(schedule.receivers),
            'rank_k': rk,
            'rank_v': rv,
            'total_native_rank': sum(rk) + sum(rv),
            'factor_bytes': engine.total_factor_bytes(rk, rv),
            'runtime_config': config,
            'aggregate': aggregate(case_rows),
            'windows': case_rows,
            'elapsed_seconds': time.time() - case_start,
        }
        print(json.dumps({'event': 'case_done', 'case': label, 'J': case_by_key[key]['aggregate']['J'], 'elapsed_seconds': case_by_key[key]['elapsed_seconds']}), flush=True)
        del case_rows
        gc.collect()
        torch.cuda.empty_cache()

    control_key = (0, tuple([384] * 8), tuple([576] * 8))
    control = case_by_key[control_key]
    decisions = []
    schedule_rows = []
    for name, (signal, seed) in signals.items():
        arm = []
        for schedule in engine.nested_schedules(name, signal, seed=seed):
            key = (schedule.T, schedule.rank_k, schedule.rank_v)
            case = case_by_key[key]
            deltas = []
            for control_row, arm_row in zip(control['windows'], case['windows']):
                deltas.append({
                    'dataset': arm_row['dataset'],
                    'document_id': arm_row['document_id'],
                    'control_mean_nll': control_row['mean_nll'],
                    'arm_mean_nll': arm_row['mean_nll'],
                    'delta_mean_nll': arm_row['mean_nll'] - control_row['mean_nll'],
                })
            delta_j = case['aggregate']['J'] - control['aggregate']['J']
            boot = paired_bootstrap(deltas)
            record = {
                'signal': name,
                'seed': seed,
                'T': schedule.T,
                'donors': list(schedule.donors),
                'receivers': list(schedule.receivers),
                'rank_k': list(schedule.rank_k),
                'rank_v': list(schedule.rank_v),
                'total_native_rank': case['total_native_rank'],
                'factor_bytes': case['factor_bytes'],
                'resource_match_exact': case['factor_bytes'] == control['factor_bytes'],
                'J_control': control['aggregate']['J'],
                'J_arm': case['aggregate']['J'],
                'delta_J': delta_j,
                'bootstrap': boot,
                'windows': deltas,
            }
            arm.append(record)
            schedule_rows.append(record)
        selected = min(arm, key=lambda row: (row['J_arm'], row['T']))
        decisions.append({
            'signal': name,
            'seed': seed,
            'selected_T': selected['T'],
            'selection_rule': 'minimum calibration J over nested T=0..4; tie-break lower T',
            'selected_delta_J': selected['delta_J'],
            'selected_bootstrap': selected['bootstrap'],
            'all_T': [{'T': row['T'], 'J_arm': row['J_arm'], 'delta_J': row['delta_J'], 'ci95_delta_J': row['bootstrap']['ci95_delta_J']} for row in arm],
        })

    doc = {
        'schema': 'XKV_FDP_NATIVE_EXCHANGE_CALIBRATION_V1',
        'status': 'COMPLETE',
        'formal_quality_started': False,
        'model': '/diskF/jiaqizhang/Llama-3.1',
        'runtime_fixed': True,
        'gpu_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'manifest': {'path': str(MANIFEST), 'sha256': sha(MANIFEST), 'split': 'train', 'counts': manifest['counts'], 'sequence_length': manifest['sequence_length'], 'prefix_tokens': 4096, 'continuation_tokens': 64, 'objective': '0.5 WikiText-2 mean continuation NLL + 0.5 C4 mean continuation NLL'},
        'native_marginal_lock': {'path': str(MARGINAL_LOCK), 'sha256': sha(MARGINAL_LOCK), 'q_sha': lock['q_sha'], 'source_result_sha256': lock['source_result_sha256']},
        'signals': {'q_prior': q_group, 'native_phi': phi_group, 'random_seed_0': signals['random_seed_0'][0], 'random_seed_1': signals['random_seed_1'][0], 'random_seed_2': signals['random_seed_2'][0]},
        'control': control,
        'schedule_records': schedule_rows,
        'decisions': decisions,
        'engine': {'module': str(ENGINE_ROOT / 'xkv_native_exchange_engine.py'), 'resource_formula': 'official xKV grouped factor-memory formula', 'group_unit': '8 native four-layer groups', 'quantum': {'delta_k': 32, 'delta_v': 48}, 'nested_T': [0, 1, 2, 3, 4], 'iterative_reprofiling': False},
        'old_results_modified': False,
        'formal_manifests_modified': False,
        'foreign_jobs_interrupted': 0,
        'elapsed_seconds': time.time() - started,
    }
    atomic_json(OUT_JSON, doc)
    with OUT_CSV.open('w', newline='', encoding='utf-8') as handle:
        fields = ['signal', 'seed', 'T', 'donors', 'receivers', 'J_control', 'J_arm', 'delta_J', 'ci95_low', 'ci95_high', 'fraction_FDP_better', 'median_paired_delta', 'factor_bytes', 'resource_match_exact']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in schedule_rows:
            writer.writerow({
                'signal': row['signal'], 'seed': row['seed'], 'T': row['T'],
                'donors': ';'.join(map(str, row['donors'])), 'receivers': ';'.join(map(str, row['receivers'])),
                'J_control': row['J_control'], 'J_arm': row['J_arm'], 'delta_J': row['delta_J'],
                'ci95_low': row['bootstrap']['ci95_delta_J'][0], 'ci95_high': row['bootstrap']['ci95_delta_J'][1],
                'fraction_FDP_better': row['bootstrap']['fraction_FDP_better'], 'median_paired_delta': row['bootstrap']['median_paired_delta'],
                'factor_bytes': row['factor_bytes'], 'resource_match_exact': row['resource_match_exact'],
            })
    print(json.dumps({'status': 'COMPLETE', 'out': str(OUT_JSON), 'decisions': decisions}, indent=2), flush=True)


if __name__ == '__main__':
    main()
