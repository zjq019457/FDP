#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path('/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1')
OLD = Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1')
OFFLINE_TASKS = Path('/diskF/jiaqizhang/a1804298693/offline_datasets')
LOCAL_TASKS = OLD / 'local_zero_shot_tasks'
MODEL = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf')
TASKS = ['openbookqa', 'hellaswag', 'piqa', 'arc_easy', 'arc_challenge', 'winogrande']
BINDINGS = {
    'openbookqa': 'openbookqa_localjsonl',
    'hellaswag': 'hellaswag_localjsonl',
    'piqa': 'piqa_localjsonl',
    'arc_easy': 'arc_easy_localjsonl',
    'arc_challenge': 'arc_challenge_localjsonl',
    'winogrande': 'winogrande_localjsonl',
}

sys.path.insert(0, str(ROOT / 'code'))
import run_official_formal_quality as official


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare_cache_env():
    cache = ROOT / 'zero_shot_cache'
    paths = {
        'HF_HOME': cache / 'hf_home',
        'HF_DATASETS_CACHE': cache / 'datasets',
        'TRANSFORMERS_CACHE': cache / 'transformers',
        'TMPDIR': cache / 'tmp',
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['TEMP'] = str(paths['TMPDIR'])
    os.environ['TMP'] = str(paths['TMPDIR'])
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    return {key: str(path) for key, path in paths.items()}


def selected_accuracy(metrics):
    for key in ('acc,none', 'acc_norm,none'):
        if key in metrics:
            return key, metrics[key]
    return None, None


def schedule_for(ratio, method):
    selection, path = official.schedule(ratio, method)
    schedule_doc = load(path)
    if not selection or any(len(values) != 8 for values in selection.values()):
        raise RuntimeError('official action-space assertion failed: expected eight native groups per K/V module')
    return selection, path, schedule_doc


def dataset_binding_audit():
    path = OLD / 'audit' / 'PALU_ZERO_SHOT_LOCAL_DATASET_BINDING.json'
    if not path.exists():
        raise FileNotFoundError(str(path))
    doc = load(path)
    return str(path), sha(path), doc.get('dataset_bindings', doc.get('bindings', {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, required=True, choices=[0.3, 0.5, 0.7])
    parser.add_argument('--method', required=True, choices=['fisher', 'fdp_prior', 'fdp_native', 'random_0', 'random_1', 'random_2'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cache_env = prepare_cache_env()
    tag = ('%.1f' % args.ratio).replace('.', 'p')
    outdir = ROOT / 'official_zero_shot' / ('ratio_' + tag) / args.method
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / 'zeroshot.json'
    if result_path.exists():
        print(json.dumps({'status': 'ALREADY_COMPLETE', 'path': str(result_path)}), flush=True)
        return

    selection, schedule_path, schedule_doc = schedule_for(args.ratio, args.method)
    fisher_selection, fisher_path, _ = schedule_for(args.ratio, 'fisher')
    resource_control = official.resource(fisher_selection)
    resource_method = official.resource(selection)
    if resource_method != resource_control:
        raise RuntimeError('official-native resource mismatch vs Fisher: %s != %s' % (resource_method, resource_control))

    action_space = load(ROOT / 'PALU_OFFICIAL_NATIVE_ACTION_SPACE.json')
    protocol_lock = ROOT / 'PALU_OFFICIAL_PROTOCOL_LOCK.json'
    q_sha = load(protocol_lock).get('frozen_q_sha')
    binding_path, binding_sha, bindings = dataset_binding_audit()

    model, tokenizer, _raw, _parents, _factors, _module_class = official.build(args.device, selection)
    model.seqlen = 4096
    from lm_eval import simple_evaluate
    from lm_eval import tasks as lm_tasks
    from lm_eval.models.huggingface import HFLM

    lm_model = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=1)
    evaluated = simple_evaluate(
        model=lm_model,
        tasks=list(BINDINGS.values()),
        num_fewshot=0,
        batch_size=1,
        limit=None,
        log_samples=False,
        task_manager=lm_tasks.TaskManager(include_path=[str(OFFLINE_TASKS), str(LOCAL_TASKS)]),
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    results = evaluated.get('results', {})
    selected = {}
    for task, bound_task in BINDINGS.items():
        metric, value = selected_accuracy(results.get(bound_task, {}))
        selected[task] = {'bound_task': bound_task, 'metric': metric, 'accuracy': value}
    values = [row['accuracy'] for row in selected.values() if row['accuracy'] is not None]
    if len(values) != len(TASKS):
        raise RuntimeError('official zero-shot did not return all six task scores')

    doc = {
        'status': 'OFFICIAL_ZERO_SHOT_COMPLETE',
        'campaign': 'PALU_OFFICIAL_PROTOCOL_REPRO_V1',
        'lineage': 'OFFICIAL_NATIVE_8_GROUP',
        'old_anchor_schema_zero_shot_used': False,
        'ratio': args.ratio,
        'method': 'PaLu Automatic (Fisher)' if args.method == 'fisher' else ('FDP Prior' if args.method == 'fdp_prior' else ('FDP Native' if args.method == 'fdp_native' else args.method)),
        'method_key': args.method,
        'tasks': TASKS,
        'bound_tasks': list(BINDINGS.values()),
        'num_fewshot': 0,
        'batch_size': 1,
        'limit': None,
        'seed': {'random_seed': 0, 'numpy_random_seed': 1234, 'torch_random_seed': 1234, 'fewshot_random_seed': 1234},
        'model': str(MODEL),
        'schedule_path': str(schedule_path),
        'schedule_sha256': sha(schedule_path),
        'schedule_status': schedule_doc.get('status'),
        'fisher_reference_path': str(fisher_path),
        'fisher_reference_sha256': sha(fisher_path),
        'resource': resource_method,
        'resource_control_fisher': resource_control,
        'resource_match': 'EXACT',
        'selected_accuracy': selected,
        'macro_accuracy': sum(values) / len(values),
        'lm_eval_results': results,
        'protocol': 'official PaLu six-task zero-shot protocol; common HFLM wrapper; batch_size=1; local offline bindings',
        'official_protocol_lock_path': str(protocol_lock),
        'official_protocol_lock_sha256': sha(protocol_lock),
        'official_action_space_path': str(ROOT / 'PALU_OFFICIAL_NATIVE_ACTION_SPACE.json'),
        'official_action_space_sha256': action_space['action_space_sha256'],
        'q_sha256': q_sha,
        'dataset_binding_audit_path': binding_path,
        'dataset_binding_audit_sha256': binding_sha,
        'dataset_bindings': bindings,
        'cache_environment': cache_env,
        'historical_results_modified': False,
        'old_anchor_schema_results_mixed': False,
    }
    result_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': doc['status'], 'ratio': args.ratio, 'method': doc['method'], 'macro_accuracy': doc['macro_accuracy'], 'resource': resource_method, 'out': str(result_path)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
