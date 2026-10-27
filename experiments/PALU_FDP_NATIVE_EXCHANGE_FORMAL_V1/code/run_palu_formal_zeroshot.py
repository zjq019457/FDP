import argparse
import json
import hashlib
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_palu_formal_quality as formal


TASKS = ['openbookqa', 'hellaswag', 'piqa', 'arc_easy', 'arc_challenge', 'winogrande']
OFFLINE_TASKS = Path('/diskF/jiaqizhang/a1804298693/offline_datasets')
LOCAL_TASKS = formal.BRANCH / 'local_zero_shot_tasks'
BINDINGS = {
    'openbookqa': 'openbookqa_localjsonl',
    'hellaswag': 'hellaswag_localjsonl',
    'piqa': 'piqa_localjsonl',
    'arc_easy': 'arc_easy_localjsonl',
    'arc_challenge': 'arc_challenge_localjsonl',
    'winogrande': 'winogrande_localjsonl',
}
BOUND_TASKS = list(BINDINGS.values())
sys.path.insert(0, str(OFFLINE_TASKS))

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def binding_record(paths):
    out = []
    for raw in paths:
        p = Path(raw)
        rec = {'path': str(p), 'exists': p.exists()}
        if p.exists():
            rec.update({'bytes': p.stat().st_size, 'sha256': file_sha256(p)})
            if p.suffix == '.parquet':
                try:
                    import pyarrow.parquet as pq
                    rec['samples'] = pq.read_metadata(p).num_rows
                except Exception:
                    rec['samples'] = None
            else:
                with p.open('rb') as f:
                    rec['samples'] = sum(1 for _ in f)
        out.append(rec)
    return out

def dataset_bindings():
    root = '/diskF/jiaqizhang/a1804298693/offline_datasets'
    return {
        'openbookqa': binding_record([
            '/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/datasets/allenai/openbookqa/main/test-00000-of-00001.parquet',
            '/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/datasets/allenai/openbookqa/main/validation-00000-of-00001.parquet',
        ]),
        'hellaswag': binding_record([
            root + '/hellaswag-train-dev/valid_labeled.jsonl',
        ]),
        'piqa': binding_record([root + '/piqa/dev_labeled.jsonl']),
        'arc_easy': binding_record([
            root + '/ai2_arc/ARC-Easy/ARC-Easy-Test.jsonl',
        ]),
        'arc_challenge': binding_record([
            root + '/ai2_arc/ARC-Challenge/ARC-Challenge-Test.jsonl',
        ]),
        'winogrande': binding_record([root + '/winogrande/dev.jsonl']),
    }


def selected_accuracy(task, metrics):
    for key in ('acc,none', 'acc_norm,none'):
        if key in metrics:
            return key, metrics[key]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, required=True, choices=[0.3, 0.5, 0.7])
    ap.add_argument('--method', required=True, choices=['uniform', 'fisher', 'fdp_prior', 'fdp_native'])
    args = ap.parse_args()
    outdir = formal.BRANCH / 'quality' / 'llama2' / f'ratio_{args.ratio:.1f}' / f'PALU_{args.method.upper()}'
    outdir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, schedule_path, schedule_doc, selection, _wiki_ids, _c4_ids = formal.load_runtime(args.ratio, args.method)
    import lm_eval
    from lm_eval import tasks as lm_tasks
    from lm_eval.models.huggingface import HFLM

    model.seqlen = 4096
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=1)
    raw = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=BOUND_TASKS,
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
    results = raw.get('results', {})
    selected = {}
    for task, bound_task in BINDINGS.items():
        key, value = selected_accuracy(bound_task, results.get(bound_task, {}))
        selected[task] = {'bound_task': bound_task, 'metric': key, 'accuracy': value}
    values = [x['accuracy'] for x in selected.values() if x['accuracy'] is not None]
    doc = {
        'status': 'FORMAL_ZERO_SHOT_COMPLETE',
        'campaign': 'PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1',
        'ratio': args.ratio,
        'method': f'PALU_{args.method.upper()}',
        'tasks': TASKS,
        'bound_tasks': BOUND_TASKS,
        'dataset_bindings': dataset_bindings(),
        'task_config_dir': str(LOCAL_TASKS),
        'num_fewshot': 0,
        'batch_size': 1,
        'limit': None,
        'seed': {'random_seed': 0, 'numpy_random_seed': 1234, 'torch_random_seed': 1234, 'fewshot_random_seed': 1234},
        'model': str(formal.MODEL),
        'schedule_path': str(schedule_path),
        'schedule_sha': schedule_doc['schedule_sha'],
        'resource': formal.resource(selection),
        'selected_accuracy': selected,
        'macro_accuracy': sum(values) / len(values) if len(values) == len(TASKS) else None,
        'lm_eval_results': results,
        'protocol': 'official PaLu run_lm_eval.py task list; common HFLM wrapper; GPU batch constrained to 1',
        'historical_results_modified': False,
    }
    (outdir / 'zeroshot.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': doc['status'], 'ratio': args.ratio, 'method': doc['method'], 'macro_accuracy': doc['macro_accuracy'], 'out': str(outdir / 'zeroshot.json')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
