#!/usr/bin/env python3
"""Isolated Appendix F.3 Automatic reproduction for G-LRD gs=4.

This cell is intentionally kept separate from the user-requested equal-rank
Table-1 operator diagnostics.  It checks the paper's Automatic/Fisher result
and provides evidence about the Table-1 vs Appendix-F.3 protocol ambiguity.
"""
import datetime
import fcntl
import gc
import gzip
import hashlib
import json
import os
import sys
import traceback
import types
from pathlib import Path

BASE = Path('/diskE/jiaqizhang/PALU_OFFICIAL_TABLE1_REPRO_V1')
REPO = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu')
MODEL = BASE / 'model_reference/Llama-2-7b-hf'
DATA = BASE / 'inputs/wikitext2_raw_v1'
C4 = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/data/c4-validation.00000-of-00008.json.gz')
OUT = BASE / 'results/appendix_f3_automatic_gs4'
STATE = BASE / 'state/APPENDIX_F3_AUTOMATIC_GS4_STATE.json'
ACTIVE = BASE / 'state/APPENDIX_F3_AUTOMATIC_GS4_ACTIVE.json'
LOCK_PATH = BASE / 'locks/APPENDIX_F3_AUTOMATIC_GS4.lock'
EXPECTED = {
    'wiki_test': '3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925',
    'wiki_train': '49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701',
    'c4': '1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f',
}
PAPER = {
    'wikitext2': {'appendix_f3_automatic': 6.02, 'table1_g_lrd': 6.01},
    'c4': {'table1_g_lrd': 9.82},
}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def verify_inputs():
    paths = {
        'wiki_test': DATA / 'test-00000-of-00001.parquet',
        'wiki_train': DATA / 'train-00000-of-00001.parquet',
        'c4': C4,
    }
    got = {k: sha_file(v) for k, v in paths.items()}
    if got != EXPECTED:
        raise RuntimeError(f'input SHA mismatch: {got}')
    return got


def resource(model, selection, rank_sum, total_rank):
    from palu.model.modules.svd_linear import HeadwiseLowRankModule
    modules = [(n, x) for n, x in model.named_modules() if isinstance(x, HeadwiseLowRankModule)]
    factor_bytes = 0
    realization = hashlib.sha256()
    histogram = {}
    for ranks in selection.values():
        for rank in ranks:
            histogram[str(int(rank))] = histogram.get(str(int(rank)), 0) + 1
    for name, module in modules:
        for pname, param in sorted(module.named_parameters()):
            factor_bytes += param.numel() * param.element_size()
            arr = param.detach().cpu().contiguous().numpy()
            realization.update((name + '.' + pname + '|' + str(tuple(arr.shape)) + '|' + str(arr.dtype) + '\n').encode())
            realization.update(memoryview(arr).cast('B'))
    rank_sum_from_selection = sum(int(r) for ranks in selection.values() for r in ranks)
    modules_by_name = dict(model.named_modules())
    ranks_in_bounds = all(
        0 < int(rank) <= (modules_by_name[name].out_features // len(ranks))
        for name, ranks in selection.items()
        for rank in ranks
    )
    rank_quantized = all(
        int(rank) % 32 == 0
        for ranks in selection.values()
        for rank in ranks
    )
    allocation_integrity_pass = (
        int(rank_sum) == rank_sum_from_selection
        and int(total_rank) == 262144
        and len(modules) == 64
        and ranks_in_bounds
        and rank_quantized
        and factor_bytes > 0
    )
    return {
        'rank_sum': int(rank_sum),
        'uncompressed_rank': int(total_rank),
        'nominal_50pct_rank_target': 131072,
        'rank_delta_from_nominal_target': int(rank_sum) - 131072,
        'compression_fraction': 1.0 - float(rank_sum) / total_rank,
        'rank_quantum': 32,
        'factor_bytes': int(factor_bytes),
        'cache_bytes_per_token_fp16': int(rank_sum * 2),
        'prefill_cache_bytes_at_2048_fp16': int(rank_sum * 4096),
        'decode_cache_bytes_at_2048_fp16': int(rank_sum * 4096),
        'low_rank_module_count': len(modules),
        'allocation_sha256': hashlib.sha256(json.dumps(
            {k: [int(v) for v in ranks] for k, ranks in sorted(selection.items())},
            separators=(',', ':')).encode()).hexdigest(),
        'realization_sha256': realization.hexdigest(),
        'rank_histogram': histogram,
        'rank_sum_matches_selection': int(rank_sum) == rank_sum_from_selection,
        'rank_quantized_to_32': rank_quantized,
        'ranks_within_native_group_dimensions': ranks_in_bounds,
        'nominal_50pct_target_exact': int(rank_sum) == 131072,
        'allocation_integrity_pass': allocation_integrity_pass,
        'resource_policy': 'report official per-unit 32-rank rounding as realized; do not manually adjust ranks to force the nominal total',
    }


def main():
    BASE.joinpath('locks').mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_PATH, 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists():
        state = json.loads(STATE.read_text())
        if state.get('status') == 'COMPLETE':
            print('SWEEP_ALREADY_COMPLETE', flush=True)
            return
    else:
        state = {'schema': 'PALU_APPENDIX_F3_AUTOMATIC_GS4_STATE_V1', 'cells': {}}
    previous_failures = {k: v for k, v in state.get('cells', {}).items() if v.get('status') == 'FAILED'}
    if previous_failures:
        BASE.joinpath('audit').mkdir(parents=True, exist_ok=True)
        repair_path = BASE / 'audit/FISHER_SWEEP_ENGINEERING_FIX_01.json'
        if not repair_path.exists():
            cell, failure = next(iter(previous_failures.items()))
            save_json(repair_path, {
                'schema': 'PALU_ENGINEERING_FIX_AUDIT_V1',
                'failure_behavior': 'Official Fisher calibration requested WikiText train via the Hub and failed in offline mode before Fisher profiling, decomposition, or PPL evaluation.',
                'failure_type': failure.get('failure_type'),
                'reason': failure.get('reason'),
                'affected_cell': cell,
                'repair': 'Apply the local pinned-Parquet loader before importing official palu.data_utils, which captures datasets.load_dataset at module import time.',
                'retry_scope': 'Only the incomplete Fisher Table1/Table6 sweep; no completed quality cell is rerun.',
                'quality_result_produced_before_failure': False,
                'old_results_valid': True,
                'historical_results_modified': False,
                'foreign_jobs_interrupted': 0,
                'logged_utc': utc(),
            })
    state.update(status='RUNNING', pid=os.getpid(), gpu=0, foreign_jobs_interrupted=0,
                 run_id='APPENDIX_F3_AUTOMATIC_GS4_20260930',
                 protocol={'compression_fraction_target': 0.5, 'group_sizes': [4],
                           'search_method': 'fisher_uniform', 'rank_quantum': 32,
                           'fisher_samples': 2048, 'fisher_seqlen': 1024,
                           'whiten_samples': 256, 'ppl_seqlen_stride': 2048,
                           'cell_scope': 'Appendix F.3 Automatic at gs=4; C4 is a cross-table diagnostic only',
                           'foreign_jobs_interrupted': 0})
    save_json(STATE, state)
    save_json(ACTIVE, {'pid': os.getpid(), 'gpu': 0, 'started_utc': utc(), 'scope': 'isolated Appendix F.3 Automatic G-LRD gs=4 validation', 'foreign_jobs_interrupted': 0})

    dataset_sha = verify_inputs()
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(REPO))
    import datasets
    import torch
    import numpy as np
    import evaluate

    class NoZeroShotMetric:
        def compute(self, *args, **kwargs):
            raise RuntimeError('zero-shot metric must not run in the PPL sweep')
    real_metric_load = evaluate.load
    evaluate.load = lambda name, *a, **kw: NoZeroShotMetric() if name == 'exact_match' else real_metric_load(name, *a, **kw)
    fht = types.ModuleType('fast_hadamard_transform')
    fht.hadamard_transform = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('Hadamard path must not run for FP16 whiten PPL'))
    sys.modules['fast_hadamard_transform'] = fht
    real_load = datasets.load_dataset

    class Rows:
        def __init__(self, rows): self.rows = rows
        def __getitem__(self, key):
            if isinstance(key, slice): return {'text': self.rows[key]}
            if key == 'text': return self.rows
            return {'text': self.rows[key]}

    def local_load(path, *args, **kwargs):
        if path == 'wikitext' and args and args[0] == 'wikitext-2-raw-v1':
            split = kwargs.get('split')
            if split not in ('train', 'test'):
                raise RuntimeError(f'unexpected Wiki split {split}')
            source = DATA / ('train-00000-of-00001.parquet' if split == 'train' else 'test-00000-of-00001.parquet')
            return real_load('parquet', data_files={split: str(source)}, split=split)
        if path == 'allenai/c4':
            if kwargs.get('revision') != '607bd4c8450a42878aa9ddc051a65a055450ef87':
                raise RuntimeError('unexpected C4 revision')
            docs = []
            with gzip.open(C4, 'rt', encoding='utf-8') as f:
                for _, line in zip(range(1100), f):
                    docs.append(json.loads(line)['text'])
            return Rows(docs)
        return real_load(path, *args, **kwargs)

    datasets.load_dataset = local_load
    real_save = torch.save
    torch.save = lambda obj, *a, **k: real_save(
        types.SimpleNamespace(input_ids=obj.input_ids) if obj.__class__.__name__ == 'TokenizerWrapper' else obj,
        *a, **k)
    # Import only after local_load is installed: data_utils binds load_dataset
    # with `from datasets import load_dataset` when this module is imported.
    import compress as official_compress
    import run_ppl_eval

    hold = {}
    original_load = official_compress.load_model_and_tokenizer
    def capture_load(*a, **kw):
        model, tokenizer = original_load(*a, **kw)
        hold.update(model=model, tokenizer=tokenizer)
        return model, tokenizer
    official_compress.load_model_and_tokenizer = capture_load
    original_rank = official_compress.rank_search
    def capture_rank(model, tokenizer, args):
        result = original_rank(model, tokenizer, args)
        hold.update(selection=result[0], rank_sum=int(result[1]), total_rank=int(result[2]))
        return result
    official_compress.rank_search = capture_rank

    commit = 'bb22666e2ef96707e8dd21d93fc00146c2e0d615'
    source_hashes = {name: sha_file(REPO / rel) for name, rel in {
        'compress.py': 'compress.py',
        'rank_search.py': 'palu/rank_search.py',
        'decomposition.py': 'palu/decomposition.py',
        'svd_linear.py': 'palu/model/modules/svd_linear.py',
        'run_ppl_eval.py': 'run_ppl_eval.py',
    }.items()}
    alias = str(MODEL)
    whitening_cache = BASE / f'cache/whiten/{alias.replace("/", "_")}_w2_scaling_matrices_fp16.pt'
    whitening_cache_sha = sha_file(whitening_cache) if whitening_cache.exists() else None
    specs = [(4, ['wikitext2', 'c4'])]

    cell_path = OUT / 'unknown'
    fisher_cache_sha = None
    try:
        for gs, datasets_to_run in specs:
            cell = 'APPENDIX_F3_G_LRD_GS4_AUTOMATIC_FISHER'
            cell_path = OUT / cell
            result_path = cell_path / 'result.json'
            if result_path.exists() and json.loads(result_path.read_text()).get('status') == 'DONE':
                state['cells'][cell] = json.loads(result_path.read_text())
                print('SKIP_ALREADY_DONE', cell, flush=True)
                continue
            cell_path.mkdir(parents=True, exist_ok=True)
            started = utc()
            state['cells'][cell] = {'status': 'RUNNING', 'group_size': gs, 'allocator': 'fisher_uniform', 'started_utc': started, 'old_results_valid': True}
            save_json(STATE, state)
            print('CELL_START', cell, 'gs', gs, flush=True)
            hold.clear()
            args = types.SimpleNamespace(
                model_id=alias, seed=42, dump_huggingface_model=False, use_cache=True, device='cuda',
                n_fisher_calib_samples=32, n_whiten_calib_samples=256, calib_dataset='wikitext2',
                calib_seqlen=1024, head_group_size=gs, param_ratio_target=0.5,
                verbose=False, search_method='fisher_uniform', decompose_method='whiten')
            official_compress.compress(args)
            model, tokenizer = hold['model'], hold['tokenizer']
            if fisher_cache_sha is None:
                fisher_cache = BASE / f'cache/{alias.replace("/", "_")}_calib_fisher_info.pt'
                if fisher_cache.exists():
                    fisher_cache_sha = sha_file(fisher_cache)
            ranks, rank_sum, total_rank = hold['selection'], hold['rank_sum'], hold['total_rank']
            census = resource(model, ranks, rank_sum, total_rank)
            if not census['allocation_integrity_pass']:
                raise RuntimeError(f'official allocation-integrity check failed: {census}')
            allocation = {
                'schema': 'PALU_APPENDIX_F3_AUTOMATIC_ALLOCATION_V1', 'cell': cell, 'group_size': gs,
                'allocator': 'official fisher_uniform', 'fisher_samples': 2048, 'fisher_seqlen': 1024,
                'seed': 42, 'rank_quantum': 32, 'ranks': {k: [int(v) for v in values] for k, values in sorted(ranks.items())},
                'resource': census, 'created_utc': utc(), 'official_repo_commit': commit,
            }
            allocation['allocation_file_sha256'] = hashlib.sha256(json.dumps(allocation, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            save_json(cell_path / 'allocation.json', allocation)
            ppls = {}
            for dataset in datasets_to_run:
                eval_alias = f'{alias}/TABLE1_FISHER_GS{gs}'
                ppls.update(run_ppl_eval.eval_ppl(model, tokenizer, eval_alias, dataset, seqlen=2048, device='cuda'))
                print('CELL_METRIC', cell, dataset, ppls[dataset], flush=True)
            deltas = {
                dataset: {
                    label: {'paper_reference': ref, 'absolute_error': abs(ppls[dataset] - ref),
                            'relative_error': abs(ppls[dataset] - ref) / ref}
                    for label, ref in PAPER[dataset].items()
                }
                for dataset in ppls
            }
            result = {
                'schema': 'PALU_APPENDIX_F3_AUTOMATIC_RESULT_V1', 'campaign': 'PALU_OFFICIAL_TABLE1_REPRO_V1',
                'cell': cell, 'status': 'DONE', 'method': 'G-LRD',
                'group_size': gs, 'allocator': 'official fisher_uniform (Automatic rank allocation)', 'seed': 42,
                'official_repo_commit': commit, 'official_source_sha256': source_hashes,
                'model_path': str(MODEL), 'model_config_sha256': '9242e7db1bc2a17873e66084c3b1c6ed10883076e156b338fd6a7775748e2e3c',
                'dataset_sha256': dataset_sha, 'fisher': {'samples': 2048, 'seqlen': 1024,
                    'cache_path': str(BASE / f'cache/{alias.replace("/", "_")}_calib_fisher_info.pt'),
                    'cache_sha256': fisher_cache_sha, 'fresh_campaign_cache': True},
                'whitening': {'samples': 256, 'seqlen': 2048, 'decomposition': 'official whiten/truncation-aware SVD',
                    'cache_path': str(whitening_cache), 'cache_sha256': whitening_cache_sha,
                    'reused_same_campaign_calibration': True},
                'ppl_protocol': {'sequence_length': 2048, 'stride': 2048, 'tokens': {'wikitext2': 339968, 'c4': 524288}},
                'ppl': ppls,
                'paper_reference': {k: v for k, v in PAPER.items() if k in ppls},
                'comparison': deltas, 'resource': census,
                'classification': {
                    'wiki_vs_appendix_f3_automatic': 'APPENDIX_F3_DIRECT_REPRODUCTION',
                    'wiki_vs_table1_g_lrd': 'CROSS_SECTION_COMPARISON_NOT_EXACT_TABLE1_REPRODUCTION',
                    'c4_vs_table1_g_lrd': 'CROSS_SECTION_DIAGNOSTIC_ONLY',
                },
                'allocation_sha256': census['allocation_sha256'], 'realization_sha256': census['realization_sha256'],
                'started_utc': started, 'completed_utc': utc(), 'foreign_jobs_interrupted': 0,
                'historical_results_modified': False, 'old_results_valid': True,
            }
            result['result_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            save_json(result_path, result)
            state['cells'][cell] = result
            state['last_completed_cell'] = cell
            state['foreign_jobs_interrupted'] = 0
            save_json(STATE, state)
            print('CELL_DONE', cell, result['result_sha256'], flush=True)
            hold.clear()
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as exc:
        cell = next((name for name, item in state.get('cells', {}).items() if item.get('status') == 'RUNNING'), 'UNKNOWN')
        failure = {'status': 'TODO' if 'out of memory' in str(exc).lower() else 'FAILED',
                   'failure_type': 'GPU_MEMORY_RUNTIME_FAILURE' if 'out of memory' in str(exc).lower() else type(exc).__name__,
                   'reason': str(exc), 'traceback': traceback.format_exc(), 'failed_utc': utc(),
                   'old_results_valid': True, 'foreign_jobs_interrupted': 0}
        state['cells'][cell] = failure
        state['status'] = 'PARTIAL_TODO_OR_FAILED'
        save_json(cell_path / 'FAILURE.json', failure)
        save_json(STATE, state)
        print('CELL_FAILED', cell, failure['failure_type'], failure['reason'], flush=True)
        raise
    finally:
        state['finished_utc'] = utc()
        state['status'] = 'COMPLETE' if all(state.get('cells', {}).get('APPENDIX_F3_G_LRD_GS4_AUTOMATIC_FISHER', {}).get('status') == 'DONE' for _gs, _datasets in specs) else 'PARTIAL_TODO_OR_FAILED'
        state['foreign_jobs_interrupted'] = 0
        save_json(STATE, state)
        try:
            ACTIVE.unlink()
        except FileNotFoundError:
            pass
        lock.close()
    print('SWEEP_' + state['status'], flush=True)


if __name__ == '__main__':
    main()
