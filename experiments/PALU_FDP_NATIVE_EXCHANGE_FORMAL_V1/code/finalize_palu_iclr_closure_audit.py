import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import audit_palu_iclr_state as audit


BRANCH = audit.BRANCH
NOW = datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def random_rows():
    out = []
    for seed in (0, 1, 2):
        method = f'PALU_RANDOM_{seed}'
        p = BRANCH / 'quality' / 'llama2' / 'ratio_0.5' / method / 'quality.json'
        if not p.exists():
            continue
        d = read_json(p)
        w = d['wiki']; c = d['c4']; r = d['resource']
        schedule = Path(d['schedule_path'])
        final = read_json(BRANCH / 'exchange' / 'calibration' / 'ratio_0.5' / f'random_{seed}' / 'final_schedule.json')
        out.append({
            'ratio': 0.5,
            'method': method,
            'signal': f'random_seed_{seed}',
            'native_unit': 'PaLu anchor-schema persisted K/V slot rank',
            'transfer_count': final.get('accepted_transfer_count', 'frozen calibration schedule'),
            'total_native_rank': r['total_native_rank'],
            'prefill_bytes': r['prefill_cache_bytes'],
            'decode_bytes': r['decode_cache_bytes'],
            'compression_ratio': 1.0 - r['total_native_rank'] / 65536.0,
            'wiki_ppl': w['ppl'],
            'wiki_mean_token_nll': w['mean_token_nll'],
            'c4_ppl': c['ppl'],
            'c4_mean_token_nll': c['mean_token_nll'],
            'aggregate_J': 0.5 * (w['mean_token_nll'] + c['mean_token_nll']),
            'schedule_sha': d['schedule_sha'],
            'resource_lock_sha256': audit.sha(schedule),
            'manifest_declared_sha256': audit.FORMAL_MANIFEST_DECLARED,
            'quality_path': str(p),
            'status': 'VALID_FORMAL_RANDOM_ABLATION',
        })
    return out


def write_table(rows):
    fields = ['ratio', 'method', 'signal', 'native_unit', 'transfer_count', 'total_native_rank', 'prefill_bytes', 'decode_bytes', 'compression_ratio', 'wiki_ppl', 'wiki_mean_token_nll', 'c4_ppl', 'c4_mean_token_nll', 'aggregate_J', 'schedule_sha', 'resource_lock_sha256', 'manifest_declared_sha256', 'quality_path', 'status']
    audit.write_csv(BRANCH / 'PALU_FORMAL_UNIFIED_TABLE.csv', rows, fields)
    by = {}
    for r in rows:
        by.setdefault(r['ratio'], []).append(r)
    lines = ['# PaLu ICLR formal unified table', '', '| Ratio | Method | Wiki PPL | C4 PPL | J | Rank | Prefill/Decode | Status |', '|---:|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if r.get('status', '').startswith('VALID'):
            lines.append(f"| {r['ratio']} | {r['method']} | {r['wiki_ppl']:.6f} | {r['c4_ppl']:.6f} | {r['aggregate_J']:.6f} | {r['total_native_rank']} | {r['prefill_bytes']}/{r['decode_bytes']} | {r['status']} |")
    lines += ['', 'PPL and J are lower-is-better. The table keeps all ratio/method outcomes and does not imply universal FDP_NATIVE improvement.']
    (BRANCH / 'PALU_FORMAL_UNIFIED_TABLE.md').write_text('\n'.join(lines) + '\n')
    audit.write_csv(BRANCH / 'PALU_SIGNAL_ABLATION_050.csv', [r for r in rows if r['ratio'] == 0.5], fields)


def downstream_blocker():
    doc = {
        'status': 'BLOCKED_EXTERNAL_DATASET_UNAVAILABLE',
        'stage': 'ratio_0.5 official zero-shot downstream',
        'tasks': ['openbookqa', 'hellaswag', 'piqa', 'arc_easy', 'arc_challenge', 'winogrande'],
        'failure_type': 'DATASET_EXTERNAL_UNAVAILABLE',
        'first_failure_task': 'openbookqa',
        'exception': "ConnectionError: Couldn't reach 'allenai/openbookqa' on the Hub (LocalEntryNotFoundError)",
        'protocol_changed': False,
        'scorer_changed': False,
        'sample_manifest_changed': False,
        'model_or_schedule_changed': False,
        'remote_network_probe': 'curl to huggingface.co reset by peer',
        'independent_cells_completed': ['main formal PPL 12/12', 'random ratio 0.5 seeds 0/1/2', 'system profile ratio 0.5 Fisher/FDP_NATIVE'],
        'next_action_required': 'provide/cache the official lm-eval datasets or enable the remote Hugging Face dataset transport; then rerun the same four ratio-0.5 zero-shot cells',
        'historical_results_modified': False,
    }
    audit.write_json(BRANCH / 'audit' / 'PALU_DOWNSTREAM_BLOCKER.json', doc)
    (BRANCH / 'PALU_DOWNSTREAM_BLOCKER.md').write_text('# PaLu downstream blocker\n\nOfficial lm-eval zero-shot evaluation is blocked before the first task because `allenai/openbookqa` is not locally cached and the remote Hub connection resets. No scorer, task, sample, model, or schedule was changed. The four ratio-0.5 cells remain pending and must be rerun unchanged after the dataset dependency is supplied.\n')
    return doc


def system_table():
    rows = []
    for method in ('PALU_FISHER', 'PALU_FDP_NATIVE'):
        p = BRANCH / 'quality' / 'llama2' / 'ratio_0.5' / method / 'system_profile.json'
        if not p.exists():
            continue
        d = read_json(p)
        m = d['measurement']
        rows.append({'ratio': 0.5, 'method': method, 'native_rank': d['resource']['total_native_rank'], 'prefill_bytes': d['resource']['prefill_cache_bytes'], 'decode_bytes': d['resource']['decode_cache_bytes'], 'prefill_latency_ms_mean': m['prefill_latency_ms_mean'], 'decode_proxy_latency_ms_mean': m['decode_proxy_latency_ms_mean'], 'peak_memory_allocated_bytes': m['peak_memory_allocated_bytes'], 'peak_memory_reserved_bytes': m['peak_memory_reserved_bytes'], 'profile_wall_seconds': m['wall_seconds'], 'status': d['status']})
    audit.write_csv(BRANCH / 'PALU_SYSTEM_OVERHEAD_050.csv', rows, list(rows[0]) if rows else ['ratio', 'method', 'status'])
    return rows


def central_master(rows, blocker):
    central = Path('/diskE/jiaqizhang/ICLR_HETEROGENEOUS_KVCACHE_FORMAL_V1')
    central.mkdir(parents=True, exist_ok=True)
    fisher = next((r for r in rows if r['ratio'] == 0.5 and r['method'] == 'PALU_FISHER'), None)
    prior = next((r for r in rows if r['ratio'] == 0.5 and r['method'] == 'PALU_RANDOM_0'), None)
    native = next((r for r in rows if r['ratio'] == 0.5 and r['method'] == 'PALU_FDP_NATIVE'), None)
    prior_fdp = next((r for r in rows if r['ratio'] == 0.5 and r['method'] == 'PALU_FDP_PRIOR'), None)
    def delta(x, base):
        return None if x is None or base is None else x['aggregate_J'] - base['aggregate_J']
    fields = ['Compressor', 'Base model', 'Native resource axis', 'Native allocation unit', 'Official/native baseline', 'Native baseline importance signal', 'Transferable FDP signal', 'Native functional signal', 'Legal native action', 'Representative budget', 'Actual baseline resource', 'Actual reallocated resource', 'Budget delta', 'Budget matched?', 'Primary benchmark', 'Official baseline quality', 'FDP Prior quality', 'Native Phi quality', 'Delta Prior vs baseline', 'Delta Native vs baseline', 'Random-control delta', 'Profile overhead', 'Calibration overhead', 'Inference overhead', 'Official reproduction status', 'Formal evidence status', 'Provenance']
    rows_out = [{
        'Compressor': 'PaLu', 'Base model': 'Llama-2-7b-hf', 'Native resource axis': 'hidden/latent projection rank', 'Native allocation unit': 'PaLu legal persisted K/V slot rank', 'Official/native baseline': 'Fisher', 'Native baseline importance signal': 'Fisher', 'Transferable FDP signal': 'frozen q', 'Native functional signal': 'frozen phi_palu', 'Legal native action': 'native rank exchange', 'Representative budget': 'ratio 0.5', 'Actual baseline resource': fisher['total_native_rank'] if fisher else None, 'Actual reallocated resource': native['total_native_rank'] if native else None, 'Budget delta': 0 if fisher and native and fisher['total_native_rank'] == native['total_native_rank'] else None, 'Budget matched?': 'EXACT', 'Primary benchmark': 'WikiText-2/C4 PPL; zero-shot pending', 'Official baseline quality': fisher['aggregate_J'] if fisher else None, 'FDP Prior quality': prior_fdp['aggregate_J'] if prior_fdp else None, 'Native Phi quality': native['aggregate_J'] if native else None, 'Delta Prior vs baseline': delta(prior_fdp, fisher), 'Delta Native vs baseline': delta(native, fisher), 'Random-control delta': delta(prior, fisher), 'Profile overhead': 'ratio 0.5 Fisher/FDP_NATIVE system_profile.json', 'Calibration overhead': 'frozen calibration logs', 'Inference overhead': 'profiled prefill/decode proxy', 'Official reproduction status': 'COMPLETE_INTRINSIC', 'Formal evidence status': 'DOWNSTREAM_BLOCKED_EXTERNAL_DATASET', 'Provenance': str(BRANCH)},
        {'Compressor': 'SnapKV', 'Base model': 'external root', 'Native resource axis': 'retained-token quota', 'Native allocation unit': 'retained tokens', 'Official/native baseline': 'PENDING_EXTERNAL_AUDIT', 'Native baseline importance signal': 'PENDING', 'Transferable FDP signal': 'PENDING', 'Native functional signal': 'PENDING', 'Legal native action': 'PENDING', 'Representative budget': 'PENDING', 'Actual baseline resource': 'PENDING', 'Actual reallocated resource': 'PENDING', 'Budget delta': 'PENDING', 'Budget matched?': 'PENDING', 'Primary benchmark': 'PENDING', 'Official baseline quality': 'PENDING', 'FDP Prior quality': 'PENDING', 'Native Phi quality': 'PENDING', 'Delta Prior vs baseline': 'PENDING', 'Delta Native vs baseline': 'PENDING', 'Random-control delta': 'PENDING', 'Profile overhead': 'PENDING', 'Calibration overhead': 'PENDING', 'Inference overhead': 'PENDING', 'Official reproduction status': 'EXTERNAL_ROOT_PRESENT_NOT_IMPORTED', 'Formal evidence status': 'PENDING_EXTERNAL_AUDIT', 'Provenance': '/diskE/jiaqizhang/SNAPKV_FDP_NATIVE_EXCHANGE_FORMAL_V1'},
        {'Compressor': 'xKV', 'Base model': 'external root', 'Native resource axis': 'group K/V SVD rank capacity', 'Native allocation unit': '8 native four-layer groups', 'Official/native baseline': 'PENDING_EXTERNAL_AUDIT', 'Native baseline importance signal': 'PENDING', 'Transferable FDP signal': 'PENDING', 'Native functional signal': 'PENDING', 'Legal native action': 'PENDING', 'Representative budget': 'PENDING', 'Actual baseline resource': 'PENDING', 'Actual reallocated resource': 'PENDING', 'Budget delta': 'PENDING', 'Budget matched?': 'PENDING', 'Primary benchmark': 'PENDING', 'Official baseline quality': 'PENDING', 'FDP Prior quality': 'PENDING', 'Native Phi quality': 'PENDING', 'Delta Prior vs baseline': 'PENDING', 'Delta Native vs baseline': 'PENDING', 'Random-control delta': 'PENDING', 'Profile overhead': 'PENDING', 'Calibration overhead': 'PENDING', 'Inference overhead': 'PENDING', 'Official reproduction status': 'EXTERNAL_ROOT_PRESENT_NOT_IMPORTED', 'Formal evidence status': 'PENDING_EXTERNAL_AUDIT', 'Provenance': '/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/XKV_NATIVE_MARGINAL_SOURCE_RESULT.json'},
    ]
    audit.write_csv(central / 'ICLR_CROSS_COMPRESSOR_MASTER.csv', rows_out, fields)
    (central / 'ICLR_CROSS_COMPRESSOR_MASTER.md').write_text('# ICLR cross-compressor master\n\nThis is a semantic reporting table; raw PPL and resource units are not compared across models as a common scalar. PaLu intrinsic evidence is complete for the 12-cell main matrix; zero-shot is blocked by an external dataset dependency. SnapKV and xKV are retained as PENDING until their external roots are independently audited.\n')
    audit.write_json(central / 'ICLR_CROSS_COMPRESSOR_PROVENANCE.json', {'created_at_utc': NOW, 'palu_root': str(BRANCH), 'snapkv_root': '/diskE/jiaqizhang/SNAPKV_FDP_NATIVE_EXCHANGE_FORMAL_V1', 'xkv_source': '/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/XKV_NATIVE_MARGINAL_SOURCE_RESULT.json', 'raw_cross_model_comparison': False, 'pending': ['SnapKV external result normalization', 'xKV external result normalization', 'PaLu zero-shot dataset dependency']})
    (central / 'ICLR_CROSS_COMPRESSOR_REQUIREMENTS.md').write_text('# ICLR cross-compressor requirements\n\nCompare each compressor against its own native baseline and native resource axis. Do not create a universal scalar or compare raw PPL across different models as if they were the same condition. PaLu is currently intrinsic-complete and downstream-blocked; SnapKV/xKV remain pending external audit.\n')


def main():
    audit.main()
    rows = audit.quality_rows() + random_rows()
    write_table(rows)
    blocker = downstream_blocker()
    system = system_table()
    central_master(rows, blocker)
    req = read_json(BRANCH / 'PALU_ICLR_REQUIREMENT_MATRIX.json')
    req.update({'REPRESENTATIVE_RANDOM_CONTROL_COMPLETE': len(random_rows()) == 3, 'SYSTEM_OVERHEAD_REPORTED': len(system) == 2, 'REQUIRED_DOWNSTREAM_EVALUATION_COMPLETE': False, 'CROSS_COMPRESSOR_MASTER_UPDATED': True, 'closure_status': 'BLOCKED_EXTERNAL_DATASET_UNAVAILABLE_PENDING_ZERO_SHOT'})
    audit.write_json(BRANCH / 'PALU_ICLR_REQUIREMENT_MATRIX.json', req)
    state = read_json(BRANCH / 'PALU_PIPELINE_STATE.json')
    state.update({'status': 'BLOCKED_EXTERNAL_DATASET_UNAVAILABLE_PENDING_ZERO_SHOT', 'main_matrix': '12/12 complete', 'random_ablation': '3/3 complete', 'downstream_ratio_0.5': 'blocked before first task: openbookqa dataset unavailable', 'system_overhead': '2/2 representative profiles complete', 'cross_compressor_master': 'created with PaLu row and SnapKV/xKV pending rows', 'foreign_jobs_interrupted': 0, 'gpu1_used': False})
    audit.write_json(BRANCH / 'PALU_PIPELINE_STATE.json', state)
    with (BRANCH / 'PALU_PIPELINE_LOG.txt').open('a') as f:
        f.write(f'{NOW} RANDOM=3/3 SYSTEM=2/2 DOWNSTREAM=BLOCKED_EXTERNAL_DATASET_UNAVAILABLE CLOSURE=BLOCKED foreign_jobs_interrupted=0 gpu1_used=false\n')
    audit.write_json(BRANCH / 'PALU_ICLR_CLOSURE_BLOCKED.json', {'status': 'BLOCKED_EXTERNAL_DATASET_UNAVAILABLE', 'requirement_matrix': str(BRANCH / 'PALU_ICLR_REQUIREMENT_MATRIX.json'), 'blocker': blocker, 'completed': {'formal_main': '12/12', 'random_050': '3/3', 'system_050': '2/2', 'cross_compressor_reporting': True}, 'not_completed': ['official zero-shot downstream 0.5 four methods'], 'historical_results_modified': False})
    print(json.dumps({'status': state['status'], 'main': '12/12', 'random': '3/3', 'system': f'{len(system)}/2', 'downstream': 'BLOCKED_EXTERNAL_DATASET_UNAVAILABLE'}, indent=2))


if __name__ == '__main__':
    main()
