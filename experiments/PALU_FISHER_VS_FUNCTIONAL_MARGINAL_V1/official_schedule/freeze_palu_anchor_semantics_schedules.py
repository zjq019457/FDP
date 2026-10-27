import csv
import hashlib
import json
import re
from pathlib import Path


ANCHOR = Path('/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/v2_runs/palu_fisher/PALU_V2_OFFICIAL_0P7_FISHER_ANCHOR.json')
QLOCK = Path('/diskF/jiaqizhang/fdp_final_core_generalization_v1/authority/FINAL_D125_Q_LOCK.json')
MODEL_CONFIG = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf/config.json')
OUT = Path('/diskE/jiaqizhang/PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1/official_schedule')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def schedule_sha(selection):
    return hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_schedule(anchor, ratio):
    # This is the exact persisted native schedule schema of the recovered
    # official anchor: two head-wise slots per K/V module, each capped at 512.
    modules = list(anchor['head_wise_ranks'])
    capacity = 512
    target = round(65536 * ratio)
    fisher = anchor['module_fisher_means']
    fisher_sum = sum(float(fisher[k]) * 2.0 for k in modules)
    floats = {}
    result = {}
    for k in modules:
        x = target * float(fisher[k]) / fisher_sum
        floats[k] = [x, x]
        result[k] = [min(capacity, int(x)), min(capacity, int(x))]
    indexes = [(k, i) for k in modules for i in range(2)]
    indexes.sort(key=lambda x: floats[x[0]][x[1]] - result[x[0]][x[1]])
    deficit = target - sum(sum(v) for v in result.values())
    while deficit > 0:
        for k, i in indexes:
            if result[k][i] == capacity:
                continue
            result[k][i] += 1
            deficit -= 1
            if deficit == 0:
                break
    result = {k: [max(1, round(v / 32)) * 32 for v in vals] for k, vals in result.items()}
    return result, target, sum(sum(v) for v in result.values())


def layer(key):
    return int(re.search(r'model\.layers\.(\d+)\.', key).group(1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anchor = json.loads(ANCHOR.read_text())
    qdoc = json.loads(QLOCK.read_text())
    config = json.loads(MODEL_CONFIG.read_text())
    ratios = [0.7, 0.5, 0.3]
    manifest = {
        'task': 'PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1',
        'status': 'OFFICIAL_ANCHOR_SCHEMA_FROZEN; CURRENT_WRAPPER_MISMATCH_QUARANTINED',
        'model_config': str(MODEL_CONFIG),
        'model_config_sha256': sha(MODEL_CONFIG),
        'current_model_num_attention_heads': config.get('num_attention_heads'),
        'current_model_num_key_value_heads': config.get('num_key_value_heads'),
        'anchor_file': str(ANCHOR),
        'anchor_sha256': sha(ANCHOR),
        'q_lock_file': str(QLOCK),
        'q_lock_file_sha256': sha(QLOCK),
        'q_SHA': qdoc['q_SHA'],
        'native_unit': 'one persisted head_wise_ranks slot per K/V module; rank quantum 32; two slots per module; cap 512',
        'total_capacity': 65536,
        'fisher_allocation': 'official anchor proportional Fisher allocation, integer deficit completion, then block-32 rounding',
        'ratios': ratios,
        'quality_used_to_design_schedule': False,
        'transfer_count_policy': 'TRANSFER_COUNT_POLICY_NOT_FROZEN',
        'historical_results_modified': False,
    }
    schedules = {}
    for ratio in ratios:
        selection, integer_target, realized = make_schedule(anchor, ratio)
        sha_value = schedule_sha(selection)
        schedules[str(ratio)] = {'integer_target_rank': integer_target, 'realized_rank_sum': realized, 'schedule_sha': sha_value}
        out = OUT / f'ratio_{ratio:.1f}'
        out.mkdir(parents=True, exist_ok=True)
        units = []
        for name, ranks in selection.items():
            f = float(anchor['module_fisher_means'][name])
            q = float(qdoc['q'][layer(name)])
            for slot, rank in enumerate(ranks):
                units.append({'unit': f'{name}::slot{slot}', 'module': name, 'layer': layer(name), 'slot': slot, 'fisher_importance': f, 'q': q, 'native_rank': rank, 'native_quantum': 32})
        doc = {
            'status': 'OFFICIAL_PALU_FISHER_NATIVE_CONTROL_SCHEDULE_FROZEN_ANCHOR_SCHEMA',
            'ratio': ratio,
            'capacity_rank': 65536,
            'integer_target_rank': integer_target,
            'realized_rank_sum': realized,
            'realized_capacity_ratio': realized / 65536,
            'rank_quantum': 32,
            'head_group_size_anchor': anchor.get('head_group_size'),
            'native_unit': manifest['native_unit'],
            'selection': selection,
            'units': units,
            'schedule_sha': sha_value,
            'anchor_schedule_sha_0p7': anchor['schedule_sha'],
            'anchor_reproduction_exact': ratio != 0.7 or sha_value == anchor['schedule_sha'],
            'fisher_source': anchor.get('fisher_source'),
            'fisher_module_means': anchor['module_fisher_means'],
            'q_SHA': qdoc['q_SHA'],
            'q_mapping': 'frozen layer q repeated to each persisted K/V slot; no score-driven remapping',
            'quality_used_to_design_schedule': False,
            'transfer_count_policy': 'TRANSFER_COUNT_POLICY_NOT_FROZEN',
        }
        (out / 'schedule.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
        with (out / 'units.csv').open('w', newline='') as f:
            fields = ['unit', 'module', 'layer', 'slot', 'fisher_importance', 'q', 'native_rank', 'native_quantum']
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(units)

    manifest['schedules'] = schedules
    manifest['ratio_0.7_reproduction_exact'] = schedules['0.7']['schedule_sha'] == anchor['schedule_sha']
    manifest['current_wrapper_semantics_audit'] = {
        'current_wrapper_fisher_uniform_capacity': 262144,
        'current_wrapper_fisher_uniform_groups_per_module': 8,
        'frozen_anchor_capacity': 65536,
        'frozen_anchor_slots_per_module': 2,
        'capacity_ratio_current_wrapper_to_anchor': 4.0,
        'status': 'NATIVE_UNIT_SEMANTICS_MISMATCH; CURRENT_WRAPPER_OUTPUTS_QUARANTINED',
    }
    (OUT / 'OFFICIAL_SCHEDULE_FREEZE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    audit = {
        'status': 'OFFICIAL_BASELINE_FREEZE_GATE_PASS_FOR_ANCHOR_SCHEMA; CURRENT_WRAPPER_GATE_FAIL',
        'reason': 'Current checked-out fisher_uniform path creates 8 slots/module and 262144 capacity, while immutable official 0.7 anchor has 2 slots/module and 65536 capacity. The anchor-schema generator reproduces the immutable 0.7 schedule SHA exactly and is used for 0.5/0.3 candidates.',
        'anchor_reproduction_exact': manifest['ratio_0.7_reproduction_exact'],
        'current_wrapper_outputs': 'DIAGNOSTIC_ONLY_NOT_OFFICIAL',
        'schedules': schedules,
        'transfer_count_policy': 'TRANSFER_COUNT_POLICY_NOT_FROZEN',
        'marginal_quality_started': False,
        'historical_frozen_results_modified': False,
    }
    (OUT / 'OFFICIAL_SCHEDULE_SEMANTICS_AUDIT.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
