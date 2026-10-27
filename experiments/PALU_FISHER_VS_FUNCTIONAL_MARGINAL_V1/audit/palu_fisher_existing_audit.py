import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path


OUT = Path('/diskE/jiaqizhang/PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1')
LOCK_ROOT = Path('/diskF/jiaqizhang/FDP_NATIVE_TRANSFER_V2_WORK/FINAL_FDP_CORE_V2/resource_locks')
MATRIX = Path('/diskF/jiaqizhang/FDP_NATIVE_TRANSFER_V2_WORK/reports/FRAGILITY_PROFILE_MATRIX.csv')
FISHER = Path('/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/v2_runs/palu_fisher/PALU_V2_OFFICIAL_0P7_FISHER_ANCHOR.json')
QDOC = Path('/diskF/jiaqizhang/fdp_final_core_generalization_v1/authority/FINAL_D125_Q_LOCK.json')
TIERS = ['LIGHT', 'MEDIUM', 'AGGRESSIVE']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        r = (i + 1 + j) / 2.0
        for k in order[i:j]:
            out[k] = r
        i = j
    return out


def pearson(a, b):
    if len(a) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def corr(a, b):
    return pearson(a, b), pearson(rankdata(a), rankdata(b))


def layer_of(key):
    m = re.search(r'model\.layers\.(\d+)\.', key)
    return int(m.group(1)) if m else None


def module_name(key):
    return key.split('.')[-2] if key.endswith('_proj') else key


def module_fisher_by_layer(doc):
    out = {i: {'k': None, 'v': None} for i in range(32)}
    for key, value in doc['module_fisher_means'].items():
        layer = layer_of(key)
        if layer is None:
            continue
        kind = 'k' if '.k_proj' in key else 'v' if '.v_proj' in key else None
        if kind:
            out[layer][kind] = float(value)
    for layer, x in out.items():
        vals = [v for v in (x['k'], x['v']) if v is not None]
        x['importance'] = sum(vals) / len(vals) if vals else None
    return out


def schedule_layer(schedule):
    out = {i: {'total': 0, 'k': 0, 'v': 0} for i in range(32)}
    for key, value in schedule.items():
        layer = layer_of(key)
        if layer is None:
            continue
        s = sum(int(x) for x in value)
        out[layer]['total'] += s
        if '.k_proj' in key:
            out[layer]['k'] += s
        elif '.v_proj' in key:
            out[layer]['v'] += s
    return out


def schedule_deltas(control, fixed):
    per_key = {}
    for key in control:
        a = [int(x) for x in control[key]]
        b = [int(x) for x in fixed.get(key, [])]
        slot_deltas = [y - x for x, y in zip(a, b)]
        per_key[key] = {
            'control': a,
            'fixed': b,
            'slot_deltas': slot_deltas,
            'changed_slots': sum(d != 0 for d in slot_deltas),
            'positive_32_quanta': sum(max(d, 0) for d in slot_deltas) / 32.0,
            'negative_32_quanta': sum(max(-d, 0) for d in slot_deltas) / 32.0,
        }
    return per_key


def load_rows():
    rows = []
    with MATRIX.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['operator'] == 'PaLu':
                rows.append({
                    'tier': r['tier'],
                    'layer': int(r['layer']),
                    'phi_small': float(r['phi_small']),
                    'phi_medium': float(r['phi_medium']),
                    'q': float(r['q']),
                    'q_rank': int(float(r['q_rank'])),
                    'phi_rank': int(float(r['phi_rank'])),
                    'manifest_SHA': r.get('manifest_SHA', ''),
                    'q_SHA': r.get('q_SHA', ''),
                })
    return rows


def main():
    (OUT / 'audit').mkdir(parents=True, exist_ok=True)
    fisher_doc = json.loads(FISHER.read_text())
    fisher = module_fisher_by_layer(fisher_doc)
    rows = load_rows()
    by_tier = {t: {r['layer']: r for r in rows if r['tier'] == t} for t in TIERS}
    all_rows = []
    correlations = []
    tier_summary = {}
    source_sha = {
        'fragility_profile_matrix_sha256': sha256(MATRIX),
        'fisher_anchor_sha256': sha256(FISHER),
        'q_lock_sha256': sha256(QDOC),
        'q_sha_value': json.loads(QDOC.read_text()).get('q_SHA'),
    }

    for tier in TIERS:
        lock = json.loads((LOCK_ROOT / f'PaLu_{tier}.json').read_text())
        c_layer = schedule_layer(lock['control_schedule'])
        f_layer = schedule_layer(lock['fdp_schedule'])
        key_deltas = schedule_deltas(lock['control_schedule'], lock['fdp_schedule'])
        for layer in range(32):
            r = dict(by_tier[tier][layer])
            r.update({
                'model': 'llama3_8b',
                'operator': 'PaLu',
                'tier': tier,
                'layer': layer,
                'fisher_k_importance': fisher[layer]['k'],
                'fisher_v_importance': fisher[layer]['v'],
                'fisher_importance': fisher[layer]['importance'],
                'control_native_rank': c_layer[layer]['total'],
                'fixed_native_rank': f_layer[layer]['total'],
                'delta_native_rank': f_layer[layer]['total'] - c_layer[layer]['total'],
                'transfer_status': 'receiver' if f_layer[layer]['total'] > c_layer[layer]['total'] else 'donor' if f_layer[layer]['total'] < c_layer[layer]['total'] else 'unchanged',
                'transfer_32_quanta_abs': abs(f_layer[layer]['total'] - c_layer[layer]['total']) / 32.0,
                'key_changed_slots': sum(key_deltas[k]['changed_slots'] for k in key_deltas if layer_of(k) == layer),
                'key_positive_32_quanta': sum(key_deltas[k]['positive_32_quanta'] for k in key_deltas if layer_of(k) == layer),
                'key_negative_32_quanta': sum(key_deltas[k]['negative_32_quanta'] for k in key_deltas if layer_of(k) == layer),
                'resource_lock_sha256': sha256(LOCK_ROOT / f'PaLu_{tier}.json'),
                'schedule_sha': lock.get('schedule_sha'),
                'native_source_sha': lock.get('native_source_sha'),
                'q_lock_sha': lock.get('q_SHA'),
            })
            all_rows.append(r)

        tier_rows = [r for r in all_rows if r['tier'] == tier]
        def values(name): return [float(r[name]) for r in tier_rows]
        pair = {'fisher_importance': 'phi_small', 'q': 'phi_small', 'fisher_importance_vs_q': 'q'}
        for a, b in [('fisher_importance', 'phi_small'), ('q', 'phi_small'), ('fisher_importance', 'q'),
                     ('fisher_importance', 'phi_medium'), ('q', 'phi_medium')]:
            x = values(a); y = values(b)
            p, s = corr(x, y)
            correlations.append({'model': 'llama3_8b', 'tier': tier, 'signal_A': a, 'signal_B': b, 'Pearson': p, 'Spearman': s, 'N_units': len(x), 'phi_primary': 'phi_small'})

        rec = [r['phi_small'] for r in tier_rows if r['transfer_status'] == 'receiver']
        don = [r['phi_small'] for r in tier_rows if r['transfer_status'] == 'donor']
        rec_m = sum(rec) / len(rec) if rec else None
        don_m = sum(don) / len(don) if don else None
        # Locks expose aggregate schedule deltas, but no immutable donor->receiver event-pair manifest.
        # Therefore this is a layer-set diagnostic, not a claim about historical pair identity.
        inferred_receiver_quanta = sum(r['key_positive_32_quanta'] for r in tier_rows)
        inferred_donor_quanta = sum(r['key_negative_32_quanta'] for r in tier_rows)
        tier_summary[tier] = {
            'control_total_native_rank': sum(c_layer[i]['total'] for i in range(32)),
            'fixed_total_native_rank': sum(f_layer[i]['total'] for i in range(32)),
            'resource_exact': sum(c_layer[i]['total'] for i in range(32)) == sum(f_layer[i]['total'] for i in range(32)),
            'receiver_layers': [r['layer'] for r in tier_rows if r['transfer_status'] == 'receiver'],
            'donor_layers': [r['layer'] for r in tier_rows if r['transfer_status'] == 'donor'],
            'mean_phi_receiver': rec_m,
            'mean_phi_donor': don_m,
            'mean_phi_receiver_gt_donor': rec_m > don_m if rec_m is not None and don_m is not None else None,
            'inferred_positive_32_quanta': inferred_receiver_quanta,
            'inferred_negative_32_quanta': inferred_donor_quanta,
            'transfer_pair_manifest': 'NOT_FOUND_IN_FROZEN_ROOTS',
            'transfer_consistency_rate': None,
            'transfer_consistency_status': 'UNAVAILABLE_EXACT_PAIRING; receiver/donor layer-set means reported',
            'lock_status': lock.get('status'),
            'lock_q_SHA': lock.get('q_SHA'),
        }

    fields = ['model', 'operator', 'tier', 'layer', 'fisher_k_importance', 'fisher_v_importance', 'fisher_importance', 'q', 'phi_small', 'phi_medium', 'q_rank', 'phi_rank', 'control_native_rank', 'fixed_native_rank', 'delta_native_rank', 'transfer_status', 'transfer_32_quanta_abs', 'key_changed_slots', 'key_positive_32_quanta', 'key_negative_32_quanta', 'resource_lock_sha256', 'schedule_sha', 'native_source_sha', 'q_lock_sha', 'q_SHA', 'manifest_SHA']
    with (OUT / 'PALU_NATIVE_MARGINAL_PER_UNIT.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    with (OUT / 'PALU_SIGNAL_CORRELATION_TABLE.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'tier', 'signal_A', 'signal_B', 'Pearson', 'Spearman', 'N_units', 'phi_primary']); w.writeheader(); w.writerows(correlations)

    audit = {
        'status': 'LLAMA3_EXISTING_FIXED_AUDIT_COMPLETE',
        'task': 'PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1',
        'immutable_inputs': source_sha,
        'fisher_aggregation': 'mean of official module_fisher_means for k_proj and v_proj at each layer',
        'phi_primary': 'phi_small; phi_medium retained as sensitivity column',
        'native_unit_observation': 'Frozen Llama-3 lock schedules expose duplicated two-slot K/V entries; deltas are reported per 32-rank schedule slot and at layer aggregate. No donor->receiver event-pair manifest was found in the frozen roots.',
        'tiers': tier_summary,
        'correlations': correlations,
        'foreign_jobs_interrupted': 0,
        'historical_frozen_results_modified': False,
    }
    (OUT / 'audit' / 'LLAMA3_EXISTING_AUDIT.json').write_text(json.dumps(audit, indent=2))

    lines = [
        '# PaLu Fisher vs Functional Marginal — existing Llama-3 Fixed audit',
        '',
        'This report is read-only over the frozen PaLu Fixed line. No historical quality, q, fragility, resource lock, or PaLu implementation was modified.',
        '',
        '## Provenance',
        '',
        f"- Fragility matrix SHA-256: `{source_sha['fragility_profile_matrix_sha256']}`",
        f"- Fisher anchor SHA-256: `{source_sha['fisher_anchor_sha256']}`",
        f"- FINAL_D125_Q_LOCK file SHA-256: `{source_sha['q_lock_sha256']}`; q_SHA value: `{source_sha['q_sha_value']}`",
        '- Fisher importance: arithmetic mean of official frozen `k_proj` and `v_proj` module Fisher means per layer.',
        '- Primary fragility: frozen `phi_small`; `phi_medium` is retained for sensitivity analysis.',
        '',
        '## Tier summary',
        '',
        '| tier | control rank | fixed rank | receiver layers | donor layers | mean phi(receiver) | mean phi(donor) | exact resource | transfer consistency |',
        '|---|---:|---:|---|---|---:|---:|---|---|',
    ]
    for t in TIERS:
        s = tier_summary[t]
        lines.append(f"| {t} | {s['control_total_native_rank']} | {s['fixed_total_native_rank']} | {s['receiver_layers']} | {s['donor_layers']} | {s['mean_phi_receiver']:.9g} | {s['mean_phi_donor']:.9g} | {s['resource_exact']} | {s['transfer_consistency_status']} |")
    lines += [
        '',
        'The frozen roots contain aggregate schedules but no immutable donor→receiver event-pair manifest. Therefore an exact per-event `phi_receiver > phi_donor` rate is not asserted; the report provides receiver/donor layer-set means and per-unit schedule deltas. This is a provenance limitation, not a rerun.',
        '',
        '## Signal correlations',
        '',
        '| tier | signal A | signal B | Pearson | Spearman | N |',
        '|---|---|---|---:|---:|---:|',
    ]
    for c in correlations:
        lines.append(f"| {c['tier']} | {c['signal_A']} | {c['signal_B']} | {c['Pearson']:.9g} | {c['Spearman']:.9g} | {c['N_units']} |")
    lines += [
        '',
        '## Files',
        '',
        '- `PALU_NATIVE_MARGINAL_PER_UNIT.csv`: layer/native schedule-unit audit rows.',
        '- `PALU_SIGNAL_CORRELATION_TABLE.csv`: Pearson and Spearman correlations.',
        '- `audit/LLAMA3_EXISTING_AUDIT.json`: machine-readable provenance and summary.',
    ]
    (OUT / 'PALU_FISHER_VS_FUNCTIONAL_MARGINAL.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
