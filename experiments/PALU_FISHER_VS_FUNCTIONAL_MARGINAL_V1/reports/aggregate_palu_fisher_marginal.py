import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path('/diskE/jiaqizhang/PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1')
LLAMA3 = ROOT / 'audit' / 'LLAMA3_EXISTING_AUDIT.json'
SCHED = ROOT / 'official_schedule' / 'OFFICIAL_SCHEDULE_FREEZE_MANIFEST.json'
SEM = ROOT / 'official_schedule' / 'OFFICIAL_SCHEDULE_SEMANTICS_AUDIT.json'
MANIFEST = ROOT / 'marginal' / 'calibration_manifest.json'
RATIOS = [0.7, 0.5, 0.3]


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); out = [0.0] * len(xs); i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]: j += 1
        r = (i + 1 + j) / 2.0
        for k in order[i:j]: out[k] = r
        i = j
    return out


def pearson(a, b):
    if len(a) < 2: return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0: return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def corr(a, b): return pearson(a, b), pearson(rankdata(a), rankdata(b))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load_units():
    rows = []
    if LLAMA3.exists():
        with (ROOT / 'PALU_NATIVE_MARGINAL_PER_UNIT.csv').open(newline='') as f:
            rows.extend(dict(r) for r in csv.DictReader(f))
    for ratio in RATIOS:
        p = ROOT / 'marginal' / f'ratio_{ratio:.1f}' / 'units.csv'
        if p.exists():
            with p.open(newline='') as f: rows.extend(dict(r, source='llama2_marginal') for r in csv.DictReader(f))
    return rows


def main():
    l3 = json.loads(LLAMA3.read_text()) if LLAMA3.exists() else None
    sched = json.loads(SCHED.read_text()) if SCHED.exists() else None
    sem = json.loads(SEM.read_text()) if SEM.exists() else None
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    profiles = {}
    for ratio in RATIOS:
        p = ROOT / 'marginal' / f'ratio_{ratio:.1f}' / 'status.json'
        profiles[str(ratio)] = json.loads(p.read_text()) if p.exists() else {'status': 'NOT_STARTED', 'ratio': ratio}

    rows = load_units()
    fields = sorted({k for r in rows for k in r})
    with (ROOT / 'PALU_NATIVE_MARGINAL_PER_UNIT.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

    correlations = []
    if l3: correlations.extend(l3.get('correlations', []))
    for ratio in RATIOS:
        p = ROOT / 'marginal' / f'ratio_{ratio:.1f}' / 'units.csv'
        if not p.exists(): continue
        with p.open(newline='') as f: rr = [r for r in csv.DictReader(f) if r.get('status') == 'COMPLETE']
        for a, b in [('fisher_importance', 'phi_palu'), ('q', 'phi_palu')]:
            if len(rr) >= 2:
                x, y = [float(r[a]) for r in rr], [float(r[b]) for r in rr]
                pe, sp = corr(x, y)
                correlations.append({'model': 'llama2_7b_hf', 'tier': f'ratio_{ratio:.1f}', 'signal_A': a, 'signal_B': b, 'Pearson': pe, 'Spearman': sp, 'N_units': len(rr), 'phi_primary': 'phi_palu'})
    with (ROOT / 'PALU_SIGNAL_CORRELATION_TABLE.csv').open('w', newline='') as f:
        cols = ['model', 'tier', 'signal_A', 'signal_B', 'Pearson', 'Spearman', 'N_units', 'phi_primary']; w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(correlations)

    complete = all(profiles[str(r)]['status'] == 'MARGINAL_PROFILE_COMPLETE' for r in RATIOS)
    result = {
        'status': 'PALU_FISHER_VS_FUNCTIONAL_MARGINAL_COMPLETE' if complete else 'PALU_FISHER_VS_FUNCTIONAL_MARGINAL_IN_PROGRESS',
        'task': 'PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1',
        'llama3_existing_fixed_audit': l3,
        'llama2_official_schedule_freeze': sched,
        'llama2_schedule_semantics_audit': sem,
        'independent_calibration_manifest': {'path': str(MANIFEST), 'sha256': sha(MANIFEST) if MANIFEST.exists() else None, 'status': manifest.get('status') if manifest else 'MISSING'},
        'llama2_marginal_profiles': profiles,
        'formal_fdp_realization': 'TRANSFER_COUNT_POLICY_NOT_FROZEN',
        'historical_frozen_results_modified': False,
        'native_palu_implementation_modified': False,
        'foreign_jobs_interrupted': 0,
        'interpretation': 'Pending Llama-2 unit profiles. Llama-3 frozen audit is descriptive; no equivalence claim is made without functional marginal phi_palu.',
    }
    (ROOT / 'PALU_FISHER_VS_FUNCTIONAL_MARGINAL.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    lines = ['# PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1', '', f"Status: **{result['status']}**", '', 'This independent branch does not modify historical frozen results, q, fragility, resource locks, or the PaLu native Fisher implementation.', '', '## Frozen Llama-3 / current Fixed audit', '']
    if l3:
        lines += ['| tier | Fisher vs phi Spearman | q vs phi Spearman | mean phi receiver | mean phi donor | exact rank |', '|---|---:|---:|---:|---:|---|']
        for t, s in l3['tiers'].items():
            cs = [x for x in l3['correlations'] if x['tier'] == t and x['signal_A'] == 'fisher_importance' and x['signal_B'] == 'phi_small'][0]
            qs = [x for x in l3['correlations'] if x['tier'] == t and x['signal_A'] == 'q' and x['signal_B'] == 'phi_small'][0]
            lines.append(f"| {t} | {cs['Spearman']:.6g} | {qs['Spearman']:.6g} | {s['mean_phi_receiver']:.6g} | {s['mean_phi_donor']:.6g} | {s['resource_exact']} |")
        lines += ['', 'The frozen roots have aggregate schedules but no immutable donor→receiver event-pair manifest, so exact per-event transfer-consistency is not claimed.']
    lines += ['', '## Llama-2 official schedule freeze', '']
    if sched:
        lines += ['| ratio | integer target | realized native rank | schedule SHA |', '|---:|---:|---:|---|']
        for r in ['0.7', '0.5', '0.3']:
            s = sched['schedules'][r]; lines.append(f"| {r} | {s['integer_target_rank']} | {s['realized_rank_sum']} | `{s['schedule_sha']}` |")
        lines += ['', f"0.7 reproduces the immutable anchor exactly: **{sched['ratio_0.7_reproduction_exact']}**.", '', 'The checked-out wrapper path is quarantined because it exposes 8 slots/module and 262144 capacity, versus the immutable anchor schema of 2 slots/module and 65536 capacity.']
    lines += ['', '## Llama-2 marginal profile', '', '| ratio | status | baseline J | completed units | legal-min units |', '|---:|---|---:|---:|---:|']
    for r in RATIOS:
        s = profiles[str(r)]; base = s.get('baseline', {}).get('J', '')
        p = ROOT / 'marginal' / f'ratio_{r:.1f}' / 'units.csv'; comp = skip = 0
        if p.exists():
            with p.open(newline='') as f:
                for row in csv.DictReader(f): comp += row.get('status') == 'COMPLETE'; skip += row.get('status') == 'SKIPPED_LEGAL_MIN_RANK'
        lines.append(f"| {r} | {s.get('status')} | {base} | {comp} | {skip} |")
    lines += ['', 'For formal FDP realization, the transfer-count policy remains `TRANSFER_COUNT_POLICY_NOT_FROZEN`; no formal transfer quality branch is launched from this task.', '', '## Outputs', '', '- `PALU_NATIVE_MARGINAL_PER_UNIT.csv`', '- `PALU_SIGNAL_CORRELATION_TABLE.csv`', '- `PALU_FISHER_VS_FUNCTIONAL_MARGINAL.json`', '- `PALU_FISHER_VS_FUNCTIONAL_MARGINAL.md`']
    (ROOT / 'PALU_FISHER_VS_FUNCTIONAL_MARGINAL.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__': main()
