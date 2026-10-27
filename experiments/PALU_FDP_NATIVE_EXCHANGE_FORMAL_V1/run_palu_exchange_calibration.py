import argparse
import copy
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


BRANCH = Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1')
PROFILE = Path('/diskE/jiaqizhang/PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1')
ROOT = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2')
OFFICIAL = ROOT / 'official'
DATA = OFFICIAL / 'data'
MODEL = ROOT / 'models' / 'Llama-2-7b-hf'
FACTOR_CACHE = PROFILE / 'marginal' / 'runtime' / 'llama2_official_whiten_factors_rank512.pt'
SEQ = 2048
FP16_BYTES = 2
N_BOOT = 10000


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def schedule_sha(selection):
    return hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def parent_and_child(model, full_name):
    parts = full_name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def build_from_factors(HeadwiseLowRankModule, raw, factors, ranks, device):
    new = HeadwiseLowRankModule(ranks, raw.in_features, raw.out_features, bias=raw.bias is not None).to(device=device, dtype=raw.weight.dtype)
    for i, r in enumerate(ranks):
        left, right = factors[i]
        new.U[i].weight.data.copy_(left[:, :r].to(device))
        if raw.bias is not None:
            bias = raw.bias.data.reshape(len(factors), -1)[i]
            new.U[i].bias.data.copy_(bias)
    new.VT.weight.data.copy_(torch.cat([x[1][:r, :] for x, r in zip(factors, ranks)], dim=0).to(device))
    return new


def layer_from_unit(unit):
    return int(re.search(r'model\.layers\.(\d+)\.', unit).group(1))


def load_signal_rows():
    rows = {}
    path = BRANCH / 'audit' / 'SOURCE_PALU_NATIVE_MARGINAL_PER_UNIT.csv'
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if row.get('model') != 'llama2_7b_hf':
                continue
            key = (f"{float(row['ratio']):.1f}", row['unit'])
            rows[key] = row
    return rows


def load_c4_docs(path, max_index):
    docs = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i > max_index:
                break
            if i not in docs:
                docs[i] = json.loads(line)['text']
    return docs


def load_runtime(ratio):
    ratio_tag = f'{ratio:.1f}'
    schedule = json.loads((BRANCH / 'schedules' / f'llama2_ratio_{ratio_tag}_anchor_schedule.json').read_text())
    manifest = json.loads((BRANCH / 'exchange_calibration' / 'manifest.json').read_text())
    sys.path.insert(0, str(OFFICIAL)); sys.path.insert(0, str(OFFICIAL / 'Palu'))
    had = types.ModuleType('fast_hadamard_transform'); had.hadamard_transform = lambda x, scale=1.0: x * scale; sys.modules['fast_hadamard_transform'] = had
    import run_official_palu_offline as wrapper
    wrapper.WIKI_TRAIN = DATA / 'wiki-train.hf_joined.raw'
    wrapper.WIKI_TEST = DATA / 'wiki-test.hf_joined.raw'
    from utils import load_model_and_tokenizer
    import palu.decomposition as decomposition
    from palu.model.modules.svd_linear import HeadwiseLowRankModule
    decomposition.get_calib_data = wrapper.offline_get_calib_data
    real_load = torch.load
    def trusted_load(*a, **kw):
        kw.setdefault('weights_only', False)
        return real_load(*a, **kw)
    torch.load = trusted_load
    model, tokenizer = load_model_and_tokenizer(str(MODEL))
    model.to(torch.device('cuda'))
    args_ns = types.SimpleNamespace(calib_dataset='wikitext2', model_id=str(MODEL), use_cache=True, n_whiten_calib_samples=256)
    runtime = BRANCH / 'runtime'; runtime.mkdir(exist_ok=True)
    os.chdir(runtime)
    decomposition.get_whiten_scale_matrix(model, tokenizer, args_ns, 'cuda')
    raw_modules = {}; parents = {}
    for name, module in list(model.named_modules()):
        if 'k_proj' not in name and 'v_proj' not in name:
            continue
        raw_modules[name] = module
        parents[name] = parent_and_child(model, name)
    factors = torch.load(FACTOR_CACHE, map_location='cpu')
    selection = schedule['selection']
    baseline_modules = {}
    for name, ranks in selection.items():
        baseline_modules[name] = build_from_factors(HeadwiseLowRankModule, raw_modules[name], factors[name], ranks, torch.device('cuda'))
        parent, child = parents[name]; setattr(parent, child, baseline_modules[name])
    model.to(torch.device('cuda'))
    wiki_ids = tokenizer((DATA / 'wiki-train.hf_joined.raw').read_text(encoding='utf-8'), return_tensors='pt', add_special_tokens=False).input_ids[0]
    c4_chunks = manifest['c4']['chunks']
    c4_docs = load_c4_docs(Path(manifest['c4']['source']), max(x['doc_index'] for x in c4_chunks))
    c4_ids = {x['doc_index']: tokenizer(c4_docs[x['doc_index']], return_tensors='pt', add_special_tokens=False).input_ids[:, x['token_start']:x['token_end']] for x in c4_chunks}
    return model, tokenizer, schedule, manifest, selection, raw_modules, parents, factors, baseline_modules, wiki_ids, c4_ids


def evaluate_samples(model, manifest, wiki_ids, c4_ids):
    device = next(model.parameters()).device
    wiki = []; c4 = []
    with torch.no_grad():
        for ch in manifest['wikitext']['chunks']:
            ids = wiki_ids[ch['token_start']:ch['token_end']].unsqueeze(0).to(device)
            pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids, device=ids.device), position_ids=pos, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='mean')
            value = float(loss.item())
            if not math.isfinite(value):
                raise FloatingPointError('non-finite Wiki loss')
            wiki.append(value)
        for ch in manifest['c4']['chunks']:
            ids = c4_ids[ch['doc_index']].to(device)
            pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids, device=ids.device), position_ids=pos, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='mean')
            value = float(loss.item())
            if not math.isfinite(value):
                raise FloatingPointError('non-finite C4 loss')
            c4.append(value)
    return {'wiki': wiki, 'c4': c4, 'J': 0.5 * (sum(wiki) / len(wiki) + sum(c4) / len(c4))}


def bootstrap_ci(delta, seed):
    rng = random.Random(seed)
    nw = len(delta['wiki']); nc = len(delta['c4'])
    values = []
    for _ in range(N_BOOT):
        w = sum(delta['wiki'][rng.randrange(nw)] for _ in range(nw)) / nw
        c = sum(delta['c4'][rng.randrange(nc)] for _ in range(nc)) / nc
        values.append(0.5 * (w + c))
    values.sort()
    return values[int(0.025 * N_BOOT)], values[int(0.975 * N_BOOT)]


def unit_records(ratio, selection, signal):
    rows = load_signal_rows()
    out = []
    for module, ranks in selection.items():
        for slot, rank in enumerate(ranks):
            unit = f'{module}::slot{slot}'
            row = rows.get((f'{ratio:.1f}', unit))
            if row is None:
                raise KeyError(f'missing frozen profile row {ratio} {unit}')
            fisher = float(row['fisher_importance'])
            q = float(row['q'])
            phi = float(row['phi_palu']) if row.get('phi_palu') not in ('', 'None', 'nan') else float('-inf')
            if signal == 'fdp_native':
                score = phi
            elif signal == 'fdp_prior':
                score = q
            elif signal == 'fisher':
                score = fisher
            elif signal.startswith('random'):
                seed = int(signal.split('_')[-1])
                stable = int(hashlib.sha256(unit.encode()).hexdigest()[:16], 16)
                score = random.Random(seed ^ stable).random()
            else:
                raise ValueError(signal)
            out.append({'unit': unit, 'module': module, 'slot': slot, 'rank': rank, 'fisher': fisher, 'q': q, 'phi': phi, 'score': score})
    return out


def candidate_pairs(records):
    donors = sorted([x for x in records if x['rank'] > 32], key=lambda x: (x['score'], x['unit']))
    receivers = sorted([x for x in records if x['rank'] < 512], key=lambda x: (-x['score'], x['unit']))
    pairs = []
    used_donors = set(); used_receivers = set()
    for receiver in receivers:
        for donor in donors:
            if donor['unit'] == receiver['unit'] or donor['unit'] in used_donors or receiver['unit'] in used_receivers:
                continue
            pairs.append((donor, receiver))
            used_donors.add(donor['unit']); used_receivers.add(receiver['unit'])
            break
        if len(pairs) == 16:
            break
    return pairs


def parse_unit(selection, unit):
    module, suffix = unit.split('::slot')
    return module, int(suffix)


def apply_exchange(selection, donor, receiver):
    out = copy.deepcopy(selection)
    dm, ds = parse_unit(selection, donor['unit']); rm, rs = parse_unit(selection, receiver['unit'])
    if out[dm][ds] <= 32 or out[rm][rs] >= 512:
        return None
    out[dm][ds] -= 32
    out[rm][rs] += 32
    return out


def actual_resource(selection):
    rank = sum(sum(v) for v in selection.values())
    cache = rank * SEQ * FP16_BYTES
    return {'total_native_rank': rank, 'prefill_cache_bytes': cache, 'decode_cache_bytes': cache}


def install_modules(model, raw_modules, parents, factors, selection, names, HeadwiseLowRankModule):
    installed = {}
    for name in names:
        new = build_from_factors(HeadwiseLowRankModule, raw_modules[name], factors[name], selection[name], torch.device('cuda'))
        parent, child = parents[name]; setattr(parent, child, new); installed[name] = new
    return installed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, required=True)
    ap.add_argument('--signal', required=True, choices=['fdp_native', 'fdp_prior', 'fisher', 'random_0', 'random_1', 'random_2'])
    args = ap.parse_args()
    ratio_tag = f'{args.ratio:.1f}'
    outdir = BRANCH / 'exchange' / 'calibration' / f'ratio_{ratio_tag}' / args.signal
    outdir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, schedule, manifest, selection, raw_modules, parents, factors, baseline_modules, wiki_ids, c4_ids = load_runtime(args.ratio)
    from palu.model.modules.svd_linear import HeadwiseLowRankModule
    records = unit_records(args.ratio, selection, args.signal)
    pairs = candidate_pairs(records)
    frozen_pairs = [{'proposal': i + 1, 'donor': d, 'receiver': r} for i, (d, r) in enumerate(pairs)]
    (outdir / 'candidate_order.json').write_text(json.dumps({'ratio': args.ratio, 'signal': args.signal, 'max_candidate_proposals': 16, 'pairs': frozen_pairs, 'ranking_frozen_before_quality': True}, indent=2, sort_keys=True) + '\n')
    current = evaluate_samples(model, manifest, wiki_ids, c4_ids)
    base_resource = actual_resource(selection)
    (outdir / 'baseline.json').write_text(json.dumps({'ratio': args.ratio, 'signal': args.signal, 'J': current['J'], 'wiki_mean_nll': sum(current['wiki']) / len(current['wiki']), 'c4_mean_nll': sum(current['c4']) / len(current['c4']), 'resource': base_resource, 'schedule_sha': schedule['schedule_sha'], 'manifest_sha256': sha(BRANCH / 'exchange_calibration' / 'manifest.json'), 'status': 'EXCHANGE_BASELINE_COMPLETE'}, indent=2, sort_keys=True) + '\n')
    current_selection = selection
    current_samples = current
    decisions = []
    for proposal in frozen_pairs:
        donor = proposal['donor']; receiver = proposal['receiver']
        candidate_selection = apply_exchange(current_selection, donor, receiver)
        decision = {'ratio': args.ratio, 'signal': args.signal, 'proposal': proposal['proposal'], 'donor': donor['unit'], 'receiver': receiver['unit'], 'donor_score': donor['score'], 'receiver_score': receiver['score'], 'status': 'INFEASIBLE_NATIVE_EXCHANGE', 'accepted': False, 'current_schedule_sha': schedule_sha(current_selection), 'manifest_sha256': sha(BRANCH / 'exchange_calibration' / 'manifest.json')}
        if candidate_selection is None:
            decisions.append(decision); continue
        before = actual_resource(current_selection); after = actual_resource(candidate_selection)
        decision.update({'resource_before': before, 'resource_after': after, 'resource_exact_match': before == after, 'native_rank_quantum': 32})
        if before != after:
            decisions.append(decision); continue
        changed = {parse_unit(current_selection, donor['unit'])[0], parse_unit(current_selection, receiver['unit'])[0]}
        old_modules = {name: getattr(parents[name][0], parents[name][1]) for name in changed}
        install_modules(model, raw_modules, parents, factors, candidate_selection, changed, HeadwiseLowRankModule)
        candidate_samples = evaluate_samples(model, manifest, wiki_ids, c4_ids)
        delta = {'wiki': [b - a for a, b in zip(current_samples['wiki'], candidate_samples['wiki'])], 'c4': [b - a for a, b in zip(current_samples['c4'], candidate_samples['c4'])]}
        delta_j = 0.5 * (sum(delta['wiki']) / len(delta['wiki']) + sum(delta['c4']) / len(delta['c4']))
        ci_low, ci_high = bootstrap_ci(delta, 20260919 + proposal['proposal'])
        accept = delta_j < 0.0 and ci_high < 0.0
        decision.update({'status': 'ACCEPTED' if accept else 'REJECTED', 'accepted': accept, 'delta_J': delta_j, 'bootstrap_ci_low': ci_low, 'bootstrap_ci_high': ci_high, 'bootstrap_n': N_BOOT, 'current_schedule_sha': schedule_sha(current_selection), 'candidate_schedule_sha': schedule_sha(candidate_selection), 'candidate_samples': candidate_samples})
        if accept:
            current_selection = candidate_selection
            current_samples = candidate_samples
            for name in changed:
                old_modules[name] = getattr(parents[name][0], parents[name][1])
        else:
            for name, old in old_modules.items():
                parent, child = parents[name]; setattr(parent, child, old)
        decisions.append(decision)
        gc.collect(); torch.cuda.empty_cache()
    (outdir / 'decisions.json').write_text(json.dumps({'ratio': args.ratio, 'signal': args.signal, 'decisions': decisions, 'accepted_transfer_count': sum(x['accepted'] for x in decisions), 'final_schedule_sha': schedule_sha(current_selection), 'final_resource': actual_resource(current_selection), 'status': 'EXCHANGE_CALIBRATION_COMPLETE'}, indent=2, sort_keys=True) + '\n')
    (outdir / 'final_schedule.json').write_text(json.dumps({'ratio': args.ratio, 'signal': args.signal, 'selection': current_selection, 'schedule_sha': schedule_sha(current_selection), 'resource': actual_resource(current_selection), 'accepted_transfer_count': sum(x['accepted'] for x in decisions), 'source_schedule_sha': schedule['schedule_sha'], 'manifest_sha256': sha(BRANCH / 'exchange_calibration' / 'manifest.json'), 'status': 'EXCHANGE_SCHEDULE_FROZEN_PRE_FORMAL_QUALITY'}, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'EXCHANGE_CALIBRATION_COMPLETE', 'ratio': args.ratio, 'signal': args.signal, 'accepted_transfer_count': sum(x['accepted'] for x in decisions), 'final_schedule_sha': schedule_sha(current_selection)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
