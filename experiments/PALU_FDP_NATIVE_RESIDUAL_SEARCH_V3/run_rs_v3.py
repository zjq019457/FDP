#!/usr/bin/env python3
"""Isolated, resumable PaLu FDP Native residual-search V3 runner.

This runner only reads V1/V2 frozen artifacts. All mutable outputs are written
under PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3. Calibration and formal inputs are
loaded from the separately frozen V3 split binding.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gzip
import hashlib
import importlib
import importlib.machinery
import json
import math
import os
import random
import struct
import sys
import time
import traceback
import types
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3')
V1 = Path('/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1')
OFFICIAL = Path('/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1/code')
REPO = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu')
DATA = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/data')
MODEL_PATH = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf')
MANIFEST_PATH = BASE / 'PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3_MANIFEST.json'
CARD_PATH = BASE / 'PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3_METHOD_CARD.md'
SPLIT_PATH = BASE / 'V3_SPLIT_BINDING.json'
STATE_PATH = BASE / 'STATE.json'
QUANTUM, FLOOR, CEILING, SEQ, BOOT_N = 32, 32, 512, 2048, 10000
TOTAL_RANK = 131072
_LOCK = None


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha_obj(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha_ids(ids):
    return hashlib.sha256(struct.pack('<' + ('q' * len(ids)), *(int(x) for x in ids))).hexdigest()


def utc_now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def set_state(**updates):
    state = read_json(STATE_PATH)
    state.update(updates)
    status = updates.get('status')
    if isinstance(status, str) and not status.endswith('_FAILED'):
        # Old exception fields were being carried forward into later successful
        # stage states. Preserve them in ENGINEERING_AUDIT before advancing.
        for key in ('error', 'traceback', 'failure_type', 'previous_status'):
            state.pop(key, None)
    state['updated_utc'] = utc_now()
    state['foreign_jobs_interrupted'] = 0
    write_json(STATE_PATH, state)


def acquire_lock():
    global _LOCK
    p = BASE / '.v3_worker.lock'
    _LOCK = p.open('a+')
    try:
        fcntl.flock(_LOCK.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('an isolated V3 worker already owns the lock') from exc


def install_import_paths():
    sys.path.insert(0, str(OFFICIAL))
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / '3rdparty' / 'fast-hadamard-transform'))
    try:
        importlib.import_module('fast_hadamard_transform')
    except Exception:
        mod = types.ModuleType('fast_hadamard_transform')
        mod.hadamard_transform = lambda x, scale=1.0: x * scale
        mod.__spec__ = importlib.machinery.ModuleSpec('fast_hadamard_transform', loader=None)
        sys.modules['fast_hadamard_transform'] = mod


def load_modules():
    install_import_paths()
    import run_official_exchange_calibration as engine
    import profile_official_native_phi as profile
    return engine, profile


def resource(selection):
    total = sum(sum(int(x) for x in ranks) for ranks in selection.values())
    return {
        'total_native_rank': total,
        'factor_bytes_fp16': total * (4096 + 512) * 2,
        'prefill_cache_bytes': total * SEQ * 2,
        'decode_cache_bytes': total * SEQ * 2,
    }


def normalize_selection(value):
    return {str(k): [int(x) for x in v] for k, v in sorted(value.items())}


def allocation_sha(selection):
    return sha_obj(normalize_selection(selection))


def validate_frozen_inputs(engine):
    manifest = read_json(MANIFEST_PATH)
    card_sha = sha_file(CARD_PATH)
    split_sha = sha_file(SPLIT_PATH)
    split = read_json(SPLIT_PATH)
    expected = {
        'method_card_sha256': card_sha,
        'split_binding_file_sha256': split_sha,
        'manifest_file_sha256': sha_file(MANIFEST_PATH),
    }
    if manifest.get('status') != 'PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3_FROZEN_PRE_CALIBRATION':
        raise RuntimeError('V3 manifest is not frozen pre-calibration')
    if card_sha != manifest['method_card_sha256'] or split_sha != manifest['split_binding_file_sha256']:
        raise RuntimeError(f'frozen V3 input SHA mismatch: {expected}')
    if sha_obj(split) != manifest['split_binding_semantic_sha256']:
        raise RuntimeError('V3 split semantic SHA mismatch')
    if split.get('status') != 'V3_SPLITS_BOUND_READ_ONLY_PRE_CALIBRATION':
        raise RuntimeError('V3 split binding is not in the expected frozen state')
    action_path = Path(manifest['native_action_space_path'])
    q_path = Path(manifest['q_mapping_path'])
    phi_path = Path(manifest['phi_official_path'])
    start_path = Path(manifest['start_allocation_path'])
    for path, expected_sha in [
        (action_path, manifest['native_action_space_sha256']),
        (phi_path, manifest['phi_official_file_sha256']),
        (start_path, manifest['start_allocation_file_sha256']),
    ]:
        actual = sha_file(path)
        if actual != expected_sha:
            raise RuntimeError(f'frozen source file changed: {path}: {actual} != {expected_sha}')
    action = read_json(action_path)
    q_mapping = read_json(q_path)
    if q_mapping.get('mapping_sha256') != manifest['q_mapping_sha256']:
        raise RuntimeError('q mapping semantic SHA mismatch')
    protocol_path = Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/quality/formal_protocol/FORMAL_QUALITY_PROTOCOL_MANIFEST.json')
    if sha_file(protocol_path) != split['FORMAL_WIKI']['formal_protocol_manifest_sha256']:
        raise RuntimeError('official formal protocol source changed')
    if engine.schedule_sha(normalize_selection(read_json(start_path)['selection'])) != manifest['start_allocation_schedule_sha256']:
        raise RuntimeError('V1 Native start schedule semantic SHA mismatch')
    if manifest['rank_quantum'] != QUANTUM or manifest['ratio'] != 0.5:
        raise RuntimeError('V3 rank quantum or ratio mismatch')
    start = normalize_selection(read_json(start_path)['selection'])
    if len(start) != 64 or sum(len(v) for v in start.values()) != 512:
        raise RuntimeError('V1 start allocation does not contain 512 native units')
    for ranks in start.values():
        if len(ranks) != 8 or any(r < FLOOR or r > CEILING or r % QUANTUM for r in ranks):
            raise RuntimeError('V1 start allocation violates legal native rank lattice')
    expected_resource = manifest['resource']
    if resource(start) != expected_resource:
        raise RuntimeError(f'V1 start resource mismatch: {resource(start)} != {expected_resource}')
    return manifest, split, start


def load_calibration_inputs(tokenizer, split, split_name):
    doc = split[split_name]
    wiki_src = Path(split['sources']['wiki_train']['path'])
    c4_src = Path(split['sources']['c4_train']['path'])
    if sha_file(wiki_src) != split['sources']['wiki_train']['sha256']:
        raise RuntimeError('V3 Wiki train source SHA mismatch')
    if sha_file(c4_src) != split['sources']['c4_train']['sha256']:
        raise RuntimeError('V3 C4 train source SHA mismatch')
    wiki_text = wiki_src.read_text(encoding='utf-8')
    wiki = tokenizer(wiki_text, return_tensors='pt', add_special_tokens=False).input_ids[0]
    for chunk in doc['wikitext_chunks']:
        ids = wiki[int(chunk['token_start']):int(chunk['token_end'])].tolist()
        if len(ids) != SEQ or sha_ids(ids) != chunk['token_ids_sha256']:
            raise RuntimeError(f'V3 Wiki token binding mismatch: {chunk["sample_id"]}')
    c4rows = {int(x['doc_index']): x for x in doc['c4_chunks']}
    wanted = set(c4rows)
    c4texts = {}
    last = max(wanted)
    with gzip.open(c4_src, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i in wanted:
                c4texts[i] = json.loads(line)['text']
            if i >= last:
                break
    if set(c4texts) != wanted:
        raise RuntimeError(f'V3 C4 train docs missing: {sorted(wanted - set(c4texts))}')
    c4 = {}
    for i, chunk in c4rows.items():
        ids = tokenizer(c4texts[i], return_tensors='pt', add_special_tokens=False).input_ids
        if ids.shape[1] != int(chunk['token_count']):
            raise RuntimeError(f'V3 C4 token count mismatch at line {i}')
        start, end = int(chunk['token_start']), int(chunk['token_end'])
        sliced = ids[0, start:end].tolist()
        if len(sliced) != SEQ or sha_ids(sliced) != chunk['token_ids_sha256']:
            raise RuntimeError(f'V3 C4 token binding mismatch at line {i}')
        c4[i] = ids[:, start:end].contiguous()
    manifest = {
        'wikitext': {'chunks': doc['wikitext_chunks']},
        'c4': {'chunks': doc['c4_chunks']},
    }
    return manifest, wiki, c4


def load_runtime(engine, profile, selection):
    profile.install_hadamard()
    from utils import load_model_and_tokenizer
    from palu.model.modules.svd_linear import HeadwiseLowRankModule
    model, tokenizer = load_model_and_tokenizer(str(MODEL_PATH))
    model.to(torch.device('cuda'))
    raw, parents = {}, {}
    for name, module in list(model.named_modules()):
        if ('k_proj' in name or 'v_proj' in name) and isinstance(module, torch.nn.Linear):
            raw[name] = module
            parents[name] = profile.parent_and_child(model, name)
    if len(raw) != 64:
        raise RuntimeError(f'expected 64 native K/V modules, found {len(raw)}')
    factor_path = V1 / 'cache' / 'official_whiten_factors_rank512.pt'
    factors = torch.load(factor_path, map_location='cpu', weights_only=False)
    installed = {}
    for name in sorted(selection):
        mod = profile.build_module(HeadwiseLowRankModule, raw[name], factors[name], selection[name], 'cuda')
        parent, child = parents[name]
        setattr(parent, child, mod)
        installed[name] = mod
    model.eval()
    return model, tokenizer, raw, parents, factors, installed, HeadwiseLowRankModule


def install_selection(profile, model_parts, before, after):
    model, tokenizer, raw, parents, factors, installed, klass = model_parts
    changed = {k for k in before if before[k] != after[k]}
    for name in sorted(changed):
        mod = profile.build_module(klass, raw[name], factors[name], after[name], 'cuda')
        parent, child = parents[name]
        setattr(parent, child, mod)
        installed[name] = mod
    return model_parts


@torch.no_grad()
def evaluate(model, split_manifest, wiki_ids, c4_ids):
    device = next(model.parameters()).device
    wiki, c4 = [], []
    for chunk in split_manifest['wikitext']['chunks']:
        ids = wiki_ids[int(chunk['token_start']):int(chunk['token_end'])].unsqueeze(0).to(device)
        pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos,
                       use_cache=False).logits
        val = float(F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                                    ids[:, 1:].reshape(-1), reduction='mean').item())
        if not math.isfinite(val):
            raise FloatingPointError('non-finite V3 Wiki calibration NLL')
        wiki.append(val)
    for chunk in split_manifest['c4']['chunks']:
        ids = c4_ids[int(chunk['doc_index'])].to(device)
        pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos,
                       use_cache=False).logits
        val = float(F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                                    ids[:, 1:].reshape(-1), reduction='mean').item())
        if not math.isfinite(val):
            raise FloatingPointError('non-finite V3 C4 calibration NLL')
        c4.append(val)
    return {'wiki': wiki, 'c4': c4, 'J': 0.5 * (sum(wiki) / len(wiki) + sum(c4) / len(c4)),
            'wiki_nll': sum(wiki) / len(wiki), 'c4_nll': sum(c4) / len(c4)}


def load_eval_cache(split_name, allocation, split_manifest, wiki, c4):
    a_sha = allocation_sha(allocation)
    p = BASE / 'eval_cache' / split_name / (a_sha + '.json')
    if p.exists():
        rec = read_json(p)
        if (rec.get('allocation_sha256') != a_sha or rec.get('split') != split_name
                or rec.get('manifest_sha256') != sha_file(MANIFEST_PATH)):
            raise RuntimeError(f'evaluation cache provenance mismatch: {p}')
        return rec['evaluation']
    model_parts = load_eval_cache.model_parts
    current = load_eval_cache.current_selection
    install_selection(load_eval_cache.profile, model_parts, current, allocation)
    load_eval_cache.current_selection = copy.deepcopy(allocation)
    result = evaluate(model_parts[0], split_manifest, wiki, c4)
    write_json(p, {'split': split_name, 'allocation_sha256': a_sha,
                   'manifest_sha256': sha_file(MANIFEST_PATH), 'evaluation': result})
    return result


def bind_eval_cache(model_parts, profile, start):
    load_eval_cache.model_parts = model_parts
    load_eval_cache.profile = profile
    load_eval_cache.current_selection = copy.deepcopy(start)


def chunks_for_half(split_manifest, half):
    out = copy.deepcopy(split_manifest)
    out['wikitext']['chunks'] = out['wikitext']['chunks'][half * 4:(half + 1) * 4]
    out['c4']['chunks'] = out['c4']['chunks'][half * 4:(half + 1) * 4]
    return out


def unit_parts(unit):
    module, group = unit.rsplit('::group', 1)
    return module, int(group)


def list_units(selection):
    return [f'{module}::group{group}' for module in sorted(selection) for group in range(len(selection[module]))]


def spearman(values_a, values_b):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: (values[i], i))
        result = [0.0] * len(values)
        pos = 0
        while pos < len(order):
            end = pos + 1
            while end < len(order) and values[order[end]] == values[order[pos]]:
                end += 1
            midrank = (pos + 1 + end) / 2.0
            for j in range(pos, end):
                result[order[j]] = midrank
            pos = end
        return result
    x, y = ranks(values_a), ranks(values_b)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float('nan')


def compute_marginals(engine, profile, model_parts, start, selection, split_manifest, wiki, c4,
                      split_name, modules=None, initial_stability=False, seed_data=None, force_refresh=False):
    units = [u for u in list_units(selection) if modules is None or unit_parts(u)[0] in modules]
    base = load_eval_cache(split_name, selection, split_manifest, wiki, c4)
    denom = QUANTUM / TOTAL_RANK
    values = {}
    out_path = BASE / 'profile' / (split_name + '_marginals_' + allocation_sha(selection) + '.json')
    existing = read_json(out_path) if out_path.exists() else (copy.deepcopy(seed_data) if seed_data else {
        'allocation_sha256': allocation_sha(selection), 'phi_minus': {}, 'phi_plus': {},
        'phi_plus_half0': {}, 'phi_plus_half1': {}})
    existing['allocation_sha256'] = allocation_sha(selection)
    for key in ('phi_minus', 'phi_plus', 'phi_plus_half0', 'phi_plus_half1'):
        existing.setdefault(key, {})
    half_manifests = [chunks_for_half(split_manifest, i) for i in range(2)] if initial_stability else None
    if initial_stability:
        half_bases = [load_eval_cache(split_name + f'_half{i}', selection, half_manifests[i], wiki, c4)
                      for i in range(2)]
    total = len(units)
    for index, unit in enumerate(units, 1):
        module, group = unit_parts(unit)
        rank = int(selection[module][group])
        row = {}
        if rank > FLOOR and (force_refresh or unit not in existing['phi_minus']):
            cand = copy.deepcopy(selection)
            cand[module][group] -= QUANTUM
            ev = load_eval_cache(split_name, cand, split_manifest, wiki, c4)
            row['phi_minus'] = (ev['J'] - base['J']) / denom
            if not math.isfinite(row['phi_minus']):
                raise FloatingPointError('nonfinite V3 phi-minus ' + unit)
            existing['phi_minus'][unit] = row['phi_minus']
        if rank < CEILING and (force_refresh or unit not in existing['phi_plus']):
            cand = copy.deepcopy(selection)
            cand[module][group] += QUANTUM
            if initial_stability:
                halves = []
                for half in range(2):
                    ev = load_eval_cache(split_name + f'_half{half}', cand, half_manifests[half], wiki, c4)
                    value = (ev['J'] - half_bases[half]['J']) / denom
                    if not math.isfinite(value):
                        raise FloatingPointError('nonfinite V3 phi-plus half ' + unit)
                    existing[f'phi_plus_half{half}'][unit] = value
                    halves.append(value)
                row['phi_plus'] = sum(halves) / 2.0
            else:
                ev = load_eval_cache(split_name, cand, split_manifest, wiki, c4)
                row['phi_plus'] = (ev['J'] - base['J']) / denom
            if not math.isfinite(row['phi_plus']):
                raise FloatingPointError('nonfinite V3 phi-plus ' + unit)
            existing['phi_plus'][unit] = row['phi_plus']
        if row:
            write_json(out_path, existing)
        if index % 16 == 0 or index == total:
            write_json(BASE / 'profile' / (split_name + '_progress.json'),
                       {'status': 'RUNNING', 'processed_units': index, 'units_in_scope': total,
                        'allocation_sha256': allocation_sha(selection), 'updated_utc': utc_now()})
    if initial_stability:
        eligible = [u for u in list_units(selection) if selection[unit_parts(u)[0]][unit_parts(u)[1]] < CEILING]
        h0 = [existing['phi_plus_half0'][u] for u in eligible if u in existing['phi_plus_half0']]
        h1 = [existing['phi_plus_half1'][u] for u in eligible if u in existing['phi_plus_half1']]
        complete_fraction = min(len(h0), len(h1)) / max(1, len(eligible))
        rho = spearman(h0, h1) if len(h0) == len(eligible) and len(h1) == len(eligible) else float('nan')
        stable = complete_fraction >= 0.95 and math.isfinite(rho) and rho >= 0.80
        existing['receiver_rule'] = 'PHI_PLUS' if stable else 'RECEIVER_RULE_CONTROLLED_VARIANT_PHI_MINUS_DESCENDING'
        existing['phi_plus_stability'] = {'eligible_units': len(eligible), 'finite_fraction': complete_fraction,
                                           'spearman_rho': rho, 'threshold': 0.80,
                                           'finite_fraction_threshold': 0.95, 'stable': stable}
    write_json(out_path, existing)
    return existing


def bootstrap_ci(delta, seed):
    rng = random.Random(seed)
    n_w, n_c = len(delta['wiki']), len(delta['c4'])
    vals = []
    for _ in range(BOOT_N):
        w = sum(delta['wiki'][rng.randrange(n_w)] for _ in range(n_w)) / n_w
        c = sum(delta['c4'][rng.randrange(n_c)] for _ in range(n_c)) / n_c
        vals.append(0.5 * (w + c))
    vals.sort()
    return [vals[int(0.025 * BOOT_N)], vals[int(0.975 * BOOT_N)]]


def compare(current, candidate, seed):
    delta = {'wiki': [b - a for a, b in zip(current['wiki'], candidate['wiki'])],
             'c4': [b - a for a, b in zip(current['c4'], candidate['c4'])]}
    dj = 0.5 * (sum(delta['wiki']) / len(delta['wiki']) + sum(delta['c4']) / len(delta['c4']))
    ci = bootstrap_ci(delta, seed)
    return dj, ci, delta


def current_phi(profile_data, selection):
    minus = profile_data.get('phi_minus', {})
    plus = profile_data.get('phi_plus', {})
    rule = profile_data.get('receiver_rule', 'PHI_PLUS')
    donors = []
    receivers = []
    for unit in list_units(selection):
        module, group = unit_parts(unit)
        rank = int(selection[module][group])
        if rank > FLOOR and unit in minus and math.isfinite(float(minus[unit])):
            donors.append((float(minus[unit]), unit))
        if rule == 'PHI_PLUS' and rank < CEILING and unit in plus and math.isfinite(float(plus[unit])):
            receivers.append((float(plus[unit]), unit))
        elif rank < CEILING and rule != 'PHI_PLUS' and unit in minus and math.isfinite(float(minus[unit])):
            receivers.append((-float(minus[unit]), unit))
    return sorted(donors), sorted(receivers)


def candidate_pairs(selection, profile_data, d_count, r_count, rmax, start):
    donors, receivers = current_phi(profile_data, selection)
    donors = donors[:d_count]
    receivers = receivers[:r_count]
    base_r = redistribution(selection, start)
    result = []
    seen = set()
    for dp, donor in donors:
        dm, dg = unit_parts(donor)
        if selection[dm][dg] <= FLOOR:
            continue
        for rp, receiver in receivers:
            if donor == receiver:
                continue
            rm, rg = unit_parts(receiver)
            if selection[rm][rg] >= CEILING:
                continue
            nxt = apply_pair(selection, donor, receiver)
            if nxt is None or redistribution(nxt, start) > rmax + 1e-15:
                continue
            key = (donor, receiver)
            if key in seen:
                continue
            seen.add(key)
            result.append({'donor': donor, 'receiver': receiver, 'predicted_delta': dp + rp,
                           'allocation': nxt, 'allocation_sha256': allocation_sha(nxt)})
    return sorted(result, key=lambda x: (x['predicted_delta'], x['donor'], x['receiver']))[:256]


def apply_pair(selection, donor, receiver):
    out = copy.deepcopy(selection)
    dm, dg = unit_parts(donor)
    rm, rg = unit_parts(receiver)
    if donor == receiver or out[dm][dg] <= FLOOR or out[rm][rg] >= CEILING:
        return None
    out[dm][dg] -= QUANTUM
    out[rm][rg] += QUANTUM
    return out


def redistribution(selection, start):
    moved = sum(abs(int(selection[m][g]) - int(start[m][g]))
                for m in start for g in range(len(start[m])))
    return 0.5 * moved / TOTAL_RANK


def strict_gate(delta, ci):
    return delta < 0.0 and ci[1] < 0.0


def config_grid(arm):
    for d, r in [(4, 4), (8, 8), (16, 16)]:
        for rmax in [0.0025, 0.005, 0.01, 0.02]:
            beams = [4, 8] if arm in {'C', 'D'} else [0]
            for beam in beams:
                yield {'D': d, 'R': r, 'Rmax': rmax, 'beam_width': beam}


def config_id(arm, cfg):
    return f"arm{arm}_D{cfg['D']}_R{cfg['R']}_Rmax{cfg['Rmax']:g}_beam{cfg['beam_width']}"


def config_seed(config):
    return int(hashlib.sha256(config.encode()).hexdigest()[:8], 16)


def trajectory_from_order(accepted, composites, action_order):
    trajectory = []
    for ref in action_order:
        kind, index = ref.get('type'), int(ref.get('index', -1))
        if kind == 'single' and 0 <= index < len(accepted):
            row = accepted[index]
            trajectory.append({'pairs': [{'donor': row['donor'], 'receiver': row['receiver']}], 'type': 'single'})
        elif kind == 'composite' and 0 <= index < len(composites):
            trajectory.append({'pairs': composites[index]['pairs'], 'type': 'composite'})
        else:
            raise RuntimeError('invalid action-order reference in V3 TUNE checkpoint/result')
    if (sum(x['type'] == 'single' for x in trajectory) != len(accepted)
            or sum(x['type'] == 'composite' for x in trajectory) != len(composites)):
        raise RuntimeError('V3 TUNE action order does not cover every accepted action exactly once')
    return trajectory


def arm_tune(engine, profile, model_parts, profile_inputs_tuple, tune_tuple, start, arm, cfg):
    pid, pw, pc = profile_inputs_tuple
    tid, tw, tc = tune_tuple
    cid = config_id(arm, cfg)
    out_dir = BASE / 'CALIBRATION_TUNE' / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / 'DONE.json'
    if done.exists():
        return read_json(done)
    tune_phi_path = BASE / 'profile' / ('PROFILE_marginals_' + allocation_sha(start) + '.json')
    checkpoint_path = out_dir / 'CHECKPOINT.json'
    if checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        if (checkpoint.get('config_id') != cid or checkpoint.get('arm') != arm
                or checkpoint.get('start_allocation_sha256') != allocation_sha(start)):
            raise RuntimeError(f'TUNE checkpoint identity mismatch for {cid}')
        selection = checkpoint['selection']
        accepted = checkpoint['accepted']
        composites = checkpoint['composites']
        rejected = checkpoint['rejected']
        action_order = checkpoint['action_order']
        tabu = {(x[0], x[1], x[2]): x[3] for x in checkpoint['tabu_entries']}
        pair_history = {tuple(x) for x in checkpoint['pair_history']}
        phi_data = checkpoint['phi_data']
        last_global_exchange_count = int(checkpoint['last_global_exchange_count'])
        refresh_count = int(checkpoint['refresh_count'])
        global_refresh_count = int(checkpoint['global_refresh_count'])
        affected_refresh_count = int(checkpoint['affected_refresh_count'])
        revisit_count = int(checkpoint['revisit_count'])
        candidate_evaluated_total = int(checkpoint['candidate_evaluated_total'])
        single_candidate_coverage = int(checkpoint['single_candidate_coverage'])
        composite_candidate_coverage = int(checkpoint['composite_candidate_coverage'])
        replay = copy.deepcopy(start)
        for action in trajectory_from_order(accepted, composites, action_order):
            for pair in action['pairs']:
                replay = apply_pair(replay, pair['donor'], pair['receiver'])
                if replay is None:
                    raise RuntimeError(f'TUNE checkpoint trajectory is illegal for {cid}')
        if allocation_sha(replay) != allocation_sha(selection) or resource(selection) != resource(start):
            raise RuntimeError(f'TUNE checkpoint replay/resource mismatch for {cid}')
        current = load_eval_cache('CALIBRATION_TUNE', selection, tid, tw, tc)
    else:
        phi_data = read_json(tune_phi_path)
        selection = copy.deepcopy(start)
        accepted, composites, rejected, action_order = [], [], [], []
        tabu = {}
        pair_history = set()
        current = load_eval_cache('CALIBRATION_TUNE', selection, tid, tw, tc)
        last_global_exchange_count = 0
        refresh_count, global_refresh_count, affected_refresh_count, revisit_count = 0, 0, 0, 0
        candidate_evaluated_total = single_candidate_coverage = composite_candidate_coverage = 0

    def checkpoint_now():
        write_json(checkpoint_path, {'status': 'TUNE_CONFIG_CHECKPOINT', 'config_id': cid, 'arm': arm,
            'start_allocation_sha256': allocation_sha(start), 'selection': selection, 'accepted': accepted,
            'composites': composites, 'rejected': rejected, 'action_order': action_order,
            'tabu_entries': [[k[0], k[1], k[2], v] for k, v in sorted(tabu.items())],
            'pair_history': [list(x) for x in sorted(pair_history)], 'phi_data': phi_data,
            'last_global_exchange_count': last_global_exchange_count,
            'refresh_count': refresh_count, 'global_refresh_count': global_refresh_count,
            'affected_refresh_count': affected_refresh_count, 'revisit_count': revisit_count,
            'candidate_evaluated_total': candidate_evaluated_total,
            'single_candidate_coverage': single_candidate_coverage,
            'composite_candidate_coverage': composite_candidate_coverage,
            'current_tune': current, 'updated_utc': utc_now()})

    max_exchange_moves = max(1, int(math.floor(cfg['Rmax'] * TOTAL_RANK / QUANTUM)))
    while len(accepted) + 2 * len(composites) < max_exchange_moves:
        state_sha = allocation_sha(selection)
        pool = candidate_pairs(selection, phi_data, cfg['D'], cfg['R'], cfg['Rmax'], start)
        candidate_evaluated_total += len(pool)
        single_candidate_coverage += len(pool)
        candidates = []
        for item in pool:
            pair = (item['donor'], item['receiver'])
            if pair in pair_history:
                revisit_count += 1
            pair_history.add(pair)
            if tabu.get((state_sha, pair[0], pair[1])):
                continue
            ev = load_eval_cache('CALIBRATION_TUNE', item['allocation'], tid, tw, tc)
            dj, ci, delta = compare(current, ev, config_seed(cid + state_sha + str(pair)))
            gate = strict_gate(dj, ci) and resource(item['allocation']) == resource(start)
            row = {'donor': pair[0], 'receiver': pair[1], 'predicted_delta': item['predicted_delta'],
                   'actual_delta_J': dj, 'paired_ci95': ci, 'accepted_gate': gate,
                   'allocation_sha256': item['allocation_sha256'], 'resource_exact': resource(item['allocation']) == resource(start)}
            candidates.append((row, item['allocation']))
            if not gate:
                reason = 'RESOURCE_MISMATCH' if not row['resource_exact'] else ('DELTA_NONNEGATIVE' if dj >= 0 else 'CI_UPPER_NOT_BELOW_ZERO')
                tabu[(state_sha, pair[0], pair[1])] = reason
                rejected.append({**row, 'reason': reason, 'state_sha256': state_sha})
        valid = [(row, alloc) for row, alloc in candidates if row['accepted_gate']]
        if valid:
            row, nxt = min(valid, key=lambda x: (x[0]['actual_delta_J'], x[0]['donor'], x[0]['receiver']))
            accepted.append({**row, 'before_sha256': state_sha, 'after_sha256': allocation_sha(nxt), 'type': 'single'})
            action_order.append({'type': 'single', 'index': len(accepted) - 1})
            current = load_eval_cache('CALIBRATION_TUNE', nxt, tid, tw, tc)
            selection = nxt
            changed_modules = {unit_parts(row['donor'])[0], unit_parts(row['receiver'])[0]}
            exchanges = len(accepted) + 2 * len(composites)
            if arm in {'B', 'D'}:
                phi_data = merge_marginals(phi_data, compute_marginals(engine, profile, model_parts, start,
                    selection, pid, pw, pc, 'PROFILE', modules=changed_modules, initial_stability=False,
                    seed_data=phi_data, force_refresh=True))
                affected_refresh_count += 1; refresh_count += 1
            if exchanges - last_global_exchange_count >= 4:
                phi_data = compute_marginals(engine, profile, model_parts, start, selection, pid, pw, pc,
                                              'PROFILE', modules=None, initial_stability=False, seed_data=phi_data,
                                              force_refresh=True)
                last_global_exchange_count = exchanges
                global_refresh_count += 1; refresh_count += 1
            checkpoint_now()
            continue
        if not pool:
            break
        if arm not in {'C', 'D'} or cfg['beam_width'] <= 0:
            break
        comp_candidates = depth2_candidates(selection, pool, cfg, start, rmax=cfg['Rmax'])
        candidate_evaluated_total += len(comp_candidates)
        composite_candidate_coverage += len(comp_candidates)
        valid_comp = []
        for cand in comp_candidates:
            ev = load_eval_cache('CALIBRATION_TUNE', cand['allocation'], tid, tw, tc)
            dj, ci, _ = compare(current, ev, config_seed(cid + state_sha + cand['allocation_sha256']))
            gate = strict_gate(dj, ci) and resource(cand['allocation']) == resource(start)
            row = {'pairs': cand['pairs'], 'predicted_delta': cand['predicted_delta'], 'actual_delta_J': dj,
                   'paired_ci95': ci, 'accepted_gate': gate, 'allocation_sha256': cand['allocation_sha256'],
                   'resource_exact': resource(cand['allocation']) == resource(start), 'type': 'composite'}
            if gate:
                valid_comp.append((row, cand['allocation']))
            else:
                reason = 'RESOURCE_MISMATCH' if not row['resource_exact'] else ('DELTA_NONNEGATIVE' if dj >= 0 else 'CI_UPPER_NOT_BELOW_ZERO')
                rejected.append({**row, 'reason': reason, 'state_sha256': state_sha})
        if not valid_comp:
            break
        row, nxt = min(valid_comp, key=lambda x: (x[0]['actual_delta_J'], str(x[0]['pairs'])))
        composites.append({**row, 'before_sha256': state_sha, 'after_sha256': allocation_sha(nxt)})
        action_order.append({'type': 'composite', 'index': len(composites) - 1})
        selection = nxt
        current = load_eval_cache('CALIBRATION_TUNE', selection, tid, tw, tc)
        changed_modules = {unit_parts(pair[k])[0] for pair in row['pairs'] for k in ('donor', 'receiver')}
        exchanges = len(accepted) + 2 * len(composites)
        if arm in {'B', 'D'}:
            phi_data = merge_marginals(phi_data, compute_marginals(engine, profile, model_parts, start,
                selection, pid, pw, pc, 'PROFILE', modules=changed_modules, initial_stability=False,
                seed_data=phi_data, force_refresh=True))
            affected_refresh_count += 1; refresh_count += 1
        if exchanges - last_global_exchange_count >= 4:
            phi_data = compute_marginals(engine, profile, model_parts, start, selection, pid, pw, pc,
                                          'PROFILE', modules=None, initial_stability=False, seed_data=phi_data,
                                          force_refresh=True)
            last_global_exchange_count = exchanges
            global_refresh_count += 1; refresh_count += 1
        checkpoint_now()
    start_eval = load_eval_cache('CALIBRATION_TUNE', start, tid, tw, tc)
    final_eval = load_eval_cache('CALIBRATION_TUNE', selection, tid, tw, tc)
    total_delta, total_ci, _ = compare(start_eval, final_eval, config_seed(cid + 'TOTAL'))
    units_changed = sum(1 for m in start for g in range(8) if start[m][g] != selection[m][g])
    moved_rank = int(round(redistribution(selection, start) * 2 * TOTAL_RANK))
    trajectory = trajectory_from_order(accepted, composites, action_order)
    result = {'status': 'TUNE_CONFIG_COMPLETE', 'arm': arm, 'config': cfg, 'config_id': cid,
              'accepted_single_count': len(accepted), 'accepted_composite_count': len(composites),
              'accepted_exchange_count': len(accepted) + 2 * len(composites),
              'changed_native_units': units_changed, 'total_moved_rank': moved_rank,
              'moved_factor_bytes': moved_rank * (4096 + 512) * 2,
              'redistribution_fraction_R': redistribution(selection, start),
              'candidate_coverage_evaluated': candidate_evaluated_total,
              'single_candidate_coverage_evaluated': single_candidate_coverage,
              'composite_candidate_coverage_evaluated': composite_candidate_coverage,
              'rejected_count': len(rejected),
              'revisit_count': revisit_count, 'refresh_count': refresh_count,
              'affected_refresh_count': affected_refresh_count, 'global_refresh_count': global_refresh_count,
              'start_J_tune': start_eval['J'], 'final_J_tune': final_eval['J'],
              'delta_J_tune_vs_v1': total_delta, 'paired_ci95_tune_vs_v1': total_ci,
              'tune_strict_vs_v1': strict_gate(total_delta, total_ci),
              'resource': resource(selection), 'allocation_sha256': allocation_sha(selection),
              'l1_rank_moved': moved_rank, 'transfer_volume_rank': moved_rank // 2,
              'selection': selection, 'accepted_singles': accepted, 'accepted_composites': composites,
              'rejected': rejected, 'action_order': action_order, 'trajectory': trajectory,
              'foreign_jobs_interrupted': 0}
    write_json(done, result)
    return result


def merge_marginals(original, changed):
    out = copy.deepcopy(original)
    for key in ('phi_minus', 'phi_plus'):
        out.setdefault(key, {}).update(changed.get(key, {}))
    out['allocation_sha256'] = changed.get('allocation_sha256', out.get('allocation_sha256'))
    return out


def depth2_candidates(selection, pool, cfg, start, rmax):
    width = int(cfg['beam_width'])
    beam = pool[:width]
    out, seen = [], set()
    for first in beam:
        inter = first['allocation']
        first_pair = {'donor': first['donor'], 'receiver': first['receiver']}
        for second in pool:
            if (second['donor'], second['receiver']) == (first['donor'], first['receiver']):
                continue
            after = apply_pair(inter, second['donor'], second['receiver'])
            if after is None or redistribution(after, start) > rmax + 1e-15:
                continue
            a_sha = allocation_sha(after)
            if a_sha == allocation_sha(selection) or a_sha in seen:
                continue
            seen.add(a_sha)
            out.append({'pairs': [first_pair, {'donor': second['donor'], 'receiver': second['receiver']}],
                        'predicted_delta': first['predicted_delta'] + second['predicted_delta'],
                        'allocation': after, 'allocation_sha256': a_sha})
    return sorted(out, key=lambda x: (x['predicted_delta'], x['allocation_sha256']))


def profile_phase(engine, profile, model_parts, inputs, start):
    out = BASE / 'profile'
    out.mkdir(parents=True, exist_ok=True)
    done = out / 'PROFILE_DONE.json'
    if done.exists():
        return read_json(done)
    set_state(status='V3_PROFILE_RUNNING', calibration_started=True,
              next_action='Finish full V1-final phi_minus/phi_plus profile on frozen PROFILE split.')
    split_manifest, wiki, c4 = inputs
    bind_eval_cache(model_parts, profile, start)
    data = compute_marginals(engine, profile, model_parts, start, start, split_manifest, wiki, c4,
                             'PROFILE', initial_stability=True)
    result = {'status': 'PROFILE_COMPLETE', 'allocation_sha256': allocation_sha(start),
              'phi_minus_count': len(data['phi_minus']), 'phi_plus_count': len(data['phi_plus']),
              'receiver_rule': data['receiver_rule'], 'phi_plus_stability': data['phi_plus_stability'],
              'resource': resource(start), 'manifest_sha256': sha_file(MANIFEST_PATH),
              'profile_marginals_sha256': sha_file(out / ('PROFILE_marginals_' + allocation_sha(start) + '.json')),
              'foreign_jobs_interrupted': 0}
    write_json(done, result)
    set_state(status='V3_PROFILE_COMPLETE', calibration_started=True,
              profile_result_sha256=sha_file(done), next_action='Run frozen TUNE grids for arms A-D.')
    return result


def tune_phase(engine, profile, model_parts, profile_inputs_tuple, tune_tuple, start, arms):
    # Unlike PROFILE and LOCK, the original TUNE path did not bind the
    # process-local evaluation cache context before requesting its baseline.
    bind_eval_cache(model_parts, profile, start)
    set_state(status='V3_TUNE_RUNNING', calibration_started=True,
              next_action='Evaluate frozen D/R, Rmax and beam grids on CALIBRATION-TUNE only.')
    all_results = []
    for arm in arms:
        for cfg in config_grid(arm):
            result = arm_tune(engine, profile, model_parts, profile_inputs_tuple, tune_tuple, start, arm, cfg)
            all_results.append(result)
    tune_root = BASE / 'CALIBRATION_TUNE'
    for arm in arms:
        candidates = [x for x in all_results if x['arm'] == arm]
        strict = [x for x in candidates if x['tune_strict_vs_v1']]
        selected = min(strict or candidates, key=lambda x: (x['final_J_tune'], x['config_id']))
        write_json(tune_root / f'ARM_{arm}_SELECTED.json', {
            'status': 'ARM_TUNE_CONFIG_SELECTED', 'selection_rule': 'minimum final TUNE J among strict candidates; if none, minimum final TUNE J and lock will decide',
            'selected_config_id': selected['config_id'], 'tune_strict_vs_v1': selected['tune_strict_vs_v1'],
            'final_J_tune': selected['final_J_tune'], 'delta_J_tune_vs_v1': selected['delta_J_tune_vs_v1'],
            'paired_ci95_tune_vs_v1': selected['paired_ci95_tune_vs_v1'], 'foreign_jobs_interrupted': 0})
    done = {'status': 'TUNE_GRID_COMPLETE', 'config_count': len(all_results),
            'arm_count': len(arms), 'foreign_jobs_interrupted': 0}
    write_json(tune_root / 'TUNE_DONE.json', done)
    set_state(status='V3_TUNE_COMPLETE', calibration_started=True,
              tune_summary_sha256=sha_file(tune_root / 'TUNE_DONE.json'), next_action='Replay selected arm trajectories on independent CALIBRATION-LOCK.')
    return done


def lock_phase(engine, profile, model_parts, lock_inputs, tune_inputs, start, arms):
    lock_root = BASE / 'CALIBRATION_LOCK'
    lock_root.mkdir(parents=True, exist_ok=True)
    set_state(status='V3_LOCK_RUNNING', calibration_started=True,
              next_action='Replay frozen TUNE trajectories on independent CALIBRATION-LOCK; no re-optimization.')
    split_manifest, wiki, c4 = lock_inputs
    tune_manifest, tune_wiki, tune_c4 = tune_inputs
    bind_eval_cache(model_parts, profile, start)
    start_eval = load_eval_cache('CALIBRATION_LOCK', start, split_manifest, wiki, c4)
    lock_results = []
    for arm in arms:
        selected_doc = read_json(BASE / 'CALIBRATION_TUNE' / f'ARM_{arm}_SELECTED.json')
        tune = read_json(BASE / 'CALIBRATION_TUNE' / selected_doc['selected_config_id'] / 'DONE.json')
        selection = copy.deepcopy(start)
        replay = []
        for action in tune['trajectory']:
            for pair in action['pairs']:
                nxt = apply_pair(selection, pair['donor'], pair['receiver'])
                if nxt is None or resource(nxt) != resource(start):
                    raise RuntimeError(f'LOCK trajectory replay illegal for arm {arm}')
                selection = nxt
            replay.append({'type': action['type'], 'pairs': action['pairs'], 'allocation_sha256': allocation_sha(selection)})
        if allocation_sha(selection) != tune['allocation_sha256']:
            raise RuntimeError(f'LOCK trajectory replay SHA mismatch for arm {arm}')
        final_eval = load_eval_cache('CALIBRATION_LOCK', selection, split_manifest, wiki, c4)
        delta, ci, _ = compare(start_eval, final_eval, config_seed(arm + tune['config_id'] + 'LOCK'))
        passed = strict_gate(delta, ci) and resource(selection) == resource(start)
        result = {'status': 'LOCK_PASS' if passed else 'LOCK_FAIL', 'arm': arm,
                  'config_id': tune['config_id'], 'allocation_sha256': allocation_sha(selection),
                  'selection': selection, 'start_J_lock': start_eval['J'], 'final_J_lock': final_eval['J'],
                  'delta_J_lock_vs_v1': delta, 'paired_ci95_lock_vs_v1': ci, 'strict_lock_pass': passed,
                  'accepted_single_count': tune['accepted_single_count'],
                  'accepted_composite_count': tune['accepted_composite_count'],
                  'accepted_exchange_count': tune['accepted_exchange_count'],
                  'changed_native_units': tune['changed_native_units'], 'total_moved_rank': tune['total_moved_rank'],
                  'redistribution_fraction_R': redistribution(selection, start), 'resource': resource(selection),
                  'trajectory': replay, 'tune_result_sha256': sha_file(BASE / 'CALIBRATION_TUNE' / tune['config_id'] / 'DONE.json'),
                  'foreign_jobs_interrupted': 0}
        write_json(lock_root / f'ARM_{arm}_LOCK.json', result)
        lock_results.append(result)
    passing = [x for x in lock_results if x['strict_lock_pass']]
    if passing:
        order = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        passing.sort(key=lambda x: (order[x['arm']], int(read_json(BASE / 'CALIBRATION_TUNE' / x['config_id'] / 'DONE.json')['config']['beam_width'] or 0),
                                    float(read_json(BASE / 'CALIBRATION_TUNE' / x['config_id'] / 'DONE.json')['config']['Rmax']),
                                    int(read_json(BASE / 'CALIBRATION_TUNE' / x['config_id'] / 'DONE.json')['config']['D']),
                                    x['accepted_composite_count']))
        chosen = passing[0]
        chosen_doc = {'status': 'V3_CALIBRATION_LOCK_FROZEN', 'chosen_arm': chosen['arm'],
                      'chosen_config_id': chosen['config_id'], 'allocation_sha256': chosen['allocation_sha256'],
                      'selection': chosen['selection'], 'resource': chosen['resource'],
                      'lock_delta_J_vs_v1': chosen['delta_J_lock_vs_v1'],
                      'lock_ci95_vs_v1': chosen['paired_ci95_lock_vs_v1'],
                      'trajectory': chosen['trajectory'], 'method_card_sha256': sha_file(CARD_PATH),
                      'manifest_sha256': sha_file(MANIFEST_PATH), 'foreign_jobs_interrupted': 0}
        write_json(lock_root / 'V3_LOCK.json', chosen_doc)
        set_state(status='V3_LOCK_PASS', lock_result_sha256=sha_file(lock_root / 'V3_LOCK.json'),
                  chosen_arm=chosen['arm'], chosen_config_id=chosen['config_id'],
                  next_action='Run frozen official Wiki/C4 formal PPL; do not tune after formal.')
    else:
        write_json(lock_root / 'V3_LOCK.json', {'status': 'NO_V3_GO_QUALITY',
            'arms': [{'arm': x['arm'], 'delta_J_lock_vs_v1': x['delta_J_lock_vs_v1'],
                      'paired_ci95_lock_vs_v1': x['paired_ci95_lock_vs_v1']} for x in lock_results],
            'foreign_jobs_interrupted': 0})
        set_state(status='NO_V3_GO_QUALITY', lock_result_sha256=sha_file(lock_root / 'V3_LOCK.json'),
                  next_action='Do not run formal V3 PPL; preserve calibration evidence.')
    return read_json(lock_root / 'V3_LOCK.json')


def formal_phase(engine, profile, model_parts, start, lock_doc):
    out = BASE / 'FORMAL_PPL'
    out.mkdir(parents=True, exist_ok=True)
    done = out / 'DONE.json'
    if done.exists():
        return read_json(done)
    if lock_doc.get('status') != 'V3_CALIBRATION_LOCK_FROZEN':
        raise RuntimeError('formal V3 PPL is gated on a strict calibration lock pass')
    set_state(status='V3_FORMAL_PPL_RUNNING', formal_ppl_started=True,
              next_action='Evaluate Fisher, frozen V1 Native and locked V3 on the official formal windows.')
    formal_module = importlib.import_module('run_official_formal_quality')
    model, tokenizer = model_parts[0], model_parts[1]
    fisher_doc = read_json(V1 / 'results' / 'fisher_0p5_ratio_0.5_fisher_uniform.json')
    fisher_sel = normalize_selection(fisher_doc['selection'])
    v3_sel = normalize_selection(lock_doc['selection'])
    control = formal_module.windows
    install_selection(profile, model_parts, start, fisher_sel)
    fisher = control(model, tokenizer)
    install_selection(profile, model_parts, fisher_sel, start)
    v1 = control(model, tokenizer)
    install_selection(profile, model_parts, start, v3_sel)
    v3 = control(model, tokenizer)
    d_v1 = {'wiki': [b - a for a, b in zip(v1['wiki'], v3['wiki'])],
            'c4': [b - a for a, b in zip(v1['c4'], v3['c4'])]}
    d_f = {'wiki': [b - a for a, b in zip(fisher['wiki'], v3['wiki'])],
           'c4': [b - a for a, b in zip(fisher['c4'], v3['c4'])]}
    j_v1 = 0.5 * (sum(d_v1['wiki']) / len(d_v1['wiki']) + sum(d_v1['c4']) / len(d_v1['c4']))
    j_f = 0.5 * (sum(d_f['wiki']) / len(d_f['wiki']) + sum(d_f['c4']) / len(d_f['c4']))
    seed = 2026092305
    result = {'status': 'V3_FORMAL_PPL_COMPLETE', 'experiment': 'PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3',
              'ratio': 0.5, 'chosen_arm': lock_doc['chosen_arm'], 'chosen_config_id': lock_doc['chosen_config_id'],
              'fisher': fisher, 'v1_native': v1, 'v3': v3,
              'delta_J_v3_vs_v1': j_v1, 'paired_ci95_v3_vs_v1': bootstrap_ci(d_v1, seed),
              'delta_J_v3_vs_fisher': j_f, 'paired_ci95_v3_vs_fisher': bootstrap_ci(d_f, seed + 1),
              'resource_fisher': resource(fisher_sel), 'resource_v1': resource(start),
              'resource_v3': resource(v3_sel), 'resource_match': 'EXACT' if resource(start) == resource(v3_sel) == resource(fisher_sel) else 'MISMATCH',
              'formal_manifest_sha256': read_json(SPLIT_PATH)['FORMAL_WIKI']['formal_protocol_manifest_sha256'],
              'v3_manifest_sha256': sha_file(MANIFEST_PATH), 'lock_sha256': sha_file(BASE / 'CALIBRATION_LOCK' / 'V3_LOCK.json'),
              'runner_sha256': sha_file(Path(__file__)), 'formal_quality_used_to_select': False,
              'foreign_jobs_interrupted': 0}
    write_json(done, result)
    set_state(status='V3_FORMAL_PPL_COMPLETE', formal_result_sha256=sha_file(done),
              next_action='If V3 movement exceeds V1, create quality-blind Random-Matched-V3 seeds 0/1/2; audit RULER protocol separately.')
    return result


def arm_phase(args):
    acquire_lock()
    engine, profile = load_modules()
    manifest, split, start = validate_frozen_inputs(engine)
    runner_sha = sha_file(Path(__file__))
    lock_path = BASE / 'RUNNER_LOCK.json'
    prior_state = read_json(STATE_PATH)
    runner_rebound = None
    if lock_path.exists():
        bound = read_json(lock_path)
        if bound.get('runner_sha256') != runner_sha:
            profile_done = BASE / 'profile' / 'PROFILE_DONE.json'
            profile_is_bound = (profile_done.exists()
                and prior_state.get('profile_result_sha256') == sha_file(profile_done)
                and read_json(profile_done).get('status') == 'PROFILE_COMPLETE')
            tune_root = BASE / 'CALIBRATION_TUNE'
            tune_cache_root = BASE / 'eval_cache' / 'CALIBRATION_TUNE'
            tune_outputs = (any(x.is_file() for x in tune_root.rglob('*')) if tune_root.exists() else False)
            tune_outputs = tune_outputs or (any(x.is_file() for x in tune_cache_root.rglob('*'))
                                             if tune_cache_root.exists() else False)
            lock_outputs = (BASE / 'CALIBRATION_LOCK').exists()
            formal_outputs = any((BASE / name).exists() for name in ('FORMAL', 'FORMAL_PPL', 'FORMAL_RESULTS'))
            formal_started = bool(prior_state.get('formal_ppl_started'))
            if args.phase not in {'preflight', 'tune'} or not profile_is_bound or tune_outputs or lock_outputs or formal_outputs or formal_started:
                raise RuntimeError('runner rebind is allowed only in the verified post-PROFILE/pre-TUNE window with no TUNE/LOCK/formal artifacts')
            addendum_no = len(list(BASE.glob('RUNNER_LOCK_ADDENDUM_*.json'))) + 1
            addendum = BASE / f'RUNNER_LOCK_ADDENDUM_{addendum_no:03d}.json'
            runner_rebound = {'status': 'RUNNER_LOCK_REBOUND_POST_PROFILE_PRE_TUNE',
                'old_runner_sha256': bound.get('runner_sha256'), 'new_runner_sha256': runner_sha,
                'affected_stage': 'PRE_TUNE_ENGINEERING_ONLY',
                'reason': 'Engineering-only TUNE runner correction after frozen PROFILE; no TUNE, LOCK, or formal output/cache exists.',
                'profile_result_sha256': sha_file(profile_done),
                'profile_marginals_sha256': read_json(profile_done).get('profile_marginals_sha256'),
                'tune_outputs_absent': True, 'lock_outputs_absent': True, 'formal_outputs_absent': True,
                'old_results_valid': True, 'experimental_outputs_written': False,
                'foreign_jobs_interrupted': 0, 'created_utc': utc_now()}
            write_json(addendum, runner_rebound)
            bound.update({'runner_sha256': runner_sha, 'last_addendum_path': str(addendum),
                          'last_addendum_sha256': sha_file(addendum), 'updated_utc': utc_now()})
            write_json(lock_path, bound)
    else:
        write_json(lock_path, {'status': 'V3_RUNNER_FROZEN_PRE_CALIBRATION', 'runner_path': str(Path(__file__)),
            'runner_sha256': runner_sha, 'manifest_sha256': sha_file(MANIFEST_PATH),
            'method_card_sha256': sha_file(CARD_PATH), 'split_binding_sha256': sha_file(SPLIT_PATH),
            'frozen_utc': utc_now(), 'foreign_jobs_interrupted': 0})
    set_state(runner_sha256=runner_sha, calibration_started=bool(read_json(STATE_PATH).get('calibration_started')),
              runner_status='FROZEN_CPU_PREFLIGHT_PASSED')
    if args.phase == 'preflight':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, use_fast=True)
        audits = {}
        for name in ['PROFILE', 'CALIBRATION_TUNE', 'CALIBRATION_LOCK']:
            manifest, wiki, c4 = load_calibration_inputs(tokenizer, split, name)
            audits[name] = {'wiki_chunks': len(manifest['wikitext']['chunks']),
                            'wiki_tokens': sum(int(x['sequence_length']) for x in manifest['wikitext']['chunks']),
                            'c4_chunks': len(manifest['c4']['chunks']),
                            'c4_tokens': sum(int(x['sequence_length']) for x in manifest['c4']['chunks']),
                            'wiki_tensor_tokens': int(wiki.numel()),
                            'c4_doc_indices': sorted(c4),
                            'token_bindings_verified': True}
        preflight = {'status': 'V3_CPU_PREFLIGHT_PASS', 'manifest_sha256': sha_file(MANIFEST_PATH),
                     'method_card_sha256': sha_file(CARD_PATH), 'split_binding_sha256': sha_file(SPLIT_PATH),
                     'runner_sha256': runner_sha, 'allocation_sha256': allocation_sha(start),
                     'resource': resource(start), 'splits': audits, 'formal_split_read': False,
                     'foreign_jobs_interrupted': 0}
        write_json(BASE / 'CPU_PREFLIGHT.json', preflight)
        audit_path = BASE / 'ENGINEERING_AUDIT.json'
        audit = read_json(audit_path) if audit_path.exists() else {'attempts': []}
        if runner_rebound and not any(x.get('failure_type') == 'V3_RUNNER_LOCK_POST_PROFILE_PRE_TUNE_REBIND' for x in audit['attempts']):
            audit['attempts'].append({**runner_rebound,
                'failure_type': 'V3_RUNNER_LOCK_POST_PROFILE_PRE_TUNE_REBIND',
                'repair': 'Versioned addendum binds engineering-corrected TUNE code after verified PROFILE and before any TUNE/LOCK/formal output.'})
        if not any(x.get('failure_type') == 'V3_TUNE_ENGINEERING_FIXES_002' for x in audit['attempts']):
            audit['attempts'].append({
                'failure': 'TUNE cached old marginals during refresh, grouped accepted composite actions after single actions in the LOCK trajectory, omitted composite candidate coverage, and did not restore checkpoints.',
                'failure_type': 'V3_TUNE_ENGINEERING_FIXES_002',
                'repair': 'Force recomputation for refreshed units; serialize/replay action acceptance order; count single and composite candidates separately; restore and validate full deterministic checkpoint state; report L1 moved rank separately from transfer volume.',
                'affected_files': ['run_rs_v3.py', 'RUNNER_LOCK.json', 'RUNNER_LOCK_ADDENDUM_002.json'],
                'affected_cells': ['V3 TUNE arms A-D only; TUNE has not started and no TUNE cells existed before this repair.'],
                'profile_result_sha256': runner_rebound.get('profile_result_sha256') if runner_rebound else prior_state.get('profile_result_sha256'),
                'old_results_valid': True, 'profile_result_unchanged': True,
                'experimental_outputs_written_before_repair': False,
                'foreign_jobs_interrupted': 0})
        if not any(x.get('failure_type') == 'V3_TUNE_EVAL_CACHE_CONTEXT_BINDING_MISSING' for x in audit['attempts']):
            audit['attempts'].append({
                'failure': 'First TUNE startup failed before evaluating its initial baseline because load_eval_cache.model_parts was never bound in tune_phase.',
                'failure_type': 'V3_TUNE_EVAL_CACHE_CONTEXT_BINDING_MISSING',
                'repair': 'Bind the existing model/profile/start-allocation evaluation context at TUNE phase entry before any candidate or baseline evaluation.',
                'affected_cell': 'TUNE arm A first configuration startup; no evaluation or result was produced.',
                'affected_files': ['run_rs_v3.py', 'ENGINEERING_AUDIT.json', 'STATE.json'],
                'failed_run_id': 'TUNE_ALL_GPU2_ADDENDUM_002_20260923T1010Z',
                'failed_worker_pid': 3667994,
                'failure_log': 'logs/TUNE_ALL_GPU2_ADDENDUM_002_20260923T1010Z.log',
                'partial_files_created': ['empty CALIBRATION_TUNE config directory only'],
                'eval_cache_written': False, 'checkpoint_written': False,
                'profile_result_sha256': sha_file(BASE / 'profile' / 'PROFILE_DONE.json'),
                'old_results_valid': True, 'formal_ppl_started': False,
                'foreign_jobs_interrupted': 0})
        prior_state_now = read_json(STATE_PATH)
        if ((prior_state_now.get('error') or prior_state_now.get('traceback'))
                and not any(x.get('verified_profile_result_sha256') == sha_file(BASE / 'profile' / 'PROFILE_DONE.json')
                            for x in audit.get('state_reconciliations', []))):
            audit.setdefault('state_reconciliations', []).append({
                'observed_state': 'V3_PROFILE_COMPLETE with stale runner-rebind exception fields',
                'prior_status': prior_state_now.get('status'),
                'prior_failure_type': prior_state_now.get('failure_type'),
                'prior_error': prior_state_now.get('error'),
                'prior_traceback': prior_state_now.get('traceback'),
                'verified_profile_result_sha256': sha_file(BASE / 'profile' / 'PROFILE_DONE.json'),
                'reconciliation': 'Retain exception in this audit record; successful preflight status update clears stale error fields from active state.',
                'old_results_valid': True, 'foreign_jobs_interrupted': 0,
                'recorded_utc': utc_now()})
        if not any(x.get('failure_type') == 'V3_Q_MAPPING_HASH_DOMAIN_CONFUSION' for x in audit['attempts']):
            audit['attempts'].append({
                'failure': 'Initial CPU preflight compared q_mapping_sha256 (semantic mapping SHA) to the JSON file-byte SHA.',
                'failure_type': 'V3_Q_MAPPING_HASH_DOMAIN_CONFUSION',
                'repair': 'Compare the q mapping semantic mapping_sha256 field to the frozen manifest; retain byte-SHA checks only for files whose manifest fields are byte SHAs.',
                'affected_cell': 'V3_CPU_PREFLIGHT_ONLY',
                'experimental_outputs_written': False,
                'old_results_valid': True,
                'foreign_jobs_interrupted': 0,
            })
        if not any(x.get('failure_type') == 'V3_ACTION_SPACE_HASH_DOMAIN_CONFUSION' for x in audit['attempts']):
            audit['attempts'].append({
                'failure': 'First hash-domain repair treated native_action_space_sha256 as an embedded semantic SHA; manifest binds the action-space JSON file bytes.',
                'failure_type': 'V3_ACTION_SPACE_HASH_DOMAIN_CONFUSION',
                'repair': 'Restore byte-SHA verification for the action-space file and do not compare its embedded semantic hash to the file SHA.',
                'affected_cell': 'V3_CPU_PREFLIGHT_ONLY',
                'experimental_outputs_written': False,
                'old_results_valid': True,
                'foreign_jobs_interrupted': 0,
            })
        if not any(x.get('failure_type') == 'V3_FORMAL_PROTOCOL_FILE_BINDING_FIX' for x in audit['attempts']):
            audit['attempts'].append({
                'failure': 'Preflight initially compared the formal-protocol digest to the generic Official protocol lock, not the frozen formal-quality protocol manifest bound in the split audit.',
                'failure_type': 'V3_FORMAL_PROTOCOL_FILE_BINDING_FIX',
                'repair': 'Verify the exact formal-quality protocol manifest at PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/quality/formal_protocol/FORMAL_QUALITY_PROTOCOL_MANIFEST.json.',
                'affected_cell': 'V3_CPU_PREFLIGHT_ONLY',
                'experimental_outputs_written': False,
                'old_results_valid': True,
                'foreign_jobs_interrupted': 0,
            })
        if not any(x.get('failure_type') == 'V3_TOKEN_ID_HASH_ENCODING_FIX' for x in audit['attempts']):
            audit['attempts'].append({
                'failure': 'CPU preflight initially used the comma-delimited token hash convention from V2; frozen V3 split token hashes bind little-endian int64 token IDs.',
                'failure_type': 'V3_TOKEN_ID_HASH_ENCODING_FIX',
                'repair': 'Verify each frozen V3 chunk using SHA256 over packed little-endian int64 token IDs, matching the precomputed V3 token binding.',
                'affected_cell': 'V3_CPU_PREFLIGHT_ONLY',
                'experimental_outputs_written': False,
                'old_results_valid': True,
                'foreign_jobs_interrupted': 0,
            })
        write_json(audit_path, audit)
        profile_done_path = BASE / 'profile' / 'PROFILE_DONE.json'
        profile_complete = profile_done_path.exists() and read_json(profile_done_path).get('status') == 'PROFILE_COMPLETE'
        next_action = ('Run frozen TUNE grids for arms A-D; PROFILE remains frozen and valid.' if profile_complete
                       else 'Run complete PROFILE marginal census on the isolated PROFILE split.')
        set_state(status='V3_PROFILE_COMPLETE' if profile_complete else 'V3_CPU_PREFLIGHT_PASS',
                  runner_status='FROZEN_CPU_PREFLIGHT_PASSED',
                  cpu_preflight_sha256=sha_file(BASE / 'CPU_PREFLIGHT.json'),
                  next_action=next_action)
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return
    model_parts = load_runtime(engine, profile, start)
    split_inputs = {}
    for name in ['PROFILE', 'CALIBRATION_TUNE', 'CALIBRATION_LOCK']:
        split_inputs[name] = load_calibration_inputs(model_parts[1], split, name)
    if args.phase in {'profile', 'all'}:
        profile_phase(engine, profile, model_parts, split_inputs['PROFILE'], start)
    if args.phase in {'tune', 'all'}:
        if not (BASE / 'profile' / 'PROFILE_DONE.json').exists():
            raise RuntimeError('TUNE cannot begin before PROFILE_DONE')
        tune_phase(engine, profile, model_parts, split_inputs['PROFILE'], split_inputs['CALIBRATION_TUNE'],
                   start, [args.arm] if args.arm != 'all' else ['A', 'B', 'C', 'D'])
    if args.phase in {'lock', 'all'}:
        if not (BASE / 'CALIBRATION_TUNE' / 'TUNE_DONE.json').exists():
            raise RuntimeError('LOCK cannot begin before TUNE_DONE')
        lock_doc = lock_phase(engine, profile, model_parts, split_inputs['CALIBRATION_LOCK'],
                              split_inputs['CALIBRATION_TUNE'], start,
                              [args.arm] if args.arm != 'all' else ['A', 'B', 'C', 'D'])
    else:
        lock_doc = read_json(BASE / 'CALIBRATION_LOCK' / 'V3_LOCK.json') if (BASE / 'CALIBRATION_LOCK' / 'V3_LOCK.json').exists() else {}
    if args.phase in {'formal', 'all'}:
        if not lock_doc:
            lock_doc = read_json(BASE / 'CALIBRATION_LOCK' / 'V3_LOCK.json')
        formal_phase(engine, profile, model_parts, start, lock_doc)
    print(json.dumps({'phase': args.phase, 'status': read_json(STATE_PATH).get('status'),
                      'runner_sha256': runner_sha, 'foreign_jobs_interrupted': 0}, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--phase', choices=['preflight', 'profile', 'tune', 'lock', 'formal', 'all'], required=True)
    p.add_argument('--arm', choices=['A', 'B', 'C', 'D', 'all'], default='all')
    args = p.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    try:
        arm_phase(args)
    except Exception as exc:
        state = read_json(STATE_PATH)
        set_state(status='V3_RUNNER_FAILED', failure_type='RUNTIME_OR_ENGINEERING_FAILURE',
                  error=repr(exc), traceback=traceback.format_exc(),
                  previous_status=state.get('status'), old_results_valid=True,
                  next_action='Inspect isolated V3 failure; do not alter frozen inputs or V1/V2 outputs.')
        raise


if __name__ == '__main__':
    main()
