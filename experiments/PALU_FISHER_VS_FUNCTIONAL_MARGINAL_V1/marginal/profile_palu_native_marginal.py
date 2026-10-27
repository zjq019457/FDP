import argparse
import copy
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import re
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2')
OFFICIAL = ROOT / 'official'
DATA = OFFICIAL / 'data'
MODEL = ROOT / 'models' / 'Llama-2-7b-hf'
BRANCH = Path('/diskE/jiaqizhang/PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1')
MANIFEST = BRANCH / 'marginal' / 'calibration_manifest.json'
SEQ = 2048
FP16_BYTES = 2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layer_of(name):
    return int(re.search(r'model\.layers\.(\d+)\.', name).group(1))


def c4_docs(path, n=1100):
    out = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            out.append(json.loads(line)['text'])
            if len(out) == n:
                break
    return out


def evaluate(model, tokenizer, manifest, wiki_ids, c4_ids):
    device = next(model.parameters()).device
    totals = {'wiki': [0.0, 0], 'c4': [0.0, 0]}
    with torch.no_grad():
        for ch in manifest['wikitext']['chunks']:
            ids = wiki_ids[ch['token_start']:ch['token_end']].unsqueeze(0).to(device)
            pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='sum')
            totals['wiki'][0] += float(loss.item()); totals['wiki'][1] += ids.shape[1] - 1
        for ch in manifest['c4']['chunks']:
            ids = c4_ids[ch['doc_index']].to(device)
            pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='sum')
            totals['c4'][0] += float(loss.item()); totals['c4'][1] += ids.shape[1] - 1
    wn = totals['wiki'][0] / totals['wiki'][1]
    cn = totals['c4'][0] / totals['c4'][1]
    return {'wiki_nll': wn, 'wiki_ppl': math.exp(wn), 'c4_nll': cn, 'c4_ppl': math.exp(cn), 'J': 0.5 * (wn + cn), 'tokens': totals['wiki'][1] + totals['c4'][1]}


def parent_and_child(model, full_name):
    parts = full_name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def build_from_factors(HeadwiseLowRankModule, raw, factors, ranks, device):
    new = HeadwiseLowRankModule(ranks, raw.in_features, raw.out_features, bias=raw.bias is not None).to(device=device, dtype=raw.weight.dtype)
    for i, r in enumerate(ranks):
        l, rr = factors[i]
        new.U[i].weight.data.copy_(l[:, :r].to(device))
        if raw.bias is not None:
            b = raw.bias.data.reshape(len(factors), -1)[i]
            new.U[i].bias.data.copy_(b)
    new.VT.weight.data.copy_(torch.cat([x[1][:r, :] for x, r in zip(factors, ranks)], dim=0).to(device))
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    ratio_tag = f'{args.ratio:.1f}'
    outdir = BRANCH / 'marginal' / f'ratio_{ratio_tag}'
    outdir.mkdir(parents=True, exist_ok=True)
    runtime = BRANCH / 'marginal' / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    os.chdir(runtime)
    manifest = json.loads(MANIFEST.read_text())
    schedule_doc = json.loads((BRANCH / 'official_schedule' / f'ratio_{ratio_tag}' / 'schedule.json').read_text())
    selection = schedule_doc['selection']

    sys.path.insert(0, str(OFFICIAL)); sys.path.insert(0, str(OFFICIAL / 'Palu'))
    had = types.ModuleType('fast_hadamard_transform'); had.hadamard_transform = lambda x, scale=1.0: x * scale; sys.modules['fast_hadamard_transform'] = had
    import run_official_palu_offline as wrapper
    wrapper.WIKI_TRAIN = DATA / 'wiki-train.hf_joined.raw'; wrapper.WIKI_TEST = DATA / 'wiki-test.hf_joined.raw'
    from utils import load_model_and_tokenizer
    import palu.decomposition as decomposition
    from palu.model.modules.svd_linear import HeadwiseLowRankModule, _per_head_whiten_decomposition_from_weight
    decomposition.get_calib_data = wrapper.offline_get_calib_data
    real_load = torch.load
    def trusted_load(*a, **kw): kw.setdefault('weights_only', False); return real_load(*a, **kw)
    torch.load = trusted_load
    model, tokenizer = load_model_and_tokenizer(str(MODEL))
    model.to(torch.device(args.device))
    device = next(model.parameters()).device
    args_ns = types.SimpleNamespace(calib_dataset='wikitext2', model_id=str(MODEL), use_cache=True, n_whiten_calib_samples=256)
    decomposition.get_whiten_scale_matrix(model, tokenizer, args_ns, args.device)

    raw_modules = {}; parents = {}
    for name, module in list(model.named_modules()):
        if 'k_proj' not in name and 'v_proj' not in name:
            continue
        raw_modules[name] = module
        parents[name] = parent_and_child(model, name)
    max_rank = max(max(v) for v in selection.values())
    factor_cache = runtime / 'llama2_official_whiten_factors_rank512.pt'
    if factor_cache.exists():
        factors = torch.load(factor_cache, map_location='cpu')
    else:
        factors = {}
        for name, module in raw_modules.items():
            weights = module.weight.data.reshape(2, -1, module.in_features)
            factors[name] = []
            for i in range(2):
                l, r = _per_head_whiten_decomposition_from_weight(weights[i], module.scaling_diag_matrix, max_rank)
                factors[name].append((l.detach().cpu().half(), r.detach().cpu().half()))
        torch.save(factors, factor_cache)

    baseline_modules = {}
    for name, raw in raw_modules.items():
        baseline_modules[name] = build_from_factors(HeadwiseLowRankModule, raw, factors[name], selection[name], device)
        parent, child = parents[name]; setattr(parent, child, baseline_modules[name])

    wiki_ids = tokenizer((DATA / 'wiki-test.hf_joined.raw').read_text(encoding='utf-8'), return_tensors='pt').input_ids[0]
    docs = c4_docs(DATA / 'c4-validation.00000-of-00008.json.gz', manifest['c4']['pool_docs'])
    c4_ids = {i: tokenizer(docs[i], return_tensors='pt').input_ids[:, :SEQ] for i in {ch['doc_index'] for ch in manifest['c4']['chunks']}}
    baseline = evaluate(model, tokenizer, manifest, wiki_ids, c4_ids)
    total_rank = sum(sum(v) for v in selection.values())
    baseline.update({'ratio': args.ratio, 'total_rank': total_rank, 'latent_cache_bytes_B1_T2048_fp16': total_rank * SEQ * FP16_BYTES, 'status': 'MARGINAL_BASELINE_COMPLETE', 'native_conversion': 'official HeadwiseLowRankModule primitives; official whiten scaling matrix', 'schedule_sha': schedule_doc['schedule_sha'], 'manifest_sha256': sha(MANIFEST)})
    (outdir / 'baseline.json').write_text(json.dumps(baseline, indent=2, sort_keys=True) + '\n')

    fieldnames = ['model', 'ratio', 'unit', 'module', 'layer', 'slot', 'status', 'fisher_importance', 'q', 'native_rank_before', 'native_rank_after', 'native_rank_delta', 'actual_latent_cache_bytes_before', 'actual_latent_cache_bytes_after', 'relative_native_resource_reduction', 'delta_J', 'phi_palu', 'candidate_Wiki_PPL', 'candidate_C4_PPL', 'schedule_sha', 'manifest_sha256']
    csv_path = outdir / 'units.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for name, module_ranks in selection.items():
            for slot, before in enumerate(module_ranks):
                unit = f'{name}::slot{slot}'; lay = layer_of(name); q = float(json.loads((BRANCH / 'official_schedule' / f'ratio_{ratio_tag}' / 'schedule.json').read_text())['units'][2 * list(selection).index(name) + slot]['q'])
                row = {'model': 'llama2_7b_hf', 'ratio': args.ratio, 'unit': unit, 'module': name, 'layer': lay, 'slot': slot, 'fisher_importance': float(schedule_doc['fisher_module_means'][name]), 'q': q, 'native_rank_before': before, 'schedule_sha': schedule_doc['schedule_sha'], 'manifest_sha256': sha(MANIFEST)}
                if before <= 32:
                    row.update({'status': 'SKIPPED_LEGAL_MIN_RANK', 'native_rank_after': before, 'native_rank_delta': 0, 'actual_latent_cache_bytes_before': total_rank * SEQ * FP16_BYTES, 'actual_latent_cache_bytes_after': total_rank * SEQ * FP16_BYTES, 'relative_native_resource_reduction': 0.0, 'delta_J': None, 'phi_palu': None})
                    w.writerow(row); f.flush(); continue
                alt_ranks = list(module_ranks); alt_ranks[slot] = before - 32
                parent, child = parents[name]
                candidate = build_from_factors(HeadwiseLowRankModule, raw_modules[name], factors[name], alt_ranks, device)
                setattr(parent, child, candidate)
                cand = evaluate(model, tokenizer, manifest, wiki_ids, c4_ids)
                after_total = total_rank - 32
                delta = cand['J'] - baseline['J']; rel = 32.0 / total_rank
                row.update({'status': 'COMPLETE', 'native_rank_after': before - 32, 'native_rank_delta': -32, 'actual_latent_cache_bytes_before': total_rank * SEQ * FP16_BYTES, 'actual_latent_cache_bytes_after': after_total * SEQ * FP16_BYTES, 'relative_native_resource_reduction': rel, 'delta_J': delta, 'phi_palu': delta / rel, 'candidate_Wiki_PPL': cand['wiki_ppl'], 'candidate_C4_PPL': cand['c4_ppl']})
                w.writerow(row); f.flush()
                setattr(parent, child, baseline_modules[name]); del candidate; gc.collect(); torch.cuda.empty_cache()
    (outdir / 'status.json').write_text(json.dumps({'status': 'MARGINAL_PROFILE_COMPLETE', 'ratio': args.ratio, 'baseline': baseline, 'schedule_sha': schedule_doc['schedule_sha'], 'manifest_sha256': sha(MANIFEST), 'native_conversion': 'official HeadwiseLowRankModule primitives; no native implementation modification', 'quality_used_to_design_selection': False}, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'MARGINAL_PROFILE_COMPLETE', 'ratio': args.ratio, 'baseline_J': baseline['J'], 'output': str(outdir)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
