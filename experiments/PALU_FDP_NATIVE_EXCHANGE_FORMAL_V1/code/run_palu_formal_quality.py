import argparse
import gzip
import hashlib
import importlib.machinery
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


BRANCH = Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1')
ROOT = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2')
OFFICIAL = ROOT / 'official'
DATA = OFFICIAL / 'data'
MODEL = ROOT / 'models' / 'Llama-2-7b-hf'
PROFILE = Path('/diskE/jiaqizhang/PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1')
FACTOR_CACHE = PROFILE / 'marginal' / 'runtime' / 'llama2_official_whiten_factors_rank512.pt'
SEQ = 2048
DTYPE_BYTES = 2


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def parent_and_child(model, full_name):
    parts = full_name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def build_from_factors(HeadwiseLowRankModule, raw, factors, ranks, device):
    new = HeadwiseLowRankModule(
        ranks,
        raw.in_features,
        raw.out_features,
        bias=raw.bias is not None,
    ).to(device=device, dtype=raw.weight.dtype)
    for i, rank in enumerate(ranks):
        left, _right = factors[i]
        new.U[i].weight.data.copy_(left[:, :rank].to(device))
        if raw.bias is not None:
            bias = raw.bias.data.reshape(len(factors), -1)[i]
            new.U[i].bias.data.copy_(bias)
    new.VT.weight.data.copy_(
        torch.cat([x[1][:rank, :] for x, rank in zip(factors, ranks)], dim=0).to(device)
    )
    return new


def load_schedule(ratio: float, method: str):
    tag = f'{ratio:.1f}'
    if method == 'uniform':
        path = BRANCH / 'schedules' / f'llama2_ratio_{tag}_uniform_schedule.json'
    elif method == 'fisher':
        path = BRANCH / 'schedules' / f'llama2_ratio_{tag}_anchor_schedule.json'
    elif method in {'fdp_prior', 'fdp_native', 'random_0', 'random_1', 'random_2'}:
        path = BRANCH / 'exchange' / 'calibration' / f'ratio_{tag}' / method / 'final_schedule.json'
    else:
        raise ValueError(method)
    doc = json.loads(path.read_text())
    selection = doc['selection']
    rank = sum(sum(v) for v in selection.values())
    if rank != int(doc.get('realized_rank_sum', doc.get('resource', {}).get('total_native_rank', rank))):
        raise RuntimeError(f'schedule self-check failed: {path}')
    return path, doc, selection


def load_c4_docs(path: Path, count: int):
    docs = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            docs.append(json.loads(line)['text'])
            if len(docs) == count:
                break
    if len(docs) != count:
        raise RuntimeError(f'C4 docs: got {len(docs)}, expected {count}')
    return docs


def load_runtime(ratio: float, method: str):
    schedule_path, schedule_doc, selection = load_schedule(ratio, method)

    sys.path.insert(0, str(OFFICIAL))
    sys.path.insert(0, str(OFFICIAL / 'Palu'))
    had = types.ModuleType('fast_hadamard_transform')
    had.hadamard_transform = lambda x, scale=1.0: x * scale
    # transformers probes optional dependencies with importlib.util.find_spec;
    # a runtime stub must therefore carry a valid module spec.
    had.__spec__ = importlib.machinery.ModuleSpec('fast_hadamard_transform', loader=None)
    sys.modules['fast_hadamard_transform'] = had

    from utils import load_model_and_tokenizer
    from palu.model.modules.svd_linear import HeadwiseLowRankModule

    model, tokenizer = load_model_and_tokenizer(str(MODEL))
    raw_modules = {}
    parents = {}
    for name, module in list(model.named_modules()):
        if 'k_proj' not in name and 'v_proj' not in name:
            continue
        raw_modules[name] = module
        parents[name] = parent_and_child(model, name)

    try:
        factors = torch.load(FACTOR_CACHE, map_location='cpu', weights_only=False)
    except TypeError:
        factors = torch.load(FACTOR_CACHE, map_location='cpu')

    for name, ranks in selection.items():
        if name not in raw_modules:
            raise KeyError(f'schedule module missing from model: {name}')
        replacement = build_from_factors(
            HeadwiseLowRankModule,
            raw_modules[name],
            factors[name],
            ranks,
            torch.device('cpu'),
        )
        parent, child = parents[name]
        setattr(parent, child, replacement)

    model.to(torch.device('cuda'))
    model.eval()
    model.config.use_cache = False

    wiki_text = (DATA / 'wiki-test.hf_joined.raw').read_text(encoding='utf-8')
    wiki_ids = tokenizer(wiki_text, return_tensors='pt').input_ids
    c4_docs = load_c4_docs(DATA / 'c4-validation.00000-of-00008.json.gz', 1100)
    c4_ids = tokenizer(' '.join(c4_docs), return_tensors='pt').input_ids
    c4_ids = c4_ids[:, :256 * SEQ]

    return model, tokenizer, schedule_path, schedule_doc, selection, wiki_ids, c4_ids


@torch.no_grad()
def eval_loader(model, ids):
    device = next(model.parameters()).device
    nsamples = ids.numel() // SEQ
    nlls = []
    for index in range(nsamples):
        batch = ids[:, index * SEQ:(index + 1) * SEQ].to(device)
        outputs = model.model(batch)
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        value = float((loss.float() * SEQ).item())
        if not math.isfinite(value):
            raise FloatingPointError(f'non-finite loss at chunk {index}')
        nlls.append(value)
    if not nlls:
        raise RuntimeError('empty evaluation loader')
    total_tokens = len(nlls) * SEQ
    mean_token_nll = sum(nlls) / total_tokens
    return {
        'chunks': len(nlls),
        'tokens': total_tokens,
        'summed_nll': sum(nlls),
        'mean_token_nll': mean_token_nll,
        'ppl': math.exp(mean_token_nll),
        'chunk_nll': nlls,
    }


def resource(selection):
    rank = sum(sum(v) for v in selection.values())
    cache = rank * SEQ * DTYPE_BYTES
    return {
        'total_native_rank': rank,
        'prefill_cache_bytes': cache,
        'decode_cache_bytes': cache,
    }


def module_census(model, selection):
    result = {}
    for name, ranks in selection.items():
        module = dict(model.named_modules())[name]
        result[name] = {
            'requested_ranks': ranks,
            'effective_rank_sum': int(sum(ranks)),
            'u_shapes': [list(factor.weight.shape) for factor in module.U],
            'vt_shape': list(module.VT.weight.shape),
        }
    return result


def update_status(cell, status, error=None):
    path = BRANCH / 'audit' / 'CAMPAIGN_STATUS.json'
    doc = json.loads(path.read_text())
    doc['quality_started'] = True
    doc['status'] = status
    doc['active_quality_cell'] = cell if status.endswith('RUNNING') else None
    if error:
        doc['last_quality_error'] = error
    write_json(path, doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, required=True, choices=[0.3, 0.5, 0.7])
    parser.add_argument('--method', required=True, choices=['uniform', 'fisher', 'fdp_prior', 'fdp_native', 'random_0', 'random_1', 'random_2'])
    args = parser.parse_args()
    cell = f'ratio_{args.ratio:.1f}/{args.method}'
    outdir = BRANCH / 'quality' / 'llama2' / f'ratio_{args.ratio:.1f}' / f'PALU_{args.method.upper()}'
    outdir.mkdir(parents=True, exist_ok=True)
    update_status(cell, 'FORMAL_QUALITY_CELL_RUNNING')
    try:
        model, tokenizer, schedule_path, schedule_doc, selection, wiki_ids, c4_ids = load_runtime(args.ratio, args.method)
        res = resource(selection)
        census = module_census(model, selection)
        expected = int(schedule_doc.get('realized_rank_sum', schedule_doc.get('resource', {}).get('total_native_rank', res['total_native_rank'])))
        if res['total_native_rank'] != expected:
            raise RuntimeError('runtime native rank does not match schedule')
        wiki = eval_loader(model, wiki_ids)
        c4 = eval_loader(model, c4_ids)
        quality = {
            'status': 'FORMAL_QUALITY_COMPLETE',
            'campaign': 'PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1',
            'model': str(MODEL),
            'ratio': args.ratio,
            'method': f'PALU_{args.method.upper()}',
            'protocol_manifest': str(BRANCH / 'quality' / 'formal_protocol' / 'FORMAL_QUALITY_PROTOCOL_MANIFEST.json'),
            'schedule_path': str(schedule_path),
            'schedule_sha': schedule_doc['schedule_sha'],
            'resource': res,
            'resource_match_gate': 'PASS',
            'native_conversion': 'official Palu HeadwiseLowRankModule with frozen anchor-schema factors; no schedule search during quality',
            'wiki': {k: v for k, v in wiki.items() if k != 'chunk_nll'},
            'c4': {k: v for k, v in c4.items() if k != 'chunk_nll'},
            'scorer': 'historical official run_ppl_eval semantics; common implementation for all methods',
            'module_census': census,
        }
        write_json(outdir / 'quality.json', quality)
        write_json(outdir / 'window_results.json', {'wiki': wiki, 'c4': c4})
        update_status(cell, 'FORMAL_QUALITY_CELL_COMPLETE')
        print(json.dumps({
            'status': quality['status'],
            'cell': cell,
            'wiki_ppl': wiki['ppl'],
            'c4_ppl': c4['ppl'],
            'resource': res,
            'outdir': str(outdir),
        }, indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        update_status(cell, 'FORMAL_QUALITY_CELL_FAILED', repr(exc))
        raise


if __name__ == '__main__':
    main()
