import argparse
import copy
import gc
import gzip
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import torch
import torch.nn.functional as F


ROOT = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2')
OFFICIAL = ROOT / 'official'
DATA = OFFICIAL / 'data'
MODEL = ROOT / 'models' / 'Llama-2-7b-hf'
BRANCH = Path('/diskE/jiaqizhang/PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1')
MANIFEST = BRANCH / 'marginal' / 'calibration_manifest.json'


def load_manifest():
    return json.loads(MANIFEST.read_text())


def c4_docs(path, n=1100):
    out = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            out.append(json.loads(line)['text'])
            if len(out) == n:
                break
    return out


def evaluate(model, tokenizer, manifest):
    device = next(model.parameters()).device
    wiki_ids = tokenizer((DATA / 'wiki-test.hf_joined.raw').read_text(encoding='utf-8'), return_tensors='pt').input_ids[0]
    docs = c4_docs(DATA / 'c4-validation.00000-of-00008.json.gz', manifest['c4']['pool_docs'])
    totals = {'wiki': [0.0, 0], 'c4': [0.0, 0]}
    with torch.no_grad():
        for ch in manifest['wikitext']['chunks']:
            ids = wiki_ids[ch['token_start']:ch['token_end']].unsqueeze(0).to(device)
            pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='sum')
            totals['wiki'][0] += float(loss.item()); totals['wiki'][1] += ids.shape[1] - 1
        for ch in manifest['c4']['chunks']:
            ids = tokenizer(docs[ch['doc_index']], return_tensors='pt').input_ids[:, :manifest['sequence_length']].to(device)
            if ids.shape[1] != manifest['sequence_length']:
                raise RuntimeError(f"short C4 manifest chunk: {ch}")
            pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='sum')
            totals['c4'][0] += float(loss.item()); totals['c4'][1] += ids.shape[1] - 1
    wiki_nll = totals['wiki'][0] / totals['wiki'][1]
    c4_nll = totals['c4'][0] / totals['c4'][1]
    return {'wiki_nll': wiki_nll, 'wiki_ppl': math.exp(wiki_nll), 'c4_nll': c4_nll, 'c4_ppl': math.exp(c4_nll), 'J': 0.5 * (wiki_nll + c4_nll), 'token_counts': {'wiki': totals['wiki'][1], 'c4': totals['c4'][1]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, required=True)
    ap.add_argument('--arm', choices=['control', 'candidate'], required=True)
    ap.add_argument('--module', default=None)
    ap.add_argument('--slot', type=int, default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.arm == 'candidate' and (args.module is None or args.slot is None):
        raise ValueError('candidate requires module and slot')
    manifest = load_manifest()
    schedule_doc = json.loads((BRANCH / 'official_schedule' / f'ratio_{args.ratio:.1f}' / 'schedule.json').read_text())
    selection = copy.deepcopy(schedule_doc['selection'])
    change = None
    if args.arm == 'candidate':
        before = selection[args.module][args.slot]
        after = before - 32
        if after < 32:
            raise ValueError('candidate reduction would violate min native rank 32')
        selection[args.module][args.slot] = after
        change = {'module': args.module, 'slot': args.slot, 'before': before, 'after': after, 'delta': -32}
    sys.path.insert(0, str(OFFICIAL))
    sys.path.insert(0, str(OFFICIAL / 'Palu'))
    had = types.ModuleType('fast_hadamard_transform')
    had.hadamard_transform = lambda x, scale=1.0: x * scale
    sys.modules['fast_hadamard_transform'] = had
    import run_official_palu_offline as wrapper
    wrapper.WIKI_TRAIN = DATA / 'wiki-train.hf_joined.raw'
    wrapper.WIKI_TEST = DATA / 'wiki-test.hf_joined.raw'
    from utils import load_model_and_tokenizer
    import palu.decomposition as decomposition
    decomposition.get_calib_data = wrapper.offline_get_calib_data
    real_load = torch.load
    def trusted_load(*a, **kw):
        kw.setdefault('weights_only', False)
        return real_load(*a, **kw)
    torch.load = trusted_load
    model, tokenizer = load_model_and_tokenizer(str(MODEL))
    model.to(torch.device(args.device))
    search_args = SimpleNamespace(
        calib_dataset='wikitext2', model_id=str(MODEL), calib_seqlen=1024,
        n_whiten_calib_samples=256, use_cache=True, decompose_method='whiten',
    )
    decomposition.compress_model(model, tokenizer, search_args, args.device, selection)
    result = evaluate(model, tokenizer, manifest)
    result.update({'status': 'MARGINAL_CONVERSION_CANARY_COMPLETE', 'ratio': args.ratio, 'arm': args.arm, 'schedule_sha': schedule_doc['schedule_sha'], 'selection_rank': sum(sum(x) for x in selection.values()), 'change': change, 'manifest_path': str(MANIFEST), 'manifest_sha256': __import__('hashlib').sha256(MANIFEST.read_bytes()).hexdigest(), 'native_conversion': 'official HeadwiseLowRankModule.from_linear_whiten via official decomposition.compress_model', 'quality_used_to_design_selection': False})
    out = BRANCH / 'marginal' / f'canary_ratio_{args.ratio:.1f}_{args.arm}.json'
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
