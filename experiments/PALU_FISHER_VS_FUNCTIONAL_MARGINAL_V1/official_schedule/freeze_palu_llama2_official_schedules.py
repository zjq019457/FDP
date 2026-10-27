import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2')
REPO = ROOT / 'official' / 'Palu'
OUT = Path('/diskE/jiaqizhang/PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1/official_schedule')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def schedule_sha(selection):
    payload = json.dumps(selection, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'runtime').mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / 'official'))
    sys.path.insert(0, str(REPO))
    os.chdir(OUT / 'runtime')
    (OUT / 'runtime' / 'cache').mkdir(parents=True, exist_ok=True)

    # The official utility imports the optional Hadamard CUDA extension even
    # though Fisher schedule extraction does not execute any Hadamard path.
    # Keep the official code untouched and provide an import-only stub here.
    if 'fast_hadamard_transform' not in sys.modules:
        had = types.ModuleType('fast_hadamard_transform')
        had.hadamard_transform = lambda x, scale=1.0: x * scale
        sys.modules['fast_hadamard_transform'] = had

    from utils import load_model_and_tokenizer
    from palu import rank_search
    from palu.model import AVAILABLE_MODELS

    model_path = str(ROOT / 'models' / 'Llama-2-7b-hf')
    fisher_anchor = Path('/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/v2_runs/palu_fisher/PALU_V2_OFFICIAL_0P7_FISHER_ANCHOR.json')
    anchor_doc = json.loads(fisher_anchor.read_text())

    # Use only the frozen official Fisher means from the anchor. The prior raw
    # 7.6 GB torch cache is incomplete because diskF filled during its write;
    # no attempt is made to repair or overwrite that historical file.
    rank_search.get_calib_data = lambda *a, **k: []
    anchor_means = anchor_doc['module_fisher_means']
    def attach_frozen_anchor(model, calib_loader, device, use_cache):
        for name, module in model.named_modules():
            if ('k_proj' in name or 'v_proj' in name) and name in anchor_means:
                module.fisher_info = torch.full(module.weight.shape, float(anchor_means[name]), device=module.weight.device, dtype=torch.float32)
    rank_search.calib_fisher_info = attach_frozen_anchor
    real_load = torch.load
    def trusted_load(*a, **k):
        k.setdefault('weights_only', False)
        return real_load(*a, **k)
    torch.load = trusted_load

    model, tokenizer = load_model_and_tokenizer(model_path)
    target_model_class = AVAILABLE_MODELS[model.config.model_type]['ModelForCausalLM']
    info = target_model_class.get_kv_info(model, model.config.num_key_value_heads)
    ratios = [0.7, 0.5, 0.3]
    manifest = {
        'task': 'PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1',
        'model': model_path,
        'model_config_sha256': sha256(Path(model_path) / 'config.json'),
        'official_repo': str(REPO),
        'native_rank_search_source': str(REPO / 'palu' / 'rank_search.py'),
        'native_rank_search_source_sha256': sha256(REPO / 'palu' / 'rank_search.py'),
        'official_fisher_anchor': str(fisher_anchor),
        'official_fisher_anchor_sha256': sha256(fisher_anchor),
        'fisher_method': 'official calib_fisher_info elementwise squared-gradient accumulation, sqrt(mean); frozen per-module means reused read-only from official anchor',
        'search_method': 'fisher_uniform',
        'head_group_size': 4,
        'rank_quantum': 32,
        'num_key_value_heads': int(model.config.num_key_value_heads),
        'native_info': {'num_lr_groups': int(info.num_lr_groups), 'lr_group_dims': int(info.lr_group_dims)},
        'ratios': ratios,
        'official_baseline_freeze': True,
        'quality_used_to_design_schedule': False,
    }
    (OUT / 'OFFICIAL_SCHEDULE_FREEZE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')

    for ratio in ratios:
        search_args = SimpleNamespace(
            search_method='fisher_uniform',
            calib_dataset='wikitext2',
            model_id=model_path,
            calib_seqlen=1024,
            use_cache=True,
            device=args.device,
            head_group_size=4,
            param_ratio_target=ratio,
        )
        selection, rank_sum, total_rank = rank_search.rank_search(model, tokenizer, search_args)
        # Official fisher_uniform computes one Fisher scalar for each module
        # before split_values replicates it across native group slots.
        fisher_modules = {}
        fisher_units = {}
        for name, module in model.named_modules():
            if 'k_proj' not in name and 'v_proj' not in name:
                continue
            fisher = module.fisher_info.reshape(info.num_lr_groups, -1, module.in_features)
            vals = [float(torch.mean(fisher[i]).item()) for i in range(info.num_lr_groups)]
            fisher_modules[name] = vals
            fisher_units[name] = [vals[0]] * len(selection[name])
        out = OUT / f'ratio_{ratio:.1f}'
        out.mkdir(parents=True, exist_ok=True)
        doc = {
            'status': 'OFFICIAL_PALU_FISHER_NATIVE_CONTROL_SCHEDULE_FROZEN',
            'model': model_path,
            'ratio': ratio,
            'rank_sum': int(rank_sum),
            'total_rank': int(total_rank),
            'realized_ratio': rank_sum / total_rank,
            'rank_quantum': 32,
            'head_group_size': 4,
            'selection': selection,
            'fisher_module_means': fisher_modules,
            'fisher_native_unit_means': fisher_units,
            'selection_sha256': schedule_sha(selection),
            'source_fisher_anchor_sha256': sha256(fisher_anchor),
            'official_native_source_sha256': manifest['native_rank_search_source_sha256'],
            'quality_used_to_design_schedule': False,
            'immutable_status': 'NEW_BRANCH_ONLY',
        }
        (out / 'schedule.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'ratio': ratio, 'rank_sum': rank_sum, 'total_rank': total_rank, 'sha': doc['selection_sha256']}), flush=True)


if __name__ == '__main__':
    main()
