#!/usr/bin/env python3
"""Read-only inventory of indexed model checkpoints on FDP-owned F storage."""
import json
import os
from pathlib import Path

ROOT = Path('/diskF/jiaqizhang')
PRESERVE = [
    Path('/diskF/jiaqizhang/Llama-3.1'),
    Path('/diskF/jiaqizhang/models/新建文件夹'),
    Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf'),
    Path('/diskF/jiaqizhang/FDP125_DIRECTION_CORRECTED_V2/models/mistral_7b_v0_2_official/mistral-7B-v0.2'),
]

def beneath(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

records = []
for base, dirs, files in os.walk(ROOT, followlinks=False):
    dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
    indices = [name for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json') if name in files]
    if not indices:
        continue
    d = Path(base)
    config_path = d / 'config.json'
    try:
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    except Exception as e:
        config = {'_config_read_error': repr(e)}
    shards = set()
    for index_name in indices:
        try:
            idx = json.loads((d / index_name).read_text())
            shards.update(idx.get('weight_map', {}).values())
        except Exception:
            pass
    shard_rows = []
    for shard in sorted(shards):
        f = d / shard
        if f.is_file() and not f.is_symlink():
            shard_rows.append({'path': str(f), 'bytes': f.stat().st_size})
    arch = config.get('architectures', [])
    lowrank = any('eigenattn' in str(x).lower() or 'lowrank' in str(x).lower() for x in arch)
    raw = any(beneath(d, p) for p in PRESERVE)
    records.append({
        'directory': str(d),
        'indices': indices,
        'model_type': config.get('model_type'),
        'architectures': arch,
        'name_or_path': config.get('_name_or_path'),
        'layers': config.get('num_hidden_layers', config.get('n_layers')),
        'max_positions': config.get('max_position_embeddings', config.get('max_seq_len')),
        'shard_count': len(shard_rows),
        'weight_bytes': sum(x['bytes'] for x in shard_rows),
        'shards': shard_rows,
        'classification': 'PRESERVE_ORIGINAL' if raw else ('COMPRESSED_EIGEN_MODEL' if lowrank else 'REVIEW_ARCH_UNCLEAR'),
    })

print(json.dumps({'root': str(ROOT), 'indexed_checkpoint_count': len(records), 'records': records}, indent=2, ensure_ascii=False))
