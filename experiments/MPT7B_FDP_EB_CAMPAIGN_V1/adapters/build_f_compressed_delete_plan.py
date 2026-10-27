#!/usr/bin/env python3
"""Build a guarded, metadata-only deletion plan for generated model weights."""
import hashlib
import json
from pathlib import Path

INV = Path('/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/audits/F_COMPRESSED_MODEL_INVENTORY_20261005.json')
OUT = Path('/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/audits/F_COMPRESSED_MODEL_DELETE_PLAN_20261005.json')
ROOT = Path('/diskF/jiaqizhang')
PROTECTED = [
    Path('/diskF/jiaqizhang/Llama-3.1'),
    Path('/diskF/jiaqizhang/models/新建文件夹'),
    Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf'),
    Path('/diskF/jiaqizhang/FDP125_DIRECTION_CORRECTED_V2/models/mistral_7b_v0_2_official/mistral-7B-v0.2'),
]
PALU_ROOT = Path('/diskF/jiaqizhang/fdp_recalkv_transfer_confirmation_v1_results/models')

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def under(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

inv = json.loads(INV.read_text())
targets = []
checkpoint_rows = []
for r in inv['records']:
    is_eigen = r['classification'] == 'COMPRESSED_EIGEN_MODEL'
    is_palu_derived = r['model_type'] == 'palullama' and under(Path(r['directory']), PALU_ROOT)
    if not (is_eigen or is_palu_derived) or not r['weight_bytes']:
        continue
    directory = Path(r['directory']).resolve(strict=True)
    if not under(directory, ROOT) or any(under(directory, p) for p in PROTECTED):
        raise RuntimeError(f'protected/out-of-scope candidate: {directory}')
    config = directory / 'config.json'
    index_paths = [directory / name for name in r['indices']]
    row_targets = []
    for shard in r['shards']:
        p = Path(shard['path']).resolve(strict=True)
        if not under(p, ROOT) or any(under(p, x) for x in PROTECTED):
            raise RuntimeError(f'protected/out-of-scope weight: {p}')
        if p.parent != directory or p.suffix not in ('.safetensors', '.bin'):
            raise RuntimeError(f'unexpected model weight path: {p}')
        st = p.stat()
        if st.st_size != shard['bytes']:
            raise RuntimeError(f'size changed since inventory: {p}')
        item = {
            'path': str(p), 'bytes': st.st_size, 'device': st.st_dev,
            'inode': st.st_ino, 'link_count': st.st_nlink,
            'mtime_ns': st.st_mtime_ns,
        }
        targets.append(item)
        row_targets.append(item)
    checkpoint_rows.append({
        'directory': str(directory), 'classification': 'GENERATED_COMPRESSED_MODEL',
        'model_type': r['model_type'], 'architectures': r['architectures'],
        'source_reference': r['name_or_path'], 'checkpoint_bytes': sum(x['bytes'] for x in row_targets),
        'config_sha256': sha256(config) if config.is_file() else None,
        'index_sha256': {p.name: sha256(p) for p in index_paths if p.is_file()},
        'weight_files': row_targets,
    })

plan = {
    'schema': 'F_COMPRESSED_MODEL_DELETE_PLAN_V1',
    'inventory_path': str(INV), 'inventory_sha256': sha256(INV),
    'scope': 'Only indexed generated EigenAttention model checkpoints and indexed derived PALU model checkpoints under /diskF/jiaqizhang; preserve raw source checkpoints and all non-weight evidence.',
    'protected_original_roots': [str(p) for p in PROTECTED],
    'checkpoint_count': len(checkpoint_rows), 'weight_file_count': len(targets),
    'planned_weight_bytes': sum(x['bytes'] for x in targets),
    'deletion_semantics': 'unlink exact verified weight shard files only; retain config, index, manifests, logs, result tables, basis/profile artifacts, and directories.',
    'checkpoints': checkpoint_rows,
}
OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: plan[k] for k in ('schema','checkpoint_count','weight_file_count','planned_weight_bytes','inventory_sha256','scope')}, indent=2))
print('PLAN_PATH', OUT)
