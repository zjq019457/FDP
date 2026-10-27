#!/usr/bin/env python3
"""Guarded removal of exact generated checkpoint weight shards from a reviewed plan."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PLAN = Path('/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/audits/F_COMPRESSED_MODEL_DELETE_PLAN_20261005.json')
RECEIPT = Path('/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/audits/F_COMPRESSED_MODEL_DELETE_RECEIPT_20261005.json')
ROOT = Path('/diskF/jiaqizhang')
PROTECTED = [
    Path('/diskF/jiaqizhang/Llama-3.1'),
    Path('/diskF/jiaqizhang/models/新建文件夹'),
    Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf'),
    Path('/diskF/jiaqizhang/FDP125_DIRECTION_CORRECTED_V2/models/mistral_7b_v0_2_official/mistral-7B-v0.2'),
]

def under(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

def save_receipt(receipt):
    receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')

plan = json.loads(PLAN.read_text())
if plan.get('schema') != 'F_COMPRESSED_MODEL_DELETE_PLAN_V1':
    raise SystemExit('abort: unexpected plan schema')
paths = [Path(x['path']) for cp in plan['checkpoints'] for x in cp['weight_files']]
if len(paths) != plan['weight_file_count'] or len(set(paths)) != len(paths):
    raise SystemExit('abort: duplicate or count-mismatched target list')

# Fail closed if any target is referenced by an active command line.
needles = [str(p).encode() for p in paths]
active_refs = []
for entry in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        cmdline = entry.read_bytes()
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        continue
    if any(n in cmdline for n in needles):
        active_refs.append(str(entry))
if active_refs:
    raise SystemExit('abort: active process command line references target checkpoint weights: ' + ','.join(active_refs))

# Fail closed if any target shard is open in any process.
lsof = subprocess.run(['lsof', '-nP', *map(str, paths)], capture_output=True, text=True)
if lsof.stderr.strip():
    raise SystemExit('abort: lsof could not verify open-file state: ' + lsof.stderr.strip()[:1000])
if lsof.stdout.strip():
    raise SystemExit('abort: target checkpoint weight files are open; no files removed')

# Verify every exact path against the audited plan before unlinking anything.
for cp in plan['checkpoints']:
    for item in cp['weight_files']:
        p = Path(item['path'])
        resolved = p.resolve(strict=True)
        if resolved != p or not under(p, ROOT) or any(under(p, x) for x in PROTECTED):
            raise SystemExit('abort: path escaped the user-owned F scope or entered a protected source: ' + str(p))
        if p.suffix not in ('.safetensors', '.bin') or not p.is_file() or p.is_symlink():
            raise SystemExit('abort: target is not a regular indexed model weight file: ' + str(p))
        st = p.stat()
        expected = (item['device'], item['inode'], item['bytes'], item['link_count'], item['mtime_ns'])
        actual = (st.st_dev, st.st_ino, st.st_size, st.st_nlink, st.st_mtime_ns)
        if actual != expected:
            raise SystemExit('abort: target changed since plan: ' + str(p))
        if st.st_nlink != 1:
            raise SystemExit('abort: hard-linked model weight requires review; no files removed')

receipt = {
    'schema': 'F_COMPRESSED_MODEL_DELETE_RECEIPT_V1',
    'plan_path': str(PLAN),
    'plan_sha256': __import__('hashlib').sha256(PLAN.read_bytes()).hexdigest(),
    'status': 'DELETING_VERIFIED_WEIGHT_SHARDS',
    'target_checkpoint_count': plan['checkpoint_count'],
    'target_file_count': plan['weight_file_count'],
    'planned_bytes': plan['planned_weight_bytes'],
    'deleted_files': [], 'deleted_bytes': 0, 'failures': [],
    'preserved': [str(p) for p in PROTECTED],
    'preserved_evidence': 'configs, shard indexes, manifests, profiles/bases, logs, metrics, tables, and all non-weight files',
    'foreign_jobs_interrupted': 0,
}
for cp in plan['checkpoints']:
    for item in cp['weight_files']:
        p = Path(item['path'])
        try:
            p.unlink()
            receipt['deleted_files'].append({'path': str(p), 'bytes': item['bytes']})
            receipt['deleted_bytes'] += item['bytes']
        except OSError as e:
            receipt['failures'].append({'path': str(p), 'error': repr(e)})
            receipt['status'] = 'PARTIAL_CLEANUP_REVIEW_REQUIRED'
            save_receipt(receipt)
            raise
save_receipt(receipt)
print(json.dumps({k: receipt[k] for k in ('status','target_checkpoint_count','target_file_count','planned_bytes','deleted_bytes','failures','foreign_jobs_interrupted','finished_utc')}, indent=2))
print('RECEIPT_PATH', RECEIPT)
