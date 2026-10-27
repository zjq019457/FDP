import hashlib
import json
import math
from pathlib import Path


BRANCH = Path('/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1')
ROOT = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2')
DATA = ROOT / 'official' / 'data'
SCHEDULES = BRANCH / 'schedules'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def schedule_sha(selection: dict) -> str:
    raw = json.dumps(selection, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def uniform_selection(anchor: dict, realized_rank: int) -> dict:
    """Distribute the anchor's realized native blocks evenly over all slots.

    The formal comparison must have exact native resource equality with the
    frozen Fisher schedule.  The integer remainder is spread deterministically
    across slots, while every slot remains a legal multiple of 32 and within
    the native [32, 512] range.
    """
    modules = sorted(anchor['selection'])
    units = [(name, slot) for name in modules for slot in range(len(anchor['selection'][name]))]
    blocks = realized_rank // 32
    base, remainder = divmod(blocks, len(units))
    extra = set()
    if remainder:
        for j in range(remainder):
            extra.add((j * len(units)) // remainder)
    values = [32 * (base + (1 if i in extra else 0)) for i in range(len(units))]
    if min(values) < 32 or max(values) > 512:
        raise RuntimeError(f'uniform schedule violates native bounds: {min(values)}..{max(values)}')
    out = {name: [0] * len(anchor['selection'][name]) for name in modules}
    for (name, slot), rank in zip(units, values):
        out[name][slot] = rank
    if sum(sum(v) for v in out.values()) != realized_rank:
        raise RuntimeError('uniform schedule rank mismatch')
    return out


def main() -> None:
    SCHEDULES.mkdir(parents=True, exist_ok=True)
    formal_dir = BRANCH / 'quality' / 'formal_protocol'
    formal_dir.mkdir(parents=True, exist_ok=True)

    wiki = DATA / 'wiki-test.hf_joined.raw'
    c4 = DATA / 'c4-validation.00000-of-00008.json.gz'
    if not wiki.exists() or not c4.exists():
        raise FileNotFoundError('historical formal WikiText-2/C4 source is missing')

    ratios = [0.7, 0.5, 0.3]
    uniform_records = {}
    for ratio in ratios:
        tag = f'{ratio:.1f}'
        anchor_path = SCHEDULES / f'llama2_ratio_{tag}_anchor_schedule.json'
        anchor = json.loads(anchor_path.read_text())
        realized = int(anchor['realized_rank_sum'])
        selection = uniform_selection(anchor, realized)
        record = {
            'status': 'FORMAL_UNIFORM_NATIVE_SCHEDULE_FROZEN',
            'method': 'PALU_UNIFORM',
            'ratio': ratio,
            'ratio_semantics': 'retained native rank fraction; compression=1-ratio',
            'capacity_rank': 65536,
            'integer_target_rank': int(round(65536 * ratio)),
            'realized_rank_sum': realized,
            'rank_quantum': 32,
            'native_unit': 'two persisted K/V slots per K/V module; anchor schema',
            'selection': selection,
            'schedule_sha': schedule_sha(selection),
            'resource': {
                'total_native_rank': realized,
                'prefill_cache_bytes': realized * 2048 * 2,
                'decode_cache_bytes': realized * 2048 * 2,
            },
            'source_anchor_schedule_sha': anchor['schedule_sha'],
            'remainder_policy': 'evenly-spaced deterministic native block remainder; exact rank match to Fisher anchor',
            'quality_used_to_design_schedule': False,
        }
        out = SCHEDULES / f'llama2_ratio_{tag}_uniform_schedule.json'
        write_json(out, record)
        uniform_records[tag] = {
            'schedule': str(out),
            'schedule_sha': record['schedule_sha'],
            'realized_rank_sum': realized,
        }

    protocol = {
        'status': 'FORMAL_QUALITY_PROTOCOL_MANIFEST_FROZEN_BEFORE_QUALITY',
        'campaign': 'PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1',
        'model': str(ROOT / 'models' / 'Llama-2-7b-hf'),
        'official_repo': str(ROOT / 'official' / 'Palu'),
        'official_repo_commit': 'bb22666e2ef96707e8dd21d93fc00146c2e0d615',
        'ratios': ratios,
        'methods': ['PALU_UNIFORM', 'PALU_FISHER', 'PALU_FDP_PRIOR', 'PALU_FDP_NATIVE'],
        'wiki': {
            'source': str(wiki),
            'source_sha256': sha(wiki),
            'split': 'test',
            'sequence_length': 2048,
            'chunk_policy': 'all floor(total_token_count / sequence_length) contiguous chunks',
            'joiner': 'historical hf_joined.raw materialization',
        },
        'c4': {
            'source': str(c4),
            'source_sha256': sha(c4),
            'split': 'validation/offline corpus',
            'docs_scanned': 1100,
            'chunks': 256,
            'sequence_length': 2048,
            'joiner': 'single ASCII space between documents',
            'chunk_ids': list(range(256)),
        },
        'scorer': {
            'implementation': 'historical official run_ppl_eval semantics, one common evaluator for every method',
            'logits': 'model.model hidden states followed by lm_head; native model dtype',
            'cross_entropy': 'torch.nn.CrossEntropyLoss default mean over shifted tokens',
            'masking': 'none beyond contiguous shift; no padding tokens in materialized chunks',
            'reduction': 'mean loss * sequence_length per chunk; PPL=exp(sum NLL/(chunks*sequence_length))',
            'use_cache': False,
            'special_tokens': 'historical tokenizer call defaults',
            'finite_value_policy': 'fail on non-finite loss',
        },
        'resource': {
            'native_rank_unit': 'rank',
            'rank_quantum': 32,
            'dtype_bytes': 2,
            'prefill_formula': 'native_rank * 2048 * 2',
            'decode_formula': 'native_rank * 2048 * 2',
            'exact_match_required': True,
        },
        'uniform_schedules': uniform_records,
        'formal_quality_started': False,
        'historical_results_modified': False,
    }
    manifest_path = formal_dir / 'FORMAL_QUALITY_PROTOCOL_MANIFEST.json'
    write_json(manifest_path, protocol)
    protocol['manifest_sha256'] = sha(manifest_path)
    write_json(manifest_path, protocol)

    status_path = BRANCH / 'audit' / 'CAMPAIGN_STATUS.json'
    status = json.loads(status_path.read_text())
    status.update({
        'status': 'FORMAL_QUALITY_READY',
        'blocker': None,
        'formal_quality_manifest_frozen': True,
        'formal_quality_manifest': str(manifest_path),
        'formal_quality_manifest_sha256': sha(manifest_path),
        'quality_started': False,
        'foreign_jobs_interrupted': 0,
        'gpu1_used': False,
    })
    write_json(status_path, status)
    print(json.dumps({
        'status': status['status'],
        'manifest': str(manifest_path),
        'manifest_sha256': status['formal_quality_manifest_sha256'],
        'uniform_schedules': uniform_records,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
