import gzip
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path('/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2')
DATA = ROOT / 'official' / 'data'
MODEL = ROOT / 'models' / 'Llama-2-7b-hf'
OUT = Path('/diskE/jiaqizhang/PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1/marginal/calibration_manifest.json')
SEED = 20260918
SEQ = 2048
N_WIKI = 8
N_C4 = 8


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def wiki_text(path):
    return path.read_text(encoding='utf-8')


def c4_docs(path, n=1100):
    out = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            out.append(json.loads(line)['text'])
            if len(out) == n:
                break
    return out


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    wiki_path = DATA / 'wiki-test.hf_joined.raw'
    c4_path = DATA / 'c4-validation.00000-of-00008.json.gz'
    wiki_ids = tokenizer(wiki_text(wiki_path), return_tensors='pt').input_ids[0].tolist()
    wiki_rng = random.Random(SEED)
    wiki_max = len(wiki_ids) - SEQ - 1
    wiki_starts = sorted(wiki_rng.sample(range(wiki_max), N_WIKI))
    docs = c4_docs(c4_path, 1100)
    c4_rng = random.Random(SEED + 1)
    candidates = []
    for i, doc in enumerate(docs):
        ids = tokenizer(doc, return_tensors='pt').input_ids[0].tolist()
        if len(ids) >= SEQ + 1:
            candidates.append((i, len(ids)))
    chosen = c4_rng.sample(candidates, N_C4)
    chosen.sort()
    manifest = {
        'status': 'FROZEN_BEFORE_MARGINAL_QUALITY',
        'task': 'PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1',
        'model': str(MODEL),
        'tokenizer': str(MODEL),
        'sequence_length': SEQ,
        'seed': SEED,
        'n_wikitext_chunks': N_WIKI,
        'n_c4_chunks': N_C4,
        'J_definition': '0.5 * mean token NLL over WikiText-2 chunks + 0.5 * mean token NLL over C4 chunks; PPL=exp(mean NLL) reported separately',
        'loss': 'torch.nn.functional.cross_entropy over next-token labels; no padding; fixed full 2048-token windows; fp32 reduction from model logits',
        'wikitext': {
            'source': str(wiki_path),
            'source_sha256': sha(wiki_path),
            'token_count': len(wiki_ids),
            'token_starts': wiki_starts,
            'chunks': [{'chunk_id': f'wiki_{i:02d}', 'token_start': s, 'token_end': s + SEQ} for i, s in enumerate(wiki_starts)],
        },
        'c4': {
            'source': str(c4_path),
            'source_sha256': sha(c4_path),
            'pool_docs': 1100,
            'chunks': [{'chunk_id': f'c4_{j:02d}', 'doc_index': int(i), 'token_count_at_manifest': int(n), 'token_start': 0, 'token_end': SEQ} for j, (i, n) in enumerate(chosen)],
        },
        'overlap_policy': 'independent calibration manifest; not used for Fisher calibration; fixed across every unit and ratio',
        'quality_parameter_search': False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'path': str(OUT), 'wiki_tokens': len(wiki_ids), 'wiki_starts': wiki_starts, 'c4_docs': [x[0] for x in chosen], 'sha256': sha(OUT)}, indent=2))


if __name__ == '__main__':
    main()
