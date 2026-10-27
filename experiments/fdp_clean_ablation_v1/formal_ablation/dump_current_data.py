#!/usr/bin/env python3
import json, math
from pathlib import Path

p = Path('/diskF/jiaqizhang/fdp_clean_ablation_v1/formal_ablation/CORE_FORMAL_PER_WINDOW')

auth_files = [
    Path('/diskF/jiaqizhang/fdp_clean_ablation_v1/formal_ablation/CORE_RESOURCE_AUTHORITY.json'),
    Path('/diskF/jiaqizhang/fdp_clean_ablation_v1/formal_ablation/LOO_CONTROLS_AUTHORITY.json')
]

sha_map = {}
if auth_files[0].exists():
    for r in json.load(open(auth_files[0])):
        sha_map[r['natural_rank_sha']] = {'tier': r['tier'], 'group': 'CORE_NATURAL', 'config': r['method'], 'tot_rank': r['natural_tot_rank']}
        sha_map[r['matched_rank_sha']] = {'tier': r['tier'], 'group': 'CORE_MATCHED', 'config': r['method'], 'tot_rank': r['matched_tot_rank']}
if auth_files[1].exists():
    for r in json.load(open(auth_files[1])):
        sha_map[r['sha']] = {'tier': r['tier'], 'group': r['group'], 'config': r['config_name'], 'tot_rank': r['tot_rank']}

results = []
for sdir in sorted(p.iterdir()):
    if not sdir.is_dir(): continue
    sha = sdir.name
    wiki_win = sdir / 'wiki_windows.jsonl'
    c4_win = sdir / 'c4_windows.jsonl'
    done_f = sdir / 'STATE_EVAL_DONE.json'
    
    w_ppl, c_ppl, zs_macro = None, None, None
    if wiki_win.exists():
        lines = [json.loads(l) for l in wiki_win.read_text().splitlines() if l.strip()]
        if lines:
            w_nll = sum(float(l['total_nll']) for l in lines) / sum(int(l['token_count']) for l in lines)
            w_ppl = math.exp(w_nll)
    if c4_win.exists():
        lines = [json.loads(l) for l in c4_win.read_text().splitlines() if l.strip()]
        if lines:
            c_nll = sum(float(l['total_nll']) for l in lines) / sum(int(l['token_count']) for l in lines)
            c_ppl = math.exp(c_nll)
    if done_f.exists():
        d = json.loads(done_f.read_text())
        zs_macro = d.get('zero_shot_macro')
        if not w_ppl: w_ppl = d.get('wiki_ppl')
        if not c_ppl: c_ppl = d.get('c4_ppl')

    meta = sha_map.get(sha, {})
    if w_ppl is not None or c_ppl is not None or zs_macro is not None:
        results.append({
            'sha': sha[:12],
            'tier': meta.get('tier', 'unknown'),
            'group': meta.get('group', 'other'),
            'config': meta.get('config', 'unmapped'),
            'tot_rank': meta.get('tot_rank', '-'),
            'wiki_ppl': round(w_ppl, 4) if w_ppl else None,
            'c4_ppl': round(c_ppl, 4) if c_ppl else None,
            'zs_macro': round(zs_macro, 4) if zs_macro else None,
            'status': 'DONE' if done_f.exists() else 'PPL_DONE'
        })

print(json.dumps(results, indent=2))
