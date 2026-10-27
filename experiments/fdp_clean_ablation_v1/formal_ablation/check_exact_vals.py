
import json, math
from pathlib import Path

p = Path('/diskF/jiaqizhang/fdp_clean_ablation_v1/formal_ablation/CORE_FORMAL_PER_WINDOW')

targets = ['b80472cc', '76bd0eb1', '31635586', 'fa4831cd', 'fdf1ef67']
for pref in targets:
    matches = list(p.glob(f"{pref}*"))
    if matches:
        d = matches[0]
        w = d / "wiki_windows.jsonl"
        lines = [json.loads(l) for l in w.read_text().splitlines() if l.strip()]
        total_nll = sum(float(l['total_nll']) for l in lines)
        total_tok = sum(int(l['token_count']) for l in lines)
        nll = total_nll / total_tok
        ppl = math.exp(nll)
        first_win_nll = float(lines[0]['total_nll']) / int(lines[0]['token_count'])
        print(f"{pref}: PPL={ppl:.8f} NLL={nll:.8f} Win0_NLL={first_win_nll:.8f}")
