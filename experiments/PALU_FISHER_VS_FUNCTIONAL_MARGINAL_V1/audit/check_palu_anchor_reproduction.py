import json
from pathlib import Path

anchor = json.loads(Path('/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/v2_runs/palu_fisher/PALU_V2_OFFICIAL_0P7_FISHER_ANCHOR.json').read_text())
f = anchor['module_fisher_means']
modules = list(anchor['head_wise_ranks'])
def build(ratio):
    base = {k: [512, 512] for k in modules}
    target = round(65536 * ratio)
    fisher_sum = sum(f[k] * 2 for k in modules)
    floats = {}
    for k in modules:
        floats[k] = [target * f[k] / fisher_sum] * 2
        base[k] = [min(512, int(x)) for x in floats[k]]
    idx = [(k, i) for k in modules for i in range(2)]
    idx.sort(key=lambda x: floats[x[0]][x[1]] - base[x[0]][x[1]])
    dif = target - sum(sum(v) for v in base.values())
    while dif > 0:
        for k, i in idx:
            if base[k][i] == 512:
                continue
            base[k][i] += 1
            dif -= 1
            if dif == 0:
                break
    rounded = {k: [max(1, round(v / 32)) * 32 for v in vals] for k, vals in base.items()}
    return rounded, sum(sum(v) for v in rounded.values())

got, total = build(0.7)
want = anchor['head_wise_ranks']
same = got == want
print(json.dumps({'same': same, 'got_total': total, 'want_total': anchor['realized_rank_sum'], 'first_got': list(got.items())[:5], 'first_want': list(want.items())[:5]}, indent=2))
