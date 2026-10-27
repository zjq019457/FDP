#!/usr/bin/env python3
"""FDP_PROFILE_ROBUSTNESS_V1 analysis (CPU). Gate: the frozen step1.json replays to the paper's 0.74x FDP-EB ranks."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/code")
import c45lib as L
sys.path.insert(0, str(L.EIGEN))
from realloc_rank_from_fragility import fragility_vector, minmax

R = Path("/diskF/jiaqizhang/FDP_PROFILE_ROBUSTNESS_V1")
FROZEN = Path("/diskF/jiaqizhang/fdp125_final_campaign_v1/results/llama/medium/step1.json")
PAPER_FDP = [[808, 992], [384, 872], [400, 776], [712, 936], [680, 928], [776, 952], [792, 960], [792, 952], [624, 904], [792, 960], [696, 928],
             [720, 928], [896, 992], [688, 936], [776, 960], [624, 912], [688, 936], [632, 904], [504, 840], [448, 824], [392, 800], [496, 856],
             [440, 824], [432, 832], [400, 824], [400, 824], [360, 816], [488, 856], [504, 856], [1016, 1024], [888, 992], [696, 928]]
E = 0.06


def q_of(p):
    pl = json.loads(Path(p).read_text())["per_layer"]
    raw, _ = fragility_vector(pl, sorted(int(x) for x in pl), 1, 1, 0, 0, 1, False)
    return np.array(minmax(raw))


def ranks_of(p):
    eb = L.derive_eb(p, E, (1, 1, 0, 0, 1)); eb[0] = E
    return L.select(eb, False)[0]


def rk(x): return np.argsort(np.argsort(x))
def spear(a, b): return float(np.corrcoef(rk(a), rk(b))[0, 1])
def topk(a, b, k): return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k]))


qF = q_of(FROZEN); rF = ranks_of(FROZEN)
out = {"gate_frozen_replays_paper_ranks": rF == PAPER_FDP, "B_paper": L.B_of(PAPER_FDP)}
for name, f in [("default_probe_rerun", R / "step1_default_probe.json"), ("wikitext2_train_8x512", R / "step1_wikitext2_train8.json")]:
    if not f.exists(): continue
    q = q_of(f); r = ranks_of(f)
    diff = [l for l in range(32) if r[l] != PAPER_FDP[l]]
    out[name] = {"spearman_q": spear(qF, q), "pearson_q": float(np.corrcoef(qF, q)[0, 1]), "top4_overlap": topk(qF, q, 4), "top8_overlap": topk(qF, q, 8),
                 "bottom8_overlap": topk(-qF, -q, 8), "max_abs_dq": float(np.max(np.abs(q - qF))), "B": L.B_of(r),
                 "B_gap_pct_vs_paper": 100 * (L.B_of(r) - L.B_of(PAPER_FDP)) / L.B_of(PAPER_FDP), "layers_with_rank_change": diff,
                 "n_layers_changed": len(diff), "total_abs_rank_change": int(sum(abs(a - b) for x, y in zip(r, PAPER_FDP) for a, b in zip(x, y))),
                 "q": q.tolist()}
out["q_frozen"] = qF.tolist()
(R / "ROBUSTNESS_RESULT.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "q"}) for k, v in out.items() if k != "q_frozen"}, indent=1))
