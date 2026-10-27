import sys
import json
import math
import torch
from pathlib import Path

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")

TIER_CONF = {
    "aggressive": {
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "eb_base": 0.090,
        "target_rank": 38600
    },
    "ultra": {
        "report": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb_report.json",
        "basis": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb.pt",
        "eb_base": 0.120,
        "target_rank": 30328
    }
}

def minmax(x, eps=1e-12):
    lo, hi = min(x), max(x)
    span = hi - lo
    if span < eps:
        return [0.5 for _ in x]
    return [(v - lo) / span for v in x]

def oracle_lookup(trace, eb, svd_floor=0.30):
    best_rk, best_rv = 1024, 1024
    for step in trace:
        th = step["threshold"]
        if step["err"] <= eb and th >= svd_floor:
            best_rk = step["rank_kq"]
            best_rv = step["rank_v"]
        else:
            break
    return best_rk, best_rv

def clip_ranks_to_basis(ranks, basis_dict):
    clipped = []
    for li, (rk, rv) in enumerate(ranks):
        q_key = f"base_model.model.model.layers.{li}.self_attn.q_proj_basis"
        v_key = f"base_model.model.model.layers.{li}.self_attn.v_proj_basis"
        max_k = basis_dict[q_key].shape[0] * 8
        max_v = basis_dict[v_key].shape[0] * 8
        clipped.append([min(rk, max_k), min(rv, max_v)])
    return clipped

def derive_eb_schedule(s_tilde, eb_base, lambda_eb, rho, c_scale=1.0):
    eb_inv = [eb_base / (1.0 + lambda_eb * st) for st in s_tilde]
    m = sum(eb_inv) / len(eb_inv)
    scale = (eb_base / m) if m > 0 else 1.0
    eb_norm = [e * scale for e in eb_inv]
    eps0 = math.exp(rho * math.log(eb_base) + (1.0 - rho) * math.log(eb_norm[0]))
    eb_norm[0] = eps0
    eb_final = [e * c_scale for e in eb_norm]
    if rho >= 0.999:
        eb_final[0] = eb_base
    return eb_final

spec_tilde = minmax([0.5 + (l + 1)/32.0 for l in range(32)])

for tier in ["aggressive", "ultra"]:
    conf = TIER_CONF[tier]
    with open(conf["report"]) as f:
        rep = json.load(f)
    traces = [l["runs"][0]["threshold_trace"] for l in rep["layers"]]
    basis = torch.load(conf["basis"], map_location="cpu")
    tgt_r = conf["target_rank"]
    
    best_c, best_diff, best_tot = 1.0, 999999, 0
    # Fine search from 0.500 to 2.000 in steps of 0.001
    for c_int in range(500, 2000):
        c = c_int / 1000.0
        eb_sched = derive_eb_schedule(spec_tilde, conf["eb_base"], 1.2, 0.75, c)
        ranks = []
        for li in range(32):
            rk, rv = oracle_lookup(traces[li], eb_sched[li])
            ranks.append([rk, rv])
        ranks_safe = clip_ranks_to_basis(ranks, basis)
        tot_r = sum(sum(r) for r in ranks_safe)
        diff = abs(tot_r - tgt_r)
        if diff < best_diff:
            best_diff = diff
            best_c = c
            best_tot = tot_r
    gap_pct = best_diff / tgt_r * 100.0
    print(f"Tier {tier}: best_c={best_c}, best_tot={best_tot}, tgt={tgt_r}, diff={best_diff} (gap={gap_pct:.4f}%)")
