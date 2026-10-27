import sys
import json
import math
import torch
from pathlib import Path

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
sys.path.insert(0, str(WORKSPACE / "formal_ablation"))
from run_formal_suite import TIER_CONF, TRACES, BASIS_DICTS, derive_eb_schedule, oracle_lookup, clip_ranks_to_basis, minmax

spec_tilde = minmax([0.5 + (l + 1)/32.0 for l in range(32)])
for tier in ["aggressive", "ultra"]:
    conf = TIER_CONF[tier]
    traces = TRACES[tier]
    basis = BASIS_DICTS[tier]
    tgt_r = conf["target_rank"]
    best_c, best_diff, best_tot = 1.0, 999999, 0
    # Let's search broader c from 0.2 to 3.0
    for c_int in range(20, 300):
        c = c_int / 100.0
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
    print("Tier " + tier + ": best_c=" + str(best_c) + ", best_tot=" + str(best_tot) + ", tgt=" + str(tgt_r) + ", diff=" + str(best_diff) + " (gap=" + str(round(gap_pct, 4)) + "%)")
