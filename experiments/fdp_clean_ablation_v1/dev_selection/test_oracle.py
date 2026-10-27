
import json

rep_path = "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb_report.json"
with open(rep_path) as f:
    rep = json.load(f)

with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/per_layer_eb_ultra_120.json") as f:
    eb_schedule = json.load(f)["error_budget_by_layer"]

# Oracle lookup function
def oracle_lookup(trace, eb, svd_floor=0.30, min_rank=64):
    best_rk = 1024
    best_rv = 1024
    best_th = 1.0
    for step in trace:
        th = step["threshold"]
        rk = step["rank_kq"]
        rv = step["rank_v"]
        err = step["err"]
        if err <= eb and th >= svd_floor:
            best_rk = rk
            best_rv = rv
            best_th = th
        else:
            break
    return best_rk, best_rv, best_th

exact_matches = 0
for li, layer in enumerate(rep["layers"]):
    run = layer["runs"][0]
    actual_rk = run["rank_kq_final"]
    actual_rv = run["rank_v_final"]
    
    eb_l = eb_schedule[li]
    trace = run["threshold_trace"]
    pred_rk, pred_rv, pred_th = oracle_lookup(trace, eb_l)
    
    match = (actual_rk == pred_rk and actual_rv == pred_rv)
    if match:
        exact_matches += 1
    else:
        print(f"Layer {li}: actual=({actual_rk}, {actual_rv}) vs pred=({pred_rk}, {pred_rv})")

print(f"Exact rank oracle matches: {exact_matches} / {len(rep['layers'])} ({(exact_matches/len(rep['layers']))*100:.1f}%)")
