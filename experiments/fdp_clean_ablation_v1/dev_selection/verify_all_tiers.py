
import json
from pathlib import Path

tiers = {
    "medium": {
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "eb": 0.060
    },
    "aggressive": {
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "eb": 0.090
    },
    "ultra": {
        "report": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb_report.json",
        "basis": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb.pt",
        "eb": 0.120
    }
}

def oracle_lookup(trace, eb, svd_floor=0.30):
    best_rk = 1024
    best_rv = 1024
    for step in trace:
        th = step["threshold"]
        rk = step["rank_kq"]
        rv = step["rank_v"]
        err = step["err"]
        if err <= eb and th >= svd_floor:
            best_rk = rk
            best_rv = rv
        else:
            break
    return best_rk, best_rv

for name, t in tiers.items():
    with open(t["report"]) as f:
        rep = json.load(f)
    budgets = rep.get("budgets", [t["eb"]])
    matches = 0
    for li, layer in enumerate(rep["layers"]):
        run = layer["runs"][0]
        actual_rk = run["rank_kq_final"]
        actual_rv = run["rank_v_final"]
        eb_l = run["error_budget"]
        pred_rk, pred_rv = oracle_lookup(run["threshold_trace"], eb_l)
        if actual_rk == pred_rk and actual_rv == pred_rv:
            matches += 1
    print(f"Tier {name.upper()}: matches = {matches}/{len(rep['layers'])} (basis exists: {Path(t['basis']).exists()})")
