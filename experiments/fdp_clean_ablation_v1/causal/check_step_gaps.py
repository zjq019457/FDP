import json
from pathlib import Path

TIER_CONF = {
    "light": {
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "eb_base": 0.045
    },
    "medium": {
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "eb_base": 0.060
    },
    "aggressive": {
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "eb_base": 0.090
    },
    "ultra": {
        "report": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb_report.json",
        "eb_base": 0.120
    }
}

for tier, conf in TIER_CONF.items():
    print(f"\n=== TIER: {tier.upper()} (base={conf['eb_base']}) ===")
    rep = json.load(open(conf["report"]))
    step_gaps = []
    for li in range(32):
        t = rep["layers"][li]["runs"][0]["threshold_trace"]
        # find base step index
        base_idx = 0
        for i, s in enumerate(t):
            if s["err"] <= conf["eb_base"] and s["threshold"] >= 0.30:
                base_idx = i
            else:
                break
        next_err = t[base_idx + 1]["err"] if base_idx + 1 < len(t) else t[base_idx]["err"]
        gap = next_err - conf["eb_base"]
        step_gaps.append(gap)
    print(f"Base step gap: min={min(step_gaps):.4f}, mean={sum(step_gaps)/len(step_gaps):.4f}, max={max(step_gaps):.4f}")
    print(f"Sample layer 0..4 gaps: {[round(g, 4) for g in step_gaps[:5]]}")
