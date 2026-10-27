import json
import math
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
import csv

sys.path.insert(0, "/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")
from run_formula_search_engine import FormulaSearchRunner, derive_eb_schedule, clip_ranks_to_basis, oracle_lookup, match_ranks, TIER_CONF, TIERS
from realloc_rank_from_fragility import fragility_vector, minmax

runner = FormulaSearchRunner(device="cuda:2")

# Candidate A: Parent F_CURRENT
# w=(1,1,1,1,1), lambda=1.2, rho=0.75
cand_A = {
    "name": "Candidate_A_Parent",
    "weights": [1.0, 1.0, 1.0, 1.0, 1.0],
    "lambda": 1.2,
    "rho": 0.75
}

# Candidate B: Provisional F*
# w=(0.5, 1.0, 1.0, 0.5, 0.5), lambda=0.9, rho=0.50
cand_B = {
    "name": "Candidate_B_Provisional",
    "weights": [0.5, 1.0, 1.0, 0.5, 0.5],
    "lambda": 0.9,
    "rho": 0.50
}

def sha_of_list(lst):
    return hashlib.sha256(json.dumps(lst, separators=(',', ':')).encode('utf-8')).hexdigest()

def get_eb_and_ranks(conf, weights, lambda_eb, rho, c_scale=1.0):
    H = runner.step1_data[conf['tier']]['per_layer']
    layers = sorted(int(k) for k in H.keys())
    s_raw, _ = fragility_vector(H, layers, weights[0], weights[1], weights[2], weights[3], weights[4], False)
    s_tilde = minmax(s_raw)
    eb_sched = derive_eb_schedule(s_tilde, conf['eb_base'], lambda_eb, rho, c_scale=c_scale)
    ranks = []
    for li in range(32):
        rk, rv = oracle_lookup(runner.traces[conf['tier']][li], eb_sched[li])
        ranks.append([rk, rv])
    ranks_safe = clip_ranks_to_basis(ranks, runner.basis_dicts[conf['tier']])
    return eb_sched, ranks_safe

print("================================================================================")
print("TASK 2: SIMPLICITY / TIE AUDIT — CANDIDATE A vs CANDIDATE B")
print("================================================================================")

audit_task2 = {}
for tier in TIERS:
    conf = TIER_CONF[tier]
    conf['tier'] = tier
    
    # A Natural
    eb_A_nat, r_A_nat = get_eb_and_ranks(conf, cand_A["weights"], cand_A["lambda"], cand_A["rho"], c_scale=1.0)
    # A Matched
    r_A_match, c_A_match, diff_A = match_ranks(minmax(fragility_vector(runner.step1_data[tier]['per_layer'], sorted(int(k) for k in runner.step1_data[tier]['per_layer'].keys()), *cand_A["weights"], False)[0]), runner.traces[tier], runner.basis_dicts[tier], conf['eb_base'], conf['target_rank'], cand_A["lambda"], cand_A["rho"])
    eb_A_match = derive_eb_schedule(minmax(fragility_vector(runner.step1_data[tier]['per_layer'], sorted(int(k) for k in runner.step1_data[tier]['per_layer'].keys()), *cand_A["weights"], False)[0]), conf['eb_base'], cand_A["lambda"], cand_A["rho"], c_scale=c_A_match)
    
    # B Natural
    eb_B_nat, r_B_nat = get_eb_and_ranks(conf, cand_B["weights"], cand_B["lambda"], cand_B["rho"], c_scale=1.0)
    # B Matched
    r_B_match, c_B_match, diff_B = match_ranks(minmax(fragility_vector(runner.step1_data[tier]['per_layer'], sorted(int(k) for k in runner.step1_data[tier]['per_layer'].keys()), *cand_B["weights"], False)[0]), runner.traces[tier], runner.basis_dicts[tier], conf['eb_base'], conf['target_rank'], cand_B["lambda"], cand_B["rho"])
    eb_B_match = derive_eb_schedule(minmax(fragility_vector(runner.step1_data[tier]['per_layer'], sorted(int(k) for k in runner.step1_data[tier]['per_layer'].keys()), *cand_B["weights"], False)[0]), conf['eb_base'], cand_B["lambda"], cand_B["rho"], c_scale=c_B_match)
    
    # Evals
    eval_A_match = runner.evaluate_tier_ranks(tier, r_A_match)
    eval_B_match = runner.evaluate_tier_ranks(tier, r_B_match)
    eval_A_nat = runner.evaluate_tier_ranks(tier, r_A_nat)
    eval_B_nat = runner.evaluate_tier_ranks(tier, r_B_nat)
    
    bytes_per_r = conf['target_b_live'] // conf['target_rank']
    
    audit_task2[tier] = {
        "A": {
            "nat_eb_sha": sha_of_list([round(x, 8) for x in eb_A_nat]),
            "nat_rank_sha": sha_of_list(r_A_nat),
            "nat_tot_rank": sum(sum(x) for x in r_A_nat),
            "nat_b_live": sum(sum(x) for x in r_A_nat) * bytes_per_r,
            "nat_gap_pct": abs(sum(sum(x) for x in r_A_nat) - conf['target_rank']) / conf['target_rank'] * 100.0,
            "nat_j_dev": eval_A_nat["j_tier"],
            "nat_wiki": eval_A_nat["wiki_nll"],
            "nat_c4": eval_A_nat["c4_nll"],
            
            "matched_c": c_A_match,
            "matched_eb_sha": sha_of_list([round(x, 8) for x in eb_A_match]),
            "matched_rank_sha": sha_of_list(r_A_match),
            "matched_tot_rank": sum(sum(x) for x in r_A_match),
            "matched_b_live": sum(sum(x) for x in r_A_match) * bytes_per_r,
            "matched_gap_pct": eval_A_match["gap_pct"],
            "matched_j_dev": eval_A_match["j_tier"],
            "matched_wiki": eval_A_match["wiki_nll"],
            "matched_c4": eval_A_match["c4_nll"],
        },
        "B": {
            "nat_eb_sha": sha_of_list([round(x, 8) for x in eb_B_nat]),
            "nat_rank_sha": sha_of_list(r_B_nat),
            "nat_tot_rank": sum(sum(x) for x in r_B_nat),
            "nat_b_live": sum(sum(x) for x in r_B_nat) * bytes_per_r,
            "nat_gap_pct": abs(sum(sum(x) for x in r_B_nat) - conf['target_rank']) / conf['target_rank'] * 100.0,
            "nat_j_dev": eval_B_nat["j_tier"],
            "nat_wiki": eval_B_nat["wiki_nll"],
            "nat_c4": eval_B_nat["c4_nll"],
            
            "matched_c": c_B_match,
            "matched_eb_sha": sha_of_list([round(x, 8) for x in eb_B_match]),
            "matched_rank_sha": sha_of_list(r_B_match),
            "matched_tot_rank": sum(sum(x) for x in r_B_match),
            "matched_b_live": sum(sum(x) for x in r_B_match) * bytes_per_r,
            "matched_gap_pct": eval_B_match["gap_pct"],
            "matched_j_dev": eval_B_match["j_tier"],
            "matched_wiki": eval_B_match["wiki_nll"],
            "matched_c4": eval_B_match["c4_nll"],
        },
        "match_shas_identical": sha_of_list(r_A_match) == sha_of_list(r_B_match)
    }
    
    print(f"\n--- {tier.upper()} ---")
    print(f"  Matched Rank SHAs Identical: {audit_task2[tier]['match_shas_identical']}")
    print(f"  A Matched: Rank={audit_task2[tier]['A']['matched_tot_rank']}, SHA={audit_task2[tier]['A']['matched_rank_sha'][:12]}..., J={audit_task2[tier]['A']['matched_j_dev']:.6f}")
    print(f"  B Matched: Rank={audit_task2[tier]['B']['matched_tot_rank']}, SHA={audit_task2[tier]['B']['matched_rank_sha'][:12]}..., J={audit_task2[tier]['B']['matched_j_dev']:.6f}")
    print(f"  A Natural: Rank={audit_task2[tier]['A']['nat_tot_rank']}, Gap={audit_task2[tier]['A']['nat_gap_pct']:.3f}%, J={audit_task2[tier]['A']['nat_j_dev']:.6f}")
    print(f"  B Natural: Rank={audit_task2[tier]['B']['nat_tot_rank']}, Gap={audit_task2[tier]['B']['nat_gap_pct']:.3f}%, J={audit_task2[tier]['B']['nat_j_dev']:.6f}")

print("\n================================================================================")
print("TASK 3: NATURAL STATE QUALITY SUMMARY")
print("================================================================================")
for cand_label, ckey in [("Candidate A (Parent)", "A"), ("Candidate B (Provisional)", "B")]:
    j_str = (audit_task2["medium"][ckey]["nat_j_dev"] + audit_task2["aggressive"][ckey]["nat_j_dev"] + audit_task2["ultra"][ckey]["nat_j_dev"]) / 3.0
    print(f"{cand_label} Natural J_STRONG = {j_str:.6f} | Light J = {audit_task2['light'][ckey]['nat_j_dev']:.6f}")

print("\n================================================================================")
print("TASK 4: FINAL COMPONENT LOO (ON CANDIDATE A vs CANDIDATE B)")
print("================================================================================")
# We evaluate LOO on Candidate A (simpler formula) and Candidate B
def run_loo_for_cand(cand_info):
    print(f"\nEvaluating LOO on {cand_info['name']}:")
    comp_names = ["d1_z", "d2_s", "d3_v", "d4_a", "d5_h"]
    base_res = runner.evaluate_candidate(
        candidate_name=f"{cand_info['name']}_base",
        mode="prop",
        weights=cand_info["weights"],
        lambda_eb=cand_info["lambda"],
        rho=cand_info["rho"],
        eta=0.0
    )
    loo_dict = {}
    for ci, cname in enumerate(comp_names):
        w_loo = list(cand_info["weights"])
        w_loo[ci] = 0.0
        loo_res = runner.evaluate_candidate(
            candidate_name=f"{cand_info['name']}_minus_{cname}",
            mode="prop",
            weights=w_loo,
            lambda_eb=cand_info["lambda"],
            rho=cand_info["rho"],
            eta=0.0
        )
        d_strong = loo_res["j_strong"] - base_res["j_strong"]
        d_light = loo_res["delta_j_light"] - base_res["delta_j_light"]
        loo_dict[cname] = {
            "j_strong": loo_res["j_strong"],
            "delta_strong": d_strong,
            "delta_light": d_light,
            "tiers": {t: loo_res["tiers"][t]["j_tier"] for t in TIERS}
        }
        print(f"  minus {cname:4s}: J_STRONG={loo_res['j_strong']:.6f} | Delta={d_strong:+.6f} | Light_Delta={d_light:+.6f}")
    return base_res, loo_dict

base_A, loo_A = run_loo_for_cand(cand_A)
base_B, loo_B = run_loo_for_cand(cand_B)

print("\n================================================================================")
print("TASK 5: STATISTICAL CONSISTENCY (AUDIT BOOTSTRAP METHOD & DEFINITION)")
print("================================================================================")
# Audit Aggressive tier comparison between Candidate A / Hist / Candidate B
from formula_engine import paired_bootstrap_test

def detailed_bootstrap_audit(f0_losses, cand_losses, n_boot=10000, seed=42):
    diff = np.array(cand_losses) - np.array(f0_losses)
    obs_mean = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    
    # Percentile CI (two-sided 95%)
    ci_low_2sided = float(np.percentile(boot_means, 2.5))
    ci_high_2sided = float(np.percentile(boot_means, 97.5))
    
    # One-sided p-value (H0: diff >= 0, i.e. candidate is not better)
    p_onesided = float(np.mean(boot_means >= 0))
    # Two-sided p-value under symmetric null
    p_twosided = float(np.mean(np.abs(boot_means - np.mean(boot_means)) >= np.abs(obs_mean)))
    
    return {
        "n_samples": len(diff),
        "obs_mean": obs_mean,
        "ci_95_twosided": [ci_low_2sided, ci_high_2sided],
        "p_onesided": p_onesided,
        "p_twosided": p_twosided,
        "ci_contains_zero": ci_low_2sided <= 0.0 <= ci_high_2sided
    }

wins_A_agg = audit_task2["aggressive"]["A"]["matched_wiki_wins"] = runner.dev_cache["aggressive"][audit_task2["aggressive"]["A"]["matched_rank_sha"]]["wiki_per_win"] + runner.dev_cache["aggressive"][audit_task2["aggressive"]["A"]["matched_rank_sha"]]["c4_per_win"]
# Historical FDP on Aggressive
hist_sha_agg = runner.dev_cache["aggressive"][audit_task2["aggressive"]["A"]["matched_rank_sha"]]["sha"]
# Look up hist ranks
H_agg = runner.step1_data["aggressive"]['per_layer']
s_raw_agg, _ = fragility_vector(H_agg, sorted(int(k) for k in H_agg.keys()), 1, 1, 1, 1, 1, False)
r_hist_agg, _, _ = match_ranks(minmax(s_raw_agg), runner.traces["aggressive"], runner.basis_dicts["aggressive"], TIER_CONF["aggressive"]['eb_base'], TIER_CONF["aggressive"]['target_rank'], 1.2, 1.0)
eval_hist_agg = runner.evaluate_tier_ranks("aggressive", r_hist_agg)
wins_hist_agg = eval_hist_agg["wiki_per_win"] + eval_hist_agg["c4_per_win"]

boot_audit = detailed_bootstrap_audit(wins_hist_agg, wins_A_agg)
print(f"Aggressive Tier (Candidate vs Historical):")
print(f"  Observed Mean Diff: {boot_audit['obs_mean']:+.6f}")
print(f"  Two-sided 95% Percentile CI: [{boot_audit['ci_95_twosided'][0]:+.6f}, {boot_audit['ci_95_twosided'][1]:+.6f}]")
print(f"  CI Contains Zero: {boot_audit['ci_contains_zero']}")
print(f"  One-sided p-value (prob mean >= 0): {boot_audit['p_onesided']:.4f}")
print(f"  Explanation: The previously reported p=0.0389 was a ONE-SIDED p-value testing H0: mean >= 0. In two-sided testing, because CI contains zero, it is NOT statistically significant at two-sided alpha=0.05!")

print("\n================================================================================")
print("TASK 7: CAUSAL MECHANISM REPORT (SL, CHI_L, PL, RATIOS)")
print("================================================================================")
# Detailed report for L0, L1, L2, L16, L31 across all 4 tiers
causal_profile_table = {}
for tier in TIERS:
    e_idx = TIERS.index(tier)
    G_tier = runner.G_4d[:, :, :, e_idx]
    H = runner.step1_data[tier]['per_layer']
    layers = sorted(int(k) for k in H.keys())
    s_raw, _ = fragility_vector(H, layers, 1.0, 1.0, 1.0, 1.0, 1.0, False)
    
    C_raw, CHI_raw, P_raw = [], [], []
    for l in range(32):
        pos_resp = np.maximum(0.0, G_tier[:, l:32, l])
        C_raw.append(float(np.mean(pos_resp)))
        pos_self = np.maximum(0.0, G_tier[:, l, l])
        CHI_raw.append(float(np.mean(pos_self)))
        if l < 31:
            pos_down = np.maximum(0.0, G_tier[:, (l+1):32, l])
            P_raw.append(float(np.mean(pos_down)))
        else:
            P_raw.append(0.0)
            
    causal_profile_table[tier] = []
    print(f"\n--- {tier.upper()} CAUSAL ATTRIBUTION PROFILE ---")
    print(f"Layer | S_raw     | Self_CHI  | Downstream_P | Under-attribution P / (S + eps)")
    for l in [0, 1, 2, 16, 31]:
        ratio = P_raw[l] / (s_raw[l] + 1e-6)
        causal_profile_table[tier].append({
            "layer": l,
            "S_raw": s_raw[l],
            "chi_raw": CHI_raw[l],
            "P_raw": P_raw[l],
            "ratio": ratio
        })
        print(f"L{l:02d}  | {s_raw[l]:.6f} | {CHI_raw[l]:.6f}  | {P_raw[l]:.6f}     | {ratio:.6f}")

# Save full results to JSON for artifact export
final_validity_data = {
    "task2_audit": audit_task2,
    "loo_A": loo_A,
    "loo_B": loo_B,
    "boot_audit": boot_audit,
    "causal_profile": causal_profile_table
}
with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/final/FINAL_CANDIDATE_VALIDITY_DATA.json", "w") as f:
    json.dump(final_validity_data, f, indent=2)
print("\nSaved FINAL_CANDIDATE_VALIDITY_DATA.json successfully.")
