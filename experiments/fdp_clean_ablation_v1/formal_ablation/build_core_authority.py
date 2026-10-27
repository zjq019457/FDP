import sys
#!/usr/bin/env python3
"""
CORE CAMPAIGN SETUP & DISPATCH SPECIFICATION
============================================
Defines the 16 core logical rows:
4 Tiers (Light, Medium, Aggressive, Ultra) x 4 Methods:
  M1: F*_DEV (rho=0.75, lambda=1.2, w=(1,1,1,1,1), eta=0)
  M2: Historical FDP (rho=1.0, lambda=1.2, w=(1,1,1,1,1), eta=0)
  M3: Uniform resource-matched comparator
  M4: No-anchor FDP (rho=0.0, lambda=1.2, w=(1,1,1,1,1), eta=0)

Dual reporting: Natural state recorded first, then Matched-control state.
"""
import json
import math
import hashlib
import csv
from pathlib import Path

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR = WORKSPACE / "formal_ablation"
FORMAL_DIR.mkdir(parents=True, exist_ok=True)

TIER_TARGETS = {
    "light": {
        "target_b_live": 218988544,
        "target_rank": 53464,
        "bytes_per_r": 4096,
        "eb_base": 0.045,
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb045.json",
        "uniform_source": "/diskF/jiaqizhang/fdp_may2026_final_reconstruction_v1/historical_track/light/uniform_043/final_model/low_rank_config.json"
    },
    "medium": {
        "target_b_live": 198377472,
        "target_rank": 48432,
        "bytes_per_r": 4096,
        "eb_base": 0.060,
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb060.json",
        "uniform_source": "/diskF/jiaqizhang/fdp_may2026_final_reconstruction_v1/historical_track/medium/uniform_058/final_model/low_rank_config.json"
    },
    "aggressive": {
        "target_b_live": 158105600,
        "target_rank": 38600,
        "bytes_per_r": 4096,
        "eb_base": 0.090,
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb090.json",
        "uniform_source": "/diskF/jiaqizhang/fdp_may2026_final_reconstruction_v1/historical_track/aggressive/uniform_088/final_model/low_rank_config.json"
    },
    "ultra": {
        "target_b_live": 124223488,
        "target_rank": 30328,
        "bytes_per_r": 4096,
        "eb_base": 0.120,
        "report": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb_report.json",
        "basis": "/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass2_historical_fdp120/eigen_basis_llama3_8b_wikitext2_frag_eb120_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb120.json",
        "uniform_source": "authority_ultra_lock"
    }
}

def sha256_ranks(ranks):
    return hashlib.sha256(json.dumps(ranks, separators=(',', ':')).encode('utf-8')).hexdigest()

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

def match_ranks(s_tilde, traces, basis_dict, eb_base, target_rank, lambda_eb, rho):
    best_ranks = None
    best_diff = float("inf")
    best_c = 1.0
    # Coarse sweep
    for c in [round(0.60 + x * 0.02, 3) for x in range(51)]:
        eb_sched = derive_eb_schedule(s_tilde, eb_base, lambda_eb, rho, c)
        ranks = []
        for li in range(32):
            rk, rv = oracle_lookup(traces[li], eb_sched[li])
            ranks.append([rk, rv])
        ranks_safe = clip_ranks_to_basis(ranks, basis_dict)
        tot_r = sum(sum(r) for r in ranks_safe)
        diff = abs(tot_r - target_rank)
        if diff < best_diff:
            best_diff = diff
            best_ranks = ranks_safe
            best_c = c
            if diff == 0:
                break
    # Fine sweep
    for c in [round(best_c + x * 0.001, 4) for x in range(-30, 31) if x != 0]:
        if c <= 0:
            continue
        eb_sched = derive_eb_schedule(s_tilde, eb_base, lambda_eb, rho, c)
        ranks = []
        for li in range(32):
            rk, rv = oracle_lookup(traces[li], eb_sched[li])
            ranks.append([rk, rv])
        ranks_safe = clip_ranks_to_basis(ranks, basis_dict)
        tot_r = sum(sum(r) for r in ranks_safe)
        diff = abs(tot_r - target_rank)
        if diff < best_diff:
            best_diff = diff
            best_ranks = ranks_safe
            best_c = c
            if diff == 0:
                break
    return best_ranks, best_c, best_diff

def classify_res(gap_pct):
    if gap_pct == 0.0:
        return "EXACT"
    elif gap_pct <= 0.15:
        return "STRICT_MATCH"
    elif gap_pct <= 1.0:
        return "NEAR_MATCHED"
    else:
        return "FRONTIER"

sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
from realloc_rank_from_fragility import fragility_vector, minmax
import torch

rows_authority = []
configs_cache = {}

for tier in ["light", "medium", "aggressive", "ultra"]:
    conf = TIER_TARGETS[tier]
    tgt_b = conf["target_b_live"]
    tgt_r = conf["target_rank"]
    bpr = conf["bytes_per_r"]
    
    with open(conf["report"]) as f:
        rep = json.load(f)
    traces = [l["runs"][0]["threshold_trace"] for l in rep["layers"]]
    basis = torch.load(conf["basis"], map_location="cpu")
    with open(conf["step1"]) as f:
        s1 = json.load(f)["per_layer"]
    
    s_raw, _ = fragility_vector(s1, list(range(32)), 1.0, 1.0, 1.0, 1.0, 1.0, False)
    s_tilde = minmax(s_raw)
    
    # 4 Methods: M1 (F* rho=0.75), M2 (Hist rho=1.0), M3 (Uniform), M4 (rho=0.0)
    methods = [
        ("M1_F_DEV_STAR", 0.75, 1.2),
        ("M2_HIST_FDP",   1.00, 1.2),
        ("M3_UNIFORM",    None, None),
        ("M4_NO_ANCHOR",  0.00, 1.2)
    ]
    
    for m_name, rho, lam in methods:
        row_id = f"{tier}_{m_name}"
        if m_name == "M3_UNIFORM":
            if tier == "ultra":
                with open("/diskF/jiaqizhang/fdp_clean_ablation_v1/authority/ULTRA_UNIFORM_COMPARATOR_LOCK.json") as f:
                    u_lock = json.load(f)
                u_ranks = u_lock["closest_comparator"]["ranks"]
                u_b = u_lock["closest_comparator"]["B_live"]
                u_tot = u_lock["closest_comparator"]["total_rank"]
                u_gap = u_lock["closest_comparator"]["gap_pct"]
                u_sha = u_lock["closest_comparator"]["rank_sha"]
            else:
                with open(conf["uniform_source"]) as f:
                    u_ranks = json.load(f)
                u_tot = sum(sum(r) for r in u_ranks)
                u_b = u_tot * bpr
                u_gap = (u_tot - tgt_r) / tgt_r * 100.0
                u_sha = sha256_ranks(u_ranks)
                
            res_class = classify_res(abs(u_gap))
            r_entry = {
                "tier": tier,
                "method": m_name,
                "row_id": row_id,
                "target_b_live": tgt_b,
                "target_rank": tgt_r,
                "natural_ranks": u_ranks,
                "natural_tot_rank": u_tot,
                "natural_b_live": u_b,
                "natural_gap_pct": round(u_gap, 4),
                "natural_rank_sha": u_sha,
                "natural_eb_vector": None,
                "matched_c": 1.0,
                "matched_ranks": u_ranks,
                "matched_tot_rank": u_tot,
                "matched_b_live": u_b,
                "matched_gap_pct": round(u_gap, 4),
                "matched_rank_sha": u_sha,
                "matched_eb_vector": None,
                "resource_class": res_class,
                "note": "Uniform baseline (identical for natural and matched)"
            }
        else:
            # Non-uniform methods: compute NATURAL state first
            eb_nat = derive_eb_schedule(s_tilde, conf["eb_base"], lam, rho, c_scale=1.0)
            r_nat = []
            for li in range(32):
                rk, rv = oracle_lookup(traces[li], eb_nat[li])
                r_nat.append([rk, rv])
            r_nat = clip_ranks_to_basis(r_nat, basis)
            tot_nat = sum(sum(r) for r in r_nat)
            b_nat = tot_nat * bpr
            gap_nat = (tot_nat - tgt_r) / tgt_r * 100.0
            sha_nat = sha256_ranks(r_nat)
            
            # Compute MATCHED state
            r_match, c_match, diff_m = match_ranks(s_tilde, traces, basis, conf["eb_base"], tgt_r, lam, rho)
            tot_match = sum(sum(r) for r in r_match)
            b_match = tot_match * bpr
            gap_match = (tot_match - tgt_r) / tgt_r * 100.0
            sha_match = sha256_ranks(r_match)
            eb_match = derive_eb_schedule(s_tilde, conf["eb_base"], lam, rho, c_scale=c_match)
            res_class = classify_res(abs(gap_match))
            
            r_entry = {
                "tier": tier,
                "method": m_name,
                "row_id": row_id,
                "target_b_live": tgt_b,
                "target_rank": tgt_r,
                "natural_ranks": r_nat,
                "natural_tot_rank": tot_nat,
                "natural_b_live": b_nat,
                "natural_gap_pct": round(gap_nat, 4),
                "natural_rank_sha": sha_nat,
                "natural_eb_vector": [round(x, 8) for x in eb_nat],
                "matched_c": round(c_match, 4),
                "matched_ranks": r_match,
                "matched_tot_rank": tot_match,
                "matched_b_live": b_match,
                "matched_gap_pct": round(gap_match, 4),
                "matched_rank_sha": sha_match,
                "matched_eb_vector": [round(x, 8) for x in eb_match],
                "resource_class": res_class,
                "note": f"FDP variant with rho={rho}, lambda={lam}"
            }
        rows_authority.append(r_entry)

# Write CORE_RESOURCE_AUTHORITY.csv and .json
auth_csv = FORMAL_DIR / "CORE_RESOURCE_AUTHORITY.csv"
auth_json = FORMAL_DIR / "CORE_RESOURCE_AUTHORITY.json"

with open(auth_json, "w") as f:
    json.dump(rows_authority, f, indent=2)

with open(auth_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "row_id", "tier", "method", "target_b_live", "target_rank",
        "nat_b_live", "nat_tot_rank", "nat_gap_pct", "nat_rank_sha",
        "matched_c", "matched_b_live", "matched_tot_rank", "matched_gap_pct", "matched_rank_sha", "resource_class"
    ])
    for r in rows_authority:
        writer.writerow([
            r["row_id"], r["tier"], r["method"], r["target_b_live"], r["target_rank"],
            r["natural_b_live"], r["natural_tot_rank"], r["natural_gap_pct"], r["natural_rank_sha"],
            r["matched_c"], r["matched_b_live"], r["matched_tot_rank"], r["matched_gap_pct"], r["matched_rank_sha"], r["resource_class"]
        ])

print(f"Written {auth_json} and {auth_csv} with 16 core rows.")
