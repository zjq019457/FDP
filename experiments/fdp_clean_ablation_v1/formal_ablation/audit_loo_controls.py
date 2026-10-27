#!/usr/bin/env python3
import sys, os, json, math, hashlib, random
from pathlib import Path

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
FORMAL_DIR = WORKSPACE / "formal_ablation"
PER_WIN_DIR = FORMAL_DIR / "CORE_FORMAL_PER_WINDOW"

sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/EigenAttn")
from realloc_rank_from_fragility import fragility_vector, minmax
import torch

TIER_TARGETS = {
    "light": {
        "target_b_live": 218988544,
        "target_rank": 53464,
        "bytes_per_r": 4096,
        "eb_base": 0.045,
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb045/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb045.json",
    },
    "medium": {
        "target_b_live": 198377472,
        "target_rank": 48432,
        "bytes_per_r": 4096,
        "eb_base": 0.060,
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb060.json",
    },
    "aggressive": {
        "target_b_live": 158105600,
        "target_rank": 38600,
        "bytes_per_r": 4096,
        "eb_base": 0.090,
        "report": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json",
        "basis": "/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb090/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt",
        "step1": "/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb090.json",
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

def derive_eb_schedule(s_tilde, eb_base, lambda_eb=1.2, rho=0.75, c_scale=1.0):
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

# Load medium step1 for static transfer
with open(TIER_TARGETS["medium"]["step1"]) as f:
    s060_raw, _ = fragility_vector(json.load(f)["per_layer"], list(range(32)), 1, 1, 1, 1, 1, False)
s060_tilde = minmax(s060_raw)

audit_records = []
unique_states = {}

for tier in ["light", "medium", "aggressive"]:
    spec = TIER_TARGETS[tier]
    with open(spec["step1"]) as f:
        H_data = json.load(f)["per_layer"]
    with open(spec["report"]) as f:
        rep = json.load(f)
    traces = [l["runs"][0]["threshold_trace"] for l in rep["layers"]]
    basis_dict = torch.load(spec["basis"], map_location="cpu")
    
    eb_base = spec["eb_base"]
    tgt_r = spec["target_rank"]
    b_tgt = spec["target_b_live"]
    bytes_per_r = spec["bytes_per_r"]
    
    s_raw, _ = fragility_vector(H_data, list(range(32)), 1.0, 1.0, 1.0, 1.0, 1.0, False)
    s_tilde_base = minmax(s_raw)
    
    tier_configs = {}
    
    # 1. LOO
    loo_weights = {
        "LOO_MINUS_D1": (0.0, 1.0, 1.0, 1.0, 1.0),
        "LOO_MINUS_D2": (1.0, 0.0, 1.0, 1.0, 1.0),
        "LOO_MINUS_D3": (1.0, 1.0, 0.0, 1.0, 1.0),
        "LOO_MINUS_D4": (1.0, 1.0, 1.0, 0.0, 1.0),
        "LOO_MINUS_D5": (1.0, 1.0, 1.0, 1.0, 0.0),
    }
    for k, w in loo_weights.items():
        s_loo, _ = fragility_vector(H_data, list(range(32)), *w, False)
        tier_configs[k] = {"s_tilde": minmax(s_loo), "group": "LOO"}
        
    # 2. Controls
    for s_idx in [0, 1, 2]:
        rng = random.Random(s_idx + 42)
        perm = list(s_tilde_base)
        rng.shuffle(perm)
        tier_configs[f"RANDOM_SEED{s_idx}"] = {"s_tilde": perm, "group": "CONTROL"}
        
    spec_tilde = minmax([0.5 + (l + 1)/32.0 for l in range(32)])
    tier_configs["SPECTRAL_CONTROL"] = {"s_tilde": spec_tilde, "group": "CONTROL"}
    
    rec_tilde = minmax([1.25 - (l / 31.0) * 0.5 for l in range(32)])
    tier_configs["RECONSTRUCTION_CONTROL"] = {"s_tilde": rec_tilde, "group": "CONTROL"}
    
    tier_configs["STATIC_S060_TRANSFER"] = {"s_tilde": s060_tilde, "group": "CONTROL"}
    
    for cfg_name, cfg_info in tier_configs.items():
        st = cfg_info["s_tilde"]
        eb_nat = derive_eb_schedule(st, eb_base, lambda_eb=1.2, rho=0.75, c_scale=1.0)
        r_nat = []
        for li in range(32):
            rk, rv = oracle_lookup(traces[li], eb_nat[li])
            r_nat.append([rk, rv])
        r_nat = clip_ranks_to_basis(r_nat, basis_dict)
        tot_r = sum(sum(r) for r in r_nat)
        b_live = tot_r * bytes_per_r
        gap_pct = abs(tot_r - tgt_r) / tgt_r * 100.0
        sha = sha256_ranks(r_nat)
        
        is_done = (PER_WIN_DIR / sha / "STATE_EVAL_DONE.json").exists()
        
        entry = {
            "tier": tier,
            "group": cfg_info["group"],
            "config_name": cfg_name,
            "tot_rank": tot_r,
            "b_live": b_live,
            "gap_pct": round(gap_pct, 4),
            "sha": sha,
            "ranks": r_nat,
            "basis_pt": spec["basis"],
            "is_done": is_done,
        }
        audit_records.append(entry)
        
        if sha not in unique_states:
            unique_states[sha] = entry

with open(FORMAL_DIR / "LOO_CONTROLS_AUTHORITY.json", "w") as f:
    json.dump(audit_records, f, indent=2)

print("="*65)
print(f"AUDIT SUMMARY: LOO + CONTROLS (Light, Medium, Aggressive)")
print("="*65)
print(f"Total rows: {len(audit_records)} (3 tiers * 11 configs = 33 rows)")
print(f"Unique physical SHAs: {len(unique_states)}")
done_count = sum(1 for s, e in unique_states.items() if e["is_done"])
need_count = len(unique_states) - done_count
print(f"Already EVALUATED & DONE: {done_count}")
print(f"FRESH EVALUATION NEEDED: {need_count}")
print("-"*65)
i = 1
for s, e in unique_states.items():
    if not e["is_done"]:
        print(f"[{i:>2}/{need_count}] {e['tier']:<11} {e['group']:<8} {e['config_name']:<22} Rank={e['tot_rank']:<6} Gap={e['gap_pct']:.2f}% SHA={s[:12]}")
        i += 1
