
import json
import math
from pathlib import Path

OUT_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection")

f2_path = OUT_DIR / "LOO_EVAL_PROGRESS_GPU2.json"
f3_path = OUT_DIR / "LOO_EVAL_PROGRESS_GPU3.json"

if not f2_path.exists() or not f3_path.exists():
    print("Progress files not ready")
    sys.exit(1)

with open(f2_path) as f:
    d2 = json.load(f)
with open(f3_path) as f:
    d3 = json.load(f)

all_data = {}
all_data.update(d2)
all_data.update(d3)

TIERS = ["light", "medium", "aggressive", "ultra"]
CIDS = [
    ("minus_d1_z", "Drop d1 (delta_z)"),
    ("minus_d2_s", "Drop d2 (delta_score)"),
    ("minus_d3_v", "Drop d3 (delta_v)"),
    ("minus_d4_a", "Drop d4 (attn_entropy)"),
    ("minus_d5_h", "Drop d5 (attn_energy)")
]

print("================================================================================")
print("FAMILY 2: COMPONENT LEAVE-ONE-OUT (LOO) ABLATION SYNTHESIS")
print("Baseline: F_CURRENT (rho=0.75, lambda=1.2, unit weights)")
print("================================================================================")

# 1. Per-tier detailed table
for tier in TIERS:
    tdata = all_data.get(tier, {})
    base = tdata.get("baseline", {})
    base_j = base.get("J_dev", 0.0)
    print(f"\n--- TIER: {tier.upper()} (Baseline J_dev = {base_j:.6f}) ---")
    print(f"{'Candidate':12s} | {'Rank':6s} | {'Gap %':7s} | {'DEV Wiki NLL':12s} | {'DEV C4 NLL':10s} | {'J_dev':10s} | {'Delta J':10s} | {'95% CI':22s} | {'p-val':6s} | {'Interpretation'}")
    print("-" * 115)
    cands = tdata.get("candidates", {})
    for cid, desc in CIDS:
        c = cands.get(cid, {})
        rk = c.get("total_rank", 0)
        gap = c.get("gap_pct", 0.0)
        wnll = c.get("dev_wiki_nll", 0.0)
        cnll = c.get("dev_c4_nll", 0.0)
        jdev = c.get("J_dev", 0.0)
        dj = c.get("delta_J", 0.0)
        ci_l = c.get("ci_low", 0.0)
        ci_h = c.get("ci_high", 0.0)
        pval = c.get("p_value", 1.0)
        
        ci_str = f"[{ci_l:+.4f}, {ci_h:+.4f}]"
        
        if dj > 0.001:
            interp = "BENEFICIAL (Drop causes degradation)"
        elif dj < -0.001:
            interp = "HARMFUL (Drop improves quality)"
        else:
            interp = "NEUTRAL / REDUNDANT"
            
        print(f"{cid:12s} | {rk:6d} | {gap:+6.2f}% | {wnll:12.4f} | {cnll:10.4f} | {jdev:10.6f} | {dj:+10.6f} | {ci_str:22s} | {pval:6.4f} | {interp}")

# 2. Cross-tier J_STRONG synthesis
print("\n================================================================================")
print("CROSS-TIER J_STRONG SYNTHESIS & LIGHT GUARD")
print("================================================================================")
base_m = all_data["medium"]["baseline"]["J_dev"]
base_a = all_data["aggressive"]["baseline"]["J_dev"]
base_u = all_data["ultra"]["baseline"]["J_dev"]
base_l = all_data["light"]["baseline"]["J_dev"]
base_strong = (base_m + base_a + base_u) / 3.0

print(f"BASELINE F_CURRENT: J_STRONG = {base_strong:.6f} | Light J_L = {base_l:.6f}")
print("-" * 90)
print(f"{'Ablation':12s} | {'Light J_L':10s} | {'Delta J_L':10s} | {'J_MEDIUM':10s} | {'J_AGGR':10s} | {'J_ULTRA':10s} | {'J_STRONG':10s} | {'Delta_STRONG':12s} | {'Guard Status'}")
print("-" * 90)

synthesis_rows = []
for cid, desc in CIDS:
    jl = all_data["light"]["candidates"][cid]["J_dev"]
    jm = all_data["medium"]["candidates"][cid]["J_dev"]
    ja = all_data["aggressive"]["candidates"][cid]["J_dev"]
    ju = all_data["ultra"]["candidates"][cid]["J_dev"]
    
    jstrong = (jm + ja + ju) / 3.0
    dstrong = jstrong - base_strong
    dl = jl - base_l
    
    guard_pass = (dl <= 0.0030)
    guard_str = "PASSED" if guard_pass else "FAILED (Regressed)"
    
    print(f"{cid:12s} | {jl:10.6f} | {dl:+10.6f} | {jm:10.6f} | {ja:10.6f} | {ju:10.6f} | {jstrong:10.6f} | {dstrong:+12.6f} | {guard_str}")
    
    synthesis_rows.append({
        "cid": cid,
        "desc": desc,
        "j_light": jl,
        "delta_light": dl,
        "guard_pass": guard_pass,
        "j_medium": jm,
        "j_aggressive": ja,
        "j_ultra": ju,
        "j_strong": jstrong,
        "delta_strong": dstrong
    })

# Save synthesis JSON
master_loo = {
    "baseline_f_current": {
        "j_light": base_l,
        "j_medium": base_m,
        "j_aggressive": base_a,
        "j_ultra": base_u,
        "j_strong": base_strong
    },
    "loo_candidates": synthesis_rows,
    "detailed_per_tier": all_data
}

with open(OUT_DIR / "COMPONENT_LOO_SYNTHESIS.json", "w") as f:
    json.dump(master_loo, f, indent=2)

print("\nSynthesis saved to dev_selection/COMPONENT_LOO_SYNTHESIS.json")
