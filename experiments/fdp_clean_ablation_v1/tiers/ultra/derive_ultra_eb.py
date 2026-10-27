#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

# Add EigenAttn to path
sys.path.insert(0, '/home/bai-project/jiaqizhang/EigenAttn/EigenAttn')
from realloc_rank_from_fragility import fragility_vector, minmax

def derive_ultra_eb(
    step1_json: str,
    out_json: str,
    eb_base: float = 0.120,
    lambda_eb: float = 1.2,
    layer0_override: bool = True
):
    print(f"Loading step1 from {step1_json}...")
    with open(step1_json) as f:
        data = json.load(f)
        
    per_layer = data["per_layer"]
    layers = sorted(int(k) for k in per_layer.keys())
    
    s_raw, breakdown = fragility_vector(per_layer, layers, 1.0, 1.0, 1.0, 1.0, 1.0, False)
    s_tilde = minmax(s_raw)
    
    eb_inv = [eb_base / (1.0 + lambda_eb * st) for st in s_tilde]
    m = sum(eb_inv) / len(eb_inv)
    scale = eb_base / m if m > 0 else 1.0
    eb_final = [e * scale for e in eb_inv]
    
    if layer0_override:
        eb_final[0] = float(eb_base)
        
    res = {
        "meta": {
            "tier": "ultra",
            "step1_source": step1_json,
            "eb_base": eb_base,
            "lambda_eb": lambda_eb,
            "normalize": "mean",
            "weights": {"wz": 1.0, "ws": 1.0, "wv": 1.0, "wa": 1.0, "wh": 1.0},
            "layer0_override": layer0_override,
            "mean_eb": sum(eb_final) / len(eb_final),
            "min_eb": min(eb_final),
            "max_eb": max(eb_final)
        },
        "error_budget_by_layer": eb_final,
        "fragility_raw": s_raw,
        "fragility_s_tilde": s_tilde
    }
    
    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(res, f, indent=2)
        
    print(f"Saved Ultra EB schedule to {out_json}!")
    print(f"  eb[0] = {eb_final[0]:.6f}")
    print(f"  eb[:5] = {[round(x, 6) for x in eb_final[:5]]}")
    print(f"  range = [{min(eb_final):.6f}, {max(eb_final):.6f}], mean = {sum(eb_final)/len(eb_final):.6f}")
    return res

if __name__ == "__main__":
    derive_ultra_eb(
        step1_json="/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb120.json",
        out_json="/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/per_layer_eb_ultra_120.json",
        eb_base=0.120,
        lambda_eb=1.2,
        layer0_override=True
    )
