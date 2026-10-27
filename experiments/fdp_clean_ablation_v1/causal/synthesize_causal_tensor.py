#!/usr/bin/env python3
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

OUT_DIR = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1/causal")
TIERS = ["light", "medium", "aggressive", "ultra"]
COMPONENTS = ["d1_z", "d2_s", "d3_v", "d4_a", "d5_h"]
WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0]

def minmax(x: List[float], eps: float = 1e-12) -> List[float]:
    lo, hi = min(x), max(x)
    span = hi - lo
    if span < eps:
        return [0.5 for _ in x]
    return [(t - lo) / span for t in x]

def main():
    f2 = OUT_DIR / "causal_worker_results_gpu2.json"
    f3 = OUT_DIR / "causal_worker_results_gpu3.json"
    
    if not f2.exists() or not f3.exists():
        print(f"Error: Missing worker results {f2} or {f3}")
        return
        
    with open(f2) as f:
        d2 = json.load(f)
    with open(f3) as f:
        d3 = json.load(f)
        
    all_data = {}
    all_data.update(d2)
    all_data.update(d3)
    
    # Construct 4D tensor G[j, k, l, e] shape [5, 32, 32, 4]
    G_4d = np.zeros((5, 32, 32, 4), dtype=np.float64)
    d_base_all = np.zeros((5, 32, 4), dtype=np.float64)
    
    for e_idx, tier in enumerate(TIERS):
        t_data = all_data[tier]
        G_tier = np.array(t_data["G_tier"], dtype=np.float64) # [5, 32, 32]
        d_base = np.array(t_data["d_base"], dtype=np.float64) # [5, 32]
        G_4d[:, :, :, e_idx] = G_tier
        d_base_all[:, :, e_idx] = d_base
        
    # Save PyTorch tensor
    tensor_path = OUT_DIR / "causal_response_tensor.pt"
    save_dict = {
        "tensor": torch.from_numpy(G_4d).float(),
        "components": COMPONENTS,
        "tiers": TIERS,
        "layers": list(range(32)),
        "delta": 0.015,
        "d_base": torch.from_numpy(d_base_all).float()
    }
    torch.save(save_dict, tensor_path)
    print(f"Saved causal response tensor to {tensor_path}")
    
    # Compute SHA256 of tensor
    with open(tensor_path, "rb") as f:
        t_sha = hashlib.sha256(f.read()).hexdigest()
        
    # Compute C, P, CHI, S per tier
    causal_rows = []
    prop_rows = []
    self_rows = []
    
    summary_analysis = {}
    
    for e_idx, tier in enumerate(TIERS):
        G_tier = G_4d[:, :, :, e_idx] # [5, 32, 32] -> G[j, k, l]
        d_base = d_base_all[:, :, e_idx] # [5, 32]
        
        # S_l: observed baseline fragility
        S_raw = [float(sum(WEIGHTS[j] * d_base[j, l] for j in range(5))) for l in range(32)]
        S_tilde = minmax(S_raw)
        
        # C_l: Total causal fragility = sum_j w_j mean_{k=l..31} max(0, G[j, k, l])
        C_raw = []
        for l in range(32):
            val = 0.0
            for j in range(5):
                pos_resp = np.maximum(0.0, G_tier[j, l:32, l])
                val += WEIGHTS[j] * float(np.mean(pos_resp))
            C_raw.append(val)
        C_tilde = minmax(C_raw)
        
        # CHI_l (R_l): Self EB susceptibility = sum_j w_j max(0, G[j, l, l])
        CHI_raw = []
        for l in range(32):
            val = 0.0
            for j in range(5):
                pos_self = max(0.0, float(G_tier[j, l, l]))
                val += WEIGHTS[j] * pos_self
            CHI_raw.append(val)
        CHI_tilde = minmax(CHI_raw)
        
        # P_l: Downstream propagation = sum_j w_j mean_{k>l} max(0, G[j, k, l])
        P_raw = []
        for l in range(32):
            if l < 31:
                val = 0.0
                for j in range(5):
                    pos_down = np.maximum(0.0, G_tier[j, (l+1):32, l])
                    val += WEIGHTS[j] * float(np.mean(pos_down))
                P_raw.append(val)
            else:
                P_raw.append(0.0)
        P_tilde = minmax(P_raw)
        
        summary_analysis[tier] = {
            "S_raw": S_raw,
            "S_tilde": S_tilde,
            "C_raw": C_raw,
            "C_tilde": C_tilde,
            "CHI_raw": CHI_raw,
            "CHI_tilde": CHI_tilde,
            "P_raw": P_raw,
            "P_tilde": P_tilde
        }
        
        for l in range(32):
            causal_rows.append({
                "tier": tier,
                "layer": l,
                "S_raw": f"{S_raw[l]:.8e}",
                "S_tilde": f"{S_tilde[l]:.6f}",
                "C_raw": f"{C_raw[l]:.8e}",
                "C_tilde": f"{C_tilde[l]:.6f}"
            })
            prop_rows.append({
                "tier": tier,
                "layer": l,
                "P_raw": f"{P_raw[l]:.8e}",
                "P_tilde": f"{P_tilde[l]:.6f}"
            })
            self_rows.append({
                "tier": tier,
                "layer": l,
                "CHI_raw": f"{CHI_raw[l]:.8e}",
                "CHI_tilde": f"{CHI_tilde[l]:.6f}"
            })
            
    # Write CSVs
    def write_csv(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")
        
    write_csv(OUT_DIR / "LAYER_CAUSAL_FRAGILITY.csv", causal_rows, ["tier", "layer", "S_raw", "S_tilde", "C_raw", "C_tilde"])
    write_csv(OUT_DIR / "LAYER_PROPAGATION_FRAGILITY.csv", prop_rows, ["tier", "layer", "P_raw", "P_tilde"])
    write_csv(OUT_DIR / "LAYER_SELF_SUSCEPTIBILITY.csv", self_rows, ["tier", "layer", "CHI_raw", "CHI_tilde"])
    
    # Write Manifest
    manifest = {
        "artifact": "causal_response_tensor",
        "pt_file": "causal/causal_response_tensor.pt",
        "sha256": t_sha,
        "shape": [5, 32, 32, 4],
        "dimension_meaning": ["functional_components (5)", "measured_layer_k (32)", "intervened_layer_l (32)", "operating_tier_e (4)"],
        "components": COMPONENTS,
        "tiers": TIERS,
        "intervention_details": {
            "method": "single_layer_eb_perturbation",
            "delta_eb": 0.015,
            "non_intervened_layers": "fixed_at_tier_base_uniform_eb",
            "calibration_input": "DEFAULT_PROMPTS (2 seqs, 512 tokens)",
            "causal_lower_triangular_property": "G[:, k, l] == 0 for k < l"
        },
        "layer0_emergent_analysis": {
            tier: {
                "S_0_tilde": summary_analysis[tier]["S_tilde"][0],
                "C_0_tilde": summary_analysis[tier]["C_tilde"][0],
                "CHI_0_tilde": summary_analysis[tier]["CHI_tilde"][0],
                "P_0_tilde": summary_analysis[tier]["P_tilde"][0]
            } for tier in TIERS
        }
    }
    
    with open(OUT_DIR / "Causal_Response_Tensor_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {OUT_DIR / 'Causal_Response_Tensor_manifest.json'}")
    
    print("\n=== LAYER 0 vs OTHER LAYERS CAUSAL SUMMARY ===")
    for tier in TIERS:
        sa = summary_analysis[tier]
        print(f"\n--- {tier.upper()} ---")
        print(f"Layer 0: S~={sa['S_tilde'][0]:.4f}, C~={sa['C_tilde'][0]:.4f}, Self(CHI)~={sa['CHI_tilde'][0]:.4f}, Downstream(P)~={sa['P_tilde'][0]:.4f}")
        print(f"Layer 1: S~={sa['S_tilde'][1]:.4f}, C~={sa['C_tilde'][1]:.4f}, Self(CHI)~={sa['CHI_tilde'][1]:.4f}, Downstream(P)~={sa['P_tilde'][1]:.4f}")
        print(f"Layer 2: S~={sa['S_tilde'][2]:.4f}, C~={sa['C_tilde'][2]:.4f}, Self(CHI)~={sa['CHI_tilde'][2]:.4f}, Downstream(P)~={sa['P_tilde'][2]:.4f}")
        print(f"Layer16: S~={sa['S_tilde'][16]:.4f}, C~={sa['C_tilde'][16]:.4f}, Self(CHI)~={sa['CHI_tilde'][16]:.4f}, Downstream(P)~={sa['P_tilde'][16]:.4f}")
        print(f"Layer31: S~={sa['S_tilde'][31]:.4f}, C~={sa['C_tilde'][31]:.4f}, Self(CHI)~={sa['CHI_tilde'][31]:.4f}, Downstream(P)~={sa['P_tilde'][31]:.4f}")

if __name__ == "__main__":
    main()
