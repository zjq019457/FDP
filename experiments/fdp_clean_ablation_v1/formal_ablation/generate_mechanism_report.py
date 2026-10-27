#!/usr/bin/env python3
"""
FORMAL MECHANISM REPORT & CAUSAL VISUALIZATION GENERATOR
=========================================================
Generates:
1. 32x32 Source -> Downstream Causal Response Heatmap (G_{k <- l})
2. Layer-wise Profile comparison: Observed Local Fragility S_l vs Downstream Propagation P_l
3. Layer 0 & Early layer rank attribution audit
Uses exclusively the frozen causal response tensor from DEV.
Zero modification to formulas or evaluation metrics.
"""
import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKSPACE = Path("/diskF/jiaqizhang/fdp_clean_ablation_v1")
CAUSAL_DIR = WORKSPACE / "causal"
FORMAL_DIR = WORKSPACE / "formal_ablation"
MECH_DIR = FORMAL_DIR / "mechanism_report"
MECH_DIR.mkdir(parents=True, exist_ok=True)

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"[{utc()}] Loading frozen causal response tensor from {CAUSAL_DIR / 'causal_response_tensor.pt'}...")
tensor_path = CAUSAL_DIR / "causal_response_tensor.pt"
loaded_dict = torch.load(tensor_path, map_location="cpu")
G_4d = loaded_dict["tensor"].numpy() # [5, 32, 32, 4]

TIERS = ["light", "medium", "aggressive", "ultra"]

# 1. Generate 32x32 heatmap across tiers
for t_idx, tier in enumerate(TIERS):
    G_mat = np.mean(np.maximum(0.0, G_4d[:, :, :, t_idx]), axis=0) # [32, 32]
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    cax = ax.imshow(G_mat, cmap="viridis", aspect="auto", origin="upper")
    fig.colorbar(cax, label="Mean Causal Response $G_{k \leftarrow l}$")
    ax.set_title(f"Source-to-Downstream Causal Propagation Matrix ({tier.capitalize()} Tier)", fontsize=13, pad=12)
    ax.set_xlabel("Source Layer $l$ (Intervened)", fontsize=11)
    ax.set_ylabel("Downstream Layer $k$ (Measured)", fontsize=11)
    ax.set_xticks(range(0, 32, 4))
    ax.set_yticks(range(0, 32, 4))
    plt.tight_layout()
    out_img = MECH_DIR / f"causal_heatmap_{tier}.png"
    plt.savefig(out_img)
    plt.close()
    print(f"[{utc()}] Saved {out_img}")

# 2. Generate Dual-Axis Layer-Wise Profile: S_l vs P_l
profile_data = {}
for t_idx, tier in enumerate(TIERS):
    G_tier = G_4d[:, :, :, t_idx]
    S_raw = [float(np.mean(np.maximum(0.0, G_tier[:, l, l]))) for l in range(32)]
    P_raw = [float(np.mean(np.maximum(0.0, G_tier[:, (l+1):32, l]))) if l < 31 else 0.0 for l in range(32)]
    
    s_ranks = np.argsort(np.argsort(S_raw)) # 0 = lowest, 31 = highest
    p_ranks = np.argsort(np.argsort(P_raw))
    
    profile_data[tier] = {
        "S_raw": S_raw,
        "P_raw": P_raw,
        "S_ranks": s_ranks.tolist(),
        "P_ranks": p_ranks.tolist(),
        "L0_S_rank": int(s_ranks[0]),
        "L0_P_rank": int(p_ranks[0]),
    }
    
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    color1 = 'tab:blue'
    ax1.set_xlabel("Layer Index $l$", fontsize=11)
    ax1.set_ylabel("Observed Local Susceptibility $S_l$ (Self)", color=color1, fontsize=11)
    l1 = ax1.plot(range(32), S_raw, color=color1, marker='o', linewidth=2, label="Local Fragility $S_l$")
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("Downstream Forward Propagation $P_l$", color=color2, fontsize=11)
    l2 = ax2.plot(range(32), P_raw, color=color2, marker='s', linestyle='--', linewidth=2, label="Downstream Cascade $P_l$")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
    annot_text = f"Layer 0\nLow S_0 (Rank {s_ranks[0]+1}/32)\nHigh P_0 (Rank {p_ranks[0]+1}/32)"
    ax1.annotate(annot_text, 
                 xy=(0, S_raw[0]), xytext=(2, max(S_raw)*0.7 if max(S_raw)>0 else 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
                 
    plt.title(f"Mechanistic Dissociation: Local Fragility $S_l$ vs. Forward Propagation $P_l$ ({tier.capitalize()})", fontsize=12, pad=12)
    plt.tight_layout()
    out_profile = MECH_DIR / f"layer_profile_{tier}.png"
    plt.savefig(out_profile)
    plt.close()
    print(f"[{utc()}] Saved {out_profile}")

# Save JSON data
with open(MECH_DIR / "CAUSAL_MECHANISM_DATA.json", "w") as f:
    json.dump(profile_data, f, indent=2)

print(f"[{utc()}] Causal mechanism visualization completed successfully!")
