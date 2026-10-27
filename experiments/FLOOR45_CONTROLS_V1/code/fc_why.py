#!/usr/bin/env python3
"""Read-only analysis: why does Spectral-tail beat FDP-EB under the 45% floor at Aggressive+ ?"""
import json, sys
from pathlib import Path
import numpy as np
SRC = Path("/diskF/jiaqizhang/FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1")
sys.path.insert(0, str(SRC / "code"))
import c45lib as L

P = json.loads(Path("/diskF/jiaqizhang/FLOOR45_CONTROLS_V1/PLAN.json").read_text())
for t in ["light", "medium", "aggressive"]:
    rp = json.loads((SRC / "tiers" / t / "RESOURCE_PLAN.json").read_text())
    U = np.array(rp["uniform_predicted_ranks"]); F = np.array(rp["fdp_predicted_ranks"]); T = np.array(P["tiers"][t]["tail45"]["ranks"])
    ebF = rp["fdp_eb_vector"]; ebF = [rp["base_e"]] + ebF[1:]
    _, sF = L.select(ebF, True); _, sU = L.select([rp["uniform_eb"]] * 32, True); _, sT = L.select(P["tiers"][t]["tail45"]["eps_by_layer"], True)
    qT = np.array(P["tiers"][t]["tail45"]["q"])
    # FDP q from step1 (same rule as derive_eb)
    sys.path.insert(0, str(L.EIGEN))
    from realloc_rank_from_fragility import fragility_vector, minmax
    pl = json.loads((SRC / "tiers" / t / "step1.json").read_text())["per_layer"]
    raw, _ = fragility_vector(pl, sorted(int(x) for x in pl), 1, 1, 0, 0, 1, False); qF = np.array(minmax(raw))
    rho = np.corrcoef(np.argsort(np.argsort(qF)), np.argsort(np.argsort(qT)))[0, 1]
    print(f"\n=== {t}  B: U {L.B_of(U.tolist())} F {L.B_of(F.tolist())} T {L.B_of(T.tolist())}   spearman(qF,qT)={rho:+.2f}")
    print("stops U", L.stopdist(sU), "F", L.stopdist(sF), "T", L.stopdist(sT))
    for name, X in (("F", F), ("T", T)):
        d = X - U
        print(f"{name}-U  QK total {d[:,0].sum():+5d}  V total {d[:,1].sum():+5d} | layers 0-7 QK {d[:8,0].sum():+5d} V {d[:8,1].sum():+5d} | 8-19 QK {d[8:20,0].sum():+5d} V {d[8:20,1].sum():+5d} | 20-31 QK {d[20:,0].sum():+5d} V {d[20:,1].sum():+5d}")
    print("layer  U(qk,v)    F(qk,v)    T(qk,v)   stopU stopF stopT   qF    qT")
    for l in range(32):
        if (U[l] != F[l]).any() or (U[l] != T[l]).any():
            print(f"{l:3d}  {U[l][0]:4d},{U[l][1]:4d}  {F[l][0]:4d},{F[l][1]:4d}  {T[l][0]:4d},{T[l][1]:4d}   {sU[l][:6]:6s} {sF[l][:6]:6s} {sT[l][:6]:6s} {qF[l]:.2f} {qT[l]:.2f}")
