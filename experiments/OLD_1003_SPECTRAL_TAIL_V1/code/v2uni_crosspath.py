#!/usr/bin/env python3
"""CPU-only, read-only: is the existing Severe V2 Uniform checkpoint (ranks 823e850e = the full-trace 10/3 Uniform point
at +0.217%) operator-equivalent to what the 10/3 materialization path (10/3 master slice, fill_arm2 rule) would produce?
Checks: (a) V2 basis == 10/3 master slice at the V2 ranks; (b) V3 operator-parity of every V2 attention factor against
references recomputed from the 10/3 master slice + dense weights (thresholds of OPERATOR_PARITY_V3_DTYPE_AWARE_FROZEN);
(c) every non-attention tensor bit-equal to dense Llama-3. Nothing is written except the result JSON."""
import hashlib, json
from pathlib import Path
import torch
from safetensors import safe_open

torch.set_num_threads(16)
V2 = Path("/diskF/jiaqizhang/TACL_LLAMA3_SPECTRAL_CROSS_TIER_V1/materialized_v2/severe_uniform")
FM = V2 / "final_model"
MASTER = json.loads(Path("/diskF/jiaqizhang/FDP_OFFICIAL_PROTOCOL_V1/records/old_master/result.json").read_text())["basis_pt"]
BASE = Path("/diskD/jiaqizhang/Llama-3")
OUT = Path("/diskF/jiaqizhang/OLD_1003_SPECTRAL_TAIL_V1/V2_UNIFORM_CROSSPATH_EQUIVALENCE.json")
U = 2.0 ** -8
ranks = json.loads((FM / "low_rank_config.json").read_text())
jsha = lambda o: hashlib.sha256(json.dumps(o, separators=(",", ":")).encode()).hexdigest()
res = {"schema": "V2_UNIFORM_VS_1003_PATH_EQUIVALENCE_V1", "checkpoint": str(FM), "rank_sha": jsha(ranks), "master": MASTER}
mb = torch.load(MASTER, map_location="cpu", weights_only=False)
vb = torch.load(V2 / "basis/severe_uniform_v2_perlayer_eb.pt", map_location="cpu", weights_only=False)
idx = json.loads((FM / "model.safetensors.index.json").read_text())["weight_map"]
bidx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
H = {}
def T(root, index, key):
    sh = index[key]
    if (root, sh) not in H: H[(root, sh)] = safe_open(str(root / sh), framework="pt", device="cpu")
    return H[(root, sh)].get_tensor(key)
def ordered(t):
    b = t.contiguous().view(torch.int16).to(torch.int32); return torch.where(b < 0, -(b & 0x7FFF), b)
def ulp(a, b): return int((ordered(a) - ordered(b)).abs().max())
def rel(a, b): return float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(1e-30))

basis_eq, per = [], []
ok3a, wu, wk, wv = True, 0, 0.0, 0.0
for li, (kq, v) in enumerate(ranks):
    bp = f"base_model.model.model.layers.{li}.self_attn"
    rk, rv = kq // 8, v // 8
    V = mb[f"{bp}.v_proj_basis"]; kc_m = V.shape[1] // 8
    Uq = mb[f"{bp}.q_proj_basis"][:, :rk].float()
    Uv = V[:, [h * kc_m + c for h in range(8) for c in range(rv)]].float()
    # (a) basis: the V2 builder basis truncated by the same rule
    V2v = vb[f"{bp}.v_proj_basis"]; kc2 = V2v.shape[1] // 8
    b_ok = {"q": torch.equal(vb[f"{bp}.q_proj_basis"][:, :rk].float(), Uq),
            "v": torch.equal(V2v[:, [h * kc2 + c for h in range(8) for c in range(rv)]].float(), Uv)}
    basis_eq.append(all(b_ok.values()))
    p = f"model.layers.{li}.self_attn."
    obs = {n: T(FM, idx, p + n + ".weight") for n in ("q_proj", "k_proj_low", "k_proj_up", "v_proj_low", "o_proj_up")}
    Wq, Wk, Wv, Wo = (T(BASE, bidx, p + n + ".weight") for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    dt = obs["k_proj_up"].dtype
    a3 = {"q_proj": torch.equal(obs["q_proj"], Wq), "k_proj_up": tuple(obs["k_proj_up"].shape) == tuple(Uq.shape) and torch.equal(obs["k_proj_up"], Uq.to(dt))}
    ok3a &= all(a3.values())
    rkl = torch.cat([(Uq.T @ Wk.float()[h * 128:(h + 1) * 128]) for h in range(8)]).to(obs["k_proj_low"].dtype)
    rvl = torch.cat([(Uv[h * 128:(h + 1) * 128, h * rv:(h + 1) * rv].T @ Wv.float()[h * 128:(h + 1) * 128]) for h in range(8)]).to(obs["v_proj_low"].dtype)
    rou = torch.cat([(Wo.float()[:, qh * 128:(qh + 1) * 128] @ Uv[(qh // 4) * 128:(qh // 4 + 1) * 128, (qh // 4) * rv:(qh // 4 + 1) * rv]) for qh in range(32)], 1).to(obs["o_proj_up"].dtype)
    shapes_ok = obs["k_proj_low"].shape == rkl.shape and obs["v_proj_low"].shape == rvl.shape and obs["o_proj_up"].shape == rou.shape
    u3 = {"k_proj_low": ulp(obs["k_proj_low"], rkl), "v_proj_low": ulp(obs["v_proj_low"], rvl), "o_proj_up": ulp(obs["o_proj_up"], rou)} if shapes_ok else None
    g = torch.Generator().manual_seed(20261005 + li); x = torch.randn((4, 4096), generator=g)
    kr = vr = 0.0
    if shapes_ok:
        for h in range(8):
            s = slice(h * rk, (h + 1) * rk)
            kr = max(kr, rel((x @ obs["k_proj_low"][s].float().T) @ obs["k_proj_up"].float().T, (x @ rkl[s].float().T) @ Uq.to(dt).float().T))
            vs = slice(h * rv, (h + 1) * rv)
            for gq in range(4):
                o = slice((h * 4 + gq) * rv, (h * 4 + gq + 1) * rv)
                vr = max(vr, rel((x @ obs["v_proj_low"][vs].float().T) @ obs["o_proj_up"][:, o].float().T, (x @ rvl[vs].float().T) @ rou[:, o].float().T))
        wu = max(wu, *u3.values())
    else:
        ok3a = False
    wk, wv = max(wk, kr), max(wv, vr)
    per.append({"layer": li, "basis_equal": b_ok, "gate3A": a3, "shapes_ok": shapes_ok, "gate3B_max_ulp": u3, "relL2_K": kr, "relL2_VO": vr})
    print(li, b_ok, a3, u3, f"{kr:.2e} {vr:.2e}", flush=True)
other = [k for k in idx if ".self_attn." not in k]
neq = [k for k in other if not torch.equal(T(FM, idx, k), T(BASE, bidx, k))]
res.update({"basis_equal_all_layers": all(basis_eq), "gate3A_all_exact": ok3a, "gate3B_worst_max_ulp": wu, "gate3C_worst_relL2_K": wk, "gate3C_worst_relL2_VO": wv,
            "non_attention_tensors_checked": len(other), "non_attention_tensors_not_equal_dense": neq, "per_layer": per})
res["OPERATOR_EQUIVALENT_TO_1003_PATH"] = ok3a and wu <= 1 and wk <= U and wv <= U and not neq
res["BIT_EXACT_TO_1003_PATH_REFERENCE"] = ok3a and wu == 0 and not neq
OUT.write_text(json.dumps(res, indent=1))
print({k: res[k] for k in ("basis_equal_all_layers", "gate3A_all_exact", "gate3B_worst_max_ulp", "gate3C_worst_relL2_K", "gate3C_worst_relL2_VO",
                           "non_attention_tensors_not_equal_dense", "OPERATOR_EQUIVALENT_TO_1003_PATH", "BIT_EXACT_TO_1003_PATH_REFERENCE")})
