import json
import torch

with open('/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb_report.json') as f:
    rep = json.load(f)

t0 = rep["layers"][0]["runs"][0]["threshold_trace"]
for s in t0:
    print(f"th={s['threshold']:.3f}, err={s['err']:.4f}, rkq={s['rank_kq']}, rv={s['rank_v']}")

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

print("eb=0.045 ->", oracle_lookup(t0, 0.045))
print("eb=0.060 ->", oracle_lookup(t0, 0.060))
print("eb=0.075 ->", oracle_lookup(t0, 0.075))

basis = torch.load('/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt', map_location='cpu')
qk = "base_model.model.model.layers.0.self_attn.q_proj_basis"
vk = "base_model.model.model.layers.0.self_attn.v_proj_basis"
print(f"qk shape: {basis[qk].shape}, vk shape: {basis[vk].shape}")
