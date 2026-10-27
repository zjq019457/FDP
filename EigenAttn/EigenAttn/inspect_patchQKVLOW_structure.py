import json, torch
from transformers import AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"

def cos_rel(a,b,eps=1e-12):
    a=a.flatten().float(); b=b.flatten().float()
    cos=float(torch.dot(a,b)/(a.norm()*b.norm()+eps))
    rel=float((a-b).norm()/(a.norm()+eps))
    return cos, rel

low=json.load(open(LOW_JSON,"r"))
B=torch.load(BASIS_PT, map_location="cpu")

print("Load BASE (cpu)...")
base=AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True, torch_dtype=torch.float16,
    device_map=None, low_cpu_mem_usage=True
).cpu().eval()

print("Load CMP (cpu)...")
cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=torch.float16,
    device_map=None, low_cpu_mem_usage=True, low_rank_config=low
).cpu().eval()

layers_to_check=[0,1,2,47]
print("\n[Module shapes]")
for i in layers_to_check:
    a=cmpm.model.decoder.layers[i].self_attn
    print(f"layer{i}:",
          "q_low", tuple(a.q_proj_low.weight.shape),
          "k_low", tuple(a.k_proj_low.weight.shape),
          "v_low", tuple(a.v_proj_low.weight.shape),
          "out_up", tuple(a.out_proj_up.weight.shape))

print("\n[Weight-formula consistency checks]")
for i in layers_to_check:
    b=base.model.decoder.layers[i].self_attn
    c=cmpm.model.decoder.layers[i].self_attn

    Uq=B[f"base_model.model.model.decoder.layers.{i}.self_attn.q_proj_basis"].float()
    Uk=B[f"base_model.model.model.decoder.layers.{i}.self_attn.k_proj_basis"].float()
    Uv=B[f"base_model.model.model.decoder.layers.{i}.self_attn.v_proj_basis"].float()

    # expected: Wlow = U^T W, blow = b @ U
    Wq_exp = Uq.t() @ b.q_proj.weight.detach().float()
    bq_exp = b.q_proj.bias.detach().float() @ Uq
    Wk_exp = Uk.t() @ b.k_proj.weight.detach().float()
    bk_exp = b.k_proj.bias.detach().float() @ Uk
    Wv_exp = Uv.t() @ b.v_proj.weight.detach().float()
    bv_exp = b.v_proj.bias.detach().float() @ Uv
    Wo_exp = b.out_proj.weight.detach().float() @ Uv
    bo_exp = b.out_proj.bias.detach().float()

    c1, r1 = cos_rel(c.q_proj_low.weight, Wq_exp)
    c2, r2 = cos_rel(c.q_proj_low.bias,   bq_exp)
    c3, r3 = cos_rel(c.k_proj_low.weight, Wk_exp)
    c4, r4 = cos_rel(c.k_proj_low.bias,   bk_exp)
    c5, r5 = cos_rel(c.v_proj_low.weight, Wv_exp)
    c6, r6 = cos_rel(c.v_proj_low.bias,   bv_exp)
    c7, r7 = cos_rel(c.out_proj_up.weight, Wo_exp)
    c8, r8 = cos_rel(c.out_proj_up.bias,   bo_exp)

    print(f"\nlayer{i}:")
    print(f"  q_low.weight  cos={c1:.6f} relL2={r1:.6f}")
    print(f"  q_low.bias    cos={c2:.6f} relL2={r2:.6f}")
    print(f"  k_low.weight  cos={c3:.6f} relL2={r3:.6f}")
    print(f"  k_low.bias    cos={c4:.6f} relL2={r4:.6f}")
    print(f"  v_low.weight  cos={c5:.6f} relL2={r5:.6f}")
    print(f"  v_low.bias    cos={c6:.6f} relL2={r6:.6f}")
    print(f"  out_up.weight cos={c7:.6f} relL2={r7:.6f}")
    print(f"  out_up.bias   cos={c8:.6f} relL2={r8:.6f}")

# Q/K basis overlap (layer0)
Uq0=B["base_model.model.model.decoder.layers.0.self_attn.q_proj_basis"].float()
Uk0=B["base_model.model.model.decoder.layers.0.self_attn.k_proj_basis"].float()
G=Uq0.t()@Uk0
overlap=float((G.pow(2).sum()/Uq0.shape[1]).item())
print("\n[layer0 Uq/Uk subspace overlap] (1 is ideal):", overlap)
