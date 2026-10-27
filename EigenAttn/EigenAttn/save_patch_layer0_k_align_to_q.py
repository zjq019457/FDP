import os, json, torch
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

CMP_IN="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
OUT="/diskD/eigen_outputs_opt30b_r5712/final_model_patch_layer0_KalignQ"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"
L=0

low=json.load(open(LOW_JSON,"r"))
B=torch.load(BASIS_PT, map_location="cpu")

print("Load CMP_IN (cpu)...")
m=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP_IN, local_files_only=True, torch_dtype=torch.float16,
    device_map=None, low_rank_config=low, low_cpu_mem_usage=True
).cpu().eval()

Uq=B[f"base_model.model.model.decoder.layers.{L}.self_attn.q_proj_basis"].float()
Uk=B[f"base_model.model.model.decoder.layers.{L}.self_attn.k_proj_basis"].float()
T=(Uk.t()@Uq).float()  # [r,r]

attn=m.model.decoder.layers[L].self_attn
W=attn.k_proj_low.weight.detach().float()  # [r,7168]
b=attn.k_proj_low.bias.detach().float()    # [r]

# k_low_new = k_low_old @ T  => W_new = T^T @ W_old, b_new = b_old @ T
Wn = (T.t() @ W)
bn = (b @ T)

with torch.no_grad():
    attn.k_proj_low.weight.copy_(Wn.to(dtype=attn.k_proj_low.weight.dtype))
    attn.k_proj_low.bias.copy_(bn.to(dtype=attn.k_proj_low.bias.dtype))

os.makedirs(OUT, exist_ok=True)
m.save_pretrained(OUT, safe_serialization=True)
print("saved:", OUT)
