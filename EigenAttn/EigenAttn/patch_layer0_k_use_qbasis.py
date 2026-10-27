import os, json, torch
from transformers import AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP_IN="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
OUT="/diskD/eigen_outputs_opt30b_r5712/final_model_patch_layer0_KeqQ"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"
L=0

low=json.load(open(LOW_JSON,"r"))
B=torch.load(BASIS_PT, map_location="cpu")

Uq = B[f"base_model.model.model.decoder.layers.{L}.self_attn.q_proj_basis"].float()  # [7168,r]

base=AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.float16, device_map=None, low_cpu_mem_usage=True).cpu().eval()
cmpm=OPTForCausalLM_EigenAttn.from_pretrained(CMP_IN, local_files_only=True, torch_dtype=torch.float16, device_map=None, low_cpu_mem_usage=True, low_rank_config=low).cpu().eval()

b_attn=base.model.decoder.layers[L].self_attn
c_attn=cmpm.model.decoder.layers[L].self_attn

# rebuild ONLY k_proj_low using Uq
Wk=b_attn.k_proj.weight.detach().float()
bk=b_attn.k_proj.bias.detach().float()
Wk_low = Uq.t() @ Wk
bk_low = bk @ Uq

with torch.no_grad():
    c_attn.k_proj_low.weight.copy_(Wk_low.to(dtype=c_attn.k_proj_low.weight.dtype))
    c_attn.k_proj_low.bias.copy_(bk_low.to(dtype=c_attn.k_proj_low.bias.dtype))

os.makedirs(OUT, exist_ok=True)
cmpm.save_pretrained(OUT, safe_serialization=True)
print("saved:", OUT)
