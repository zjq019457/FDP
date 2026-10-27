import os, json, torch
from transformers import AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP_IN="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
OUT="/diskD/eigen_outputs_opt30b_r5712/final_model_KeqQ_alllayers"   # 新模型目录
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"

dtype=torch.float16
low=json.load(open(LOW_JSON,"r"))
B=torch.load(BASIS_PT, map_location="cpu")

print("Load BASE (cpu)...")
base=AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True
).cpu().eval()

print("Load CMP_IN (cpu)...")
m=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP_IN, local_files_only=True, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True, low_rank_config=low
).cpu().eval()

n=m.config.num_hidden_layers
print("n_layers =", n)

with torch.no_grad():
    for i in range(n):
        Uq = B[f"base_model.model.model.decoder.layers.{i}.self_attn.q_proj_basis"].float()  # [7168,r]

        b_attn = base.model.decoder.layers[i].self_attn
        c_attn = m.model.decoder.layers[i].self_attn

        # rebuild ONLY k_proj_low using Uq (KeqQ)
        Wk = b_attn.k_proj.weight.detach().float()
        bk = b_attn.k_proj.bias.detach().float()
        Wk_low = Uq.t() @ Wk          # [r,7168]
        bk_low = bk @ Uq              # [r]

        c_attn.k_proj_low.weight.copy_(Wk_low.to(dtype=c_attn.k_proj_low.weight.dtype))
        c_attn.k_proj_low.bias.copy_(bk_low.to(dtype=c_attn.k_proj_low.bias.dtype))

        if i % 8 == 0:
            print("patched layer", i)

os.makedirs(OUT, exist_ok=True)
m.save_pretrained(OUT, safe_serialization=True)
print("saved:", OUT)
