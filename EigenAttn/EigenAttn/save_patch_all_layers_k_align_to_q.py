import os, json, torch
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

CMP_IN="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
OUT="/diskD/eigen_outputs_opt30b_r5712/final_model_QK_aligned"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"

low=json.load(open(LOW_JSON,"r"))
B=torch.load(BASIS_PT, map_location="cpu")

print("Load CMP_IN (cpu)...")
m=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP_IN, local_files_only=True, torch_dtype=torch.float16,
    device_map=None, low_rank_config=low, low_cpu_mem_usage=True
).cpu().eval()

n=m.config.num_hidden_layers
print("n_layers =", n)

with torch.no_grad():
    for i in range(n):
        Uq=B[f"base_model.model.model.decoder.layers.{i}.self_attn.q_proj_basis"].float()
        Uk=B[f"base_model.model.model.decoder.layers.{i}.self_attn.k_proj_basis"].float()
        T=(Uk.t()@Uq).float()  # [r,r]

        attn=m.model.decoder.layers[i].self_attn
        W=attn.k_proj_low.weight.detach().float()
        b=attn.k_proj_low.bias.detach().float()

        attn.k_proj_low.weight.copy_((T.t()@W).to(dtype=attn.k_proj_low.weight.dtype))
        attn.k_proj_low.bias.copy_((b@T).to(dtype=attn.k_proj_low.bias.dtype))

        if i%8==0: print("patched layer", i)

os.makedirs(OUT, exist_ok=True)
m.save_pretrained(OUT, safe_serialization=True)
print("saved:", OUT)
