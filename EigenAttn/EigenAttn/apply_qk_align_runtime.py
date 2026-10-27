import json, torch
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"

low=json.load(open(LOW_JSON,"r"))
B=torch.load(BASIS_PT, map_location="cpu")

model=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=torch.bfloat16, device_map=None, low_rank_config=low, low_cpu_mem_usage=True
).to("cuda").eval()

n=model.config.num_hidden_layers
with torch.no_grad():
    for i in range(n):
        Uq=B[f"base_model.model.model.decoder.layers.{i}.self_attn.q_proj_basis"].float()
        Uk=B[f"base_model.model.model.decoder.layers.{i}.self_attn.k_proj_basis"].float()
        T=(Uk.t()@Uq).float()

        attn=model.model.decoder.layers[i].self_attn
        W=attn.k_proj_low.weight.detach().float().cpu()
        b=attn.k_proj_low.bias.detach().float().cpu()
        Wn=(T.t()@W)
        bn=(b@T)
        attn.k_proj_low.weight.copy_(Wn.to(device="cuda", dtype=attn.k_proj_low.weight.dtype))
        attn.k_proj_low.bias.copy_(bn.to(device="cuda", dtype=attn.k_proj_low.bias.dtype))
        if i%8==0: print("patched layer", i)

print("runtime patch done. Now run your eval in the SAME process.")
