import os, json, torch
from transformers import AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP_IN="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed"
CMP_OUT="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"

dtype=torch.float16  # 够用；计算时转 float32

low=json.load(open(LOW_JSON,"r"))
B=torch.load(BASIS_PT, map_location="cpu")

print("Load BASE (cpu)...")
base=AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True
).cpu().eval()

print("Load CMP (cpu)...")
cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP_IN, local_files_only=True, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True,
    low_rank_config=low
).cpu().eval()

n_layers=base.config.num_hidden_layers
print("n_layers =", n_layers)

@torch.no_grad()
def patch_one(layer_idx: int):
    b_attn=base.model.decoder.layers[layer_idx].self_attn
    c_attn=cmpm.model.decoder.layers[layer_idx].self_attn

    def getU(kind):
        key=f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{kind}_proj_basis"
        U=B[key].float()  # [7168,r]
        return U

    # ---- Q/K/V low weights (FIX HERE) ----
    for kind, proj_name in [("q","q_proj"), ("k","k_proj"), ("v","v_proj")]:
        U=getU(kind)  # [7168,r]
        W=getattr(b_attn, proj_name).weight.detach().float()  # [7168,7168]
        b=getattr(b_attn, proj_name).bias.detach().float()    # [7168]

        Wlow = U.t() @ W          # [r,7168]  ✅ correct
        blow = b @ U              # [r]       ✅ correct

        low_layer = getattr(c_attn, f"{proj_name}_low")
        low_layer.weight.copy_(Wlow.to(dtype=low_layer.weight.dtype))
        low_layer.bias.copy_(blow.to(dtype=low_layer.bias.dtype))

    # ---- optional: recompute out_proj_up (yours already OK; keep consistent) ----
    Uv=getU("v")  # [7168,r]
    Wo=b_attn.out_proj.weight.detach().float()
    bo=b_attn.out_proj.bias.detach().float()
    Wup = Wo @ Uv                # [7168,r]
    c_attn.out_proj_up.weight.copy_(Wup.to(dtype=c_attn.out_proj_up.weight.dtype))
    c_attn.out_proj_up.bias.copy_(bo.to(dtype=c_attn.out_proj_up.bias.dtype))

for i in range(n_layers):
    patch_one(i)
    if i % 8 == 0:
        print("patched layer", i)

os.makedirs(CMP_OUT, exist_ok=True)
cmpm.save_pretrained(CMP_OUT, safe_serialization=True)
print("saved:", CMP_OUT)
