import os, json, torch
from transformers import AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE_PATH="/diskE/opt-30b"
CMP_PATH="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"  # ✅ patched
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"

LAYER=0
DEVICE_BASE="cuda:0"
DEVICE_CMP ="cuda:1"
DTYPE=torch.bfloat16
BATCH=16
SEED=0

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def cos_rel(a,b,eps=1e-12):
    a=a.flatten(); b=b.flatten()
    cos = torch.dot(a,b) / (a.norm()*b.norm() + eps)
    rel = (a-b).norm() / (a.norm() + eps)
    return float(cos), float(rel)

print("Load low_rank_config...")
low=json.load(open(LOW_JSON,"r"))

print("Load eigen_basis.pt...")
B=torch.load(BASIS_PT, map_location="cpu")
k=f"base_model.model.model.decoder.layers.{LAYER}.self_attn.v_proj_basis"
Uv=B[k].float()  # [7168,5712]
print("Uv key:", k)
print("Uv shape:", tuple(Uv.shape), "dtype:", Uv.dtype)

UtU=Uv.t()@Uv
orth_err=(UtU-torch.eye(Uv.shape[1])).abs().max().item()
print("orth_err_max |U^T U - I|:", orth_err)

Uv_base=Uv.to(DEVICE_BASE)
Uv_cmp =Uv.to(DEVICE_CMP)

print("Load BASE...")
base=AutoModelForCausalLM.from_pretrained(
    BASE_PATH, local_files_only=True, torch_dtype=DTYPE,
    device_map=None, low_cpu_mem_usage=True
).to(DEVICE_BASE).eval()

print("Load CMP (patched)...")
cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP_PATH, local_files_only=True, torch_dtype=DTYPE,
    device_map=None, low_cpu_mem_usage=True, low_rank_config=low
).to(DEVICE_CMP).eval()

b_attn=base.model.decoder.layers[LAYER].self_attn
c_attn=cmpm.model.decoder.layers[LAYER].self_attn

Wv_full=b_attn.v_proj.weight.detach().to(torch.float32)
bv_full=b_attn.v_proj.bias.detach().to(torch.float32)
Wo_full=b_attn.out_proj.weight.detach().to(torch.float32)
bo_full=b_attn.out_proj.bias.detach().to(torch.float32)

Wv_low_ckpt=c_attn.v_proj_low.weight.detach().to(torch.float32)
bv_low_ckpt=c_attn.v_proj_low.bias.detach().to(torch.float32)
Wup_ckpt   =c_attn.out_proj_up.weight.detach().to(torch.float32)
bup_ckpt   =c_attn.out_proj_up.bias.detach().to(torch.float32)

torch.manual_seed(SEED)
x=torch.randn(BATCH, 7168, device=DEVICE_BASE, dtype=torch.float32)

v_full = x @ Wv_full.t() + bv_full

v_low_expected = (v_full @ Uv_base).to(DEVICE_CMP)
x_cmp = x.to(DEVICE_CMP)
v_low_ckpt = x_cmp @ Wv_low_ckpt.t() + bv_low_ckpt
cos_vlow, rel_vlow = cos_rel(v_low_expected, v_low_ckpt)

print("\n[v_proj_low check] (should be cos~1, relL2~0)")
print(f"v_low_expected vs v_low_ckpt: cos={cos_vlow:.6f} relL2={rel_vlow:.6f}")

v_low_base = (v_full @ Uv_base)
out_expected = (v_low_base @ Uv_base.t()) @ Wo_full.t() + bo_full
out_ckpt = (v_low_base.to(DEVICE_CMP) @ Wup_ckpt.t()) + bup_ckpt
cos_outp, rel_outp = cos_rel(out_expected.to(DEVICE_CMP), out_ckpt)

print("\n[out_proj_up check] (should remain cos~1)")
print(f"out_expected vs out_ckpt: cos={cos_outp:.6f} relL2={rel_outp:.6f}")
