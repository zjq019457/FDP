import os, json, torch
from transformers import AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

# ----------------
# config
# ----------------
BASE_PATH="/diskE/opt-30b"
CMP_PATH="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"

# TODO: 替换成你 layer0 的 Uv 文件路径
UV_PATH="REPLACE_WITH_LAYER0_V_BASIS_PATH.pt"

LAYER=0
DEVICE_BASE="cuda:0"
DEVICE_CMP ="cuda:1"
DTYPE=torch.bfloat16
BATCH=32
SEED=0

# 强制 eager（避免 sdp/flash 干扰）
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def cos_rel(a,b,eps=1e-12):
    a=a.flatten()
    b=b.flatten()
    cos = torch.dot(a,b) / (a.norm()*b.norm() + eps)
    rel = (a-b).norm() / (a.norm() + eps)
    return float(cos), float(rel)

def load_Uv(path):
    Uv=torch.load(path, map_location="cpu")
    if isinstance(Uv, dict):
        for k in ["Uv","U","basis","v","value","V"]:
            if k in Uv:
                Uv=Uv[k]; break
    assert torch.is_tensor(Uv), f"Uv not tensor, got {type(Uv)}"
    Uv=Uv.float()
    if Uv.shape[0] != 7168 and Uv.shape[1] == 7168:
        Uv=Uv.t()
    return Uv

print("Load low_rank_config...")
low=json.load(open(LOW_JSON,"r"))

print("Load Uv...")
Uv=load_Uv(UV_PATH)
d, r = Uv.shape
print(f"Uv shape = {Uv.shape}")
assert d==7168, "Uv first dim must be 7168 for OPT-30B"
print("Check orthogonality (columns): ||U^T U - I||_max ...")
UtU = (Uv.t() @ Uv)
I = torch.eye(r)
orth_err = (UtU - I).abs().max().item()
print("orth_err_max =", orth_err)

# ----------------
# Load BASE and grab layer weights
# ----------------
print("Load BASE model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_PATH, local_files_only=True, torch_dtype=DTYPE,
    device_map=None, low_cpu_mem_usage=True
).to(DEVICE_BASE).eval()

b_attn = base.model.decoder.layers[LAYER].self_attn
Wv_full = b_attn.v_proj.weight.detach().to(torch.float32)     # [7168,7168]
bv_full = b_attn.v_proj.bias.detach().to(torch.float32)        # [7168]
Wo_full = b_attn.out_proj.weight.detach().to(torch.float32)    # [7168,7168]
bo_full = b_attn.out_proj.bias.detach().to(torch.float32)      # [7168]

# ----------------
# Load CMP and grab low-rank weights
# ----------------
print("Load CMP model...")
cmpm = OPTForCausalLM_EigenAttn.from_pretrained(
    CMP_PATH, local_files_only=True, torch_dtype=DTYPE,
    device_map=None, low_cpu_mem_usage=True, low_rank_config=low
).to(DEVICE_CMP).eval()

c_attn = cmpm.model.decoder.layers[LAYER].self_attn
Wv_low_ckpt = c_attn.v_proj_low.weight.detach().to(torch.float32)    # [r,7168]
bv_low_ckpt = c_attn.v_proj_low.bias.detach().to(torch.float32)      # [r]
Wup_ckpt    = c_attn.out_proj_up.weight.detach().to(torch.float32)   # [7168,r]
bup_ckpt    = c_attn.out_proj_up.bias.detach().to(torch.float32)     # [7168]

# move Uv to both devices as needed
Uv_base = Uv.to(DEVICE_BASE)
Uv_cmp  = Uv.to(DEVICE_CMP)

# ----------------
# 1) 判断 Uv 是输出空间 basis 还是输入空间 basis（用投影重构误差）
# ----------------
torch.manual_seed(SEED)
x = torch.randn(BATCH, 7168, device=DEVICE_BASE, dtype=torch.float32)

# full value output: v_full = x Wv^T + b
v_full = x @ Wv_full.t() + bv_full

# output-space projection: v_rec_out = (v_full U) U^T
v_low_from_full = v_full @ Uv_base
v_rec_out = v_low_from_full @ Uv_base.t()
cos_out, rel_out = cos_rel(v_full, v_rec_out)

# input-space projection: x_rec = (x U) U^T, then v_rec_in = x_rec Wv^T + b
x_rec = (x @ Uv_base) @ Uv_base.t()
v_rec_in = x_rec @ Wv_full.t() + bv_full
cos_in, rel_in = cos_rel(v_full, v_rec_in)

print("\n[U-side test]")
print(f"Output-space proj: cos={cos_out:.6f} relL2={rel_out:.6f}")
print(f" Input-space proj: cos={cos_in:.6f} relL2={rel_in:.6f}")
print("=> 哪个 relL2 更小，Uv 就更像哪个空间的 basis（output vs input）")

# ----------------
# 2) 检查 ckpt 的 v_proj_low 是否符合正确公式：Wv_low = U^T Wv_full, bv_low = bv_full U
#    但我们不直接做矩阵乘法（也可以做），直接对同一 x 比对 v_low
# ----------------
# expected low: v_low_expected = (x Wv^T + b) U
v_low_expected = v_low_from_full  # already computed on base device

# ckpt low: v_low_ckpt = x Wv_low^T + b_low
# 注意 ckpt 在另一张卡
x_cmp = x.to(DEVICE_CMP)
v_low_ckpt = x_cmp @ Wv_low_ckpt.t() + bv_low_ckpt

cos_vlow, rel_vlow = cos_rel(v_low_expected.to(DEVICE_CMP), v_low_ckpt)
print("\n[v_proj_low check]")
print(f"v_low_expected vs v_low_ckpt: cos={cos_vlow:.6f} relL2={rel_vlow:.6f}")
print("=> 这里如果很差：要么 Wv_low 构造方向错，要么 Uv 不是 output-space basis（或 layer/file 对错）")

# ----------------
# 3) 检查 ckpt 的 out_proj_up 是否符合正确公式
#    期望：out_expected = (v_low U^T) Wo^T + bo
#    ckpt： out_ckpt = v_low Wup^T + bup
# ----------------
# 用同一个 v_low（用 expected 的 v_low 更“理论干净”）
v_low = v_low_expected.to(DEVICE_BASE)

out_expected = (v_low @ Uv_base.t()) @ Wo_full.t() + bo_full
out_ckpt = (v_low.to(DEVICE_CMP) @ Wup_ckpt.t()) + bup_ckpt

cos_outp, rel_outp = cos_rel(out_expected.to(DEVICE_CMP), out_ckpt)
print("\n[out_proj_up check]")
print(f"out_expected vs out_ckpt: cos={cos_outp:.6f} relL2={rel_outp:.6f}")
print("=> 这里如果很差，而 v_proj_low check 很好：基本就锁定 out_proj_up.weight 构造错")
