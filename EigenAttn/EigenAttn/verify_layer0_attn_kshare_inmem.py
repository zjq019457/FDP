import gc, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"
TOK="/diskE/opt-30b"
PROMPT="Hello, my name is Jiaqi Zhang and I like"
L=0
DEVICE="cuda"
DTYPE=torch.bfloat16

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def cos_rel(a,b,eps=1e-12):
    a=a.flatten().float().cpu()
    b=b.flatten().float().cpu()
    cos=float(torch.dot(a,b)/(a.norm()*b.norm()+eps))
    rel=float((a-b).norm()/(a.norm()+eps))
    return cos, rel

B=torch.load(BASIS_PT, map_location="cpu")
Uq=B[f"base_model.model.model.decoder.layers.{L}.self_attn.q_proj_basis"].float()

tok=AutoTokenizer.from_pretrained(TOK, local_files_only=True, use_fast=False)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
enc=tok(PROMPT, return_tensors="pt")
enc={k:v.to(DEVICE) for k,v in enc.items()}

# --- BASE capture layer0 attn out (last token) + get Wk,bk ---
print("Load BASE...")
base=AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True, torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True
).to(DEVICE).eval()

base_out=None
def hook_base(m, inp, out):
    nonlocal base_out
    x = out[0] if isinstance(out,(tuple,list)) else out
    base_out = x.detach()

h = base.model.decoder.layers[L].self_attn.register_forward_hook(hook_base)
with torch.no_grad():
    _ = base(**enc, use_cache=False)
h.remove()

base_last = base_out[:, -1, :].detach().cpu()
b_attn = base.model.decoder.layers[L].self_attn
Wk = b_attn.k_proj.weight.detach().float().cpu()
bk = b_attn.k_proj.bias.detach().float().cpu()

del base
gc.collect()
torch.cuda.empty_cache()

# --- CMP before patch ---
print("Load CMP...")
low=json.load(open(LOW_JSON,"r"))
cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True, low_rank_config=low
).to(DEVICE).eval()

cmp_out=None
def hook_cmp(m, inp, out):
    nonlocal cmp_out
    x = out[0] if isinstance(out,(tuple,list)) else out
    cmp_out = x.detach()

h2 = cmpm.model.decoder.layers[L].self_attn.register_forward_hook(hook_cmp)
with torch.no_grad():
    _ = cmpm(**enc, use_cache=False)
h2.remove()

cmp_last = cmp_out[:, -1, :].detach().cpu()
cos0, rel0 = cos_rel(base_last, cmp_last)
print("\n[Before patch] cos:", cos0, "relL2:", rel0)

# --- in-memory patch: rebuild layer0 k_proj_low using Uq ---
c_attn = cmpm.model.decoder.layers[L].self_attn
Wk_low = (Uq.t() @ Wk)          # [5712,7168]
bk_low = (bk @ Uq)             # [5712]

with torch.no_grad():
    c_attn.k_proj_low.weight.copy_(Wk_low.to(device=DEVICE, dtype=c_attn.k_proj_low.weight.dtype))
    c_attn.k_proj_low.bias.copy_(bk_low.to(device=DEVICE, dtype=c_attn.k_proj_low.bias.dtype))

# run again
cmp_out2=None
def hook_cmp2(m, inp, out):
    nonlocal cmp_out2
    x = out[0] if isinstance(out,(tuple,list)) else out
    cmp_out2 = x.detach()

h3 = cmpm.model.decoder.layers[L].self_attn.register_forward_hook(hook_cmp2)
with torch.no_grad():
    _ = cmpm(**enc, use_cache=False)
h3.remove()

cmp_last2 = cmp_out2[:, -1, :].detach().cpu()
cos1, rel1 = cos_rel(base_last, cmp_last2)
print("\n[After patch (K uses Q-basis)] cos:", cos1, "relL2:", rel1)
