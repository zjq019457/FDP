import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"
TOK="/diskE/opt-30b"
PROMPT="Hello, my name is Jiaqi Zhang and I like"
LAYER=0
DEVICE="cuda:0"
DTYPE=torch.bfloat16

def q(x,p): return torch.quantile(x, p).item()
def summ(name, t):
    t=t.detach().float().flatten().cpu()
    print(f"{name}: absmax={t.abs().max().item():.6g} std={t.std().item():.6g} "
          f"p99={q(t.abs(),0.99):.6g} p999={q(t.abs(),0.999):.6g} p9999={q(t.abs(),0.9999):.6g}")

def cos_rel(a,b,eps=1e-12):
    a=a.flatten(); b=b.flatten()
    cos=float(torch.dot(a,b)/(a.norm()*b.norm()+eps))
    rel=float((a-b).norm()/(a.norm()+eps))
    return cos, rel

low=json.load(open(LOW_JSON,"r"))
B=torch.load(BASIS_PT, map_location="cpu")
Uq=B[f"base_model.model.model.decoder.layers.{LAYER}.self_attn.q_proj_basis"].float()  # [7168,r]
r=Uq.shape[1]
print("Uq shape:", tuple(Uq.shape), "r=", r)

tok=AutoTokenizer.from_pretrained(TOK, local_files_only=True, use_fast=False)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
enc=tok(PROMPT, return_tensors="pt")
enc={k:v.to(DEVICE) for k,v in enc.items()}

# ---- BASE: get attn_in + q_full on EXACT attn_in ----
base=AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True, torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True
).to(DEVICE).eval()

b_layer = base.model.decoder.layers[LAYER]
b_attn  = b_layer.self_attn
b_ln    = b_layer.self_attn_layer_norm

with torch.no_grad():
    dec = base.model.decoder(output_hidden_states=True, use_cache=False, return_dict=True, **enc)
    hs0 = dec.hidden_states[0]
    attn_in = b_ln(hs0)                       # [1,T,7168] 这是“真实 self_attn 输入”
    q_full  = b_attn.q_proj(attn_in)          # [1,T,7168]

# ---- CMP: run q_proj_low on the SAME attn_in (排除上游差异) ----
del dec, hs0
torch.cuda.empty_cache()

cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True, low_rank_config=low
).to(DEVICE).eval()
c_attn = cmpm.model.decoder.layers[LAYER].self_attn

with torch.no_grad():
    q_low_cmp = c_attn.q_proj_low(attn_in)    # [1,T,r]

# ---- expected: q_full @ Uq ----
Uq_dev = Uq.to(DEVICE)
with torch.no_grad():
    q_low_exp = q_full @ Uq_dev               # [1,T,r]

print("\n[Core check: q_low_cmp ?= q_full@Uq]")
cos, rel = cos_rel(q_low_exp, q_low_cmp)
print(f"cos={cos:.6f} relL2={rel:.6f}")

summ("base q_full", q_full)
summ("exp  q_full@Uq", q_low_exp)
summ("cmp  q_proj_low(attn_in)", q_low_cmp)

# ---- amplitude explanation: ratio of tail quantiles ----
def ratio(p):
    a=q_low_exp.detach().float().abs().flatten()
    b=q_full.detach().float().abs().flatten()
    return float(torch.quantile(a, p)/ (torch.quantile(b, p)+1e-12))
print("\n[Amplitude ratios: (q_full@Uq) / q_full]")
for p in [0.99, 0.999, 0.9999]:
    print(f"  p={p}: ratio={ratio(p):.6g}")
