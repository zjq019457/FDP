import os, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"          # ←改成你要验证的 CMP（r=7168 情况）
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
    print(f"{name}: absmax={t.abs().max().item():.6g} std={t.std().item():.6g} p99={q(t.abs(),0.99):.6g} p999={q(t.abs(),0.999):.6g}")

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

# ---------- BASE: grab attn_in and q_full ----------
base=AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True, torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True
).to(DEVICE).eval()

base_attn = base.model.decoder.layers[LAYER].self_attn
base_ln   = base.model.decoder.layers[LAYER].self_attn_layer_norm

stash={}
def hook_q_full(m, inp, out):
    stash["base_q_in"] = inp[0].detach()
    stash["base_q_out"]= out.detach()

h1 = base_attn.q_proj.register_forward_hook(hook_q_full)

with torch.no_grad():
    # 取 layer0 的真实 attn 输入：hidden_states[0] 过 self_attn_layer_norm
    dec = base.model.decoder(output_hidden_states=True, use_cache=False, return_dict=True, **enc)
    hs0 = dec.hidden_states[0]
    attn_in = base_ln(hs0).detach()
    stash["base_attn_in"]=attn_in
    _ = base_attn.q_proj(attn_in)  # 触发 hook，得到 q_full

h1.remove()

# ---------- CMP: grab q_low actual ----------
del base
torch.cuda.empty_cache()

cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True, low_rank_config=low
).to(DEVICE).eval()

cmp_attn = cmpm.model.decoder.layers[LAYER].self_attn

stash2={}
def hook_q_low(m, inp, out):
    stash2["cmp_q_in"] = inp[0].detach()
    stash2["cmp_q_out"]= out.detach()

assert hasattr(cmp_attn, "q_proj_low"), "CMP has no q_proj_low"
h2 = cmp_attn.q_proj_low.register_forward_hook(hook_q_low)

with torch.no_grad():
    # 关键：用 BASE 的 attn_in 来喂 CMP 的 q_proj_low（排除上游差异）
    _ = cmp_attn.q_proj_low(stash["base_attn_in"])

h2.remove()

# ---------- checks ----------
print("\n[Input check inside CMP]")
c_in = stash2["cmp_q_in"]
b_in = stash["base_attn_in"]
cos_in, rel_in = cos_rel(b_in, c_in)
print(f"cmp q_in vs base attn_in: cos={cos_in:.6f} relL2={rel_in:.6f}")

q_full = stash["base_q_out"]              # [B,T,7168]
q_low_cmp = stash2["cmp_q_out"]           # [B,T,r]
q_low_expected = q_full @ Uq.to(q_full.device)  # [B,T,r]

print("\n[Core functional check]")
cos_q, rel_q = cos_rel(q_low_expected, q_low_cmp)
print(f"q_low_expected (=q_full@Uq) vs q_low_cmp: cos={cos_q:.6f} relL2={rel_q:.6f}")

summ("base q_full", q_full)
summ("cmp  q_low",  q_low_cmp)
summ("exp  q_full@Uq", q_low_expected)

# 如果 r==7168，额外：把 cmp q_low 投回 7168 再比 q_full（需要 Uq 正交）
if r == 7168:
    q_full_hat = q_low_cmp @ Uq.t().to(q_low_cmp.device)  # [B,T,7168]
    cos_b, rel_b = cos_rel(q_full, q_full_hat)
    print("\n[r==7168 back-project check]")
    print(f"q_full vs (q_low_cmp@Uq^T): cos={cos_b:.6f} relL2={rel_b:.6f}")
