import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"
TOK="/diskE/opt-30b"
PROMPT="Hello, my name is Jiaqi Zhang and I like"
LAYER=0

def q(x,p): return torch.quantile(x, p).item()
def summ(name, t):
    t=t.detach().float().flatten()
    print(f"{name}: absmax={t.abs().max().item():.6g} std={t.std().item():.6g} "
          f"p99={q(t.abs(),0.99):.6g} p999={q(t.abs(),0.999):.6g} p9999={q(t.abs(),0.9999):.6g}")

def cos_rel(a,b,eps=1e-12):
    a=a.flatten().float(); b=b.flatten().float()
    cos=float(torch.dot(a,b)/(a.norm()*b.norm()+eps))
    rel=float((a-b).norm()/(a.norm()+eps))
    return cos, rel

# --- load basis ---
B=torch.load(BASIS_PT, map_location="cpu")
Uq=B[f"base_model.model.model.decoder.layers.{LAYER}.self_attn.q_proj_basis"].float()  # [7168,r]
r=Uq.shape[1]
print("Uq shape:", tuple(Uq.shape), "r=", r)

# --- tokenizer ---
tok=AutoTokenizer.from_pretrained(TOK, local_files_only=True, use_fast=False)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
enc=tok(PROMPT, return_tensors="pt")
input_ids=enc["input_ids"]
attn_mask=enc.get("attention_mask", torch.ones_like(input_ids))

# --- load models on CPU only (no OOM) ---
base=AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True, torch_dtype=torch.float16, device_map=None, low_cpu_mem_usage=True
).cpu().eval()

low=json.load(open(LOW_JSON,"r"))
cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=torch.float16, device_map=None, low_cpu_mem_usage=True, low_rank_config=low
).cpu().eval()

# --- compute the TRUE layer0 attn_in (same logic as你之前 sanity：hidden_states -> layer0 self_attn_layer_norm) ---
dec = base.model.decoder
layer0 = dec.layers[LAYER]

with torch.no_grad():
    # OPT decoder input: token_emb + pos_emb (+ project_in if exists)
    tok_emb = dec.embed_tokens(input_ids)  # [1,T,7168]
    pos_emb = dec.embed_positions(attn_mask, 0)  # learned pos emb: [1,T,7168]
    hs0 = tok_emb + pos_emb
    if getattr(dec, "project_in", None) is not None:
        hs0 = dec.project_in(hs0)
    # dropout in eval is no-op
    attn_in = layer0.self_attn_layer_norm(hs0)  # [1,T,7168]

# --- BASE q_full on SAME attn_in ---
Wq = layer0.self_attn.q_proj.weight.detach()
bq = layer0.self_attn.q_proj.bias.detach()
with torch.no_grad():
    q_full = F.linear(attn_in, Wq, bq)  # [1,T,7168]

# --- CMP q_low on SAME attn_in ---
c_attn = cmpm.model.decoder.layers[LAYER].self_attn
Wql = c_attn.q_proj_low.weight.detach()
bql = c_attn.q_proj_low.bias.detach()
with torch.no_grad():
    q_low_cmp = F.linear(attn_in, Wql, bql)  # [1,T,r]

# --- expected: q_full @ Uq (do in fp32 to avoid dtype issues, then compare) ---
with torch.no_grad():
    q_low_exp = (q_full.float() @ Uq)  # [1,T,r]

print("\n[Input check] (same attn_in fed to both) => PASS by construction")

print("\n[Core check] q_low_cmp ?= q_full@Uq")
cos, rel = cos_rel(q_low_exp, q_low_cmp)
print(f"cos={cos:.6f} relL2={rel:.6f}")

summ("base q_full", q_full)
summ("exp  q_full@Uq", q_low_exp)
summ("cmp  q_proj_low(attn_in)", q_low_cmp)

# amplitude ratio (解释 q_out 幅度变小是否由投影足够解释)
def ratio(p):
    a=q(q_low_exp.abs().flatten().float(), p)
    b=q(q_full.abs().flatten().float(), p)
    return a/(b+1e-12)

print("\n[Amplitude ratios] (q_full@Uq) / q_full")
for p in [0.99, 0.999, 0.9999]:
    print(f"  p={p}: ratio={ratio(p):.6g}")
