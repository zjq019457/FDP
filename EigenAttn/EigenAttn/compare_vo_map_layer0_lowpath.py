import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed_patchQKVLOW"
LOW="./low_rank_config_layerwise_5712_opt30b.json"
TOK="/diskE/opt-30b"
PROMPT="Hello, my name is Jiaqi Zhang and I like"
L=0

def cos_rel(a,b,eps=1e-12):
    a=a.flatten(); b=b.flatten()
    cos=float(torch.dot(a,b)/(a.norm()*b.norm()+eps))
    rel=float((a-b).norm()/(a.norm()+eps))
    return cos, rel

low=json.load(open(LOW,"r"))
tok=AutoTokenizer.from_pretrained(TOK, local_files_only=True, use_fast=False)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
enc=tok(PROMPT, return_tensors="pt")
input_ids=enc["input_ids"]
attn_mask=enc.get("attention_mask", torch.ones_like(input_ids))

# CPU 即可（只算一层线性）
base=AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.float16, device_map=None).cpu().eval()
cmpm=OPTForCausalLM_EigenAttn.from_pretrained(CMP, local_files_only=True, torch_dtype=torch.float16, device_map=None, low_rank_config=low).cpu().eval()

# 取 BASE 的 embedding hidden_states[0]
with torch.no_grad():
    dec_out = base.model.decoder(
        input_ids=input_ids, attention_mask=attn_mask,
        output_hidden_states=True, use_cache=False, return_dict=True
    )
hs0 = dec_out.hidden_states[0]                        # [1, T, 7168]
hs = base.model.decoder.layers[L].self_attn_layer_norm(hs0)  # self_attn 的真实输入
hs_last = hs[:, -1, :]                                # [1,7168]

b_attn = base.model.decoder.layers[L].self_attn
c_attn = cmpm.model.decoder.layers[L].self_attn

with torch.no_grad():
    # BASE full path
    v_base = b_attn.v_proj(hs_last)                   # [1,7168]
    o_base = b_attn.out_proj(v_base)                  # [1,7168]

    # CMP low path (你修的是这条)
    v_low  = c_attn.v_proj_low(hs_last)               # [1,r]
    o_cmp  = c_attn.out_proj_up(v_low)                # [1,7168]

cos, rel = cos_rel(o_base.float(), o_cmp.float())
print("[LOW-PATH V->O map] on real hs (last token):")
print("cos:", cos)
print("relL2:", rel)

# bias-only：输入 0
z = torch.zeros_like(hs_last)
with torch.no_grad():
    o_base_b = b_attn.out_proj(b_attn.v_proj(z))
    o_cmp_b  = c_attn.out_proj_up(c_attn.v_proj_low(z))
cosb, relb = cos_rel(o_base_b.float(), o_cmp_b.float())
print("\n[LOW-PATH V->O map] on ZERO input (bias-only):")
print("cos:", cosb)
print("relL2:", relb)
