import json, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

def cos(a,b): return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
def rel_l2(a,b): return (a-b).norm().item() / (a.norm().item() + 1e-8)

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_sanitized_KeqQ"  # 改成你的输出目录
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"
KEQK=1  # KeqQ 就设 1，不做 KeqQ 就设 0

low=json.load(open(LOW_JSON,"r"))
B=torch.load(BASIS_PT, map_location="cpu")

tok=AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=False)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
enc=tok("Hello, my name is", return_tensors="pt")
x=enc["input_ids"]
attn_mask=torch.ones_like(x)

base=AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True, torch_dtype=torch.float16, device_map=None, low_cpu_mem_usage=True
).cpu().eval()

cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=torch.float16, device_map=None,
    low_cpu_mem_usage=True, low_rank_config=low
).cpu().eval()

def build_opt_embed(decoder, input_ids, attention_mask):
    hs = decoder.embed_tokens(input_ids)
    if hasattr(decoder, "project_in") and decoder.project_in is not None:
        hs = decoder.project_in(hs)
    pos = None
    if hasattr(decoder, "embed_positions") and decoder.embed_positions is not None:
        try:
            pos = decoder.embed_positions(attention_mask=attention_mask)
        except TypeError:
            pos = decoder.embed_positions(attention_mask)
    if pos is not None:
        hs = hs + pos
    if hasattr(decoder, "layernorm_embedding") and decoder.layernorm_embedding is not None:
        hs = decoder.layernorm_embedding(hs)
    return hs

def getU(layer, kind):
    key=f"base_model.model.model.decoder.layers.{layer}.self_attn.{kind}_proj_basis"
    return B[key].float()  # [d,r] 用 float32 做对齐计算

with torch.no_grad():
    # ✅ 关键：hs dtype 要跟 base 权重一致（通常是 float16）
    base_dtype = base.model.decoder.embed_tokens.weight.dtype
    hs = build_opt_embed(base.model.decoder, x, attn_mask).to(base_dtype)

    n_layers = base.config.num_hidden_layers
    print("n_layers =", n_layers, "KEQK =", KEQK)

    # 统计最差（最大 relL2）方便定位哪层坏
    worst = {"q":(-1.0,None), "k":(-1.0,None), "v":(-1.0,None)}  # relL2, layer

    for L in range(n_layers):
        b_attn=base.model.decoder.layers[L].self_attn
        c_attn=cmpm.model.decoder.layers[L].self_attn

        Uq=getU(L,"q")
        Uk=Uq if KEQK==1 else getU(L,"k")
        Uv=getU(L,"v")

        # forward 用 float16，比较用 float32
        q_full=b_attn.q_proj(hs).float()   # [B,T,d]
        k_full=b_attn.k_proj(hs).float()
        v_full=b_attn.v_proj(hs).float()

        q_low=c_attn.q_proj_low(hs).float()  # [B,T,r]
        k_low=c_attn.k_proj_low(hs).float()
        v_low=c_attn.v_proj_low(hs).float()

        q_t=q_full @ Uq
        k_t=k_full @ Uk
        v_t=v_full @ Uv

        q_c, q_r = cos(q_low,q_t), rel_l2(q_low,q_t)
        k_c, k_r = cos(k_low,k_t), rel_l2(k_low,k_t)
        v_c, v_r = cos(v_low,v_t), rel_l2(v_low,v_t)

        if q_r > worst["q"][0]: worst["q"]=(q_r,L)
        if k_r > worst["k"][0]: worst["k"]=(k_r,L)
        if v_r > worst["v"][0]: worst["v"]=(v_r,L)

        if L % 6 == 0 or L == n_layers-1:
            print(f"L{L:02d} | Q cos {q_c:.6f} relL2 {q_r:.6e} | "
                  f"K cos {k_c:.6f} relL2 {k_r:.6e} | "
                  f"V cos {v_c:.6f} relL2 {v_r:.6e}")

    print("worst relL2:", worst)
