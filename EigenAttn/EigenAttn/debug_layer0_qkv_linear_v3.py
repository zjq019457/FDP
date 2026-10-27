import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

def cos(a,b): return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
def rel_l2(a,b): return (a-b).norm().item() / (a.norm().item() + 1e-8)

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_sanitized_KeqQ"   # 按实际改
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"

low=json.load(open(LOW_JSON,"r"))

tok=AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=False)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
enc=tok("Hello, my name is", return_tensors="pt")
x=enc["input_ids"]                      # [1,T]
attn_mask=torch.ones_like(x)            # OPT pos emb 常用 attention_mask

base=AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True, torch_dtype=torch.float16, device_map=None, low_cpu_mem_usage=True
).cpu().eval()

cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
    CMP, local_files_only=True, torch_dtype=torch.float16, device_map=None,
    low_cpu_mem_usage=True, low_rank_config=low
).cpu().eval()

def find_basis(attn, proj_name):
    cand = [f"{proj_name}_basis", f"{proj_name}_proj_basis"]
    for c in cand:
        if hasattr(attn, c):
            v = getattr(attn, c)
            if torch.is_tensor(v):
                return c, v
    for n in dir(attn):
        if ("basis" in n) and (proj_name in n):
            v = getattr(attn, n)
            if torch.is_tensor(v):
                return n, v
    return None, None

def proj_full(attn, proj_name, hs, hidden_dim):
    low_name = f"{proj_name}_low"
    up_name  = f"{proj_name}_up"

    if not hasattr(attn, low_name):
        raise AttributeError(f"missing {low_name} in OPTEigenAttention")
    low = getattr(attn, low_name)(hs)  # [B,T,r]

    # A) 显式 up Linear
    if hasattr(attn, up_name):
        return getattr(attn, up_name)(low)

    # B) basis 升维
    bname, basis = find_basis(attn, proj_name)
    if basis is None:
        raise AttributeError(f"missing {up_name} and no basis found for {proj_name}")

    r = low.size(-1)
    if basis.dim() != 2:
        raise RuntimeError(f"{bname} dim != 2, got {basis.shape}")

    # basis 期望是 [d,r] 作为 F.linear 的 weight
    if basis.shape == (hidden_dim, r):
        W = basis
    elif basis.shape == (r, hidden_dim):
        W = basis.t()
    else:
        raise RuntimeError(f"{bname} shape {tuple(basis.shape)} not compatible with r={r}, d={hidden_dim}")

    return F.linear(low, W)  # [B,T,d]

def build_opt_embed(decoder, input_ids, attention_mask):
    # tokens
    hs = decoder.embed_tokens(input_ids)

    # OPT 可能有 project_in（当 word_embed_proj_dim != hidden_size）
    if hasattr(decoder, "project_in") and decoder.project_in is not None:
        hs = decoder.project_in(hs)

    # positions：不同版本签名略不同，做 try 兼容
    pos = None
    if hasattr(decoder, "embed_positions") and decoder.embed_positions is not None:
        try:
            pos = decoder.embed_positions(attention_mask=attention_mask)
        except TypeError:
            try:
                pos = decoder.embed_positions(attention_mask)
            except TypeError:
                # 兜底：用 shape 构造（极少数实现）
                pos = decoder.embed_positions(input_ids.shape)

    if pos is not None:
        hs = hs + pos

    # layernorm_embedding（有的版本有）
    if hasattr(decoder, "layernorm_embedding") and decoder.layernorm_embedding is not None:
        hs = decoder.layernorm_embedding(hs)

    return hs

with torch.no_grad():
    d = base.config.hidden_size
    hs = build_opt_embed(base.model.decoder, x, attn_mask)

    b_attn = base.model.decoder.layers[0].self_attn
    c_attn = cmpm.model.decoder.layers[0].self_attn

    print("=== CMP layer0 attr check ===")
    for pn in ["q_proj","k_proj","v_proj"]:
        has_low = hasattr(c_attn, f"{pn}_low")
        has_up  = hasattr(c_attn, f"{pn}_up")
        bname, basis = find_basis(c_attn, pn)
        bs = None if basis is None else tuple(basis.shape)
        print(f"{pn}: low={has_low} up={has_up} basis_name={bname} basis_shape={bs}")

    qb = b_attn.q_proj(hs); kb = b_attn.k_proj(hs); vb = b_attn.v_proj(hs)
    qc = proj_full(c_attn, "q_proj", hs, d)
    kc = proj_full(c_attn, "k_proj", hs, d)
    vc = proj_full(c_attn, "v_proj", hs, d)

    print("Q: cos", cos(qb,qc), "relL2", rel_l2(qb,qc))
    print("K: cos", cos(kb,kc), "relL2", rel_l2(kb,kc))
    print("V: cos", cos(vb,vc), "relL2", rel_l2(vb,vc))
