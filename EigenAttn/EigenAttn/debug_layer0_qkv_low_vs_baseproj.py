import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

def cos(a,b): return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
def rel_l2(a,b): return (a-b).norm().item() / (a.norm().item() + 1e-8)

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_sanitized_KeqQ"   # 改成你的cmp目录
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"
BASIS_PT="qkv_outputs/eigen_basis.pt"
L=0

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
    CMP, local_files_only=True, torch_dtype=torch.float16, device_map=None, low_cpu_mem_usage=True,
    low_rank_config=low
).cpu().eval()

def getU(kind):
    key=f"base_model.model.model.decoder.layers.{L}.self_attn.{kind}_proj_basis"
    return B[key].float()  # [d,r]

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

with torch.no_grad():
    hs = build_opt_embed(base.model.decoder, x, attn_mask)

    b_attn=base.model.decoder.layers[L].self_attn
    c_attn=cmpm.model.decoder.layers[L].self_attn

    # base full
    q_full=b_attn.q_proj(hs).float()   # [B,T,d]
    k_full=b_attn.k_proj(hs).float()
    v_full=b_attn.v_proj(hs).float()

    # cmp low
    q_low=c_attn.q_proj_low(hs).float()   # [B,T,r]
    k_low=c_attn.k_proj_low(hs).float()
    v_low=c_attn.v_proj_low(hs).float()

    Uq=getU("q")   # [d,r]
    Uk = Uq
    Uv=getU("v")

    # target low = full @ U  (最后一维 matmul)
    q_t = torch.matmul(q_full, Uq)    # [B,T,r]
    k_t = torch.matmul(k_full, Uk)
    v_t = torch.matmul(v_full, Uv)

    print("shapes:",
          "q_low", tuple(q_low.shape), "q_t", tuple(q_t.shape),
          "k_low", tuple(k_low.shape), "k_t", tuple(k_t.shape),
          "v_low", tuple(v_low.shape), "v_t", tuple(v_t.shape))

    print("Q_low vs base@Uq: cos", cos(q_low,q_t), "relL2", rel_l2(q_low,q_t))
    print("K_low vs base@Uk: cos", cos(k_low,k_t), "relL2", rel_l2(k_low,k_t))
    print("V_low vs base@Uv: cos", cos(v_low,v_t), "relL2", rel_l2(v_low,v_t))
