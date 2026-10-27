import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

def cos(a,b): return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
def rel_l2(a,b): return (a-b).norm().item() / (a.norm().item() + 1e-8)

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r5712/final_model_sanitized_KeqQ"
LOW_JSON="./low_rank_config_layerwise_5712_opt30b.json"

low=json.load(open(LOW_JSON,"r"))

tok=AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=False)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
enc=tok("Hello, my name is", return_tensors="pt")
x=enc["input_ids"]
# 取 base 的 embedding 输出当作 hidden_states 输入（避免整模型跑一遍）
base=AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.float16, device_map=None).cpu().eval()
cmpm=OPTForCausalLM_EigenAttn.from_pretrained(CMP, local_files_only=True, torch_dtype=torch.float16, device_map=None, low_rank_config=low).cpu().eval()

with torch.no_grad():
    hs = base.model.decoder.embed_tokens(x) + base.model.decoder.embed_positions(torch.arange(x.size(1)).unsqueeze(0))
    b_attn=base.model.decoder.layers[0].self_attn
    c_attn=cmpm.model.decoder.layers[0].self_attn

    # base q/k/v
    qb = b_attn.q_proj(hs)
    kb = b_attn.k_proj(hs)
    vb = b_attn.v_proj(hs)

    # cmp q/k/v (low->up)
    qc = c_attn.q_proj_up(c_attn.q_proj_low(hs))
    kc = c_attn.k_proj_up(c_attn.k_proj_low(hs))
    vc = c_attn.v_proj_up(c_attn.v_proj_low(hs))

    print("Q: cos", cos(qb,qc), "relL2", rel_l2(qb,qc))
    print("K: cos", cos(kb,kc), "relL2", rel_l2(kb,kc))
    print("V: cos", cos(vb,vc), "relL2", rel_l2(vb,vc))
