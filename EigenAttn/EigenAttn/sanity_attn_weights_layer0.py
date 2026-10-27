import json, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r7168/final_model"
LOW_JSON="./low_rank_config_layerwise_7168_opt30b.json"
TOK="/diskE/opt-30b"
device="cuda"
L=0

torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True, use_fast=False)
inp = tok("Hello, my name is", return_tensors="pt")
inp = {k:v.to(device) for k,v in inp.items()}

def summ(x):
    x = x.detach().float().flatten().cpu()
    return torch.quantile(x, 0.999).item(), torch.quantile(x, 0.001).item(), x.std().item()

def run_base():
    m = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m(**inp, output_attentions=True, return_dict=True)
        w = out.attentions[L]   # (bs, heads, tgt, src)
    del m; torch.cuda.empty_cache(); gc.collect()
    return w

def run_cmp():
    cfg = json.load(open(LOW_JSON))
    m = OPTForCausalLM_EigenAttn.from_pretrained(CMP, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, low_rank_config=cfg).to(device).eval()
    m = m.to(dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m(**inp, output_attentions=True, return_dict=True)
        w = out.attentions[L]
    del m; torch.cuda.empty_cache(); gc.collect()
    return w

bw = run_base()
cw = run_cmp()

b_hi,b_lo,b_std = summ(bw)
c_hi,c_lo,c_std = summ(cw)
print("attn_w q999/q001/std base:", b_hi, b_lo, b_std)
print("attn_w q999/q001/std cmp :", c_hi, c_lo, c_std)
