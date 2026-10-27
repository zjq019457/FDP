import json, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r7168/final_model"
LOW_JSON="./low_rank_config_layerwise_7168_opt30b.json"
TOK="/diskE/opt-30b"
device="cuda"

tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True, use_fast=False)
inp = tok("Hello, my name is", return_tensors="pt")
inp = {k:v.to(device) for k,v in inp.items()}

def run_base():
    m = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m.model.decoder(**inp, return_dict=True)  # decoder output
        hs = out.last_hidden_state.float().cpu()
        logits = m.lm_head(out.last_hidden_state).float().cpu()
    del m; torch.cuda.empty_cache(); gc.collect()
    return hs, logits

def run_cmp():
    cfg = json.load(open(LOW_JSON))
    m = OPTForCausalLM_EigenAttn.from_pretrained(CMP, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, low_rank_config=cfg).to(device).eval()
    m = m.to(dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m.model.decoder(**inp, return_dict=True)
        hs = out.last_hidden_state.float().cpu()
        logits = m.lm_head(out.last_hidden_state).float().cpu()
    del m; torch.cuda.empty_cache(); gc.collect()
    return hs, logits

def summary(x):
    ax = x.abs().flatten()
    p999 = torch.quantile(ax, 0.999).item()
    std  = x.flatten().std().item()
    return p999, std

b_hs, b_log = run_base()
c_hs, c_log = run_cmp()

bh_p, bh_s = summary(b_hs)
ch_p, ch_s = summary(c_hs)
bl_p, bl_s = summary(b_log)
cl_p, cl_s = summary(c_log)

print("last_hidden p99.9 base/cmp/ratio:", bh_p, ch_p, ch_p/(bh_p+1e-12))
print("last_hidden std   base/cmp/ratio:", bh_s, ch_s, ch_s/(bh_s+1e-12))
print("logits      p99.9 base/cmp/ratio:", bl_p, cl_p, cl_p/(bl_p+1e-12))
print("logits      std   base/cmp/ratio:", bl_s, cl_s, cl_s/(bl_s+1e-12))
