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
        logits = m(**inp).logits.float().cpu()
    del m; torch.cuda.empty_cache(); gc.collect()
    return logits

def run_cmp():
    cfg = json.load(open(LOW_JSON))
    m = OPTForCausalLM_EigenAttn.from_pretrained(CMP, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, low_rank_config=cfg).to(device).eval()
    m = m.to(dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = m(**inp).logits.float().cpu()
    del m; torch.cuda.empty_cache(); gc.collect()
    return logits

base = run_base()
cmp  = run_cmp()

base_std = base.flatten().std().item()
cmp_std  = cmp.flatten().std().item()
g = base_std / (cmp_std + 1e-12)

diff0 = (cmp-base).abs().mean().item()
diff1 = ((cmp*g)-base).abs().mean().item()

print("base_std:", base_std, "cmp_std:", cmp_std, "gain g:", g)
print("diff_mean raw:", diff0)
print("diff_mean scaled:", diff1)
