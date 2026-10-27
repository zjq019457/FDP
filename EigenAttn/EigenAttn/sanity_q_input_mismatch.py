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
    return torch.quantile(x.abs(), 0.999).item(), x.std().item()

def run_base():
    m = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    attn = m.model.decoder.layers[L].self_attn
    rec = {}

    def attn_pre(mod, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args)>0: hs=args[0]
        rec["attn_in"] = summ(hs)
    h_attn = attn.register_forward_pre_hook(attn_pre, with_kwargs=True)

    def q_pre(mod, args, kwargs):
        x = kwargs.get("input", None)
        # Linear 通常走 positional args
        if x is None and len(args)>0: x=args[0]
        rec["q_in"] = summ(x)
    def q_out(mod, args, out):
        rec["q_out"] = summ(out)
    hq1 = attn.q_proj.register_forward_pre_hook(q_pre, with_kwargs=True)
    hq2 = attn.q_proj.register_forward_hook(q_out)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = m(**inp)

    for h in [h_attn,hq1,hq2]: h.remove()
    del m; torch.cuda.empty_cache(); gc.collect()
    return rec

def run_cmp():
    cfg = json.load(open(LOW_JSON))
    m = OPTForCausalLM_EigenAttn.from_pretrained(CMP, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, low_rank_config=cfg).to(device).eval()
    m = m.to(dtype=torch.bfloat16)
    attn = m.model.decoder.layers[L].self_attn
    rec = {}

    def attn_pre(mod, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args)>0: hs=args[0]
        rec["attn_in"] = summ(hs)
    h_attn = attn.register_forward_pre_hook(attn_pre, with_kwargs=True)

    def q_pre(mod, args, kwargs):
        x = kwargs.get("input", None)
        if x is None and len(args)>0: x=args[0]
        rec["q_in"] = summ(x)
    def q_out(mod, args, out):
        rec["q_out"] = summ(out)
    hq1 = attn.q_proj_low.register_forward_pre_hook(q_pre, with_kwargs=True)
    hq2 = attn.q_proj_low.register_forward_hook(q_out)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = m(**inp)

    for h in [h_attn,hq1,hq2]: h.remove()
    del m; torch.cuda.empty_cache(); gc.collect()
    return rec

b = run_base()
c = run_cmp()

print("BASE attn_in(p999,std):", b["attn_in"], "q_in:", b["q_in"], "q_out:", b["q_out"])
print("CMP  attn_in(p999,std):", c["attn_in"], "q_in:", c["q_in"], "q_out:", c["q_out"])
