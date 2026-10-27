import json, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r7168/final_model"
LOW_JSON="./low_rank_config_layerwise_7168_opt30b.json"
TOK="/diskE/opt-30b"
device="cuda"
LAYER=0

torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True, use_fast=False)
inp = tok("Hello, my name is", return_tensors="pt")
inp = {k:v.to(device) for k,v in inp.items()}

def summ(x):
    x = x.detach().float().flatten().cpu()
    return {
        "p999": torch.quantile(x.abs(), 0.999).item(),
        "std": x.std().item(),
        "mean": x.mean().item(),
    }

def run_base():
    m = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    attn = m.model.decoder.layers[LAYER].self_attn
    rec = {}

    # hook q_proj/k_proj/v_proj outputs
    def hook_q(mod, args, out): rec["q"] = summ(out)
    def hook_k(mod, args, out): rec["k"] = summ(out)
    def hook_v(mod, args, out): rec["v"] = summ(out)
    hq = attn.q_proj.register_forward_hook(hook_q)
    hk = attn.k_proj.register_forward_hook(hook_k)
    hv = attn.v_proj.register_forward_hook(hook_v)

    # hook whole attention output + (optionally) attn_weights
    def hook_attn(mod, args, out):
        rec["attn_out"] = summ(out[0])
        if len(out) > 1 and out[1] is not None:
            rec["attn_w"] = summ(out[1])
    ha = attn.register_forward_hook(hook_attn)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = m(**inp)

    for h in [hq,hk,hv,ha]: h.remove()
    del m; torch.cuda.empty_cache(); gc.collect()
    return rec

def run_cmp():
    cfg = json.load(open(LOW_JSON))
    m = OPTForCausalLM_EigenAttn.from_pretrained(CMP, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, low_rank_config=cfg).to(device).eval()
    m = m.to(dtype=torch.bfloat16)
    attn = m.model.decoder.layers[LAYER].self_attn
    rec = {}

    def hook_q(mod, args, out): rec["q_low"] = summ(out)
    def hook_k(mod, args, out): rec["k_low"] = summ(out)
    def hook_v(mod, args, out): rec["v_low"] = summ(out)
    hq = attn.q_proj_low.register_forward_hook(hook_q)
    hk = attn.k_proj_low.register_forward_hook(hook_k)
    hv = attn.v_proj_low.register_forward_hook(hook_v)

    def hook_attn(mod, args, out):
        rec["attn_out"] = summ(out[0])
        if len(out) > 1 and out[1] is not None:
            rec["attn_w"] = summ(out[1])
    ha = attn.register_forward_hook(hook_attn)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = m(**inp)

    for h in [hq,hk,hv,ha]: h.remove()
    del m; torch.cuda.empty_cache(); gc.collect()
    return rec

b = run_base()
c = run_cmp()

def pr(name_b, name_c):
    bb = b.get(name_b, None)
    cc = c.get(name_c, None)
    if bb is None or cc is None:
        print(f"{name_b}/{name_c}: missing")
        return
    print(f"{name_b:8s} vs {name_c:8s} | p99.9 {bb['p999']:.4f} -> {cc['p999']:.4f} (ratio {cc['p999']/(bb['p999']+1e-12):.3f}) | std {bb['std']:.4f} -> {cc['std']:.4f} (ratio {cc['std']/(bb['std']+1e-12):.3f})")

print("=== Layer0 internals ===")
pr("q","q_low")
pr("k","k_low")
pr("v","v_low")
pr("attn_w","attn_w")
pr("attn_out","attn_out")
