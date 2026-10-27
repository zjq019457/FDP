import json, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r7168/final_model"
LOW_JSON="./low_rank_config_layerwise_7168_opt30b.json"
TOK="/diskE/opt-30b"
device="cuda"

layers_to_check = [0, 19]  # 你可以改成 [0, 9, 19, 47]

torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True, use_fast=False)
inp = tok("Hello, my name is", return_tensors="pt")
inp = {k:v.to(device) for k,v in inp.items()}

def summarize(x):
    x = x.detach().float().flatten().cpu()
    return {
        "p999": torch.quantile(x.abs(), 0.999).item(),
        "std": x.std().item(),
        "mean": x.mean().item(),
    }

def attach_attn_hooks(attn_mod, rec, L):
    # input: use pre-hook with kwargs
    def pre_hook(mod, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None:
            # fallback: some implementations pass as first positional
            if len(args) > 0:
                hs = args[0]
        rec[L]["in"] = summarize(hs)

    # output: standard forward hook
    def fwd_hook(mod, args, out):
        # OPT attention returns (attn_output, attn_weights, past_key_value)
        if isinstance(out, (tuple, list)):
            attn_out = out[0]
        else:
            attn_out = out
        rec[L]["out"] = summarize(attn_out)

    h1 = attn_mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = attn_mod.register_forward_hook(fwd_hook)
    return [h1, h2]

def run_base():
    m = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()

    rec = {L:{} for L in layers_to_check}
    hooks=[]
    for L in layers_to_check:
        attn = m.model.decoder.layers[L].self_attn
        hooks += attach_attn_hooks(attn, rec, L)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = m(**inp)

    for h in hooks: h.remove()
    del m; torch.cuda.empty_cache(); gc.collect()
    return rec

def run_cmp():
    cfg = json.load(open(LOW_JSON))
    m = OPTForCausalLM_EigenAttn.from_pretrained(
        CMP, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        low_rank_config=cfg
    ).to(device).eval()
    m = m.to(dtype=torch.bfloat16)

    rec = {L:{} for L in layers_to_check}
    hooks=[]
    for L in layers_to_check:
        attn = m.model.decoder.layers[L].self_attn
        hooks += attach_attn_hooks(attn, rec, L)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = m(**inp)

    for h in hooks: h.remove()
    del m; torch.cuda.empty_cache(); gc.collect()
    return rec

b = run_base()
c = run_cmp()

for L in layers_to_check:
    print(f"\n=== Layer {L} self_attn IO ===")
    for k in ["in","out"]:
        bp, bs = b[L][k]["p999"], b[L][k]["std"]
        cp, cs = c[L][k]["p999"], c[L][k]["std"]
        print(f"{k:3s} base p99.9={bp:.4f} std={bs:.4f} | cmp p99.9={cp:.4f} std={cs:.4f} | ratio p={cp/(bp+1e-12):.3f} std={cs/(bs+1e-12):.3f}")
