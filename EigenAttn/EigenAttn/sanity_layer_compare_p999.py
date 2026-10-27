import json, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

BASE="/diskE/opt-30b"
CMP="/diskD/eigen_outputs_opt30b_r7168/final_model"
LOW_JSON="./low_rank_config_layerwise_7168_opt30b.json"
TOK="/diskE/opt-30b"
device="cuda"

torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True, use_fast=False)
inp = tok("Hello, my name is", return_tensors="pt")
inp = {k:v.to(device) for k,v in inp.items()}

def collect_stats(model):
    stats = {}
    hooks=[]
    layers = model.model.decoder.layers

    def make_hook(i):
        def hook(mod, ins, outs):
            hs = outs[0] if isinstance(outs, (tuple,list)) else outs
            x = hs.detach().float().flatten().cpu()
            p999 = torch.quantile(x.abs(), 0.999).item()
            std  = x.std().item()
            stats[i] = (p999, std)
        return hook

    for i,layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = model(**inp)

    for h in hooks: h.remove()
    return stats

def run_base():
    m = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    s = collect_stats(m)
    del m; torch.cuda.empty_cache(); gc.collect()
    return s

def run_cmp():
    cfg = json.load(open(LOW_JSON))
    m = OPTForCausalLM_EigenAttn.from_pretrained(
        CMP, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        low_rank_config=cfg
    ).to(device).eval()
    m = m.to(dtype=torch.bfloat16)
    s = collect_stats(m)
    del m; torch.cuda.empty_cache(); gc.collect()
    return s

base = run_base()
cmp  = run_cmp()

print("idx  base_p99.9  cmp_p99.9  ratio    base_std  cmp_std  ratio")
for i in range(len(base)):
    bp, bs = base[i]
    cp, cs = cmp[i]
    r1 = cp/(bp+1e-12)
    r2 = cs/(bs+1e-12)
    print(f"{i:2d}  {bp:10.4f}  {cp:10.4f}  {r1:6.3f}   {bs:8.4f}  {cs:8.4f}  {r2:6.3f}")
