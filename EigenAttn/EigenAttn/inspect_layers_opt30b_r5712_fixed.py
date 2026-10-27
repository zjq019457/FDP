import json, torch
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

low=json.load(open("./low_rank_config_layerwise_5712_opt30b.json","r"))

m=OPTForCausalLM_EigenAttn.from_pretrained(
    "/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed",
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map=None,
    low_cpu_mem_usage=True,
    low_rank_config=low
).cuda().eval()

def tn(x):
    x=x.detach()
    return float(x.float().norm().item()), float(x.abs().max().item())

print("layer  vscale_finite  vscale_val")
for i, layer in enumerate(m.model.decoder.layers):
    attn = layer.self_attn
    v = getattr(attn, "vscale_log", None)
    vfin = torch.isfinite(v).all().item()
    vval = float(v.detach().float().mean().item())
    print(f"{i:02d}    {vfin}    {vval:+.6f}")
