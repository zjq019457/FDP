import json, torch
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

low=json.load(open("./low_rank_config_layerwise_5712_opt30b.json","r"))

m=OPTForCausalLM_EigenAttn.from_pretrained(
    "/diskD/eigen_outputs_opt30b_r5712/final_model",
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map=None,
    low_cpu_mem_usage=True,
    low_rank_config=low
).cuda().eval()

def tn(x):  # tensor norm / absmax helper
    x=x.detach()
    return float(x.float().norm().item()), float(x.abs().max().item())

print("layer  rank_kq  rank_v  vscale_finite  vscale_val  q_norm  k_norm  v_norm  up_norm  up_absmax")
for i, layer in enumerate(m.model.decoder.layers):
    attn = layer.self_attn
    rkq = getattr(attn, "rank_kq", None)
    rv  = getattr(attn, "rank_v", None)

    vscale = getattr(attn, "vscale_log", None)
    if vscale is None:
        vfin, vval = True, 0.0
    else:
        vfin = torch.isfinite(vscale).all().item()
        vval = float(vscale.detach().float().mean().item())

    qn,_ = tn(attn.q_proj_low.weight)
    kn,_ = tn(attn.k_proj_low.weight)
    vn,_ = tn(attn.v_proj_low.weight)
    upn, upabs = tn(attn.out_proj_up.weight)

    print(f"{i:02d}    {rkq}    {rv}    {vfin}    {vval:+.4f}    {qn:.3e}  {kn:.3e}  {vn:.3e}  {upn:.3e}  {upabs:.3e}")
