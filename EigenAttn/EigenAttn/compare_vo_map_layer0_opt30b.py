import json, gc, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def cosine(a,b): return F.cosine_similarity(a,b,dim=-1).mean().item()
def rel_l2(a,b):
    num=(a-b).norm(dim=-1).mean().item()
    den=(a.norm(dim=-1).mean().item()+1e-8)
    return num/den

@torch.no_grad()
def capture_attn_input_hs(model, tok, prompt, layer_id=0, device="cuda:0"):
    enc = tok(prompt, return_tensors="pt")
    enc = {k:v.to(device) for k,v in enc.items()}
    attn = model.model.decoder.layers[layer_id].self_attn
    box={"hs":None}
    def prehook(mod, args, kwargs):
        if box["hs"] is None:
            if len(args)>0 and torch.is_tensor(args[0]):
                box["hs"]=args[0].detach()
            else:
                box["hs"]=kwargs["hidden_states"].detach()
    h=attn.register_forward_pre_hook(prehook, with_kwargs=True)
    model(**enc, use_cache=False)
    h.remove()
    return box["hs"]  # [B,T,embed]

@torch.no_grad()
def base_vo(base_attn, hs):
    # align dtype/device
    hs = hs.to(device=base_attn.v_proj.weight.device, dtype=base_attn.v_proj.weight.dtype)
    v = base_attn.v_proj(hs)[:, -1, :]  # last token [B,embed]
    v = v.to(device=base_attn.out_proj.weight.device, dtype=base_attn.out_proj.weight.dtype)
    y = base_attn.out_proj(v)           # [B,embed]
    return y.detach().float().cpu()

@torch.no_grad()
def cmp_vo(cmp_attn, hs):
    hs = hs.to(device=cmp_attn.v_proj_low.weight.device, dtype=cmp_attn.v_proj_low.weight.dtype)
    v = cmp_attn.v_proj_low(hs)[:, -1, :]  # [B,rank]
    v = v.to(device=cmp_attn.out_proj_up.weight.device, dtype=cmp_attn.out_proj_up.weight.dtype)
    y = cmp_attn.out_proj_up(v)            # [B,embed]
    return y.detach().float().cpu()

def main():
    base_path="/diskE/opt-30b"
    cmp_path="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed"
    low=json.load(open("./low_rank_config_layerwise_5712_opt30b.json","r"))

    tok=AutoTokenizer.from_pretrained(base_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token

    prompt="Hello, my name is"
    device="cuda:0"
    layer_id=0

    print("Load BASE...")
    base=AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True
    ).to(device).eval()

    hs_b = capture_attn_input_hs(base, tok, prompt, layer_id, device)
    base_attn = base.model.decoder.layers[layer_id].self_attn
    yb = base_vo(base_attn, hs_b)

    # bias-only probe: hs=0
    yb0 = base_vo(base_attn, torch.zeros_like(hs_b))

    del base
    gc.collect(); torch.cuda.empty_cache()

    print("Load CMP...")
    cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
        cmp_path, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True,
        low_rank_config=low
    ).to(device).eval()

    # force vscale=1
    with torch.no_grad():
        for mm in cmpm.modules():
            if hasattr(mm, "vscale_log"):
                mm.vscale_log.data.fill_(0.0)

    hs_c = capture_attn_input_hs(cmpm, tok, prompt, layer_id, device)
    cmp_attn = cmpm.model.decoder.layers[layer_id].self_attn
    yc = cmp_vo(cmp_attn, hs_c)
    yc0 = cmp_vo(cmp_attn, torch.zeros_like(hs_c))

    print("\n[V->O map] on real hs (last token):")
    print("cos:", cosine(yb, yc))
    print("relL2:", rel_l2(yb, yc))

    print("\n[V->O map] on ZERO input (bias-only behavior):")
    print("cos:", cosine(yb0, yc0))
    print("relL2:", rel_l2(yb0, yc0))

if __name__=="__main__":
    main()
