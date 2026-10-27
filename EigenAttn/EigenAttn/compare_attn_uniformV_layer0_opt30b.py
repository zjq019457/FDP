import json, gc, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def cosine(a,b):
    return F.cosine_similarity(a, b, dim=-1).mean().item()

def rel_l2(a,b):
    num=(a-b).norm(dim=-1).mean().item()
    den=(a.norm(dim=-1).mean().item()+1e-8)
    return num/den

@torch.no_grad()
def capture_attn_hidden_states(model, tok, prompt, layer_id=0, device="cuda:0", max_len=128):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k:v.to(device) for k,v in enc.items()}

    attn = model.model.decoder.layers[layer_id].self_attn
    box = {"hs": None}

    # OPTAttention is called with kwargs hidden_states=...
    def prehook(mod, args, kwargs):
        if box["hs"] is None:
            if len(args) > 0 and torch.is_tensor(args[0]):
                box["hs"] = args[0].detach()
            else:
                hs = kwargs.get("hidden_states", None)
                if hs is None:
                    raise RuntimeError("cannot find hidden_states in kwargs")
                box["hs"] = hs.detach()

    h = attn.register_forward_pre_hook(prehook, with_kwargs=True)
    model(**enc, use_cache=False)
    h.remove()

    return box["hs"]  # [B,T,embed]

@torch.no_grad()
def uniform_attn_out_base(attn_mod, hs):
    # hs: [B,T,embed]
    # align dtype/device to v_proj
    w = attn_mod.v_proj.weight
    if (hs.device != w.device) or (hs.dtype != w.dtype):
        hs = hs.to(device=w.device, dtype=w.dtype)

    B,T,_ = hs.shape
    V = attn_mod.v_proj(hs)  # [B,T,embed]
    H = attn_mod.num_heads
    D = attn_mod.head_dim
    V = V.view(B, T, H, D).transpose(1,2).contiguous()  # [B,H,T,D]
    v_mean = V.mean(dim=2)                               # [B,H,D]
    v_last = v_mean.reshape(B, H*D)                      # [B,embed]

    # align to out_proj dtype/device too (extra safety)
    ow = attn_mod.out_proj.weight
    if (v_last.device != ow.device) or (v_last.dtype != ow.dtype):
        v_last = v_last.to(device=ow.device, dtype=ow.dtype)

    out = attn_mod.out_proj(v_last)                      # [B,embed]
    return out

@torch.no_grad()
def uniform_attn_out_cmp(attn_mod, hs):
    # hs: [B,T,embed]
    # align dtype/device to v_proj_low
    w = attn_mod.v_proj_low.weight
    if (hs.device != w.device) or (hs.dtype != w.dtype):
        hs = hs.to(device=w.device, dtype=w.dtype)

    B,T,_ = hs.shape
    V = attn_mod.v_proj_low(hs)  # [B,T,rank_v]
    H = attn_mod.num_heads
    Dv = attn_mod.rank_v // H
    V = V.view(B, T, H, Dv).transpose(1,2).contiguous()  # [B,H,T,Dv]
    v_mean = V.mean(dim=2)                                # [B,H,Dv]
    v_last = v_mean.reshape(B, H*Dv)                      # [B,rank_v]

    # align to out_proj_up dtype/device
    ow = attn_mod.out_proj_up.weight
    if (v_last.device != ow.device) or (v_last.dtype != ow.dtype):
        v_last = v_last.to(device=ow.device, dtype=ow.dtype)

    out = attn_mod.out_proj_up(v_last)                    # [B,embed]
    return out

def main():
    base_path="/diskE/opt-30b"
    cmp_path="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed"
    low=json.load(open("./low_rank_config_layerwise_5712_opt30b.json","r"))

    tok=AutoTokenizer.from_pretrained(base_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    prompt="Hello, my name is"
    device="cuda:0"
    layer_id=0

    # -------- BASE --------
    print("Load BASE...")
    base=AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True
    ).to(device).eval()

    hs_b = capture_attn_hidden_states(base, tok, prompt, layer_id=layer_id, device=device)
    base_attn = base.model.decoder.layers[layer_id].self_attn
    out_b = uniform_attn_out_base(base_attn, hs_b).detach().float().cpu()

    del base
    gc.collect()
    torch.cuda.empty_cache()

    # -------- CMP --------
    print("Load CMP...")
    cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
        cmp_path, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True,
        low_rank_config=low
    ).to(device).eval()

    # force vscale=1 (should already be 0 in vscale_fixed, but keep for safety)
    with torch.no_grad():
        n=0
        for mm in cmpm.modules():
            if hasattr(mm, "vscale_log"):
                mm.vscale_log.data.fill_(0.0)
                n += 1
    print(f"[CMP] forced vscale=1 on {n} modules")

    hs_c = capture_attn_hidden_states(cmpm, tok, prompt, layer_id=layer_id, device=device)
    cmp_attn = cmpm.model.decoder.layers[layer_id].self_attn
    out_c = uniform_attn_out_cmp(cmp_attn, hs_c).detach().float().cpu()

    print("\nUniform-V attention output compare (layer0):")
    print("cos:", cosine(out_b, out_c))
    print("relL2:", rel_l2(out_b, out_c))

if __name__=="__main__":
    main()
