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
    return box["hs"]

@torch.no_grad()
def base_uniform(attn_mod, hs):
    w = attn_mod.v_proj.weight
    hs = hs.to(device=w.device, dtype=w.dtype)

    B,T,_ = hs.shape
    V = attn_mod.v_proj(hs)  # [B,T,embed]
    # flat average then out_proj
    v_flat = V.mean(dim=1)   # [B,embed]
    ow = attn_mod.out_proj.weight
    v_flat = v_flat.to(device=ow.device, dtype=ow.dtype)
    out_flat = attn_mod.out_proj(v_flat)

    # head average then out_proj (should be same as flat in full model)
    H = attn_mod.num_heads
    D = attn_mod.head_dim
    Vh = V.view(B,T,H,D).transpose(1,2).contiguous()  # [B,H,T,D]
    v_head = Vh.mean(dim=2).reshape(B, H*D)           # [B,embed]
    v_head = v_head.to(device=ow.device, dtype=ow.dtype)
    out_head = attn_mod.out_proj(v_head)

    return out_flat, out_head

@torch.no_grad()
def cmp_uniform(attn_mod, hs):
    w = attn_mod.v_proj_low.weight
    hs = hs.to(device=w.device, dtype=w.dtype)

    B,T,_ = hs.shape
    V = attn_mod.v_proj_low(hs)  # [B,T,rank]
    # flat average then out_proj_up
    v_flat = V.mean(dim=1)       # [B,rank]
    ow = attn_mod.out_proj_up.weight
    v_flat = v_flat.to(device=ow.device, dtype=ow.dtype)
    out_flat = attn_mod.out_proj_up(v_flat)

    # head average then out_proj_up (your current assumption)
    H = attn_mod.num_heads
    Dv = attn_mod.rank_v // H
    Vh = V.view(B,T,H,Dv).transpose(1,2).contiguous() # [B,H,T,Dv]
    v_head = Vh.mean(dim=2).reshape(B, H*Dv)          # [B,rank]
    v_head = v_head.to(device=ow.device, dtype=ow.dtype)
    out_head = attn_mod.out_proj_up(v_head)

    return out_flat, out_head

def main():
    base_path="/diskE/opt-30b"
    cmp_path="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed"
    low=json.load(open("./low_rank_config_layerwise_5712_opt30b.json","r"))

    tok=AutoTokenizer.from_pretrained(base_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    prompt="Hello, my name is"
    device="cuda:0"
    layer_id=0

    print("Load BASE...")
    base=AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True
    ).to(device).eval()

    hs_b = capture_attn_hidden_states(base, tok, prompt, layer_id=layer_id, device=device)
    base_attn = base.model.decoder.layers[layer_id].self_attn
    out_b_flat, out_b_head = base_uniform(base_attn, hs_b)
    out_b_flat = out_b_flat.detach().float().cpu()
    out_b_head = out_b_head.detach().float().cpu()

    del base
    gc.collect()
    torch.cuda.empty_cache()

    print("Load CMP...")
    cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
        cmp_path, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True,
        low_rank_config=low
    ).to(device).eval()

    with torch.no_grad():
        for mm in cmpm.modules():
            if hasattr(mm, "vscale_log"):
                mm.vscale_log.data.fill_(0.0)

    hs_c = capture_attn_hidden_states(cmpm, tok, prompt, layer_id=layer_id, device=device)
    cmp_attn = cmpm.model.decoder.layers[layer_id].self_attn
    out_c_flat, out_c_head = cmp_uniform(cmp_attn, hs_c)
    out_c_flat = out_c_flat.detach().float().cpu()
    out_c_head = out_c_head.detach().float().cpu()

    print("\n[Uniform-V] compare BASE vs CMP")
    print("BASE flat vs CMP flat: cos=%.4f relL2=%.4f" % (cosine(out_b_flat, out_c_flat), rel_l2(out_b_flat, out_c_flat)))
    print("BASE flat vs CMP head: cos=%.4f relL2=%.4f" % (cosine(out_b_flat, out_c_head), rel_l2(out_b_flat, out_c_head)))
    print("(sanity) BASE flat vs BASE head: cos=%.4f relL2=%.4f" % (cosine(out_b_flat, out_b_head), rel_l2(out_b_flat, out_b_head)))

if __name__=="__main__":
    main()
