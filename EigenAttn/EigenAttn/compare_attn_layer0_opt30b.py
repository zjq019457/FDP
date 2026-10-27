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
def grab_attn_out(model, tok, prompt, layer_id=0, device="cuda:0", max_len=128):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k:v.to(device) for k,v in enc.items()}

    out_box = {}

    # OPT decoder layer: layer.self_attn returns (attn_output, attn_weights, past_kv)
    def hook(mod, inp, out):
        attn_out = out[0] if isinstance(out,(tuple,list)) else out
        out_box["x"] = attn_out[:, -1, :].detach().float().cpu()

    h = model.model.decoder.layers[layer_id].self_attn.register_forward_hook(hook)
    model(**enc, use_cache=False)
    h.remove()

    return out_box["x"]

def main():
    base_path="/diskE/opt-30b"
    cmp_path="/diskD/eigen_outputs_opt30b_r5712/final_model_KeqQ_alllayers"
    low=json.load(open("./low_rank_config_layerwise_5712_opt30b.json","r"))

    tok=AutoTokenizer.from_pretrained(base_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token

    prompt="Hello, my name is"
    device="cuda:0"
    layer_id=0

    # -------- BASE (load -> run -> free) --------
    print("Load BASE...")
    base=AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True
    ).to(device).eval()

    a0 = grab_attn_out(base, tok, prompt, layer_id=layer_id, device=device)
    print("[BASE] grabbed layer0 self_attn out.")

    del base
    gc.collect()
    torch.cuda.empty_cache()

    # -------- CMP (load -> (force vscale=1) -> run) --------
    print("Load CMP...")
    cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
        cmp_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
        low_rank_config=low
    ).to(device).eval()

    # 强制 vscale=1，避免缩放污染（你现在 fixed 模型本身就是 0，但这里再保险）
    with torch.no_grad():
        n=0
        for mm in cmpm.modules():
            if hasattr(mm, "vscale_log"):
                mm.vscale_log.data.fill_(0.0)
                n+=1
    print(f"[CMP] forced vscale=1 on {n} modules")

    c0 = grab_attn_out(cmpm, tok, prompt, layer_id=layer_id, device=device)
    print("[CMP] grabbed layer0 self_attn out.")

    print("\nlayer0 self_attn out compare:")
    print("cos:", cosine(a0, c0))
    print("relL2:", rel_l2(a0, c0))

if __name__=="__main__":
    main()
