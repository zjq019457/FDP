import json, gc, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

@torch.no_grad()
def run_and_capture(model, tok, prompt, max_len=128, device="cuda:0"):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k:v.to(device) for k,v in enc.items()}

    hs = []
    hooks = []

    # hook each decoder layer output (hidden_states)
    for i, layer in enumerate(model.model.decoder.layers):
        def mk(i):
            def _h(mod, inp, out):
                x = out[0] if isinstance(out,(tuple,list)) else out
                # take last token
                hs.append((i, x[:, -1, :].detach().float().cpu()))
            return _h
        hooks.append(layer.register_forward_hook(mk(i)))

    out = model(**enc, use_cache=False)
    for h in hooks: h.remove()

    hs.sort(key=lambda t: t[0])
    return [v for _,v in hs], out.logits[:, -1, :].detach().float().cpu()

def cosine(a,b):
    return F.cosine_similarity(a, b, dim=-1).mean().item()

def rel_l2(a,b):
    num = (a-b).norm(dim=-1).mean().item()
    den = (a.norm(dim=-1).mean().item() + 1e-8)
    return num/den

def main():
    base_path="/diskE/opt-30b"
    cmp_path="/diskD/eigen_outputs_opt30b_r5712/final_model_vscale_fixed"
    low=json.load(open("./low_rank_config_layerwise_5712_opt30b.json","r"))

    tok=AutoTokenizer.from_pretrained(base_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token

    prompt="Hello, my name is"
    device="cuda:0"

    print("Load BASE...")
    base=AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map=None, low_cpu_mem_usage=True
    ).to(device).eval()

    base_hs, base_lg = run_and_capture(base, tok, prompt, device=device)
    del base
    gc.collect()
    torch.cuda.empty_cache()

    print("Load CMP...")
    cmpm=OPTForCausalLM_EigenAttn.from_pretrained(
        cmp_path, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map=None, low_cpu_mem_usage=True, low_rank_config=low
    ).to(device).eval()
    with torch.no_grad():
        n = 0
        for mm in cmpm.modules():
            if hasattr(mm, "vscale_log"):
                mm.vscale_log.data.fill_(0.0)  # vscale = exp(0) = 1
                n += 1
    print(f"[CMP] forced vscale=1 on {n} modules")
    cmp_hs, cmp_lg = run_and_capture(cmpm, tok, prompt, device=device)

    print("Layerwise compare (last token hidden):")
    for i,(hb,hc) in enumerate(zip(base_hs, cmp_hs)):
        print(f"layer {i:02d} cos={cosine(hb,hc):.4f} relL2={rel_l2(hb,hc):.4f}")

    print("\nLogits compare:")
    print("logits cos:", cosine(base_lg, cmp_lg))
    print("logits relL2:", rel_l2(base_lg, cmp_lg))

if __name__=="__main__":
    main()
