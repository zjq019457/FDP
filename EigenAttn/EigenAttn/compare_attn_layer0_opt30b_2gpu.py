import os, json, gc, argparse, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def cosine(a,b): return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
def rel_l2(a,b):
    num = (a-b).norm().item()
    den = (a.norm().item() + 1e-8)
    return num/den

def _load_on_device(load_fn, device: str):
    # 优先尝试直接 device_map 加载（避免先CPU后GPU的峰值）
    try:
        return load_fn(device_map={"": device})
    except Exception:
        m = load_fn(device_map=None)
        return m.to(device)

@torch.no_grad()
def grab_layer_attn_out(model, tok, prompt: str, layer_id: int, device: str):
    model.eval()
    attn = model.model.decoder.layers[layer_id].self_attn

    box = {}
    def hook(_m, _inp, out):
        # OPT attn forward 通常 out 是 tuple: (attn_output, attn_weights, past_key_value)
        if isinstance(out, (tuple, list)):
            box["y"] = out[0].detach()
        else:
            box["y"] = out.detach()

    h = attn.register_forward_hook(hook)

    enc = tok(prompt, return_tensors="pt")
    enc = {k:v.to(device) for k,v in enc.items()}
    _ = model(**enc, use_cache=False)

    h.remove()
    y = box["y"]
    return y
def force_vscale_to_one(cmpm):
    # 兼容 vscale_log / vscale / vscale_param
    touched = 0
    with torch.no_grad():
        for _, mod in cmpm.named_modules():
            if hasattr(mod, "vscale_log") and torch.is_tensor(getattr(mod, "vscale_log")):
                getattr(mod, "vscale_log").fill_(0.0)   # log(1)=0
                touched += 1
            elif hasattr(mod, "vscale") and torch.is_tensor(getattr(mod, "vscale")):
                getattr(mod, "vscale").fill_(1.0)
                touched += 1
            elif hasattr(mod, "vscale_param") and torch.is_tensor(getattr(mod, "vscale_param")):
                getattr(mod, "vscale_param").fill_(1.0)
                touched += 1
    return touched
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prompt", default="Hello, my name is")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16"])
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device", default="cuda:1")
    ap.add_argument("--force_vscale1", type=int, default=1)
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    low = json.load(open(args.low_json,"r"))

    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    # ----------------------------
    # BASE on base_device, then free
    # ----------------------------
    print(f"Load BASE on {args.base_device} ...")
    def load_base(device_map=None):
        return AutoModelForCausalLM.from_pretrained(
            args.base, local_files_only=True, torch_dtype=dtype,
            low_cpu_mem_usage=True, device_map=device_map
        ).eval()

    base = _load_on_device(load_base, args.base_device)
    y_base = grab_layer_attn_out(base, tok, args.prompt, args.layer, args.base_device).float().cpu()
    print("[BASE] grabbed layer", args.layer, "self_attn out:", tuple(y_base.shape))

    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------
    # CMP on cmp_device, then free
    # ----------------------------
    print(f"Load CMP on {args.cmp_device} ...")
    def load_cmp(device_map=None):
        return OPTForCausalLM_EigenAttn.from_pretrained(
            args.cmp, local_files_only=True, torch_dtype=dtype,
            low_cpu_mem_usage=True, device_map=device_map,
            low_rank_config=low
        ).eval()

    cmpm = _load_on_device(load_cmp, args.cmp_device)

    if args.force_vscale1 == 1:
        n = force_vscale_to_one(cmpm)
        print(f"[CMP] forced vscale=1 on {n} modules")

    y_cmp = grab_layer_attn_out(cmpm, tok, args.prompt, args.layer, args.cmp_device).float().cpu()
    print("[CMP] grabbed layer", args.layer, "self_attn out:", tuple(y_cmp.shape))

    del cmpm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------
    # Compare on CPU
    # ----------------------------
    print("\nlayer{} self_attn out compare:".format(args.layer))
    print("cos:", cosine(y_base, y_cmp))
    print("relL2:", rel_l2(y_base, y_cmp))

if __name__ == "__main__":
    main()
