import os, json, argparse, math, gc
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

# -------------------------
# determinism / stability knobs
# -------------------------
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def q(x, p):
    return torch.quantile(x, p).item()

def summarize(x: torch.Tensor):
    x = x.detach().float().flatten().cpu()
    return {
        "absmax": x.abs().max().item(),
        "mean": x.mean().item(),
        "std": x.std().item(),
        "p99_abs": q(x.abs(), 0.99),
        "p999_abs": q(x.abs(), 0.999),
    }

def fmt(d):
    return " ".join([f"{k}={d[k]:.6g}" for k in ["absmax","p999_abs","p99_abs","std","mean"] if k in d])

def is_finite_tensor(t: torch.Tensor):
    return torch.isfinite(t).all().item()

def get_last_logits(logits: torch.Tensor):
    # logits: [B, T, V]
    return logits[:, -1, :]

def topk_agreement(a: torch.Tensor, b: torch.Tensor, k: int):
    # a,b: [B, V]
    ta = torch.topk(a, k=k, dim=-1).indices
    tb = torch.topk(b, k=k, dim=-1).indices
    # compute intersection size / k
    inter = []
    for i in range(a.size(0)):
        sa = set(ta[i].tolist())
        sb = set(tb[i].tolist())
        inter.append(len(sa & sb) / k)
    return sum(inter) / len(inter)

def load_tok(tok_path):
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok

def load_model_base(path, dtype, device_map):
    m, info = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        output_loading_info=True,
    )
    m.eval()
    return m, info

def load_model_cmp(path, dtype, device_map):
    m, info = OPTForCausalLM_EigenAttn.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        output_loading_info=True,
    )
    m.eval()
    return m, info

def attach_layer_finite_hooks(model, prefix=""):
    hooks = []
    bad = {"hit": False}

    def hook_fn(i):
        def _h(mod, inp, out):
            if bad["hit"]:
                return
            t = out
            if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                t = out[0]
            if torch.is_tensor(t) and (not torch.isfinite(t).all()):
                print(f"[NaN/Inf] {prefix} layer={i} module={mod.__class__.__name__}")
                bad["hit"] = True
        return _h

    # OPT decoder layers are usually at model.model.decoder.layers
    layers = None
    for attr in ["model", "model_decoder", "decoder"]:
        pass
    try:
        layers = model.model.decoder.layers
    except Exception:
        try:
            layers = model.decoder.layers
        except Exception:
            layers = None

    if layers is None:
        print("[warn] cannot find decoder layers for hooks; skip layer hooks")
        return hooks

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(hook_fn(i)))
    return hooks

@torch.no_grad()
def run_logits(model, tok, prompts, max_len, device_for_inputs=None):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    if device_for_inputs is None:
        # if device_map is used, inputs can stay on cuda:0
        device_for_inputs = "cuda:0" if torch.cuda.is_available() else "cpu"
    enc = {k: v.to(device_for_inputs) for k, v in enc.items()}
    out = model(**enc, use_cache=False)
    logits = out.logits  # [B,T,V]
    return logits

def print_loading_info(tag, info):
    mk = info.get("missing_keys", []) or []
    uk = info.get("unexpected_keys", []) or []
    mm = info.get("mismatched_keys", []) or []
    print(f"[{tag}] missing_keys={len(mk)} unexpected_keys={len(uk)} mismatched_keys={len(mm)}")
    if len(mk) > 0:
        print(f"[{tag}] missing_keys_head: {mk[:20]}")
    if len(uk) > 0:
        print(f"[{tag}] unexpected_keys_head: {uk[:20]}")
    if len(mm) > 0:
        print(f"[{tag}] mismatched_keys_head: {mm[:5]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="/diskE/opt-30b")
    ap.add_argument("--cmp_model", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default="/diskE/opt-30b")
    ap.add_argument("--device_map", type=str, default="auto", choices=["auto", "none"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--check_layers", type=int, default=1)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device_map = "auto" if args.device_map == "auto" else None

    tok = load_tok(args.tokenizer)

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In a shocking finding, scientists discovered that",
        "Write a short explanation of attention in transformers:",
    ]

    # -------------------------
    # 1) baseline logits
    # -------------------------
    print("=== Load BASE ===")
    base, info_b = load_model_base(args.base_model, dtype=dtype, device_map=device_map)
    print_loading_info("BASE", info_b)

    hooks = []
    if args.check_layers:
        hooks = attach_layer_finite_hooks(base, prefix="BASE")

    print("=== Run BASE ===")
    logits_b = run_logits(base, tok, prompts, max_len=args.max_len)
    if hooks:
        for h in hooks: h.remove()

    if not is_finite_tensor(logits_b):
        print("[BASE] logits has NaN/Inf -> stop")
        return

    sb = summarize(logits_b)
    print(f"[BASE] logits: {fmt(sb)}")

    last_b = get_last_logits(logits_b).float().cpu()  # store on CPU
    # free baseline
    del logits_b
    del base
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------
    # 2) cmp logits
    # -------------------------
    print("\n=== Load CMP ===")
    cmpm, info_c = load_model_cmp(args.cmp_model, dtype=dtype, device_map=device_map)
    print_loading_info("CMP", info_c)

    hooks = []
    if args.check_layers:
        hooks = attach_layer_finite_hooks(cmpm, prefix="CMP")

    print("=== Run CMP ===")
    logits_c = run_logits(cmpm, tok, prompts, max_len=args.max_len)
    if hooks:
        for h in hooks: h.remove()

    if not is_finite_tensor(logits_c):
        print("[CMP] logits has NaN/Inf -> investigate attention softmax/scale/mask")
        sc = summarize(torch.nan_to_num(logits_c))
        print(f"[CMP] logits(nan_to_num): {fmt(sc)}")
        return

    sc = summarize(logits_c)
    print(f"[CMP] logits: {fmt(sc)}")

    last_c = get_last_logits(logits_c).float().cpu()
    del logits_c
    del cmpm
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------
    # 3) compare
    # -------------------------
    print("\n=== Compare (last token) ===")
    diff = (last_c - last_b)
    print(f"[DIFF] absmax={diff.abs().max().item():.6g} mean_abs={diff.abs().mean().item():.6g}")

    # Top-1 agreement
    top1_b = torch.argmax(last_b, dim=-1)
    top1_c = torch.argmax(last_c, dim=-1)
    top1_acc = (top1_b == top1_c).float().mean().item()
    print(f"[Top1] agreement={top1_acc:.4f}")

    # Top-5 overlap
    top5 = topk_agreement(last_b, last_c, k=5)
    print(f"[Top5] overlap={top5:.4f}")

    # KL on last token distribution
    pb = F.softmax(last_b, dim=-1)
    logpc = F.log_softmax(last_c, dim=-1)
    kl = F.kl_div(logpc, pb, reduction="batchmean").item()
    print(f"[KL] batchmean={kl:.6g}")

if __name__ == "__main__":
    main()
