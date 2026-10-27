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

def sym_kl(p_logits, q_logits):
    # p,q: [V] float32 on CPU
    p_logp = F.log_softmax(p_logits, dim=-1)
    q_logp = F.log_softmax(q_logits, dim=-1)
    p = p_logp.exp()
    q = q_logp.exp()
    kl_pq = (p * (p_logp - q_logp)).sum().item()
    kl_qp = (q * (q_logp - p_logp)).sum().item()
    return 0.5*(kl_pq + kl_qp), kl_pq, kl_qp

@torch.no_grad()
def grab_last_logits(model, tok, prompt, device, token_index=-1):
    model.eval()
    enc = tok(prompt, return_tensors="pt")
    enc = {k:v.to(device) for k,v in enc.items()}
    out = model(**enc, use_cache=False)
    # logits: [B,T,V]
    logits = out.logits[:, token_index, :].detach().float().cpu().squeeze(0)  # [V]
    return logits

def _load_on_device(load_fn, device: str):
    # 尽量直接 device_map，避免峰值
    try:
        return load_fn(device_map={"": device})
    except Exception:
        m = load_fn(device_map=None)
        return m.to(device)

def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--prompt", default="Hello, my name is")
    ap.add_argument("--token_index", type=int, default=-1)  # -1 = last token
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16"])
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device", default="cuda:1")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    low = json.load(open(args.low_json, "r"))

    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---------------- BASE ----------------
    print(f"Load BASE on {args.base_device} ...")
    def load_base(device_map=None):
        return AutoModelForCausalLM.from_pretrained(
            args.base, local_files_only=True, torch_dtype=dtype,
            low_cpu_mem_usage=True, device_map=device_map
        ).eval()

    base = _load_on_device(load_base, args.base_device)
    base_logits = grab_last_logits(base, tok, args.prompt, args.base_device, args.token_index)
    print("[BASE] grabbed logits:", tuple(base_logits.shape))

    del base
    free_cuda()

    # ---------------- CMP ----------------
    print(f"Load CMP on {args.cmp_device} ...")
    def load_cmp(device_map=None):
        return OPTForCausalLM_EigenAttn.from_pretrained(
            args.cmp, local_files_only=True, torch_dtype=dtype,
            low_cpu_mem_usage=True, device_map=device_map,
            low_rank_config=low
        ).eval()

    cmpm = _load_on_device(load_cmp, args.cmp_device)
    cmp_logits = grab_last_logits(cmpm, tok, args.prompt, args.cmp_device, args.token_index)
    print("[CMP] grabbed logits:", tuple(cmp_logits.shape))

    del cmpm
    free_cuda()

    # ---------------- Compare (CPU) ----------------
    c = cosine(base_logits, cmp_logits)
    r = rel_l2(base_logits, cmp_logits)
    skl, kl_pq, kl_qp = sym_kl(base_logits, cmp_logits)

    base_argmax = int(torch.argmax(base_logits).item())
    cmp_argmax  = int(torch.argmax(cmp_logits).item())
    argmax_same = (base_argmax == cmp_argmax)

    k = args.topk
    b_top = torch.topk(base_logits, k).indices.tolist()
    c_top = torch.topk(cmp_logits,  k).indices.tolist()
    inter = len(set(b_top).intersection(set(c_top)))

    # 一些额外统计
    diff = (base_logits - cmp_logits)
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    print("\nlogits(last token) compare:")
    print("cos:", c)
    print("relL2:", r)
    print("symKL:", skl, " (KL(base||cmp)=", kl_pq, ", KL(cmp||base)=", kl_qp, ")")
    print(f"argmax_same: {argmax_same}  base_argmax={base_argmax}  cmp_argmax={cmp_argmax}")
    print(f"top{k}_overlap: {inter}/{k}")
    print("max_abs_diff:", max_abs)
    print("mean_abs_diff:", mean_abs)

if __name__ == "__main__":
    main()
