import os, argparse, gc, json, math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

# Force math SDPA path for stability / determinism
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def summarize(x: torch.Tensor):
    x = x.detach().float().flatten().cpu()
    q = lambda p: torch.quantile(x.abs(), p).item()
    return {
        "absmax": x.abs().max().item(),
        "mean": x.mean().item(),
        "std": x.std().item(),
        "p99_abs": q(0.99),
        "p999_abs": q(0.999),
    }


def fmt(d):
    return " ".join([f"{k}={d[k]:.6g}" for k in ["absmax", "p999_abs", "p99_abs", "std", "mean"]])


@torch.no_grad()
def last_logits_no_pad(model, tok, prompts, max_len, device_for_inputs="cuda:0"):
    # run each prompt separately (no padding) to avoid pad-mask NaNs
    outs = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=max_len)  # no padding
        enc = {k: v.to(device_for_inputs) for k, v in enc.items()}
        o = model(**enc, use_cache=False)
        lg = o.logits[:, -1, :]  # [1, V]
        if not torch.isfinite(lg).all():
            raise RuntimeError(f"logits NaN/Inf on prompt: {p[:80]}")
        outs.append(lg.detach().float().cpu())
    return torch.cat(outs, dim=0)  # [B, V]


def topk_overlap(a, b, k):
    ta = torch.topk(a, k=k, dim=-1).indices
    tb = torch.topk(b, k=k, dim=-1).indices
    acc = 0.0
    for i in range(a.size(0)):
        acc += len(set(ta[i].tolist()) & set(tb[i].tolist())) / k
    return acc / a.size(0)


def load_tok(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def maybe_freeze_vscale(model):
    # Set vscale_log = 0 => vscale = exp(0) = 1
    with torch.no_grad():
        n = 0
        for m in model.modules():
            if hasattr(m, "vscale_log") and isinstance(getattr(m, "vscale_log"), torch.nn.Parameter):
                m.vscale_log.data.fill_(0.0)
                n += 1
        print(f"[CMP] freeze_vscale applied to {n} modules (vscale=1.0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="/diskE/opt-30b")
    ap.add_argument("--cmp_model", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default="/diskE/opt-30b")
    ap.add_argument("--low_rank_json", type=str, required=True)
    ap.add_argument("--device_map", type=str, default="auto", choices=["auto", "none"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--freeze_vscale", type=int, default=0, help="1: force all vscale=1 by setting vscale_log=0")
    ap.add_argument("--extra_prompts", type=int, default=0, help="1: use more prompts for more stable metrics")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device_map = "auto" if args.device_map == "auto" else None

    tok = load_tok(args.tokenizer)
    low_rank_config = json.load(open(args.low_rank_json, "r"))

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In a shocking finding, scientists discovered that",
        "Write a short explanation of attention in transformers:",
    ]

    if args.extra_prompts:
        prompts += [
            "The quick brown fox jumps over the lazy dog.",
            "Explain what a GPU does in one paragraph.",
            "Translate the following sentence to Chinese: I like apples.",
            "Summarize the plot of Romeo and Juliet in three sentences.",
            "What is the difference between TCP and UDP?",
            "Define entropy in information theory.",
            "Give me a short recipe for pancakes.",
            "Describe the main idea of gradient descent.",
            "What is a transformer model in machine learning?",
            "Write a haiku about winter.",
            "List three reasons why caching improves performance.",
            "In Python, how do you open a file and read lines?",
        ]

    print("=== Load BASE ===")
    base, info_b = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        output_loading_info=True,
    )
    base.eval()
    print(f"[BASE] missing={len(info_b.get('missing_keys', []) or [])} "
          f"unexpected={len(info_b.get('unexpected_keys', []) or [])} "
          f"mismatched={len(info_b.get('mismatched_keys', []) or [])}")

    print("=== Run BASE (last-token, no-pad) ===")
    last_b = last_logits_no_pad(base, tok, prompts, args.max_len)
    print(f"[BASE] last_logits: {fmt(summarize(last_b))}")

    del base
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== Load CMP ===")
    cmpm, info_c = OPTForCausalLM_EigenAttn.from_pretrained(
        args.cmp_model,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        output_loading_info=True,
        low_rank_config=low_rank_config,
    )
    cmpm.eval()
    print(f"[CMP] missing={len(info_c.get('missing_keys', []) or [])} "
          f"unexpected={len(info_c.get('unexpected_keys', []) or [])} "
          f"mismatched={len(info_c.get('mismatched_keys', []) or [])}")

    if args.freeze_vscale:
        maybe_freeze_vscale(cmpm)

    print("=== Run CMP (last-token, no-pad) ===")
    last_c = last_logits_no_pad(cmpm, tok, prompts, args.max_len)
    print(f"[CMP] last_logits: {fmt(summarize(last_c))}")

    del cmpm
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== Compare (last-token) ===")
    diff = last_c - last_b
    print(f"[DIFF] absmax={diff.abs().max().item():.6g} mean_abs={diff.abs().mean().item():.6g}")

    top1_b = torch.argmax(last_b, dim=-1)
    top1_c = torch.argmax(last_c, dim=-1)
    print(f"[Top1] agreement={(top1_b == top1_c).float().mean().item():.4f}")
    print(f"[Top5] overlap={topk_overlap(last_b, last_c, 5):.4f}")

    pb = F.softmax(last_b, dim=-1)
    logpc = F.log_softmax(last_c, dim=-1)
    kl = F.kl_div(logpc, pb, reduction="batchmean").item()
    print(f"[KL] batchmean={kl:.6g}")


if __name__ == "__main__":
    main()
