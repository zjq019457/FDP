#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def stat(x: torch.Tensor):
    x = x.detach()
    if x.dtype in (torch.float16, torch.bfloat16):
        xf = x.float()
    else:
        xf = x
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "nan": bool(torch.isnan(xf).any().item()) if xf.is_floating_point() else False,
        "inf": bool(torch.isinf(xf).any().item()) if xf.is_floating_point() else False,
        "mean": float(xf.mean().item()) if xf.is_floating_point() else None,
        "std": float(xf.std(unbiased=False).item()) if xf.is_floating_point() else None,
        "norm": float(xf.norm().item()) if xf.is_floating_point() else None,
        "maxabs": float(xf.abs().max().item()) if xf.is_floating_point() else None,
    }

@torch.no_grad()
def eval_one_batch(model, tok, text, device="cuda", max_len=64):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
    loss = out.loss.detach().float().item()
    ppl = float(math.exp(min(loss, 20.0)))  # clamp 防止exp溢出
    logits = out.logits
    return loss, ppl, stat(logits)

def save_arch_report(model, out_path):
    rep = {}
    rep["model_class"] = str(type(model))
    rep["num_params"] = int(sum(p.numel() for p in model.parameters()))
    rep["trainable_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 记录一些关键模块名字（避免文件太大）
    layers = []
    base = getattr(model, "model", None)
    dec = getattr(base, "decoder", None) if base is not None else None
    if dec is not None and hasattr(dec, "layers"):
        for i, layer in enumerate(dec.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            layers.append({
                "i": i,
                "attn_param_prefix_samples": sorted([n for n,_ in attn.named_parameters()])[:80],
                "attn_module_samples": sorted([n for n,_ in attn.named_modules()])[:120],
            })
    rep["layers"] = layers
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="你的 final_model 目录")
    ap.add_argument("--tokenizer_path", required=True, help="tokenizer目录(通常是 baseline OPT-30B tokenizer)")
    ap.add_argument("--device", default="cuda", choices=["cpu","cuda"], help="cuda 或 cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16","bf16","fp32"], help="加载dtype")
    ap.add_argument("--text", default="Hello world. This is a quick sanity check.", help="用于sanity的短文本")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--write_arch", action="store_true", help="写 arch_report.json 到 model_path")
    ap.add_argument("--print_keys", action="store_true", help="打印部分 state_dict keys 命中情况")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    print("="*80)
    print("[Load] model_path:", args.model_path)
    print("[Load] tokenizer_path:", args.tokenizer_path)
    print("[Load] device:", args.device, "dtype:", args.dtype)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    # 关键：输出加载信息（missing/unexpected/mismatch）
    model, info = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None if args.device == "cpu" else {"": 0},
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )

    print("="*80)
    print("[LoadingInfo]")
    mk = info.get("missing_keys", [])
    uk = info.get("unexpected_keys", [])
    mm = info.get("mismatched_keys", [])
    print("  missing_keys    :", len(mk))
    print("  unexpected_keys :", len(uk))
    print("  mismatched_keys :", len(mm))
    if len(mk) > 0:
        print("  sample missing_keys (first 30):")
        for k in mk[:30]:
            print("   -", k)
    if len(uk) > 0:
        print("  sample unexpected_keys (first 30):")
        for k in uk[:30]:
            print("   -", k)
    if len(mm) > 0:
        print("  sample mismatched_keys (first 10):")
        for k in mm[:10]:
            print("   -", k)

    # tie 检查
    print("="*80)
    print("[TieWeights]")
    tied = None
    try:
        w1 = model.model.decoder.embed_tokens.weight
        w2 = model.lm_head.weight
        tied = (w1.data_ptr() == w2.data_ptr())
        print("  tied? :", tied)
        print("  embed_tokens.weight shape:", tuple(w1.shape), "norm:", float(w1.float().norm().item()))
        print("  lm_head.weight      shape:", tuple(w2.shape), "norm:", float(w2.float().norm().item()))
    except Exception as e:
        print("  tie check failed:", repr(e))

    # 建议：显式 tie 一次（不改变数值语义时应该是安全的）
    try:
        model.tie_weights()
        print("  model.tie_weights() done")
    except Exception as e:
        print("  model.tie_weights() failed:", repr(e))

    # 可选：打印 key 命中
    if args.print_keys:
        print("="*80)
        print("[KeyHints]")
        sd = model.state_dict()
        keys = list(sd.keys())
        for kw in ["eigen", "q_proj_low", "k_proj_low", "v_proj_low", "q_proj_up", "k_proj_up", "v_proj_up", "dora", "lora"]:
            hit = [k for k in keys if kw in k]
            print(f"  {kw:12s} hits={len(hit)}")
            for k in hit[:8]:
                print("   -", k)

    # Sanity PPL
    model.eval()
    if args.device == "cuda":
        model.to("cuda")
    print("="*80)
    print("[SanityPPL] text:", args.text)
    loss, ppl, logits_stat = eval_one_batch(model, tok, args.text, device=args.device, max_len=args.max_len)
    print(f"  loss={loss:.6f}  ppl≈{ppl:.4f}")
    print("  logits:", logits_stat)

    # 输出 arch_report
    if args.write_arch:
        out = os.path.join(args.model_path, "arch_report.json")
        save_arch_report(model, out)
        print("="*80)
        print("[ArchReport] saved:", out)

    print("="*80)
    print("[ResultSummary]")
    # 简单判定建议
    if len(mk) > 0 or len(mm) > 0:
        print("  !! 风险：存在 missing/mismatched keys。你的模型可能没正确加载权重 → PPL 会极高。")
    if tied is False:
        print("  !! 注意：embed_tokens 与 lm_head 可能未 tie。建议确认是否需要 tie。")
    if logits_stat["nan"] or logits_stat["inf"]:
        print("  !! 风险：logits 出现 NaN/Inf。")
    print("  Done.")

if __name__ == "__main__":
    main()
