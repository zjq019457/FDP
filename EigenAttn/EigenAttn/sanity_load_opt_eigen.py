#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn


def load_sharded_safetensors(model_dir: str) -> dict:
    """Load sharded safetensors using model.safetensors.index.json -> merged state_dict (cpu)."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index: {index_path}")

    idx = json.load(open(index_path, "r"))
    weight_map = idx.get("weight_map", {})
    if not weight_map:
        raise RuntimeError(f"Empty weight_map in {index_path}")

    shard_files = sorted(set(weight_map.values()))
    shard_paths = [os.path.join(model_dir, f) for f in shard_files]

    sd = {}
    for p in shard_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Shard not found: {p}")
        part = load_file(p, device="cpu")
        sd.update(part)
    return sd


@torch.no_grad()
def sanity_ppl(model, tokenizer, text: str, device: str = "cuda", max_len: int = 64):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, labels=enc["input_ids"])
    loss = float(out.loss.detach().float().item())
    ppl = math.exp(min(loss, 20.0))
    return loss, ppl


def main():
    # ====== paths (edit here if needed) ======
    #MODEL_DIR = "/diskD/eigen_outputs_opt30b_r448/final_model"
    import os
    MODEL_DIR = os.getenv("MODEL_DIR", "/diskD/eigen_outputs_opt30b_r448/final_model")

    TOK_DIR = "/diskE/opt-30b"
    LOW_RANK_JSON = "./low_rank_config_layerwise_448_opt30b.json"

    TEXT = "Hello world. This is a quick sanity check."
    DEVICE = "cuda"
    DTYPE = torch.float16

    print("=" * 80)
    print("[Env] EIGEN_SCALE_ORIG =", os.getenv("EIGEN_SCALE_ORIG", "0"))
    print("[Paths]")
    print("  MODEL_DIR   =", MODEL_DIR)
    print("  TOK_DIR     =", TOK_DIR)
    print("  LOW_RANK_JSON =", LOW_RANK_JSON)

    # 1) load sd
    print("=" * 80)
    sd = load_sharded_safetensors(MODEL_DIR)
    print("[StateDict] keys =", len(sd))

    # 2) build model
    low_rank_config = json.load(open(LOW_RANK_JSON, "r"))
    config = AutoConfig.from_pretrained(MODEL_DIR)

    model = OPTForCausalLM_EigenAttn(config, low_rank_config=low_rank_config).to(DEVICE, dtype=DTYPE)

    # 3) load weights
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("=" * 80)
    print("[LoadStateDict] missing =", len(missing), "unexpected =", len(unexpected))
    if missing:
        print("  missing sample:", missing[:20])
    if unexpected:
        print("  unexpected sample:", unexpected[:20])

    # 4) tie weights (remove lm_head noise)
    model.tie_weights()
    w_embed = model.model.decoder.embed_tokens.weight
    w_head = model.lm_head.weight
    print("=" * 80)
    print("[Tie check] tied? :", (w_embed.data_ptr() == w_head.data_ptr()))
    print("[Tie check] embed norm:", float(w_embed.float().norm().item()))
    print("[Tie check] head  norm:", float(w_head.float().norm().item()))

    # 5) sanity ppl
    tok = AutoTokenizer.from_pretrained(TOK_DIR, use_fast=True)
    model.eval()
    loss, ppl = sanity_ppl(model, tok, TEXT, device=DEVICE)
    print("=" * 80)
    print("[Sanity] loss =", loss, "ppl =", ppl)
    print("=" * 80)


if __name__ == "__main__":
    main()
