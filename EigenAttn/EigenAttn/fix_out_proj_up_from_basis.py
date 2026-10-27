#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM

# -------------------------
# paths (EDIT if needed)
# -------------------------
FINAL_MODEL_DIR = "/diskD/eigen_outputs_opt30b_r448/final_model"
OUT_DIR         = "/diskD/eigen_outputs_opt30b_r448/final_model_fixed_up"

BASELINE_DIR    = "/diskE/opt-30b"              # baseline OPT-30B HF checkpoint dir
EIGEN_BASIS_PT  = "qkv_outputs/eigen_basis.pt"  # your basis file

LOW_RANK_JSON   = "./low_rank_config_layerwise_448_opt30b.json"  # only used to sanity-check r_v

DEVICE_BASELINE = "cpu"  # keep baseline on cpu for this rewrite job

# -------------------------
# helpers
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_hf_skeleton(src_dir, dst_dir):
    ensure_dir(dst_dir)
    # copy config/tokenizer/generation files + index + all shards initially
    for fn in os.listdir(src_dir):
        src = os.path.join(src_dir, fn)
        dst = os.path.join(dst_dir, fn)
        if os.path.isdir(src):
            continue
        # copy everything; we will overwrite shards we modify
        shutil.copy2(src, dst)

def load_sharded_safetensors(model_dir):
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing index: {idx_path}")
    idx = json.load(open(idx_path, "r"))
    weight_map = idx["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    shard_paths = [os.path.join(model_dir, f) for f in shard_files]
    shards = []
    for p in shard_paths:
        shards.append((p, load_file(p, device="cpu")))
    return idx, shards

def write_shards(out_dir, idx, shards):
    # save each shard tensor dict back, preserving names
    for p, sd in shards:
        out_path = os.path.join(out_dir, os.path.basename(p))
        save_file(sd, out_path)

    # write index json (same mapping)
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f, indent=2, sort_keys=True)

import re
import torch

def extract_basis_v_per_layer(obj, num_layers, d_model, r_v):
    """
    eigen_basis.pt is a dict with keys like:
      base_model.model.model.decoder.layers.{i}.self_attn.v_proj_basis
      base_model.model.model.decoder.layers.{i}.self_attn.q_proj_basis
      base_model.model.model.decoder.layers.{i}.self_attn.k_proj_basis
    We extract v_proj_basis for each layer i -> [d_model, r_v]
    """
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected eigen_basis.pt to be a dict, got {type(obj)}")

    # Find one v key to infer prefix
    # Accept any prefix before ".decoder.layers.{i}.self_attn.v_proj_basis"
    pat = re.compile(r"^(.*)\.decoder\.layers\.(\d+)\.self_attn\.v_proj_basis$")
    matches = []
    for k in obj.keys():
        m = pat.match(k)
        if m:
            prefix = m.group(1)
            layer = int(m.group(2))
            matches.append((prefix, layer, k))

    if not matches:
        # print a helpful hint
        sample = list(obj.keys())[:30]
        raise RuntimeError(
            "Cannot find keys matching '*.decoder.layers.{i}.self_attn.v_proj_basis'. "
            f"Sample keys: {sample}"
        )

    # Choose the most common prefix (robust)
    from collections import Counter
    prefix_counts = Counter([p for p, _, _ in matches])
    best_prefix, _ = prefix_counts.most_common(1)[0]
    print(f"[Basis] detected prefix = {best_prefix}")

    V_list = []
    for i in range(num_layers):
        key = f"{best_prefix}.decoder.layers.{i}.self_attn.v_proj_basis"
        if key not in obj:
            raise KeyError(f"Missing V basis key for layer {i}: {key}")
        V = obj[key]
        if not torch.is_tensor(V):
            raise TypeError(f"Key {key} is not a tensor: {type(V)}")
        V = V.detach().cpu()
        if V.shape != (d_model, r_v):
            raise ValueError(f"Layer {i} V shape {tuple(V.shape)} != ({d_model},{r_v})")
        V_list.append(V)

    return V_list

def main():
    # load config to get layer count
    cfg = AutoConfig.from_pretrained(FINAL_MODEL_DIR)
    num_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    print("[Config] layers=", num_layers, "d_model=", d_model)

    # read rank_v from json (optional sanity)
    lr = json.load(open(LOW_RANK_JSON, "r"))
    r_v = int(lr[0][1])
    print("[LowRank] rank_v=", r_v)

    # load eigen basis
    basis_obj = torch.load(EIGEN_BASIS_PT, map_location="cpu")
    V_list = extract_basis_v_per_layer(basis_obj, num_layers, d_model, r_v)
    print("[Basis] V_list len=", len(V_list), "V[0] shape=", tuple(V_list[0].shape))

    # quick shape checks
    for i in [0, num_layers//2, num_layers-1]:
        v = V_list[i]
        if v.shape != (d_model, r_v):
            raise ValueError(f"Layer {i} V shape {tuple(v.shape)} != ({d_model},{r_v})")

    # load baseline model (CPU) to get Wo, bo per layer
    print("[Baseline] loading baseline OPT from:", BASELINE_DIR)
    base = AutoModelForCausalLM.from_pretrained(BASELINE_DIR, torch_dtype=torch.float16, device_map={"": "cpu"})
    base.eval()

    # prepare output dir and copy skeleton
    print("[Out] copying model dir to:", OUT_DIR)
    if os.path.exists(OUT_DIR):
        raise RuntimeError(f"OUT_DIR exists already: {OUT_DIR} (delete it or change name)")
    copy_hf_skeleton(FINAL_MODEL_DIR, OUT_DIR)

    # load shards of final_model
    idx, shards = load_sharded_safetensors(FINAL_MODEL_DIR)
    weight_map = idx["weight_map"]

    # patch per layer
    patched = 0
    for layer in range(num_layers):
        # key names in your checkpoint
        k_w = f"model.decoder.layers.{layer}.self_attn.out_proj_up.weight"
        k_b = f"model.decoder.layers.{layer}.self_attn.out_proj_up.bias"

        if k_w not in weight_map:
            raise KeyError(f"Key not found in index: {k_w}")
        if k_b not in weight_map:
            raise KeyError(f"Key not found in index: {k_b}")

        # baseline Wo, bo (HF OPT naming)
        Wo = base.model.decoder.layers[layer].self_attn.out_proj.weight.detach().cpu().to(torch.float32)  # [d,d]
        bo = base.model.decoder.layers[layer].self_attn.out_proj.bias.detach().cpu().to(torch.float32)    # [d]

        V  = V_list[layer].to(torch.float32)  # [d,r_v]

        # compute out_proj_up = Wo @ V
        out_w = Wo.matmul(V).contiguous()  # [d, r_v]
        out_b = bo.contiguous()            # [d]

        # write into the correct shard dict
        shard_w = weight_map[k_w]
        shard_b = weight_map[k_b]

        for p, sd in shards:
            bn = os.path.basename(p)
            if bn == shard_w:
                sd[k_w] = out_w.to(sd[k_w].dtype)  # keep checkpoint dtype (usually fp16)
            if bn == shard_b:
                sd[k_b] = out_b.to(sd[k_b].dtype)

        patched += 1
        if layer in [0, num_layers//2, num_layers-1]:
            print(f"[Patch] layer {layer}: out_w {tuple(out_w.shape)} out_b {tuple(out_b.shape)}")

    print("[Patch] patched layers:", patched)

    # write modified shards + index into OUT_DIR
    write_shards(OUT_DIR, idx, shards)

    print("[Done] wrote patched model to:", OUT_DIR)
    print("Next: run sanity on OUT_DIR by editing sanity_load_opt_eigen.py MODEL_DIR or passing env var (see below).")

if __name__ == "__main__":
    main()
