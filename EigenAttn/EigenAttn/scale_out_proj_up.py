#!/usr/bin/env python3
import os, json, shutil
import torch
from safetensors.torch import load_file, save_file

SRC = "/diskD/eigen_outputs_opt30b_r448/final_model"
DST = "/diskD/eigen_outputs_opt30b_r448/final_model_scaled_up"
VSCALE = float(os.getenv("VSCALE", "0.1"))  # set VSCALE env

def copy_skeleton(src, dst):
    if os.path.exists(dst):
        raise RuntimeError(f"DST exists: {dst}")
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if os.path.isdir(p):
            continue
        shutil.copy2(p, os.path.join(dst, fn))

def main():
    print("[Scale] VSCALE =", VSCALE)
    copy_skeleton(SRC, DST)

    idx_path = os.path.join(SRC, "model.safetensors.index.json")
    idx = json.load(open(idx_path, "r"))
    weight_map = idx["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    # load shards
    shards = {}
    for sf in shard_files:
        shards[sf] = load_file(os.path.join(SRC, sf), device="cpu")

    # scale all out_proj_up.weight keys
    n = 0
    for k, sf in weight_map.items():
        if k.endswith(".self_attn.out_proj_up.weight"):
            t = shards[sf][k]
            shards[sf][k] = (t.float() * VSCALE).to(t.dtype)
            n += 1
    print("[Scale] scaled keys =", n)

    # save shards
    for sf, sd in shards.items():
        save_file(sd, os.path.join(DST, sf))

    # write index
    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f, indent=2, sort_keys=True)

    print("[Done] wrote:", DST)

if __name__ == "__main__":
    main()
