#!/usr/bin/env python3
import os, json, argparse
from safetensors.torch import load_file, save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    args = ap.parse_args()

    idx_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    idx = json.load(open(idx_path, "r"))
    wm = idx["weight_map"]

    # only shards that contain out_proj_up weights/bias
    target_prefix = "self_attn.out_proj_up."
    need_shards = set()
    for k, sf in wm.items():
        if target_prefix in k:
            need_shards.add(sf)

    need_shards = sorted(need_shards)
    if not need_shards:
        raise RuntimeError("No out_proj_up keys found in weight_map; check checkpoint format.")

    print("[Fix] rewriting shards with metadata={'format':'pt'}")
    print("[Fix] shards:", need_shards)

    for sf in need_shards:
        fp = os.path.join(args.model_dir, sf)
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        sd = load_file(fp, device="cpu")
        tmp = fp + ".tmp"
        save_file(sd, tmp, metadata={"format": "pt"})
        os.replace(tmp, fp)
        print("fixed:", sf, "tensors:", len(sd))

    print("[Done] metadata fixed. You can retry from_pretrained now.")

if __name__ == "__main__":
    main()
