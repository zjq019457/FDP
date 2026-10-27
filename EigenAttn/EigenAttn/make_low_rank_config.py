import json, argparse
from typing import Any

def walk(obj: Any, r: int):
    # 只改“rank相关”的 key，避免误改 hidden_size / input_dim
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = k.lower()
            if isinstance(v, int) and ("rank" in lk or lk in ("r", "avg_dim", "avgdim")):
                # 如果原来是 full rank 7168，或者更大，统一截到 r
                out[k] = min(v, r)
            else:
                out[k] = walk(v, r)
        return out
    if isinstance(obj, list):
        return [walk(v, r) for v in obj]
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--rank", type=int, required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.in_path, "r"))
    new_cfg = walk(cfg, args.rank)

    json.dump(new_cfg, open(args.out_path, "w"), indent=2)
    print(f"saved: {args.out_path}")

if __name__ == "__main__":
    main()
