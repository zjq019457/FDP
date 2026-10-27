#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
from typing import Dict, List, Tuple

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_index_file(model_dir: str) -> Tuple[str, str]:
    """
    返回 (index_path, format_name)
    支持：
      - pytorch_model.bin.index.json
      - model.safetensors.index.json
    """
    p_bin = os.path.join(model_dir, "pytorch_model.bin.index.json")
    p_st  = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(p_st):
        return p_st, "safetensors"
    if os.path.exists(p_bin):
        return p_bin, "pytorch_bin"
    return "", "unknown"

def load_weight_keys(model_dir: str) -> Dict:
    idx_path, fmt = find_index_file(model_dir)
    if not idx_path:
        return {
            "format": fmt,
            "index_path": None,
            "num_keys": 0,
            "shards": [],
            "keys": [],
        }
    j = read_json(idx_path)
    weight_map = j.get("weight_map", {})
    keys = sorted(weight_map.keys())
    shards = sorted(set(weight_map.values()))
    return {
        "format": fmt,
        "index_path": idx_path,
        "num_keys": len(keys),
        "shards": shards,
        "keys": keys,
    }

def config_summary(cfg: Dict) -> Dict:
    if not cfg:
        return {}
    keys = [
        "model_type", "architectures",
        "hidden_size", "ffn_dim",
        "num_hidden_layers", "num_attention_heads",
        "vocab_size", "max_position_embeddings",
        "tie_word_embeddings", "torch_dtype",
    ]
    return {k: cfg.get(k) for k in keys if k in cfg}

def diff_key_sets(keys_a: List[str], keys_b: List[str]) -> Dict:
    A, B = set(keys_a), set(keys_b)
    only_a = sorted(A - B)
    only_b = sorted(B - A)
    common = sorted(A & B)
    return {
        "only_in_A": only_a,
        "only_in_B": only_b,
        "common": common,
        "only_in_A_count": len(only_a),
        "only_in_B_count": len(only_b),
        "common_count": len(common),
    }

def get_opt_layers(model):
    """
    OPTForCausalLM 通常：model.model.decoder.layers
    这里写得稍微鲁棒一点。
    """
    cand_paths = [
        ("model.model.decoder.layers", lambda m: m.model.decoder.layers),
        ("model.decoder.layers",       lambda m: m.model.decoder.layers),
        ("decoder.layers",             lambda m: m.decoder.layers),
    ]
    for name, fn in cand_paths:
        try:
            layers = fn(model)
            _ = len(layers)
            return name, layers
        except Exception:
            pass
    raise RuntimeError("Cannot locate decoder layers on this model object.")

def describe_module_params(mod, max_params=30):
    """
    打印一个模块的 direct parameters（不递归子模块太深，避免输出爆炸）
    """
    out = []
    for n, p in list(mod.named_parameters(recurse=False))[:max_params]:
        out.append({"name": n, "shape": list(p.shape), "dtype": str(p.dtype)})
    return out

def inspect_arch(model_dir: str, mode: str, n_layers: int, trust_remote_code: bool):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

    if mode == "meta":
        # 只建空模型，不加载权重
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    elif mode == "load":
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
    else:
        raise ValueError("mode must be meta or load")

    layers_path, layers = get_opt_layers(model)
    L = min(n_layers, len(layers))

    layers_info = []
    for i in range(L):
        layer = layers[i]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            layers_info.append({"layer": i, "error": "no self_attn"})
            continue

        def pick(obj, name):
            return getattr(obj, name, None)

        q = pick(attn, "q_proj")
        k = pick(attn, "k_proj")
        v = pick(attn, "v_proj")
        o = pick(attn, "out_proj")

        info = {
            "layer": i,
            "self_attn_type": type(attn).__name__,
            "q_proj_type": type(q).__name__ if q is not None else None,
            "k_proj_type": type(k).__name__ if k is not None else None,
            "v_proj_type": type(v).__name__ if v is not None else None,
            "out_proj_type": type(o).__name__ if o is not None else None,
            "q_proj_params": describe_module_params(q) if q is not None else [],
            "k_proj_params": describe_module_params(k) if k is not None else [],
            "v_proj_params": describe_module_params(v) if v is not None else [],
            "out_proj_params": describe_module_params(o) if o is not None else [],
        }
        layers_info.append(info)

    return {
        "model_dir": model_dir,
        "mode": mode,
        "layers_path": layers_path,
        "config_summary": config_summary(cfg.to_dict()),
        "layers_info": layers_info,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline model dir (e.g. /diskE/opt-30b)")
    ap.add_argument("--comp", required=True, help="compressed model dir")
    ap.add_argument("--mode", default="meta", choices=["meta", "load"], help="meta=不加载权重; load=加载权重")
    ap.add_argument("--layers", type=int, default=3, help="inspect first N layers")
    ap.add_argument("--out", default="arch_report.json", help="output json")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    # 1) config + index keys
    cfg_base = read_json(os.path.join(args.base, "config.json")) if os.path.exists(os.path.join(args.base, "config.json")) else {}
    cfg_comp = read_json(os.path.join(args.comp, "config.json")) if os.path.exists(os.path.join(args.comp, "config.json")) else {}
    idx_base = load_weight_keys(args.base)
    idx_comp = load_weight_keys(args.comp)
    key_diff = diff_key_sets(idx_base["keys"], idx_comp["keys"])

    # 2) arch inspect
    arch_base = inspect_arch(args.base, args.mode, args.layers, args.trust_remote_code)
    arch_comp = inspect_arch(args.comp, args.mode, args.layers, args.trust_remote_code)

    report = {
        "base": {
            "dir": args.base,
            "config_summary": config_summary(cfg_base),
            "weights": {k: idx_base[k] for k in ["format","index_path","num_keys","shards"]},
        },
        "comp": {
            "dir": args.comp,
            "config_summary": config_summary(cfg_comp),
            "weights": {k: idx_comp[k] for k in ["format","index_path","num_keys","shards"]},
        },
        "key_diff": {
            "only_in_base_count": key_diff["only_in_A_count"],
            "only_in_comp_count": key_diff["only_in_B_count"],
            "only_in_base_examples": key_diff["only_in_A"][:200],
            "only_in_comp_examples": key_diff["only_in_B"][:200],
        },
        "arch_base": arch_base,
        "arch_comp": arch_comp,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 终端简要输出
    print("=== WEIGHTS FORMAT ===")
    print("BASE:", report["base"]["weights"])
    print("COMP:", report["comp"]["weights"])
    print("\n=== KEY DIFF ===")
    print("only_in_base:", report["key_diff"]["only_in_base_count"])
    print("only_in_comp:", report["key_diff"]["only_in_comp_count"])
    print("\n=== FIRST LAYERS (types) ===")
    for side, arch in [("BASE", arch_base), ("COMP", arch_comp)]:
        print(f"\n[{side}] layers_path = {arch['layers_path']}")
        for li in arch["layers_info"]:
            if "error" in li:
                print(f"layer {li['layer']:02d}: {li['error']}")
            else:
                print(
                    f"layer {li['layer']:02d}: attn={li['self_attn_type']}  "
                    f"q={li['q_proj_type']} k={li['k_proj_type']} v={li['v_proj_type']} o={li['out_proj_type']}"
                )

    print(f"\nWrote report: {args.out}")

if __name__ == "__main__":
    main()
