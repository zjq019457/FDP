#!/usr/bin/env python3
import os, json, math, shutil, argparse
from typing import List, Dict

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn


def load_safetensors_sharded(model_dir: str) -> Dict[str, torch.Tensor]:
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json"), "r"))
    weight_map = idx["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    sd = {}
    for sf in shard_files:
        sd.update(load_file(os.path.join(model_dir, sf), device="cpu"))
    return sd


def load_safetensors_shards_by_file(model_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json"), "r"))
    weight_map = idx["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    shards = {}
    for sf in shard_files:
        shards[sf] = load_file(os.path.join(model_dir, sf), device="cpu")
    return idx, shards


def copy_model_skeleton(src: str, dst: str):
    if os.path.exists(dst):
        raise RuntimeError(f"DST exists: {dst}")
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if os.path.isdir(p):
            continue
        shutil.copy2(p, os.path.join(dst, fn))


@torch.no_grad()
def get_text_batches(tok, texts: List[str], seq_len: int, batch_size: int, device: str):
    # simple packing: tokenize each text individually, truncate/pad to seq_len
    # (good enough for calibration)
    batches = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        batches.append((input_ids, attn_mask))
    return batches


@torch.no_grad()
def run_and_collect(
    base_model,
    comp_model,
    tok,
    batches,
    num_layers: int,
    device: str,
):
    """
    Collect per-layer pairs:
      X_i: compressed attn_low before out_proj_up, shape [N, r]
      Y_i: baseline attention output AFTER out_proj (full dim), shape [N, d]
    We'll flatten over batch*seq.
    """
    # containers (lists of chunks)
    X = [[] for _ in range(num_layers)]
    Y = [[] for _ in range(num_layers)]

    # --- hook baseline attention output (after out_proj) ---
    base_cache = [None for _ in range(num_layers)]
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            # HF OPTAttention forward returns (attn_output, attn_weights, past_key_value)
            attn_out = out[0]  # (bsz, tgt, d)
            base_cache[layer_idx] = attn_out.detach()
        return hook_fn

    for i in range(num_layers):
        h = base_model.model.decoder.layers[i].self_attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    # run
    os.environ["EIGEN_SAVE_LOW"] = "1"
    base_model.eval()
    comp_model.eval()

    for (input_ids, attention_mask) in batches:
        # baseline forward
        _ = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # compressed forward
        _ = comp_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # collect for each layer
        for i in range(num_layers):
            # baseline Y: (bsz,tgt,d)
            y = base_cache[i]
            if y is None:
                raise RuntimeError(f"Baseline hook missing layer {i}")
            # compressed X: saved on module: (bsz,tgt,r)
            attn_low = getattr(comp_model.model.decoder.layers[i].self_attn, "_last_attn_low", None)
            if attn_low is None:
                raise RuntimeError(
                    f"Compressed attn_low missing at layer {i}. "
                    "Did you add EIGEN_SAVE_LOW block in OPTEigenAttention.forward?"
                )
            # flatten
            bsz, tgt, d = y.shape
            r = attn_low.shape[-1]
            X[i].append(attn_low.reshape(bsz*tgt, r).float().cpu())
            Y[i].append(y.reshape(bsz*tgt, d).float().cpu())

    # cleanup hooks
    for h in hooks:
        h.remove()

    # concat
    Xcat, Ycat = [], []
    for i in range(num_layers):
        Xcat.append(torch.cat(X[i], dim=0))  # [N, r]
        Ycat.append(torch.cat(Y[i], dim=0))  # [N, d]
    return Xcat, Ycat


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float):
    """
    Fit W,b minimizing ||X W^T + b - Y||^2 + lam||W||^2
    X: [N,r], Y: [N,d]
    returns:
      W: [d,r], b: [d]
    """
    # center to fit bias
    x_mean = X.mean(dim=0, keepdim=True)  # [1,r]
    y_mean = Y.mean(dim=0, keepdim=True)  # [1,d]
    Xc = X - x_mean
    Yc = Y - y_mean

    # W^T = (X^T X + lam I)^-1 X^T Y  => W = Y^T X (X^T X + lam I)^-1
    XtX = Xc.t().matmul(Xc)  # [r,r]
    r = XtX.shape[0]
    XtX = XtX + lam * torch.eye(r)
    XtY = Xc.t().matmul(Yc)  # [r,d]
    # solve (XtX) A = XtY  => A = [r,d]
    A = torch.linalg.solve(XtX, XtY)      # [r,d]
    W = A.t().contiguous()                # [d,r]
    b = (y_mean - x_mean.matmul(W.t())).squeeze(0).contiguous()  # [d]
    return W, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="compressed model dir (safetensors shards)")
    ap.add_argument("--base_dir", required=True, help="baseline OPT-30B dir")
    ap.add_argument("--tok_dir", required=True, help="tokenizer dir")
    ap.add_argument("--low_rank_json", required=True)
    ap.add_argument("--out_dir", required=True, help="output calibrated model dir")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=64, help="number of batches (approx)")
    ap.add_argument("--lam", type=float, default=1e-3, help="ridge lambda")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    print("[Args]", vars(args))

    # load tokenizer
    tok = AutoTokenizer.from_pretrained(args.tok_dir, use_fast=True)

    # make some texts (simple + stable). If you have a file, you can extend this later.
    texts = [
        "Hello world. This is a quick sanity check.",
        "Large language models compress key-value caches to reduce memory.",
        "The quick brown fox jumps over the lazy dog.",
        "In a transformer, attention computes weighted sums of value vectors.",
        "This calibration uses a small dataset to fit a linear map.",
        "We want the compressed model to match the baseline attention output.",
        "OPT is a decoder-only architecture with multi-head self-attention.",
        "Numerical stability matters in fp16 softmax computations.",
    ]
    # repeat to reach steps*batch_size
    texts = (texts * ((args.steps * args.batch_size + len(texts) - 1) // len(texts)))[: args.steps * args.batch_size]
    batches = get_text_batches(tok, texts, args.seq_len, args.batch_size, device)

    # load base model (cpu to save gpu mem; move to cuda if you can afford)
    print("[Baseline] loading:", args.base_dir)
    base = AutoModelForCausalLM.from_pretrained(args.base_dir, torch_dtype=torch.float16, device_map={"": "cpu"})
    base.eval()

    # load compressed model
    low_rank = json.load(open(args.low_rank_json, "r"))
    cfg = AutoConfig.from_pretrained(args.model_dir)
    comp = OPTForCausalLM_EigenAttn(cfg, low_rank_config=low_rank).to(device, dtype=torch.float16)
    sd = load_safetensors_sharded(args.model_dir)
    missing, unexpected = comp.load_state_dict(sd, strict=False)
    print("[Compressed load] missing", len(missing), "unexpected", len(unexpected))
    comp.tie_weights()
    comp.eval()

    L = cfg.num_hidden_layers
    d = cfg.hidden_size
    r_v = int(low_rank[0][1])
    print(f"[Shape] layers={L} d={d} r_v={r_v}")

    # NOTE: baseline currently on CPU; run base forward on CPU, comp on GPU => slow but works.
    # If you have memory, you can put base on GPU too.
    # We'll move input to CPU for base forward separately.
    Xs, Ys = [], []

    # manual loop to keep base+comp inputs on correct device
    os.environ["EIGEN_SAVE_LOW"] = "1"
    Xs = [[] for _ in range(L)]
    Ys = [[] for _ in range(L)]
    base_cache = [None for _ in range(L)]
    hooks = []

    def make_hook(i):
        def hook_fn(module, inp, out):
            base_cache[i] = out[0].detach()  # (bsz,tgt,d)
        return hook_fn

    for i in range(L):
        hooks.append(base.model.decoder.layers[i].self_attn.register_forward_hook(make_hook(i)))

    for (input_ids_gpu, attn_mask_gpu) in batches:
        # BASE on CPU
        input_ids_cpu = input_ids_gpu.to("cpu")
        attn_mask_cpu = attn_mask_gpu.to("cpu")
        _ = base(input_ids=input_ids_cpu, attention_mask=attn_mask_cpu, use_cache=False)

        # COMP on GPU
        _ = comp(input_ids=input_ids_gpu, attention_mask=attn_mask_gpu, use_cache=False)

        for i in range(L):
            y = base_cache[i]
            if y is None:
                raise RuntimeError(f"Baseline hook missing layer {i}")
            x = getattr(comp.model.decoder.layers[i].self_attn, "_last_attn_low", None)
            if x is None:
                raise RuntimeError(f"Compressed _last_attn_low missing layer {i} (EIGEN_SAVE_LOW?)")
            bsz, tgt, _ = y.shape
            Xs[i].append(x.reshape(bsz*tgt, r_v).float().cpu())
            Ys[i].append(y.reshape(bsz*tgt, d).float().cpu())

    for h in hooks:
        h.remove()

    Xcat = [torch.cat(Xs[i], dim=0) for i in range(L)]
    Ycat = [torch.cat(Ys[i], dim=0) for i in range(L)]
    print("[Collect] done. N per layer:", Xcat[0].shape[0])

    # fit per layer W,b
    W_list, b_list = [], []
    for i in range(L):
        W, b = ridge_fit(Xcat[i], Ycat[i], lam=args.lam)
        W_list.append(W)  # [d,r]
        b_list.append(b)  # [d]
        if i in (0, 10, 20, 30, 47):
            print(f"[Fit] layer {i} W_norm={W.norm().item():.3f} b_norm={b.norm().item():.3f}")

    # write new model with patched out_proj_up
    idx, shards = load_safetensors_shards_by_file(args.model_dir)
    weight_map = idx["weight_map"]

    nW = nB = 0
    for i in range(L):
        wkey = f"model.decoder.layers.{i}.self_attn.out_proj_up.weight"
        bkey = f"model.decoder.layers.{i}.self_attn.out_proj_up.bias"
        if wkey not in weight_map or bkey not in weight_map:
            raise KeyError(f"Missing keys in checkpoint for layer {i}: {wkey} / {bkey}")

        sf_w = weight_map[wkey]
        sf_b = weight_map[bkey]

        w_old = shards[sf_w][wkey]
        b_old = shards[sf_b][bkey]

        shards[sf_w][wkey] = W_list[i].to(dtype=w_old.dtype)
        shards[sf_b][bkey] = b_list[i].to(dtype=b_old.dtype)
        nW += 1
        nB += 1

    copy_model_skeleton(args.model_dir, args.out_dir)
    for sf, sd in shards.items():
        save_file(sd, os.path.join(args.out_dir, sf))
    with open(os.path.join(args.out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f, indent=2, sort_keys=True)

    print(f"[Done] wrote calibrated model to: {args.out_dir} (patched {nW} weights, {nB} bias)")


if __name__ == "__main__":
    main()
