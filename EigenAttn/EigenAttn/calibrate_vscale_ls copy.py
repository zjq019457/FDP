#!/usr/bin/env python3
import os, json, math, argparse, random, glob, gzip, shutil
from typing import Dict, List

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def set_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _iter_lines(fp: str):
    opener = gzip.open if fp.endswith(".gz") else open
    with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def _extract_text(line: str):
    if line.startswith("{") and line.endswith("}"):
        try:
            j = json.loads(line)
            if isinstance(j, dict):
                t = j.get("text", None) or j.get("content", None)
                if isinstance(t, str) and t.strip():
                    return t.strip()
        except Exception:
            return None
    return line

def reservoir_sample_texts_from_dir(data_dir: str, nsamples: int, seed: int, max_scan_docs: int = 800000) -> List[str]:
    exts = ["*.jsonl", "*.json", "*.gz", "*.txt"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", e), recursive=True))
    files = [fp for fp in files if os.path.isfile(fp)]
    if not files:
        raise FileNotFoundError(f"No shards found under data_dir={data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)

    reservoir, seen = [], 0
    for fp in files:
        try:
            for line in _iter_lines(fp):
                t = _extract_text(line)
                if not isinstance(t, str) or not t.strip():
                    continue
                seen += 1
                if len(reservoir) < nsamples:
                    reservoir.append(t)
                else:
                    j = rng.randint(1, seen)
                    if j <= nsamples:
                        reservoir[j - 1] = t
                if seen >= max_scan_docs:
                    break
        except Exception:
            continue
        if seen >= max_scan_docs:
            break

    if len(reservoir) < nsamples:
        print(f"[Warn] only sampled {len(reservoir)} texts (<{nsamples})")
    rng.shuffle(reservoir)
    return reservoir[:nsamples]

def build_token_stream_join(tok, texts: List[str]) -> torch.Tensor:
    big = "\n\n".join(texts)
    enc = tok(big, return_tensors="pt")
    return enc.input_ids[0].to(torch.long).cpu()

def make_random_batches(stream: torch.Tensor, seq_len: int, batch_size: int, steps: int, seed: int) -> List[torch.Tensor]:
    rng = random.Random(seed)
    T = stream.numel()
    if T < seq_len + 2:
        raise RuntimeError(f"stream too short: T={T} < seq_len={seq_len}")
    max_start = T - seq_len - 1
    batches = []
    for _ in range(steps):
        starts = [rng.randint(0, max_start) for _ in range(batch_size)]
        batches.append(torch.stack([stream[s:s+seq_len] for s in starts], dim=0))
    return batches

def ensure_dir_empty(path: str):
    if os.path.exists(path):
        raise RuntimeError(f"out_dir exists: {path}")
    os.makedirs(path, exist_ok=True)

def copy_non_safetensors_files(src_dir: str, dst_dir: str):
    for fn in os.listdir(src_dir):
        sp = os.path.join(src_dir, fn)
        dp = os.path.join(dst_dir, fn)
        if os.path.isdir(sp): 
            continue
        if fn.endswith(".safetensors"):
            continue
        shutil.copy2(sp, dp)

def load_index(model_dir: str) -> Dict:
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    return json.load(open(idx_path, "r"))

def list_shard_files_from_index(idx: Dict) -> List[str]:
    return sorted(set(idx["weight_map"].values()))

def patch_and_write_shards(src_dir: str, out_dir: str, idx: Dict, patched: Dict[str, torch.Tensor]):
    wm = idx["weight_map"]
    shard_files = list_shard_files_from_index(idx)

    shard2keys: Dict[str, List[str]] = {}
    for k in patched.keys():
        sf = wm.get(k, None)
        if sf is None:
            raise KeyError(f"Key not in weight_map: {k}")
        shard2keys.setdefault(sf, []).append(k)

    for sf in shard_files:
        src_sf = os.path.join(src_dir, sf)
        dst_sf = os.path.join(out_dir, sf)
        keys = shard2keys.get(sf, [])
        if not keys:
            shutil.copy2(src_sf, dst_sf)
            continue
        sd = load_file(src_sf, device="cpu")
        for k in keys:
            old = sd[k]
            sd[k] = patched[k].to("cpu", dtype=old.dtype)
        # 关键：写 metadata，避免 transformers 读取时报 NoneType.get 崩溃
        save_file(sd, dst_sf, metadata={"format": "pt"})
        del sd

    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f, indent=2, sort_keys=True)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--cmp_dir", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device", default="cuda:1")

    ap.add_argument("--calib_nsamples", type=int, default=2048)
    ap.add_argument("--calib_seed", type=int, default=11)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=800)

    ap.add_argument("--clip_min", type=float, default=0.25)
    ap.add_argument("--clip_max", type=float, default=4.0)

    args = ap.parse_args()
    set_seed(args.calib_seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    tok = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    low_rank = json.load(open(args.low_json, "r"))

    print("[Load] BASE", args.base_dir, "->", args.base_device)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_dir, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=True, device_map={"": args.base_device}
    )
    base.eval(); base.config.use_cache = False

    print("[Load] CMP", args.cmp_dir, "->", args.cmp_device)
    cmpm = OPTForCausalLM_EigenAttn.from_pretrained(
        args.cmp_dir, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=True, device_map={"": args.cmp_device},
        low_rank_config=low_rank
    )
    cmpm.eval(); cmpm.config.use_cache = False

    L = base.config.num_hidden_layers
    print("[Model] layers =", L)

    # ---- detect whether vscale_log is log-space (has negatives) or scale-space ----
    vmins = []
    for i in range(L):
        attn = cmpm.model.decoder.layers[i].self_attn
        if hasattr(attn, "vscale_log"):
            v = attn.vscale_log.detach().float()
            vmins.append(float(v.min().item()))
    store_log = (len(vmins)>0 and min(vmins) < 0.0)
    print("[Detect] vscale_log store_log =", store_log, "(min =", (min(vmins) if vmins else None), ")")

    # ---- temporarily set vscale to 1 (or log=0) to measure unscaled y0 ----
    for i in range(L):
        attn = cmpm.model.decoder.layers[i].self_attn
        if hasattr(attn, "vscale_log"):
            if store_log:
                attn.vscale_log.data.zero_()     # log(1)=0
            else:
                attn.vscale_log.data.fill_(1.0)  # scale=1

    # ---- build calib batches ----
    texts = reservoir_sample_texts_from_dir(args.data_dir, args.calib_nsamples, seed=args.calib_seed)
    stream = build_token_stream_join(tok, texts)
    print("[Data] token_stream =", stream.numel())
    batches = make_random_batches(stream, args.seq_len, args.batch_size, args.steps, seed=args.calib_seed+999)

    # ---- hooks: baseline attn_out (after out_proj) and cmp out_proj_up output ----
    base_y: Dict[int, torch.Tensor] = {}
    cmp_y0: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_base_hook(i):
        def fn(module, inp, out):
            base_y[i] = out[0].detach()
        return fn

    def make_cmp_up_hook(i):
        def fn(module, inp, out):
            cmp_y0[i] = out.detach()
        return fn

    for i in range(L):
        hooks.append(base.model.decoder.layers[i].self_attn.register_forward_hook(make_base_hook(i)))
        hooks.append(cmpm.model.decoder.layers[i].self_attn.out_proj_up.register_forward_hook(make_cmp_up_hook(i)))

    # ---- accumulate numerator/denominator per layer ----
    num = [0.0 for _ in range(L)]
    den = [0.0 for _ in range(L)]
    eps = 1e-12

    for step, ids in enumerate(batches, start=1):
        base_y.clear(); cmp_y0.clear()
        ids0 = ids.to(args.base_device)
        ids1 = ids.to(args.cmp_device)
        attn0 = torch.ones_like(ids0, dtype=torch.long)
        attn1 = torch.ones_like(ids1, dtype=torch.long)

        with torch.autocast(device_type="cuda", dtype=dtype):
            _ = base(input_ids=ids0, attention_mask=attn0, use_cache=False)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _ = cmpm(input_ids=ids1, attention_mask=attn1, use_cache=False)

        for i in range(L):
            yb = base_y[i].to(args.cmp_device).float()      # [B,S,D]
            y0 = cmp_y0[i].float()                          # [B,S,D]
            b = cmpm.model.decoder.layers[i].self_attn.out_proj_up.bias.float().view(1,1,-1)

            a = (y0 - b).reshape(-1)                         # vector
            t = (yb - b).reshape(-1)
            num[i] += float((a*t).sum().item())
            den[i] += float((a*a).sum().item())

        if step % 50 == 0:
            print(f"[Calib] step {step}/{args.steps}")

    for h in hooks:
        h.remove()

    # ---- compute s and write back to vscale_log ----
    s_list = []
    for i in range(L):
        s = num[i] / (den[i] + eps)
        s = max(args.clip_min, min(args.clip_max, s))
        s_list.append(s)
        attn = cmpm.model.decoder.layers[i].self_attn
        if hasattr(attn, "vscale_log"):
            if store_log:
                attn.vscale_log.data.fill_(math.log(max(s, 1e-8)))
            else:
                attn.vscale_log.data.fill_(float(s))

    print("[Calib] vscale stats:",
          "min", min(s_list), "max", max(s_list), "mean", sum(s_list)/len(s_list))

    # ---- patch checkpoint (only vscale keys) ----
    idx = load_index(args.cmp_dir)
    wm_keys = list(idx["weight_map"].keys())
    vscale_keys = [k for k in wm_keys if "vscale_log" in k]

    if not vscale_keys:
        raise RuntimeError("No vscale_log keys found in checkpoint index. Can't patch safely.")

    patched = {}
    sd = cmpm.state_dict()
    for k in vscale_keys:
        patched[k] = sd[k].detach().cpu()

    print("[Write] patch keys:", len(patched), "->", args.out_dir)
    ensure_dir_empty(args.out_dir)
    copy_non_safetensors_files(args.cmp_dir, args.out_dir)
    patch_and_write_shards(args.cmp_dir, args.out_dir, idx, patched)
    print("[Done] wrote calibrated model to:", args.out_dir)

if __name__ == "__main__":
    main()
