#!/usr/bin/env python3
import os, json, math, argparse, random, glob, gzip, shutil
from typing import Dict, List, Tuple, Optional

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

# ---- make attention path stable ----
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# -------------------------
# utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _parse_cuda_index(dev: str) -> Optional[int]:
    if not isinstance(dev, str):
        return None
    dev = dev.strip()
    if dev.startswith("cuda:"):
        try:
            return int(dev.split(":")[1])
        except Exception:
            return None
    return None

def _validate_cuda_devices(base_device: str, cmp_device: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    n = torch.cuda.device_count()
    bi = _parse_cuda_index(base_device)
    ci = _parse_cuda_index(cmp_device)
    if bi is None or ci is None:
        raise RuntimeError(f"Device must be like 'cuda:0'. Got base_device={base_device}, cmp_device={cmp_device}")
    if bi < 0 or bi >= n:
        raise RuntimeError(f"Invalid base_device={base_device}. Visible cuda device_count={n}. "
                           f"Check CUDA_VISIBLE_DEVICES.")
    if ci < 0 or ci >= n:
        raise RuntimeError(f"Invalid cmp_device={cmp_device}. Visible cuda device_count={n}. "
                           f"Check CUDA_VISIBLE_DEVICES.")

def _iter_lines(fp: str):
    opener = gzip.open if fp.endswith(".gz") else open
    with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def _extract_text(line: str) -> Optional[str]:
    if line.startswith("{") and line.endswith("}"):
        try:
            j = json.loads(line)
            if isinstance(j, dict):
                t = j.get("text", None) or j.get("content", None)
                if isinstance(t, str) and t.strip():
                    return t.strip()
        except Exception:
            return None
    return line if isinstance(line, str) and line.strip() else None

def reservoir_sample_texts_from_dir(
    data_dir: str,
    nsamples: int,
    seed: int,
    max_scan_docs: int = 800000,
) -> List[str]:
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
                if not t:
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

def build_token_stream_concat(tok, texts: List[str], add_sep: bool = True) -> torch.Tensor:
    sep_ids = tok("\n\n", add_special_tokens=False).input_ids if add_sep else []
    all_ids: List[int] = []
    for i, t in enumerate(texts):
        ids = tok(t, add_special_tokens=False).input_ids
        if not ids:
            continue
        all_ids.extend(ids)
        if add_sep and i != len(texts) - 1 and sep_ids:
            all_ids.extend(sep_ids)
    if len(all_ids) == 0:
        raise RuntimeError("Empty token stream after tokenization.")
    return torch.tensor(all_ids, dtype=torch.long)

def make_random_batches(stream: torch.Tensor, seq_len: int, batch_size: int, steps: int, seed: int) -> List[torch.Tensor]:
    rng = random.Random(seed)
    T = int(stream.numel())
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

        save_file(sd, dst_sf, metadata={"format": "pt"})
        del sd

    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f, indent=2, sort_keys=True)

def _fill_like_(t: torch.Tensor, val: float):
    t.data.fill_(float(val))

def _set_vscale_identity(attn, store_log: bool):
    if not hasattr(attn, "vscale_log"):
        return
    if store_log:
        _fill_like_(attn.vscale_log, 0.0)   # log(1)=0
    else:
        _fill_like_(attn.vscale_log, 1.0)   # scale=1

def _set_vscale_value(attn, store_log: bool, s: float):
    if not hasattr(attn, "vscale_log"):
        return
    if store_log:
        _fill_like_(attn.vscale_log, math.log(max(s, 1e-12)))
    else:
        _fill_like_(attn.vscale_log, float(s))

@torch.no_grad()
def probe_vscale_semantics(
    cmpm,
    stream: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    probe_layers: List[int],
    seq_len: int,
    steps: int,
    seed: int,
    val_a: float = 1.0,
    val_b: float = 2.0,
) -> Tuple[bool, float]:
    L = cmpm.config.num_hidden_layers
    probe_layers = [i for i in probe_layers if 0 <= i < L]
    if not probe_layers:
        probe_layers = [0, L // 2, L - 1]

    batches = make_random_batches(stream.cpu(), seq_len=seq_len, batch_size=1, steps=steps, seed=seed)

    ratios = []
    for li in probe_layers:
        attn = cmpm.model.decoder.layers[li].self_attn
        if not hasattr(attn, "vscale_log"):
            continue

        saved = []
        for i in range(L):
            a = cmpm.model.decoder.layers[i].self_attn
            if hasattr(a, "vscale_log"):
                saved.append((i, a.vscale_log.detach().clone()))

        y_cap = {}
        def hook_fn(mod, inp, out):
            y_cap["y"] = out.detach()

        h = attn.out_proj_up.register_forward_hook(hook_fn)

        _fill_like_(attn.vscale_log, val_a)
        ids = batches[0]
        y_cap.clear()
        ids1 = ids.to(device)
        attn1 = torch.ones_like(ids1, dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _ = cmpm(input_ids=ids1, attention_mask=attn1, use_cache=False)
        yA = y_cap["y"]

        _fill_like_(attn.vscale_log, val_b)
        y_cap.clear()
        with torch.autocast(device_type="cuda", dtype=dtype):
            _ = cmpm(input_ids=ids1, attention_mask=attn1, use_cache=False)
        yB = y_cap["y"]

        b = attn.out_proj_up.bias
        if b is not None:
            bc = b.float().view(1, 1, -1)
            aA = (yA.float() - bc).reshape(-1)
            aB = (yB.float() - bc).reshape(-1)
        else:
            aA = yA.float().reshape(-1)
            aB = yB.float().reshape(-1)

        nA = float(aA.norm().item()) + 1e-12
        nB = float(aB.norm().item()) + 1e-12
        ratios.append(nB / nA)

        h.remove()
        for i, v in saved:
            a = cmpm.model.decoder.layers[i].self_attn
            if hasattr(a, "vscale_log"):
                a.vscale_log.data.copy_(v.to(a.vscale_log.device))

    if not ratios:
        return False, float("nan")

    r_mean = sum(ratios) / len(ratios)
    target_lin = 2.0
    target_exp = math.e  # e^(2-1)
    store_log = abs(r_mean - target_exp) < abs(r_mean - target_lin)
    return store_log, r_mean

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--cmp_dir", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device", default="cuda:1")

    ap.add_argument("--calib_nsamples", type=int, default=4096)
    ap.add_argument("--calib_seed", type=int, default=11)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--max_scan_docs", type=int, default=800000)

    ap.add_argument("--clip_min", type=float, default=0.05)
    ap.add_argument("--clip_max", type=float, default=8.0)

    ap.add_argument("--detect", default="probe", choices=["probe", "force_log", "force_linear"])
    ap.add_argument("--probe_layers", default="0,24,47")
    ap.add_argument("--probe_seq_len", type=int, default=256)
    ap.add_argument("--probe_steps", type=int, default=8)
    ap.add_argument("--probe_val_a", type=float, default=1.0)
    ap.add_argument("--probe_val_b", type=float, default=2.0)

    args = ap.parse_args()
    set_seed(args.calib_seed)

    _validate_cuda_devices(args.base_device, args.cmp_device)

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
    base.eval()
    base.config.use_cache = False

    print("[Load] CMP", args.cmp_dir, "->", args.cmp_device)
    cmpm = OPTForCausalLM_EigenAttn.from_pretrained(
        args.cmp_dir, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=True, device_map={"": args.cmp_device},
        low_rank_config=low_rank
    )
    cmpm.eval()
    cmpm.config.use_cache = False

    L = base.config.num_hidden_layers
    print("[Model] layers =", L)

    # ---- build calib batches ----
    texts = reservoir_sample_texts_from_dir(
        args.data_dir, args.calib_nsamples,
        seed=args.calib_seed, max_scan_docs=args.max_scan_docs
    )
    stream = build_token_stream_concat(tok, texts, add_sep=True)
    print("[Data] token_stream =", stream.numel())

    # ---- detect vscale semantics ----
    if args.detect == "force_log":
        store_log = True
        print("[Detect] forced store_log=True (write log(s), runtime uses exp)")
    elif args.detect == "force_linear":
        store_log = False
        print("[Detect] forced store_log=False (write s, runtime uses linear)")
    else:
        probe_layers = [int(x) for x in args.probe_layers.split(",") if x.strip()]
        store_log, r_mean = probe_vscale_semantics(
            cmpm=cmpm, stream=stream,
            device=args.cmp_device, dtype=dtype,
            probe_layers=probe_layers,
            seq_len=args.probe_seq_len,
            steps=args.probe_steps,
            seed=args.calib_seed + 123,
            val_a=args.probe_val_a,
            val_b=args.probe_val_b,
        )
        print(f"[Detect][probe] ratio_mean={r_mean:.6g} -> store_log={store_log} "
              f"(~2 means linear, ~{math.e:.6g} means exp/log)")

    # ---- set all vscale to identity (unscaled) ----
    for i in range(L):
        _set_vscale_identity(cmpm.model.decoder.layers[i].self_attn, store_log=store_log)

    # ---- cache biases on cmp_device to avoid device mismatch + reduce overhead ----
    base_bias = [None] * L
    cmp_bias  = [None] * L
    for i in range(L):
        b0 = getattr(base.model.decoder.layers[i].self_attn.out_proj, "bias", None)
        if b0 is not None:
            base_bias[i] = b0.detach().to(args.cmp_device).float().view(1, 1, -1)
        b1 = getattr(cmpm.model.decoder.layers[i].self_attn.out_proj_up, "bias", None)
        if b1 is not None:
            cmp_bias[i] = b1.detach().to(args.cmp_device).float().view(1, 1, -1)

    # ---- random batches ----
    batches = make_random_batches(stream.cpu(), args.seq_len, args.batch_size, args.steps, seed=args.calib_seed + 999)
    print(f"[Data] batches={len(batches)} seq_len={args.seq_len} batch_size={args.batch_size}")

    # ---- hooks ----
    base_y: Dict[int, torch.Tensor] = {}
    cmp_y0: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_base_hook(i):
        def fn(module, inp, out):
            if isinstance(out, (tuple, list)):
                base_y[i] = out[0].detach()
            else:
                base_y[i] = out.detach()
        return fn

    def make_cmp_up_hook(i):
        def fn(module, inp, out):
            cmp_y0[i] = out.detach()
        return fn

    for i in range(L):
        hooks.append(base.model.decoder.layers[i].self_attn.register_forward_hook(make_base_hook(i)))
        hooks.append(cmpm.model.decoder.layers[i].self_attn.out_proj_up.register_forward_hook(make_cmp_up_hook(i)))

    # ---- accumulate ----
    eps = 1e-12
    num = [0.0 for _ in range(L)]
    den = [0.0 for _ in range(L)]

    for step, ids in enumerate(batches, start=1):
        base_y.clear()
        cmp_y0.clear()

        ids0 = ids.to(args.base_device)
        ids1 = ids.to(args.cmp_device)

        attn0 = torch.ones_like(ids0, dtype=torch.long, device=args.base_device)
        attn1 = torch.ones_like(ids1, dtype=torch.long, device=args.cmp_device)

        with torch.autocast(device_type="cuda", dtype=dtype):
            _ = base(input_ids=ids0, attention_mask=attn0, use_cache=False)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _ = cmpm(input_ids=ids1, attention_mask=attn1, use_cache=False)

        for i in range(L):
            yb = base_y[i].to(args.cmp_device).float()
            y0 = cmp_y0[i].float()

            tb = base_bias[i]
            bc = cmp_bias[i]

            t = (yb - tb).reshape(-1) if tb is not None else yb.reshape(-1)
            a = (y0 - bc).reshape(-1) if bc is not None else y0.reshape(-1)

            nn = float((a * t).sum().item())
            dd = float((a * a).sum().item())

            num[i] += nn
            den[i] += dd

        if step % 50 == 0:
            print(f"[Calib] step {step}/{args.steps}")

    for h in hooks:
        h.remove()

    # ---- compute s + clip ----
    s_list = []
    for i in range(L):
        s = num[i] / (den[i] + eps)
        s = max(args.clip_min, min(args.clip_max, s))
        s_list.append(s)

    print("[Calib] vscale stats:",
          "min", min(s_list), "max", max(s_list), "mean", sum(s_list) / len(s_list))
    n_clipmin = sum(1 for s in s_list if abs(s - args.clip_min) < 1e-12)
    n_clipmax = sum(1 for s in s_list if abs(s - args.clip_max) < 1e-12)
    print(f"[Calib] clipped: at_min={n_clipmin}/{L}, at_max={n_clipmax}/{L}, clip_min={args.clip_min}, clip_max={args.clip_max}")

    # ---- write back vscale_log ----
    for i in range(L):
        _set_vscale_value(cmpm.model.decoder.layers[i].self_attn, store_log=store_log, s=s_list[i])

    # ---- patch checkpoint ----
    idx = load_index(args.cmp_dir)
    wm_keys = list(idx["weight_map"].keys())
    vscale_keys = [k for k in wm_keys if "vscale_log" in k]
    if not vscale_keys:
        raise RuntimeError("No vscale_log keys found in checkpoint index. Can't patch safely.")

    sd = cmpm.state_dict()
    patched = {k: sd[k].detach().cpu() for k in vscale_keys}

    print("[Write] patch keys:", len(patched), "->", args.out_dir)
    ensure_dir_empty(args.out_dir)
    copy_non_safetensors_files(args.cmp_dir, args.out_dir)
    patch_and_write_shards(args.cmp_dir, args.out_dir, idx, patched)
    print("[Done] wrote calibrated model to:", args.out_dir)

if __name__ == "__main__":
    main()