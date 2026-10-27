#!/usr/bin/env python3
import os, json, math, argparse, random, shutil, glob, gzip
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

# ---- stable attention path ----
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing index: {idx_path}")
    return json.load(open(idx_path, "r"))

def list_shard_files_from_index(idx: Dict) -> List[str]:
    wm = idx["weight_map"]
    return sorted(set(wm.values()))

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

    reservoir = []
    seen = 0

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
        print(f"[Warn] only sampled {len(reservoir)} texts (<{nsamples}); data_dir may be small.")
    rng.shuffle(reservoir)
    return reservoir[:nsamples]

def build_token_stream_join(tok, texts: List[str]) -> torch.Tensor:
    big = "\n\n".join(texts)
    enc = tok(big, return_tensors="pt")
    return enc.input_ids[0].to(torch.long).cpu()

def make_random_batches_from_stream(
    token_stream: torch.Tensor,
    seq_len: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> List[torch.Tensor]:
    rng = random.Random(seed)
    T = token_stream.numel()
    if T < seq_len + 2:
        raise RuntimeError(f"token_stream too short: T={T} < seq_len={seq_len}")

    batches = []
    max_start = T - seq_len
    for _ in range(steps):
        starts = [rng.randint(0, max_start - 1) for _ in range(batch_size)]
        batch = torch.stack([token_stream[s:s+seq_len] for s in starts], dim=0)  # [bsz, seq]
        batches.append(batch)
    return batches

@torch.no_grad()
def eval_ppl_repo_c4(
    model,
    tok,
    data_dir: str,
    device: str,
    seqlen: int = 2048,
    nsamples_text: int = 256,
    seed: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    loss_fp32: bool = True,
) -> Tuple[float, Dict]:
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    texts = reservoir_sample_texts_from_dir(data_dir, nsamples_text, seed=seed)
    ids = build_token_stream_join(tok, texts)  # [T]
    T = ids.numel()
    n_chunks = T // seqlen
    used = n_chunks * seqlen
    if n_chunks <= 0:
        return float("inf"), {"tokens": T, "chunks": 0, "bad_chunks": 0}

    nll_sum = 0.0
    bad = 0
    for i in range(n_chunks):
        batch = ids[i*seqlen:(i+1)*seqlen].unsqueeze(0).to(device)  # [1,seqlen]
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=batch, use_cache=False)
            logits = out.logits  # [1,seqlen,vocab]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        if loss_fp32:
            shift_logits = shift_logits.float()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        if not torch.isfinite(loss):
            bad += 1
            continue

        # keep consistent with your sanity_AB repo_c4
        nll_sum += float(loss.item()) * seqlen

    denom = max(n_chunks * seqlen, 1)
    ppl = math.exp(nll_sum / denom)
    return ppl, {"tokens": used, "chunks": n_chunks, "bad_chunks": bad}

def get_layer_indices(num_layers: int, train_last_n: int) -> List[int]:
    train_last_n = min(max(train_last_n, 1), num_layers)
    return list(range(num_layers - train_last_n, num_layers))

def patch_and_write_model_shards(
    src_model_dir: str,
    out_dir: str,
    idx: Dict,
    patched_tensors: Dict[str, torch.Tensor],
):
    wm = idx["weight_map"]
    shard_files = list_shard_files_from_index(idx)

    shard2keys: Dict[str, List[str]] = {}
    for k in patched_tensors.keys():
        sf = wm.get(k, None)
        if sf is None:
            raise KeyError(f"Key not in weight_map: {k}")
        shard2keys.setdefault(sf, []).append(k)

    for sf in shard_files:
        src_sf = os.path.join(src_model_dir, sf)
        dst_sf = os.path.join(out_dir, sf)
        keys = shard2keys.get(sf, [])
        if not keys:
            shutil.copy2(src_sf, dst_sf)
            continue

        sd = load_file(src_sf, device="cpu")
        for k in keys:
            old = sd[k]
            new = patched_tensors[k].to(device="cpu", dtype=old.dtype)
            sd[k] = new
        save_file(sd, dst_sf)
        del sd

    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f, indent=2, sort_keys=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--cmp_dir", required=True)
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--low_rank_json", required=True)
    ap.add_argument("--data_dir", required=True)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device", default="cuda:1")

    # calibration
    ap.add_argument("--calib_nsamples", type=int, default=512)
    ap.add_argument("--calib_seed", type=int, default=11)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400)

    ap.add_argument("--train_last_n", type=int, default=8)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=1e-4)

    # eval ppl
    ap.add_argument("--eval_ppl", type=int, default=1)
    ap.add_argument("--eval_nsamples", type=int, default=256)
    ap.add_argument("--eval_seed", type=int, default=2)
    ap.add_argument("--eval_seqlen", type=int, default=2048)
    ap.add_argument("--loss_fp32", type=int, default=1)

    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    set_seed(1234)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    loss_fp32 = bool(args.loss_fp32)

    print("[Args]", vars(args))

    tok = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("[Load] BASE on", args.base_device)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_dir,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": args.base_device},
    )
    base.eval()
    if hasattr(base, "config"):
        base.config.use_cache = False

    low_rank = json.load(open(args.low_rank_json, "r"))

    print("[Load] CMP on", args.cmp_device)
    cmpm = OPTForCausalLM_EigenAttn.from_pretrained(
        args.cmp_dir,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": args.cmp_device},
        low_rank_config=low_rank,
    )
    cmpm.eval()
    if hasattr(cmpm, "config"):
        cmpm.config.use_cache = False

    L = base.config.num_hidden_layers
    d = base.config.hidden_size
    layers = get_layer_indices(L, args.train_last_n)
    print(f"[Model] layers={L} hidden={d} calibrate_layers={layers[0]}..{layers[-1]} (count={len(layers)})")

    # ---- eval before ----
    if args.eval_ppl:
        print("\n[Eval-Before] Repo-C4 PPL")
        bppl, bi = eval_ppl_repo_c4(
            base, tok, args.data_dir, args.base_device,
            seqlen=args.eval_seqlen, nsamples_text=args.eval_nsamples, seed=args.eval_seed,
            dtype=dtype, loss_fp32=loss_fp32
        )
        cppl, ci = eval_ppl_repo_c4(
            cmpm, tok, args.data_dir, args.cmp_device,
            seqlen=args.eval_seqlen, nsamples_text=args.eval_nsamples, seed=args.eval_seed,
            dtype=dtype, loss_fp32=loss_fp32
        )
        print(f"BASE PPL: {bppl:.6f} info={bi}")
        print(f"CMP  PPL: {cppl:.6f} info={ci}")
        if math.isfinite(bppl) and bppl > 0 and math.isfinite(cppl):
            print(f"ratio CMP/BASE: {cppl/bppl:.6f}")

    # ---- calib token stream ----
    print("\n[Calib] Sampling texts + building token stream")
    texts = reservoir_sample_texts_from_dir(args.data_dir, nsamples=args.calib_nsamples, seed=args.calib_seed)
    stream = build_token_stream_join(tok, texts)
    print(f"[Calib] token_stream length = {stream.numel()}")

    batches = make_random_batches_from_stream(
        stream, seq_len=args.seq_len, batch_size=args.batch_size, steps=args.steps, seed=args.calib_seed + 999
    )

    # ---- hooks: BASE self_attn output (Y), CMP out_proj_up input/output (X, Y0) ----
    base_cache: Dict[int, torch.Tensor] = {}
    comp_x: Dict[int, torch.Tensor] = {}
    comp_y0: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_base_hook(i: int):
        def hook_fn(module, inp, out):
            # out[0] = attn_output after out_proj, shape [bsz,seq,d]
            base_cache[i] = out[0].detach()
        return hook_fn

    def make_comp_up_hook(i: int):
        def hook_fn(module, inp, out):
            # inp[0] is the low-dim tensor fed to out_proj_up: [bsz,seq,r]
            comp_x[i] = inp[0].detach()
            # out is out_proj_up(x): [bsz,seq,d]
            comp_y0[i] = out.detach()
        return hook_fn

    for i in layers:
        hooks.append(base.model.decoder.layers[i].self_attn.register_forward_hook(make_base_hook(i)))
        hooks.append(cmpm.model.decoder.layers[i].self_attn.out_proj_up.register_forward_hook(make_comp_up_hook(i)))

    # ---- LoRA residual params per layer ----
    params = []
    A: Dict[int, torch.nn.Parameter] = {}
    B: Dict[int, torch.nn.Parameter] = {}
    db: Dict[int, torch.nn.Parameter] = {}

    for i in layers:
        W = cmpm.model.decoder.layers[i].self_attn.out_proj_up.weight  # [d, r]
        r = W.shape[1]
        k = min(args.lora_rank, r, d)
       # A[i] = torch.nn.Parameter(torch.zeros((k, r), device=args.cmp_device, dtype=torch.float32))
        A[i] = torch.nn.Parameter(1e-4 * torch.randn((k, r), device=args.cmp_device, dtype=torch.float32))
        B[i] = torch.nn.Parameter(torch.zeros((d, k), device=args.cmp_device, dtype=torch.float32))
        #B[i] = torch.nn.Parameter(torch.zeros((d, k), device=args.cmp_device, dtype=torch.float32))
        db[i] = torch.nn.Parameter(torch.zeros((d,), device=args.cmp_device, dtype=torch.float32))
        params.extend([A[i], B[i], db[i]])
        print(f"[Init] layer {i}: r={r} k={k}")

    opt = torch.optim.AdamW(params, lr=args.lr)

    # ---- calib loop ----
    print("\n[Calib] Starting LoRA-style residual fit (hooks for X/Y0, no _last_attn_low needed)")
    for step, batch_ids in enumerate(batches, start=1):
        opt.zero_grad(set_to_none=True)
        base_cache.clear()
        comp_x.clear()
        comp_y0.clear()

        ids0 = batch_ids.to(args.base_device)
        ids1 = batch_ids.to(args.cmp_device)
        attn0 = torch.ones_like(ids0, dtype=torch.long)
        attn1 = torch.ones_like(ids1, dtype=torch.long)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            _ = base(input_ids=ids0, attention_mask=attn0, use_cache=False)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            _ = cmpm(input_ids=ids1, attention_mask=attn1, use_cache=False)

        total = 0.0
        mse_sum = 0.0

        for i in layers:
            if i not in base_cache:
                raise RuntimeError(f"Missing baseline attn_out hook at layer {i}")
            if i not in comp_x or i not in comp_y0:
                raise RuntimeError(
                    f"Missing comp out_proj_up hook at layer {i}. "
                    "Is out_proj_up called in this model forward?"
                )

            y = base_cache[i].to(args.cmp_device).float()   # target [bsz,seq,d]
            x = comp_x[i].float()                           # low-dim [bsz,seq,r]
            y0 = comp_y0[i].float()                         # current up-proj output [bsz,seq,d]

            # delta = (x @ A^T) @ B^T + db
            z = torch.matmul(x, A[i].t())                   # [bsz,seq,k]
            delta = torch.matmul(z, B[i].t()) + db[i].view(1, 1, -1)

            pred = y0 + delta
            mse = (pred - y).pow(2).mean()
            reg = args.lam * (A[i].pow(2).mean() + B[i].pow(2).mean())
            total = total + mse + reg
            mse_sum += float(mse.detach().item())

        total.backward()
        opt.step()

        if step == 1 or (step % args.log_every) == 0:
            print(f"[Calib] step={step}/{args.steps} loss={float(total.detach().item()):.6g} avg_mse={mse_sum/len(layers):.6g}")

    for h in hooks:
        h.remove()

    # ---- merge into out_proj_up ----
    print("\n[Merge] Applying residual into CMP out_proj_up weights/bias")
    patched: Dict[str, torch.Tensor] = {}
    for i in layers:
        lin = cmpm.model.decoder.layers[i].self_attn.out_proj_up
        W0 = lin.weight.data
        b0 = lin.bias.data
        dW = (B[i].detach() @ A[i].detach()).to(device=args.cmp_device, dtype=W0.dtype)
        dbi = db[i].detach().to(device=args.cmp_device, dtype=b0.dtype)

        lin.weight.data = (W0 + dW)
        lin.bias.data = (b0 + dbi)

        wkey = f"model.decoder.layers.{i}.self_attn.out_proj_up.weight"
        bkey = f"model.decoder.layers.{i}.self_attn.out_proj_up.bias"
        patched[wkey] = lin.weight.detach().to("cpu")
        patched[bkey] = lin.bias.detach().to("cpu")

        print(f"[Merge] layer {i}: |dW|_fro={float(dW.float().norm().item()):.3f} |db|={float(dbi.float().norm().item()):.3f}")

    # ---- eval after (in-memory) ----
    if args.eval_ppl:
        print("\n[Eval-After] Repo-C4 PPL (CMP calibrated in-memory)")
        cppl2, ci2 = eval_ppl_repo_c4(
            cmpm, tok, args.data_dir, args.cmp_device,
            seqlen=args.eval_seqlen, nsamples_text=args.eval_nsamples, seed=args.eval_seed,
            dtype=dtype, loss_fp32=loss_fp32
        )
        print(f"CMP(calib) PPL: {cppl2:.6f} info={ci2}")

    # ---- write model ----
    print("\n[Write] Saving calibrated model shards:", args.out_dir)
    ensure_dir_empty(args.out_dir)
    copy_non_safetensors_files(args.cmp_dir, args.out_dir)

    idx = load_index(args.cmp_dir)
    patch_and_write_model_shards(
        src_model_dir=args.cmp_dir,
        out_dir=args.out_dir,
        idx=idx,
        patched_tensors=patched
    )
    print("[Done] wrote calibrated model to:", args.out_dir)

if __name__ == "__main__":
    main()
