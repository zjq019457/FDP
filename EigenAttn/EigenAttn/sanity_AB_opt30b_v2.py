#!/usr/bin/env python3
import os, json, glob, gzip, argparse, math, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_base_model(path, dtype, device, low_cpu_mem=True):
    return AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem, device_map={"": device},
    )

def load_cmp_model(path, dtype, device, low_rank_config, low_cpu_mem=True):
    return OPTForCausalLM_EigenAttn.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem, device_map={"": device},
        low_rank_config=low_rank_config,
    )

# -------------------------
# Robust gz JSON reader
# -------------------------
def _iter_json_texts_gz(fp, text_key="text", read_size=1 << 20):
    dec = json.JSONDecoder()
    buf = ""

    def skip_seps(s, i, n):
        while i < n and s[i] in " \t\r\n,\ufeff\x00":
            i += 1
        if i < n and s[i] == "\x1e":
            i += 1
        return i

    with gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(read_size)
            if not chunk:
                break
            buf += chunk
            i = 0
            while True:
                n = len(buf)
                i = skip_seps(buf, i, n)
                if i >= n:
                    break
                if buf[i] not in "{[":
                    nxt = buf.find("{", i)
                    if nxt == -1:
                        buf = ""
                        i = 0
                        break
                    i = nxt
                try:
                    obj, j = dec.raw_decode(buf, i)
                except json.JSONDecodeError as e:
                    if e.pos >= n - 32:
                        break
                    nxt = buf.find("{", i + 1)
                    if nxt == -1:
                        break
                    i = nxt
                    continue
                i = j
                if isinstance(obj, dict):
                    v = obj.get(text_key, None)
                    if isinstance(v, str):
                        t = v.strip()
                        if t:
                            yield t
            buf = buf[i:]

def build_doc_pool(
    data_dir: str,
    glob_pat: str,
    pool_docs: int,
    scan_docs: int,
    seed: int,
    files_limit: int = -1,
):
    files = sorted(glob.glob(os.path.join(data_dir, glob_pat)))
    if files_limit > 0:
        files = files[:files_limit]
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pat} under {data_dir}")

    rng = random.Random(seed)
    reservoir = []
    seen = 0
    for fp in files:
        for t in _iter_json_texts_gz(fp, text_key="text"):
            seen += 1
            if len(reservoir) < pool_docs:
                reservoir.append(t)
            else:
                j = rng.randint(1, seen)
                if j <= pool_docs:
                    reservoir[j - 1] = t
            if seen >= scan_docs:
                break
        if seen >= scan_docs:
            break
    if not reservoir:
        raise RuntimeError(f"Failed to read any text from {files[0]}")
    return reservoir, {"seen_docs": seen, "pool_docs": len(reservoir), "file0": files[0], "nfiles": len(files)}

@torch.no_grad()
def build_valenc_from_pool(tok, doc_pool, seqlen, n_chunks, val_seed, add_special_tokens=False, max_tries_per_chunk=2000):
    rng = random.Random(val_seed)
    chunks = []
    for k in range(n_chunks):
        ok = False
        for _ in range(max_tries_per_chunk):
            text = doc_pool[rng.randint(0, len(doc_pool) - 1)]
            tmp = tok(text, return_tensors="pt", add_special_tokens=add_special_tokens)
            L = tmp.input_ids.shape[1]
            if L > seqlen:
                i = rng.randint(0, L - seqlen - 1)
                chunks.append(tmp.input_ids[:, i:i+seqlen])
                ok = True
                break
        if not ok:
            raise RuntimeError("No doc long enough. Increase scan_docs/pool_docs or reduce seqlen.")
    return torch.hstack(chunks)  # [1, n_chunks*seqlen]

# -------------------------
# PPL eval
# -------------------------
@torch.no_grad()
def _forward_logits(model, input_ids, device, dtype):
    # Use standard forward to avoid decoder/lm_head path differences
    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model(input_ids=input_ids.to(device), use_cache=False)
    return out.logits

@torch.no_grad()
def eval_ppl_disjoint(model, testenc, device, seqlen, dtype):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tok = 0

    nsamples = testenc.numel() // seqlen
    for i in range(nsamples):
        batch = testenc[:, i*seqlen:(i+1)*seqlen]
        logits = _forward_logits(model, batch, device, dtype)  # [1, S, V]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous().to(device)
        nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_nll += float(nll.item())
        total_tok += shift_labels.numel()

    ppl = math.exp(total_nll / max(1, total_tok))
    return ppl, {"windows": nsamples, "tokens": total_tok}

@torch.no_grad()
def eval_ppl_stride(model, testenc, device, seqlen, stride, dtype):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_nll = 0.0
    total_tok = 0

    ids = testenc.to(device)
    T = ids.shape[1]
    # Standard strided eval (GPTQ-style)
    for end in range(stride, T, stride):
        begin = max(0, end - seqlen)
        input_ids = ids[:, begin:end]
        trg_len = end - (end - stride)  # = stride, except last window (handled below)
        target = input_ids.clone()
        # Only count the last `trg_len` tokens (excluding the first token due to shift)
        target[:, :-trg_len] = -100

        logits = _forward_logits(model, input_ids, device, dtype)  # [1, L, V]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target[:, 1:].contiguous()

        nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_nll += float(nll.item())
        total_tok += int((shift_labels != -100).sum().item())

    # tail (if T not multiple of stride)
    if T % stride != 0:
        end = T
        begin = max(0, end - seqlen)
        input_ids = ids[:, begin:end]
        trg_len = end - (T - (T % stride))
        target = input_ids.clone()
        target[:, :-trg_len] = -100

        logits = _forward_logits(model, input_ids, device, dtype)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target[:, 1:].contiguous()
        nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_nll += float(nll.item())
        total_tok += int((shift_labels != -100).sum().item())

    ppl = math.exp(total_nll / max(1, total_tok))
    return ppl, {"stride": stride, "tokens": total_tok}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok", required=True)

    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=2)

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)

    # pool + valenc
    ap.add_argument("--c4_glob", default="c4-validation.*.json.gz")
    ap.add_argument("--c4_scan_docs", type=int, default=800000)
    ap.add_argument("--c4_pool_docs", type=int, default=80000)
    ap.add_argument("--c4_val_chunks", type=int, default=256)
    ap.add_argument("--c4_val_seed", type=int, default=0)
    ap.add_argument("--files_limit", type=int, default=-1)
    ap.add_argument("--add_special_tokens", type=int, default=0)

    # ppl mode
    ap.add_argument("--ppl_mode", default="stride", choices=["stride", "disjoint"])
    ap.add_argument("--stride", type=int, default=512)

    # cache testenc
    ap.add_argument("--save_testenc", default=None)
    ap.add_argument("--load_testenc", default=None)

    args = ap.parse_args()
    set_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    low_rank_config = json.load(open(args.low_json))

    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("[Load] BASE...")
    base = load_base_model(args.base, dtype=dtype, device=args.base_device)
    base.eval()

    print("[Load] CMP...")
    cmp = load_cmp_model(args.cmp, dtype=dtype, device=args.cmp_device, low_rank_config=low_rank_config)
    cmp.eval()

    # build / load testenc
    if args.load_testenc and os.path.exists(args.load_testenc):
        print("[Data] Loading testenc:", args.load_testenc)
        testenc = torch.load(args.load_testenc, map_location="cpu")
    else:
        pool, info = build_doc_pool(
            args.data_dir, args.c4_glob,
            pool_docs=args.c4_pool_docs,
            scan_docs=args.c4_scan_docs,
            seed=args.seed,
            files_limit=args.files_limit,
        )
        print(f"[C4-Pool] {info}")
        testenc = build_valenc_from_pool(
            tok, pool, seqlen=args.seqlen,
            n_chunks=args.c4_val_chunks,
            val_seed=args.c4_val_seed,
            add_special_tokens=bool(args.add_special_tokens),
        )
        if args.save_testenc:
            print("[Data] Saving testenc:", args.save_testenc)
            torch.save(testenc, args.save_testenc)

    assert testenc.shape[1] == args.c4_val_chunks * args.seqlen, testenc.shape

    if args.ppl_mode == "stride":
        base_ppl, bi = eval_ppl_stride(base, testenc, args.base_device, args.seqlen, args.stride, dtype)
        cmp_ppl,  ci = eval_ppl_stride(cmp,  testenc, args.cmp_device,  args.seqlen, args.stride, dtype)
    else:
        base_ppl, bi = eval_ppl_disjoint(base, testenc, args.base_device, args.seqlen, dtype)
        cmp_ppl,  ci = eval_ppl_disjoint(cmp,  testenc, args.cmp_device,  args.seqlen, dtype)

    print(f"BASE PPL: {base_ppl:.6f} info={bi}")
    print(f"CMP  PPL: {cmp_ppl:.6f} info={ci}")
    if math.isfinite(base_ppl) and base_ppl > 0 and math.isfinite(cmp_ppl):
        print(f"ratio CMP/BASE: {cmp_ppl/base_ppl:.6f}")

if __name__ == "__main__":
    main()