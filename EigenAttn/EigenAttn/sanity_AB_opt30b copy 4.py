import os, json, glob, gzip, argparse, math, random
import torch
import torch.nn.functional as F
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

def q(x: torch.Tensor, p: float) -> float:
    return torch.quantile(x, p).item()

def summarize(x: torch.Tensor):
    x = x.detach().float().flatten().cpu()
    ax = x.abs()
    return {
        "absmax": ax.max().item(),
        "p999_abs": q(ax, 0.999),
        "p99_abs": q(ax, 0.99),
        "std": x.std().item(),
        "mean": x.mean().item(),
    }

def fmt_sum(s: dict):
    return f"absmax={s['absmax']:.6g} p99.9={s['p999_abs']:.6g} std={s['std']:.6g} mean={s['mean']:.6g}"

def load_base_model(path, dtype, device, low_cpu_mem=True):
    return AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem,
        device_map={"": device},
    )

def load_cmp_model(path, dtype, device, low_rank_config, low_cpu_mem=True):
    return OPTForCausalLM_EigenAttn.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem,
        device_map={"": device},
        low_rank_config=low_rank_config,
    )

# -------------------------
# A) multi-prompt logits compare
# -------------------------
@torch.no_grad()
def compare_prompts(base, cmp, tok, prompts, base_device, cmp_device, dtype):
    print("\n[A] Multi-prompt logits alignment (BASE vs CMP)\n")

    for i, text in enumerate(prompts):
        enc = tok(text, return_tensors="pt")
        base_inp = {k: v.to(base_device) for k, v in enc.items()}
        cmp_inp  = {k: v.to(cmp_device)  for k, v in enc.items()}

        with torch.autocast(device_type="cuda", dtype=dtype):
            bout = base(**base_inp)
            cout = cmp(**cmp_inp)

        blogits = bout.logits.detach().float().cpu()
        clogits = cout.logits.detach().float().cpu()

        bs = summarize(blogits)
        cs = summarize(clogits)
        diff = (blogits - clogits).abs()
        ds = summarize(diff)

        ratio_p999 = cs["p999_abs"] / (bs["p999_abs"] + 1e-12)
        ratio_std  = cs["std"] / (bs["std"] + 1e-12)

        print(f"--- Prompt {i} ---")
        print("text:", repr(text[:160] + ("..." if len(text) > 160 else "")))
        print("BASE logits:", fmt_sum(bs))
        print("CMP  logits:", fmt_sum(cs))
        print(f"ratio: p99.9={ratio_p999:.6f} std={ratio_std:.6f}")
        print(f"ABS diff:", fmt_sum(ds))
        print()

# -------------------------
# C4 (local shard) helpers
# -------------------------
def _iter_jsonl_texts_gz(fp, text_key="text", read_size=1 << 20):
    """
    Robust reader for:
      - jsonl.gz (c4-train.*.json.gz)
      - concatenated JSON objects without newlines (your c4-validation.*.json.gz)
      - JSON-seq with 0x1e separators
    Strategy:
      - stream read into buffer
      - parse with JSONDecoder.raw_decode
      - on decode failure not near buffer end: resync to next '{' and continue
    """
    dec = json.JSONDecoder()
    buf = ""

    def skip_seps(s, i, n):
        # include BOM / NUL just in case; keeps train behavior unchanged
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

                # If we are not at a JSON object start, resync forward.
                if buf[i] not in "{[":
                    nxt_obj = buf.find("{", i)
                    if nxt_obj == -1:
                        # can't find any object start in current buffer; drop it
                        buf = ""
                        i = 0
                        break
                    i = nxt_obj

                try:
                    obj, j = dec.raw_decode(buf, i)
                except json.JSONDecodeError as e:
                    # If the error is near the end of buffer, likely incomplete -> read more.
                    if e.pos >= n - 32:
                        break
                    # Otherwise, resync to next '{' and keep going.
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

            # keep unparsed tail only
            buf = buf[i:]

def build_doc_pool_from_c4_train_dir(
    data_dir: str,
    glob_pat: str = "c4-train.*.json.gz",
    pool_docs: int = 20000,
    scan_lines: int = 200000,  # doc-count; keep arg name unchanged
    seed: int = 2,
):
    """
    Stream first `scan_lines` docs (JSON objects) and reservoir-sample `pool_docs`.
    """
    files = sorted(glob.glob(os.path.join(data_dir, glob_pat)))
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pat} under {data_dir}")

    rng = random.Random(seed)
    reservoir = []
    seen = 0

    for fp in files:
        for t in _iter_jsonl_texts_gz(fp, text_key="text"):
            seen += 1
            if len(reservoir) < pool_docs:
                reservoir.append(t)
            else:
                j = rng.randint(1, seen)
                if j <= pool_docs:
                    reservoir[j - 1] = t
            if seen >= scan_lines:
                break
        if seen >= scan_lines:
            break

    if not reservoir:
        raise RuntimeError(f"Failed to read any text from {files[0]}")
    return reservoir, {"seen_docs": seen, "pool_docs": len(reservoir), "file": files[0]}

@torch.no_grad()
def build_utkarsh_like_valenc_from_pool(
    tok,
    doc_pool,
    seqlen: int,
    n_chunks: int = 256,
    val_seed: int = 0,
    max_tries_per_chunk: int = 2000,
):
    rng = random.Random(val_seed)
    valenc = []

    for k in range(n_chunks):
        ok = False
        for _ in range(max_tries_per_chunk):
            text = doc_pool[rng.randint(0, len(doc_pool) - 1)]
            tmp = tok(text, return_tensors="pt")
            L = tmp.input_ids.shape[1]
            if L > seqlen:
                i = rng.randint(0, L - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"Unable to sample a doc with token_len > seqlen after {max_tries_per_chunk} tries. "
                f"Try increasing --c4_scan_lines / --c4_pool_docs."
            )

    return torch.hstack(valenc)

# -------------------------
# Utkarsh-aligned OPT PPL
# -------------------------
@torch.no_grad()
def eval_ppl_utkarsh_opt(
    model,
    testenc: torch.Tensor,
    device: str,
    seqlen: int,
    limit: int = -1,
):
    model.eval()
    use_cache = getattr(model.config, "use_cache", False) if hasattr(model, "config") else False
    if hasattr(model, "config"):
        model.config.use_cache = False

    nsamples = testenc.numel() // seqlen
    loss_fct = torch.nn.CrossEntropyLoss()
    nlls = []

    for i in range(nsamples):
        batch = testenc[:, i*seqlen:(i+1)*seqlen].to(device)

        outputs = model.model.decoder(batch)
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)

        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:].to(model.lm_head.weight.device)

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * seqlen)

        if i == limit:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    if hasattr(model, "config"):
        model.config.use_cache = use_cache

    return ppl.item(), {"nsamples": nsamples, "used_chunks": len(nlls)}

def ppl_sanity_utkarsh_c4_train(base, cmp, tok, args):
    print("\n[B] PPL sanity (Utkarsh-aligned, using your c4-train shard as source)\n")
    if args.data_dir is None or (not os.path.isdir(args.data_dir)):
        print("data_dir not provided or not a directory. Skip PPL.")
        return

    pool, pool_info = build_doc_pool_from_c4_train_dir(
        args.data_dir,
        glob_pat=args.c4_glob,
        pool_docs=args.c4_pool_docs,
        scan_lines=args.c4_scan_lines,
        seed=args.seed,
    )
    print(f"[C4-Pool] {pool_info}")

    testenc = build_utkarsh_like_valenc_from_pool(
        tok,
        pool,
        seqlen=args.seqlen,
        n_chunks=args.c4_val_chunks,
        val_seed=args.c4_val_seed,
    )
    assert testenc.shape[1] == args.c4_val_chunks * args.seqlen, (testenc.shape, args.c4_val_chunks, args.seqlen)

    base_ppl, bi = eval_ppl_utkarsh_opt(base, testenc, args.base_device, args.seqlen, limit=args.limit)
    cmp_ppl,  ci = eval_ppl_utkarsh_opt(cmp,  testenc, args.cmp_device,  args.seqlen, limit=args.limit)

    print(f"BASE PPL: {base_ppl:.6f}  info={bi}")
    print(f"CMP  PPL: {cmp_ppl:.6f}  info={ci}")
    if math.isfinite(base_ppl) and base_ppl > 0 and math.isfinite(cmp_ppl):
        print(f"ratio CMP/BASE: {cmp_ppl/base_ppl:.6f}")
    print()

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok", required=True)

    ap.add_argument("--dtype", default="fp16", choices=["bf16", "fp16"],
                    help="For Utkarsh alignment, fp16 is recommended.")
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=2)

    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--max_prompts", type=int, default=12)

    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=-1,
                    help="Match repo behavior: break if i == limit. Default -1 means never break.")

    ap.add_argument("--c4_glob", default="c4-train.*.json.gz")
    ap.add_argument("--c4_scan_lines", type=int, default=200000,
                    help="How many docs to scan (stream) from gz for building pool.")
    ap.add_argument("--c4_pool_docs", type=int, default=20000,
                    help="Reservoir pool size from scanned docs.")
    ap.add_argument("--c4_val_chunks", type=int, default=256,
                    help="Utkarsh uses 256 chunks for c4 validation.")
    ap.add_argument("--c4_val_seed", type=int, default=0,
                    help="Utkarsh hardcodes random.seed(0) for valenc sampling.")

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

    default_prompts = [
        "Hello, my name is",
        "The quick brown fox jumps over the lazy dog.",
        "Write a short Python function to compute Fibonacci numbers.",
        "Explain what a transformer attention head does in one paragraph.",
        "In 200 words, summarize the purpose of layer normalization in deep nets.",
        "Translate to Chinese: 'Perplexity is a measure of language model quality.'",
        "Given x86-64 assembly, what does 'rdtsc' return and why use 'cpuid' before it?",
        "Consider a matrix A. What does its SVD represent geometrically?",
        "I have a dataset of Airbnb listings; suggest a robust outlier filtering strategy.",
        "Tell me a short story about a robot learning to paint.",
        "Complete the sentence: The capital of California is",
        "Write 5 bullet points about KV-cache compression.",
    ]

    prompts = default_prompts
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, "r", encoding="utf-8", errors="ignore") as f:
            pl = [ln.strip() for ln in f if ln.strip()]
        if pl:
            prompts = pl[:args.max_prompts]

    compare_prompts(base, cmp, tok, prompts[:args.max_prompts], args.base_device, args.cmp_device, dtype)
    ppl_sanity_utkarsh_c4_train(base, cmp, tok, args)
    print("[Done] A+B sanity completed.")

if __name__ == "__main__":
    main()
