import os, json, glob, gzip, argparse, math, random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

# ---- make attention path stable (avoid flash/mem_efficient surprises) ----
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
# local C4-ish reader: json/jsonl/gz shards
# -------------------------
def _iter_lines(fp):
    opener = gzip.open if fp.endswith(".gz") else open
    with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def _extract_text_from_line(line: str):
    # common C4 jsonl: {"text": "...", ...}
    if line.startswith("{") and line.endswith("}"):
        try:
            j = json.loads(line)
            if isinstance(j, dict):
                t = j.get("text", None) or j.get("content", None)
                if isinstance(t, str):
                    return t
        except Exception:
            return None
    return line

def sample_texts_from_c4_dir(data_dir, nsamples=256, seed=2, max_scan_docs=2000000):
    """
    Try to mimic HF datasets shuffle+select(256) roughly:
    - scan a lot of docs, do reservoir sampling to get ~uniform sample
    - then return nsamples texts
    """
    exts = ["*.jsonl", "*.json", "*.gz", "*.txt"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", e), recursive=True))
    files = [fp for fp in files if os.path.isfile(fp)]
    if not files:
        raise FileNotFoundError(f"No readable shards found under data_dir={data_dir}")

    rng = random.Random(seed)
    rng.shuffle(files)

    reservoir = []
    seen = 0

    for fp in files:
        try:
            for line in _iter_lines(fp):
                t = _extract_text_from_line(line)
                if not isinstance(t, str):
                    continue
                t = t.strip()
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
        # still return what we have; better than failing
        print(f"[Warn] Only sampled {len(reservoir)} texts (<{nsamples}). data_dir may be small.")
    rng.shuffle(reservoir)
    return reservoir[:nsamples]

# -------------------------
# repo-aligned C4 PPL
# -------------------------
@torch.no_grad()
def eval_ppl_repo_c4(model, tok, data_dir, device,
                    seqlen=2048, nsamples_text=256, seed=2,
                    dtype=torch.bfloat16, loss_fp32=True):
    """
    Align to typical repo eval loop:
    - sample 256 texts (c4)
    - join with "\\n\\n"
    - tokenize once => testenc
    - nsamples = numel // seqlen (drop tail)
    - for each chunk: forward, shift, CE(mean), nll += loss * seqlen
    - ppl = exp(sum(nll) / (nsamples * seqlen))
    """
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    texts = sample_texts_from_c4_dir(data_dir, nsamples=nsamples_text, seed=seed)
    big = "\n\n".join(texts)

    # NOTE: to mimic many repos, keep tokenizer default add_special_tokens=True
    enc = tok(big, return_tensors="pt")
    testenc = enc.input_ids  # shape [1, T]
    T = testenc.numel()
    if T < seqlen + 2:
        return float("inf"), {"tokens": T, "chunks": 0}

    n_chunks = T // seqlen
    used_tokens = n_chunks * seqlen

    nll_sum = 0.0
    bad_chunks = 0

    for i in range(n_chunks):
        batch = testenc[:, i*seqlen:(i+1)*seqlen].to(device)

        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=batch)
            logits = out.logits  # [1, seqlen, vocab]

        # shift like repo
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        if loss_fp32:
            shift_logits = shift_logits.float()

        # CE over (seqlen-1) tokens, reduction='mean'
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean"
        )

        if not torch.isfinite(loss):
            bad_chunks += 1
            # keep going but this indicates model instability
            continue

        # repo-style scaling: multiply by seqlen (not seqlen-1)
        nll_sum += loss.detach().item() * seqlen

    denom = max(n_chunks * seqlen, 1)
    ppl = math.exp(nll_sum / denom)
    info = {"tokens": used_tokens, "chunks": n_chunks, "bad_chunks": bad_chunks}
    return ppl, info

# -------------------------
# legacy stream PPL (kept for debug)
# -------------------------
def iter_texts_from_dir(data_dir, max_files=50, max_lines=200000):
    exts = ["*.txt", "*.jsonl", "*.json", "*.gz"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", e), recursive=True))
    files = sorted(files)[:max_files]
    if not files:
        return
    n = 0
    for fp in files:
        opener = gzip.open if fp.endswith(".gz") else open
        try:
            with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
                if fp.endswith(".json") and (not fp.endswith(".json.gz")):
                    try:
                        obj = json.load(f)
                        if isinstance(obj, list):
                            for it in obj:
                                if isinstance(it, dict):
                                    t = it.get("text", None) or it.get("content", None)
                                    if isinstance(t, str) and t.strip():
                                        yield t.strip(); n += 1
                                        if n >= max_lines: return
                                elif isinstance(it, str) and it.strip():
                                    yield it.strip(); n += 1
                                    if n >= max_lines: return
                        elif isinstance(obj, dict):
                            for k in ["text", "content"]:
                                t = obj.get(k, None)
                                if isinstance(t, str) and t.strip():
                                    yield t.strip(); n += 1
                                    if n >= max_lines: return
                    except Exception:
                        pass
                    continue

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = None
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            j = json.loads(line)
                            if isinstance(j, dict):
                                t = j.get("text", None) or j.get("content", None)
                        except Exception:
                            t = None
                    if t is None:
                        t = line
                    if isinstance(t, str) and t.strip():
                        yield t.strip()
                        n += 1
                        if n >= max_lines:
                            return
        except Exception:
            continue

def build_token_stream(tok, texts_iter, target_tokens=500000, seed=2):
    rng = random.Random(seed)
    ids = []
    total = 0
    for t in texts_iter:
        enc = tok(t, add_special_tokens=False)["input_ids"]
        if not enc:
            continue
        # legacy mode: keep eos insertion (debug-only)
        ids.extend(enc + [tok.eos_token_id])
        total += len(enc) + 1
        if total >= target_tokens:
            break
    if total == 0:
        return None
    return torch.tensor(ids, dtype=torch.long)

@torch.no_grad()
def eval_ppl_on_stream(model, token_stream, device, seqlen=2048, stride=2048, dtype=torch.bfloat16):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    nll_sum = 0.0
    tok_count = 0
    N = token_stream.numel()
    if N < 2:
        return float("inf")

    for start in range(0, N, stride):
        end = min(start + seqlen, N)
        if end - start < 2:
            break
        inp = token_stream[start:end].unsqueeze(0).to(device)
        attn = torch.ones_like(inp, dtype=torch.long)
        labels = inp.clone()

        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=inp, attention_mask=attn, labels=labels)
            loss = out.loss

        chunk_tokens = inp.numel() - 1
        nll_sum += loss.detach().float().item() * chunk_tokens
        tok_count += chunk_tokens
        if end >= N:
            break

    return math.exp(nll_sum / max(tok_count, 1))

def ppl_sanity(base, cmp, tok, args, dtype):
    print("\n[B] PPL sanity\n")
    if args.data_dir is None or (not os.path.isdir(args.data_dir)):
        print("data_dir not provided or not a directory. Skip PPL.")
        return

    if args.ppl_impl == "repo_c4":
        print(f"[Repo-C4] nsamples_text={args.c4_nsamples} seqlen={args.seqlen} (drop tail) loss_fp32={int(args.loss_fp32)}")
        base_ppl, bi = eval_ppl_repo_c4(
            base, tok, args.data_dir, args.base_device,
            seqlen=args.seqlen, nsamples_text=args.c4_nsamples, seed=args.seed,
            dtype=dtype, loss_fp32=args.loss_fp32
        )
        cmp_ppl, ci = eval_ppl_repo_c4(
            cmp, tok, args.data_dir, args.cmp_device,
            seqlen=args.seqlen, nsamples_text=args.c4_nsamples, seed=args.seed,
            dtype=dtype, loss_fp32=args.loss_fp32
        )
        print(f"BASE PPL: {base_ppl:.6f}  info={bi}")
        print(f"CMP  PPL: {cmp_ppl:.6f}  info={ci}")
        if math.isfinite(base_ppl) and base_ppl > 0 and math.isfinite(cmp_ppl):
            print(f"ratio CMP/BASE: {cmp_ppl/base_ppl:.6f}")
        print()
        return

    # legacy stream mode (debug)
    print(f"[Stream] token_budget={args.token_budget} seqlen={args.seqlen} stride={args.stride}")
    texts = iter_texts_from_dir(args.data_dir, max_files=80, max_lines=300000)
    stream = build_token_stream(tok, texts, target_tokens=args.token_budget, seed=args.seed)
    if stream is None:
        print(f"Failed to read any text from data_dir={args.data_dir}. Skip PPL.")
        return

    print(f"Built token stream: {stream.numel()} tokens (budget={args.token_budget})")
    base_ppl = eval_ppl_on_stream(base, stream, args.base_device, seqlen=args.seqlen, stride=args.stride, dtype=dtype)
    cmp_ppl  = eval_ppl_on_stream(cmp,  stream, args.cmp_device,  seqlen=args.seqlen, stride=args.stride, dtype=dtype)

    print(f"BASE PPL: {base_ppl:.6f}")
    print(f"CMP  PPL: {cmp_ppl:.6f}")
    if base_ppl > 0 and math.isfinite(base_ppl) and math.isfinite(cmp_ppl):
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

    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=2)

    # A
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--max_prompts", type=int, default=12)

    # B
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--token_budget", type=int, default=300000)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=2048)

    # repo-aligned ppl
    ap.add_argument("--ppl_impl", default="repo_c4", choices=["repo_c4", "stream"],
                    help="repo_c4: mimic repo eval (C4 256 texts + join + drop tail); stream: old local stream debug")
    ap.add_argument("--c4_nsamples", type=int, default=256, help="repo_c4: number of C4 texts to sample")
    ap.add_argument("--loss_fp32", type=int, default=1, help="repo_c4: compute CE in fp32 to reduce numeric noise")

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

    # Prompts
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
    ppl_sanity(base, cmp, tok, args, dtype)

    print("[Done] A+B sanity completed.")

if __name__ == "__main__":
    main()