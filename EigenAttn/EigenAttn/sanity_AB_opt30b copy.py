import os, json, glob, gzip, argparse, math, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

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
# B) small PPL sanity (local text/json/jsonl/gz)
# -------------------------
def iter_texts_from_dir(data_dir, max_files=50, max_lines=200000):
    """
    Reads text samples from local files:
    - *.txt
    - *.jsonl (expects {"text": "..."} or {"content": "..."} or raw line text)
    - *.json  (list/dict; tries to find 'text'/'content' fields)
    - *.gz    (gzip of txt/jsonl)
    """
    exts = ["*.txt", "*.jsonl", "*.json", "*.gz"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", e), recursive=True))
    files = sorted(files)[:max_files]

    if not files:
        return

    n = 0
    for fp in files:
        opener = open
        is_gz = fp.endswith(".gz")
        if is_gz:
            opener = gzip.open

        try:
            with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
                # json (non-line-delimited)
                if fp.endswith(".json") and (not is_gz):
                    try:
                        obj = json.load(f)
                        if isinstance(obj, list):
                            for it in obj:
                                if isinstance(it, dict):
                                    t = it.get("text", None) or it.get("content", None)
                                    if isinstance(t, str) and t.strip():
                                        yield t.strip()
                                        n += 1
                                        if n >= max_lines: return
                                elif isinstance(it, str) and it.strip():
                                    yield it.strip()
                                    n += 1
                                    if n >= max_lines: return
                        elif isinstance(obj, dict):
                            # try common keys
                            for k in ["text", "content"]:
                                t = obj.get(k, None)
                                if isinstance(t, str) and t.strip():
                                    yield t.strip()
                                    n += 1
                                    if n >= max_lines: return
                    except Exception:
                        pass
                    continue

                # line-based
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
    """
    Concatenate many texts into a 1D token stream for chunked LM eval.
    """
    random.seed(seed)
    ids = []
    total = 0
    for t in texts_iter:
        enc = tok(t, add_special_tokens=False)["input_ids"]
        if not enc:
            continue
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
    # 避免返回/构建 cache（省显存 + 不影响 loss）
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

        # ✅ 关键：labels 不要手动右移，让 HF 内部完成 shift
        labels = inp.clone()

        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=inp, attention_mask=attn, labels=labels)
            loss = out.loss  # mean over (seq_len - 1)

        chunk_tokens = inp.numel() - 1  # HF 的 loss 实际预测 token 数
        nll_sum += loss.detach().float().item() * chunk_tokens
        tok_count += chunk_tokens

        if end >= N:
            break

    return math.exp(nll_sum / max(tok_count, 1))
    
def ppl_sanity(base, cmp, tok, data_dir, token_budget, seqlen, stride, base_device, cmp_device, dtype, seed):
    print("\n[B] Small PPL sanity (local data)\n")
    if data_dir is None or (not os.path.isdir(data_dir)):
        print("data_dir not provided or not a directory. Skip PPL sanity.")
        return

    texts = iter_texts_from_dir(data_dir, max_files=80, max_lines=300000)
    stream = build_token_stream(tok, texts, target_tokens=token_budget, seed=seed)
    if stream is None:
        print(f"Failed to read any text from data_dir={data_dir}. Skip PPL sanity.")
        return

    print(f"Built token stream: {stream.numel()} tokens (budget={token_budget})")
    base_ppl = eval_ppl_on_stream(base, stream, base_device, seqlen=seqlen, stride=stride, dtype=dtype)
    cmp_ppl  = eval_ppl_on_stream(cmp,  stream, cmp_device,  seqlen=seqlen, stride=stride, dtype=dtype)

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
    ap.add_argument("--prompts_file", default=None, help="Optional: a txt file, one prompt per line.")
    ap.add_argument("--max_prompts", type=int, default=12)

    # B
    ap.add_argument("--data_dir", default=None, help="Local directory containing txt/jsonl/json/gz shards")
    ap.add_argument("--token_budget", type=int, default=300000, help="How many tokens to build for PPL stream")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=2048)

    args = ap.parse_args()
    set_seed(args.seed)

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

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

    ppl_sanity(
        base, cmp, tok,
        data_dir=args.data_dir,
        token_budget=args.token_budget,
        seqlen=args.seqlen,
        stride=args.stride,
        base_device=args.base_device,
        cmp_device=args.cmp_device,
        dtype=dtype,
        seed=args.seed,
    )

    print("[Done] A+B sanity completed.")

if __name__ == "__main__":
    main()
