import os, json, glob, gzip, math, random, argparse
import torch
from transformers import AutoTokenizer
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

# -------------------------
# dataset reader (txt/jsonl/json/gz) -> token stream (CPU)
# -------------------------
def iter_texts_from_dir(data_dir, max_files=80, max_lines=300000):
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
                if fp.endswith(".json") and (not fp.endswith(".gz")):
                    try:
                        obj = json.load(f)
                        if isinstance(obj, list):
                            for it in obj:
                                t = None
                                if isinstance(it, dict):
                                    t = it.get("text") or it.get("content")
                                elif isinstance(it, str):
                                    t = it
                                if isinstance(t, str) and t.strip():
                                    yield t.strip()
                                    n += 1
                                    if n >= max_lines: return
                        elif isinstance(obj, dict):
                            t = obj.get("text") or obj.get("content")
                            if isinstance(t, str) and t.strip():
                                yield t.strip()
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
                                t = j.get("text") or j.get("content")
                        except Exception:
                            t = None
                    if t is None:
                        t = line
                    if isinstance(t, str) and t.strip():
                        yield t.strip()
                        n += 1
                        if n >= max_lines: return
        except Exception:
            continue

def build_token_stream(tok, texts_iter, target_tokens=500000, seed=2):
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
    return torch.tensor(ids, dtype=torch.long)  # CPU

def batchify_stream_cpu(stream_cpu, seqlen, batch_size):
    N = stream_cpu.numel()
    assert N > seqlen + 1
    while True:
        starts = torch.randint(low=0, high=N - seqlen - 1, size=(batch_size,))
        x = torch.stack([stream_cpu[s:s+seqlen] for s in starts], dim=0)
        attn = torch.ones_like(x, dtype=torch.long)
        yield x, attn

# -------------------------
# trainable selector
# -------------------------
def set_trainable(model, mode="vscale_only", train_last_n=0):
    """
    mode:
      - vscale_only: only vscale_log
      - up_vscale: out_proj_up + vscale_log
      - qkv_up_vscale: q/k/v low + out_proj_up + vscale_log
      - all: everything

    train_last_n:
      - 0 => all layers
      - >0 => only last N decoder layers are unfrozen (for modes except 'all')
    """
    for p in model.parameters():
        p.requires_grad_(False)

    def mark(module):
        for p in module.parameters():
            p.requires_grad_(True)

    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)
        return

    layers = list(model.model.decoder.layers)
    if train_last_n and train_last_n > 0:
        layers = layers[-train_last_n:]

    for layer in layers:
        attn = layer.self_attn

        if hasattr(attn, "vscale_log"):
            try:
                attn.vscale_log.requires_grad_(True)
            except Exception:
                pass

        if mode in ["up_vscale", "qkv_up_vscale"]:
            if hasattr(attn, "out_proj_up"):
                mark(attn.out_proj_up)

        if mode == "qkv_up_vscale":
            for name in ["q_proj_low", "k_proj_low", "v_proj_low"]:
                if hasattr(attn, name):
                    mark(getattr(attn, name))

# -------------------------
# eval ppl (correct; labels=input_ids)
# -------------------------
@torch.no_grad()
def eval_ppl(model, stream_cpu, device, seqlen=2048, stride=2048, dtype=torch.bfloat16):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    nll_sum = 0.0
    tok_count = 0
    N = stream_cpu.numel()
    if N < 2:
        return float("inf")

    for start in range(0, N, stride):
        end = min(start + seqlen, N)
        if end - start < 2:
            break

        inp = stream_cpu[start:end].unsqueeze(0).to(device, non_blocking=True)
        attn = torch.ones_like(inp, dtype=torch.long)
        labels = inp.clone()  # ✅

        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=inp, attention_mask=attn, labels=labels)
            loss = out.loss

        chunk_tokens = inp.numel() - 1
        nll_sum += loss.detach().float().item() * chunk_tokens
        tok_count += chunk_tokens

        if end >= N:
            break

    return math.exp(nll_sum / max(tok_count, 1))

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--data_dir", required=True)

    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--device", default="cuda:0")  # under CUDA_VISIBLE_DEVICES=2,3 => cuda:0=phys2, cuda:1=phys3

    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--token_budget", type=int, default=600000)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument("--mode", default="vscale_only",
                    choices=["vscale_only","up_vscale","qkv_up_vscale","all"])
    ap.add_argument("--train_last_n", type=int, default=0,
                    help="only unfreeze last N decoder layers (0=all layers)")

    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_tokens", type=int, default=200000)
    ap.add_argument("--grad_ckpt", type=int, default=1)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    low_rank_config = json.load(open(args.low_json))

    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    print("[Args]", vars(args))

    print("[Load] student CMP...")
    student = OPTForCausalLM_EigenAttn.from_pretrained(
        args.cmp,
        local_files_only=True,
        torch_dtype=dtype,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        low_rank_config=low_rank_config,
    )
    student.config.use_cache = False
    if args.grad_ckpt == 1 and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        print("[GradCkpt] enabled")
    student.eval()

    set_trainable(student, args.mode, train_last_n=args.train_last_n)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in trainable_params)
    total = sum(p.numel() for p in student.parameters())
    print(f"[Trainable] {trainable:,} / {total:,} ({100.0*trainable/total:.6f}%) mode={args.mode} lastN={args.train_last_n}")

    # Data stream (CPU)
    texts = iter_texts_from_dir(args.data_dir)
    stream_cpu = build_token_stream(tok, texts, target_tokens=args.token_budget, seed=args.seed)
    assert stream_cpu is not None, "No data read from data_dir"
    print(f"[Data] token_stream={stream_cpu.numel()} seqlen={args.seqlen} bs={args.batch_size}")

    loader = batchify_stream_cpu(stream_cpu, args.seqlen, args.batch_size)

    # Eval stream
    texts2 = iter_texts_from_dir(args.data_dir)
    eval_stream_cpu = build_token_stream(tok, texts2, target_tokens=args.eval_tokens, seed=args.seed+1)
    assert eval_stream_cpu is not None

    print("[Eval] initial PPL ...")
    base_ppl = eval_ppl(student, eval_stream_cpu, device=args.device, seqlen=2048, stride=2048, dtype=dtype)
    print(f"CMP PPL init: {base_ppl:.6f}")

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    student.train()
    opt.zero_grad(set_to_none=True)

    for step in range(1, args.steps + 1):
        x_cpu, attn_cpu = next(loader)
        x = x_cpu.to(args.device, non_blocking=True)
        attn = attn_cpu.to(args.device, non_blocking=True)
        labels = x  # ✅ PPL 对应的监督

        with torch.autocast(device_type="cuda", dtype=dtype):
            out = student(input_ids=x, attention_mask=attn, labels=labels)
            loss = out.loss / max(args.grad_accum, 1)

        loss.backward()

        if step % args.grad_accum == 0:
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 20 == 0:
            # 这里打印的是 (per-token) NLL
            print(f"[Step {step:6d}] nll={loss.item()*max(args.grad_accum,1):.6f}")

        if args.eval_every > 0 and step % args.eval_every == 0:
            student.eval()
            ppl = eval_ppl(student, eval_stream_cpu, device=args.device, seqlen=2048, stride=2048, dtype=dtype)
            print(f"[Eval step {step}] CMP PPL: {ppl:.6f}")
            student.train()

    print("[Done] training finished (no checkpoints saved).")

if __name__ == "__main__":
    main()