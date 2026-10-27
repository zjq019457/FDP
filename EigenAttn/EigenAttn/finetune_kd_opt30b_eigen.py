import os, json, glob, gzip, math, random, argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

# -------------------------
# GPU memory caps (best-effort)
# -------------------------
def apply_memory_caps(mem_caps_gb: str):
    """
    mem_caps_gb format: "0:20,1:20,2:80,3:80"  (indices are VISIBLE cuda indices)
    This uses torch.cuda.set_per_process_memory_fraction which limits PyTorch allocations.
    """
    if not mem_caps_gb:
        return
    if not torch.cuda.is_available():
        print("[MemCap] CUDA not available; skip.")
        return

    # parse
    caps = {}
    for part in mem_caps_gb.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad --mem_caps_gb segment: {part} (expect i:GB)")
        i_str, gb_str = part.split(":")
        i = int(i_str.strip())
        gb = float(gb_str.strip())
        caps[i] = gb

    # apply
    for dev in sorted(caps.keys()):
        if dev < 0 or dev >= torch.cuda.device_count():
            print(f"[MemCap] skip dev {dev}: out of visible range (count={torch.cuda.device_count()})")
            continue
        total = torch.cuda.get_device_properties(dev).total_memory  # bytes
        cap_bytes = caps[dev] * (1024 ** 3)
        # leave a little headroom: if user asks >= total, cap to 97% total to reduce fragmentation risk
        if cap_bytes >= total:
            cap_bytes = total * 0.97
        frac = max(0.01, min(0.99, cap_bytes / total))
        torch.cuda.set_per_process_memory_fraction(frac, device=dev)
        print(f"[MemCap] cuda:{dev} total={total/1e9:.2f}GB cap={cap_bytes/1e9:.2f}GB frac={frac:.4f}")

# -------------------------
# dataset reader (txt/jsonl/json/gz) -> token stream
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
        opener = open
        is_gz = fp.endswith(".gz")
        if is_gz:
            opener = gzip.open
        try:
            with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
                if fp.endswith(".json") and (not is_gz):
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
                                    if n >= max_lines:
                                        return
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
                        if n >= max_lines:
                            return
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
    return torch.tensor(ids, dtype=torch.long)  # keep on CPU

def batchify_stream_cpu(stream_cpu, seqlen, batch_size):
    N = stream_cpu.numel()
    assert N > seqlen + 1
    while True:
        starts = torch.randint(low=0, high=N - seqlen - 1, size=(batch_size,))
        x = torch.stack([stream_cpu[s:s+seqlen] for s in starts], dim=0)
        y = torch.stack([stream_cpu[s+1:s+seqlen+1] for s in starts], dim=0)
        attn = torch.ones_like(x, dtype=torch.long)
        yield x, y, attn

# -------------------------
# KD (top-k) loss
# -------------------------
def kd_kl_topk(student_logits_topk, teacher_logits_topk, T=2.0):
    """
    KL(teacher || student) on restricted top-k support
    student_logits_topk: [B, L, K]
    teacher_logits_topk: [B, L, K]
    """
    s = F.log_softmax(student_logits_topk / T, dim=-1)
    t = F.softmax(teacher_logits_topk / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)

def set_trainable(model, mode="vscale_only"):
    """
    mode:
      - vscale_only: only vscale_log
      - up_vscale: out_proj_up + vscale_log
      - qkv_up_vscale: q/k/v low + out_proj_up + vscale_log
      - all: everything
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

    for layer in model.model.decoder.layers:
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
# eval ppl (correct; avoid double-shift)
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

        # ✅ labels = input_ids; HF 内部 shift
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

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--base_device", default="cuda:2")  # recommend: put teacher on big GPU
    ap.add_argument("--cmp_device", default="cuda:3")   # recommend: put student on big GPU

    ap.add_argument("--mem_caps_gb", default="",
                    help='Best-effort cap per VISIBLE GPU, e.g. "0:20,1:20,2:80,3:80"')

    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--token_budget", type=int, default=600000)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=1)

    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--alpha_kd", type=float, default=0.9)
    ap.add_argument("--mode", default="vscale_only",
                    choices=["vscale_only","up_vscale","qkv_up_vscale","all"])

    ap.add_argument("--kd_topk", type=int, default=128)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_tokens", type=int, default=200000)
    ap.add_argument("--save_every", type=int, default=100)

    ap.add_argument("--grad_ckpt", type=int, default=1, help="Enable gradient checkpointing for student (1/0)")

    args = ap.parse_args()

    # Apply caps BEFORE big allocations
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", None))
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    apply_memory_caps(args.mem_caps_gb)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    os.makedirs(args.out_dir, exist_ok=True)
    low_rank_config = json.load(open(args.low_json))

    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("[Args]", vars(args))

    print("[Load] teacher BASE...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.base,
        local_files_only=True,
        torch_dtype=dtype,
        device_map={"": args.base_device},
        low_cpu_mem_usage=True,
    ).eval()
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("[Load] student CMP...")
    student = OPTForCausalLM_EigenAttn.from_pretrained(
        args.cmp,
        local_files_only=True,
        torch_dtype=dtype,
        device_map={"": args.cmp_device},
        low_cpu_mem_usage=True,
        low_rank_config=low_rank_config,
    )
    student.config.use_cache = False
    if args.grad_ckpt == 1 and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        print("[GradCkpt] enabled for student")
    student.eval()

    # trainable params
    set_trainable(student, args.mode)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"[Trainable] {trainable:,} / {total:,} ({100.0*trainable/total:.6f}%) mode={args.mode}")

    # Data (CPU)
    texts = iter_texts_from_dir(args.data_dir)
    stream_cpu = build_token_stream(tok, texts, target_tokens=args.token_budget, seed=args.seed)
    assert stream_cpu is not None, "No data read from data_dir"
    print(f"[Data] token_stream={stream_cpu.numel()} seqlen={args.seqlen} bs={args.batch_size}")

    loader = batchify_stream_cpu(stream_cpu, args.seqlen, args.batch_size)

    texts2 = iter_texts_from_dir(args.data_dir)
    eval_stream_cpu = build_token_stream(tok, texts2, target_tokens=args.eval_tokens, seed=args.seed+1)
    assert eval_stream_cpu is not None

    # Optimizer ONLY trainable
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable params selected"
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Initial eval
    print("[Eval] initial PPL ...")
    base_ppl = eval_ppl(teacher, eval_stream_cpu, device=args.base_device, seqlen=2048, stride=2048, dtype=dtype)
    cmp_ppl  = eval_ppl(student, eval_stream_cpu, device=args.cmp_device,  seqlen=2048, stride=2048, dtype=dtype)
    print(f"BASE PPL: {base_ppl:.6f}")
    print(f"CMP  PPL: {cmp_ppl:.6f}  ratio={cmp_ppl/base_ppl:.6f}")

    # Train loop
    student.train()
    for step in range(1, args.steps + 1):
        x_cpu, y_cpu, attn_cpu = next(loader)

        # move to devices
        x_s = x_cpu.to(args.cmp_device, non_blocking=True)
        y_s = y_cpu.to(args.cmp_device, non_blocking=True)
        attn_s = attn_cpu.to(args.cmp_device, non_blocking=True)

        x_t = x_cpu.to(args.base_device, non_blocking=True)
        attn_t = attn_cpu.to(args.base_device, non_blocking=True)

        # teacher forward
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=dtype):
                t_out = teacher(input_ids=x_t, attention_mask=attn_t)
                t_logits = t_out.logits  # [B, L, V] on base_device

            # cut last position (LM)
            t_cut = t_logits[:, :-1, :]  # [B, L-1, V]
            k = min(args.kd_topk, t_cut.size(-1))
            t_vals, t_idx = torch.topk(t_cut, k=k, dim=-1)  # [B, L-1, K]

        # student forward
        with torch.autocast(device_type="cuda", dtype=dtype):
            s_out = student(input_ids=x_s, attention_mask=attn_s)
            s_logits = s_out.logits  # [B, L, V] on cmp_device

        s_cut = s_logits[:, :-1, :]  # [B, L-1, V]
        y_cut = y_s[:, :-1]          # [B, L-1]

        # move teacher top-k to student device
        t_vals_s = t_vals.to(args.cmp_device, non_blocking=True)
        t_idx_s  = t_idx.to(args.cmp_device, non_blocking=True)

        # gather student logits on teacher support
        s_topk = torch.gather(s_cut, dim=-1, index=t_idx_s)  # [B, L-1, K]

        loss_kd = kd_kl_topk(s_topk, t_vals_s, T=args.T)
        loss_nll = F.cross_entropy(
            s_cut.reshape(-1, s_cut.size(-1)),
            y_cut.reshape(-1),
            reduction="mean",
        )
        loss = args.alpha_kd * loss_kd + (1.0 - args.alpha_kd) * loss_nll

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        opt.step()

        if step % 10 == 0:
            print(f"[Step {step:5d}] loss={loss.item():.6f} kd={loss_kd.item():.6f} nll={loss_nll.item():.6f}")

        if args.eval_every > 0 and step % args.eval_every == 0:
            student.eval()
            cmp_ppl2 = eval_ppl(student, eval_stream_cpu, device=args.cmp_device, seqlen=2048, stride=2048, dtype=dtype)
            print(f"[Eval step {step}] CMP PPL: {cmp_ppl2:.6f}  (BASE {base_ppl:.6f}) ratio={cmp_ppl2/base_ppl:.6f}")
            student.train()

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt_dir = os.path.join(args.out_dir, f"ckpt_step{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            student.save_pretrained(ckpt_dir, safe_serialization=True)
            tok.save_pretrained(ckpt_dir)
            print(f"[Save] {ckpt_dir}")

    # final save
    student.eval()
    final_dir = os.path.join(args.out_dir, "final_student")
    os.makedirs(final_dir, exist_ok=True)
    student.save_pretrained(final_dir, safe_serialization=True)
    tok.save_pretrained(final_dir)
    print(f"[Final save] {final_dir}")

if __name__ == "__main__":
    main()