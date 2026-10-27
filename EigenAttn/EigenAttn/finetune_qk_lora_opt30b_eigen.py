#!/usr/bin/env python3
import os, json, math, argparse, random, glob, gzip
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def set_seed(seed: int):
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

def make_random_batch(stream: torch.Tensor, seqlen: int, bsz: int, rng: random.Random) -> torch.Tensor:
    T = stream.numel()
    max_start = T - seqlen - 1
    starts = [rng.randint(0, max_start) for _ in range(bsz)]
    return torch.stack([stream[s:s+seqlen] for s in starts], dim=0)

@torch.no_grad()
def eval_ppl_repo_c4(model, tok, data_dir: str, device: str, seqlen=2048, nsamples_text=256, seed=2,
                    dtype=torch.bfloat16, loss_fp32=True) -> Tuple[float, Dict]:
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    texts = reservoir_sample_texts_from_dir(data_dir, nsamples_text, seed=seed)
    ids = build_token_stream_join(tok, texts)
    T = ids.numel()
    n_chunks = T // seqlen
    used = n_chunks * seqlen
    if n_chunks <= 0:
        return float("inf"), {"tokens": T, "chunks": 0, "bad_chunks": 0}

    nll_sum, bad = 0.0, 0
    for i in range(n_chunks):
        batch = ids[i*seqlen:(i+1)*seqlen].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=batch, use_cache=False)
            logits = out.logits
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
        nll_sum += float(loss.item()) * seqlen

    ppl = math.exp(nll_sum / max(used, 1))
    return ppl, {"tokens": used, "chunks": n_chunks, "bad_chunks": bad}

def freeze_lora_by_layer(peft_model, start_layer: int):
    # LoRA 默认会打到所有层；这里把 start_layer 之前的 LoRA 参数冻结，等价于“只训最后N层”
    for n, p in peft_model.named_parameters():
        if "lora_" not in n:
            continue
        # n 里通常包含 "model.decoder.layers.X."
        keep = True
        for li in range(0, start_layer):
            if f"model.decoder.layers.{li}." in n:
                keep = False
                break
        p.requires_grad = keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--device", default="cuda:0")

    # train
    ap.add_argument("--train_last_n", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--calib_nsamples", type=int, default=4096)

    # eval
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_seed", type=int, default=2)
    ap.add_argument("--eval_nsamples", type=int, default=256)
    ap.add_argument("--eval_seqlen", type=int, default=2048)
    ap.add_argument("--loss_fp32", type=int, default=1)

    args = ap.parse_args()
    set_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype=="bf16" else torch.float16
    loss_fp32 = bool(args.loss_fp32)

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    low_rank = json.load(open(args.low_json))
    print("[Load CMP]", args.cmp)
    model = OPTForCausalLM_EigenAttn.from_pretrained(
        args.cmp,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
        low_rank_config=low_rank,
    )
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    # PEFT LoRA on q/k low projections (关键：只补 Q/K)
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj_low", "k_proj_low"],  # ✅ Q/K 补偿点
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lcfg)

    # freeze earlier layers' LoRA (only last N layers)
    L = peft_model.base_model.model.config.num_hidden_layers
    start_layer = max(0, L - args.train_last_n)
    freeze_lora_by_layer(peft_model, start_layer)

    # count trainable params
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"[Params] trainable={trainable:,} total={total:,} ({100*trainable/total:.6f}%)  train_last_n={args.train_last_n}")

    # build training stream
    print("[Data] build token stream")
    texts = reservoir_sample_texts_from_dir(args.data_dir, nsamples=args.calib_nsamples, seed=args.seed+101)
    stream = build_token_stream_join(tok, texts)
    print("[Data] token_stream:", stream.numel())

    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed + 999)

    # eval before
    bppl, bi = eval_ppl_repo_c4(peft_model, tok, args.data_dir, args.device,
                               seqlen=args.eval_seqlen, nsamples_text=args.eval_nsamples, seed=args.eval_seed,
                               dtype=dtype, loss_fp32=loss_fp32)
    print(f"[Eval-Before] PPL={bppl:.6f} info={bi}")

    peft_model.train()
    running = 0.0
    opt.zero_grad(set_to_none=True)

    for step in range(1, args.steps+1):
        batch = make_random_batch(stream, args.seqlen, args.batch_size, rng).to(args.device)
        labels = batch.clone()

        with torch.autocast(device_type="cuda", dtype=dtype):
            out = peft_model(input_ids=batch, labels=labels, use_cache=False)
            loss = out.loss / args.grad_accum

        loss.backward()
        running += float(loss.detach().item())

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in peft_model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            print(f"[Train] step={step} loss(avg/50)={running:.6g}")
            running = 0.0

        if args.eval_every > 0 and step % args.eval_every == 0:
            peft_model.eval()
            ppl, info = eval_ppl_repo_c4(peft_model, tok, args.data_dir, args.device,
                                         seqlen=args.eval_seqlen, nsamples_text=args.eval_nsamples, seed=args.eval_seed,
                                         dtype=dtype, loss_fp32=loss_fp32)
            print(f"[Eval] step={step} PPL={ppl:.6f} info={info}")
            peft_model.train()

    # save adapter
    adapter_dir = os.path.join(args.out_dir, "adapter_qk")
    os.makedirs(adapter_dir, exist_ok=True)
    peft_model.save_pretrained(adapter_dir)
    tok.save_pretrained(args.out_dir)

    # final eval
    peft_model.eval()
    ppl, info = eval_ppl_repo_c4(peft_model, tok, args.data_dir, args.device,
                                 seqlen=args.eval_seqlen, nsamples_text=args.eval_nsamples, seed=args.eval_seed,
                                 dtype=dtype, loss_fp32=loss_fp32)
    print(f"[Eval-After] PPL={ppl:.6f} info={info}")
    print("[Done] saved adapter to:", adapter_dir)

if __name__ == "__main__":
    main()
