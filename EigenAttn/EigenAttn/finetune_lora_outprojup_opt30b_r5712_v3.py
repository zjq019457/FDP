import os, json, math, time, argparse, gzip
from typing import Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

def iter_texts_from_dir(data_dir: str, max_docs: Optional[int] = None) -> Iterator[str]:
    n = 0
    for root, _, files in os.walk(data_dir):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            if fn.endswith(".jsonl") or fn.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line) if line and line[0] == "{" else None
                        except Exception:
                            obj = None
                        if isinstance(obj, dict) and "text" in obj:
                            yield obj["text"]
                        else:
                            yield line
                        n += 1
                        if max_docs and n >= max_docs:
                            return
            elif fn.endswith(".json.gz") or fn.endswith(".jsonl.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict) and "text" in obj:
                                yield obj["text"]
                            else:
                                yield line
                        except Exception:
                            yield line
                        n += 1
                        if max_docs and n >= max_docs:
                            return
            elif fn.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        t = line.strip()
                        if t:
                            yield t
                            n += 1
                            if max_docs and n >= max_docs:
                                return

def token_blocks(tok, data_dir: str, seqlen: int, train_tokens: int, eos_id: int, max_docs: Optional[int]=None):
    buf: List[int] = []
    seen = 0
    for text in iter_texts_from_dir(data_dir, max_docs=max_docs):
        ids = tok(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= seqlen + 1:
            block = buf[:seqlen+1]
            buf = buf[seqlen+1:]
            seen += seqlen
            yield block
            if seen >= train_tokens:
                return

# ---------------------------
# LoRA wrapper (GPU + FP32 LoRA branch; exposes weight/bias)
# ---------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base.in_features
        out_f = base.out_features

        dev = base.weight.device
        # ✅ A/B 直接创建在 GPU 上，且用 FP32 更稳定
        self.A = nn.Parameter(torch.empty((r, in_f), device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.empty((out_f, r), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    @property
    def weight(self):  # for your EigenAttention forward that reads .weight
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)  # y dtype follows model dtype (bf16/fp16)

        # ✅ LoRA branch in FP32, then cast back
        x32 = self.dropout(x).to(torch.float32)
        l = F.linear(x32, self.A)                       # [*, r]
        l = F.linear(l, self.B) * self.scaling          # [*, out]
        l = l.to(dtype=y.dtype)

        return y + l

def inject_lora_out_proj_up(model: nn.Module, r: int, alpha: float, dropout: float):
    n = 0
    for name, mod in model.named_modules():
        if name.endswith("self_attn.out_proj_up") and isinstance(mod, nn.Linear):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
            n += 1
    return n

def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_in", required=True)
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--data_dir", required=True)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","float16"])
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--train_tokens", type=int, default=2_000_000)
    ap.add_argument("--max_docs", type=int, default=0)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    low = json.load(open(args.low_json, "r"))
    tok = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("[Load] student model ...")
    model = OPTForCausalLM_EigenAttn.from_pretrained(
        args.model_in,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        low_rank_config=low,
    ).to(args.device)

    if args.grad_ckpt == 1 and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[Cfg] gradient checkpointing ENABLED")

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    n_lora = inject_lora_out_proj_up(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    print(f"[LoRA] injected into out_proj_up modules: {n_lora}")

    trainables = get_trainable_params(model)
    n_params = sum(p.numel() for p in trainables)
    print(f"[Train] trainable params: {n_params/1e6:.3f} M")

    opt = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=args.wd)

    model.train()
    max_docs = None if args.max_docs <= 0 else args.max_docs
    stream = token_blocks(tok, args.data_dir, args.seqlen, args.train_tokens, eos_id, max_docs=max_docs)

    step = 0
    accum = 0
    t0 = time.time()
    loss_ma = None

    while step < args.max_steps:
        try:
            block = next(stream)
        except StopIteration:
            print("[Data] token stream exhausted")
            break

        x = torch.tensor(block[:-1], device=args.device).unsqueeze(0)
        y = torch.tensor(block[1:],  device=args.device).unsqueeze(0)

        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=x, labels=y, use_cache=False)
            loss = out.loss / args.grad_accum

        loss.backward()
        accum += 1

        lv = float(loss.detach().cpu().item() * args.grad_accum)
        loss_ma = lv if loss_ma is None else (0.9*loss_ma + 0.1*lv)

        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(trainables, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"step {step:5d} | loss {lv:.4f} | loss_ma {loss_ma:.4f} | {dt/max(step,1):.2f}s/step")

            if step % args.save_every == 0:
                os.makedirs(args.model_out, exist_ok=True)
                model.save_pretrained(args.model_out, safe_serialization=True)
                tok.save_pretrained(args.model_out)
                print("[Save] partial saved to:", args.model_out)

    os.makedirs(args.model_out, exist_ok=True)
    model.save_pretrained(args.model_out, safe_serialization=True)
    tok.save_pretrained(args.model_out)
    print("[Done] saved to:", args.model_out)

if __name__ == "__main__":
    main()
