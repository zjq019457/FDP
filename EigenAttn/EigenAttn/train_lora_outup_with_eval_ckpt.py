import os, json, math, time, argparse, gzip
from typing import Iterator, List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

# ---------------------------
# Data: stream texts -> token stream -> fixed-length blocks
# ---------------------------
def iter_texts_from_dir(data_dir: str, max_docs: Optional[int] = None, offset_docs: int = 0) -> Iterator[str]:
    n = 0
    skipped = 0
    for root, _, files in os.walk(data_dir):
        for fn in sorted(files):
            path = os.path.join(root, fn)

            def yield_text(line: str):
                nonlocal n, skipped
                line = line.strip()
                if not line:
                    return None
                if skipped < offset_docs:
                    skipped += 1
                    return None
                n += 1
                return line

            if fn.endswith(".jsonl") or fn.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        raw = yield_text(line)
                        if raw is None:
                            continue
                        try:
                            obj = json.loads(raw) if raw and raw[0] == "{" else None
                        except Exception:
                            obj = None
                        yield obj["text"] if isinstance(obj, dict) and "text" in obj else raw
                        if max_docs and n >= max_docs:
                            return

            elif fn.endswith(".json.gz") or fn.endswith(".jsonl.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        raw = yield_text(line)
                        if raw is None:
                            continue
                        try:
                            obj = json.loads(raw)
                            yield obj["text"] if isinstance(obj, dict) and "text" in obj else raw
                        except Exception:
                            yield raw
                        if max_docs and n >= max_docs:
                            return

            elif fn.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        raw = yield_text(line)
                        if raw is None:
                            continue
                        yield raw
                        if max_docs and n >= max_docs:
                            return

def token_blocks(tok, data_dir: str, seqlen: int, want_tokens: int, eos_id: int,
                 max_docs: Optional[int]=None, offset_docs: int = 0):
    buf: List[int] = []
    seen = 0
    for text in iter_texts_from_dir(data_dir, max_docs=max_docs, offset_docs=offset_docs):
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
            if seen >= want_tokens:
                return

# ---------------------------
# LoRA wrapper for Linear (exposes .weight/.bias to satisfy your EigenAttention forward)
# ---------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        dev = base.weight.device

        # LoRA params on same device, FP32 for stability
        self.A = nn.Parameter(torch.empty((r, in_f), device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.empty((out_f, r), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    @property
    def weight(self):  # used by your forward
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
        y = self.base(x)
        x32 = self.dropout(x).to(torch.float32)
        l = F.linear(x32, self.A)
        l = F.linear(l, self.B) * self.scaling
        return y + l.to(dtype=y.dtype)

def inject_lora_out_proj_up(model: nn.Module, r: int, alpha: float, dropout: float) -> int:
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

def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]

def get_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    st = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            st[name + ".A"] = mod.A.detach().cpu()
            st[name + ".B"] = mod.B.detach().cpu()
    return st

@torch.no_grad()
def load_lora_state(model: nn.Module, st: Dict[str, torch.Tensor]):
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            A_key = name + ".A"
            B_key = name + ".B"
            if A_key in st:
                mod.A.copy_(st[A_key].to(device=mod.A.device, dtype=mod.A.dtype))
            if B_key in st:
                mod.B.copy_(st[B_key].to(device=mod.B.device, dtype=mod.B.dtype))

def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

@torch.no_grad()
def eval_ppl(model, tok, data_dir, device, amp_dtype, seqlen, eval_tokens, eos_id,
             eval_offset_docs: int) -> Tuple[float, float]:
    model.eval()
    stream = token_blocks(tok, data_dir, seqlen, eval_tokens, eos_id, offset_docs=eval_offset_docs)
    losses = []
    for block in stream:
        x = torch.tensor(block[:-1], device=device).unsqueeze(0)
        y = torch.tensor(block[1:],  device=device).unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(input_ids=x, labels=y, use_cache=False)
            losses.append(float(out.loss.detach().cpu().item()))
    if not losses:
        return float("nan"), float("nan")
    mean_loss = sum(losses) / len(losses)
    ppl = float(math.exp(mean_loss))
    return mean_loss, ppl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_in", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","float16"])
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--train_tokens", type=int, default=2_000_000)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)

    # eval
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_seqlen", type=int, default=512)
    ap.add_argument("--eval_tokens", type=int, default=50000)
    ap.add_argument("--eval_offset_docs", type=int, default=20000)

    # checkpoint (single file, overwritten)
    ap.add_argument("--ckpt_name", default="ckpt.pt")
    ap.add_argument("--resume", type=int, default=1)

    # small csv for curves
    ap.add_argument("--csv_name", default="curve.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    csv_path  = os.path.join(args.out_dir, args.csv_name)

    low = json.load(open(args.low_json, "r"))
    tok = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id

    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("[Load] student ...")
    model = OPTForCausalLM_EigenAttn.from_pretrained(
        args.model_in, local_files_only=True, torch_dtype=amp_dtype,
        device_map=None, low_cpu_mem_usage=True, low_rank_config=low
    ).to(args.device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[Cfg] gradient checkpointing ENABLED")

    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    n = inject_lora_out_proj_up(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    print(f"[LoRA] injected modules: {n}")

    trainables = trainable_params(model)
    for p in trainables:
        p.requires_grad = True
    print(f"[Train] trainable params: {sum(p.numel() for p in trainables)/1e6:.3f} M")

    opt = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=args.wd)

    # resume
    step = 0
    loss_ma = None
    best_eval_ppl = float("inf")
    if args.resume == 1 and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        step = int(ck.get("step", 0))
        loss_ma = ck.get("loss_ma", None)
        best_eval_ppl = float(ck.get("best_eval_ppl", best_eval_ppl))
        load_lora_state(model, ck["lora"])
        opt.load_state_dict(ck["opt"])
        print(f"[Resume] loaded {ckpt_path} at step={step}, best_eval_ppl={best_eval_ppl}")

    # csv header
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("step,train_loss,train_loss_ma,eval_loss,eval_ppl,sec_per_step\n")

    model.train()
    stream = token_blocks(tok, args.data_dir, args.seqlen, args.train_tokens, eos_id)

    accum = 0
    t0 = time.time()

    while step < args.max_steps:
        try:
            block = next(stream)
        except StopIteration:
            print("[Data] exhausted")
            break

        x = torch.tensor(block[:-1], device=args.device).unsqueeze(0)
        y = torch.tensor(block[1:],  device=args.device).unsqueeze(0)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(input_ids=x, labels=y, use_cache=False)
            loss = out.loss / args.grad_accum

        loss.backward()
        accum += 1

        lv = float(loss.detach().cpu().item() * args.grad_accum)
        loss_ma = lv if loss_ma is None else (0.9 * loss_ma + 0.1 * lv)

        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(trainables, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            sec_per = (time.time() - t0) / max(step, 1)

            if step % args.log_every == 0:
                print(f"step {step:5d} | loss {lv:.4f} | loss_ma {loss_ma:.4f} | {sec_per:.2f}s/step")

            eval_loss = float("nan")
            eval_ppl_v = float("nan")
            if args.eval_every > 0 and step % args.eval_every == 0:
                eval_loss, eval_ppl_v = eval_ppl(
                    model, tok, args.data_dir, args.device, amp_dtype,
                    args.eval_seqlen, args.eval_tokens, eos_id,
                    args.eval_offset_docs
                )
                model.train()
                print(f"[Eval] step {step} | eval_loss {eval_loss:.4f} | eval_ppl {eval_ppl_v:.3f}")

                if eval_ppl_v < best_eval_ppl:
                    best_eval_ppl = eval_ppl_v

            # write CSV
            with open(csv_path, "a") as f:
                f.write(f"{step},{lv:.6f},{loss_ma:.6f},{eval_loss:.6f},{eval_ppl_v:.6f},{sec_per:.4f}\n")

            # overwrite single checkpoint every eval_every (or every 200 if eval disabled)
            if (args.eval_every > 0 and step % args.eval_every == 0) or (args.eval_every <= 0 and step % 200 == 0):
                ck = {
                    "step": step,
                    "loss_ma": loss_ma,
                    "best_eval_ppl": best_eval_ppl,
                    "lora": get_lora_state(model),
                    "opt": opt.state_dict(),
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                }
                atomic_save(ck, ckpt_path)
                print(f"[CKPT] saved (overwrite) -> {ckpt_path}")

    # final save checkpoint
    ck = {
        "step": step,
        "loss_ma": loss_ma,
        "best_eval_ppl": best_eval_ppl,
        "lora": get_lora_state(model),
        "opt": opt.state_dict(),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    atomic_save(ck, ckpt_path)
    print(f"[Done] final ckpt saved -> {ckpt_path}")
    print(f"[Curve] {csv_path}")

if __name__ == "__main__":
    main()
