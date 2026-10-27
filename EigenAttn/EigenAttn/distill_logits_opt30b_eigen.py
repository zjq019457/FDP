#!/usr/bin/env python3
import argparse
import glob
import gzip
import json
import os
import random
from typing import Iterator, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn


def iter_texts_from_dir(data_dir: str, max_files: int, max_lines: int) -> Iterator[str]:
    exts = ["*.txt", "*.jsonl", "*.json", "*.gz"]
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    files = sorted(files)[:max_files]
    n = 0
    for fp in files:
        opener = gzip.open if fp.endswith(".gz") else open
        try:
            with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    txt = None
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            rec = json.loads(line)
                            if isinstance(rec, dict):
                                txt = rec.get("text") or rec.get("content")
                        except Exception:
                            txt = None
                    if txt is None:
                        txt = line
                    if isinstance(txt, str) and txt.strip():
                        yield txt.strip()
                        n += 1
                        if n >= max_lines:
                            return
        except Exception:
            continue


def build_token_stream(tok, texts: Iterator[str], target_tokens: int, seed: int) -> torch.Tensor:
    random.seed(seed)
    ids: List[int] = []
    total = 0
    for t in texts:
        enc = tok(t, add_special_tokens=True)["input_ids"]
        if not enc:
            continue
        ids.extend(enc)
        total += len(enc)
        if total >= target_tokens:
            break
    if total == 0:
        raise RuntimeError("No tokens read from data_dir.")
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(stream_cpu: torch.Tensor, seqlen: int, batch_size: int) -> torch.Tensor:
    n = stream_cpu.numel()
    starts = torch.randint(low=0, high=n - seqlen - 1, size=(batch_size,))
    return torch.stack([stream_cpu[s : s + seqlen] for s in starts], dim=0)


def set_trainable_by_mode(model, mode: str):
    for p in model.parameters():
        p.requires_grad_(False)

    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)
        return

    for layer in model.model.decoder.layers:
        attn = layer.self_attn
        if mode in ("vscale_only", "up_vscale"):
            if hasattr(attn, "vscale_log"):
                try:
                    attn.vscale_log.requires_grad_(True)
                except Exception:
                    pass
        if mode == "up_vscale":
            if hasattr(attn, "out_proj_up"):
                for p in attn.out_proj_up.parameters():
                    p.requires_grad_(True)


def main():
    ap = argparse.ArgumentParser(description="KL+CE distillation for OPT-30B EigenAttn student")
    ap.add_argument("--teacher", type=str, required=True, help="Teacher HF path, e.g. /diskE/opt-30b")
    ap.add_argument("--student", type=str, required=True, help="Student path, e.g. /diskF/.../eigen_adaptive_rebuild")
    ap.add_argument("--low_json", type=str, required=True, help="low_rank_config json for loading student")
    ap.add_argument("--tok", type=str, required=True, help="Tokenizer path")
    ap.add_argument("--data_dir", type=str, required=True, help="Training text directory")
    ap.add_argument("--device", type=str, default="cuda:0", help="single-gpu fallback device")
    ap.add_argument("--teacher_gpus", type=str, default="", help='e.g. "0,1" for teacher model-parallel')
    ap.add_argument("--student_gpus", type=str, default="", help='e.g. "2,3" for student model-parallel')
    ap.add_argument("--per_gpu_mem", type=str, default="22GiB", help='max_memory per target gpu, e.g. "22GiB"')
    ap.add_argument("--non_target_gpu_mem", type=str, default="1GiB", help='max_memory for non-target visible gpus')
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--token_budget", type=int, default=800000)
    ap.add_argument("--max_files", type=int, default=120)
    ap.add_argument("--max_lines", type=int, default=1200000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.3, help="CE weight")
    ap.add_argument("--temperature", type=float, default=2.0, help="KD temperature")
    ap.add_argument(
        "--train_mode",
        type=str,
        default="up_vscale",
        choices=["vscale_only", "up_vscale", "all"],
        help="Which student params to train",
    )
    ap.add_argument(
        "--save_to_student",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: overwrite-save to --student; 0: save to --save_dir",
    )
    ap.add_argument("--save_dir", type=str, default="", help="Used only when --save_to_student 0")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    low_rank = json.load(open(args.low_json, "r", encoding="utf-8"))

    def _parse_gpu_list(spec: str):
        s = (spec or "").strip()
        if not s:
            return []
        return [int(x) for x in s.split(",") if x.strip() != ""]

    tgpus = _parse_gpu_list(args.teacher_gpus)
    sgpus = _parse_gpu_list(args.student_gpus)

    teacher_input_device = args.device
    student_input_device = args.device

    def _build_max_memory(target_gpus: List[int], per_target: str, per_non_target: str):
        nvis = torch.cuda.device_count()
        mm = {i: per_non_target for i in range(nvis)}
        for i in target_gpus:
            if 0 <= i < nvis:
                mm[i] = per_target
        mm["cpu"] = "256GiB"
        return mm

    print("[Load] teacher ...")
    if tgpus:
        t_mem = _build_max_memory(tgpus, args.per_gpu_mem, args.non_target_gpu_mem)
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher,
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto",
            max_memory=t_mem,
            low_cpu_mem_usage=True,
        )
        teacher_input_device = f"cuda:{tgpus[0]}"
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher,
            local_files_only=True,
            torch_dtype=dtype,
            device_map={"": args.device},
            low_cpu_mem_usage=True,
        )
    teacher.eval()
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("[Load] student ...")
    if sgpus:
        s_mem = _build_max_memory(sgpus, args.per_gpu_mem, args.non_target_gpu_mem)
        student = OPTForCausalLM_EigenAttn.from_pretrained(
            args.student,
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto",
            max_memory=s_mem,
            low_cpu_mem_usage=True,
            low_rank_config=low_rank,
        )
        student_input_device = f"cuda:{sgpus[0]}"
    else:
        student = OPTForCausalLM_EigenAttn.from_pretrained(
            args.student,
            local_files_only=True,
            torch_dtype=dtype,
            device_map={"": args.device},
            low_cpu_mem_usage=True,
            low_rank_config=low_rank,
        )
    # Training with meta/offloaded params is unstable; fail fast with actionable hint.
    meta_student = sum(1 for _, p in student.named_parameters() if str(p.device) == "meta")
    if meta_student > 0:
        raise RuntimeError(
            f"Student has {meta_student} meta parameters after loading. "
            "Increase --per_gpu_mem, reduce model overlap, or avoid auto offload for training."
        )
    student.config.use_cache = False
    set_trainable_by_mode(student, args.train_mode)
    student.train()

    trainable = [p for p in student.parameters() if p.requires_grad]
    print(f"[Trainable] {sum(p.numel() for p in trainable):,}")
    if not trainable:
        raise RuntimeError("No trainable parameters. Check --train_mode.")

    texts = iter_texts_from_dir(args.data_dir, args.max_files, args.max_lines)
    stream_cpu = build_token_stream(tok, texts, args.token_budget, args.seed)
    print(f"[Data] token_stream={stream_cpu.numel()}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    t = float(args.temperature)
    alpha = float(args.alpha)

    opt.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        x_cpu = sample_batch(stream_cpu, args.seqlen, args.batch_size)
        x_t = x_cpu.to(teacher_input_device, non_blocking=True)
        x_s = x_cpu.to(student_input_device, non_blocking=True)
        labels = x_s

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=dtype):
                t_logits = teacher(input_ids=x_t, use_cache=False).logits
                t_logits = t_logits.to(student_input_device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=dtype):
            s_logits = student(input_ids=x_s, use_cache=False).logits
            shift_s = s_logits[:, :-1, :].contiguous()
            shift_t = t_logits[:, :-1, :].contiguous()
            shift_y = labels[:, 1:].contiguous()

            ce = F.cross_entropy(
                shift_s.view(-1, shift_s.size(-1)),
                shift_y.view(-1),
                reduction="mean",
            )
            kd = F.kl_div(
                F.log_softmax(shift_s / t, dim=-1),
                F.softmax(shift_t / t, dim=-1),
                reduction="batchmean",
            ) * (t * t)
            loss = alpha * ce + (1.0 - alpha) * kd
            loss = loss / max(1, args.grad_accum)

        loss.backward()
        if step % args.grad_accum == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 20 == 0:
            print(f"[Step {step:5d}] loss={float(loss.detach().item()):.6f} (alpha={alpha}, T={t})")

    save_dir = args.student if int(args.save_to_student) == 1 else (args.save_dir or "")
    if not save_dir:
        raise ValueError("When --save_to_student 0, --save_dir must be set.")
    os.makedirs(save_dir, exist_ok=True)
    student.save_pretrained(save_dir)
    try:
        tok.save_pretrained(save_dir)
    except Exception:
        pass
    print(f"[Saved] student -> {save_dir}")


if __name__ == "__main__":
    main()

