#!/usr/bin/env python3
import os, json, math, argparse
import torch
from transformers import AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

@torch.no_grad()
def forward_logits(model, input_ids, dtype):
    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model(input_ids=input_ids, use_cache=False)
    return out.logits

def get_input_device(model):
    return next(model.parameters()).device

@torch.no_grad()
def eval_ppl_disjoint(model, testenc_cpu, seqlen, dtype):
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tok = 0
    nsamples = testenc_cpu.numel() // seqlen

    dev = get_input_device(model)
    for i in range(nsamples):
        batch = testenc_cpu[:, i*seqlen:(i+1)*seqlen].to(dev)
        logits = forward_logits(model, batch, dtype).float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_nll += float(nll.item())
        total_tok += shift_labels.numel()

    ppl = math.exp(total_nll / max(1, total_tok))
    return ppl, {"windows": nsamples, "tokens": total_tok}

@torch.no_grad()
def eval_ppl_stride(model, testenc_cpu, seqlen, stride, dtype):
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_nll = 0.0
    total_tok = 0

    dev = get_input_device(model)
    ids = testenc_cpu.to(dev)
    T = ids.shape[1]

    def step_window(begin, end, trg_len):
        nonlocal total_nll, total_tok
        input_ids = ids[:, begin:end]
        target = input_ids.clone()
        target[:, :-trg_len] = -100

        logits = forward_logits(model, input_ids, dtype).float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target[:, 1:].contiguous()
        nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_nll += float(nll.item())
        total_tok += int((shift_labels != -100).sum().item())

    for end in range(stride, T, stride):
        begin = max(0, end - seqlen)
        step_window(begin, end, trg_len=stride)

    rem = T % stride
    if rem != 0:
        end = T
        begin = max(0, end - seqlen)
        step_window(begin, end, trg_len=rem)

    ppl = math.exp(total_nll / max(1, total_tok))
    return ppl, {"stride": stride, "tokens": total_tok}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["base", "cmp"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--low_json", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device_map", default="auto", choices=["auto", "single"])
    ap.add_argument("--single_device", default="cuda:0")
    ap.add_argument("--testenc", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--ppl_mode", choices=["stride", "disjoint"], default="stride")
    ap.add_argument("--stride", type=int, default=512)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    testenc = torch.load(args.testenc, map_location="cpu")
    assert testenc.dim() == 2 and testenc.size(0) == 1, testenc.shape

    dm = "auto" if args.device_map == "auto" else {"": args.single_device}

    if args.model_type == "base":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, local_files_only=True, torch_dtype=dtype,
            low_cpu_mem_usage=True, device_map=dm
        )
    else:
        if not args.low_json:
            raise RuntimeError("--low_json is required for cmp")
        low_rank = json.load(open(args.low_json, "r"))
        model = OPTForCausalLM_EigenAttn.from_pretrained(
            args.model_dir, local_files_only=True, torch_dtype=dtype,
            low_cpu_mem_usage=True, device_map=dm, low_rank_config=low_rank
        )

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    if args.ppl_mode == "stride":
        ppl, info = eval_ppl_stride(model, testenc, args.seqlen, args.stride, dtype)
    else:
        ppl, info = eval_ppl_disjoint(model, testenc, args.seqlen, dtype)

    print(f"[{args.model_type.upper()}] PPL={ppl:.6f} info={info} "
          f"(dtype={args.dtype}, ppl_mode={args.ppl_mode}, seqlen={args.seqlen}, "
          f"stride={args.stride if args.ppl_mode=='stride' else 'NA'})")

if __name__ == "__main__":
    main()
