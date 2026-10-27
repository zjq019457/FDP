#sanity_all_opt30b_eigen.py
import os, json, gc, argparse
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 你的工程模型类
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

# -------------------------
# helpers
# -------------------------
def q(x, p):
    return torch.quantile(x, p).item()

def summarize_tensor(x: torch.Tensor):
    x = x.detach().float().flatten().cpu()
    return {
        "absmax": x.abs().max().item(),
        "mean": x.mean().item(),
        "std": x.std().item(),
        "p999_abs": q(x.abs(), 0.999),
        "p99_abs": q(x.abs(), 0.99),
    }

def print_kv(title, d):
    keys = ["absmax", "p999_abs", "p99_abs", "std", "mean"]
    s = " ".join([f"{k}={d[k]:.6g}" for k in keys if k in d])
    print(f"{title}: {s}")

def cleanup(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def autocast_for(model, device: torch.device):
    # cuda 上才 autocast；CPU 直接跑
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=next(model.parameters()).dtype)
    return nullcontext()

def load_base(base_dir, dtype, device, low_cpu_mem_usage=True):
    m = AutoModelForCausalLM.from_pretrained(
        base_dir,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    return m.to(device).eval()

def load_cmp(cmp_dir, low_json, dtype, device, low_cpu_mem_usage=True):
    cfg = json.load(open(low_json))
    m = OPTForCausalLM_EigenAttn.from_pretrained(
        cmp_dir,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        low_rank_config=cfg,
    )
    return m.to(device).eval()

def ensure_dtype(m, dtype):
    # 统一 dtype，避免 mat1/mat2 dtype mismatch
    return m.to(dtype=dtype)

def clone_inputs_for_devices(inp, dev0: torch.device, dev1: torch.device):
    # 返回两份互不共享 storage 的 inputs，分别搬到两个 device
    inp0 = {k: v.clone().to(dev0) for k, v in inp.items()}
    inp1 = {k: v.clone().to(dev1) for k, v in inp.items()}
    return inp0, inp1

# -------------------------
# tests
# -------------------------
@torch.no_grad()
def test_load_info(base_dir, cmp_dir, low_json, dtype):
    print("\n[1] Load-info check (missing/unexpected/mismatched)")
    b, binfo = AutoModelForCausalLM.from_pretrained(
        base_dir,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=None,
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    print(
        f"BASE missing={len(binfo.get('missing_keys', []))} "
        f"unexpected={len(binfo.get('unexpected_keys', []))} "
        f"mismatched={len(binfo.get('mismatched_keys', []))}"
    )
    cleanup(b)

    cfg = json.load(open(low_json))
    c, cinfo = OPTForCausalLM_EigenAttn.from_pretrained(
        cmp_dir,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=None,
        output_loading_info=True,
        low_cpu_mem_usage=True,
        low_rank_config=cfg,
    )
    print(
        f"CMP  missing={len(cinfo.get('missing_keys', []))} "
        f"unexpected={len(cinfo.get('unexpected_keys', []))} "
        f"mismatched={len(cinfo.get('mismatched_keys', []))}"
    )
    if len(cinfo.get("missing_keys", [])) or len(cinfo.get("unexpected_keys", [])):
        print("CMP missing_head:", cinfo.get("missing_keys", [])[:12])
        print("CMP unexpected_head:", cinfo.get("unexpected_keys", [])[:12])
    cleanup(c)

@torch.no_grad()
def test_tied_weights(base_dir, cmp_dir, low_json, dtype):
    print("\n[2] Tied-weights check (embed_tokens <-> lm_head)")
    base = AutoModelForCausalLM.from_pretrained(
        base_dir, local_files_only=True, torch_dtype=dtype, device_map={"": "cpu"}
    )
    cfg = json.load(open(low_json))
    cmp = OPTForCausalLM_EigenAttn.from_pretrained(
        cmp_dir, local_files_only=True, torch_dtype=dtype, device_map={"": "cpu"}, low_rank_config=cfg
    )

    b_tied = (base.model.decoder.embed_tokens.weight.data_ptr() == base.lm_head.weight.data_ptr())
    c_tied = (cmp.model.decoder.embed_tokens.weight.data_ptr() == cmp.lm_head.weight.data_ptr())
    print("BASE tied:", b_tied, " CMP tied:", c_tied)
    print("lm_head shapes:", tuple(base.lm_head.weight.shape), tuple(cmp.lm_head.weight.shape))
    cleanup(base, cmp)

@torch.no_grad()
def test_state_dict_scan(cmp_dir, low_json, dtype, scan_full=False):
    print("\n[3] State-dict numeric scan (NaN/Inf + extreme values)")
    cfg = json.load(open(low_json))
    m = OPTForCausalLM_EigenAttn.from_pretrained(
        cmp_dir, local_files_only=True, torch_dtype=dtype, device_map={"": "cpu"}, low_rank_config=cfg
    )
    sd = m.state_dict()

    nan_inf = []
    big = []
    for k, v in sd.items():
        if not torch.is_tensor(v) or v.numel() == 0:
            continue
        if (not scan_full) and ("vscale_log" not in k) and (v.numel() > 5_000_000):
            continue

        x = v.detach().float()
        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_inf.append(k)
        mx = x.abs().max().item()
        if mx > 1e4:
            big.append((k, mx, tuple(v.shape)))

    print("nan/inf count:", len(nan_inf))
    if nan_inf:
        print("nan/inf head:", nan_inf[:10])
    print("absmax>1e4 count:", len(big))
    if big:
        print("big head:", big[:10])

    vs = [(k, float(sd[k].detach().float().cpu().item())) for k in sd if k.endswith("vscale_log")]
    if vs:
        vals = torch.tensor([v for _, v in vs])
        print(
            "vscale_log: count", len(vs),
            "min", vals.min().item(),
            "max", vals.max().item(),
            "p50", torch.quantile(vals, 0.5).item(),
            "p95", torch.quantile(vals, 0.95).item(),
        )
        top = sorted(vs, key=lambda kv: abs(kv[1]), reverse=True)[:5]
        print("vscale_log top5 abs:", top)

    cleanup(m)

@torch.no_grad()
def test_forward_smoke(cmp, tok, cmp_dev: torch.device, text, max_new_tokens=16):
    print("\n[4] Forward + generation smoke test (CMP only)")
    inp = tok(text, return_tensors="pt")
    inp = {k: v.to(cmp_dev) for k, v in inp.items()}

    with autocast_for(cmp, cmp_dev):
        out = cmp(**inp, return_dict=True)
    logits = out.logits
    s = summarize_tensor(logits)
    print("CMP logits shape:", tuple(logits.shape), "dtype:", logits.dtype)
    print_kv("CMP logits", s)
    print("CMP logits_nan:", torch.isnan(logits.float()).any().item())

    with autocast_for(cmp, cmp_dev):
        gen = cmp.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    print("CMP gen:", tok.decode(gen[0], skip_special_tokens=True))

@torch.no_grad()
def test_attn_io(base, cmp, tok, base_dev: torch.device, cmp_dev: torch.device, layers):
    print("\n[5] Attention IO stats (self_attn input/output) for layers:", layers)
    raw = tok("Hello, my name is", return_tensors="pt")
    inp_base, inp_cmp = clone_inputs_for_devices(raw, base_dev, cmp_dev)

    def attach(model, rec):
        hooks = []
        for L in layers:
            attn = model.model.decoder.layers[L].self_attn

            def pre_hook(mod, args, kwargs, L=L):
                hs = kwargs.get("hidden_states", None)
                if hs is None and len(args) > 0:
                    hs = args[0]
                rec[L] = rec.get(L, {})
                rec[L]["in"] = summarize_tensor(hs)

            def fwd_hook(mod, args, out, L=L):
                rec[L] = rec.get(L, {})
                attn_out = out[0] if isinstance(out, (tuple, list)) else out
                rec[L]["out"] = summarize_tensor(attn_out)

            hooks.append(attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
            hooks.append(attn.register_forward_hook(fwd_hook))
        return hooks

    rb, rc = {}, {}
    hb = attach(base, rb)
    hc = attach(cmp, rc)

    with autocast_for(base, base_dev):
        _ = base(**inp_base, return_dict=True)
    with autocast_for(cmp, cmp_dev):
        _ = cmp(**inp_cmp, return_dict=True)

    for h in hb + hc:
        h.remove()

    for L in layers:
        bi, bo = rb[L]["in"], rb[L]["out"]
        ci, co = rc[L]["in"], rc[L]["out"]
        print(f"\n  Layer {L}")
        print_kv("    base in ", bi); print_kv("    cmp  in ", ci)
        print_kv("    base out", bo); print_kv("    cmp  out", co)
        r_in = ci["p999_abs"] / (bi["p999_abs"] + 1e-12)
        r_out = co["p999_abs"] / (bo["p999_abs"] + 1e-12)
        print(f"    ratio p99.9_abs  in={r_in:.3f} out={r_out:.3f}")

@torch.no_grad()
def test_attn_weights(base, cmp, tok, base_dev: torch.device, cmp_dev: torch.device, layers):
    print("\n[6] Attention weights distribution (output_attentions=True) layers:", layers)
    raw = tok("Hello, my name is", return_tensors="pt")
    inp_base, inp_cmp = clone_inputs_for_devices(raw, base_dev, cmp_dev)

    with autocast_for(base, base_dev):
        bout = base(**inp_base, output_attentions=True, return_dict=True)
    with autocast_for(cmp, cmp_dev):
        cout = cmp(**inp_cmp, output_attentions=True, return_dict=True)

    for L in layers:
        bw = bout.attentions[L]
        cw = cout.attentions[L]
        bwf = bw.detach().float().flatten().cpu()
        cwf = cw.detach().float().flatten().cpu()
        b = (q(bwf, 0.999), q(bwf, 0.001), bwf.std().item(), bwf.mean().item())
        c = (q(cwf, 0.999), q(cwf, 0.001), cwf.std().item(), cwf.mean().item())
        print(f"  Layer {L} attn_w base q999/q001/std/mean = {b[0]:.6g} {b[1]:.6g} {b[2]:.6g} {b[3]:.6g}")
        print(f"          attn_w cmp  q999/q001/std/mean = {c[0]:.6g} {c[1]:.6g} {c[2]:.6g} {c[3]:.6g}")

@torch.no_grad()
def test_hidden_and_logits_compare(base, cmp, tok, base_dev: torch.device, cmp_dev: torch.device):
    print("\n[7] Compare last_hidden_state + logits (base vs cmp) on same prompt")
    raw = tok("Hello, my name is", return_tensors="pt")
    inp_base, inp_cmp = clone_inputs_for_devices(raw, base_dev, cmp_dev)

    with autocast_for(base, base_dev):
        bout = base(**inp_base, output_hidden_states=True, return_dict=True)
    with autocast_for(cmp, cmp_dev):
        cout = cmp(**inp_cmp, output_hidden_states=True, return_dict=True)

    b_last = bout.hidden_states[-1]
    c_last = cout.hidden_states[-1]
    b_logits = bout.logits
    c_logits = cout.logits

    sb = summarize_tensor(b_last); sc = summarize_tensor(c_last)
    print_kv("last_hidden BASE", sb)
    print_kv("last_hidden CMP ", sc)
    print("ratio last_hidden p99.9_abs:", sc["p999_abs"] / (sb["p999_abs"] + 1e-12))

    lb = summarize_tensor(b_logits); lc = summarize_tensor(c_logits)
    print_kv("logits BASE", lb)
    print_kv("logits CMP ", lc)
    print("ratio logits p99.9_abs:", lc["p999_abs"] / (lb["p999_abs"] + 1e-12))

    # ⚠️ 两个 logits 在不同 device，上 CPU 再做差
    b_cpu = b_logits.detach().float().cpu()
    c_cpu = c_logits.detach().float().cpu()
    diff = (b_cpu - c_cpu).abs()
    print("logits diff mean:", diff.mean().item())
    print("logits diff max :", diff.max().item())

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--tok", required=True)

    # ✅ 新增：分别指定 base/cmp 放在哪张卡（配合 CUDA_VISIBLE_DEVICES=2,3）
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device",  default="cuda:1")

    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--layers", default="0,1,9,19,47", help="comma-separated layer indices")
    ap.add_argument("--prompt", default="Hello, my name is")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--scan_full_state", type=int, default=0, help="1 = scan full state_dict on CPU (slow)")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]

    base_dev = torch.device(args.base_device)
    cmp_dev  = torch.device(args.cmp_device)
    if base_dev.type == "cuda" and cmp_dev.type == "cuda" and base_dev.index == cmp_dev.index:
        raise ValueError(f"base_device and cmp_device are the same ({base_dev}). This will OOM on OPT-30B.")

    # (1) load-info checks on CPU
    test_load_info(args.base, args.cmp, args.low_json, dtype)
    test_tied_weights(args.base, args.cmp, args.low_json, dtype)
    test_state_dict_scan(args.cmp, args.low_json, dtype, scan_full=bool(args.scan_full_state))

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # (2) load models onto different devices (avoid stacking on one GPU)
    base = load_base(args.base, dtype, base_dev)
    cmp  = load_cmp(args.cmp, args.low_json, dtype, cmp_dev)
    base = ensure_dtype(base, dtype)
    cmp  = ensure_dtype(cmp, dtype)

    # (3) run tests requiring forward
    test_forward_smoke(cmp, tok, cmp_dev, args.prompt, max_new_tokens=args.max_new_tokens)
    test_attn_io(base, cmp, tok, base_dev, cmp_dev, layers)
    test_attn_weights(base, cmp, tok, base_dev, cmp_dev, layers)
    test_hidden_and_logits_compare(base, cmp, tok, base_dev, cmp_dev)

    cleanup(base, cmp)
    print("\n[Done] sanity suite completed.")

if __name__ == "__main__":
    main()
