import os, glob, json, shutil, argparse, math, random
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

def find_low_json(search_root: str):
    cands = []
    for fp in glob.glob(os.path.join(search_root, "**", "*.json"), recursive=True):
        name = os.path.basename(fp).lower()
        score = 0
        if "low_rank" in name: score += 5
        if "layerwise" in name: score += 3
        if "rank" in name: score += 2
        # try quick structural check
        try:
            obj = json.load(open(fp, "r"))
            # heuristic: config usually is a dict and has lots of entries
            if isinstance(obj, dict):
                score += 1
                if len(obj) >= 10: score += 1
            elif isinstance(obj, list):
                score += 1
        except Exception:
            continue
        cands.append((score, fp))
    cands.sort(reverse=True, key=lambda x: x[0])
    return cands[0][1] if cands else None

def shard_has_key(safetensor_path: str, substr: str) -> bool:
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if substr in k:
                return True
    return False

def patch_model(
    in_dir: str,
    out_dir: str,
    set_vscale1: bool = True,
    sanitize_nan_inf: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    # copy non-weight files
    for fn in os.listdir(in_dir):
        src = os.path.join(in_dir, fn)
        dst = os.path.join(out_dir, fn)
        if os.path.isdir(src):
            continue
        if fn.endswith(".safetensors"):
            continue
        shutil.copy2(src, dst)

    shards = sorted(glob.glob(os.path.join(in_dir, "*.safetensors")))
    assert shards, f"No *.safetensors in {in_dir}"

    patched = 0
    linked = 0

    for fp in shards:
        out_fp = os.path.join(out_dir, os.path.basename(fp))

        need_patch = False
        if set_vscale1 and shard_has_key(fp, "vscale_log"):
            need_patch = True
        if sanitize_nan_inf:
            # cheap heuristic: only load shard if we already need patch OR if you want full scan
            # here we only sanitize when patching to keep it light
            pass

        if not need_patch:
            # symlink unchanged shard to save disk
            try:
                if os.path.exists(out_fp):
                    os.remove(out_fp)
                os.symlink(fp, out_fp)
            except Exception:
                shutil.copy2(fp, out_fp)
            linked += 1
            continue

        # load & patch
        sd = load_file(fp)
        changed = False

        for k in list(sd.keys()):
            t = sd[k]
            if set_vscale1 and ("vscale_log" in k):
                # set log(vscale)=0 => vscale=1
                sd[k] = torch.zeros_like(t, dtype=torch.float32)
                changed = True
            if sanitize_nan_inf:
                if torch.isnan(t).any() or torch.isinf(t).any():
                    sd[k] = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                    changed = True

        if not changed:
            # fallback: symlink/copy
            try:
                if os.path.exists(out_fp):
                    os.remove(out_fp)
                os.symlink(fp, out_fp)
            except Exception:
                shutil.copy2(fp, out_fp)
            linked += 1
        else:
            save_file(sd, out_fp, metadata={"format": "pt"})
            patched += 1

    print(f"[Patch] out_dir={out_dir}")
    print(f"[Patch] patched_shards={patched}, linked_or_copied_shards={linked}")

@torch.no_grad()
def ppl_on_stream(model, stream, device, seqlen=2048, stride=2048, dtype=torch.bfloat16):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    nll_sum = 0.0
    tok_count = 0
    N = stream.numel()

    for start in range(0, N, stride):
        end = min(start + seqlen, N)
        if end - start < 2:
            break
        inp = stream[start:end].unsqueeze(0).to(device)
        attn = torch.ones_like(inp, dtype=torch.long)
        labels = inp.clone()  # ✅ avoid double-shift
        with torch.autocast("cuda", dtype=dtype):
            out = model(input_ids=inp, attention_mask=attn, labels=labels)
        n = inp.numel() - 1
        nll_sum += float(out.loss) * n
        tok_count += n
        if end >= N:
            break

    return math.exp(nll_sum / max(tok_count, 1))

def build_stream(tok, data_dir, target_tokens=200000, seed=3, max_files=40, max_lines=50000):
    import gzip
    random.seed(seed)
    files=[]
    for e in ["*.jsonl","*.txt","*.gz"]:
        files += glob.glob(os.path.join(data_dir,"**",e), recursive=True)
    files = sorted(files)[:max_files]

    ids=[]
    nline=0
    for fp in files:
        opener = gzip.open if fp.endswith(".gz") else open
        with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                t=line
                if line.startswith("{") and line.endswith("}"):
                    try:
                        j=json.loads(line)
                        t=j.get("text") or j.get("content") or line
                    except:
                        pass
                if not isinstance(t, str) or not t.strip():
                    continue
                enc = tok(t, add_special_tokens=False)["input_ids"]
                if enc:
                    ids += enc + [tok.eos_token_id]
                    if len(ids) >= target_tokens:
                        return torch.tensor(ids[:target_tokens], dtype=torch.long)
                nline += 1
                if nline >= max_lines:
                    break
    return torch.tensor(ids[:target_tokens], dtype=torch.long)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_model", required=True)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--search_root", default=None, help="Where to search low_rank_config json (default: parent of in_model)")
    ap.add_argument("--low_json", default="auto")
    ap.add_argument("--set_vscale1", type=int, default=1)
    ap.add_argument("--sanitize_nan_inf", type=int, default=1)

    ap.add_argument("--base_dir", default="/diskE/opt-30b")
    ap.add_argument("--tok_dir", default="/diskE/opt-30b")
    ap.add_argument("--data_dir", default="/diskE/jiaqizhang/c4-en/en")
    ap.add_argument("--device_base", default="cuda:0")
    ap.add_argument("--device_cmp", default="cuda:1")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=3)

    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    search_root = args.search_root
    if search_root is None:
        search_root = os.path.dirname(args.in_model.rstrip("/"))

    if args.low_json == "auto":
        low_json = find_low_json(search_root)
        if low_json is None:
            raise RuntimeError(f"Cannot find any usable low_rank_config json under: {search_root}")
    else:
        low_json = args.low_json

    print("[Info] using low_json:", low_json)

    patch_model(
        args.in_model,
        args.out_model,
        set_vscale1=bool(args.set_vscale1),
        sanitize_nan_inf=bool(args.sanitize_nan_inf),
    )

    # quick NaN check + PPL check
    tok = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    stream = build_stream(tok, args.data_dir, target_tokens=args.tokens, seed=args.seed)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_dir, local_files_only=True, torch_dtype=dtype,
        device_map={"": args.device_base}, low_cpu_mem_usage=True
    ).eval()
    low = json.load(open(low_json, "r"))
    cmp = OPTForCausalLM_EigenAttn.from_pretrained(
        args.out_model, local_files_only=True, torch_dtype=dtype,
        device_map={"": args.device_cmp}, low_cpu_mem_usage=True,
        low_rank_config=low
    ).eval()

    # 1-step NaN check
    enc = tok("Hello, my name is", return_tensors="pt")
    enc = {k: v.to(args.device_cmp) for k, v in enc.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out = cmp(**enc, labels=enc["input_ids"])
    print("[Check] loss finite:", bool(torch.isfinite(out.loss).detach().cpu()))
    print("[Check] logits has_nan:", bool(torch.isnan(out.logits).any().detach().cpu()))
    print("[Check] logits has_inf:", bool(torch.isinf(out.logits).any().detach().cpu()))

    bp = ppl_on_stream(base, stream, args.device_base, dtype=dtype)
    cp = ppl_on_stream(cmp,  stream, args.device_cmp,  dtype=dtype)
    print("[PPL] BASE:", bp)
    print("[PPL] CMP :", cp)

if __name__ == "__main__":
    main()