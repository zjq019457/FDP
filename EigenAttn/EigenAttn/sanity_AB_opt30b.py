import os, json, glob, gzip, argparse, math, random
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn

# ---- make attention path stable ----
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

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
    """device='auto' shards across all visible GPUs (needed for OPT-30B fp16 ≈ 60 GB)."""
    device_map = "auto" if str(device).lower() == "auto" else {"": device}
    return AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem,
        device_map=device_map,
    )


def load_tokenizer_auto(model_path: str):
    """
    Llama/OPT: use_fast=False matches repo defaults and avoids fast-tokenizer quirks.
    MPT-7B: tokenizer_config names GPTNeoXTokenizer (slow); removed in recent transformers.
    Fallback to use_fast=True → GPTNeoXTokenizerFast from tokenizer.json.
    """
    kw = dict(local_files_only=True, trust_remote_code=True)
    try:
        return AutoTokenizer.from_pretrained(model_path, use_fast=False, **kw)
    except ValueError as e:
        msg = str(e)
        if "Tokenizer class" in msg or "does not exist" in msg or "not currently imported" in msg:
            return AutoTokenizer.from_pretrained(model_path, use_fast=True, **kw)
        raise


def _detect_cmp_arch(cmp_dir: str) -> str:
    cfg_path = os.path.join(cmp_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    mt = str(cfg.get("model_type", "")).lower()
    if mt == "opt":
        return "opt"
    if mt in ("llama", "llama4", "mistral", "qwen2", "mixtral"):
        return "llama"
    raise ValueError(
        f"Unsupported model_type={mt!r} in {cfg_path}; pass --arch opt|llama after editing sanity_AB_opt30b.py if needed."
    )


def infer_low_rank_config_llama_from_shards(cmp_dir: str):
    """
    Llama EigenAttn saves weights but may omit low_rank_config in config.json.
    Recover [[rank_kq, rank_v], ...] from k_proj_low / v_proj_low output dims in safetensors.
    """
    cfg_path = os.path.join(cmp_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if str(cfg.get("model_type", "")).lower() != "llama":
        return None
    n_layers = int(cfg.get("num_hidden_layers", 0) or 0)
    if n_layers <= 0:
        return None

    try:
        from safetensors import safe_open
    except ImportError:
        return None

    index_path = os.path.join(cmp_dir, "model.safetensors.index.json")
    single_path = os.path.join(cmp_dir, "model.safetensors")

    def _rank_pair(i: int, sf_get):
        kn = f"model.layers.{i}.self_attn.k_proj_low.weight"
        vn = f"model.layers.{i}.self_attn.v_proj_low.weight"
        return [int(sf_get(kn).shape[0]), int(sf_get(vn).shape[0])]

    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        def _get_tensor(name: str):
            shard = os.path.join(cmp_dir, weight_map[name])
            with safe_open(shard, framework="pt", device="cpu") as sf:
                return sf.get_tensor(name)

        try:
            return [_rank_pair(i, _get_tensor) for i in range(n_layers)]
        except (KeyError, FileNotFoundError, ValueError):
            return None

    if os.path.isfile(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as sf:

            def _get(name: str):
                return sf.get_tensor(name)

            try:
                return [_rank_pair(i, _get) for i in range(n_layers)]
            except (KeyError, ValueError):
                return None

    return None


def load_cmp_model(
    path,
    dtype,
    device,
    low_rank_config,
    arch: str = "auto",
    low_cpu_mem=True,
    device_map=None,
):
    if arch == "auto":
        arch = _detect_cmp_arch(path)
    dm = device_map if device_map is not None else {"": device}
    if arch == "opt":
        from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn

        return OPTForCausalLM_EigenAttn.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem,
            device_map=dm,
            low_rank_config=low_rank_config,
        )
    if arch == "llama":
        return LlamaForCausalLM_EigenAttn.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem,
            device_map=dm,
            low_rank_config=low_rank_config,
        )
    raise ValueError(f"Unknown --arch={arch!r} (use auto|opt|llama)")

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
        # Match [B] eval_ppl_utkarsh_opt: full-sequence forward without KV cache. Using default
        # use_cache=True here leaves DynamicCache / module past_key_value state and can inflate CMP PPL afterward.
        base_inp["use_cache"] = False
        cmp_inp["use_cache"] = False

        # keep your autocast (debug-friendly)
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
# C4 (local shard) helpers
# -------------------------
def _iter_jsonl_texts_gz(fp, text_key="text", read_size=1 << 20):
    """
    Robust C4 reader for .json.gz that may be:
      - jsonl.gz: one JSON object per line (typical c4-train shards)
      - concatenated JSON objects without newlines (c4-validation style shards)
      - JSON-seq where objects can be separated by 0x1e record separator
    Yields: obj[text_key] (string, stripped)
    """
    dec = json.JSONDecoder()
    buf = ""

    def _skip_seps(s, i, n):
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i < n and s[i] == "\x1e":
            i += 1
        return i

    with gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(read_size)
            if not chunk:
                break
            buf += chunk

            i = 0
            n = len(buf)
            while True:
                i = _skip_seps(buf, i, n)
                if i >= n:
                    break
                try:
                    obj, j = dec.raw_decode(buf, i)
                except json.JSONDecodeError:
                    break
                i = j

                if isinstance(obj, dict):
                    v = obj.get(text_key, None)
                    if isinstance(v, str):
                        t = v.strip()
                        if t:
                            yield t

            buf = buf[i:]

        buf = buf.strip()
        if buf:
            i = 0
            n = len(buf)
            while True:
                i = _skip_seps(buf, i, n)
                if i >= n:
                    break
                try:
                    obj, j = dec.raw_decode(buf, i)
                except json.JSONDecodeError:
                    break
                i = j
                if isinstance(obj, dict):
                    v = obj.get(text_key, None)
                    if isinstance(v, str):
                        t = v.strip()
                        if t:
                            yield t

def build_doc_pool_from_c4_train_dir(
    data_dir: str,
    glob_pat: str = "c4-train.*.json.gz",
    pool_docs: int = 20000,
    scan_lines: int = 200000,
    seed: int = 2,
):
    """
    We cannot random-access inside a huge .json.gz easily.
    So we:
      - stream first `scan_lines` docs
      - do reservoir sampling -> keep `pool_docs` docs uniformly over scanned prefix
    This yields a stable pool for later "Utkarsh-style valenc sampling".
    """
    files = sorted(glob.glob(os.path.join(data_dir, glob_pat)))
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pat} under {data_dir}")

    rng = random.Random(seed)
    reservoir = []
    seen = 0

    for fp in files:
        for t in _iter_jsonl_texts_gz(fp, text_key="text"):
            seen += 1
            if len(reservoir) < pool_docs:
                reservoir.append(t)
            else:
                j = rng.randint(1, seen)
                if j <= pool_docs:
                    reservoir[j - 1] = t
            if seen >= scan_lines:
                break
        if seen >= scan_lines:
            break

    if not reservoir:
        raise RuntimeError(f"Failed to read any text from {files[0]}")
    return reservoir, {"seen_docs": seen, "pool_docs": len(reservoir), "file": files[0]}

@torch.no_grad()
def build_utkarsh_like_valenc_from_pool(
    tok,
    doc_pool,
    seqlen: int,
    n_chunks: int = 256,
    val_seed: int = 0,
    max_tries_per_chunk: int = 2000,
):
    """
    Match Utkarsh get_c4() validation construction:
      random.seed(0)
      repeat 256 times:
        - pick random doc
        - tokenize
        - if token_len > seqlen: take a random window length seqlen
      valenc = torch.hstack(windows)  -> [1, 256*seqlen]
    """
    rng = random.Random(val_seed)
    valenc = []

    for k in range(n_chunks):
        ok = False
        for _ in range(max_tries_per_chunk):
            text = doc_pool[rng.randint(0, len(doc_pool) - 1)]
            tmp = tok(text, return_tensors="pt")  # default add_special_tokens behavior (same as repo)
            L = tmp.input_ids.shape[1]
            if L > seqlen:
                i = rng.randint(0, L - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"Unable to sample a doc with token_len > seqlen after {max_tries_per_chunk} tries. "
                f"Try increasing --c4_scan_lines / --c4_pool_docs."
            )

    return torch.hstack(valenc)  # [1, n_chunks*seqlen]

# -------------------------
# Utkarsh-aligned OPT PPL
# -------------------------
@torch.no_grad()
def eval_ppl_utkarsh_opt(
    model,
    testenc: torch.Tensor,   # [1, T], where T is multiple of seqlen (ideally 256*seqlen)
    device: str,
    seqlen: int,
    limit: int = -1,
):
    """
    Utkarsh-aligned PPL chunking (same NLL aggregation as main_eigen_attn._eval_ppl_utkarsh_opt).

    - OPT: logits = lm_head(model.model.decoder(batch)[0])
    - Llama / other HF CausalLM EigenAttn: logits = model(input_ids=batch).logits
    """
    model.eval()
    _clear_attn_past_key_value_attr(model)
    use_cache = getattr(model.config, "use_cache", False) if hasattr(model, "config") else False
    if hasattr(model, "config"):
        model.config.use_cache = False

    if str(device).lower() == "auto":
        lm = getattr(model, "model", None)
        if lm is not None and hasattr(lm, "decoder") and hasattr(lm.decoder, "embed_tokens"):
            device = lm.decoder.embed_tokens.weight.device
        elif lm is not None and hasattr(lm, "embed_tokens"):
            device = lm.embed_tokens.weight.device
        else:
            device = next(model.parameters()).device

    nsamples = testenc.numel() // seqlen
    loss_fct = torch.nn.CrossEntropyLoss()
    nlls = []
    used = 0
    lm_inner = getattr(model, "model", None)
    is_opt = lm_inner is not None and hasattr(lm_inner, "decoder")
    cfg_mt = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    is_mpt = cfg_mt == "mpt" or model.__class__.__name__ == "MPTForCausalLM"

    for i in range(nsamples):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(device)

        if is_opt:
            outputs = model.model.decoder(batch)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)
            # fp16 numeric guard: same as main_eigen_attn._eval_ppl_utkarsh_opt. Without this, |logit| can
            # blow up in CE and every chunk yields non-finite loss (used_chunks=0, PPL=nan).
            # Do not apply to Llama: large vocab often has |logit|>20 legitimately; clamp skews PPL there.
            logits = torch.clamp(logits, -20, 20)
        else:
            fwd_kw = dict(input_ids=batch, use_cache=False)
            # Vendored MPTModel.forward bug: attention_mask=None uses inputs_embeds.device
            # while inputs_embeds is None when only input_ids is given.
            if is_mpt:
                fwd_kw["attention_mask"] = torch.ones(
                    batch.shape[0], batch.shape[1], dtype=torch.bool, device=batch.device
                )
            out = model(**fwd_kw)
            logits = out.logits

        # Llama/MPT: no logit clamp (Llama-3 large vocab: |logit|>20 is common; clamp skews PPL).
        # OPT: clamp applied above (matches main_eigen_attn._eval_ppl_utkarsh_opt).
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:].to(logits.device)

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
        )
        if not torch.isfinite(loss).all():
            continue
        nlls.append(loss.float() * seqlen)
        used += 1
        if i == limit:
            break

    if used == 0:
        if hasattr(model, "config"):
            model.config.use_cache = use_cache
        return float("nan"), {"nsamples": nsamples, "used_chunks": 0}

    ppl = torch.exp(torch.stack(nlls).sum() / (used * seqlen))

    if hasattr(model, "config"):
        model.config.use_cache = use_cache

    return ppl.item(), {"nsamples": nsamples, "used_chunks": used}

def _wikitext_te_to_tensor(te):
    """get_wikitext2 returns a HF BatchEncoding; eval_ppl_utkarsh_opt expects a [1,T] LongTensor."""
    if hasattr(te, "input_ids"):
        return te.input_ids
    if isinstance(te, torch.Tensor):
        return te
    raise TypeError(f"Unexpected wikitext testenc type: {type(te)}")


def _clear_attn_past_key_value_attr(model) -> None:
    """
    LlamaEigenAttention uses `past_key_value = getattr(self, "past_key_value", past_key_value)`; if [A]
    compare_prompts ran with use_cache=True, modules may retain a stale attribute and corrupt later PPL
    even when forward passes use_cache=False. Strip it before eval.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers:
        return
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "past_key_value"):
            try:
                delattr(attn, "past_key_value")
            except Exception:
                setattr(attn, "past_key_value", None)


def ppl_sanity_utkarsh_c4_train(base, cmp, tok, args):
    print("\n[B] PPL sanity (Utkarsh-aligned; C4 pool from --data_dir + --c4_glob)\n")

    def _wiki_enc():
        from datautils import get_loaders

        print("[WikiText-2] loading test tokenization via datautils.get_wikitext2 (HF datasets)...")
        _, te = get_loaders(
            "wikitext2",
            nsamples=128,
            seed=args.seed,
            seqlen=args.seqlen,
            model=args.tok,
        )
        te = _wikitext_te_to_tensor(te)
        print(f"[WikiText-2] testenc shape={tuple(te.shape)}")
        return te

    def _c4_enc():
        if args.data_dir is None or (not os.path.isdir(args.data_dir)):
            return None
        pool, pool_info = build_doc_pool_from_c4_train_dir(
            args.data_dir,
            glob_pat=args.c4_glob,
            pool_docs=args.c4_pool_docs,
            scan_lines=args.c4_scan_lines,
            seed=args.seed,
        )
        print(f"[C4-Pool] {pool_info}")
        te = build_utkarsh_like_valenc_from_pool(
            tok,
            pool,
            seqlen=args.seqlen,
            n_chunks=args.c4_val_chunks,
            val_seed=args.c4_val_seed,
        )
        assert te.shape[1] == args.c4_val_chunks * args.seqlen, (
            te.shape,
            args.c4_val_chunks,
            args.seqlen,
        )
        return te

    ran = False
    if getattr(args, "wikitext2", False):
        testenc = _wiki_enc()
        base_ppl, bi = eval_ppl_utkarsh_opt(base, testenc, args.base_device, args.seqlen, limit=args.limit)
        cmp_ppl, ci = eval_ppl_utkarsh_opt(cmp, testenc, args.cmp_device, args.seqlen, limit=args.limit)
        print(f"[WikiText-2] BASE PPL: {base_ppl:.6f}  info={bi}")
        print(f"[WikiText-2] CMP  PPL: {cmp_ppl:.6f}  info={ci}")
        if math.isfinite(base_ppl) and base_ppl > 0 and math.isfinite(cmp_ppl):
            print(f"[WikiText-2] ratio CMP/BASE: {cmp_ppl/base_ppl:.6f}")
        print()
        ran = True

    testenc_c4 = _c4_enc()
    if testenc_c4 is not None:
        base_ppl, bi = eval_ppl_utkarsh_opt(base, testenc_c4, args.base_device, args.seqlen, limit=args.limit)
        cmp_ppl, ci = eval_ppl_utkarsh_opt(cmp, testenc_c4, args.cmp_device, args.seqlen, limit=args.limit)
        print(f"[C4] BASE PPL: {base_ppl:.6f}  info={bi}")
        print(f"[C4] CMP  PPL: {cmp_ppl:.6f}  info={ci}")
        if math.isfinite(base_ppl) and base_ppl > 0 and math.isfinite(cmp_ppl):
            print(f"[C4] ratio CMP/BASE: {cmp_ppl/base_ppl:.6f}")
        print()
        ran = True

    if not ran:
        print("Provide --wikitext2 and/or a valid --data_dir for C4. Skip PPL.")


def ppl_sanity_cmp_only(
    cmp,
    tok,
    args,
    eval_device: Optional[str] = None,
    log_label: str = "CMP",
):
    """
    与 ppl_sanity_utkarsh_c4_train 的 [B] 段一致，但**只**评测单个模型（原逻辑里叫 CMP）。单卡：CUDA_VISIBLE_DEVICES=0
    --cmp_device cuda:0

    可同时传入 --wikitext2 与 --data_dir，会各跑一遍 PPL。
    eval_device：若给定，覆盖 args.cmp_device（用于 --base_only 时走 --base_device）。
    """
    dev = eval_device if eval_device is not None else args.cmp_device
    print("\n[B] PPL (Utkarsh-aligned, single checkpoint — eval_ppl_utkarsh_opt)\n")

    def _wiki_enc():
        from datautils import get_loaders

        print("[WikiText-2] loading test tokenization via datautils.get_wikitext2 (HF datasets)...")
        _, te = get_loaders(
            "wikitext2",
            nsamples=128,
            seed=args.seed,
            seqlen=args.seqlen,
            model=args.tok,
        )
        te = _wikitext_te_to_tensor(te)
        print(f"[WikiText-2] testenc shape={tuple(te.shape)}")
        return te

    def _c4_enc():
        if args.data_dir is None or (not os.path.isdir(args.data_dir)):
            return None
        pool, pool_info = build_doc_pool_from_c4_train_dir(
            args.data_dir,
            glob_pat=args.c4_glob,
            pool_docs=args.c4_pool_docs,
            scan_lines=args.c4_scan_lines,
            seed=args.seed,
        )
        print(f"[C4-Pool] {pool_info}")
        te = build_utkarsh_like_valenc_from_pool(
            tok,
            pool,
            seqlen=args.seqlen,
            n_chunks=args.c4_val_chunks,
            val_seed=args.c4_val_seed,
        )
        assert te.shape[1] == args.c4_val_chunks * args.seqlen, (
            te.shape,
            args.c4_val_chunks,
            args.seqlen,
        )
        return te

    ran = False
    if getattr(args, "wikitext2", False):
        testenc = _wiki_enc()
        cmp_ppl, ci = eval_ppl_utkarsh_opt(cmp, testenc, dev, args.seqlen, limit=args.limit)
        print(f"[WikiText-2] {log_label} PPL: {cmp_ppl:.6f}  info={ci}\n")
        ran = True


    te_c4 = _c4_enc()
    if te_c4 is not None:
        cmp_ppl, ci = eval_ppl_utkarsh_opt(cmp, te_c4, dev, args.seqlen, limit=args.limit)
        print(f"[C4] {log_label} PPL: {cmp_ppl:.6f}  info={ci}\n")
        ran = True

    if not ran:
        print("Neither --wikitext2 nor a valid --data_dir for C4. Skip PPL.")

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cmp_only",
        action="store_true",
        help="只评测 CMP 的 PPL（Utkarsh 同逻辑），不加载 --base、不跑 [A]。",
    )
    ap.add_argument(
        "--base",
        default="",
        help="全精度基座；--cmp_only 时可省略（仍建议占位）。",
    )
    ap.add_argument(
        "--base_only",
        action="store_true",
        help="只评测稠密 HuggingFace 模型路径 --base 的 PPL（与 --cmp / low_rank 无关）。",
    )
    ap.add_argument(
        "--cmp",
        default=None,
        help="EigenAttn 压缩 checkpoint 目录；--base_only 时不需要。",
    )
    ap.add_argument(
        "--arch",
        default="auto",
        choices=["auto", "opt", "llama"],
        help="CMP 权重架构：auto 读 final_model/config.json 的 model_type；Llama-3.x 用 llama 或 auto。",
    )
    ap.add_argument("--low_json", default=None,
                    help="Required only when CMP dir has no low_rank_config.json / low_rank_config.pt")
    ap.add_argument("--tok", required=True)

    ap.add_argument("--dtype", default="fp16", choices=["bf16", "fp16"],
                    help="For Utkarsh alignment, fp16 is recommended.")
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=2)

    ap.add_argument(
        "--wikitext2",
        action="store_true",
        help=(
            "WikiText-2-raw-v1：用 HF datasets 加载 test split（与 datautils.get_wikitext2 一致），"
            "可与 --data_dir 同时使用以各跑一遍 PPL。"
        ),
    )

    # prompts
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--max_prompts", type=int, default=12)

    # PPL
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=-1,
                    help="Match repo behavior: break if i == limit. Default -1 means never break.")

    # C4-train pool + Utkarsh-style valenc
    ap.add_argument("--c4_glob", default="c4-train.*.json.gz")
    ap.add_argument("--c4_scan_lines", type=int, default=200000,
                    help="How many docs to scan (stream) from gz for building pool.")
    ap.add_argument("--c4_pool_docs", type=int, default=20000,
                    help="Reservoir pool size from scanned docs.")
    ap.add_argument("--c4_val_chunks", type=int, default=256,
                    help="Utkarsh uses 256 chunks for c4 validation.")
    ap.add_argument("--c4_val_seed", type=int, default=0,
                    help="Utkarsh hardcodes random.seed(0) for valenc sampling.")

    args = ap.parse_args()
    set_seed(args.seed)

    if args.base_only and args.cmp_only:
        raise SystemExit("不要同时使用 --base_only 与 --cmp_only。")
    if args.base_only and not (args.base or "").strip():
        raise SystemExit("--base_only 需要非空的 --base（HF 模型目录）。")
    if not args.base_only and not args.cmp_only and (not args.cmp or not (args.base or "").strip()):
        raise SystemExit("对比模式需要同时提供 --base 与 --cmp，或使用 --cmp_only / --base_only。")
    if args.cmp_only and not args.cmp:
        raise SystemExit("--cmp_only 需要 --cmp。")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    # Prefer low_rank_config from CMP dir (bilevel saves it) so rank always matches checkpoint.
    # Order: low_rank_config.json -> low_rank_config.pt -> config.json["low_rank_config"] -> --low_json
    low_rank_config = None
    if not args.base_only:
        low_json_in_cmp = os.path.join(args.cmp, "low_rank_config.json")
        low_pt_in_cmp = os.path.join(args.cmp, "low_rank_config.pt")
        config_json_in_cmp = os.path.join(args.cmp, "config.json")
        if os.path.isfile(low_json_in_cmp):
            with open(low_json_in_cmp, "r") as f:
                low_rank_config = json.load(f)
            print(f"[Config] Using low_rank_config from CMP dir: {low_json_in_cmp}")
        elif os.path.isfile(low_pt_in_cmp):
            try:
                pt = torch.load(low_pt_in_cmp, map_location="cpu", weights_only=True)
            except TypeError:
                pt = torch.load(low_pt_in_cmp, map_location="cpu")
            if isinstance(pt, torch.Tensor):
                low_rank_config = [[int(pt[i][0].item()), int(pt[i][1].item())] for i in range(len(pt))]
            else:
                low_rank_config = [
                    [
                        int(x[0]) if torch.is_tensor(x[0]) else int(x[0]),
                        int(x[1]) if torch.is_tensor(x[1]) else int(x[1]),
                    ]
                    for x in pt
                ]
            print(f"[Config] Using low_rank_config from CMP dir: {low_pt_in_cmp}")
        elif os.path.isfile(config_json_in_cmp):
            with open(config_json_in_cmp, "r", encoding="utf-8") as f:
                cfgj = json.load(f)
            lrc = cfgj.get("low_rank_config")
            if isinstance(lrc, list) and len(lrc) > 0:
                low_rank_config = lrc
                print(f"[Config] Using low_rank_config from CMP config.json: {config_json_in_cmp}")
            else:
                inferred = infer_low_rank_config_llama_from_shards(args.cmp)
                if inferred is not None:
                    low_rank_config = inferred
                    print(
                        f"[Config] Inferred low_rank_config from k_proj_low/v_proj_low shapes "
                        f"({len(inferred)} layers) — config.json had no low_rank_config list."
                    )
                elif not args.low_json or not os.path.isfile(args.low_json):
                    raise SystemExit(
                        "CMP config.json has no usable low_rank_config list and could not infer from safetensors; "
                        "pass --low_json matching checkpoint ranks "
                        "(e.g. q_proj_low.bias length = that layer's rank_kq)."
                    )
                else:
                    low_rank_config = json.load(open(args.low_json))
                    print(f"[Config] Using low_rank_config from --low_json: {args.low_json}")
        else:
            if not args.low_json or not os.path.isfile(args.low_json):
                raise SystemExit("CMP dir has no config.json and no low_rank_config sidecar; pass --low_json.")
            low_rank_config = json.load(open(args.low_json))
            print(f"[Config] Using low_rank_config from --low_json: {args.low_json}")

    tok = load_tokenizer_auto(args.tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if args.base_only:
        print("[Load] BASE only (--base_only): dense HuggingFace checkpoint.")
        base = load_base_model(args.base, dtype=dtype, device=args.base_device)
        base.eval()
        ppl_sanity_cmp_only(base, tok, args, eval_device=args.base_device, log_label="BASE")
        print("[Done] BASE-only PPL completed.")
        return

    if args.cmp_only:
        print("[Load] CMP only (--cmp_only), skip BASE.")
        print("[Load] CMP...")
        cmp = load_cmp_model(
            args.cmp,
            dtype=dtype,
            device=args.cmp_device,
            low_rank_config=low_rank_config,
            arch=args.arch,
        )
        cmp.eval()
        ppl_sanity_cmp_only(cmp, tok, args)
        print("[Done] CMP-only PPL completed.")
        return

    print("[Load] BASE...")
    base = load_base_model(args.base, dtype=dtype, device=args.base_device)
    base.eval()

    print("[Load] CMP...")
    cmp = load_cmp_model(
        args.cmp,
        dtype=dtype,
        device=args.cmp_device,
        low_rank_config=low_rank_config,
        arch=args.arch,
    )
    cmp.eval()

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

    ppl_sanity_utkarsh_c4_train(base, cmp, tok, args)

    print("[Done] A+B sanity completed.")

if __name__ == "__main__":
    main()
