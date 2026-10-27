#!/usr/bin/env python3
"""
Dual-protocol PPL (EigenAttn):

  • **C4** — Legacy ROOT `/home/bai-project/jiaqizhang/sanity_AB_opt30b.py` OPT metrics:
      OPT / OPT EigenAttn CMP: `model.model.decoder` → `lm_head`, **no logit clamp**,
      CrossEntropy on logits **without** prior `.float()` cast on logits (matches old script).
      Llama / MPT / dense Llama: standard `model(...).logits`, **no clamp**, same CE wiring as legacy scalar path.

  • **WikiText-2** — “Llama / HF-forward” style for **all** architectures:
      Always `model(input_ids=..., use_cache=False).logits`, **no clamp**, CE uses **FP32 logits**,
      non-finite chunks skipped (`used_chunks`), denominator `(used * seqlen)` (matches EigenAttn `sanity_AB_opt30b` non-OPT branch intent).

Run from repo EigenAttn root with PYTHONPATH including this directory.

Example (OPT AB + Wiki + C4):
  PYTHONPATH=. python sanity_ppl_dual_protocol.py \\
    --base /diskE/opt-30b --cmp /path/to/final_model \\
    --low_json /path/to/low_rank_config.json \\
    --tok /diskE/opt-30b --dtype fp16 \\
    --base_device cuda:0 --cmp_device cuda:1 \\
    --wikitext2 \\
    --data_dir /diskE/jiaqizhang/c4-en/en \\
    --c4_glob "c4-validation.*.json.gz" --seed 2 \\
    --c4_scan_lines 500000 --c4_pool_docs 500000 \\
    --c4_val_chunks 256 --c4_val_seed 0 \\
    --wikitext2_dir /path/to/wikitext-2-raw-v1

Example (Llama CMP-only, bf16):
  PYTHONPATH=. python sanity_ppl_dual_protocol.py --cmp_only \\
    --cmp /diskD/jiaqizhang/llama3_8b_eigen_kv_eb043/final_model \\
    --tok /diskD/jiaqizhang/Llama-3 --arch llama --dtype bf16 \\
    --cmp_device cuda:0 \\
    --wikitext2 --data_dir ... (same C4 flags as OPT)
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch
import torch.nn as nn

from parallel_utils import map_layers_to_multi_gpus
from sanity_AB_opt30b import (
    build_doc_pool_from_c4_train_dir,
    build_utkarsh_like_valenc_from_pool,
    infer_low_rank_config_llama_from_shards,
    load_base_model,
    load_cmp_model,
    load_tokenizer_auto,
    set_seed,
    _clear_attn_past_key_value_attr,
    _detect_cmp_arch,
    _wikitext_te_to_tensor,
)


def detect_hf_arch(model_dir: str) -> str:
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    mt = str(cfg.get("model_type", "") or "").lower()
    if mt == "opt":
        return "opt"
    if mt in ("llama", "llama4", "mistral", "qwen2", "mixtral"):
        return "llama"
    if mt == "mpt":
        return "mpt"
    raise ValueError(f"Unsupported model_type={mt!r} in {cfg_path}")


def _resolve_eval_device(model: nn.Module, device_arg: str):
    if str(device_arg).lower() != "auto":
        return device_arg
    lm_inner = getattr(model, "model", None)
    if lm_inner is not None and hasattr(lm_inner, "decoder") and hasattr(lm_inner.decoder, "embed_tokens"):
        return lm_inner.decoder.embed_tokens.weight.device
    if lm_inner is not None and hasattr(lm_inner, "embed_tokens"):
        return lm_inner.embed_tokens.weight.device
    return next(model.parameters()).device


def _is_mpt(model: nn.Module) -> bool:
    cfg_mt = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    return cfg_mt == "mpt" or model.__class__.__name__ == "MPTForCausalLM"


def _place_multigpu_causal_lm(model: nn.Module, net: str, logger: logging.Logger) -> str:
    """
    Single-process model parallelism (same pattern as main_eigen_attn / offline lm-eval).
    Returns embed/input device string for batch placement (e.g. cuda:0 under CUDA_VISIBLE_DEVICES).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("--multigpu requires CUDA")
    nl = net.lower()
    root = model
    inner = root.model

    if "opt" in nl:
        dec = inner.decoder
        map_layers_to_multi_gpus(dec.layers, logger=logger)
        input_device = dec.layers[0].device
        output_device = dec.layers[-1].device
        if input_device != output_device:
            raise RuntimeError(f"OPT multigpu: expected same io device, got {input_device} vs {output_device}")
        dec.embed_positions.to(input_device)
        dec.embed_tokens.to(input_device)
        dec.final_layer_norm.to(output_device)
        root.lm_head.to(output_device)
        return str(input_device)

    if "llama" in nl:
        layers = inner.layers
        map_layers_to_multi_gpus(layers, logger=logger)
        input_device = layers[0].device
        output_device = layers[-1].device
        if input_device != output_device:
            raise RuntimeError(f"Llama multigpu: expected same io device, got {input_device} vs {output_device}")
        inner.embed_tokens.to(input_device)
        rotary = getattr(inner, "rotary_emb", None)
        if rotary is not None:
            rotary.to(input_device)
        inner.norm.to(output_device)
        root.lm_head.to(output_device)
        return str(input_device)

    raise NotImplementedError(f"--multigpu not implemented for arch/net={net!r}")


@torch.no_grad()
def eval_ppl_c4_legacy_opt_compatible(
    model: nn.Module,
    testenc: torch.Tensor,
    device_arg: str,
    seqlen: int,
    limit: int,
    *,
    arch: str,
) -> tuple[float, dict]:
    """
    C4: legacy ROOT OPT script semantics when arch=='opt'; else HF causal LM forward (Llama/MPT).
    Aggregation matches legacy: ppl = exp(sum(loss.float()*seqlen) / (nsamples*seqlen)), no skipped chunks.
    """
    model.eval()
    use_cache = getattr(model.config, "use_cache", False) if hasattr(model, "config") else False
    if hasattr(model, "config"):
        model.config.use_cache = False

    device = _resolve_eval_device(model, device_arg)
    nsamples = testenc.numel() // seqlen
    loss_fct = torch.nn.CrossEntropyLoss()
    nlls: list[torch.Tensor] = []

    arch_l = str(arch).lower()

    for i in range(nsamples):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(device)

        if arch_l == "opt":
            outputs = model.model.decoder(batch)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)
        else:
            fwd_kw = dict(input_ids=batch, use_cache=False)
            if _is_mpt(model):
                fwd_kw["attention_mask"] = torch.ones(
                    batch.shape[0], batch.shape[1], dtype=torch.bool, device=batch.device
                )
            logits = model(**fwd_kw).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        if arch_l == "opt":
            shift_labels = shift_labels.to(model.lm_head.weight.device)
        else:
            shift_labels = shift_labels.to(logits.device)

        # Legacy ROOT script: CE on logits dtype as-is (no logits.float()); loss scalar then .float().
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(loss.float() * seqlen)
        if i == limit:
            break

    if hasattr(model, "config"):
        model.config.use_cache = use_cache

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item(), {"nsamples": nsamples, "used_chunks": len(nlls), "c4_protocol": "legacy_opt_compatible"}


@torch.no_grad()
def eval_ppl_wikitext_hf_forward_llama_style(
    model: nn.Module,
    testenc: torch.Tensor,
    device_arg: str,
    seqlen: int,
    limit: int,
) -> tuple[float, dict]:
    """
    WikiText-2: unified HF forward for OPT/Llama/MPT — same branch intent as EigenAttn sanity_AB_opt30b
    for non-OPT models (no clamp, logits.float() for CE, skip bad chunks).
    """
    model.eval()
    _clear_attn_past_key_value_attr(model)
    use_cache = getattr(model.config, "use_cache", False) if hasattr(model, "config") else False
    if hasattr(model, "config"):
        model.config.use_cache = False

    device = _resolve_eval_device(model, device_arg)
    nsamples = testenc.numel() // seqlen
    loss_fct = torch.nn.CrossEntropyLoss()
    nlls: list[torch.Tensor] = []
    used = 0

    for i in range(nsamples):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(device)
        fwd_kw = dict(input_ids=batch, use_cache=False)
        if _is_mpt(model):
            fwd_kw["attention_mask"] = torch.ones(
                batch.shape[0], batch.shape[1], dtype=torch.bool, device=batch.device
            )
        logits = model(**fwd_kw).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:].to(logits.device)
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
        )
        if not torch.isfinite(loss).all():
            continue
        nlls.append(loss.float() * seqlen)
        used += 1
        if i == limit:
            break

    if hasattr(model, "config"):
        model.config.use_cache = use_cache

    if used == 0:
        return float("nan"), {"nsamples": nsamples, "used_chunks": 0, "wiki_protocol": "hf_forward_float_ce"}

    ppl = torch.exp(torch.stack(nlls).sum() / (used * seqlen))
    return ppl.item(), {"nsamples": nsamples, "used_chunks": used, "wiki_protocol": "hf_forward_float_ce"}


def _wiki_enc(tok, args):
    from datautils import get_loaders

    print("[WikiText-2] via datautils.get_wikitext2 (test split)...")
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


def _c4_enc(tok, args):
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


def run_ppl_suite(label: str, model, tok, args, *, arch_for_c4: str, eval_device_arg: str):
    ran = False
    if getattr(args, "wikitext2", False):
        testenc = _wiki_enc(tok, args)
        ppl, info = eval_ppl_wikitext_hf_forward_llama_style(
            model, testenc, eval_device_arg, args.seqlen, args.limit
        )
        print(f"[WikiText-2][HF-forward] {label} PPL: {ppl:.6f}  info={info}\n")
        ran = True

    te_c4 = _c4_enc(tok, args)
    if te_c4 is not None:
        ppl, info = eval_ppl_c4_legacy_opt_compatible(
            model, te_c4, eval_device_arg, args.seqlen, args.limit, arch=arch_for_c4
        )
        print(f"[C4][legacy-OPT-compatible arch={arch_for_c4}] {label} PPL: {ppl:.6f}  info={info}\n")
        ran = True

    if not ran:
        print("Neither --wikitext2 nor valid --data_dir for C4; nothing to run.")


def _resolve_low_rank_config(args) -> list | None:
    if args.base_only:
        return None
    low_json_in_cmp = os.path.join(args.cmp, "low_rank_config.json")
    low_pt_in_cmp = os.path.join(args.cmp, "low_rank_config.pt")
    config_json_in_cmp = os.path.join(args.cmp, "config.json")
    if os.path.isfile(low_json_in_cmp):
        with open(low_json_in_cmp, "r") as f:
            cfg = json.load(f)
        print(f"[Config] low_rank_config from CMP dir: {low_json_in_cmp}")
        return cfg
    if os.path.isfile(low_pt_in_cmp):
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
        print(f"[Config] low_rank_config from CMP dir: {low_pt_in_cmp}")
        return low_rank_config
    if os.path.isfile(config_json_in_cmp):
        with open(config_json_in_cmp, "r", encoding="utf-8") as f:
            cfgj = json.load(f)
        lrc = cfgj.get("low_rank_config")
        if isinstance(lrc, list) and len(lrc) > 0:
            print(f"[Config] low_rank_config from CMP config.json")
            return lrc
        inferred = infer_low_rank_config_llama_from_shards(args.cmp)
        if inferred is not None:
            print("[Config] Inferred low_rank_config from safetensors shards")
            return inferred
        if args.low_json and os.path.isfile(args.low_json):
            with open(args.low_json, "r") as f:
                cfg = json.load(f)
            print(f"[Config] low_rank_config from --low_json")
            return cfg
        raise SystemExit(
            "CMP has no usable low_rank_config; pass --low_json or add low_rank_config.json to final_model."
        )
    if args.low_json and os.path.isfile(args.low_json):
        with open(args.low_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[Config] low_rank_config from --low_json: {args.low_json}")
        return cfg
    raise SystemExit(
        "CMP dir missing config.json and low_rank sidecars; pass --low_json "
        "(e.g. --low_json /diskE/jiaqizhang/low_rank_config_opt30b_paper_eb008.json)."
    )


def _load_cmp_for_eval(args, dtype, low_rank_config, cmp_arch: str, log_ppl: logging.Logger):
    if args.multigpu:
        print("[Load] CMP --multigpu: device_map=cpu then layer-wise shard.")
        cmp_model = load_cmp_model(
            args.cmp,
            dtype=dtype,
            device="cpu",
            low_rank_config=low_rank_config,
            arch=cmp_arch,
            device_map="cpu",
        )
        cmp_eval_dev = _place_multigpu_causal_lm(cmp_model, cmp_arch, log_ppl)
        print(f"[Load] CMP embed/input device after shard: {cmp_eval_dev}")
        return cmp_model, cmp_eval_dev
    cmp_model = load_cmp_model(
        args.cmp,
        dtype=dtype,
        device=args.cmp_device,
        low_rank_config=low_rank_config,
        arch=cmp_arch,
    )
    return cmp_model, args.cmp_device


def main():
    ap = argparse.ArgumentParser(description="Dual-protocol PPL: C4 legacy OPT-compatible; Wiki HF-forward.")
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--cmp_only", action="store_true")
    ap.add_argument("--base", default="", help="Dense HF checkpoint (OPT/Llama base).")
    ap.add_argument("--cmp", default=None, help="EigenAttn final_model directory.")
    ap.add_argument("--arch", default="auto", choices=["auto", "opt", "llama"])
    ap.add_argument("--low_json", default=None)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--dtype", default="fp16", choices=["bf16", "fp16"])
    ap.add_argument("--base_device", default="cuda:0")
    ap.add_argument("--cmp_device", default="cuda:1")
    ap.add_argument(
        "--multigpu",
        action="store_true",
        help="CMP：CPU 加载后按层切到多张可见 GPU（需 CUDA_VISIBLE_DEVICES 含多卡）。"
        "开启后 batch 落在可见域 cuda:0（与 parallel_utils 一致）。",
    )
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--wikitext2", action="store_true")
    ap.add_argument(
        "--wikitext2_dir",
        default=None,
        help="本地 WikiText-2 raw 目录（含 wiki.train.raw / wiki.test.raw）；等价于设置环境变量 WIKITEXT2_DATA_DIR。",
    )
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--c4_glob", default="c4-train.*.json.gz")
    ap.add_argument("--c4_scan_lines", type=int, default=200000)
    ap.add_argument("--c4_pool_docs", type=int, default=20000)
    ap.add_argument("--c4_val_chunks", type=int, default=256)
    ap.add_argument("--c4_val_seed", type=int, default=0)
    args = ap.parse_args()

    if getattr(args, "wikitext2_dir", None):
        wt = os.path.abspath(os.path.expanduser(str(args.wikitext2_dir)))
        if not os.path.isdir(wt):
            raise SystemExit(f"--wikitext2_dir is not a directory: {wt}")
        os.environ["WIKITEXT2_DATA_DIR"] = wt

    print(
        "\n[Protocols]\n"
        "  C4: OPT → decoder+lm_head, no clamp, CE on native logits dtype (legacy ~/sanity_AB_opt30b.py).\n"
        "      Llama/MPT → model(...).logits, no clamp, same loss aggregation as legacy scalar path.\n"
        "  WikiText-2: all arches → HF forward, logits.float() CE, no clamp, skip non-finite chunks.\n"
    )

    if args.base_only and args.cmp_only:
        raise SystemExit("Use only one of --base_only / --cmp_only.")
    if args.base_only and not (args.base or "").strip():
        raise SystemExit("--base_only requires --base")
    if args.cmp_only and not args.cmp:
        raise SystemExit("--cmp_only requires --cmp")
    if args.multigpu and args.base_only:
        raise SystemExit("--multigpu 仅用于 CMP；OPT dense BASE 请用 --base_device auto + 多卡 CUDA_VISIBLE_DEVICES。")

    if not args.base_only and not args.cmp_only:
        if not args.cmp or not (args.base or "").strip():
            raise SystemExit("Need --base and --cmp, or use --base_only / --cmp_only.")

    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    log_ppl = logging.getLogger("sanity_ppl_dual_protocol")
    if not log_ppl.handlers:
        log_ppl.addHandler(logging.StreamHandler())
        log_ppl.setLevel(logging.INFO)

    tok = load_tokenizer_auto(args.tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if args.base_only:
        arch = detect_hf_arch(args.base)
        print(f"[Load] BASE only  arch={arch}  dtype={args.dtype}")
        base = load_base_model(args.base, dtype=dtype, device=args.base_device)
        base.eval()
        run_ppl_suite("BASE", base, tok, args, arch_for_c4=arch, eval_device_arg=args.base_device)
        print("[Done] dual-protocol BASE-only.")
        return

    low_rank_config = _resolve_low_rank_config(args)
    cmp_arch = args.arch if args.arch != "auto" else _detect_cmp_arch(args.cmp)

    if args.cmp_only:
        print(f"[Load] CMP only  arch={cmp_arch}  dtype={args.dtype}")
        cmp_model, cmp_eval_dev = _load_cmp_for_eval(args, dtype, low_rank_config, cmp_arch, log_ppl)
        cmp_model.eval()
        run_ppl_suite("CMP", cmp_model, tok, args, arch_for_c4=cmp_arch, eval_device_arg=cmp_eval_dev)
        print("[Done] dual-protocol CMP-only.")
        return

    base_arch = detect_hf_arch(args.base)
    print(f"[Load] BASE arch={base_arch} device={args.base_device}")
    base = load_base_model(args.base, dtype=dtype, device=args.base_device)
    base.eval()

    print(f"[Load] CMP arch={cmp_arch} dtype={args.dtype}")
    cmp_model, cmp_eval_dev = _load_cmp_for_eval(args, dtype, low_rank_config, cmp_arch, log_ppl)
    cmp_model.eval()

    run_ppl_suite("BASE", base, tok, args, arch_for_c4=base_arch, eval_device_arg=args.base_device)
    run_ppl_suite("CMP", cmp_model, tok, args, arch_for_c4=cmp_arch, eval_device_arg=cmp_eval_dev)

    # Optional ratio print when both finite
    # (re-run C4/Wiki would duplicate work — user can compare printed lines)

    print("[Done] dual-protocol BASE+CMP.")


if __name__ == "__main__":
    main()
