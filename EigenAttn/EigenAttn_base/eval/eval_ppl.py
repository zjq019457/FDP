"""Utkarsh-aligned PPL evaluation shared by OPT / MPT / Llama EigenAttn checkpoints.

Logic mirrors ``sanity_AB_opt30b.py``:

* ``seqlen``-strided non-overlapping chunks; per-chunk NLL summed and averaged
  by ``used_chunks * seqlen`` — same aggregation as ``main_eigen_attn._eval_ppl_utkarsh_opt``.
* OPT path forwards through ``model.model.decoder`` then ``lm_head`` and clamps
  logits to ``[-20, 20]`` (paper-faithful fp16 numeric guard).
* Llama / MPT path runs full ``model(input_ids=batch).logits`` with ``use_cache=False``.
  No clamp (large vocab makes >|20| legitimate; clamping skews PPL).
* MPT explicit ``attention_mask`` of all-ones to dodge a vendored MPTModel.forward NPE.

The evaluator also clears any stale ``self_attn.past_key_value`` attribute on
EigenAttn modules before iterating; otherwise a previous decoder run with cache
would leak into PPL chunks (root cause of the 6.73 -> 9.65 inflation we hit).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from datautils import (
    DEFAULT_C4_DIR,
    DEFAULT_C4_VAL_GLOB,
    DEFAULT_WIKITEXT2_DIR,
    build_utkarsh_valenc_from_c4,
    get_wikitext2,
)


# ---------------------------------------------------------------------------
# Defensive helpers
# ---------------------------------------------------------------------------
def _clear_attn_past_key_value_attr(model) -> None:
    """Strip stale ``self_attn.past_key_value`` (Eigen modules use ``getattr``)."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        decoder = getattr(inner, "decoder", None)
        if decoder is not None:
            layers = getattr(decoder, "layers", None)
    if layers is None:
        transformer = getattr(model, "transformer", None)
        if transformer is not None:
            layers = getattr(transformer, "blocks", None)
    if not layers:
        return
    for layer in layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            continue
        if hasattr(attn, "past_key_value"):
            try:
                delattr(attn, "past_key_value")
            except Exception:
                setattr(attn, "past_key_value", None)


def _wikitext_te_to_tensor(te) -> torch.Tensor:
    if hasattr(te, "input_ids"):
        return te.input_ids
    if isinstance(te, torch.Tensor):
        return te
    raise TypeError(f"Unexpected wikitext testenc type: {type(te)}")


def _resolve_eval_device(model, requested: Optional[str]) -> str:
    if requested and str(requested).lower() != "auto":
        return requested
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "decoder") and hasattr(inner.decoder, "embed_tokens"):
        return inner.decoder.embed_tokens.weight.device
    if inner is not None and hasattr(inner, "embed_tokens"):
        return inner.embed_tokens.weight.device
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "wte"):
        return transformer.wte.weight.device
    return next(model.parameters()).device


def _detect_arch(model) -> str:
    cfg_mt = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    if cfg_mt == "opt" or "OPT" in model.__class__.__name__:
        return "opt"
    if cfg_mt == "mpt" or model.__class__.__name__ == "MPTForCausalLM" or "Mpt" in model.__class__.__name__:
        return "mpt"
    return "llama"  # llama/mistral/qwen2/...


# ---------------------------------------------------------------------------
# Core PPL loop (Utkarsh-aligned).
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_ppl_utkarsh(
    model,
    testenc: torch.Tensor,
    *,
    seqlen: int,
    device: Optional[str] = None,
    limit: int = -1,
    arch: Optional[str] = None,
) -> Tuple[float, Dict[str, int]]:
    """Compute PPL on a tensor of shape ``[1, T]`` where ``T % seqlen == 0`` is preferred.

    Returns ``(ppl, info)`` where info has ``nsamples`` and ``used_chunks``.
    """
    model.eval()
    _clear_attn_past_key_value_attr(model)
    if hasattr(model, "config"):
        prev_use_cache = getattr(model.config, "use_cache", False)
        model.config.use_cache = False
    else:
        prev_use_cache = False

    arch = arch or _detect_arch(model)
    eval_dev = _resolve_eval_device(model, device)
    nsamples = testenc.numel() // seqlen
    loss_fct = torch.nn.CrossEntropyLoss()
    nlls = []
    used = 0
    for i in range(nsamples):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(eval_dev)

        if arch == "opt":
            outputs = model.model.decoder(batch)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)
            logits = torch.clamp(logits, -20, 20)
        else:
            fwd_kw = dict(input_ids=batch, use_cache=False)
            if arch == "mpt":
                fwd_kw["attention_mask"] = torch.ones(
                    batch.shape[0], batch.shape[1], dtype=torch.bool, device=batch.device
                )
            out = model(**fwd_kw)
            logits = out.logits

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

    if hasattr(model, "config"):
        model.config.use_cache = prev_use_cache

    if used == 0:
        return float("nan"), {"nsamples": nsamples, "used_chunks": 0}
    ppl = torch.exp(torch.stack(nlls).sum() / (used * seqlen)).item()
    return ppl, {"nsamples": nsamples, "used_chunks": used}


# ---------------------------------------------------------------------------
# High-level: run wiki + c4-validation PPL with one call.
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ppl(
    model,
    tokenizer_path: str,
    *,
    seqlen: int = 2048,
    seed: int = 2,
    limit: int = -1,
    device: Optional[str] = None,
    do_wikitext2: bool = True,
    do_c4: bool = True,
    wikitext2_dir: str = DEFAULT_WIKITEXT2_DIR,
    c4_data_dir: str = DEFAULT_C4_DIR,
    c4_glob: str = DEFAULT_C4_VAL_GLOB,
    c4_pool_docs: int = 500_000,
    c4_scan_lines: int = 500_000,
    c4_val_chunks: int = 256,
    c4_val_seed: int = 0,
    label: str = "MODEL",
    log_fn=print,
) -> Dict[str, dict]:
    """Run wiki-2 and/or c4-validation PPL and return a structured report.

    The model must already be loaded onto its eval device(s); only ``device`` is
    passed to the inner loop so per-arch HF parallelism can keep its own mapping.
    """
    arch = _detect_arch(model)
    log_fn(f"[ppl] arch={arch} seqlen={seqlen} seed={seed} label={label}")

    out: Dict[str, dict] = {}
    if do_wikitext2:
        log_fn("[ppl] wikitext2 testenc via local parquet")
        _, te = get_wikitext2(
            nsamples=128, seed=seed, seqlen=seqlen, model=tokenizer_path, data_dir=wikitext2_dir
        )
        te = _wikitext_te_to_tensor(te)
        log_fn(f"[ppl] wikitext2 testenc shape={tuple(te.shape)}")
        ppl, info = eval_ppl_utkarsh(model, te, seqlen=seqlen, device=device, limit=limit, arch=arch)
        log_fn(f"[ppl] wikitext2 {label} PPL={ppl:.6f}  info={info}")
        out["wikitext2"] = {"ppl": ppl, **info}

    if do_c4:
        log_fn(f"[ppl] c4 valenc via {c4_data_dir}/{c4_glob}, n_chunks={c4_val_chunks}")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=("mpt" in tokenizer_path.lower()),
        )
        valenc, info_pool = build_utkarsh_valenc_from_c4(
            tok,
            seqlen=seqlen,
            data_dir=c4_data_dir,
            glob_pat=c4_glob,
            pool_docs=c4_pool_docs,
            scan_lines=c4_scan_lines,
            n_chunks=c4_val_chunks,
            seed=seed,
            val_seed=c4_val_seed,
        )
        log_fn(f"[ppl] c4 pool {info_pool}")
        ppl, info = eval_ppl_utkarsh(model, valenc, seqlen=seqlen, device=device, limit=limit, arch=arch)
        log_fn(f"[ppl] c4 {label} PPL={ppl:.6f}  info={info}")
        out["c4"] = {"ppl": ppl, "pool": info_pool, **info}

    return out


__all__ = ["eval_ppl_utkarsh", "evaluate_ppl"]
