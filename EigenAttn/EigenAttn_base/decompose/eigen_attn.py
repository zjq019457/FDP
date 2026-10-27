"""EigenAttn per-layer SVD decomposition with optional per-layer error_budget.

Compared to the original, this version supports two extra features needed by
the dynamic-EB pipeline:

* ``args.per_layer_error_budget``: optional ``list[float]`` (length = num layers).
  When provided, layer ``i`` uses ``per_layer_error_budget[i]`` as the EB target
  for the SVD-threshold descent loop instead of the uniform ``args.error_budget``.

* ``args.freeze_layers``: optional ``list[int]``. Listed indices ignore the
  per-layer override and use ``args.error_budget`` (the uniform baseline).
  This matches ``per_layer_eb_from_step1.py``'s ``--freeze_layers 0`` semantics.

Behavior with neither set is identical to the original ``eigenattn``.
"""

from __future__ import annotations

import gc
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from decompose.eigen_attn_utils import (
    decompose_llama_layer,
    decompose_mpt_layer,
    decompose_opt_layer,
)
from models.decompose_modules import (
    LlamaEigenAttnDecoderLayer,
    MptBlockEigenAttn,
    OPTEigenAttnDecoderLayer,
)


def _normalize_per_layer_eb(
    n_layers: int,
    eb_uniform: float,
    per_layer_eb: Optional[Sequence[float]],
    freeze_layers: Optional[Sequence[int]],
) -> List[float]:
    """Build a length-``n_layers`` EB vector. Frozen layers are forced to ``eb_uniform``.

    Returning a uniform vector when ``per_layer_eb`` is None preserves
    drop-in compatibility with the previous `eigenattn` behaviour while letting
    the inner loop always read ``eb_l = eb_per_layer[i]``.
    """
    if per_layer_eb is None:
        eb_per_layer = [float(eb_uniform)] * n_layers
    else:
        if len(per_layer_eb) != n_layers:
            raise ValueError(
                f"per_layer_error_budget length {len(per_layer_eb)} != n_layers {n_layers}"
            )
        eb_per_layer = [float(x) for x in per_layer_eb]

    for idx in freeze_layers or ():
        if 0 <= idx < n_layers:
            eb_per_layer[int(idx)] = float(eb_uniform)
    return eb_per_layer


def eigenattn(
    lm,
    args,
    dataloader,
    logger=None,
    per_layer_error_budget: Optional[Sequence[float]] = None,
    freeze_layers: Optional[Sequence[int]] = None,
):
    """Run EigenAttn-style per-layer SVD compression on ``lm.model``.

    Parameters identical to the original except for the two new keyword args,
    which can also be passed through ``args`` (``args.per_layer_error_budget``
    and ``args.freeze_layers``); explicit kwargs win.
    """
    log = logger.info if logger is not None else print
    log("Starting eigenattn decomposition (offline)...")
    net = args.net.lower()
    assert (
        ("opt" in net)
        or ("mpt" in net)
        or ("llama" in net)
        or ("mistral" in net)
        or ("qwen" in net)
    ), f"Unsupported net={args.net!r}"

    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # ---- arch dispatch
    is_llama = is_mpt = is_opt = False
    if "llama" in net or "mistral" in net or "qwen" in net:
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # transformers >=4.40 keeps rotary_emb at the model level (holds inv_freq);
        # without moving it the calibration forward errors with a device mismatch.
        if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
        DecoderLayer = LlamaEigenAttnDecoderLayer
    elif "opt" in net:
        is_opt = True
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = OPTEigenAttnDecoderLayer
    elif "mpt" in net:
        is_mpt = True
        layers = model.transformer.blocks
        model.transformer.wte = model.transformer.wte.to(dev)
        DecoderLayer = MptBlockEigenAttn
    else:
        raise ValueError("Only support OPT / MPT / Llama / Llama-3 now")

    n_layers = len(layers)
    # Resolve per-layer EB (explicit kwarg > args attr > uniform).
    if per_layer_error_budget is None:
        per_layer_error_budget = getattr(args, "per_layer_error_budget", None)
    if freeze_layers is None:
        freeze_layers = getattr(args, "freeze_layers", None)

    # Coerce freeze_layers to ``Sequence[int]`` regardless of how the caller
    # supplied it (CLI string ``"0,3"``, list of ints, list of strs, or None).
    if freeze_layers is None or freeze_layers == "":
        freeze_layers_list: List[int] = []
    elif isinstance(freeze_layers, str):
        freeze_layers_list = [int(x) for x in freeze_layers.split(",") if x.strip() != ""]
    else:
        freeze_layers_list = [int(x) for x in freeze_layers]

    eb_per_layer = _normalize_per_layer_eb(
        n_layers=n_layers,
        eb_uniform=float(args.error_budget),
        per_layer_eb=per_layer_error_budget,
        freeze_layers=freeze_layers_list,
    )
    log(
        f"per-layer EB: uniform={args.error_budget} mean={sum(eb_per_layer)/n_layers:.6f} "
        f"min={min(eb_per_layer):.6f} max={max(eb_per_layer):.6f} freeze={freeze_layers_list}"
    )

    # ---- catch first-layer inputs (calibration)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.is_mpt = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs.get("position_ids", None)
            if self.is_mpt:
                cache["position_bias"] = kwargs.get("position_bias", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    layers[0].is_mpt = is_mpt

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if is_llama:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    elif is_opt:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif is_mpt:
        model.transformer.wte = model.transformer.wte.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    position_ids = cache.get("position_ids", None) if is_llama else None
    position_bias = cache.get("position_bias", None) if is_mpt else None

    # ---- per-layer decomposition with the per-layer EB target
    for i in range(n_layers):
        eb_l = float(eb_per_layer[i])
        layer = layers[i].to(dev)

        if is_opt:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    svd_floor = float(getattr(args, "svd_threshold_floor", 0.3))
                    svd_init = float(getattr(args, "svd_threshold_init", 0.98))
                    args.eigen_attn_params["threshold"] = svd_init
                    error = 0.0
                    num_heads = lm.model.config.num_attention_heads
                    basis_kq, eval_kq, basis_v, eval_v = decompose_opt_layer(
                        layer, inps, args, num_heads, i
                    )
                    rank_kq = num_heads * torch.amax(
                        (torch.cumsum(eval_kq, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                    )
                    rank_v = num_heads * torch.amax(
                        (torch.cumsum(eval_v, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                    )
                    output_hr = torch.stack(
                        [layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] for j in range(args.nsamples)]
                    )
                    qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)
                    while (
                        error < eb_l
                        and args.eigen_attn_params["threshold"] > svd_floor
                        and rank_kq > 64
                        and rank_v > 64
                    ):
                        output_lr = torch.stack(
                            [qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] for j in range(args.nsamples)]
                        )
                        error = (torch.norm(output_hr - output_lr) / torch.norm(output_hr))
                        args.eigen_attn_params["threshold"] -= 0.02
                        rank_kq = num_heads * torch.amax(
                            (torch.cumsum(eval_kq, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                        )
                        rank_v = num_heads * torch.amax(
                            (torch.cumsum(eval_v, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                        )
                        qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)

                    args.eigen_attn_params["threshold"] += 0.04
                    rank_kq = num_heads * torch.amax(
                        (torch.cumsum(eval_kq, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                    )
                    rank_v = num_heads * torch.amax(
                        (torch.cumsum(eval_v, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                    )
                    qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)

                    output_lr = torch.stack(
                        [qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] for j in range(args.nsamples)]
                    )
                    error = (torch.norm(output_hr - output_lr) / torch.norm(output_hr))

                    del basis_kq, basis_v
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            for j in range(args.nsamples):
                                inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        elif is_mpt:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    svd_floor = float(getattr(args, "svd_threshold_floor", 0.3))
                    svd_init = float(getattr(args, "svd_threshold_init", 0.98))
                    args.eigen_attn_params["threshold"] = svd_init
                    error = 0.0
                    num_heads = lm.model.config.num_attention_heads
                    basis_kq, eval_kq, basis_v, eval_v = decompose_mpt_layer(
                        layer, inps, args, num_heads, i, attention_mask, position_bias
                    )
                    output_hr = torch.stack(
                        [layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias=position_bias)[0] for j in range(args.nsamples)]
                    )
                    while error < eb_l and args.eigen_attn_params["threshold"] > svd_floor:
                        rank_kq = num_heads * torch.amax(
                            (torch.cumsum(eval_kq, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                        )
                        rank_v = num_heads * torch.amax(
                            (torch.cumsum(eval_v, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                        )
                        qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)
                        output_lr = torch.stack(
                            [qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias=position_bias)[0] for j in range(args.nsamples)]
                        )
                        error = (torch.norm(output_hr - output_lr) / torch.norm(output_hr))
                        args.eigen_attn_params["threshold"] -= 0.02

                    args.eigen_attn_params["threshold"] += 0.04
                    rank_kq = num_heads * torch.amax(
                        (torch.cumsum(eval_kq, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                    )
                    rank_v = num_heads * torch.amax(
                        (torch.cumsum(eval_v, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                    )
                    qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)
                    output_lr = torch.stack(
                        [qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias=position_bias)[0] for j in range(args.nsamples)]
                    )
                    error = (torch.norm(output_hr - output_lr) / torch.norm(output_hr))
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            for j in range(args.nsamples):
                                inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias=position_bias)[0]
                    del basis_kq, basis_v, eval_kq, eval_v, output_hr, output_lr
                    torch.cuda.empty_cache()

        elif is_llama:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    args.eigen_attn_params["threshold"] = 1.0
                    error = 0.0
                    num_heads = lm.model.config.num_key_value_heads
                    basis_kq, eval_kq, basis_v, eval_v = decompose_llama_layer(
                        layer, inps, args, num_heads, i, attention_mask, position_ids
                    )
                    max_rank_kq = layer.self_attn.k_proj.weight.shape[0]
                    max_rank_v = layer.self_attn.v_proj.weight.shape[0]
                    output_hr = torch.stack(
                        [layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, layer_idx=i)[0] for j in range(args.nsamples)]
                    )
                    rank_kq = max_rank_kq
                    rank_v = max_rank_v
                    min_thresh = 0.45
                    while error < eb_l and (rank_kq > min_thresh * max_rank_kq) and (rank_v > min_thresh * max_rank_v):
                        rank_kq = num_heads * (
                            (torch.cumsum(eval_kq, dim=0) < args.eigen_attn_params["threshold"]).sum()
                        )
                        rank_v = num_heads * torch.amax(
                            (torch.cumsum(eval_v, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                        )
                        qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config, i).to(dev)
                        qlayer = qlayer.to(dev)
                        output_lr = torch.stack(
                            [qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, layer_idx=i)[0] for j in range(args.nsamples)]
                        )
                        error = (torch.norm(output_hr - output_lr) / torch.norm(output_hr))
                        args.eigen_attn_params["threshold"] -= 0.02

                    args.eigen_attn_params["threshold"] += 0.04
                    rank_v = num_heads * torch.amax(
                        (torch.cumsum(eval_v, dim=1) < args.eigen_attn_params["threshold"]).sum(1)
                    )
                    rank_kq = num_heads * (
                        (torch.cumsum(eval_kq, dim=0) < args.eigen_attn_params["threshold"]).sum()
                    )
                    qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config, i).to(dev)

                    del basis_kq, basis_v, eval_kq, eval_v
                    torch.cuda.empty_cache()
                    output_lr = torch.stack(
                        [qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, layer_idx=i)[0] for j in range(args.nsamples)]
                    )
                    error = (torch.norm(output_hr - output_lr) / torch.norm(output_hr))
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            for j in range(args.nsamples):
                                inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, layer_idx=i)[0]

        log(
            f"layer {i} eb_l={eb_l:.6f} error={float(error):.6f} "
            f"threshold={args.eigen_attn_params['threshold']:.4f} "
            f"rank_kq={int(rank_kq)} rank_v={int(rank_v)} "
            f"max_mem={torch.cuda.max_memory_allocated(lm._device)/1024**2:.1f}MB"
        )

        qlayer.half()
        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    del inps
    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = use_cache
    return model


__all__ = ["eigenattn"]
