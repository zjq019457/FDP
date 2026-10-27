"""EigenAttn decomposition: offline SVD search OR compress-only basis replacement."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _find_basis_key(basis: Dict[str, torch.Tensor], layer_idx: int, kind: str) -> str:
    needle = f"layers.{layer_idx}.self_attn.{kind}_proj_basis"
    hits = [k for k in basis if needle in k]
    if not hits:
        raise KeyError(f"Missing basis key for layer={layer_idx} kind={kind}")
    return sorted(hits, key=len)[0]


def _unpack_v_basis_per_head(Uv: torch.Tensor, num_kv: int, head_dim: int) -> List[torch.Tensor]:
    rv = int(Uv.shape[1])
    kcol = rv // num_kv
    out: List[torch.Tensor] = []
    for h in range(num_kv):
        rs = slice(h * head_dim, (h + 1) * head_dim)
        cs = slice(h * kcol, (h + 1) * kcol)
        out.append(Uv[rs, cs].float().contiguous())
    return out


@torch.no_grad()
def _fill_llama_eigen_attn_from_basis(attn, ori_attn, Uqk: torch.Tensor, Uv: torch.Tensor, config) -> Tuple[int, int]:
    num_kv = int(config.num_key_value_heads)
    num_groups = int(config.num_attention_heads // num_kv)
    head_dim = int(config.hidden_size // config.num_attention_heads)

    rank_kq = int(Uqk.shape[1]) * num_kv
    rank_v = int(Uv.shape[1])
    k_kq = rank_kq // num_kv
    k_v = rank_v // num_kv

    attn.q_proj.weight.copy_(ori_attn.q_proj.weight)
    if attn.q_proj.bias is not None and ori_attn.q_proj.bias is not None:
        attn.q_proj.bias.copy_(ori_attn.q_proj.bias)

    u_kq = Uqk[:, :k_kq].float().contiguous()
    Wk = ori_attn.k_proj.weight.float()
    bk = ori_attn.k_proj.bias.float() if ori_attn.k_proj.bias is not None else None
    for i in range(num_kv):
        sl = slice(i * k_kq, (i + 1) * k_kq)
        hsl = slice(i * head_dim, (i + 1) * head_dim)
        attn.k_proj_low.weight.data[sl, :] = u_kq.t() @ Wk[hsl, :]
        if bk is not None and attn.k_proj_low.bias is not None:
            attn.k_proj_low.bias.data[sl] = u_kq.t() @ bk[hsl]
    attn.k_proj_up.weight.data.copy_(u_kq.to(dtype=attn.k_proj_up.weight.dtype))

    v_heads = _unpack_v_basis_per_head(Uv, num_kv, head_dim)
    Wv = ori_attn.v_proj.weight.float()
    bv = ori_attn.v_proj.bias.float() if ori_attn.v_proj.bias is not None else None
    for i in range(num_kv):
        u_v = v_heads[i][:, :k_v]
        sl = slice(i * k_v, (i + 1) * k_v)
        hsl = slice(i * head_dim, (i + 1) * head_dim)
        attn.v_proj_low.weight.data[sl, :] = u_v.t() @ Wv[hsl, :]
        if bv is not None and attn.v_proj_low.bias is not None:
            attn.v_proj_low.bias.data[sl] = u_v.t() @ bv[hsl]

    Wo = ori_attn.o_proj.weight.float()
    for i in range(num_kv):
        u_v = v_heads[i][:, :k_v]
        for j in range(num_groups):
            idx = i * num_groups + j
            osl = slice(idx * k_v, (idx + 1) * k_v)
            hsl = slice(idx * head_dim, (idx + 1) * head_dim)
            attn.o_proj_up.weight.data[:, osl] = Wo[:, hsl] @ u_v

    return rank_kq, rank_v


@torch.no_grad()
def _eigenattn_replace_from_basis(lm, args, logger=None):
    log = logger.info if logger is not None else print
    log("Starting EigenAttn replacement ...")
    basis_side = getattr(args, "basis_side", "right")
    log(f"[EigenAttn] basis_side = {basis_side}")

    basis_path = Path(getattr(args, "eigen_basis_path", "") or "").expanduser()
    if not basis_path.is_file():
        raise FileNotFoundError(f"eigen_basis_path not found: {basis_path}")
    log(f"Loading eigen_basis from: {basis_path}")
    basis = torch.load(basis_path, map_location="cpu", weights_only=False)

    model = lm.model
    config = model.config
    net = args.net.lower()

    if "opt" in net:
        layers = model.model.decoder.layers
        n_layers = len(layers)
        dummy = nn.Module()
        dummy.basis_side = basis_side
        for i in range(n_layers):
            uq = basis[_find_basis_key(basis, i, "q")].float()
            try:
                uk = basis[_find_basis_key(basis, i, "k")].float()
            except KeyError:
                uk = uq
            uv = basis[_find_basis_key(basis, i, "v")].float()
            rank_kq = int(uq.shape[1])
            rank_v = int(uv.shape[1])
            qlayer = OPTEigenAttnDecoderLayer(
                layers[i], dummy, uq, rank_kq, uv, rank_v, config, layer_name=f"layer{i}"
            )
            layers[i] = qlayer
            log(f"layer {i} rank_kq={rank_kq} rank_v={rank_v}")
        return model

    if "llama" in net or "mistral" in net or "qwen" in net:
        if "qwen" in net:
            from models.modelling_qwen2_eigen_attn import LlamaEigenAttention
        else:
            from models.modelling_llama_eigen_attn import LlamaEigenAttention

        layers = model.model.layers
        n_layers = len(layers)
        num_kv = int(config.num_key_value_heads)
        head_dim = int(config.hidden_size // config.num_attention_heads)

        dev = lm.device
        with torch.cuda.amp.autocast():
            for i in range(n_layers):
                Uqk = basis[_find_basis_key(basis, i, "q")].float()
                Uv = basis[_find_basis_key(basis, i, "v")].float()
                rank_kq = int(Uqk.shape[1]) * num_kv
                rank_v = int(Uv.shape[1])

                layer = layers[i]
                ori_attn = layer.self_attn
                new_attn = LlamaEigenAttention(rank_kq, rank_v, config, layer_idx=i).to(dev)
                _fill_llama_eigen_attn_from_basis(new_attn, ori_attn, Uqk, Uv, config)
                target_dtype = ori_attn.q_proj.weight.dtype
                layer.self_attn = new_attn.to(dtype=target_dtype)
                log(f"layer {i} rank_kq={rank_kq} rank_v={rank_v}")
        return model

    raise ValueError(f"Basis replacement unsupported for net={args.net!r}")


def _eigenattn_offline(
    lm,
    args,
    dataloader,
    logger=None,
    per_layer_error_budget: Optional[Sequence[float]] = None,
    freeze_layers: Optional[Sequence[int]] = None,
):
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

    is_llama = is_mpt = is_opt = False
    if "llama" in net or "mistral" in net or "qwen" in net:
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
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
    if per_layer_error_budget is None:
        per_layer_error_budget = getattr(args, "per_layer_error_budget", None)
    if freeze_layers is None:
        freeze_layers = getattr(args, "freeze_layers", None)

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


def eigenattn(
    lm,
    args,
    dataloader,
    logger=None,
    per_layer_error_budget: Optional[Sequence[float]] = None,
    freeze_layers: Optional[Sequence[int]] = None,
):
    ebp = (getattr(args, "eigen_basis_path", "") or "").strip()
    if ebp:
        return _eigenattn_replace_from_basis(lm, args, logger)
    return _eigenattn_offline(
        lm,
        args,
        dataloader,
        logger=logger,
        per_layer_error_budget=per_layer_error_budget,
        freeze_layers=freeze_layers,
    )


__all__ = ["eigenattn"]
