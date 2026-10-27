#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E3 fixed-rank rebuild for OPT-30B

目标
----
在 **不改变 eb、不改变 low_json、不改变每层 rank** 的前提下，做一个分层混合修复：

- score_layers: 做 Q/K score-temperature 校正（不改 rank，只缩放 q_proj_low / k_proj_low）
- vtail_layers: 用给定 basis_pt 中的 basis 重建这些层（basis_pt 可为你已做好的 V-tail 最终 basis）
- 其余层：保持 cmp_in 原样

并提供两级保护：
1) per-layer greedy accept on linesearch（Utk logPPL 下降则接受）
2) final global rollback on holdout（劣化则整包回滚）

重要说明
--------
0) 评估指标与 `sanity_AB_opt30b.py` 的 `eval_ppl_utkarsh_opt` 一致：`decoder+lm_head` 前向，
   对每块用 mean CE 后按 Utkarsh 式聚合；打出的 Utk logPPL(nats) 满足 PPL=exp(·)，可与 sanity
   的 PPL 对表。不再用 `model(input_ids).logits` 算标量。
1) 该脚本默认认为 --basis_pt 已经是“最终要注入模型的 basis”（例如你做好的 vtail basis）。
   如果你想在脚本内再把 tail_basis_pt 融进去，请同时传 --tail_basis_pt / --tail_alpha / --tail_cols。
2) score calibration 支持 global / per_head 两种模式，不改 forward；
   通过把 q_proj_low / k_proj_low 的对应权重和 bias 同乘 sqrt(temp)，使 logits 近似乘 temp。
3) 回滚规则：
   - 某一层局部修改若在 linesearch 集上没有超过 accept_delta_ce 的改善，就回滚该层。
   - 全部层处理完后，若最终模型在 holdout CE 上劣于原始 cmp_in 超过 rollback_delta_ce，
     则整体回滚：直接拷贝 cmp_in -> cmp_out。

建议用法
--------
- score_layers: 1-18
- vtail_layers: 19-27,47
- 其余层不动
"""

from __future__ import annotations

import os
import json
import math
import glob
import gzip
import random
import shutil
import argparse
import inspect as _inspect
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import no_init_weights
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME
from transformers.utils.generic import ContextManagers

from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn
from tools.build_activation_eigen_basis_opt import build_token_stream, iter_texts_from_dir

try:
    from safetensors.torch import load_file as _safe_load_file
except Exception:
    _safe_load_file = None


torch.set_grad_enabled(False)


# -----------------------------------------------------------------------------
# basic utils
# -----------------------------------------------------------------------------
def _parse_layers_spec(spec: str, n_layers: int) -> Set[int]:
    spec = (spec or "").strip().lower()
    if not spec or spec == "none":
        return set()
    if spec == "all":
        return set(range(n_layers))
    out: Set[int] = set()
    for part in spec.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            a = int(a)
            b = int(b)
            if a > b:
                a, b = b, a
            for i in range(a, b + 1):
                if 0 <= i < n_layers:
                    out.add(i)
        else:
            i = int(p)
            if 0 <= i < n_layers:
                out.add(i)
    return out


def _parse_float_grid(spec: str) -> List[float]:
    vals = []
    for x in (spec or "").split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        vals = [1.0]
    if 1.0 not in vals:
        vals = [1.0] + vals
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _normalize_low_rank_config(low, n_layers: int):
    if not isinstance(low, list) or len(low) < n_layers:
        raise ValueError(f"low_rank_config 需为长度 >= {n_layers} 的 list，实际 {type(low)} len={len(low)}")
    out = []
    for i in range(n_layers):
        row = low[i]
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError(f"low_rank_config[{i}] 无效: {row!r}")
        out.append([int(row[0]), int(row[1])])
    return out


def _basis_key(layer: int, kind: str) -> str:
    return f"base_model.model.model.decoder.layers.{layer}.self_attn.{kind}_proj_basis"


def getU(B: Dict[str, torch.Tensor], layer: int, kind: str) -> torch.Tensor:
    return B[_basis_key(layer, kind)].float()


def _load_basis_pt(path: str) -> Dict[str, torch.Tensor]:
    p = path.strip()
    if not p:
        return {}
    if "weights_only" in _inspect.signature(torch.load).parameters:
        return torch.load(p, map_location="cpu", weights_only=True)
    return torch.load(p, map_location="cpu")


def merge_tail_into_basis(
    U_main: torch.Tensor,
    U_tail: Optional[torch.Tensor],
    tail_cols: int,
    alpha: float,
) -> torch.Tensor:
    r = U_main.shape[1]
    if U_tail is None or alpha == 0.0 or tail_cols <= 0:
        return U_main
    t = min(int(tail_cols), int(U_tail.shape[1]))
    if t == 0:
        return U_main
    tail_part = (float(alpha) * U_tail[:, :t]).float()
    if t >= r:
        Q, _ = torch.linalg.qr(tail_part, mode="reduced")
        return Q[:, :r].to(dtype=U_main.dtype)
    keep = r - t
    main_part = U_main[:, :keep].float()
    M = torch.cat([tail_part, main_part], dim=1)
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q[:, :r].to(dtype=U_main.dtype)


# -----------------------------------------------------------------------------
# checkpoint loading helpers
# -----------------------------------------------------------------------------
def _load_shard_state(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        if _safe_load_file is None:
            raise ImportError("需要 safetensors 以加载 .safetensors 分片")
        return dict(_safe_load_file(path, device="cpu"))
    if "weights_only" in _inspect.signature(torch.load).parameters:
        return torch.load(path, map_location="cpu", weights_only=True)
    return torch.load(path, map_location="cpu")


def _index_json_paths(folder: str) -> List[str]:
    out: List[str] = []
    for name in (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            out.append(p)
    if not out:
        for f in sorted(os.listdir(folder)):
            if f.endswith(".index.json"):
                out.append(os.path.join(folder, f))
    return out


def load_checkpoint_shape_filtered(model: torch.nn.Module, folder: str, dtype: torch.dtype) -> None:
    folder = os.path.abspath(folder)
    single_safe = os.path.join(folder, "model.safetensors")
    single_bin = os.path.join(folder, "pytorch_model.bin")
    if os.path.isfile(single_safe):
        shard_files = [single_safe]
    elif os.path.isfile(single_bin):
        shard_files = [single_bin]
    else:
        idx_paths = _index_json_paths(folder)
        if not idx_paths:
            raise FileNotFoundError(f"未在 {folder} 找到单文件权重或 *.index.json")
        with open(idx_paths[0], "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", idx)
        shard_files = sorted({os.path.join(folder, name) for name in weight_map.values()})

    msd = model.state_dict()
    matched = 0
    skipped_mismatch = 0

    for path in shard_files:
        shard = _load_shard_state(path)
        to_load: Dict[str, torch.Tensor] = {}
        for k, v in shard.items():
            if k not in msd:
                continue
            if msd[k].shape != v.shape:
                skipped_mismatch += 1
                continue
            if v.is_floating_point():
                v = v.to(dtype=dtype)
            to_load[k] = v
        if not to_load:
            continue
        try:
            model.load_state_dict(to_load, strict=False, assign=True)
        except TypeError:
            model.load_state_dict(to_load, strict=False)
        matched += len(to_load)

    print(
        f"[load_ckpt] 已写入形状匹配参数: {matched} 个；"
        f"跳过形状不一致: {skipped_mismatch}"
    )


def _materialize_meta_parameters(model: torch.nn.Module, float_dtype: torch.dtype) -> None:
    for name, param in list(model.named_parameters(recurse=True)):
        if not getattr(param, "is_meta", False):
            continue
        path_str, leaf = name.rsplit(".", 1)
        parent = model
        for p in path_str.split("."):
            if not p:
                continue
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        dt = float_dtype if param.is_floating_point() else param.dtype
        buf = torch.nn.Parameter(
            torch.empty(tuple(param.shape), dtype=dt, device=torch.device("cpu")),
            requires_grad=param.requires_grad,
        )
        setattr(parent, leaf, buf)


# -----------------------------------------------------------------------------
# data / eval
# -----------------------------------------------------------------------------
def _iter_jsonl_texts_gz(fp: str, text_key: str = "text", read_size: int = 1 << 20):
    """
    Robust reader for C4 .json.gz shards. c4-validation shards may contain
    concatenated JSON objects, so line-by-line json.loads is not enough.
    """
    dec = json.JSONDecoder()
    buf = ""

    def _skip_seps(s: str, i: int, n: int) -> int:
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
                    t = obj.get(text_key)
                    if isinstance(t, str) and t.strip():
                        yield t.strip()
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
                t = obj.get(text_key)
                if isinstance(t, str) and t.strip():
                    yield t.strip()


def build_doc_pool_from_c4_dir(
    data_dir: str,
    glob_pat: str,
    pool_docs: int,
    scan_lines: int,
    seed: int,
) -> Tuple[List[str], Dict[str, object]]:
    files = sorted(glob.glob(os.path.join(data_dir, glob_pat)))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "**", glob_pat), recursive=True))
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pat!r} under {data_dir}")

    rng = random.Random(seed)
    reservoir: List[str] = []
    seen = 0
    first_file = files[0]
    for fp in files:
        if fp.endswith(".gz"):
            text_iter = _iter_jsonl_texts_gz(fp)
        else:
            text_iter = iter_texts_from_dir(os.path.dirname(fp), max_files=1, max_lines=scan_lines)
        for t in text_iter:
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
        raise RuntimeError(f"Failed to read any text from {first_file}")
    return reservoir, {"seen_docs": seen, "pool_docs": len(reservoir), "file": first_file}


def build_c4_window_batches(tok, doc_pool: List[str], seqlen: int, n_batches: int, seed: int) -> List[torch.Tensor]:
    rng = random.Random(seed)
    batches: List[torch.Tensor] = []
    max_tries = max(2000, n_batches * 200)
    tries = 0
    while len(batches) < n_batches and tries < max_tries:
        tries += 1
        text = doc_pool[rng.randint(0, len(doc_pool) - 1)]
        enc = tok(text, return_tensors="pt")
        ids = enc.input_ids
        if ids.shape[1] <= seqlen:
            continue
        start = rng.randint(0, ids.shape[1] - seqlen - 1)
        batches.append(ids[:, start : start + seqlen].contiguous())
    if len(batches) != n_batches:
        raise RuntimeError(
            f"Only sampled {len(batches)}/{n_batches} eval windows. "
            "Increase --c4_scan_lines / --c4_pool_docs or reduce --seqlen."
        )
    return batches


def build_token_stream_batches(tok, data_dir, seqlen, n_batches, seed, max_data_files, max_data_lines):
    texts = iter_texts_from_dir(data_dir, max_files=max_data_files, max_lines=max_data_lines)
    stream = build_token_stream(tok, texts, target_tokens=max(300000, n_batches * seqlen * 6), seed=seed)
    if stream is None:
        raise RuntimeError("empty token stream")
    g = torch.Generator().manual_seed(seed)
    N = stream.numel()
    batches = []
    for _ in range(n_batches):
        s = int(torch.randint(low=0, high=N - seqlen - 1, size=(1,), generator=g).item())
        batches.append(stream[s : s + seqlen].unsqueeze(0))
    return batches


def _batches_to_testenc(batches: List[torch.Tensor]) -> torch.Tensor:
    """[1, S] * n -> [1, n*S]，与 Utkarsh / sanity_AB_opt30b 的 chunk 拼接方式一致。"""
    if not batches:
        raise ValueError("batches 为空")
    return torch.cat(batches, dim=1)


@torch.no_grad()
def eval_nll_utkarsh_same_as_sanity(
    model,
    batches: List[torch.Tensor],
    device,
    seqlen: int,
    limit: int = -1,
) -> Tuple[float, float]:
    """
    使用与 `sanity_AB_opt30b.py` 一致的 decoder+lm_head 前向路径，但聚合改成
    token-weighted CE：

    - 前向：`outputs = model.model.decoder(batch)` -> `logits = model.lm_head(hidden_states)`
    - 每块：对 shift 后真实 token 数做 reduction=sum
    - 标量：`ce = total_nll / total_tokens`；`ppl = exp(ce)`

    返回:
      (ce, ppl) —— 第一个量用于本脚本的 accept / rollback；第二个量便于对表。
    """
    seqlen = int(seqlen)
    dev = str(device) if not isinstance(device, str) else device
    model.eval()
    use_cache = getattr(model.config, "use_cache", False) if hasattr(model, "config") else False
    if hasattr(model, "config"):
        model.config.use_cache = False
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    nll_sum = torch.zeros((), dtype=torch.float32, device=dev)
    ntokens = 0
    used = 0
    for i, batch_cpu in enumerate(batches):
        if limit >= 0 and used >= limit:
            break
        batch = batch_cpu[:, :seqlen].to(dev)
        outputs = model.model.decoder(batch)
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:].to(model.lm_head.weight.device)
        labels = shift_labels.reshape(-1)
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)).float(), labels)
        nll_sum += loss.float()
        ntokens += int(labels.numel())
        used += 1
    if ntokens == 0:
        if hasattr(model, "config"):
            model.config.use_cache = use_cache
        return float("nan"), float("nan")
    log_ppl = (nll_sum / float(ntokens)).item()
    ppl = float(math.exp(log_ppl))
    if hasattr(model, "config"):
        model.config.use_cache = use_cache
    return float(log_ppl), float(ppl)


@torch.no_grad()
def eval_ce_per_batch(
    model,
    batches: List[torch.Tensor],
    device,
    seqlen: int,
) -> List[float]:
    seqlen = int(seqlen)
    dev = str(device) if not isinstance(device, str) else device
    model.eval()
    use_cache = getattr(model.config, "use_cache", False) if hasattr(model, "config") else False
    if hasattr(model, "config"):
        model.config.use_cache = False
    loss_fct = torch.nn.CrossEntropyLoss()
    out: List[float] = []
    for batch_cpu in batches:
        batch = batch_cpu[:, :seqlen].to(dev)
        outputs = model.model.decoder(batch)
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:].to(model.lm_head.weight.device)
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
        )
        out.append(float(loss.item()))
    if hasattr(model, "config"):
        model.config.use_cache = use_cache
    return out


# -----------------------------------------------------------------------------
# score-temp helpers
# -----------------------------------------------------------------------------
def _snapshot_qk_low(layer) -> Dict[str, torch.Tensor]:
    attn = layer.self_attn
    return {
        "q_w": attn.q_proj_low.weight.detach().clone(),
        "q_b": attn.q_proj_low.bias.detach().clone(),
        "k_w": attn.k_proj_low.weight.detach().clone(),
        "k_b": attn.k_proj_low.bias.detach().clone(),
    }


def _restore_qk_low(layer, snap: Dict[str, torch.Tensor]) -> None:
    attn = layer.self_attn
    attn.q_proj_low.weight.copy_(snap["q_w"])
    attn.q_proj_low.bias.copy_(snap["q_b"])
    attn.k_proj_low.weight.copy_(snap["k_w"])
    attn.k_proj_low.bias.copy_(snap["k_b"])


def _apply_score_temp_global(layer, temp: float, snap: Dict[str, torch.Tensor]) -> None:
    """全层统一 temperature；通过 q/k low 同乘 sqrt(temp) 实现 logits × temp。"""
    s = math.sqrt(float(temp))
    attn = layer.self_attn
    attn.q_proj_low.weight.copy_(snap["q_w"] * s)
    attn.q_proj_low.bias.copy_(snap["q_b"] * s)
    attn.k_proj_low.weight.copy_(snap["k_w"] * s)
    attn.k_proj_low.bias.copy_(snap["k_b"] * s)


def _apply_score_temp_headwise(layer, temps: List[float], snap: Dict[str, torch.Tensor]) -> None:
    """每个 head 独立 temperature；缩放对应 low-rank row slice。"""
    attn = layer.self_attn
    n_heads = int(attn.num_heads)
    rank_kq = int(attn.rank_kq)
    if rank_kq % n_heads != 0:
        raise ValueError(f"rank_kq={rank_kq} must be divisible by num_heads={n_heads}")
    if len(temps) != n_heads:
        raise ValueError(f"expected {n_heads} head temps, got {len(temps)}")

    attn.q_proj_low.weight.copy_(snap["q_w"])
    attn.q_proj_low.bias.copy_(snap["q_b"])
    attn.k_proj_low.weight.copy_(snap["k_w"])
    attn.k_proj_low.bias.copy_(snap["k_b"])

    width = rank_kq // n_heads
    for h, temp in enumerate(temps):
        s = math.sqrt(float(temp))
        sl = slice(h * width, (h + 1) * width)
        attn.q_proj_low.weight[sl].mul_(s)
        attn.q_proj_low.bias[sl].mul_(s)
        attn.k_proj_low.weight[sl].mul_(s)
        attn.k_proj_low.bias[sl].mul_(s)


# -----------------------------------------------------------------------------
# rebuild helpers (vtail layers)
# -----------------------------------------------------------------------------
def _snapshot_vtail_related(layer) -> Dict[str, torch.Tensor]:
    attn = layer.self_attn
    out = {
        "q_w": attn.q_proj_low.weight.detach().clone(),
        "q_b": attn.q_proj_low.bias.detach().clone(),
        "k_w": attn.k_proj_low.weight.detach().clone(),
        "k_b": attn.k_proj_low.bias.detach().clone(),
        "v_w": attn.v_proj_low.weight.detach().clone(),
        "v_b": attn.v_proj_low.bias.detach().clone(),
        "o_w": attn.out_proj_up.weight.detach().clone(),
    }
    if getattr(attn.out_proj_up, "bias", None) is not None:
        out["o_b"] = attn.out_proj_up.bias.detach().clone()
    if hasattr(attn, "v_rot") and torch.is_tensor(getattr(attn.v_rot, "weight", None)):
        out["vrot_w"] = attn.v_rot.weight.detach().clone()
    if hasattr(attn, "vscale_log") and torch.is_tensor(attn.vscale_log):
        out["vscale"] = attn.vscale_log.detach().clone()
    return out


def _restore_vtail_related(layer, snap: Dict[str, torch.Tensor]) -> None:
    attn = layer.self_attn
    attn.q_proj_low.weight.copy_(snap["q_w"])
    attn.q_proj_low.bias.copy_(snap["q_b"])
    attn.k_proj_low.weight.copy_(snap["k_w"])
    attn.k_proj_low.bias.copy_(snap["k_b"])
    attn.v_proj_low.weight.copy_(snap["v_w"])
    attn.v_proj_low.bias.copy_(snap["v_b"])
    attn.out_proj_up.weight.copy_(snap["o_w"])
    if "o_b" in snap and getattr(attn.out_proj_up, "bias", None) is not None:
        attn.out_proj_up.bias.copy_(snap["o_b"])
    if "vrot_w" in snap and hasattr(attn, "v_rot") and torch.is_tensor(getattr(attn.v_rot, "weight", None)):
        attn.v_rot.weight.copy_(snap["vrot_w"])
    if "vscale" in snap and hasattr(attn, "vscale_log") and torch.is_tensor(attn.vscale_log):
        attn.vscale_log.copy_(snap["vscale"])


def _rebuild_layer_from_basis(
    *,
    base_model,
    cmp_model,
    layer_idx: int,
    basis_state: Dict[str, torch.Tensor],
    tail_state: Dict[str, torch.Tensor],
    tail_alpha: float,
    tail_cols: int,
    tail_branches: Set[str],
    keqk: int,
    rebuild_mode: str,
):
    b_attn = base_model.model.decoder.layers[layer_idx].self_attn
    c_attn = cmp_model.model.decoder.layers[layer_idx].self_attn
    d = base_model.config.hidden_size

    Uq = getU(basis_state, layer_idx, "q")
    Uk = getU(basis_state, layer_idx, "k")
    Uv = getU(basis_state, layer_idx, "v")

    use_tail = bool(tail_state) and float(tail_alpha) != 0.0 and len(tail_branches) > 0
    if use_tail:
        if "q" in tail_branches:
            Uq = merge_tail_into_basis(Uq, getU(tail_state, layer_idx, "q"), tail_cols, tail_alpha)
        if "k" in tail_branches:
            if int(keqk) == 1:
                if "q" not in tail_branches:
                    Uq = merge_tail_into_basis(Uq, getU(tail_state, layer_idx, "k"), tail_cols, tail_alpha)
            else:
                Uk = merge_tail_into_basis(Uk, getU(tail_state, layer_idx, "k"), tail_cols, tail_alpha)
        if "v" in tail_branches:
            Uv = merge_tail_into_basis(Uv, getU(tail_state, layer_idx, "v"), tail_cols, tail_alpha)

    if int(keqk) == 1:
        Uk = Uq

    rebuild_qk = rebuild_mode == "full"
    rebuild_specs = []
    if rebuild_qk:
        rebuild_specs.extend([("q", Uq, "q_proj"), ("k", Uk, "k_proj")])
    rebuild_specs.append(("v", Uv, "v_proj"))

    for kind, U, proj in rebuild_specs:
        W = getattr(b_attn, proj).weight.detach().float()
        b = getattr(b_attn, proj).bias.detach().float()
        low_layer = getattr(c_attn, f"{proj}_low")
        r = low_layer.weight.shape[0]
        assert U.shape[0] == d and U.shape[1] == r, f"layer {layer_idx} U{kind} shape {tuple(U.shape)} != ({d},{r})"
        Wlow = (U.t() @ W).contiguous()
        blow = (b @ U).contiguous()
        low_layer.weight.copy_(Wlow.to(dtype=low_layer.weight.dtype))
        low_layer.bias.copy_(blow.to(dtype=low_layer.bias.dtype))

    Wo = b_attn.out_proj.weight.detach().float()
    bo = b_attn.out_proj.bias.detach().float()
    out_up = c_attn.out_proj_up
    r_v = out_up.weight.shape[1]
    assert Uv.shape == (d, r_v), f"layer {layer_idx} Uv shape {tuple(Uv.shape)} != ({d},{r_v})"
    Wup = (Wo @ Uv).contiguous()
    out_up.weight.copy_(Wup.to(dtype=out_up.weight.dtype))
    if getattr(out_up, "bias", None) is not None:
        out_up.bias.copy_(bo.to(dtype=out_up.bias.dtype))

    if hasattr(c_attn, "v_rot") and torch.is_tensor(getattr(c_attn.v_rot, "weight", None)):
        c_attn.v_rot.weight.copy_(torch.eye(c_attn.v_rot.weight.shape[0], dtype=c_attn.v_rot.weight.dtype, device=c_attn.v_rot.weight.device))
    if hasattr(c_attn, "vscale_log") and torch.is_tensor(c_attn.vscale_log):
        c_attn.vscale_log.data.fill_(0.0)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp_in", required=True)
    ap.add_argument("--cmp_out", required=True)
    ap.add_argument("--low_json", required=True)
    ap.add_argument("--basis_pt", required=True)
    ap.add_argument("--tail_basis_pt", default="")
    ap.add_argument("--tail_alpha", type=float, default=0.0)
    ap.add_argument("--tail_cols", type=int, default=4)
    ap.add_argument("--tail_branches", type=str, default="v")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--keqk", type=int, default=0)

    ap.add_argument("--score_layers", type=str, default="1-18")
    ap.add_argument("--vtail_layers", type=str, default="19-27,47")
    ap.add_argument("--score_mode", type=str, default="global", choices=["global", "per_head"],
                    help="score calibration 粒度；per_head 更细但评估开销约为 num_heads 倍")
    ap.add_argument("--score_temp_grid", type=str, default="0.90,0.95,1.00,1.05,1.10")
    ap.add_argument("--score_head_accept_delta_ce", type=float, default=-1.0,
                    help="per_head 每个 head 的接受阈值；<0 时使用 --accept_delta_ce")
    ap.add_argument("--score_sanity_temp", type=float, default=0.5,
                    help="score层灵敏度诊断用的强扰动温度；<=0 则跳过")
    ap.add_argument("--vtail_rebuild_mode", type=str, default="v_only", choices=["v_only", "full"],
                    help="v_only 只重建 V/out_up，保留 cmp_in 的 Q/K；full 为旧行为，Q/K/V 都重建")
    ap.add_argument("--accept_delta_ce", type=float, default=2e-4,
                    help="局部接受阈值：candidate CE 至少比 baseline 降 accept_delta_ce 才接受")
    ap.add_argument("--rollback_delta_ce", type=float, default=0.0,
                    help="最终 holdout 若劣于原模型超过该值，则整体回滚为 cmp_in")

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--n_linesearch", type=int, default=8)
    ap.add_argument("--n_holdout", type=int, default=8)
    ap.add_argument("--eval_data_mode", type=str, default="c4_windows", choices=["c4_windows", "token_stream"],
                    help="c4_windows 与 sanity_AB_opt30b.py 的 C4 validation window 采样一致")
    ap.add_argument("--c4_glob", type=str, default="c4-validation.*.json.gz")
    ap.add_argument("--c4_scan_lines", type=int, default=500000)
    ap.add_argument("--c4_pool_docs", type=int, default=500000)
    ap.add_argument("--c4_val_seed", type=int, default=0)
    ap.add_argument("--max_data_files", type=int, default=200)
    ap.add_argument("--max_data_lines", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--debug_eval_diagnostics", type=int, default=1, choices=[0, 1],
                    help="打印 linesearch/holdout 每batch CE统计与 score 温度灵敏度诊断")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(args.device)

    # load low rank config
    with open(args.low_json, "r", encoding="utf-8") as f:
        low_raw = json.load(f)

    cmp_cfg = AutoConfig.from_pretrained(args.cmp_in, local_files_only=True)
    n_layers = int(cmp_cfg.num_hidden_layers)
    cmp_cfg.__dict__.pop("low_rank_config", None)
    cmp_cfg._attn_implementation = "eager"
    low = _normalize_low_rank_config(low_raw, n_layers)

    score_layers = _parse_layers_spec(args.score_layers, n_layers)
    vtail_layers = _parse_layers_spec(args.vtail_layers, n_layers)
    temp_grid = _parse_float_grid(args.score_temp_grid)
    print(f"[layers] score_layers={sorted(score_layers)}")
    print(f"[layers] vtail_layers={sorted(vtail_layers)}")
    print(f"[score] mode={args.score_mode} temp_grid={temp_grid}")
    print(f"[vtail] rebuild_mode={args.vtail_rebuild_mode}")

    # branch set
    tail_branches: Set[str] = set()
    for part in (args.tail_branches or "v").lower().replace(" ", "").split(","):
        if not part:
            continue
        if part in ("q", "k", "v"):
            tail_branches.add(part)
        elif part == "qk":
            tail_branches.update(("q", "k"))
        elif part == "qkv":
            tail_branches.update(("q", "k", "v"))
        else:
            raise ValueError(f"unknown --tail_branches part: {part!r}")

    # tokenizer + eval batches
    tok_path = args.tokenizer.strip() if args.tokenizer.strip() else args.base
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.eval_data_mode == "c4_windows":
        doc_pool, pool_info = build_doc_pool_from_c4_dir(
            args.data_dir,
            glob_pat=args.c4_glob,
            pool_docs=int(args.c4_pool_docs),
            scan_lines=int(args.c4_scan_lines),
            seed=int(args.seed),
        )
        print(f"[eval-data] c4_windows pool={pool_info}")
        line_batches = build_c4_window_batches(
            tok, doc_pool, int(args.seqlen), int(args.n_linesearch), int(args.c4_val_seed)
        )
        hold_batches = build_c4_window_batches(
            tok, doc_pool, int(args.seqlen), int(args.n_holdout), int(args.c4_val_seed) + 12345
        )
    else:
        line_batches = build_token_stream_batches(tok, args.data_dir, args.seqlen, args.n_linesearch, args.seed,
                                                  args.max_data_files, args.max_data_lines)
        hold_batches = build_token_stream_batches(tok, args.data_dir, args.seqlen, args.n_holdout, args.seed + 123,
                                                  args.max_data_files, args.max_data_lines)

    # load basis
    B = _load_basis_pt(args.basis_pt)
    B_tail = _load_basis_pt(args.tail_basis_pt) if args.tail_basis_pt.strip() else {}

    # load base on CPU
    print("[load] base -> cpu")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True
    ).cpu().eval()

    # load cmp on GPU skeleton + weights
    print(f"[load] cmp_in -> build on {device}")
    with ContextManagers([no_init_weights(_enable=True), init_empty_weights()]):
        cmpm = OPTForCausalLM_EigenAttn(cmp_cfg, low)
    if hasattr(cmpm, "tie_weights"):
        cmpm.tie_weights()
    load_checkpoint_shape_filtered(cmpm, args.cmp_in, dtype)
    _materialize_meta_parameters(cmpm, dtype)
    cmpm.to(device)
    cmpm.eval()

    # baseline: decoder+lm_head 路径，使用 token-weighted CE，避免 tiny eval 集上聚合/limit 误差。
    line_log_base, line_ppl_base = eval_nll_utkarsh_same_as_sanity(
        cmpm, line_batches, device, int(args.seqlen),
    )
    hold_log_base, hold_ppl_base = eval_nll_utkarsh_same_as_sanity(
        cmpm, hold_batches, device, int(args.seqlen),
    )
    print(
        f"[baseline] linesearch tokenCE={line_log_base:.6f}  PPL={line_ppl_base:.6f}",
        flush=True,
    )
    print(
        f"[baseline] holdout    tokenCE={hold_log_base:.6f}  PPL={hold_ppl_base:.6f}",
        flush=True,
    )
    if int(args.debug_eval_diagnostics) == 1:
        line_chunk_ce = eval_ce_per_batch(cmpm, line_batches, device, int(args.seqlen))
        hold_chunk_ce = eval_ce_per_batch(cmpm, hold_batches, device, int(args.seqlen))
        line_t = torch.tensor(line_chunk_ce, dtype=torch.float32)
        hold_t = torch.tensor(hold_chunk_ce, dtype=torch.float32)
        print(
            f"[diag] line batchCE: mean={line_t.mean().item():.6f} std={line_t.std(unbiased=False).item():.6e} "
            f"min={line_t.min().item():.6f} max={line_t.max().item():.6f}",
            flush=True,
        )
        print(
            f"[diag] hold batchCE: mean={hold_t.mean().item():.6f} std={hold_t.std(unbiased=False).item():.6e} "
            f"min={hold_t.min().item():.6f} max={hold_t.max().item():.6f}",
            flush=True,
        )

    report = {
        "baseline": {
            "line_log_ppl": float(line_log_base),
            "line_ppl": float(line_ppl_base),
            "hold_log_ppl": float(hold_log_base),
            "hold_ppl": float(hold_ppl_base),
            "line_ce": float(line_log_base),
            "hold_ce": float(hold_log_base),
            "note": "line_ce/hold_ce 为 token-weighted CE；PPL=exp(CE)。前向使用 decoder+lm_head 路径，eval_data_mode=c4_windows 时采样方式与 sanity_AB 的 C4 validation window 对齐。",
            "eval_data_mode": args.eval_data_mode,
            "score_mode": args.score_mode,
            "vtail_rebuild_mode": args.vtail_rebuild_mode,
        },
        "score_layers": [],
        "vtail_layers": [],
        "global": {},
    }

    # -------------------------
    # Phase A: score calibration
    # -------------------------
    if int(args.debug_eval_diagnostics) == 1 and len(score_layers) > 0 and float(args.score_sanity_temp) > 0:
        li0 = sorted(score_layers)[0]
        layer0 = cmpm.model.decoder.layers[li0]
        snap0 = _snapshot_qk_low(layer0)
        probe_base, _ = eval_nll_utkarsh_same_as_sanity(cmpm, line_batches, device, int(args.seqlen))
        if args.score_mode == "global":
            _apply_score_temp_global(layer0, float(args.score_sanity_temp), snap0)
        else:
            n_heads0 = int(layer0.self_attn.num_heads)
            _apply_score_temp_headwise(layer0, [float(args.score_sanity_temp)] * n_heads0, snap0)
        probe_cand, _ = eval_nll_utkarsh_same_as_sanity(cmpm, line_batches, device, int(args.seqlen))
        _restore_qk_low(layer0, snap0)
        delta_probe = float(probe_cand - probe_base)
        print(
            f"[diag] score sensitivity(layer {li0:02d}, temp={float(args.score_sanity_temp):.3f}): "
            f"deltaCE={delta_probe:+.6e}",
            flush=True,
        )
        if abs(delta_probe) < 1e-7:
            print(
                "[diag][warn] score perturbation has near-zero CE effect; "
                "rollback decisions may all be neutral under current metric.",
                flush=True,
            )

    for li in sorted(score_layers):
        layer = cmpm.model.decoder.layers[li]
        snap = _snapshot_qk_low(layer)
        if not report["score_layers"] and not report["vtail_layers"]:
            best_log = float(line_log_base)
        else:
            best_log, _ = eval_nll_utkarsh_same_as_sanity(cmpm, line_batches, device, int(args.seqlen))
        layer_base_log = best_log

        best_temp: object = 1.0
        per_grid: object
        if args.score_mode == "global":
            per_grid_list: List[Tuple[float, float]] = [(1.0, best_log)]
            for t in temp_grid:
                _restore_qk_low(layer, snap)
                _apply_score_temp_global(layer, t, snap)
                lg, _p = eval_nll_utkarsh_same_as_sanity(cmpm, line_batches, device, int(args.seqlen))
                per_grid_list.append((float(t), float(lg)))
                if lg < best_log - float(args.accept_delta_ce):
                    best_log = lg
                    best_temp = float(t)
            per_grid = per_grid_list
        else:
            attn = layer.self_attn
            n_heads = int(attn.num_heads)
            head_delta = float(args.score_head_accept_delta_ce)
            if head_delta < 0:
                head_delta = float(args.accept_delta_ce)
            temps = [1.0] * n_heads
            per_head_grid: List[Dict[str, object]] = []
            for h in range(n_heads):
                head_base_log = best_log
                head_best_log = best_log
                head_best_temp = temps[h]
                trials: List[Tuple[float, float]] = [(float(temps[h]), float(best_log))]
                for t in temp_grid:
                    cand_temps = list(temps)
                    cand_temps[h] = float(t)
                    _apply_score_temp_headwise(layer, cand_temps, snap)
                    lg, _p = eval_nll_utkarsh_same_as_sanity(cmpm, line_batches, device, int(args.seqlen))
                    trials.append((float(t), float(lg)))
                    if lg < head_best_log - head_delta:
                        head_best_log = lg
                        head_best_temp = float(t)
                if head_best_temp != temps[h]:
                    temps[h] = float(head_best_temp)
                    best_log = float(head_best_log)
                    _apply_score_temp_headwise(layer, temps, snap)
                else:
                    _apply_score_temp_headwise(layer, temps, snap)
                per_head_grid.append({
                    "head": int(h),
                    "baseline_log_ppl": float(head_base_log),
                    "best_log_ppl": float(head_best_log),
                    "best_temp": float(head_best_temp),
                    "accepted": bool(head_best_temp != 1.0),
                    "grid": trials,
                })
            best_temp = temps
            per_grid = per_head_grid

        # commit or rollback this layer
        _restore_qk_low(layer, snap)
        accepted = best_log < layer_base_log - float(args.accept_delta_ce)
        if accepted:
            if args.score_mode == "global":
                _apply_score_temp_global(layer, float(best_temp), snap)
            else:
                _apply_score_temp_headwise(layer, list(best_temp), snap)  # type: ignore[arg-type]
        report["score_layers"].append({
            "layer": int(li),
            "baseline_log_ppl": float(layer_base_log),
            "best_log_ppl": float(best_log),
            "score_mode": args.score_mode,
            "best_temp": best_temp,
            "baseline_ce": float(layer_base_log),
            "best_ce": float(best_log),
            "accepted": bool(accepted),
            "grid": per_grid,
        })
        temp_msg = f"{float(best_temp):.3f}" if args.score_mode == "global" else "per_head"
        print(
            f"[score] layer {li:02d} {'ACCEPT' if accepted else 'ROLLBACK'} temp={temp_msg}  "
            f"line tokenCE {layer_base_log:.6f}->{best_log:.6f}",
            flush=True,
        )

    # -------------------------
    # Phase B: V-tail rebuild on selected layers
    # -------------------------
    for li in sorted(vtail_layers):
        layer = cmpm.model.decoder.layers[li]
        snap = _snapshot_vtail_related(layer)
        base_log, _ = eval_nll_utkarsh_same_as_sanity(cmpm, line_batches, device, int(args.seqlen))
        _rebuild_layer_from_basis(
            base_model=base,
            cmp_model=cmpm,
            layer_idx=li,
            basis_state=B,
            tail_state=B_tail,
            tail_alpha=float(args.tail_alpha),
            tail_cols=int(args.tail_cols),
            tail_branches=tail_branches,
            keqk=int(args.keqk),
            rebuild_mode=args.vtail_rebuild_mode,
        )
        cand_log, _ = eval_nll_utkarsh_same_as_sanity(cmpm, line_batches, device, int(args.seqlen))
        accepted = cand_log < base_log - float(args.accept_delta_ce)
        if not accepted:
            _restore_vtail_related(layer, snap)
        report["vtail_layers"].append({
            "layer": int(li),
            "baseline_log_ppl": float(base_log),
            "candidate_log_ppl": float(cand_log),
            "baseline_ce": float(base_log),
            "candidate_ce": float(cand_log),
            "rebuild_mode": args.vtail_rebuild_mode,
            "accepted": bool(accepted),
        })
        print(
            f"[vtail] layer {li:02d} {'ACCEPT' if accepted else 'ROLLBACK'}  "
            f"line tokenCE {base_log:.6f}->{cand_log:.6f}",
            flush=True,
        )

    # final global holdout gate
    line_log_final, line_ppl_final = eval_nll_utkarsh_same_as_sanity(
        cmpm, line_batches, device, int(args.seqlen),
    )
    hold_log_final, hold_ppl_final = eval_nll_utkarsh_same_as_sanity(
        cmpm, hold_batches, device, int(args.seqlen),
    )
    delta_hold = hold_log_final - hold_log_base
    rollback_all = delta_hold > float(args.rollback_delta_ce)
    report["global"] = {
        "final_line_log_ppl": float(line_log_final),
        "final_line_ppl": float(line_ppl_final),
        "final_hold_log_ppl": float(hold_log_final),
        "final_hold_ppl": float(hold_ppl_final),
        "final_line_ce": float(line_log_final),
        "final_hold_ce": float(hold_log_final),
        "hold_delta": float(delta_hold),
        "rollback_delta_ce": float(args.rollback_delta_ce),
        "rolled_back_all": bool(rollback_all),
    }
    print(f"[global] final linesearch tokenCE = {line_log_final:.6f}  PPL = {line_ppl_final:.6f}")
    print(
        f"[global] final holdout    tokenCE = {hold_log_final:.6f}  PPL = {hold_ppl_final:.6f}  "
        f"delta={delta_hold:+.6f} (nats, vs baseline holdout)",
    )

    os.makedirs(args.cmp_out, exist_ok=True)
    report_path = os.path.join(args.cmp_out, "e3_fixedrank_report.json")

    if rollback_all:
        print("[global] HOLDOUT FAILED -> rollback all: copy cmp_in to cmp_out")
        # save report first in a temp place under cmp_out after copy
        if os.path.abspath(args.cmp_in) != os.path.abspath(args.cmp_out):
            shutil.rmtree(args.cmp_out, ignore_errors=True)
            shutil.copytree(args.cmp_in, args.cmp_out)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[saved] original cmp copied to {args.cmp_out}")
        print(f"[report] {report_path}")
        return

    # otherwise save modified checkpoint
    cmpm.save_pretrained(args.cmp_out, safe_serialization=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[saved] modified cmp -> {args.cmp_out}")
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
