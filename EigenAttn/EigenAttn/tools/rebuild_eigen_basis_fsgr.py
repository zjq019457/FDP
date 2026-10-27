#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSGR: Fisher-guided Shared-Basis Reconstruction

Offline tool to rebuild eigen_basis.pt into a more structured basis file:
  - layer-segment shared basis (default for K/V)
  - fisher-like rank allocation (calibration-driven)
  - per-layer orthogonal Procrustes alignment

This script is intentionally standalone and does NOT modify training code paths.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _basis_key(layer_idx: int, branch: str) -> str:
    return f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{branch}_proj_basis"


def load_basis_file(path: str) -> Dict[str, torch.Tensor]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"basis file not found: {p}")
    state = torch.load(p, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"basis file must be dict, got: {type(state)}")
    return state


def _iter_basis_keys(state: Dict[str, torch.Tensor], branch: str) -> List[str]:
    suffix = f".self_attn.{branch}_proj_basis"
    ks = [k for k in state.keys() if isinstance(k, str) and k.endswith(suffix)]
    return sorted(ks, key=lambda x: int(x.split(".layers.")[1].split(".")[0]))


def _parse_layer_from_key(k: str) -> int:
    return int(k.split(".layers.")[1].split(".")[0])


def check_basis_integrity(
    state: Dict[str, torch.Tensor],
    out_json: Path,
) -> Dict[str, dict]:
    summary: Dict[str, dict] = {"branches": {}, "global": {}}
    for branch in ("q", "k", "v"):
        keys = _iter_basis_keys(state, branch)
        if not keys:
            raise RuntimeError(f"no keys found for branch {branch}")
        stats = []
        for k in keys:
            t = state[k]
            if not torch.is_tensor(t):
                raise TypeError(f"{k} is not tensor: {type(t)}")
            if t.dim() != 2:
                raise ValueError(f"{k} must be 2D [d,r], got {tuple(t.shape)}")
            if not torch.isfinite(t).all():
                raise ValueError(f"{k} contains NaN/Inf")
            d, r = t.shape
            gram = (t.float().T @ t.float())
            I = torch.eye(r, dtype=gram.dtype)
            ortho_err = float((gram - I).pow(2).mean().sqrt().item())
            stats.append(
                {
                    "key": k,
                    "layer": _parse_layer_from_key(k),
                    "shape": [int(d), int(r)],
                    "dtype": str(t.dtype),
                    "absmax": float(t.abs().max().item()),
                    "mean": float(t.float().mean().item()),
                    "std": float(t.float().std().item()),
                    "orthogonality_rmse": ortho_err,
                }
            )
        summary["branches"][branch] = stats
    summary["global"]["num_layers"] = len(summary["branches"]["q"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def compute_subspace_similarity(
    basis_by_layer: Dict[int, torch.Tensor]
) -> torch.Tensor:
    layers = sorted(basis_by_layer.keys())
    n = len(layers)
    sims = torch.zeros(n, n, dtype=torch.float32)
    for i, li in enumerate(layers):
        ui = basis_by_layer[li].float()
        ri = ui.shape[1]
        for j, lj in enumerate(layers):
            uj = basis_by_layer[lj].float()
            rj = uj.shape[1]
            r = float(min(ri, rj))
            a = ui[:, : int(r)].T @ uj[:, : int(r)]
            sims[i, j] = (a.pow(2).sum() / max(1.0, r)).float()
    return sims


def _write_similarity_outputs(mat: torch.Tensor, layers: List[int], prefix: Path) -> None:
    torch.save({"layers": layers, "similarity": mat}, str(prefix) + ".pt")
    with open(str(prefix) + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + [str(x) for x in layers])
        for i, li in enumerate(layers):
            w.writerow([li] + [float(mat[i, j].item()) for j in range(len(layers))])


def cluster_layers_contiguous(
    layers: List[int],
    num_segments: int,
) -> List[List[int]]:
    if num_segments <= 1:
        return [layers]
    n = len(layers)
    segs = []
    start = 0
    for s in range(num_segments):
        end = round((s + 1) * n / num_segments)
        seg = layers[start:end]
        if seg:
            segs.append(seg)
        start = end
    return segs


def cluster_layers_global_greedy(
    layers: List[int],
    sim: torch.Tensor,
    num_segments: int,
) -> List[List[int]]:
    # Simple k-medoids-like greedy assignment using distance = 1 - similarity.
    n = len(layers)
    if num_segments <= 1 or n <= num_segments:
        return [[l] for l in layers] if n <= num_segments else [layers]
    medoids = [0]
    for _ in range(1, num_segments):
        best_idx = None
        best_gain = -1.0
        for cand in range(n):
            if cand in medoids:
                continue
            gain = 0.0
            for i in range(n):
                old = max(float(sim[i, m].item()) for m in medoids)
                new = max(old, float(sim[i, cand].item()))
                gain += (new - old)
            if gain > best_gain:
                best_gain = gain
                best_idx = cand
        medoids.append(int(best_idx))
    clusters: Dict[int, List[int]] = {m: [] for m in medoids}
    for i in range(n):
        m = max(medoids, key=lambda x: float(sim[i, x].item()))
        clusters[m].append(layers[i])
    return [clusters[m] for m in medoids]


def build_shared_basis_from_projection_mean(
    layer_basis: Dict[int, torch.Tensor],
    segment_layers: List[int],
    max_rank: int,
) -> torch.Tensor:
    # Equivalent to top eigvecs of mean projection, via low-rank PCA on concatenated bases.
    us = [layer_basis[l].float() for l in segment_layers]
    m = len(us)
    cat = torch.cat([u / math.sqrt(max(1, m)) for u in us], dim=1)  # [d, m*r]
    d, n = cat.shape
    q = min(max_rank, d, n)
    if q <= 0:
        raise ValueError(f"invalid q={q} for cat shape {tuple(cat.shape)}")

    cat_cpu = cat.detach().cpu()

    def _from_pca_lowrank(x: torch.Tensor) -> torch.Tensor:
        U, _, _ = torch.pca_lowrank(x, q=q, center=False)
        return U[:, :q].contiguous()

    def _from_gram_eigh(x64: torch.Tensor) -> torch.Tensor:
        """Top-q left singular vectors; use smaller Gram when d != n."""
        d0, n0 = x64.shape
        if d0 <= n0:
            g = x64 @ x64.t()
            g = (g + g.t()) * 0.5
            tr = torch.trace(g).clamp_min(0.0)
            jitter = float(tr.item()) * 1e-10 / max(d0, 1) + 1e-12
            g = g + jitter * torch.eye(d0, dtype=g.dtype, device=g.device)
            w, vecs = torch.linalg.eigh(g)
            return vecs[:, -q:].contiguous()
        # tall matrix: eigh on C^T C (smaller) then U = C V / sigma
        g = x64.t() @ x64
        g = (g + g.t()) * 0.5
        tr = torch.trace(g).clamp_min(0.0)
        jitter = float(tr.item()) * 1e-10 / max(n0, 1) + 1e-12
        g = g + jitter * torch.eye(n0, dtype=g.dtype, device=g.device)
        w, vecs = torch.linalg.eigh(g)
        qq = min(q, vecs.shape[1])
        V = vecs[:, -qq:]
        s = w[-qq:].clamp_min(0.0).sqrt().clamp_min(1e-15)
        return ((x64 @ V) / s.unsqueeze(0)).contiguous()

    def _from_svd_full(x64: torch.Tensor) -> torch.Tensor:
        kwargs: dict = {"full_matrices": False}
        try:
            kwargs["driver"] = "gesvd"
            U, _, _ = torch.linalg.svd(x64, **kwargs)
        except TypeError:
            kwargs.pop("driver", None)
            U, _, _ = torch.linalg.svd(x64, **kwargs)
        return U[:, :q].contiguous()

    # 1) Fast path (often fails on ill-conditioned / repeated spectrum)
    try:
        return _from_pca_lowrank(cat_cpu)
    except RuntimeError:
        pass
    
    try:
        return _from_pca_lowrank(cat_cpu.double())
    except RuntimeError:
        pass

    # 2) PSD Gram on the smaller side when d <= n (typical: d=7168, n=m*r)
    x64 = cat_cpu.double().contiguous()
    try:
        U = _from_gram_eigh(x64)
        return U.float().contiguous()
    except RuntimeError:
        pass

    # 3) Full SVD in fp64 (slower but more robust than default pca_lowrank path)
    try:
        U = _from_svd_full(x64)
        return U.float().contiguous()
    except RuntimeError:
        pass

    # 4) numpy fallback (same as orthogonal_procrustes_align style)
    u_np, _, _ = np.linalg.svd(x64.numpy(), full_matrices=False)
    U = torch.from_numpy(np.ascontiguousarray(u_np[:, :q])).float().contiguous()
    return U


def orthogonal_procrustes_align(
    u_shared: torch.Tensor,  # [d, r]
    u_orig: torch.Tensor,    # [d, r]
) -> Tuple[torch.Tensor, float]:
    """Kabsch / orthogonal Procrustes: R = argmin_R ||U_o - U_s R||_F s.t. R orthogonal.

    SVD of M = U_s^T U_o can fail to converge in fp32 or with default Lapack driver on
    ill-conditioned / near-degenerate M. We accumulate M in fp64 on CPU and prefer the
    more robust ``gesvd`` driver, with optional diagonal jitter fallback.
    """
    if u_shared.shape != u_orig.shape:
        raise ValueError(f"shape mismatch: {u_shared.shape} vs {u_orig.shape}")
    orig_dtype = u_shared.dtype
    dev = u_shared.device
    us = u_shared.to(dtype=torch.float64).cpu().contiguous()
    uo = u_orig.to(dtype=torch.float64).cpu().contiguous()
    m = us.transpose(0, 1) @ uo  # [r, r]

    def _robust_svd(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs: dict = {"full_matrices": False}
        try:
            kwargs["driver"] = "gesvd"
            return torch.linalg.svd(x, **kwargs)
        except TypeError:
            kwargs.pop("driver", None)
            return torch.linalg.svd(x, **kwargs)

    try:
        a, _, b = _robust_svd(m)
    except RuntimeError:
        rn = m.shape[0]
        eye = torch.eye(rn, dtype=m.dtype, device=m.device)
        sigma = float(torch.linalg.matrix_norm(m, ord=2).item())
        jitter = max(sigma * 1e-12, 1e-10)
        try:
            a, _, b = _robust_svd(m + jitter * eye)
        except RuntimeError:
            u, _, vh = np.linalg.svd(m.numpy(), full_matrices=False)
            a = torch.from_numpy(u.copy())
            b = torch.from_numpy(vh.copy())

    rmat = (a @ b).to(device=dev, dtype=orig_dtype)
    u_new = u_shared @ rmat
    err = float((u_orig - u_new).pow(2).mean().sqrt().item())
    return u_new, err


def _iter_local_c4_texts(calib_data_dir: str) -> Iterable[str]:
    p = Path(calib_data_dir)
    if not p.exists():
        raise FileNotFoundError(f"calib_data_dir not found: {p}")
    files = sorted(p.glob("c4-train.*.json.gz"))
    if not files:
        raise RuntimeError(f"no c4-train.*.json.gz found under {p}")
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as g:
            for line in g:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    txt = obj.get("text", "")
                    if isinstance(txt, str) and txt.strip():
                        yield txt.strip()
                except Exception:
                    continue


def _build_calib_input_ids(
    tokenizer,
    calib_data_dir: str,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> List[torch.Tensor]:
    rng = random.Random(seed)
    texts = _iter_local_c4_texts(calib_data_dir)
    out: List[torch.Tensor] = []
    for txt in texts:
        ids = tokenizer(txt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.numel() < seqlen + 1:
            continue
        for _ in range(2):
            if len(out) >= nsamples:
                break
            st = rng.randint(0, int(ids.numel() - seqlen - 1))
            chunk = ids[st : st + seqlen].clone()
            out.append(chunk.unsqueeze(0))
        if len(out) >= nsamples:
            break
    if len(out) < nsamples:
        raise RuntimeError(f"only built {len(out)} calibration samples, need {nsamples}")
    return out


@dataclass
class FisherStats:
    # per layer hidden second-moment diag mean over tokens
    diag_second_moment: Dict[int, torch.Tensor]  # [d]
    token_counts: Dict[int, int]


def _model_input_device(model: nn.Module) -> torch.device:
    """Device for input_ids / attention_mask (embeddings); works with device_map sharding."""
    emb = model.get_input_embeddings()
    return emb.weight.device


def collect_fisher_like_scores(
    model_path: str,
    basis_state: Dict[str, torch.Tensor],
    calib_data_dir: str,
    nsamples: int,
    seqlen: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Approximate Fisher-like directional sensitivity:
      I_{l,j} = E[(g_l^T u_{l,j})^2]
    with g_l approximated by layer-input activation vector x_l.
    We estimate E[x_i^2] per layer and project by (U^2)^T.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dev = device.strip().lower()
    use_device_map = dev in ("auto", "balanced", "sequential")

    if dev == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )
        input_dev = torch.device("cpu")
    elif use_device_map:
        # Requires `accelerate`. Respects CUDA_VISIBLE_DEVICES (e.g. 1,2,3 → logical cuda:0..2).
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if dev == "auto" else dev,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            raise RuntimeError(
                "device_map loading failed (install: pip install accelerate). "
                f"Original error: {e}"
            ) from e
        input_dev = _model_input_device(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model = model.to(device)
        input_dev = torch.device(device)

    model.eval()
    layers = model.model.decoder.layers
    n_layers = len(layers)
    d_model = int(model.config.hidden_size)
    sum_x2 = {i: torch.zeros(d_model, dtype=torch.float64) for i in range(n_layers)}
    cnt = {i: 0 for i in range(n_layers)}

    hooks = []

    def _mk_hook(li: int):
        def _hook(_m: nn.Module, inp, _out):
            x = inp[0].detach().float()  # [B,T,D]
            x2 = x.pow(2).sum(dim=(0, 1)).cpu().to(torch.float64)
            sum_x2[li] += x2
            cnt[li] += int(x.shape[0] * x.shape[1])
        return _hook

    for li, layer in enumerate(layers):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(_mk_hook(li)))

    batches = _build_calib_input_ids(tokenizer, calib_data_dir, nsamples, seqlen, seed)
    with torch.no_grad():
        for b in batches:
            b = b.to(input_dev, non_blocking=(input_dev.type == "cuda"))
            attn = torch.ones_like(b, device=input_dev)
            _ = model(input_ids=b, attention_mask=attn, use_cache=False)

    for h in hooks:
        h.remove()

    # Build directional scores by branch/layer:
    scores: Dict[str, Dict[int, torch.Tensor]] = {"q": {}, "k": {}, "v": {}}
    for li in range(n_layers):
        if cnt[li] <= 0:
            raise RuntimeError(f"no hook stats captured for layer {li}")
        var_diag = (sum_x2[li] / float(cnt[li])).float()  # [d]
        for br in ("q", "k", "v"):
            k = _basis_key(li, br)
            if k not in basis_state:
                raise KeyError(f"missing basis key: {k}")
            u = basis_state[k].float()  # [d, r]
            imp = (u.pow(2).T @ var_diag).clamp_min(0.0)  # [r]
            scores[br][li] = imp
    return scores


def _alloc_rank_from_importance(
    imp: torch.Tensor,
    r_max: int,
    keep_ratio: float,
    r_min: int,
) -> int:
    r_max = max(1, min(int(r_max), int(imp.numel())))
    r_min = max(1, min(int(r_min), r_max))
    v = imp[:r_max].float().clamp_min(0.0)
    s = v.sum()
    if s <= 0:
        return r_min
    c = torch.cumsum(v, dim=0) / s
    idx = int((c >= float(keep_ratio)).nonzero(as_tuple=False)[0].item()) + 1
    return max(r_min, min(r_max, idx))


def _lcm_positive(a: int, b: int) -> int:
    """LCM for positive integers (no math.lcm dependency on Python 3.9+)."""
    return a // math.gcd(a, b) * b


def _rank_snap_step(num_heads: int, rank_multiple_of: int) -> int:
    """
    Effective step = lcm(num_heads, rank_multiple_of) when each factor is enabled.
    Use num_heads<=1 or rank_multiple_of<=1 to omit that factor (treated as 1 in lcm).
    Returns 0 to disable snapping entirely.
    """
    if num_heads <= 1 and rank_multiple_of <= 1:
        return 0
    a = num_heads if num_heads > 1 else 1
    b = rank_multiple_of if rank_multiple_of > 1 else 1
    return _lcm_positive(a, b)


def _snap_rank_to_step(r: int, step: int, r_cap: int) -> int:
    """
    Round rank down to a multiple of `step`, not exceeding min(r, r_cap).
    If the floored value is 0 but a positive rank was requested and r_cap allows it,
    use `step` as the smallest positive multiple (e.g. EigenAttn needs rank % num_heads == 0).
    """
    r = min(int(r), int(r_cap))
    if step <= 1:
        return max(1, r)
    snapped = (r // step) * step
    if snapped == 0 and r > 0 and r_cap >= step:
        snapped = step
    return snapped


def _select_column_order(
    u_orig: torch.Tensor,
    r: int,
    mode: str,
    fisher_imp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Return selected columns [d, r] from u_orig.
    - none: keep original order, take first r.
    - fisher: sort columns by fisher importance descending, then take top r.
    """
    r = max(1, min(int(r), int(u_orig.shape[1])))
    mode = (mode or "none").lower()
    if mode == "none":
        return u_orig[:, :r].contiguous()
    if mode == "fisher":
        if fisher_imp is None:
            raise ValueError("basis_select_mode=fisher requires fisher importance scores")
        imp = fisher_imp[: u_orig.shape[1]].float()
        if imp.numel() < u_orig.shape[1]:
            pad = torch.zeros(u_orig.shape[1] - imp.numel(), dtype=imp.dtype)
            imp = torch.cat([imp, pad], dim=0)
        idx = torch.argsort(imp, descending=True)[:r]
        return u_orig[:, idx].contiguous()
    raise ValueError(f"unknown basis_select_mode: {mode}")


def allocate_ranks(
    basis_state: Dict[str, torch.Tensor],
    layers: List[int],
    max_rank: int,
    mode: str,
    fisher_scores: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
    snap_step: int = 0,
) -> Dict[str, Dict[int, int]]:
    ranks: Dict[str, Dict[int, int]] = {"q": {}, "k": {}, "v": {}}
    for li in layers:
        rq = min(max_rank, basis_state[_basis_key(li, "q")].shape[1])
        rk = min(max_rank, basis_state[_basis_key(li, "k")].shape[1])
        rv = min(max_rank, basis_state[_basis_key(li, "v")].shape[1])
        if mode == "uniform":
            ranks["q"][li], ranks["k"][li], ranks["v"][li] = rq, rk, rv
            continue
        if mode in ("energy", "fisher"):
            if fisher_scores is None:
                ranks["q"][li], ranks["k"][li], ranks["v"][li] = rq, rk, rv
                continue
            # Default policy: Q conservative (keep full), K more conservative rank, V relatively flexible.
            ranks["q"][li] = rq
            ranks["k"][li] = _alloc_rank_from_importance(
                fisher_scores["k"][li], rk, keep_ratio=0.97, r_min=max(1, int(0.45 * rk))
            )
            ranks["v"][li] = _alloc_rank_from_importance(
                fisher_scores["v"][li], rv, keep_ratio=0.99, r_min=max(1, int(0.60 * rv))
            )
            continue
        raise ValueError(f"unknown rank mode: {mode}")

    if snap_step > 1:
        for li in layers:
            rq = min(max_rank, basis_state[_basis_key(li, "q")].shape[1])
            rk = min(max_rank, basis_state[_basis_key(li, "k")].shape[1])
            rv = min(max_rank, basis_state[_basis_key(li, "v")].shape[1])
            ranks["q"][li] = _snap_rank_to_step(ranks["q"][li], snap_step, rq)
            ranks["k"][li] = _snap_rank_to_step(ranks["k"][li], snap_step, rk)
            ranks["v"][li] = _snap_rank_to_step(ranks["v"][li], snap_step, rv)
    return ranks


def _segment_stats_from_similarity(seg: List[int], layers: List[int], sim: torch.Tensor) -> Tuple[float, float]:
    idx = [layers.index(x) for x in seg]
    vals = []
    for i in idx:
        for j in idx:
            if i == j:
                continue
            vals.append(float(sim[i, j].item()))
    if not vals:
        return 1.0, 1.0
    return float(sum(vals) / len(vals)), float(min(vals))


def write_rebuilt_basis(
    basis_in_state: Dict[str, torch.Tensor],
    basis_out: Path,
    rebuilt: Dict[str, torch.Tensor],
    metadata: Dict[str, object],
) -> None:
    out = {}
    # Keep non-basis keys untouched if any, then overwrite basis keys.
    for k, v in basis_in_state.items():
        if isinstance(k, str) and k.endswith("_proj_basis"):
            continue
        out[k] = v
    out.update(rebuilt)
    out["__fsgr_metadata__"] = metadata
    basis_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, basis_out)


def export_analysis(analysis_dir: Path, payload: Dict[str, object]) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _to_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    return torch.float32


def run_compare_mode(basis_a: Dict[str, torch.Tensor], basis_b: Dict[str, torch.Tensor], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for br in ("q", "k", "v"):
        ka = _iter_basis_keys(basis_a, br)
        kb = _iter_basis_keys(basis_b, br)
        keys = sorted(set(ka).intersection(kb), key=lambda x: int(x.split(".layers.")[1].split(".")[0]))
        for k in keys:
            ua = basis_a[k].float()
            ub = basis_b[k].float()
            r = min(ua.shape[1], ub.shape[1])
            s = ((ua[:, :r].T @ ub[:, :r]).pow(2).sum() / max(1, r)).item()
            rel = (ua[:, :r] - ub[:, :r]).norm() / (ua[:, :r].norm() + 1e-9)
            rows.append(
                {
                    "key": k,
                    "branch": br,
                    "layer": _parse_layer_from_key(k),
                    "rank_a": int(ua.shape[1]),
                    "rank_b": int(ub.shape[1]),
                    "subspace_similarity": float(s),
                    "rel_l2": float(rel.item()),
                }
            )
    with open(out_dir / "compare_basis_stats.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser("FSGR offline basis rebuild")
    ap.add_argument("--model", required=True)
    ap.add_argument("--basis_in", required=True)
    ap.add_argument("--basis_out", required=True)
    ap.add_argument("--calib_dataset", default="c4")
    ap.add_argument("--calib_data_dir", default="")
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--max_rank", type=int, default=5712)
    ap.add_argument("--cluster_mode", choices=["contiguous", "global", "none"], default="contiguous")
    ap.add_argument("--num_segments", type=int, default=3)
    ap.add_argument("--share_targets", default="kv", help="q|k|v|qk|qv|kv|qkv")
    ap.add_argument("--rank_budget_mode", choices=["uniform", "energy", "fisher"], default="fisher")
    ap.add_argument(
        "--basis_select_mode",
        choices=["none", "fisher"],
        default="none",
        help=(
            "How to choose basis columns under fixed per-layer rank. "
            "'none': keep original order; 'fisher': reorder columns by fisher importance, "
            "then keep top-r (works with rank_budget_mode=uniform for same-rank smarter content)."
        ),
    )
    ap.add_argument("--orth_align", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--dry_run", type=int, default=0, choices=[0, 1])
    ap.add_argument("--compare_basis", default="", help="optional path to compare against basis_out-style file")
    ap.add_argument("--analysis_dir", default="")
    ap.add_argument(
        "--fisher_device",
        default="cpu",
        help=(
            "Where to run calibration forward for Fisher scores: "
            "cpu | cuda | cuda:N | auto | balanced | sequential. "
            "Use CUDA_VISIBLE_DEVICES=1,2,3 together with --fisher_device auto to shard the "
            "teacher across the visible GPUs (needs accelerate)."
        ),
    )
    ap.add_argument(
        "--num_heads",
        type=int,
        default=56,
        help=(
            "Attention heads (OPT-30B uses 56). Effective snap step is lcm(num_heads, rank_multiple_of). "
            "Use 0 to omit the head factor (only rank_multiple_of applies if >1)."
        ),
    )
    ap.add_argument(
        "--rank_multiple_of",
        type=int,
        default=12,
        help=(
            "Also require effective rank %% this == 0 (e.g. 12). Combined with --num_heads via "
            "lcm (56 and 12 -> step 168). Use 0 or 1 to disable this extra factor."
        ),
    )
    args = ap.parse_args()

    if args.calib_dataset.lower() != "c4":
        raise ValueError("this first version supports --calib_dataset c4 only")
    if args.rank_budget_mode == "fisher" and not args.calib_data_dir:
        raise ValueError("--calib_data_dir is required for fisher rank allocation")

    _set_seed(args.seed)
    basis_in = Path(args.basis_in)
    basis_out = Path(args.basis_out)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else basis_out.with_suffix("").with_name(basis_out.stem + ".analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    state = load_basis_file(str(basis_in))
    integrity = check_basis_integrity(state, analysis_dir / "basis_integrity_summary.json")

    layers = [_parse_layer_from_key(k) for k in _iter_basis_keys(state, "q")]
    layer_basis: Dict[str, Dict[int, torch.Tensor]] = {"q": {}, "k": {}, "v": {}}
    for br in ("q", "k", "v"):
        for li in layers:
            layer_basis[br][li] = state[_basis_key(li, br)].float()

    sim_mats = {br: compute_subspace_similarity(layer_basis[br]) for br in ("q", "k", "v")}
    for br in ("q", "k", "v"):
        _write_similarity_outputs(sim_mats[br], layers, analysis_dir / f"similarity_{br}")

    if args.cluster_mode == "none":
        segments = [layers]
    elif args.cluster_mode == "global":
        sim_for_cluster = 0.5 * (sim_mats["k"] + sim_mats["v"])
        segments = cluster_layers_global_greedy(layers, sim_for_cluster, args.num_segments)
    else:
        segments = cluster_layers_contiguous(layers, args.num_segments)

    fisher_scores = None
    need_fisher = (args.rank_budget_mode in ("energy", "fisher")) or (args.basis_select_mode == "fisher")
    if need_fisher:
        fisher_scores = collect_fisher_like_scores(
            model_path=args.model,
            basis_state=state,
            calib_data_dir=args.calib_data_dir,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            dtype=_to_dtype(args.dtype),
            seed=args.seed,
            device=args.fisher_device,
        )

    snap_step = _rank_snap_step(
        int(getattr(args, "num_heads", 0) or 0),
        int(getattr(args, "rank_multiple_of", 0) or 0),
    )
    ranks = allocate_ranks(
        basis_state=state,
        layers=layers,
        max_rank=args.max_rank,
        mode=args.rank_budget_mode,
        fisher_scores=fisher_scores,
        snap_step=snap_step,
    )
    print(
        f"[FSGR] rank snap: num_heads={int(args.num_heads)} rank_multiple_of={int(args.rank_multiple_of)} "
        f"-> lcm step={snap_step}"
    )

    share_set = set(args.share_targets.lower())
    share_k = "k" in share_set
    share_v = "v" in share_set
    share_q = "q" in share_set

    rebuilt: Dict[str, torch.Tensor] = {}
    layer_meta: Dict[str, dict] = {}
    seg_meta: List[dict] = []

    # Save branch-wise rank allocation
    for br in ("q", "k", "v"):
        with open(analysis_dir / f"rank_alloc_{br}.json", "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in ranks[br].items()}, f, indent=2)

    # Build shared basis per segment for requested targets
    shared_master: Dict[Tuple[int, str], torch.Tensor] = {}
    for si, seg in enumerate(segments):
        seg_entry = {"segment_id": si, "layers": seg, "stats": {}}
        for br in ("q", "k", "v"):
            avg_sim, min_sim = _segment_stats_from_similarity(seg, layers, sim_mats[br])
            seg_entry["stats"][br] = {"avg_similarity": avg_sim, "min_similarity": min_sim}
        seg_meta.append(seg_entry)

        if share_q:
            shared_master[(si, "q")] = build_shared_basis_from_projection_mean(layer_basis["q"], seg, args.max_rank)
            torch.save(shared_master[(si, "q")], analysis_dir / f"shared_q_segment{si}.pt")
        if share_k:
            shared_master[(si, "k")] = build_shared_basis_from_projection_mean(layer_basis["k"], seg, args.max_rank)
            torch.save(shared_master[(si, "k")], analysis_dir / f"shared_k_segment{si}.pt")
        if share_v:
            shared_master[(si, "v")] = build_shared_basis_from_projection_mean(layer_basis["v"], seg, args.max_rank)
            torch.save(shared_master[(si, "v")], analysis_dir / f"shared_v_segment{si}.pt")

    # Rebuild per-layer basis with optional orth alignment.
    for si, seg in enumerate(segments):
        for li in seg:
            layer_meta[str(li)] = {}
            for br in ("q", "k", "v"):
                k = _basis_key(li, br)
                u_orig = layer_basis[br][li].float()
                r = int(min(ranks[br][li], u_orig.shape[1], args.max_rank))
                fisher_imp = None if fisher_scores is None else fisher_scores[br][li]
                u_ref = _select_column_order(
                    u_orig=u_orig,
                    r=r,
                    mode=args.basis_select_mode,
                    fisher_imp=fisher_imp,
                )
                shared_on = (br == "q" and share_q) or (br == "k" and share_k) or (br == "v" and share_v)
                if not shared_on:
                    u_new = u_ref
                    ortho_err = 0.0
                else:
                    u_shared = shared_master[(si, br)][:, :r].contiguous()
                    if int(args.orth_align) == 1:
                        u_new, ortho_err = orthogonal_procrustes_align(u_shared, u_ref)
                    else:
                        u_new, ortho_err = u_shared, 0.0
                # Same dtype for matmul (e.g. u_ref fp32 vs u_new fp64 from Procrustes path)
                ur = u_ref.float()
                un = u_new.float()
                sim_old_new = float(((ur.T @ un).pow(2).sum() / max(1, r)).item())
                rebuilt[k] = u_new.to(torch.float32).contiguous()
                layer_meta[str(li)][br] = {
                    "effective_rank": int(r),
                    "shared": bool(shared_on),
                    "segment_id": int(si),
                    "basis_select_mode": str(args.basis_select_mode),
                    "orth_align_error": float(ortho_err),
                    "shared_vs_selected_subspace_similarity": float(sim_old_new),
                }

    analysis_payload = {
        "method": "fsgr",
        "share_targets": args.share_targets,
        "num_segments": int(args.num_segments),
        "cluster_mode": args.cluster_mode,
        "rank_budget_mode": args.rank_budget_mode,
        "basis_select_mode": args.basis_select_mode,
        "orth_align": int(args.orth_align),
        "num_heads": int(args.num_heads),
        "rank_multiple_of": int(args.rank_multiple_of),
        "rank_snap_step": int(snap_step),
        "layers": layers,
        "segments": seg_meta,
        "layer_meta": layer_meta,
        "integrity_summary_file": str((analysis_dir / "basis_integrity_summary.json").name),
    }
    export_analysis(analysis_dir, analysis_payload)

    if args.compare_basis:
        basis_cmp = load_basis_file(args.compare_basis)
        run_compare_mode(state, basis_cmp, analysis_dir)

    if int(args.dry_run) == 1:
        print(f"[FSGR] dry-run done. Analysis at: {analysis_dir}")
        return

    metadata = {
        "method": "fsgr",
        "share_targets": args.share_targets,
        "num_segments": int(args.num_segments),
        "cluster_mode": args.cluster_mode,
        "rank_budget_mode": args.rank_budget_mode,
        "basis_select_mode": args.basis_select_mode,
        "orth_align": int(args.orth_align),
        "num_heads": int(args.num_heads),
        "rank_multiple_of": int(args.rank_multiple_of),
        "rank_snap_step": int(snap_step),
        "effective_rank": {br: {str(li): int(ranks[br][li]) for li in layers} for br in ("q", "k", "v")},
        "layer_meta": layer_meta,
    }
    write_rebuilt_basis(state, basis_out, rebuilt, metadata)
    print(f"[FSGR] wrote rebuilt basis to: {basis_out}")
    print(f"[FSGR] analysis dir: {analysis_dir}")


if __name__ == "__main__":
    main()

