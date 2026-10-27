#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss-aware / NLL-aware basis generator for OPT models.

Compared with build_eigen_basis_opt_error_budget_offline.py:
- does NOT overwrite the original script
- keeps headwise SVD basis extraction (K/Q shared, V separate)
- builds threshold-based rank candidates per layer
- scores candidates with REAL end-to-end NLL deltas (single-layer substitution)
- performs global greedy rank allocation under a target KV budget
- optional tail-aware basis repair (A1 V / A2 KQ) before candidate scoring
- exports basis artifact (.pt) + detailed report (.json)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
except Exception:  # pragma: no cover
    AlignDevicesHook = None
    add_hook_to_module = None


def _load_py_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iter_texts_from_dir(data_dir: str, max_files: int, max_lines: int):
    import gzip
    import glob

    exts = ["*.txt", "*.jsonl", "*.json", "*.gz"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(str(Path(data_dir) / "**" / e), recursive=True))
    files = sorted(files)[: max(1, int(max_files))]
    n = 0
    for fp in files:
        opener = gzip.open if fp.endswith(".gz") else open
        try:
            with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
                if fp.endswith(".json") and (not fp.endswith(".gz")):
                    try:
                        obj = json.load(f)
                        if isinstance(obj, list):
                            for it in obj:
                                t = it.get("text") if isinstance(it, dict) else it
                                if isinstance(t, str) and t.strip():
                                    yield t.strip()
                                    n += 1
                                    if n >= max_lines:
                                        return
                        elif isinstance(obj, dict):
                            t = obj.get("text") or obj.get("content")
                            if isinstance(t, str) and t.strip():
                                yield t.strip()
                                n += 1
                                if n >= max_lines:
                                    return
                    except Exception:
                        pass
                    continue
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            j = json.loads(line)
                            if isinstance(j, dict):
                                t = j.get("text") or j.get("content")
                            else:
                                t = line
                        except Exception:
                            t = line
                    else:
                        t = line
                    if isinstance(t, str) and t.strip():
                        yield t.strip()
                        n += 1
                        if n >= max_lines:
                            return
        except Exception:
            continue


def _build_token_stream(tok, texts_iter: Iterable[str], target_tokens: int, seed: int):
    random.seed(int(seed))
    ids: List[int] = []
    total = 0
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    for t in texts_iter:
        enc = tok(t, add_special_tokens=False)["input_ids"]
        if not enc:
            continue
        if eos_id is not None:
            ids.extend(enc + [eos_id])
            total += len(enc) + 1
        else:
            ids.extend(enc)
            total += len(enc)
        if total >= int(target_tokens):
            break
    if total == 0:
        return None
    return torch.tensor(ids, dtype=torch.long)


def _basis_key(layer_idx: int, kind: str) -> str:
    return f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{kind}_proj_basis"


def _rank_from_eval(eval_per_head: torch.Tensor, threshold: float, num_heads: int) -> int:
    c = torch.cumsum(eval_per_head.float(), dim=1)
    tgt = torch.full((c.size(0), 1), float(threshold), device=c.device, dtype=c.dtype)
    per_head = torch.searchsorted(c, tgt, right=False).squeeze(1) + 1
    per_head = torch.clamp(per_head, min=1, max=c.size(1))
    return int(num_heads * torch.amax(per_head).item())


def _per_head_rank_from_eval(eval_per_head: torch.Tensor, threshold: float) -> List[int]:
    c = torch.cumsum(eval_per_head.float(), dim=1)
    tgt = torch.full((c.size(0), 1), float(threshold), device=c.device, dtype=c.dtype)
    per_head = torch.searchsorted(c, tgt, right=False).squeeze(1) + 1
    per_head = torch.clamp(per_head, min=1, max=c.size(1))
    return [int(x) for x in per_head.tolist()]


def _finalize_rank(rank: int, num_heads: int, head_dim: int, min_rank: int, strict_original: bool) -> int:
    max_r = num_heads * head_dim
    r = max(0, min(int(rank), max_r))
    k = r // num_heads if num_heads else 0
    if k < 1:
        k = 1
    if k > head_dim:
        k = head_dim
    r = num_heads * k
    if (not strict_original) and r < int(min_rank):
        k = (int(min_rank) + num_heads - 1) // num_heads
        k = max(1, min(k, head_dim))
        r = num_heads * k
    return int(min(r, max_r))


def _pack_headwise_basis(basis_headwise: torch.Tensor, rank_total: int, num_heads: int, head_dim: int) -> torch.Tensor:
    if rank_total % num_heads != 0:
        raise ValueError(f"rank_total={rank_total} not divisible by num_heads={num_heads}")
    k = int(rank_total // num_heads)
    if k <= 0 or k > head_dim:
        raise ValueError(f"invalid per-head rank {k}, head_dim={head_dim}")
    d = num_heads * head_dim
    U = torch.zeros((d, rank_total), dtype=basis_headwise.dtype, device=basis_headwise.device)
    for h in range(num_heads):
        rs = slice(h * head_dim, (h + 1) * head_dim)
        cs = slice(h * k, (h + 1) * k)
        U[rs, cs] = basis_headwise[h, :, :k]
    return U.contiguous()


def _unpack_packed_basis_to_headwise(
    packed_basis: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Inverse of _pack_headwise_basis for the kept rank.
    packed_basis: [hidden, rank_total], block-diagonal layout by head.
    returns: [num_heads, head_dim, k], where k = rank_total // num_heads.
    """
    if packed_basis.dim() != 2:
        raise ValueError(f"packed basis must be 2D, got shape={tuple(packed_basis.shape)}")
    hidden, rank_total = int(packed_basis.shape[0]), int(packed_basis.shape[1])
    if hidden != num_heads * head_dim:
        raise ValueError(
            f"packed hidden mismatch: hidden={hidden} vs num_heads*head_dim={num_heads*head_dim}"
        )
    if rank_total % num_heads != 0:
        raise ValueError(f"rank_total={rank_total} not divisible by num_heads={num_heads}")
    k = rank_total // num_heads
    out = torch.zeros((num_heads, head_dim, k), dtype=packed_basis.dtype, device=packed_basis.device)
    for h in range(num_heads):
        rs = slice(h * head_dim, (h + 1) * head_dim)
        cs = slice(h * k, (h + 1) * k)
        out[h] = packed_basis[rs, cs]
    return out


def _energy_stats(eval_per_head: torch.Tensor, per_head_ranks: List[int]) -> Dict[str, float]:
    e = eval_per_head.float().cpu()
    c = torch.cumsum(e, dim=1)
    vals = []
    hd = c.size(1)
    for h, r in enumerate(per_head_ranks):
        rr = int(max(1, min(r, hd)))
        vals.append(float(c[h, rr - 1].item()))
    t = torch.tensor(vals, dtype=torch.float32)
    return {"mean": float(t.mean().item()), "min": float(t.min().item()), "max": float(t.max().item())}


def _lm_nll(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> float:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = out.logits[:, :-1, :].float().contiguous()
    labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")
    return float(loss.item())


def _attach_like_accelerate_hook(src_module: torch.nn.Module, dst_module: torch.nn.Module) -> bool:
    """
    For sharded models (device_map=auto/balanced), cloned replacement layers must carry
    an equivalent accelerate AlignDevicesHook, otherwise cross-device transfer logic is lost.
    """
    if add_hook_to_module is None or AlignDevicesHook is None:
        return False
    src_hook = getattr(src_module, "_hf_hook", None)
    if src_hook is None or src_hook.__class__.__name__ != "AlignDevicesHook":
        return False
    try:
        new_hook = AlignDevicesHook(
            execution_device=getattr(src_hook, "execution_device", None),
            offload=getattr(src_hook, "offload", False),
            io_same_device=getattr(src_hook, "io_same_device", False),
            weights_map=getattr(src_hook, "weights_map", None),
            offload_buffers=getattr(src_hook, "offload_buffers", False),
            place_submodules=getattr(src_hook, "place_submodules", False),
            skip_keys=getattr(src_hook, "skip_keys", None),
            tied_params_map=getattr(src_hook, "tied_params_map", None),
        )
        add_hook_to_module(dst_module, new_hook)
        return True
    except Exception:
        return False


@dataclass
class Candidate:
    layer: int
    threshold: float
    rank_kq: int
    rank_v: int
    per_head_kq: List[int]
    per_head_v: List[int]
    energy_kq: Dict[str, float]
    energy_v: Dict[str, float]
    layer_err: float
    nll_delta: float = 0.0

    @property
    def cost(self) -> int:
        return int(self.rank_kq + self.rank_v)


def _combo_nll(
    model,
    layers,
    nll_batches: List[torch.Tensor],
    input_dev: torch.device,
    layer_alt_modules: List[List[torch.nn.Module | None]],
    combo_idx: List[int],
) -> float:
    replaced: List[Tuple[int, torch.nn.Module]] = []
    try:
        for li, ci in enumerate(combo_idx):
            if ci <= 0:
                continue
            qlayer = layer_alt_modules[li][ci]
            if qlayer is None:
                continue
            replaced.append((li, layers[li]))
            layers[li] = qlayer

        vals = []
        for b in nll_batches:
            bdev = b.to(input_dev)
            vals.append(_lm_nll(model, bdev, None))
        return float(sum(vals) / max(1, len(vals)))
    finally:
        for li, ori in replaced:
            layers[li] = ori


def _allocate_greedy(layer_candidates: List[List[Candidate]], n_layers: int, target_total: int, verbose: int):
    cur_idx = [0 for _ in range(n_layers)]
    cur_total = sum(layer_candidates[i][cur_idx[i]].cost for i in range(n_layers))
    alloc_trace = []
    step = 0
    while cur_total > target_total:
        best = None
        for li in range(n_layers):
            i = cur_idx[li]
            if i + 1 >= len(layer_candidates[li]):
                continue
            cur = layer_candidates[li][i]
            nxt = layer_candidates[li][i + 1]
            release = cur.cost - nxt.cost
            if release <= 0:
                continue
            penalty = max(0.0, nxt.nll_delta - cur.nll_delta)
            score = penalty / release
            cand = (score, penalty, release, li, i + 1)
            if best is None or cand < best:
                best = cand
        if best is None:
            break
        _, penalty, release, li, new_i = best
        old_i = cur_idx[li]
        cur_idx[li] = new_i
        cur_total -= release
        alloc_trace.append(
            {
                "step": step,
                "layer": li,
                "from_candidate": old_i,
                "to_candidate": new_i,
                "released_cost": int(release),
                "nll_penalty": float(penalty),
                "total_cost_after": int(cur_total),
            }
        )
        if int(verbose) == 1:
            print(
                f"[alloc-greedy] step={step} layer={li:02d} {old_i}->{new_i} release={release} "
                f"penalty={penalty:.6f} total={cur_total}",
                flush=True,
            )
        step += 1
    return cur_idx, cur_total, alloc_trace


def _allocate_beam_search(
    model,
    layers,
    layer_candidates: List[List[Candidate]],
    layer_alt_modules: List[List[torch.nn.Module | None]],
    nll_batches: List[torch.Tensor],
    input_dev: torch.device,
    target_total: int,
    beam_width: int,
    verbose: int,
):
    n_layers = len(layer_candidates)
    min_cost = [min(c.cost for c in cands) for cands in layer_candidates]
    suffix_min = [0] * (n_layers + 1)
    for i in range(n_layers - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + min_cost[i]

    # each state: (combo_idx, used_cost, nll)
    beam: List[Tuple[List[int], int, float]] = [([], 0, 0.0)]
    for li in range(n_layers):
        expanded: List[Tuple[List[int], int, float]] = []
        for prefix_idx, prefix_cost, _ in beam:
            for ci, cand in enumerate(layer_candidates[li]):
                new_cost = prefix_cost + cand.cost
                # prune states that can never satisfy budget
                if new_cost + suffix_min[li + 1] > target_total:
                    continue
                combo = prefix_idx + [ci] + [0] * (n_layers - li - 1)
                nll = _combo_nll(model, layers, nll_batches, input_dev, layer_alt_modules, combo)
                expanded.append((prefix_idx + [ci], new_cost, nll))

        if not expanded:
            # fallback: keep one feasible-by-min-cost route (may exceed target eventually)
            for prefix_idx, prefix_cost, _ in beam:
                ci = len(layer_candidates[li]) - 1
                cand = layer_candidates[li][ci]
                combo = prefix_idx + [ci] + [0] * (n_layers - li - 1)
                nll = _combo_nll(model, layers, nll_batches, input_dev, layer_alt_modules, combo)
                expanded.append((prefix_idx + [ci], prefix_cost + cand.cost, nll))

        expanded.sort(key=lambda x: (x[2], x[1]))  # lower nll then lower cost
        beam = expanded[: max(1, int(beam_width))]
        if int(verbose) == 1:
            print(
                f"[alloc-beam] layer={li:02d} explored={len(expanded)} keep={len(beam)} "
                f"best_nll={beam[0][2]:.6f} best_cost={beam[0][1]}",
                flush=True,
            )

    feasible = [st for st in beam if st[1] <= target_total]
    if feasible:
        feasible.sort(key=lambda x: (x[2], x[1]))
        best = feasible[0]
    else:
        beam.sort(key=lambda x: (x[1], x[2]))
        best = beam[0]

    alloc_trace = [
        {
            "method": "beam",
            "beam_width": int(beam_width),
            "best_nll": float(best[2]),
            "best_cost": int(best[1]),
            "target_total": int(target_total),
            "feasible_found": bool(len(feasible) > 0),
        }
    ]
    return best[0], best[1], alloc_trace, float(best[2])


def _downsample_candidates(cands: List[Candidate], keep_n: int) -> List[Candidate]:
    if len(cands) <= keep_n:
        return cands
    # cands sorted by descending threshold; keep endpoints and roughly uniform interior
    idx = sorted(set([0, len(cands) - 1] + [int(round(i * (len(cands) - 1) / (keep_n - 1))) for i in range(keep_n)]))
    return [cands[i] for i in idx]


@torch.no_grad()
def _precompute_calib_token_nll_weights(
    model,
    calib_batches: List[torch.Tensor],
    input_dev: torch.device,
) -> torch.Tensor:
    """
    对每个校准序列的每个位置给出与 LM loss 一致的逐 token CE（长度 seqlen，
    末位用倒数第二位复制填充），形状 (nsamples, seqlen)，CPU float32。
    """
    ns = len(calib_batches)
    if ns == 0:
        raise ValueError("empty calib_batches")
    L = int(calib_batches[0].shape[1])
    W = torch.zeros((ns, L), dtype=torch.float32, device="cpu")
    for j, b in enumerate(calib_batches):
        ids = b.to(input_dev)
        out = model(input_ids=ids, use_cache=False, return_dict=True)
        logits = out.logits[:, :-1, :].float().contiguous()
        labels = ids[:, 1:].contiguous()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )
        ce_1d = ce.detach().float().cpu()
        Lm1 = int(ce_1d.numel())
        if Lm1 != L - 1:
            raise RuntimeError(f"unexpected CE len {Lm1} vs seqlen-1 {L - 1}")
        W[j, : L - 1] = ce_1d
        W[j, L - 1] = W[j, L - 2]
        del out, logits, ce, ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return W


@torch.no_grad()
def _match_calib_weights_to_avg_features(
    calib_token_weights: torch.Tensor,
    nsamples: int,
    seqlen: int,
    avg_dim: int,
    expect_Ntok: int,
) -> torch.Tensor | None:
    """
    `get_kqv_opt` averages every `avg_dim` calibration samples into one feature block,
    so per-head features have Ntok = (nsamples // avg_dim) * seqlen.
    The token-level NLL weights are (nsamples, seqlen); we must average them along the
    same grouping axis to align, otherwise `_inject_tail_last_kept_col_safe` silently
    falls back to un-weighted covariance (this was the real reason the previous
    nll_cov branch never actually took effect).
    """
    if calib_token_weights is None:
        return None
    ns = int(calib_token_weights.shape[0])
    L = int(calib_token_weights.shape[1])
    if ns != int(nsamples) or L != int(seqlen):
        return None
    avg = int(max(1, avg_dim))
    g = ns // avg
    if g * avg != ns or g * L != int(expect_Ntok):
        return None
    w = calib_token_weights.reshape(g, avg, L).float().mean(dim=1).reshape(-1)
    return w


@torch.no_grad()
def _solve_tail_directions(
    X: torch.Tensor,
    U_prefix: torch.Tensor,
    w_token: torch.Tensor | None,
    tail_mode: str,
    need: int,
) -> List[torch.Tensor]:
    """Top-`need` orthonormal directions in span(U_prefix)^perp, ranked by (weighted) residual energy."""
    N, d = int(X.shape[0]), int(X.shape[1])
    if N == 0 or d == 0 or int(need) <= 0:
        return []
    Xf = X.float()
    P = U_prefix.float()
    if P.shape[1] > 0:
        R = Xf - (Xf @ P) @ P.T
    else:
        R = Xf
    mode = (tail_mode or "nll_cov").strip().lower()
    if mode == "nll_cov" and w_token is not None and int(w_token.numel()) == N:
        w = w_token.float().reshape(-1).clamp(min=0.0)
        s = float(w.sum().item())
        if s > 1e-12:
            w = w * (float(N) / s)
            rw = torch.sqrt(w).view(N, 1) * R
            C = rw.T @ rw
        else:
            C = R.T @ R
    elif mode == "mean":
        m = R.mean(dim=0, keepdim=True)
        Rc = R - m
        C = Rc.T @ Rc
    else:
        C = R.T @ R
    C = 0.5 * (C + C.T)
    try:
        evals, evecs = torch.linalg.eigh(C)
    except Exception:
        return []
    order = torch.argsort(evals, descending=True)
    evecs = evecs[:, order]
    dirs: List[torch.Tensor] = []
    basis_accum = P.clone() if P.shape[1] > 0 else P.new_zeros((d, 0))
    for i in range(min(int(need), evecs.shape[1])):
        v = evecs[:, i].clone()
        if basis_accum.shape[1] > 0:
            v = v - basis_accum @ (basis_accum.T @ v)
        nv = float(v.norm().item())
        if nv < 1e-8:
            continue
        v = v / nv
        dirs.append(v)
        basis_accum = torch.cat([basis_accum, v.view(d, 1)], dim=1)
        if len(dirs) >= int(need):
            break
    return dirs


@torch.no_grad()
def _inject_tail_last_kept_col_safe(
    U_kept: torch.Tensor,
    feat_head: torch.Tensor,
    alpha: float,
    tail_mode: str,
    w_token: torch.Tensor | None,
    companion_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Safe tail repair that *only* touches U_kept[:, k-1].

    - U_kept: (d, k) orthonormal (k = per-head kept rank actually read by OPTEigenAttention)
    - Returns (U_kept_new, U_comp) with:
        * U_kept_new[:, : k-1] == U_kept[:, : k-1] exactly (no destructive QR),
        * U_kept_new[:, k-1] = normalize((1-alpha)*old_last + alpha*v_top), deflated against the prefix,
        * U_comp: (d, T) orthonormal vectors in span(U_kept_new)^perp.

    This addresses the three main failure modes of the previous tail patch:
      1) global QR perturbed the main prefix subspace preserved by SVD;
      2) inject_col might have fallen outside the kept range (no forward effect);
      3) nll_cov direction was computed but never aligned to avg-pooled features.
    """
    d, k = int(U_kept.shape[0]), int(U_kept.shape[1])
    T = int(max(0, companion_rank))
    U_comp = torch.zeros((d, T), dtype=U_kept.dtype)
    if k <= 0 or feat_head.numel() == 0:
        return U_kept.clone(), U_comp

    a = float(max(0.0, min(1.0, alpha)))
    Uf = U_kept.float()

    if k == 1:
        prefix = Uf.new_zeros((d, 0))
    else:
        prefix = Uf[:, : k - 1].contiguous()
    old_last = Uf[:, k - 1]

    need = 1 + T
    dirs = _solve_tail_directions(feat_head, prefix, w_token, tail_mode, need=need)
    if not dirs:
        return U_kept.clone(), U_comp

    v_top = dirs[0]

    if a <= 0.0:
        n0 = max(float(old_last.norm().item()), 1e-12)
        new_last = old_last / n0
    else:
        new_last = (1.0 - a) * old_last + a * v_top
        if prefix.shape[1] > 0:
            new_last = new_last - prefix @ (prefix.T @ new_last)
        nrm = float(new_last.norm().item())
        if nrm < 1e-8:
            new_last = v_top
        else:
            new_last = new_last / nrm

    if prefix.shape[1] > 0:
        basis_accum = torch.cat([prefix, new_last.view(d, 1)], dim=1)
    else:
        basis_accum = new_last.view(d, 1)
    comp_list: List[torch.Tensor] = []
    for v in dirs[1:]:
        v2 = v - basis_accum @ (basis_accum.T @ v)
        nv = float(v2.norm().item())
        if nv < 1e-8:
            continue
        v2 = v2 / nv
        comp_list.append(v2)
        basis_accum = torch.cat([basis_accum, v2.view(d, 1)], dim=1)
        if len(comp_list) >= T:
            break

    U_kept_new = U_kept.clone()
    U_kept_new[:, k - 1] = new_last.to(dtype=U_kept.dtype)
    for i, v in enumerate(comp_list):
        U_comp[:, i] = v.to(dtype=U_kept.dtype)
    return U_kept_new, U_comp


@torch.no_grad()
def _inject_tail_last_kept_cols_multi_safe(
    U_kept: torch.Tensor,
    feat_head: torch.Tensor,
    alpha: float,
    tail_mode: str,
    w_token: torch.Tensor | None,
    tail_k: int,
    companion_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Path-2 的核心实现：安全的多列 V-Tail 替换。

    - U_kept: (d, k) 列正交，k = OPTEigenAttention 实际读取的 per-head kept rank
    - tail_k: 要替换的尾部列数（1 <= tail_k <= k）；tail_k=1 时行为严格等价
      `_inject_tail_last_kept_col_safe`（调用方也可直接退化到单列路径以保持二进制一致）
    - 返回 (U_kept_new, U_comp)：
        * U_kept_new[:, : k - tail_k] == U_kept[:, : k - tail_k]（前缀严格不动）
        * U_kept_new[:, k - tail_k + i] = normalize( (1-α)·old_col_i + α·v_tail_i )
          并对 "前缀 ∪ 已固定新列" 做 Gram–Schmidt deflate（逐列）
        * U_comp: (d, T) 位于 span(U_kept_new)^perp 的补充 tail 方向
    - 若某列 deflate 后 norm 过小，退回原 old_col_i（这一列不修，其它列继续）
    - 论文意义：在固定 rank / 固定 KV 预算下，把最后 tail_k 列从 "能量主方向" 重新分配到
      "NLL 加权协方差主方向"，其他列保持不动 → 保 main subspace，同时让 V 子空间
      多覆盖 task-relevant tail directions。
    """
    d, k = int(U_kept.shape[0]), int(U_kept.shape[1])
    T = int(max(0, companion_rank))
    U_comp = torch.zeros((d, T), dtype=U_kept.dtype)
    if k <= 0 or feat_head.numel() == 0:
        return U_kept.clone(), U_comp

    tk = int(max(1, min(tail_k, k)))
    if tk == 1:
        # 严格等价单列安全路径（避免重复实现 + 100% 向后兼容）
        return _inject_tail_last_kept_col_safe(U_kept, feat_head, alpha, tail_mode, w_token, T)

    a = float(max(0.0, min(1.0, alpha)))
    Uf = U_kept.float()
    prefix = Uf[:, : k - tk].contiguous() if k > tk else Uf.new_zeros((d, 0))
    old_tail_cols = Uf[:, k - tk : k].contiguous()  # (d, tk)

    # 一次性解出 tk + T 个"前缀外"方向（按 NLL-加权残差能量排序），再分给 kept-tail 和 companion
    need = tk + T
    dirs = _solve_tail_directions(feat_head, prefix, w_token, tail_mode, need=need)
    if not dirs:
        return U_kept.clone(), U_comp

    # 逐列处理：每固定一列就把它加入 basis_accum，用于下一列的 deflate
    basis_accum = prefix.clone() if prefix.shape[1] > 0 else prefix.new_zeros((d, 0))
    new_tail_cols: List[torch.Tensor] = []
    used_dirs_idx = 0

    for i in range(tk):
        old_col = old_tail_cols[:, i]
        # 取下一个可用 tail direction（跳过退化情况）
        v_tail: torch.Tensor | None = None
        while used_dirs_idx < len(dirs):
            cand = dirs[used_dirs_idx]
            used_dirs_idx += 1
            # 在当前 basis_accum 下重新 deflate 一次（理论上 solve 已经正交，但稳妥起见）
            if basis_accum.shape[1] > 0:
                cand = cand - basis_accum @ (basis_accum.T @ cand)
            cn = float(cand.norm().item())
            if cn >= 1e-8:
                v_tail = cand / cn
                break

        if v_tail is None:
            # 没有可用 tail direction → 这一列保持原样（仍对前缀 deflate 以维持正交）
            col = old_col.clone()
            if basis_accum.shape[1] > 0:
                col = col - basis_accum @ (basis_accum.T @ col)
            cn = float(col.norm().item())
            if cn < 1e-8:
                # 极端退化：放一个单位阵的某个空方向占位
                col = torch.zeros_like(old_col)
                if basis_accum.shape[1] < d:
                    col[basis_accum.shape[1]] = 1.0
                else:
                    col = old_col / max(float(old_col.norm().item()), 1e-12)
            else:
                col = col / cn
            new_tail_cols.append(col)
            basis_accum = torch.cat([basis_accum, col.view(d, 1)], dim=1)
            continue

        # 软混合 (1-α)·old_col + α·v_tail, 再对 basis_accum 做 deflate
        if a <= 0.0:
            cand_col = old_col.clone()
        else:
            cand_col = (1.0 - a) * old_col + a * v_tail
        if basis_accum.shape[1] > 0:
            cand_col = cand_col - basis_accum @ (basis_accum.T @ cand_col)
        cn = float(cand_col.norm().item())
        if cn < 1e-8:
            # 混合后共线 / 退化 → 退回纯 tail direction（也已正交于 basis_accum）
            cand_col = v_tail
        else:
            cand_col = cand_col / cn

        new_tail_cols.append(cand_col)
        basis_accum = torch.cat([basis_accum, cand_col.view(d, 1)], dim=1)

    # 剩下的 dirs 作为 companion (位于 span(U_kept_new)^perp)
    comp_list: List[torch.Tensor] = []
    while used_dirs_idx < len(dirs) and len(comp_list) < T:
        v = dirs[used_dirs_idx]
        used_dirs_idx += 1
        if basis_accum.shape[1] > 0:
            v = v - basis_accum @ (basis_accum.T @ v)
        nv = float(v.norm().item())
        if nv < 1e-8:
            continue
        v = v / nv
        comp_list.append(v)
        basis_accum = torch.cat([basis_accum, v.view(d, 1)], dim=1)

    # 写回
    U_kept_new = U_kept.clone()
    for i, col in enumerate(new_tail_cols):
        U_kept_new[:, k - tk + i] = col.to(dtype=U_kept.dtype)
    for i, v in enumerate(comp_list):
        U_comp[:, i] = v.to(dtype=U_kept.dtype)
    return U_kept_new, U_comp


@torch.no_grad()
def _repair_basis_v_tail_safe(
    basis_v_h: torch.Tensor,
    feat_v: torch.Tensor,
    num_heads: int,
    head_dim: int,
    alpha: float,
    tail_mode: str,
    calib_token_weights: torch.Tensor | None,
    companion_rank: int,
    *,
    nsamples: int,
    seqlen: int,
    avg_dim: int,
    tail_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    tail_k: 尾部替换列数。
      - tail_k=1 → 严格等价 `_inject_tail_last_kept_col_safe` 原行为（Path 1）；
      - tail_k>1 → 调用 `_inject_tail_last_kept_cols_multi_safe`（Path 2，多列替换）。
    """
    H, d, k = basis_v_h.shape
    if H != num_heads or d != head_dim:
        raise ValueError("basis_v_h shape mismatch")
    fv = feat_v.reshape(-1, num_heads, head_dim).float()
    Ntok = int(fv.shape[0])
    w_flat = None
    if (tail_mode or "").lower() == "nll_cov":
        w_flat = _match_calib_weights_to_avg_features(
            calib_token_weights, nsamples, seqlen, avg_dim, Ntok
        )
    T = int(max(0, companion_rank))
    tk = int(max(1, min(tail_k, k)))
    new_h = basis_v_h.clone()
    comp_h = torch.zeros((H, d, T), dtype=basis_v_h.dtype)
    for h in range(num_heads):
        if tk == 1:
            U_new, U_comp = _inject_tail_last_kept_col_safe(
                basis_v_h[h], fv[:, h, :], alpha, tail_mode, w_flat, T
            )
        else:
            U_new, U_comp = _inject_tail_last_kept_cols_multi_safe(
                basis_v_h[h], fv[:, h, :], alpha, tail_mode, w_flat, tk, T
            )
        new_h[h] = U_new
        if T > 0:
            comp_h[h] = U_comp
    return new_h, comp_h


@torch.no_grad()
def _repair_basis_kq_tail_safe(
    basis_kq_h: torch.Tensor,
    feat_k: torch.Tensor,
    feat_q: torch.Tensor,
    num_heads: int,
    head_dim: int,
    alpha: float,
    tail_mode: str,
    calib_token_weights: torch.Tensor | None,
    companion_rank: int,
    *,
    nsamples: int,
    seqlen: int,
    avg_dim: int,
    tail_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """tail_k: 同 `_repair_basis_v_tail_safe`。"""
    H, d, k = basis_kq_h.shape
    fk = feat_k.reshape(-1, num_heads, head_dim).float()
    fq = feat_q.reshape(-1, num_heads, head_dim).float()
    Ntok = int(fk.shape[0])
    w_kq = None
    if (tail_mode or "").lower() == "nll_cov":
        w0 = _match_calib_weights_to_avg_features(
            calib_token_weights, nsamples, seqlen, avg_dim, Ntok
        )
        if w0 is not None:
            w_kq = torch.cat([w0, w0], dim=0)
    T = int(max(0, companion_rank))
    tk = int(max(1, min(tail_k, k)))
    new_h = basis_kq_h.clone()
    comp_h = torch.zeros((H, d, T), dtype=basis_kq_h.dtype)
    for h in range(num_heads):
        Xh = torch.cat([fk[:, h, :], fq[:, h, :]], dim=0)
        if tk == 1:
            U_new, U_comp = _inject_tail_last_kept_col_safe(
                basis_kq_h[h], Xh, alpha, tail_mode, w_kq, T
            )
        else:
            U_new, U_comp = _inject_tail_last_kept_cols_multi_safe(
                basis_kq_h[h], Xh, alpha, tail_mode, w_kq, tk, T
            )
        new_h[h] = U_new
        if T > 0:
            comp_h[h] = U_comp
    return new_h, comp_h


@torch.no_grad()
def _inject_tail_one_column(
    U: torch.Tensor,
    feat_head: torch.Tensor,
    prefix_rank: int,
    inject_col: int,
    tail_mode: str,
    w_token: torch.Tensor | None,
) -> torch.Tensor:
    """
    U: (head_dim, D) 列正交；feat_head: (N, head_dim) 校准特征（每行一个 token 位置）。

    tail_mode == "mean": 用均值在主前缀外的残差方向（旧行为）。
    tail_mode == "nll_cov": r_t = x_t - U_p U_p^T x_t，C_tail = sum_t w_t r_t r_t^T，
    取 C_tail 最大特征值对应特征向量，经 QR 重正交化后替换 inject_col 列。

    OPTEigenAttention 仍只读 U[:, :k]，故 rank / 导出格式不变。
    """
    d, D = U.shape
    if D <= 1 or feat_head.numel() == 0:
        return U
    pr = int(max(1, min(prefix_rank, D - 1)))
    ic = int(max(0, min(inject_col, D - 1)))
    Uf = U.float()
    X = feat_head.float()
    N = int(X.shape[0])
    Up = Uf[:, :pr]

    mode = (tail_mode or "nll_cov").strip().lower()
    r: torch.Tensor | None = None
    if mode == "nll_cov" and w_token is not None and w_token.numel() == N:
        w = w_token.float().reshape(-1).clamp(min=0.0)
        s = float(w.sum().item())
        if s > 1e-12:
            w = w * (float(N) / s)
            R = X - (X @ Up) @ Up.T
            rw = torch.sqrt(w.clamp(min=0.0)).view(N, 1) * R
            C = rw.T @ rw
            C = 0.5 * (C + C.T)
            try:
                evals, evecs = torch.linalg.eigh(C)
                r = evecs[:, int(evals.argmax().item())].contiguous()
            except Exception:
                r = None
    if r is None:
        m = X.mean(dim=0)
        r = m - Up @ (Up.t() @ m)

    nr = float(r.norm().item())
    if nr < 1e-8:
        return U
    r = r / nr
    Wmat = Uf.clone()
    Wmat[:, ic] = r
    try:
        Q, _ = torch.linalg.qr(Wmat, mode="reduced")
    except Exception:
        return U
    if Q.shape[1] < D:
        return U
    Q = Q[:, :D]
    return Q.to(dtype=U.dtype, device=U.device)


@torch.no_grad()
def _repair_basis_v_tail(
    basis_v_h: torch.Tensor,
    feat_v: torch.Tensor,
    num_heads: int,
    head_dim: int,
    prefix_rank: int,
    inject_col: int,
    tail_mode: str,
    calib_token_weights: torch.Tensor | None,
) -> torch.Tensor:
    H, d, D = basis_v_h.shape
    if H != num_heads or d != head_dim:
        raise ValueError("basis_v_h shape mismatch")
    fv = feat_v.reshape(-1, num_heads, head_dim).float()
    Ntok = int(fv.shape[0])
    w_flat = None
    if calib_token_weights is not None and tail_mode == "nll_cov":
        w_flat = calib_token_weights.reshape(-1).float()
        if w_flat.numel() != Ntok:
            w_flat = None
    out = basis_v_h.clone()
    for h in range(num_heads):
        out[h] = _inject_tail_one_column(out[h], fv[:, h, :], prefix_rank, inject_col, tail_mode, w_flat)
    return out


@torch.no_grad()
def _repair_basis_kq_tail(
    basis_kq_h: torch.Tensor,
    feat_k: torch.Tensor,
    feat_q: torch.Tensor,
    num_heads: int,
    head_dim: int,
    prefix_rank: int,
    inject_col: int,
    tail_mode: str,
    calib_token_weights: torch.Tensor | None,
) -> torch.Tensor:
    H, d, D = basis_kq_h.shape
    fk = feat_k.reshape(-1, num_heads, head_dim).float()
    fq = feat_q.reshape(-1, num_heads, head_dim).float()
    Ntok = int(fk.shape[0])
    w_kq = None
    if calib_token_weights is not None and tail_mode == "nll_cov":
        w0 = calib_token_weights.reshape(-1).float()
        if w0.numel() == Ntok:
            w_kq = torch.cat([w0, w0], dim=0)
    out = basis_kq_h.clone()
    for h in range(num_heads):
        Xh = torch.cat([fk[:, h, :], fq[:, h, :]], dim=0)
        out[h] = _inject_tail_one_column(out[h], Xh, prefix_rank, inject_col, tail_mode, w_kq)
    return out


@torch.no_grad()
def _relative_layer_error_streaming(
    layer,
    qlayer,
    inps: torch.Tensor,
    attention_mask: torch.Tensor | None,
    nsamples: int,
) -> float:
    """
    Compute ||Y_hr - Y_lr|| / ||Y_hr|| without stacking all sample outputs,
    to avoid peak VRAM spikes on large models.
    """
    num_sq = 0.0
    den_sq = 0.0
    for j in range(int(nsamples)):
        x = inps[j].unsqueeze(0)
        y_hr = layer(x, attention_mask=attention_mask)[0]
        y_lr = qlayer(x, attention_mask=attention_mask)[0]
        diff = y_hr - y_lr
        num_sq += float((diff.float() * diff.float()).sum().item())
        den_sq += float((y_hr.float() * y_hr.float()).sum().item())
        del x, y_hr, y_lr, diff
    if den_sq <= 0.0:
        return 0.0
    return float(math.sqrt(num_sq / den_sq))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_prefix", type=str, default="eigen_basis_opt30b_lossaware")
    ap.add_argument("--nsamples", type=int, default=128, help="calibration samples for SVD/error")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--token_budget", type=int, default=1_500_000)
    ap.add_argument("--max_data_files", type=int, default=300)
    ap.add_argument("--max_data_lines", type=int, default=3_000_000)
    ap.add_argument("--avg_dim", type=int, default=32)
    ap.add_argument("--svd_threshold_init", type=float, default=1.0)
    ap.add_argument("--svd_threshold_floor", type=float, default=0.30)
    ap.add_argument("--threshold_step", type=float, default=0.02)
    ap.add_argument("--min_rank", type=int, default=64)
    ap.add_argument("--strict_original_rank", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--device_map",
        type=str,
        default="",
        help=(
            "模型分片: 空/single => 整模在 --device 单卡; "
            "balanced / auto / sequential => accelerate 多卡; "
            "balanced_4gpu => 等价 balanced 且未指定 max_memory 时为每张可见卡设 19GiB 上限（4 卡跑 30B 常用）; "
            "若以 { 开头则解析为 JSON device_map"
        ),
    )
    ap.add_argument(
        "--max_memory_json",
        type=str,
        default="",
        help='可选，传给 from_pretrained(max_memory=...)，例如 {"0":"18GiB","1":"18GiB","2":"18GiB","3":"18GiB"}',
    )
    ap.add_argument(
        "--per_gpu_max_memory_gib",
        type=float,
        default=0.0,
        help="若 >0 且未设 --max_memory_json，则为当前可见的每张 CUDA 卡设置 max_memory[i]=f'{值}GiB'（4 卡跑 30B 时常用）",
    )
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tokenizer", type=str, default="")
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1])

    # loss-aware controls
    ap.add_argument("--kv_target_gb", type=float, default=2.10, help="target KV cache GB for global allocation")
    ap.add_argument("--kv_baseline_gb", type=float, default=2.625)
    ap.add_argument("--max_candidates_per_layer", type=int, default=6)
    ap.add_argument("--nll_eval_batches", type=int, default=2, help="number of eval batches for NLL scoring")
    ap.add_argument("--nll_eval_seqlen", type=int, default=1024)
    ap.add_argument("--alloc_method", type=str, default="beam", choices=["greedy", "beam"])
    ap.add_argument("--beam_width", type=int, default=4)

    # A1/A2: tail-aware basis repair (same rank / same export format; only changes orthonormal columns)
    ap.add_argument(
        "--basis_repair",
        type=str,
        default="none",
        choices=["none", "v_tail", "kq_rotate", "both"],
        help="v_tail: 用校准 V 在主前缀外残差构造 tail（默认 NLL 加权协方差主方向，见 --basis_repair_tail_mode）后替换一列并 QR; "
        "kq_rotate: 对 K/Q 共享 basis 用 [K;Q] 特征; both: 两者都做",
    )
    ap.add_argument(
        "--basis_repair_prefix_rank",
        type=int,
        default=48,
        help="投影残差时用的「前缀主子空间」列数上界（会按每头实际 D 截断）",
    )
    ap.add_argument(
        "--basis_repair_inject_col",
        type=int,
        default=-1,
        help="替换的列索引；-1 表示自动取 min(prefix_rank, D-1)",
    )
    ap.add_argument(
        "--basis_repair_tail_mode",
        type=str,
        default="nll_cov",
        choices=["nll_cov", "mean"],
        help="tail 方向: nll_cov=逐 token CE 加权残差协方差主特征向量; mean=前缀外均值残差",
    )
    ap.add_argument(
        "--base_basis_pt",
        type=str,
        default="",
        help="可选：基于已有 basis.pt 做 tail 修复，不重跑候选/分配",
    )
    ap.add_argument(
        "--freeze_rank_from_basis_pt",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 表示严格沿用 base_basis_pt 的每层 rank，仅修 basis 方向",
    )
    ap.add_argument(
        "--basis_repair_version",
        type=str,
        default="safe",
        choices=["legacy", "safe"],
        help="safe: 仅改 kept 的最后一列，保留前缀，支持软插入+companion basis+NLL 门控 (推荐); "
        "legacy: 原脚本的 QR+整体替换方案",
    )
    ap.add_argument(
        "--basis_repair_alpha",
        type=float,
        default=0.3,
        help="safe 模式的软插入比例: 0=不动; 0.3 默认; 1=完全替换最后一列为 tail 方向",
    )
    ap.add_argument(
        "--basis_repair_gate_nll",
        type=int,
        default=0,
        choices=[0, 1],
        help="safe 模式: 1 表示逐层做真实 NLL 单层替换对比，仅当修复不劣化 NLL 时才接受",
    )
    ap.add_argument(
        "--basis_repair_gate_tol",
        type=float,
        default=0.0,
        help="safe 模式接受条件: nll_new <= nll_old + tol (tol 为正放宽，负收紧)",
    )
    ap.add_argument(
        "--tail_companion_rank",
        type=int,
        default=0,
        help=">0 时额外导出 tail 补偿 basis（每头 T 个位于 kept^perp 内的方向，用于 attention 公式补偿项）",
    )
    ap.add_argument(
        "--basis_repair_tail_k",
        type=int,
        default=1,
        help="safe 模式下替换尾部的列数：1=Path1(严格等价原 safe 单列), >=2=Path2(多列 V-Tail). "
             "只在 --basis_repair_version=safe 生效；rank 不变、导出格式不变。"
             "建议取值：1/2/4/8 (受 kept rank 上限约束自动截断)。",
    )
    ap.add_argument(
        "--tail_companion_out_suffix",
        type=str,
        default="_tailcomp",
        help="companion basis 输出文件后缀",
    )

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    base_root = repo_root.parent / "EigenAttn_base"
    if not base_root.exists():
        raise FileNotFoundError(f"Cannot find EigenAttn_base at: {base_root}")

    mod_utils = _load_py_module("eigen_base_utils_lossaware", base_root / "decompose" / "eigen_attn_utils.py")
    mod_dm = _load_py_module("eigen_base_decompose_modules_lossaware", base_root / "models" / "decompose_modules.py")
    decompose_opt_layer = mod_utils.decompose_opt_layer
    get_kqv_opt = mod_utils.get_kqv_opt
    OPTEigenAttnDecoderLayer = mod_dm.OPTEigenAttnDecoderLayer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_path = args.tokenizer.strip() if args.tokenizer.strip() else args.model
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    texts = _iter_texts_from_dir(args.data_dir, args.max_data_files, args.max_data_lines)
    stream = _build_token_stream(tok, texts, args.token_budget, args.seed)
    if stream is None:
        raise RuntimeError("empty token stream from data_dir")

    dt = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    dm_raw = (args.device_map or "").strip()
    use_balanced_4gpu_preset = dm_raw.lower() == "balanced_4gpu"
    if use_balanced_4gpu_preset:
        dm_raw = "balanced"
    dm = dm_raw.lower()

    max_memory = None
    mj = (args.max_memory_json or "").strip()
    if mj:
        max_memory = json.loads(mj)
        if not isinstance(max_memory, dict):
            raise ValueError("--max_memory_json 必须解析为 JSON 对象（字典）")
    elif float(args.per_gpu_max_memory_gib) > 0.0 and torch.cuda.is_available():
        n = int(torch.cuda.device_count())
        g = float(args.per_gpu_max_memory_gib)
        max_memory = {i: f"{g}GiB" for i in range(n)}
    elif use_balanced_4gpu_preset and torch.cuda.is_available():
        # 4 卡跑大模型：未手写 max_memory / per_gpu 时为每张可见卡设上限，便于 accelerate 均分权重
        n = int(torch.cuda.device_count())
        max_memory = {i: "19GiB" for i in range(n)}

    load_kw: Dict[str, object] = dict(
        torch_dtype=dt,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if max_memory is not None:
        load_kw["max_memory"] = max_memory

    if not dm_raw or dm in ("single", "none"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": args.device},
            **load_kw,
        )
    elif dm in ("auto", "balanced", "sequential"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=dm,
            **load_kw,
        )
    elif dm_raw.startswith("{"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=json.loads(dm_raw),
            **load_kw,
        )
    else:
        raise ValueError(
            f"未知 --device_map={args.device_map!r}；请用空/single、auto、balanced、sequential 或 JSON 字典"
        )

    model.eval()
    model.config.use_cache = False

    if use_balanced_4gpu_preset and max_memory is not None:
        print(f"[device_map] balanced_4gpu: 使用 per-GPU max_memory={max_memory}", flush=True)

    hf_dm = getattr(model, "hf_device_map", None)
    if isinstance(hf_dm, dict) and hf_dm:
        head = list(hf_dm.items())[:6]
        print(
            f"[device_map] accelerate 分片: {len(hf_dm)} 个模块项, 前若干项={head}{' ...' if len(hf_dm) > 6 else ''}",
            flush=True,
        )
    else:
        p0 = next(model.parameters())
        print(f"[device_map] 单栈: 首参数 device={p0.device} dtype={p0.dtype}", flush=True)

    multi_cuda_devices = set()
    if isinstance(hf_dm, dict) and hf_dm:
        for dv in hf_dm.values():
            if isinstance(dv, int):
                multi_cuda_devices.add(int(dv))
            elif isinstance(dv, str) and dv.startswith("cuda:"):
                try:
                    multi_cuda_devices.add(int(dv.split(":", 1)[1]))
                except Exception:
                    pass
    multi_shard_mode = len(multi_cuda_devices) > 1
    if multi_shard_mode:
        print("[info] 检测到多卡分片，候选替换层将尝试复制 accelerate hook。", flush=True)

    layers = model.model.decoder.layers
    n_layers = len(layers)
    num_heads = int(model.config.num_attention_heads)
    hidden = int(model.config.hidden_size)
    head_dim = hidden // num_heads

    print(
        f"[config] nsamples={args.nsamples} seqlen={args.seqlen} th_init={args.svd_threshold_init:.4f} "
        f"floor={args.svd_threshold_floor:.4f} step={args.threshold_step:.4f} strict_original={args.strict_original_rank} "
        f"target_kv_gb={args.kv_target_gb:.4f} basis_repair={args.basis_repair} "
        f"basis_repair_tail_mode={args.basis_repair_tail_mode}",
        flush=True,
    )

    # build calibration batches for layerwise decomposition
    N = stream.numel()
    if N <= args.seqlen + 2:
        raise RuntimeError(f"token stream too short ({N}) for seqlen={args.seqlen}")
    g = torch.Generator().manual_seed(args.seed)
    calib_batches = []
    for _ in range(args.nsamples):
        s = int(torch.randint(low=0, high=N - args.seqlen - 1, size=(1,), generator=g).item())
        calib_batches.append(stream[s : s + args.seqlen].unsqueeze(0))

    # build eval batches for loss-aware scoring
    nll_batches = []
    nll_len = int(args.nll_eval_seqlen)
    if N <= nll_len + 2:
        raise RuntimeError(f"token stream too short ({N}) for nll_eval_seqlen={nll_len}")
    for _ in range(int(args.nll_eval_batches)):
        s = int(torch.randint(low=0, high=N - nll_len - 1, size=(1,), generator=g).item())
        nll_batches.append(stream[s : s + nll_len].unsqueeze(0))

    # capture layer0 inps for decompose workflow
    dev = model.get_input_embeddings().weight.device
    inps = torch.zeros((args.nsamples, args.seqlen, hidden), dtype=dt, device=dev)
    cache: Dict[str, object] = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for b in calib_batches:
        if cache["i"] >= args.nsamples:
            break
        try:
            _ = model(input_ids=b.to(dev), use_cache=False, return_dict=True)
        except ValueError:
            pass
    layers[0] = layers[0].module
    attention_mask = cache.get("attention_mask", None)

    # fast path: repair an existing basis.pt in-place (rank frozen)
    if int(args.freeze_rank_from_basis_pt) == 1:
        base_pt = (args.base_basis_pt or "").strip()
        if not base_pt:
            raise ValueError("--freeze_rank_from_basis_pt=1 时必须提供 --base_basis_pt")
        base_basis = torch.load(base_pt, map_location="cpu")
        if not isinstance(base_basis, dict):
            raise ValueError(f"invalid base basis file: {base_pt}")

        repair = (args.basis_repair or "none").strip().lower()
        T_comp = int(max(0, args.tail_companion_rank))
        if repair == "none" and T_comp <= 0:
            raise ValueError(
                "freeze_rank 模式下请至少设置 --basis_repair (v_tail / kq_rotate / both) "
                "或 --tail_companion_rank > 0"
            )

        version = str(args.basis_repair_version).lower()
        use_gate = (version == "safe") and int(args.basis_repair_gate_nll) == 1
        if use_gate and multi_shard_mode:
            # Multi-shard + in-place layer swapping for gate NLL is unstable with accelerate hooks.
            # Fall back to "no gate" instead of crashing with CUDA illegal memory access.
            print(
                "[warn] basis_repair_gate_nll=1 with multi-shard device_map is unstable; "
                "disable gate in this run. For strict gate, use single-GPU device_map.",
                flush=True,
            )
            use_gate = False

        calib_token_weights: torch.Tensor | None = None
        if str(args.basis_repair_tail_mode) == "nll_cov":
            calib_token_weights = _precompute_calib_token_nll_weights(model, calib_batches, dev)
            if int(args.verbose) == 1:
                print(
                    f"[basis_repair/freeze] calib token CE weights: shape={tuple(calib_token_weights.shape)} "
                    f"mean={float(calib_token_weights.mean()):.6f}",
                    flush=True,
                )

        # Optional: per-layer single-layer NLL gating needs a baseline NLL-by-layer reference.
        # We compute NLL with the *current* (possibly cumulatively-replaced) model, so the
        # comparison naturally reflects "given already-accepted upstream layers, does this repair help?"
        gate_nll_batches: List[torch.Tensor] = []
        if use_gate:
            glen = int(args.nll_eval_seqlen)
            gN = int(max(1, args.nll_eval_batches))
            if stream.numel() <= glen + 2:
                raise RuntimeError(f"token stream too short for nll_eval_seqlen={glen}")
            g_gate = torch.Generator().manual_seed(args.seed + 7)
            for _ in range(gN):
                s = int(torch.randint(low=0, high=stream.numel() - glen - 1, size=(1,), generator=g_gate).item())
                gate_nll_batches.append(stream[s : s + glen].unsqueeze(0))

        repaired_basis: Dict[str, torch.Tensor] = {}
        companion_basis: Dict[str, torch.Tensor] = {}
        layer_reports: List[Dict] = []
        accept_count = 0

        for li in range(n_layers):
            kq_key = _basis_key(li, "q")
            v_key = _basis_key(li, "v")
            if kq_key not in base_basis or v_key not in base_basis:
                raise KeyError(f"missing layer-{li} keys in base basis: {kq_key} / {v_key}")
            U_kq_packed = base_basis[kq_key].float().cpu()
            U_v_packed = base_basis[v_key].float().cpu()
            rank_kq = int(U_kq_packed.shape[1])
            rank_v = int(U_v_packed.shape[1])
            basis_kq_h_old = _unpack_packed_basis_to_headwise(U_kq_packed, num_heads, head_dim)
            basis_v_h_old = _unpack_packed_basis_to_headwise(U_v_packed, num_heads, head_dim)

            layer = layers[li]
            lay_dev = next(layer.parameters()).device
            inps = inps.to(lay_dev)
            if attention_mask is not None and torch.is_tensor(attention_mask):
                attention_mask = attention_mask.to(lay_dev)
            w_dtype = layer.self_attn.k_proj.weight.dtype
            decomp_args = SimpleNamespace(
                nsamples=int(args.nsamples),
                low_rank=False,
                eigen_attn_params={"threshold": float(args.svd_threshold_init), "avg_dim_features": int(args.avg_dim)},
            )
            feat_k, feat_q, feat_v = get_kqv_opt(layer, inps, decomp_args)
            fk = feat_k.to(dtype=torch.float32, device="cpu")
            fq = feat_q.to(dtype=torch.float32, device="cpu")
            fv = feat_v.to(dtype=torch.float32, device="cpu")
            w_calib = calib_token_weights.float().cpu() if calib_token_weights is not None else None
            tail_mode = str(args.basis_repair_tail_mode)
            alpha = float(args.basis_repair_alpha)

            basis_v_h_new = basis_v_h_old.clone()
            basis_kq_h_new = basis_kq_h_old.clone()
            comp_v_h = torch.zeros((num_heads, head_dim, T_comp), dtype=basis_v_h_old.dtype)
            comp_kq_h = torch.zeros((num_heads, head_dim, T_comp), dtype=basis_kq_h_old.dtype)

            if version == "safe":
                tail_k_cfg = int(max(1, args.basis_repair_tail_k))
                if repair in ("v_tail", "both") or T_comp > 0:
                    basis_v_h_new, comp_v_h = _repair_basis_v_tail_safe(
                        basis_v_h_old, fv, num_heads, head_dim,
                        alpha if repair in ("v_tail", "both") else 0.0,
                        tail_mode, w_calib, T_comp,
                        nsamples=int(args.nsamples), seqlen=int(args.seqlen), avg_dim=int(args.avg_dim),
                        tail_k=tail_k_cfg,
                    )
                if repair in ("kq_rotate", "both") or T_comp > 0:
                    basis_kq_h_new, comp_kq_h = _repair_basis_kq_tail_safe(
                        basis_kq_h_old, fk, fq, num_heads, head_dim,
                        alpha if repair in ("kq_rotate", "both") else 0.0,
                        tail_mode, w_calib, T_comp,
                        nsamples=int(args.nsamples), seqlen=int(args.seqlen), avg_dim=int(args.avg_dim),
                        tail_k=tail_k_cfg,
                    )
            else:
                # legacy: original QR-based replace; no companion export
                Dv = int(basis_v_h_old.shape[2])
                Dkq = int(basis_kq_h_old.shape[2])
                pr_v = int(max(1, min(int(args.basis_repair_prefix_rank), Dv - 1))) if Dv > 1 else 1
                pr_kq = int(max(1, min(int(args.basis_repair_prefix_rank), Dkq - 1))) if Dkq > 1 else 1
                inj_user = int(args.basis_repair_inject_col)
                inj_v = 0 if Dv <= 1 else (min(pr_v, Dv - 1) if inj_user < 0 else int(max(0, min(inj_user, Dv - 1))))
                inj_kq = 0 if Dkq <= 1 else (min(pr_kq, Dkq - 1) if inj_user < 0 else int(max(0, min(inj_user, Dkq - 1))))
                if repair in ("v_tail", "both") and Dv > 1:
                    basis_v_h_new = _repair_basis_v_tail(
                        basis_v_h_old, fv, num_heads, head_dim, pr_v, inj_v, tail_mode, w_calib
                    )
                if repair in ("kq_rotate", "both") and Dkq > 1:
                    basis_kq_h_new = _repair_basis_kq_tail(
                        basis_kq_h_old, fk, fq, num_heads, head_dim, pr_kq, inj_kq, tail_mode, w_calib
                    )

            accepted = True
            nll_old_val: float | None = None
            nll_new_val: float | None = None

            if use_gate and (repair in ("v_tail", "kq_rotate", "both")):
                try:
                    basis_kq_dev_old = basis_kq_h_old.to(device=lay_dev, dtype=w_dtype)
                    basis_v_dev_old = basis_v_h_old.to(device=lay_dev, dtype=w_dtype)
                    basis_kq_dev_new = basis_kq_h_new.to(device=lay_dev, dtype=w_dtype)
                    basis_v_dev_new = basis_v_h_new.to(device=lay_dev, dtype=w_dtype)
                    qlayer_old = OPTEigenAttnDecoderLayer(
                        layer, decomp_args, basis_kq_dev_old, rank_kq, basis_v_dev_old, rank_v, model.config
                    ).to(device=lay_dev, dtype=dt)
                    qlayer_new = OPTEigenAttnDecoderLayer(
                        layer, decomp_args, basis_kq_dev_new, rank_kq, basis_v_dev_new, rank_v, model.config
                    ).to(device=lay_dev, dtype=dt)
                    if multi_shard_mode:
                        _attach_like_accelerate_hook(layer, qlayer_old)
                        _attach_like_accelerate_hook(layer, qlayer_new)
                    orig = layers[li]
                    try:
                        layers[li] = qlayer_old
                        vals_old = [_lm_nll(model, b.to(dev), None) for b in gate_nll_batches]
                        nll_old_val = float(sum(vals_old) / max(1, len(vals_old)))
                        layers[li] = qlayer_new
                        vals_new = [_lm_nll(model, b.to(dev), None) for b in gate_nll_batches]
                        nll_new_val = float(sum(vals_new) / max(1, len(vals_new)))
                    finally:
                        layers[li] = orig

                    accepted = (nll_new_val <= nll_old_val + float(args.basis_repair_gate_tol))
                    del qlayer_old, qlayer_new
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as gate_e:
                    # Keep run alive and choose monotone-safe fallback (reject this layer change).
                    accepted = False
                    print(
                        f"[warn] gate-nll failed at layer={li:02d}: {gate_e}. "
                        "Fallback to REJECT for this layer.",
                        flush=True,
                    )

            if accepted:
                accept_count += 1
                final_v_h = basis_v_h_new
                final_kq_h = basis_kq_h_new
            else:
                final_v_h = basis_v_h_old
                final_kq_h = basis_kq_h_old
                if T_comp > 0:
                    comp_v_h = torch.zeros_like(comp_v_h)
                    comp_kq_h = torch.zeros_like(comp_kq_h)

            U_kq_new = _pack_headwise_basis(final_kq_h.float().cpu(), rank_kq, num_heads, head_dim).float().cpu()
            U_v_new = _pack_headwise_basis(final_v_h.float().cpu(), rank_v, num_heads, head_dim).float().cpu()
            repaired_basis[_basis_key(li, "q")] = U_kq_new
            repaired_basis[_basis_key(li, "k")] = U_kq_new
            repaired_basis[_basis_key(li, "v")] = U_v_new

            if T_comp > 0:
                rank_comp = int(T_comp * num_heads)
                U_kq_comp_packed = _pack_headwise_basis(
                    comp_kq_h.float().cpu(), rank_comp, num_heads, head_dim
                ).float().cpu()
                U_v_comp_packed = _pack_headwise_basis(
                    comp_v_h.float().cpu(), rank_comp, num_heads, head_dim
                ).float().cpu()
                companion_basis[_basis_key(li, "q") + "_tail"] = U_kq_comp_packed
                companion_basis[_basis_key(li, "k") + "_tail"] = U_kq_comp_packed
                companion_basis[_basis_key(li, "v") + "_tail"] = U_v_comp_packed

            layer_reports.append(
                {
                    "layer": li,
                    "rank_kq": rank_kq,
                    "rank_v": rank_v,
                    "k_kept_kq": int(rank_kq // num_heads),
                    "k_kept_v": int(rank_v // num_heads),
                    "repair": repair,
                    "version": version,
                    "alpha": float(args.basis_repair_alpha),
                    "tail_k": int(args.basis_repair_tail_k),
                    "tail_companion_rank": T_comp,
                    "nll_old": nll_old_val,
                    "nll_new": nll_new_val,
                    "accepted": bool(accepted),
                }
            )

            if int(args.verbose) == 1:
                gate_note = ""
                if use_gate and nll_old_val is not None and nll_new_val is not None:
                    gate_note = (
                        f" nll_old={nll_old_val:.6f} nll_new={nll_new_val:.6f} "
                        f"delta={nll_new_val - nll_old_val:+.6f} "
                        f"{'ACCEPT' if accepted else 'REJECT'}"
                    )
                print(
                    f"[freeze-repair] layer={li:02d} rk={rank_kq} rv={rank_v} "
                    f"ver={version} mode={repair}/{args.basis_repair_tail_mode} alpha={alpha:.3f} "
                    f"tail_k={int(args.basis_repair_tail_k)} T_comp={T_comp}{gate_note}",
                    flush=True,
                )

            del feat_k, feat_q, feat_v, fk, fq, fv
            del basis_v_h_old, basis_v_h_new, basis_kq_h_old, basis_kq_h_new, comp_v_h, comp_kq_h
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out_pt = out_dir / f"{args.out_prefix}.pt"
        torch.save(repaired_basis, out_pt)
        print(f"[saved] {out_pt}", flush=True)

        if T_comp > 0:
            out_comp_pt = out_dir / f"{args.out_prefix}{args.tail_companion_out_suffix}.pt"
            torch.save(companion_basis, out_comp_pt)
            print(f"[saved] {out_comp_pt}", flush=True)

        report = {
            "model": args.model,
            "mode": "basis_repair_freeze_rank_from_basis_pt",
            "config": {
                "base_basis_pt": base_pt,
                "basis_repair": str(args.basis_repair),
                "basis_repair_tail_mode": str(args.basis_repair_tail_mode),
                "basis_repair_version": version,
                "basis_repair_alpha": float(args.basis_repair_alpha),
                "basis_repair_gate_nll": int(args.basis_repair_gate_nll),
                "basis_repair_gate_tol": float(args.basis_repair_gate_tol),
                "basis_repair_prefix_rank": int(args.basis_repair_prefix_rank),
                "basis_repair_inject_col": int(args.basis_repair_inject_col),
                "basis_repair_tail_k": int(args.basis_repair_tail_k),
                "tail_companion_rank": T_comp,
                "tail_companion_out_suffix": str(args.tail_companion_out_suffix),
                "nsamples": int(args.nsamples),
                "seqlen": int(args.seqlen),
                "nll_eval_batches": int(args.nll_eval_batches),
                "nll_eval_seqlen": int(args.nll_eval_seqlen),
                "avg_dim": int(args.avg_dim),
            },
            "accept_count": int(accept_count),
            "total_layers": int(n_layers),
            "layers": layer_reports,
        }
        out_report = out_dir / f"{args.out_prefix}_report.json"
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_report}", flush=True)
        return

    # baseline NLL for loss-aware scoring
    baseline_nll_vals = []
    for b in nll_batches:
        bdev = b.to(dev)
        baseline_nll_vals.append(_lm_nll(model, bdev, None))
    baseline_nll = float(sum(baseline_nll_vals) / max(1, len(baseline_nll_vals)))
    print(f"[loss-aware] baseline_nll={baseline_nll:.6f} (batches={len(nll_batches)})", flush=True)

    calib_token_weights: torch.Tensor | None = None
    repair_glob = (args.basis_repair or "none").strip().lower()
    if repair_glob != "none" and str(args.basis_repair_tail_mode) == "nll_cov":
        calib_token_weights = _precompute_calib_token_nll_weights(model, calib_batches, dev)
        if int(args.verbose) == 1:
            print(
                f"[basis_repair] calib token CE weights: shape={tuple(calib_token_weights.shape)} "
                f"mean={float(calib_token_weights.mean()):.6f}",
                flush=True,
            )

    # collect layer candidates
    layer_candidates: List[List[Candidate]] = []
    layer_alt_modules: List[List[torch.nn.Module | None]] = []
    keep_alt_modules_for_combo = (args.alloc_method == "beam") and (not multi_shard_mode)
    if args.alloc_method == "beam" and multi_shard_mode:
        print(
            "[warn] 多卡分片下为避免候选模块累计导致 CUDA 非法访问，"
            "将禁用跨层组合缓存并回退到 greedy 分配（仍保留单层真实 NLL 打分）。",
            flush=True,
        )
    report_layers = []
    stored_basis_kq_cpu: List[torch.Tensor] = []
    stored_basis_v_cpu: List[torch.Tensor] = []

    for li in range(n_layers):
        layer = layers[li]
        lay_dev = next(layer.parameters()).device
        inps = inps.to(lay_dev)
        if attention_mask is not None and torch.is_tensor(attention_mask):
            attention_mask = attention_mask.to(lay_dev)

        decomp_args = SimpleNamespace(
            nsamples=int(args.nsamples),
            low_rank=False,
            eigen_attn_params={"threshold": float(args.svd_threshold_init), "avg_dim_features": int(args.avg_dim)},
        )
        basis_kq_h, eval_kq, basis_v_h, eval_v = decompose_opt_layer(layer, inps, decomp_args, num_heads, li)
        w_dtype = layer.self_attn.k_proj.weight.dtype
        # Keep repair path on CPU to avoid cross-device pressure/illegal access in multi-shard mode.
        basis_kq_h = basis_kq_h.to(dtype=torch.float32, device="cpu")
        basis_v_h = basis_v_h.to(dtype=torch.float32, device="cpu")

        repair = (args.basis_repair or "none").strip().lower()
        if repair != "none":
            feat_k, feat_q, feat_v = get_kqv_opt(layer, inps, decomp_args)
            fk = feat_k.to(dtype=torch.float32, device="cpu")
            fq = feat_q.to(dtype=torch.float32, device="cpu")
            fv = feat_v.to(dtype=torch.float32, device="cpu")
            w_calib = calib_token_weights.float().cpu() if calib_token_weights is not None else None
            tail_mode = str(args.basis_repair_tail_mode)
            Dv = int(basis_v_h.shape[2])
            Dkq = int(basis_kq_h.shape[2])
            pr_v = int(max(1, min(int(args.basis_repair_prefix_rank), Dv - 1)))
            pr_kq = int(max(1, min(int(args.basis_repair_prefix_rank), Dkq - 1)))
            inj_user = int(args.basis_repair_inject_col)
            inj_v = min(pr_v, Dv - 1) if inj_user < 0 else int(max(0, min(inj_user, Dv - 1)))
            inj_kq = min(pr_kq, Dkq - 1) if inj_user < 0 else int(max(0, min(inj_user, Dkq - 1)))
            if repair in ("v_tail", "both"):
                basis_v_h = _repair_basis_v_tail(
                    basis_v_h, fv, num_heads, head_dim, pr_v, inj_v, tail_mode, w_calib
                )
            if repair in ("kq_rotate", "both"):
                basis_kq_h = _repair_basis_kq_tail(
                    basis_kq_h, fk, fq, num_heads, head_dim, pr_kq, inj_kq, tail_mode, w_calib
                )
            del feat_k, feat_q, feat_v, fk, fq, fv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        stored_basis_kq_cpu.append(basis_kq_h.detach().float().cpu().clone())
        stored_basis_v_cpu.append(basis_v_h.detach().float().cpu().clone())

        basis_kq_h = basis_kq_h.to(device=lay_dev, dtype=w_dtype)
        basis_v_h = basis_v_h.to(device=lay_dev, dtype=w_dtype)

        # enumerate unique candidate ranks over thresholds
        cands: List[Candidate] = []
        seen = set()
        th = float(args.svd_threshold_init)
        while th >= float(args.svd_threshold_floor) - 1e-12:
            raw_kq = _rank_from_eval(eval_kq, th, num_heads)
            raw_v = _rank_from_eval(eval_v, th, num_heads)
            rkq = _finalize_rank(raw_kq, num_heads, head_dim, args.min_rank, bool(args.strict_original_rank))
            rv = _finalize_rank(raw_v, num_heads, head_dim, args.min_rank, bool(args.strict_original_rank))
            key = (rkq, rv)
            if key not in seen:
                seen.add(key)
                qlayer = OPTEigenAttnDecoderLayer(
                    layer, decomp_args, basis_kq_h, rkq, basis_v_h, rv, model.config
                ).to(device=lay_dev, dtype=dt)
                lerr = _relative_layer_error_streaming(
                    layer=layer,
                    qlayer=qlayer,
                    inps=inps,
                    attention_mask=attention_mask,
                    nsamples=int(args.nsamples),
                )
                per_kq = _per_head_rank_from_eval(eval_kq, th)
                per_v = _per_head_rank_from_eval(eval_v, th)
                cands.append(
                    Candidate(
                        layer=li,
                        threshold=float(th),
                        rank_kq=int(rkq),
                        rank_v=int(rv),
                        per_head_kq=per_kq,
                        per_head_v=per_v,
                        energy_kq=_energy_stats(eval_kq, per_kq),
                        energy_v=_energy_stats(eval_v, per_v),
                        layer_err=lerr,
                    )
                )
                del qlayer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            th -= float(args.threshold_step)

        # sort from high-cost to low-cost, keep limited points
        cands = sorted(cands, key=lambda x: (-x.cost, -x.threshold))
        cands = _downsample_candidates(cands, max(2, int(args.max_candidates_per_layer)))

        if int(args.verbose) == 1:
            print(
                f"[layer {li:02d}] candidates={len(cands)} "
                f"cost_range=[{min(c.cost for c in cands)}, {max(c.cost for c in cands)}]",
                flush=True,
            )
            for ci, c in enumerate(cands):
                print(
                    f"  cand#{ci} th={c.threshold:.4f} rkq={c.rank_kq} rv={c.rank_v} "
                    f"lerr={c.layer_err:.6f} energy_kq={c.energy_kq['mean']:.4f} energy_v={c.energy_v['mean']:.4f}",
                    flush=True,
                )

        # NLL scoring per candidate (single-layer substitution)
        layer_dev_for_nll = next(layer.parameters()).device
        alt_modules_for_layer: List[torch.nn.Module | None] = [None for _ in range(len(cands))]
        for ci, c in enumerate(cands):
            qlayer = OPTEigenAttnDecoderLayer(
                layer, decomp_args, basis_kq_h, c.rank_kq, basis_v_h, c.rank_v, model.config
            ).to(device=layer_dev_for_nll, dtype=dt)
            if multi_shard_mode and (not _attach_like_accelerate_hook(layer, qlayer)):
                raise RuntimeError(
                    f"layer {li} 候选模块无法复制 accelerate hook，无法在多卡分片下做真实 NLL 打分。"
                )
            if keep_alt_modules_for_combo:
                alt_modules_for_layer[ci] = qlayer

            orig = layers[li]
            try:
                layers[li] = qlayer
                nll_vals = []
                for b in nll_batches:
                    bdev = b.to(dev)
                    nll_vals.append(_lm_nll(model, bdev, None))
                cand_nll = float(sum(nll_vals) / max(1, len(nll_vals)))
                c.nll_delta = float(cand_nll - baseline_nll)
            finally:
                layers[li] = orig

            if int(args.verbose) == 1:
                print(
                    f"  [layer {li:02d}] NLL cand#{ci} cost={c.cost} nll={cand_nll:.6f} delta={c.nll_delta:.6f}",
                    flush=True,
                )
            if not keep_alt_modules_for_combo:
                del qlayer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # IMPORTANT: keep original device_map placement for end-to-end NLL scoring.
        # Moving processed layers to CPU here will break later full-model forward
        # with device mismatch (some layers on CPU, inputs on CUDA).

        layer_candidates.append(cands)
        layer_alt_modules.append(alt_modules_for_layer)
        report_layers.append(
            {
                "layer": li,
                "candidates": [
                    {
                        "threshold": c.threshold,
                        "rank_kq": c.rank_kq,
                        "rank_v": c.rank_v,
                        "cost": c.cost,
                        "layer_error": c.layer_err,
                        "nll_delta": c.nll_delta,
                        "energy_kq": c.energy_kq,
                        "energy_v": c.energy_v,
                    }
                    for c in cands
                ],
            }
        )

    # global allocation under KV budget
    full_cost = n_layers * (2 * hidden)
    target_total = int(round((float(args.kv_target_gb) / float(args.kv_baseline_gb)) * full_cost))

    print(
        f"[alloc] method={args.alloc_method} full_cost={full_cost} target_total={target_total} "
        f"target_kv_gb={args.kv_target_gb:.4f}",
        flush=True,
    )
    final_combo_nll = None
    use_beam_combo = (args.alloc_method == "beam") and keep_alt_modules_for_combo
    if use_beam_combo:
        cur_idx, cur_total, alloc_trace, final_combo_nll = _allocate_beam_search(
            model=model,
            layers=layers,
            layer_candidates=layer_candidates,
            layer_alt_modules=layer_alt_modules,
            nll_batches=nll_batches,
            input_dev=dev,
            target_total=target_total,
            beam_width=int(args.beam_width),
            verbose=int(args.verbose),
        )
    else:
        if args.alloc_method == "beam" and (not keep_alt_modules_for_combo):
            print("[alloc] 已自动回退到 greedy（多卡稳态模式）。", flush=True)
        cur_idx, cur_total, alloc_trace = _allocate_greedy(
            layer_candidates=layer_candidates,
            n_layers=n_layers,
            target_total=target_total,
            verbose=int(args.verbose),
        )

    selected = [layer_candidates[i][cur_idx[i]] for i in range(n_layers)]
    final_total = sum(c.cost for c in selected)
    final_kv_est = float(args.kv_baseline_gb) * (float(final_total) / float(full_cost))
    print(
        f"[alloc] done total={final_total} est_kv_gb={final_kv_est:.6f} "
        f"(target={args.kv_target_gb:.6f})",
        flush=True,
    )

    # export basis（与打分阶段一致：用每层缓存的 headwise basis，含 basis_repair）
    basis_dict: Dict[str, torch.Tensor] = {}
    for c in selected:
        li = c.layer
        if li >= len(stored_basis_kq_cpu) or li >= len(stored_basis_v_cpu):
            raise RuntimeError("internal: missing cached basis for export")
        basis_kq_h = stored_basis_kq_cpu[li]
        basis_v_h = stored_basis_v_cpu[li]
        U_kq = _pack_headwise_basis(basis_kq_h, int(c.rank_kq), num_heads, head_dim).float()
        U_v = _pack_headwise_basis(basis_v_h, int(c.rank_v), num_heads, head_dim).float()
        basis_dict[_basis_key(li, "q")] = U_kq
        basis_dict[_basis_key(li, "k")] = U_kq
        basis_dict[_basis_key(li, "v")] = U_v

    out_pt = out_dir / f"{args.out_prefix}.pt"
    torch.save(basis_dict, out_pt)
    print(f"[saved] {out_pt}", flush=True)

    report = {
        "model": args.model,
        "mode": f"loss_aware_{args.alloc_method}_single_layer_nll_candidates",
        "config": {
            "nsamples": int(args.nsamples),
            "seqlen": int(args.seqlen),
            "svd_threshold_init": float(args.svd_threshold_init),
            "svd_threshold_floor": float(args.svd_threshold_floor),
            "threshold_step": float(args.threshold_step),
            "strict_original_rank": int(args.strict_original_rank),
            "max_candidates_per_layer": int(args.max_candidates_per_layer),
            "nll_eval_batches": int(args.nll_eval_batches),
            "nll_eval_seqlen": int(args.nll_eval_seqlen),
            "kv_target_gb": float(args.kv_target_gb),
            "kv_baseline_gb": float(args.kv_baseline_gb),
            "alloc_method": str(args.alloc_method),
            "beam_width": int(args.beam_width),
            "basis_repair": str(args.basis_repair),
            "basis_repair_prefix_rank": int(args.basis_repair_prefix_rank),
            "basis_repair_inject_col": int(args.basis_repair_inject_col),
            "basis_repair_tail_mode": str(args.basis_repair_tail_mode),
        },
        "baseline_nll": baseline_nll,
        "allocation": {
            "full_cost": int(full_cost),
            "target_total": int(target_total),
            "final_total": int(final_total),
            "final_kv_gb_est": float(final_kv_est),
            "final_combo_nll": None if final_combo_nll is None else float(final_combo_nll),
            "trace": alloc_trace,
        },
        "layers": report_layers,
        "selected": [
            {
                "layer": c.layer,
                "threshold": c.threshold,
                "rank_kq": c.rank_kq,
                "rank_v": c.rank_v,
                "cost": c.cost,
                "layer_error": c.layer_err,
                "nll_delta": c.nll_delta,
                "energy_kq": c.energy_kq,
                "energy_v": c.energy_v,
            }
            for c in selected
        ],
    }
    out_report = out_dir / f"{args.out_prefix}_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_report}", flush=True)


if __name__ == "__main__":
    main()

