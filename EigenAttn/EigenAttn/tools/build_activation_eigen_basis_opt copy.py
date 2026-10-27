#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从校准数据上的真实 Q/K/V 输出（按 head）做协方差 PCA，生成与 eigen_basis.pt 同键、同 rank 的激活感知 basis，
并与参考 basis 做子空间/投影矩阵对比（rank 不变）。

用法（在 EigenAttn 目录下，PYTHONPATH 含当前目录）:
  python tools/build_activation_eigen_basis_opt.py \\
    --model /path/to/opt-30b \\
    --ref_basis qkv_outputs/eigen_basis.pt \\
    --out_basis qkv_outputs/eigen_basis_activation.pt \\
    --data_dir /path/to/c4-or-texts \\
    --max_batches 32 --batch_size 1 --seqlen 512 \\
    --compare_json qkv_outputs/basis_activation_compare.json

依赖: transformers, torch；大模型建议 accelerate（device_map=auto）。
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _basis_key(layer_idx: int, branch: str) -> str:
    return f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{branch}_proj_basis"


def _parse_layers(spec: str, n_layers: int) -> List[int]:
    s = spec.strip().lower()
    if s == "all" or s == "":
        return list(range(n_layers))
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted({i for i in out if 0 <= i < n_layers})
    if not out:
        raise ValueError(f"no valid layers in {spec!r} (n_layers={n_layers})")
    return out


def iter_texts_from_dir(data_dir: str, max_files: int = 80, max_lines: int = 300000):
    exts = ["*.txt", "*.jsonl", "*.json", "*.gz"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", e), recursive=True))
    files = sorted(files)[:max_files]
    if not files:
        return
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
                                t = None
                                if isinstance(it, dict):
                                    t = it.get("text") or it.get("content")
                                elif isinstance(it, str):
                                    t = it
                                if isinstance(t, str) and t.strip():
                                    yield t.strip()
                                    n += 1
                                    if n >= max_lines:
                                        return
                        elif isinstance(obj, dict):
                            t = obj.get("text") or obj.get("content")
                            if isinstance(t, str) and t.strip():
                                yield t.strip()
                    except Exception:
                        pass
                    continue

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = None
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            j = json.loads(line)
                            if isinstance(j, dict):
                                t = j.get("text") or j.get("content")
                        except Exception:
                            t = None
                    if t is None:
                        t = line
                    if isinstance(t, str) and t.strip():
                        yield t.strip()
                        n += 1
                        if n >= max_lines:
                            return
        except Exception:
            continue


def build_token_stream(tok, texts_iter, target_tokens: int, seed: int):
    random.seed(seed)
    ids: List[int] = []
    total = 0
    for t in texts_iter:
        enc = tok(t, add_special_tokens=False)["input_ids"]
        if not enc:
            continue
        ids.extend(enc + [tok.eos_token_id])
        total += len(enc) + 1
        if total >= target_tokens:
            break
    if total == 0:
        return None
    return torch.tensor(ids, dtype=torch.long)


def batchify_stream_cpu(stream_cpu: torch.Tensor, seqlen: int, batch_size: int):
    N = stream_cpu.numel()
    assert N > seqlen + 1
    while True:
        starts = torch.randint(low=0, high=N - seqlen - 1, size=(batch_size,))
        x = torch.stack([stream_cpu[s : s + seqlen] for s in starts], dim=0)
        attn = torch.ones_like(x, dtype=torch.long)
        yield x, attn


@torch.no_grad()
def _pca_cols_from_cov(YtY: torch.Tensor, n: int, r: int) -> torch.Tensor:
    """YtY: [d,d] symmetric, n token count. Return V [d,r] orthonormal (largest variance)."""
    d = YtY.shape[0]
    if n <= 0:
        return torch.eye(d, dtype=YtY.dtype, device=YtY.device)[:, :r]
    C = (YtY / float(n)).float()
    w, V = torch.linalg.eigh(C)
    return V[:, -r:].contiguous().float()


def _orthonormalize(U: torch.Tensor) -> torch.Tensor:
    """U: [d,r] -> QR thin Q."""
    if U.numel() == 0:
        return U
    q, _ = torch.linalg.qr(U, mode="reduced")
    return q[:, : U.shape[1]].contiguous()


def principal_angle_stats(U_old: torch.Tensor, U_new: torch.Tensor) -> Dict[str, float]:
    """
    U_old, U_new: [d,r] column orthonormal (approximately).
    Principal angles via SVD of U_old^T U_new.
    """
    Uo = _orthonormalize(U_old.float().cpu())
    Un = _orthonormalize(U_new.float().cpu())
    r = min(Uo.shape[1], Un.shape[1])
    if r == 0:
        return {"mean_deg": float("nan"), "max_deg": float("nan"), "min_cos": float("nan")}
    M = Uo.t() @ Un
    s = torch.linalg.svdvals(M)
    s = torch.clamp(s, -1.0, 1.0)
    ang = torch.acos(s) * (180.0 / math.pi)
    Po = Uo @ Uo.t()
    Pn = Un @ Un.t()
    fro = torch.linalg.matrix_norm(Po - Pn, ord="fro").item()
    r_eff = float(r)
    return {
        "mean_deg": float(ang.mean().item()),
        "max_deg": float(ang.max().item()),
        "min_cos": float(s.min().item()),
        "proj_fro_diff": float(fro),
        "proj_fro_diff_norm": float(fro / max(math.sqrt(2.0 * r_eff), 1e-12)),
    }


class _PerHeadCovAccumulator:
    def __init__(self, num_heads: int, head_dim: int, device_cpu: torch.device):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device_cpu = device_cpu
        z = torch.zeros(head_dim, head_dim, dtype=torch.float32, device=device_cpu)
        self.yty_q = [z.clone() for _ in range(num_heads)]
        self.yty_k = [z.clone() for _ in range(num_heads)]
        self.yty_v = [z.clone() for _ in range(num_heads)]
        self.n_tokens = 0

    def _add_branch(self, y_list: List[torch.Tensor], y_flat: torch.Tensor):
        # y_flat: [N, hidden] on CPU float32
        N = y_flat.shape[0]
        if N == 0:
            return
        hd = self.head_dim
        for h in range(self.num_heads):
            sl = y_flat[:, h * hd : (h + 1) * hd]
            y_list[h].addmm_(sl.t(), sl)

    def update_from_outputs(self, yq: torch.Tensor, yk: torch.Tensor, yv: torch.Tensor):
        """y*: [bs, seq, hidden] on any device."""
        yq = yq.detach().float().cpu()
        yk = yk.detach().float().cpu()
        yv = yv.detach().float().cpu()
        bs, seq, h = yq.shape
        assert h == self.num_heads * self.head_dim
        N = bs * seq
        self.n_tokens += N
        fq = yq.reshape(N, h)
        fk = yk.reshape(N, h)
        fv = yv.reshape(N, h)
        self._add_branch(self.yty_q, fq)
        self._add_branch(self.yty_k, fk)
        self._add_branch(self.yty_v, fv)


def _build_u_from_covs(
    acc: _PerHeadCovAccumulator,
    rank_kq_per_head: int,
    rank_v_per_head: int,
    repair_keqk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = acc.num_heads * acc.head_dim
    rank_kq = rank_kq_per_head * acc.num_heads
    rank_v = rank_v_per_head * acc.num_heads
    u_q = torch.zeros(hidden, rank_kq, dtype=torch.float32)
    u_k = torch.zeros(hidden, rank_kq, dtype=torch.float32)
    u_v = torch.zeros(hidden, rank_v, dtype=torch.float32)
    n = acc.n_tokens
    for h in range(acc.num_heads):
        row = slice(h * acc.head_dim, (h + 1) * acc.head_dim)
        col_kq = slice(h * rank_kq_per_head, (h + 1) * rank_kq_per_head)
        col_v = slice(h * rank_v_per_head, (h + 1) * rank_v_per_head)
        Vq = _pca_cols_from_cov(acc.yty_q[h], n, rank_kq_per_head)
        Vk = _pca_cols_from_cov(acc.yty_k[h], n, rank_kq_per_head) if repair_keqk != 1 else Vq
        Vv = _pca_cols_from_cov(acc.yty_v[h], n, rank_v_per_head)
        u_q[row, col_kq] = Vq
        u_k[row, col_kq] = Vk if repair_keqk != 1 else Vq
        u_v[row, col_v] = Vv
    return u_q, u_k, u_v


@torch.no_grad()
def collect_activation_basis_one_layer(
    model: torch.nn.Module,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    rank_kq_per_head: int,
    rank_v_per_head: int,
    repair_keqk: int,
    data_iter,
    input_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    layer = model.model.decoder.layers[layer_idx]
    attn = layer.self_attn
    acc = _PerHeadCovAccumulator(num_heads, head_dim, torch.device("cpu"))
    hook_out: Dict[str, torch.Tensor] = {}

    def q_hook(_m, _inp, out):
        hook_out["q"] = out

    def k_hook(_m, _inp, out):
        hook_out["k"] = out

    def v_hook(_m, _inp, out):
        hook_out["v"] = out

    hq = attn.q_proj.register_forward_hook(q_hook)
    hk = attn.k_proj.register_forward_hook(k_hook)
    hv = attn.v_proj.register_forward_hook(v_hook)
    hooks = [hq, hk, hv]

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    try:
        for batch in data_iter:
            hook_out.clear()
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0]
            elif isinstance(batch, dict):
                input_ids = batch.get("input_ids")
            else:
                input_ids = batch
            if not torch.is_tensor(input_ids):
                continue
            input_ids = input_ids.to(input_device)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            _ = model.model.decoder(input_ids)
            if "q" in hook_out and "k" in hook_out and "v" in hook_out:
                acc.update_from_outputs(hook_out["q"], hook_out["k"], hook_out["v"])
    finally:
        for h in hooks:
            h.remove()
        model.config.use_cache = use_cache

    u_q, u_k, u_v = _build_u_from_covs(acc, rank_kq_per_head, rank_v_per_head, repair_keqk)
    return u_q, u_k, u_v, acc.n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--ref_basis", type=str, required=True, help="参考 eigen_basis.pt（定 rank / 键名）")
    ap.add_argument("--out_basis", type=str, required=True)
    ap.add_argument("--compare_json", type=str, default="", help="写出新旧 basis 对比指标")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--token_budget", type=int, default=500000)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--layers", type=str, default="all", help='e.g. "all" or "46" or "32-47"')
    ap.add_argument("--repair_keqk", type=int, default=1, choices=[0, 1], help="1: K basis 与 Q 相同（与现有 repair 一致）")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--device_map", type=str, default="auto", help='transformers device_map, 或 "cuda:0" 单卡')
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ref_path = Path(args.ref_basis)
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    ref_state = torch.load(str(ref_path), map_location="cpu")
    if not isinstance(ref_state, dict):
        raise TypeError("ref basis must be a dict")

    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    n_layers = int(cfg.num_hidden_layers)
    num_heads = int(cfg.num_attention_heads)
    hidden = int(cfg.hidden_size)
    head_dim = hidden // num_heads

    Uq0 = ref_state[_basis_key(0, "q")]
    if Uq0.shape[0] != hidden:
        raise ValueError(f"ref basis hidden {Uq0.shape[0]} != config hidden {hidden}")
    rank_kq = int(Uq0.shape[1])
    rank_v = int(ref_state[_basis_key(0, "v")].shape[1])
    if rank_kq % num_heads or rank_v % num_heads:
        raise ValueError(f"rank_kq={rank_kq} or rank_v={rank_v} not divisible by num_heads={num_heads}")
    rank_kq_per_head = rank_kq // num_heads
    rank_v_per_head = rank_v // num_heads

    layer_ids = _parse_layers(args.layers, n_layers)

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    texts = iter_texts_from_dir(args.data_dir)
    stream = build_token_stream(tok, texts, target_tokens=args.token_budget, seed=args.seed)
    if stream is None:
        raise RuntimeError("no text read from data_dir")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.device_map != "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dt,
            device_map={"": args.device_map},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dt,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    input_device = model.get_input_embeddings().weight.device

    # 固定数据迭代器：每层复用同一批 batch 序列（可复现）
    def make_batches():
        g = torch.Generator()
        g.manual_seed(args.seed)
        N = stream.numel()
        for _ in range(args.max_batches):
            starts = torch.randint(low=0, high=N - args.seqlen - 1, size=(args.batch_size,), generator=g)
            x = torch.stack([stream[s : s + args.seqlen] for s in starts], dim=0)
            yield x

    out_state: Dict[str, torch.Tensor] = {}
    report: Dict[str, object] = {
        "model": args.model,
        "ref_basis": str(ref_path),
        "layers": layer_ids,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "seqlen": args.seqlen,
        "repair_keqk": args.repair_keqk,
        "per_layer": [],
    }

    batch_list = list(make_batches())
    if not batch_list:
        raise RuntimeError("no batches")

    for li in layer_ids:
        u_q, u_k, u_v, n_tok = collect_activation_basis_one_layer(
            model,
            layer_idx=li,
            num_heads=num_heads,
            head_dim=head_dim,
            rank_kq_per_head=rank_kq_per_head,
            rank_v_per_head=rank_v_per_head,
            repair_keqk=args.repair_keqk,
            data_iter=batch_list,
            input_device=input_device,
        )
        out_state[_basis_key(li, "q")] = u_q
        out_state[_basis_key(li, "k")] = u_k
        out_state[_basis_key(li, "v")] = u_v

        rec: Dict[str, object] = {
            "layer": li,
            "n_tokens_seen": n_tok,
        }
        for br, Un in [("q", u_q), ("k", u_k), ("v", u_v)]:
            key = _basis_key(li, br)
            if key in ref_state:
                st = principal_angle_stats(ref_state[key].float(), Un)
                rec[f"vs_ref_{br}"] = st
        report["per_layer"].append(rec)
        print(f"[layer {li}] tokens={n_tok} q_mean_angle_deg={rec.get('vs_ref_q', {}).get('mean_deg', 'n/a')}")

    # 未计算的层：从参考拷贝（保持完整 checkpoint 可加载）
    for li in range(n_layers):
        for br in ("q", "k", "v"):
            key = _basis_key(li, br)
            if key not in out_state:
                out_state[key] = ref_state[key].float().clone()

    out_path = Path(args.out_basis)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, out_path)
    print(f"[saved] {out_path} (activation layers={layer_ids})")

    if args.compare_json:
        cmp_path = Path(args.compare_json)
        cmp_path.parent.mkdir(parents=True, exist_ok=True)
        # 汇总标量
        summary = {"activation_layers": layer_ids}
        for br in ("q", "k", "v"):
            angles = []
            for pl in report["per_layer"]:
                d = pl.get(f"vs_ref_{br}")
                if isinstance(d, dict) and "mean_deg" in d and not math.isnan(d["mean_deg"]):
                    angles.append(d["mean_deg"])
            if angles:
                summary[f"{br}_mean_angle_deg_avg_over_layers"] = float(sum(angles) / len(angles))
        report["summary"] = summary
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[compare] {cmp_path}")


if __name__ == "__main__":
    main()
