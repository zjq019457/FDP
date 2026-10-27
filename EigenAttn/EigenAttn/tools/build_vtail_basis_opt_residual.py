#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线 B：全秩 OPT 在 calibration 上采集 v_proj 输出，对主 V basis 做正交补残差 R = V - Q Q^T V，
对残差协方差做特征值分解，导出 tail v_proj_basis（与 rebuild 的 tail_basis_pt 键一致）。

默认 lowram=1：逐层累协方差、多遍数据，内存 O(d^2) 不随层数倍增。
仅单卡：device_map 指向 --device（如 cuda:0）。
"""

from __future__ import annotations

import argparse
import gc
import glob
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _basis_key(layer: int) -> str:
    return f"base_model.model.model.decoder.layers.{layer}.self_attn.v_proj_basis"


def _iter_jsonl_texts_gz(fp: str, text_key: str = "text", read_size: int = 1 << 20) -> Iterable[str]:
    import json

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

    buf2 = buf.strip()
    if buf2:
        import json as _j

        dec2 = _j.JSONDecoder()
        i, n = 0, len(buf2)

        def _sk(s, j, m):
            while j < m and s[j] in " \t\r\n,":
                j += 1
            return j

        while True:
            i = _sk(buf2, i, n)
            if i >= n:
                break
            try:
                obj, j = dec2.raw_decode(buf2, i)
            except json.JSONDecodeError:
                break
            i = j
            if isinstance(obj, dict):
                t = obj.get(text_key)
                if isinstance(t, str) and t.strip():
                    yield t.strip()


def _iter_texts_from_dir(data_dir: str, max_files: int, max_lines: int) -> Iterable[str]:
    exts = ["*.txt", "*.jsonl", "*.json", "*.gz", "*.parquet"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(str(Path(data_dir) / "**" / e), recursive=True))
    _skip_prefixes = ("test-", "validation-", "val-")
    files = [f for f in files if not any(Path(f).name.startswith(p) for p in _skip_prefixes)]
    files = sorted(files)[: max(1, int(max_files))]
    n = 0
    for fp in files:
        if fp.endswith(".json.gz") or (fp.endswith(".gz") and "c4" in Path(fp).name.lower()):
            for t in _iter_jsonl_texts_gz(fp):
                yield t
                n += 1
                if n >= max_lines:
                    return
            continue
        if fp.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq  # type: ignore

                tbl = pq.read_table(fp, columns=["text"])
                for txt in tbl["text"].to_pylist():
                    if isinstance(txt, str) and txt.strip():
                        yield txt.strip()
                        n += 1
                        if n >= max_lines:
                            return
            except Exception:
                pass
            continue
        import json

        opener = gzip.open if fp.endswith(".gz") else open
        try:
            with opener(fp, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = line
                    if line.startswith("{"):
                        try:
                            j = json.loads(line)
                            if isinstance(j, dict):
                                t = j.get("text") or j.get("content") or line
                        except Exception:
                            t = line
                    if isinstance(t, str) and t.strip():
                        yield t.strip()
                        n += 1
                        if n >= max_lines:
                            return
        except Exception:
            continue


def _build_token_stream(
    tok, texts_iter: Iterable[str], target_tokens: int, seed: int
) -> Optional[torch.Tensor]:
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


@torch.no_grad()
def _ortho_q_from_u(U: torch.Tensor) -> torch.Tensor:
    x = U.float()
    if x.shape[1] == 0:
        raise ValueError("empty U")
    q, _ = torch.linalg.qr(x, mode="reduced")
    return q


@torch.no_grad()
def _accum_residual_cov_one_layer(
    model,
    batches: List[torch.Tensor],
    Q: torch.Tensor,
    li: int,
    d_model: int,
    d_dev: torch.device,
    adtype: torch.dtype,
) -> torch.Tensor:
    """C = (1/N) sum r r^T 在 CPU, adtype。"""
    cmat = torch.zeros((d_model, d_model), dtype=adtype, device="cpu")
    n_rows = 0
    buf: Dict[int, torch.Tensor] = {}

    def _h(_m, _inp, out):
        t = out[0] if isinstance(out, (list, tuple)) else out
        buf[li] = t.detach()

    mod = model.model.decoder.layers[li].self_attn.v_proj
    hook = mod.register_forward_hook(_h)
    try:
        q_cpu = Q.float()
        for b in batches:
            buf.clear()
            ids = b.to(d_dev, non_blocking=True)
            _ = model.model(input_ids=ids, attention_mask=None, use_cache=False)
            v = buf.get(li)
            if v is None:
                continue
            if v.dim() != 3 or v.size(-1) != d_model:
                continue
            q = q_cpu.to(device=v.device, dtype=torch.float32)
            vf = v.float().reshape(-1, d_model)
            proj = (vf @ q) @ q.mT
            r = (vf - proj).cpu().to(adtype)
            cmat += r.mT @ r
            n_rows += int(r.shape[0])
            del ids, v, vf, proj, r
        hook.remove()
    except Exception:
        hook.remove()
        raise
    if n_rows > 0:
        cmat /= float(n_rows)
    return cmat


@torch.no_grad()
def _top_eigenvectors_sym(C: torch.Tensor, k: int) -> torch.Tensor:
    w, V = torch.linalg.eigh(C.float())
    k2 = min(k, V.shape[1])
    return V[:, -k2:].contiguous()  # 大特征值在末尾


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description="V-tail: residual w.r.t. main v_proj_basis, PCA of residual covariance"
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--main_basis_pt", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--out_json", default="", help="可选 report json；默认同目录 .json")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--tail_cols", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--token_budget", type=int, default=0)
    ap.add_argument("--max_data_files", type=int, default=300)
    ap.add_argument("--max_data_lines", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--layers", type=str, default="all", help="如 all 或 19-27,47")
    ap.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"]
    )
    ap.add_argument(
        "--accum_dtype", type=str, default="float64", choices=["float32", "float64"]
    )
    ap.add_argument(
        "--lowram",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=逐层一协方差(推荐,省RAM); 0=一遍前向多hook(多占 RAM)",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--local_files_only", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    lfo = bool(int(args.local_files_only))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    adtype = torch.float64 if args.accum_dtype == "float64" else torch.float32
    d_dev = torch.device(args.device)

    tok = AutoTokenizer.from_pretrained(
        (args.tokenizer or "").strip() or args.model, local_files_only=lfo, use_fast=False
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    sig = __import__("inspect").signature(torch.load)
    w_only = "weights_only" in sig.parameters
    main: Dict = torch.load(
        args.main_basis_pt,
        map_location="cpu",
        **({"weights_only": True} if w_only else {}),
    )
    if not isinstance(main, dict):
        raise TypeError("main_basis_pt 应为 state dict like")

    n_layers: Optional[int] = None
    for k in main.keys():
        if "decoder.layers" in k and "v_proj_basis" in k:
            p = k.split("decoder.layers.")[1].split(".")[0]
            n_layers = int(p) + 1 if n_layers is None else max(n_layers, int(p) + 1)
    if n_layers is None:
        n_layers = 48

    s = (args.layers or "all").strip().lower()
    if s in ("all", ""):
        layer_list = list(range(n_layers))
    else:
        layer_set = set()
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                for i in range(int(a), int(b) + 1):
                    if 0 <= i < 512:
                        layer_set.add(i)
            else:
                i = int(part)
                if 0 <= i < 512:
                    layer_set.add(i)
        layer_list = sorted(layer_set)

    target_toks = int(args.token_budget) if int(args.token_budget) > 0 else int(
        args.n_batches * args.seqlen * 2
    )
    stream = _build_token_stream(
        tok,
        _iter_texts_from_dir(args.data_dir, args.max_data_files, args.max_data_lines),
        target_tokens=max(500_000, target_toks),
        seed=int(args.seed),
    )
    if stream is None or stream.numel() < int(args.seqlen) + 2:
        raise RuntimeError("token stream 太短，请检查 data_dir")

    g = torch.Generator().manual_seed(int(args.seed))
    N = int(stream.numel())
    seqlen = int(args.seqlen)
    nb = int(args.n_batches)
    batches: List[torch.Tensor] = []
    for _ in range(nb):
        s0 = int(
            torch.randint(
                low=0, high=max(1, N - seqlen - 1), size=(1,), generator=g
            ).item()
        )
        batches.append(stream[s0 : s0 + seqlen].unsqueeze(0))

    print(
        f"[data] n_batches={nb} seqlen={seqlen} stream_N={N} | lowram={int(args.lowram)} "
        f"accum={args.accum_dtype}",
        flush=True,
    )

    print(
        f"[load] {args.model} | device={args.device} | dtype={args.dtype}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=lfo,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": str(d_dev)},
    )
    model.eval()
    d_model = int(model.config.hidden_size)
    n_heads = int(model.config.num_attention_heads)

    Q_by: Dict[int, torch.Tensor] = {}
    for li in layer_list:
        k = _basis_key(li)
        if k not in main:
            print(f"[warn] missing {k}, skip", flush=True)
            continue
        U = main[k].float().cpu()
        if U.shape[0] != d_model:
            print(f"[warn] layer {li} U shape {tuple(U.shape)} vs d={d_model}, skip", flush=True)
            continue
        if U.shape[1] % n_heads != 0:
            print(
                f"[warn] layer {li} r_v={U.shape[1]} not multiple of n_heads={n_heads}",
                flush=True,
            )
        Q_by[li] = _ortho_q_from_u(U)

    layer_list = sorted(Q_by.keys())
    if not layer_list:
        raise RuntimeError("no valid layers in main_basis_pt")

    tail_out: Dict[str, torch.Tensor] = {}
    meta: List[dict] = []
    tc = int(args.tail_cols)

    if int(args.lowram) == 1:
        for j, li in enumerate(layer_list):
            print(
                f"[layer] {li:02d} ({j + 1}/{len(layer_list)}) 累积残差协方差…",
                flush=True,
            )
            C = _accum_residual_cov_one_layer(
                model, batches, Q_by[li], li, d_model, d_dev, adtype
            )
            T = _top_eigenvectors_sym(C, tc)
            tail_out[_basis_key(li)] = T
            w, _V = torch.linalg.eigh(C.float())
            emax = float(w.max().item())
            emin = float(w[min(10, w.numel() - 1)].item()) if w.numel() else 0.0
            meta.append({"layer": li, "C_eigmax": emax, "C_trace_approx": "see_tensor"})
            del C, T, w, _V
            gc.collect()
    else:
        cov = {li: torch.zeros((d_model, d_model), dtype=adtype) for li in layer_list}
        buf: Dict[int, torch.Tensor] = {}

        def _mk_hook(layer_idx: int):
            def _h(_m, _inp, out):
                t = out[0] if isinstance(out, (list, tuple)) else out
                buf[layer_idx] = t.detach()

            return _h

        hooks = []
        for li in layer_list:
            h = model.model.decoder.layers[li].self_attn.v_proj.register_forward_hook(
                _mk_hook(li)
            )
            hooks.append(h)
        n_tok = 0
        for b in batches:
            buf.clear()
            ids = b.to(d_dev, non_blocking=True)
            _ = model.model(input_ids=ids, attention_mask=None, use_cache=False)
            for li in layer_list:
                v = buf.get(li)
                if v is None or v.dim() != 3 or v.size(-1) != d_model:
                    continue
                q = Q_by[li].to(device=v.device, dtype=torch.float32)
                vf = v.float().reshape(-1, d_model)
                proj = (vf @ q) @ q.mT
                r = (vf - proj).cpu().to(adtype)
                cov[li] += r.mT @ r
            n_tok += int(ids.size(0) * ids.size(1))
        for h in hooks:
            h.remove()
        nt = max(1, n_tok)
        for li in layer_list:
            cov[li] /= float(nt)
            T = _top_eigenvectors_sym(cov[li], tc)
            tail_out[_basis_key(li)] = T
            del T

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tail_out, out_path)
    jpath = (args.out_json or "").strip()
    if not jpath:
        jpath = str(out_path.with_suffix(".json"))
    rep = {
        "description": (
            "Tail V basis: top eigenvectors of Cov(r), r = v - Q Q^T v on full-model v_proj outputs; "
            "Q is orthonormalized main v_proj_basis U_v (same column space). "
            "Columns live in span(U_v)^perp (numerically). File does NOT contain main U_v."
        ),
        "model": args.model,
        "main_basis_pt": args.main_basis_pt,
        "out_path": str(out_path),
        "n_batches": nb,
        "seqlen": seqlen,
        "tail_cols": tc,
        "layers": layer_list,
        "lowram": int(args.lowram),
        "meta": meta,
    }
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_path}", flush=True)
    print(f"[report] {jpath}", flush=True)


if __name__ == "__main__":
    main()
