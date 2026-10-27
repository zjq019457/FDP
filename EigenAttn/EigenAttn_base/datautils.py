"""Offline data loaders for EigenAttn calibration and PPL evaluation.

All loaders are *strictly* offline:
  * WikiText-2: local parquet files (HF datasets `wikitext`/wikitext-2-raw-v1 layout).
  * C4: local ``.json.gz`` shards (allenai/c4 layout: one JSON per line with ``text`` key,
        but our reader also tolerates concatenated objects without newlines, which is what
        c4-validation shards in this workspace look like).
  * Tokenizers: ``local_files_only=True``; OPT/Llama default to ``use_fast=False``,
                MPT to ``use_fast=True`` (matches existing behavior in the project).

Public API: ``get_loaders(name, ...)`` (compat with the original signature) and the
new ``build_utkarsh_valenc_from_c4(...)`` used by :mod:`eval.eval_ppl` for the
paper-aligned C4 PPL evaluation (256 chunks, ``random.seed(0)``).
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import random
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Project-wide default offline locations (override via CLI/env if needed).
# Keep paths absolute and explicit so Hub access is never attempted.
# ---------------------------------------------------------------------------
DEFAULT_WIKITEXT2_DIR = os.environ.get(
    "WIKITEXT2_DATA_DIR", "/home/bai-project/jiaqizhang/wikitext2"
)
DEFAULT_C4_DIR = os.environ.get(
    "C4_DATA_DIR", "/diskE/jiaqizhang/c4-en/en"
)
DEFAULT_C4_TRAIN_GLOB = os.environ.get("C4_TRAIN_GLOB", "c4-train.*.json.gz")
DEFAULT_C4_VAL_GLOB = os.environ.get("C4_VAL_GLOB", "c4-validation.*.json.gz")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def _autodetect_use_fast(model: str) -> bool:
    return "mpt" in model.lower()


def _load_tokenizer(model: str):
    use_fast = _autodetect_use_fast(model)
    kw = dict(local_files_only=True, trust_remote_code=True)
    try:
        return AutoTokenizer.from_pretrained(model, use_fast=use_fast, legacy=False, **kw)
    except (TypeError, ValueError) as e:
        msg = str(e)
        if not use_fast and (
            "Tokenizer class" in msg or "does not exist" in msg or "not currently imported" in msg
        ):
            return AutoTokenizer.from_pretrained(model, use_fast=True, **kw)
        if "legacy" in msg:
            return AutoTokenizer.from_pretrained(model, use_fast=use_fast, **kw)
        raise


# ---------------------------------------------------------------------------
# WikiText-2: read parquet, tokenize once, return testenc as [1, T] int64 tensor.
# ---------------------------------------------------------------------------
def _read_wikitext2_parquet(split: str, data_dir: str) -> List[str]:
    path = Path(data_dir) / f"{split}-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"[wikitext2] missing local parquet: {path}; set WIKITEXT2_DATA_DIR or pass --wikitext2_dir"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "[wikitext2] reading parquet requires `pyarrow`; install via `pip install pyarrow`"
        ) from e
    table = pq.read_table(str(path), columns=["text"])
    return table.column("text").to_pylist()


def get_wikitext2(
    nsamples: int,
    seed: int,
    seqlen: int,
    model: str,
    data_dir: str = DEFAULT_WIKITEXT2_DIR,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], "torch.utils.data.Dataset"]:
    print("get_wikitext2")
    train_texts = _read_wikitext2_parquet("train", data_dir)
    test_texts = _read_wikitext2_parquet("test", data_dir)
    print(
        f"[wikitext2] OFFLINE/local parquet: train_lines={len(train_texts)}, "
        f"test_lines={len(test_texts)}, dir={data_dir}"
    )

    tokenizer = _load_tokenizer(model)
    trainenc = tokenizer("\n\n".join(train_texts), return_tensors="pt")
    testenc = tokenizer("\n\n".join(test_texts), return_tensors="pt")

    random.seed(seed)
    trainloader: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# ---------------------------------------------------------------------------
# C4 streaming reader (handles JSONL.gz with or without newlines / record-sep).
# Borrowed from sanity_AB_opt30b.py to keep the same byte-for-byte parsing.
# ---------------------------------------------------------------------------
def _iter_jsonl_texts_gz(
    fp: str, text_key: str = "text", read_size: int = 1 << 20
) -> Iterator[str]:
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


def _resolve_c4_files(data_dir: str, glob_pat: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, glob_pat)))
    if not files:
        raise FileNotFoundError(f"[c4] no files matched {glob_pat} under {data_dir}")
    return files


def build_doc_pool_from_c4_dir(
    data_dir: str,
    glob_pat: str,
    pool_docs: int,
    scan_lines: int,
    seed: int,
) -> Tuple[List[str], dict]:
    """Reservoir-sample ``pool_docs`` docs out of the first ``scan_lines`` docs of
    matching C4 shards. Stable across runs given the same seed."""
    files = _resolve_c4_files(data_dir, glob_pat)
    rng = random.Random(seed)
    reservoir: List[str] = []
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
        raise RuntimeError(f"[c4] no text decoded from {files[0]}")
    return reservoir, {"seen_docs": seen, "pool_docs": len(reservoir), "file": files[0]}


@torch.no_grad()
def build_utkarsh_valenc_from_pool(
    tok,
    doc_pool: Sequence[str],
    seqlen: int,
    n_chunks: int = 256,
    val_seed: int = 0,
    max_tries_per_chunk: int = 2000,
) -> torch.Tensor:
    """Mirror Utkarsh ``get_c4()`` validation construction:
    ``random.seed(0)`` -> 256 random windows of length ``seqlen`` from random docs.
    Returns ``[1, n_chunks*seqlen]`` LongTensor.
    """
    rng = random.Random(val_seed)
    valenc: List[torch.Tensor] = []
    for _ in range(n_chunks):
        ok = False
        for _try in range(max_tries_per_chunk):
            text = doc_pool[rng.randint(0, len(doc_pool) - 1)]
            tmp = tok(text, return_tensors="pt")
            L = tmp.input_ids.shape[1]
            if L > seqlen:
                i = rng.randint(0, L - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"[c4] cannot sample doc with token_len > {seqlen} after {max_tries_per_chunk} tries; "
                f"increase pool_docs / scan_lines"
            )
    return torch.hstack(valenc)


def build_utkarsh_valenc_from_c4(
    tokenizer,
    seqlen: int,
    *,
    data_dir: str = DEFAULT_C4_DIR,
    glob_pat: str = DEFAULT_C4_VAL_GLOB,
    pool_docs: int = 500_000,
    scan_lines: int = 500_000,
    n_chunks: int = 256,
    seed: int = 2,
    val_seed: int = 0,
) -> Tuple[torch.Tensor, dict]:
    """One-shot helper used by ``eval/eval_ppl``: returns ``[1, n_chunks*seqlen]`` testenc and pool info."""
    pool, info = build_doc_pool_from_c4_dir(
        data_dir=data_dir,
        glob_pat=glob_pat,
        pool_docs=pool_docs,
        scan_lines=scan_lines,
        seed=seed,
    )
    valenc = build_utkarsh_valenc_from_pool(
        tokenizer, pool, seqlen=seqlen, n_chunks=n_chunks, val_seed=val_seed
    )
    return valenc, info


# ---------------------------------------------------------------------------
# C4 calibration loader (random ``seqlen`` windows from random long docs in train shards).
# ---------------------------------------------------------------------------
def get_c4_local(
    nsamples: int,
    seed: int,
    seqlen: int,
    model: str,
    *,
    data_dir: str = DEFAULT_C4_DIR,
    train_glob: str = DEFAULT_C4_TRAIN_GLOB,
    val_glob: str = DEFAULT_C4_VAL_GLOB,
    pool_docs: int = 200_000,
    scan_lines: int = 200_000,
    n_val_chunks: int = 256,
    val_seed: int = 0,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Streaming-friendly C4 calibration loader (no Hub).

    * trainloader: ``nsamples`` random ``seqlen`` windows from long train docs.
    * valenc: Utkarsh-style 256-chunk validation testenc.
    """
    print(f"get_c4_local data_dir={data_dir} train_glob={train_glob} val_glob={val_glob}")
    tokenizer = _load_tokenizer(model)

    train_pool, train_info = build_doc_pool_from_c4_dir(
        data_dir=data_dir, glob_pat=train_glob, pool_docs=pool_docs, scan_lines=scan_lines, seed=seed
    )
    print(f"[c4] train pool: {train_info}")

    rng = random.Random(seed)
    trainloader: List[Tuple[torch.Tensor, torch.Tensor]] = []
    tries_total = 0
    while len(trainloader) < nsamples and tries_total < 10_000:
        text = train_pool[rng.randint(0, len(train_pool) - 1)]
        enc = tokenizer(text, return_tensors="pt")
        if enc.input_ids.shape[1] > seqlen:
            i = rng.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = enc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        tries_total += 1
    if len(trainloader) < nsamples:
        raise RuntimeError(
            f"[c4] only got {len(trainloader)}/{nsamples} calib chunks; "
            f"increase pool_docs / scan_lines or use a different seed"
        )

    val_pool, val_info = build_doc_pool_from_c4_dir(
        data_dir=data_dir, glob_pat=val_glob, pool_docs=pool_docs, scan_lines=scan_lines, seed=seed
    )
    print(f"[c4] val pool: {val_info}")
    valenc = build_utkarsh_valenc_from_pool(
        tokenizer, val_pool, seqlen=seqlen, n_chunks=n_val_chunks, val_seed=val_seed
    )
    return trainloader, valenc


# ---------------------------------------------------------------------------
# Synthetic offline calibration (last-resort fallback).
# ---------------------------------------------------------------------------
def build_synthetic_calib_dataloader(
    model: str, nsamples: int, seqlen: int, seed: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    random.seed(seed)
    torch.manual_seed(seed)
    tok = _load_tokenizer(model)
    vocab = tok.vocab_size
    trainloader: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(nsamples):
        inp = torch.randint(low=0, high=vocab, size=(1, seqlen), dtype=torch.long)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


# ---------------------------------------------------------------------------
# Unified entrypoint expected by main_eigen_attn / eigenattn.
# ---------------------------------------------------------------------------
def get_loaders(
    name: str,
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    model: str = "",
    *,
    wikitext2_dir: str = DEFAULT_WIKITEXT2_DIR,
    c4_data_dir: str = DEFAULT_C4_DIR,
    c4_train_glob: str = DEFAULT_C4_TRAIN_GLOB,
    c4_val_glob: str = DEFAULT_C4_VAL_GLOB,
    c4_pool_docs: int = 200_000,
    c4_scan_lines: int = 200_000,
    c4_n_val_chunks: int = 256,
    c4_val_seed: int = 0,
    # Back-compat shims: original signature accepted these kwargs.
    c4_train_file: Optional[str] = None,
    c4_val_file: Optional[str] = None,
):
    name_lower = name.lower()
    if "wikitext2" in name_lower:
        return get_wikitext2(nsamples, seed, seqlen, model, data_dir=wikitext2_dir)
    if "c4" in name_lower:
        # If caller pinned a single train shard via ``c4_train_file``, honour it as a 1-file glob.
        if c4_train_file:
            tdir = os.path.dirname(os.path.abspath(c4_train_file)) or "."
            tglob = os.path.basename(c4_train_file)
            vdir = c4_data_dir if not c4_val_file else (os.path.dirname(os.path.abspath(c4_val_file)) or ".")
            vglob = c4_val_glob if not c4_val_file else os.path.basename(c4_val_file)
            return get_c4_local(
                nsamples, seed, seqlen, model,
                data_dir=tdir, train_glob=tglob,
                pool_docs=c4_pool_docs, scan_lines=c4_scan_lines,
                n_val_chunks=c4_n_val_chunks, val_seed=c4_val_seed,
                val_glob=vglob,
            ) if tdir == vdir else get_c4_local(
                nsamples, seed, seqlen, model,
                data_dir=tdir, train_glob=tglob,
                pool_docs=c4_pool_docs, scan_lines=c4_scan_lines,
                n_val_chunks=c4_n_val_chunks, val_seed=c4_val_seed,
                val_glob=DEFAULT_C4_VAL_GLOB,
            )
        return get_c4_local(
            nsamples, seed, seqlen, model,
            data_dir=c4_data_dir, train_glob=c4_train_glob, val_glob=c4_val_glob,
            pool_docs=c4_pool_docs, scan_lines=c4_scan_lines,
            n_val_chunks=c4_n_val_chunks, val_seed=c4_val_seed,
        )
    if "mix" in name_lower:
        wiki_train, _ = get_wikitext2(nsamples // 2, seed, seqlen, model, data_dir=wikitext2_dir)
        c4_train, _ = get_c4_local(
            nsamples // 2, seed, seqlen, model,
            data_dir=c4_data_dir, train_glob=c4_train_glob, val_glob=c4_val_glob,
            pool_docs=c4_pool_docs, scan_lines=c4_scan_lines,
            n_val_chunks=c4_n_val_chunks, val_seed=c4_val_seed,
        )
        return wiki_train + c4_train, None
    raise ValueError(
        f"Unsupported dataset name {name!r}; choose from {{wikitext2, c4, mix}} (offline)."
    )


__all__ = [
    "DEFAULT_WIKITEXT2_DIR",
    "DEFAULT_C4_DIR",
    "DEFAULT_C4_TRAIN_GLOB",
    "DEFAULT_C4_VAL_GLOB",
    "set_seed",
    "get_loaders",
    "get_wikitext2",
    "get_c4_local",
    "build_doc_pool_from_c4_dir",
    "build_utkarsh_valenc_from_pool",
    "build_utkarsh_valenc_from_c4",
    "build_synthetic_calib_dataloader",
]
