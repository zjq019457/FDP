#!/usr/bin/env python3
"""Run the pinned PaLu implementation on the frozen offline protocol.

This runner intentionally calls the official rank_search/decomposition path
from the pinned repository.  It keeps the compressed model in memory, so the
new branch does not materialize six large Hugging Face checkpoints on disk.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.machinery
import json
import math
import os
import random
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


REPO = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu")
BRANCH = Path("/diskE/jiaqizhang/PALU_OFFICIAL_PROTOCOL_REPRO_V1")
MODEL = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf")
DATA = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/data")
SEQ = 2048
DTYPE_BYTES = 2


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def install_optional_hadamard_fallback() -> None:
    """Avoid importing the unavailable CUDA extension for non-quantized runs.

    The official source imports fast_hadamard_transform at module import time,
    although the FP16 non-quantized decomposition path never calls it.  This
    fallback is only an engineering dependency repair; quantized experiments
    are not run by this pipeline.
    """
    try:
        import fast_hadamard_transform  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    mod = types.ModuleType("fast_hadamard_transform")
    mod.__spec__ = importlib.machinery.ModuleSpec("fast_hadamard_transform", loader=None)

    def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        n = x.shape[-1]
        if n <= 0 or n & (n - 1):
            raise ValueError(f"fallback Hadamard requires power-of-two last dim, got {n}")
        y = x
        width = 1
        while width < n:
            y = y.reshape(*y.shape[:-1], n // (2 * width), 2, width)
            a = y[..., 0, :].clone()
            b = y[..., 1, :].clone()
            y = torch.stack((a + b, a - b), dim=-2).reshape(*y.shape[:-3], n)
            width *= 2
        return y * scale

    mod.hadamard_transform = hadamard_transform
    sys.modules["fast_hadamard_transform"] = mod


def load_official_imports():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "3rdparty" / "fast-hadamard-transform"))
    install_optional_hadamard_fallback()
    from utils import load_model_and_tokenizer  # type: ignore
    from palu.decomposition import compress_model  # type: ignore
    from palu.rank_search import rank_search  # type: ignore

    # The pinned official code calls datasets.load_dataset() for the
    # whitening calibration even when its own whitening cache is present.
    # The official raw WikiText-2 training export is already frozen locally;
    # patch only this data-loader edge so the official sampler/tokenization
    # remains unchanged in offline execution.
    import palu.data_utils as data_utils  # type: ignore
    import palu.decomposition as decomposition  # type: ignore
    import palu.rank_search as rank_search_module  # type: ignore

    def local_get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
        if name != "wikitext2":
            return data_utils.get_calib_data(name, tokenizer, model_id, nsamples, seqlen, seed)
        cache_file = BRANCH / "cache" / f"{name}_{model_id.replace('/', '_')}_{nsamples}_{seqlen}_{seed}.pt"
        if cache_file.exists():
            return torch.load(cache_file, map_location="cpu")
        raw_path = DATA / "wiki-train.hf_joined.raw"
        tot_text = raw_path.read_text(encoding="utf-8")
        random.seed(seed)
        traindataset = []
        for _ in range(nsamples):
            i = random.randint(0, len(tot_text) - seqlen - 1)
            j = i + seqlen * 10
            trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
            inp = trainenc.input_ids[:, :seqlen]
            attention_mask = torch.ones_like(inp)
            traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(traindataset, cache_file)
        return traindataset

    data_utils.get_calib_data = local_get_calib_data
    decomposition.get_calib_data = local_get_calib_data
    rank_search_module.get_calib_data = local_get_calib_data

    return load_model_and_tokenizer, compress_model, rank_search


def load_protocol_ids(tokenizer):
    wiki_text = (DATA / "wiki-test.hf_joined.raw").read_text(encoding="utf-8")
    wiki_ids = tokenizer(wiki_text, return_tensors="pt").input_ids
    docs = []
    with gzip.open(DATA / "c4-validation.00000-of-00008.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if len(docs) == 1100:
                break
    if len(docs) != 1100:
        raise RuntimeError(f"expected 1100 C4 docs, got {len(docs)}")
    c4_ids = tokenizer(" ".join(docs), return_tensors="pt").input_ids[:, : 256 * SEQ]
    return wiki_ids, c4_ids


@torch.no_grad()
def eval_loader(model, ids):
    device = next(model.parameters()).device
    n = ids.numel() // SEQ
    nlls = []
    for i in range(n):
        batch = ids[:, i * SEQ : (i + 1) * SEQ].to(device)
        position_ids = torch.arange(batch.shape[1], device=device).unsqueeze(0)
        out = model.model(batch, position_ids=position_ids)
        logits = model.lm_head(out[0])
        loss = nn.CrossEntropyLoss()(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
        )
        value = float((loss.float() * SEQ).item())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite NLL at chunk {i}")
        nlls.append(value)
    tokens = len(nlls) * SEQ
    mean_nll = sum(nlls) / tokens
    return {
        "chunks": len(nlls),
        "tokens": tokens,
        "summed_nll": sum(nlls),
        "mean_token_nll": mean_nll,
        "ppl": math.exp(mean_nll),
    }


def flatten_selection(selection: dict) -> tuple[int, dict[int, int]]:
    ranks = [int(x) for values in selection.values() for x in values]
    return sum(ranks), {len(values): values[0] for values in selection.values() if values}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, choices=[0.3, 0.5, 0.7, 1.0], required=True)
    ap.add_argument("--search_method", choices=["uniform", "fisher_uniform"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="official")
    ap.add_argument("--rebuild-fisher-cache", action="store_true")
    args = ap.parse_args()

    BRANCH.mkdir(parents=True, exist_ok=True)
    os.chdir(BRANCH)
    load_model_and_tokenizer, compress_model, rank_search = load_official_imports()

    class Args:
        model_id = str(MODEL)
        seed = 42
        device = args.device
        # Rebuild only the corrupt Fisher artifact.  The official whitening
        # cache remains read-only and is still used by compression.
        use_cache = not args.rebuild_fisher_cache
        n_fisher_calib_samples = 2048
        n_whiten_calib_samples = 256
        calib_dataset = "wikitext2"
        calib_seqlen = 1024
        head_group_size = 4
        param_ratio_target = args.ratio
        search_method = args.search_method
        decompose_method = "whiten"

    cfg = Args()
    load_model_and_tokenizer_fn = load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer_fn(str(MODEL))
    model.to(torch.device(args.device))
    selection, rank_sum, total_rank = rank_search(model, tokenizer, cfg)
    if args.rebuild_fisher_cache:
        cfg.use_cache = True
    compress_model(model, tokenizer, cfg, args.device, selection)
    model.eval()
    wiki_ids, c4_ids = load_protocol_ids(tokenizer)
    wiki = eval_loader(model, wiki_ids)
    c4 = eval_loader(model, c4_ids)
    realized = sum(sum(int(x) for x in v) for v in selection.values())
    resource_bytes = realized * SEQ * DTYPE_BYTES
    result = {
        "status": "OFFICIAL_PATH_QUALITY_COMPLETE",
        "campaign": "PALU_OFFICIAL_PROTOCOL_REPRO_V1",
        "tag": args.tag,
        "ratio_target": args.ratio,
        "compression_fraction": 1.0 - args.ratio,
        "search_method": args.search_method,
        "decompose_method": "whiten",
        "head_group_size": 4,
        "official_repo": str(REPO),
        "official_repo_commit": "bb22666e2ef96707e8dd21d93fc00146c2e0d615",
        "model": str(MODEL),
        "protocol": {
            "sequence_length": SEQ,
            "wiki_source_sha256": sha(DATA / "wiki-test.hf_joined.raw"),
            "c4_source_sha256": sha(DATA / "c4-validation.00000-of-00008.json.gz"),
            "c4_docs_scanned": 1100,
            "c4_chunks": 256,
            "fisher_samples": 2048,
            "fisher_seqlen": 1024,
            "whiten_samples": 256,
        },
        "selection": selection,
        "total_rank": realized,
        "total_rank_reported_by_search": int(rank_sum),
        "total_uncompressed_rank": int(total_rank),
        "prefill_cache_bytes": resource_bytes,
        "decode_cache_bytes": resource_bytes,
        "wiki": wiki,
        "c4": c4,
        "source_dependency_repair": "pure-Python fast_hadamard_transform fallback; unused by FP16 whiten path",
        "historical_results_modified": False,
        "old_anchor_schema_results_modified": False,
    }
    out = BRANCH / "results" / f"{args.tag}_ratio_{args.ratio:.1f}_{args.search_method}.json"
    write_json(out, result)
    print(json.dumps({"out": str(out), "wiki_ppl": wiki["ppl"], "c4_ppl": c4["ppl"], "rank": realized}, indent=2))


if __name__ == "__main__":
    main()
