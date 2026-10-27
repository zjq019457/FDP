"""Run the official PaLu rank search/decomposition with frozen local manifests.

This wrapper only replaces the unavailable Hugging Face dataset transport with
the exact raw WikiText-2 and C4 files stored under the new experiment root.
The PaLu rank-search and whiten-decomposition code is imported unchanged from
the pinned official repository.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path("/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2")
REPO = ROOT / "official" / "Palu"
DATA = ROOT / "official" / "data"
WIKI_TRAIN = DATA / "wiki.train.raw"
WIKI_TEST = DATA / "wiki.test.raw"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def wiki_join(path: Path) -> str:
    # The HF parquet extraction already materializes the official
    # "\n\n".join(testdata["text"]) string.  Do not split/rejoin it again:
    # parquet text rows retain their source line endings.
    if path.name.endswith(".hf_joined.raw"):
        return path.read_text(encoding="utf-8")
    return "\n\n".join(path.read_text(encoding="utf-8").splitlines())


def c4_first_docs(path: Path, n: int = 1100) -> list[str]:
    docs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if len(docs) == n:
                break
    if len(docs) != n:
        raise RuntimeError(f"C4 file has only {len(docs)} docs; expected {n}")
    return docs


def build_calibration_manifest(nsamples: int, seqlen: int, seed: int = 3) -> dict:
    text = wiki_join(WIKI_TRAIN)
    rng = random.Random(seed)
    indices = [rng.randint(0, len(text) - seqlen - 1) for _ in range(nsamples)]
    return {
        "dataset": "wikitext2",
        "source_file": str(WIKI_TRAIN),
        "source_sha256": sha256(WIKI_TRAIN),
        "joiner": "\\n\\n",
        "seed": seed,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "char_start_indices": indices,
    }


def offline_get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    if name != "wikitext2":
        raise NotImplementedError("The official reproduction freezes WikiText-2 calibration")
    text = wiki_join(WIKI_TRAIN)
    rng = random.Random(seed)
    samples = []
    for _ in range(nsamples):
        i = rng.randint(0, len(text) - seqlen - 1)
        enc = tokenizer(text[i : i + seqlen * 10], return_tensors="pt")
        inp = enc.input_ids[:, :seqlen]
        if inp.shape[1] != seqlen:
            raise RuntimeError(f"short calibration sample at char offset {i}: {inp.shape}")
        samples.append({"input_ids": inp, "attention_mask": torch.ones_like(inp)})
    return samples


def offline_get_ppl_loader(name, tokenizer, seqlen=2048):
    if name == "wikitext2":
        ids = tokenizer(wiki_join(WIKI_TEST), return_tensors="pt").input_ids
        return SimpleNamespace(input_ids=ids)
    if name == "c4":
        docs = c4_first_docs(DATA / "c4-validation.00000-of-00008.json.gz", 1100)
        ids = tokenizer(" ".join(docs), return_tensors="pt").input_ids
        return SimpleNamespace(input_ids=ids[:, : 256 * seqlen])
    raise NotImplementedError(name)


def patch_offline_data():
    # Imports are intentionally after PYTHONPATH is set by the launcher.
    import palu.decomposition as decomposition
    import palu.rank_search as rank_search
    import run_ppl_eval as ppl

    rank_search.get_calib_data = offline_get_calib_data
    decomposition.get_calib_data = offline_get_calib_data
    ppl.get_ppl_eval_loaders = offline_get_ppl_loader

    # PyTorch 2.6+ defaults torch.load to weights_only=True, while the
    # official script caches a SimpleNamespace/BatchEncoding loader.
    real_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return real_load(*args, **kwargs)

    torch.load = trusted_load
    return rank_search, decomposition, ppl


def main() -> None:
    global WIKI_TRAIN, WIKI_TEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["control", "fixed"], required=True)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--wiki-train", default=str(DATA / "wiki.train.raw"))
    parser.add_argument("--wiki-test", default=str(DATA / "wiki.test.raw"))
    args = parser.parse_args()

    if args.mode == "fixed" and args.ratio not in {0.3, 0.5, 0.7}:
        raise ValueError("fixed ratio must be one of 0.3, 0.5, 0.7")

    WIKI_TRAIN = Path(args.wiki_train)
    WIKI_TEST = Path(args.wiki_test)

    from utils import load_model_and_tokenizer, set_seed

    rank_search, decomposition, ppl = patch_offline_data()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    protocol = {
        "model": str(ROOT / "models" / "Llama-2-7b-hf"),
        "mode": args.mode,
        "param_ratio_target": args.ratio,
        "compression_rate_target": None if args.ratio is None else 1.0 - args.ratio,
        "calib_dataset": "wikitext2",
        "calib_seqlen": 1024,
        "fisher_samples": 2048,
        "whiten_samples": 256,
        "eval_seqlen": 2048,
        "c4_docs_scanned": 1100,
        "c4_chunks": 256,
        "c4_source_sha256": sha256(DATA / "c4-validation.00000-of-00008.json.gz"),
        "wiki_train_sha256": sha256(WIKI_TRAIN),
        "wiki_test_sha256": sha256(WIKI_TEST),
    }
    save_json(out / "protocol.json", protocol)
    save_json(out / "calibration_manifest.json", build_calibration_manifest(2048, 1024, 3))

    model, tokenizer = load_model_and_tokenizer(protocol["model"])
    # The official evaluator's cache key is based only on model_name.  Give
    # every new manifest/output lane a distinct namespace so cached tokenized
    # loaders cannot cross-contaminate diagnostic and formal runs.
    eval_model_name = protocol["model"] + "__" + out.name
    if args.mode == "control":
        results = ppl.eval_ppl(model, tokenizer, eval_model_name, "wikitext2,c4", 2048, args.device)
        save_json(out / "quality.json", {"protocol": protocol, "ppl": results})
        return

    search_args = SimpleNamespace(
        search_method="fisher_uniform",
        calib_dataset="wikitext2",
        calib_seqlen=1024,
        model_id=protocol["model"],
        head_group_size=4,
        param_ratio_target=args.ratio,
        device=args.device,
        use_cache=args.use_cache,
        decompose_method="whiten",
        n_fisher_calib_samples=2048,
        n_whiten_calib_samples=256,
    )
    selection, rank_sum, total_rank = rank_search.rank_search(model, tokenizer, search_args)
    save_json(
        out / "rank_schedule.json",
        {
            "protocol": protocol,
            "rank_sum": rank_sum,
            "total_rank": total_rank,
            "realized_ratio": rank_sum / total_rank,
            "realized_compression": 1.0 - rank_sum / total_rank,
            "selection": selection,
        },
    )
    decomposition.compress_model(model, tokenizer, search_args, args.device, selection)
    results = ppl.eval_ppl(model, tokenizer, eval_model_name, "wikitext2,c4", 2048, args.device)
    save_json(out / "quality.json", {"protocol": protocol, "ppl": results})


if __name__ == "__main__":
    main()
