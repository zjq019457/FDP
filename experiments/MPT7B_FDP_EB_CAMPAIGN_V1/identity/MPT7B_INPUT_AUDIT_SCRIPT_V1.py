#!/usr/bin/env python3
"""Create a read-only, content-addressed MPT evaluation-input manifest.

This prepares only dataset/tokenizer input identities. It does not load model
weights, run inference, or authorize a quality job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_sha256(tensor) -> str:
    import torch

    return hashlib.sha256(tensor.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wikitext2", required=True)
    ap.add_argument("--c4-texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method-lock", required=True)
    ap.add_argument("--accuracy-dir", required=True)
    args = ap.parse_args()

    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    repo = Path(args.repo).resolve()
    model = Path(args.model).resolve()
    wiki = Path(args.wikitext2).resolve()
    c4 = Path(args.c4_texts).resolve()
    acc = Path(args.accuracy_dir).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, "/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/runners")

    from transformers import AutoTokenizer
    import datautils
    from ppl_instrumented import _load_c4_windows

    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    datautils._load_tokenizer = lambda _model: tokenizer
    _, testenc = datautils.get_wikitext2(
        nsamples=128, seed=2, seqlen=2048, model=str(model), data_dir=str(wiki)
    )
    test_ids = testenc.input_ids if hasattr(testenc, "input_ids") else testenc
    n_wiki = int(test_ids.numel() // 2048)
    wiki_windows = [
        {
            "window_id": f"wiki_{i:05d}",
            "source_record_id": f"wikitext2_test_chunk_{i}",
            "input_ids_sha256": tensor_sha256(test_ids[:, i * 2048 : (i + 1) * 2048]),
        }
        for i in range(n_wiki)
    ]
    c4_windows = _load_c4_windows(c4, tokenizer, 2048)
    c4_manifest_windows = [
        {
            "window_id": w["window_id"],
            "source_record_id": w["source_record_id"],
            "input_ids_sha256": tensor_sha256(w["input_ids"]),
        }
        for w in c4_windows
    ]

    method_lock = Path(args.method_lock).resolve()
    method_lock_hash = sha256_file(method_lock)
    task_names = [
        "arc_challenge_localjsonl",
        "arc_easy_localjsonl",
        "hellaswag_localjsonl",
        "piqa_localjsonl",
        "winogrande_localjsonl",
    ]
    task_yamls = {
        f"{name}.yaml": sha256_file(acc / f"{name}.yaml") for name in task_names
    }
    source_files = {
        "config.json": sha256_file(model / "config.json"),
        "tokenizer.json": sha256_file(model / "tokenizer.json"),
        "tokenizer_config.json": sha256_file(model / "tokenizer_config.json"),
        "wikitext2_train_parquet": sha256_file(wiki / "train-00000-of-00001.parquet"),
        "wikitext2_test_parquet": sha256_file(wiki / "test-00000-of-00001.parquet"),
        "c4_selected_texts_jsonl": sha256_file(c4),
    }
    manifest = {
        "manifest_id": "MPT7B_EVAL_INPUT_BINDING_V1",
        "status": "INPUTS_BOUND__SCORER_ADAPTER_AND_RESOURCE_LOCK_PENDING",
        "model_content_fingerprint": "3e6db8661db48e835566a784d5ef1d1495a38a629cc3f2f0207364e07c33270f",
        "method_authority": {
            "path": str(method_lock),
            "sha256": method_lock_hash,
            "components": ["d1", "d2", "d5"],
            "allocation": "TRUE_TWO_PASS_FDP_EB",
            "lambda": 1.2,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "use_fast": True,
            "trust_remote_code": False,
            "vocab_size": int(tokenizer.vocab_size),
            "len_tokenizer": int(len(tokenizer)),
            "eos_token_id": int(tokenizer.eos_token_id),
            "pad_token_id_before_eval_normalization": tokenizer.pad_token_id,
        },
        "ppl_protocol": {
            "sequence_length": 2048,
            "wiki_split": "test",
            "wiki_seed": 2,
            "wiki_window_count": n_wiki,
            "c4_source": str(c4),
            "c4_window_count": len(c4_windows),
            "c4_tokenizer_specific_padding": "truncate to 2048; right-pad eos if shorter",
            "legacy_scorer": "ppl_instrumented.eval_windows_legacy_llama",
            "wiki_cross_entropy": "float32 logits CE",
            "c4_cross_entropy": "native logits dtype CE",
            "aggregate": "exp(sum(window NLL)/sum(window token_count))",
            "wiki_windows": wiki_windows,
            "c4_windows": c4_manifest_windows,
        },
        "accuracy_protocol": {
            "suite": task_names,
            "num_fewshot": 0,
            "include_path": str(acc),
            "suite_total_samples_from_frozen_Llama3_protocol": 16695,
            "task_yaml_sha256": task_yamls,
        },
        "source_file_sha256": source_files,
        "notes": [
            "This manifest binds inputs only; it is not a model quality result or cell launch authorization.",
            "MPT evaluation must load the official MPT attention implementation; do not use the legacy runner's Llama fallback for compressed models.",
            "The local MPT tokenizer is loaded with trust_remote_code=False; input IDs are model-tokenizer specific.",
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out.with_suffix(out.suffix + ".sha256")).write_text(
        f"{sha256_file(out)}  {out.name}\n", encoding="ascii"
    )
    print(json.dumps({
        "output": str(out),
        "file_sha256": sha256_file(out),
        "manifest_sha256": manifest["manifest_sha256"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_len": int(len(tokenizer)),
        "wiki_windows": n_wiki,
        "c4_windows": len(c4_windows),
        "wiki_id_digest": canonical_sha256(wiki_windows),
        "c4_id_digest": canonical_sha256(c4_manifest_windows),
    }, indent=2))


if __name__ == "__main__":
    main()
