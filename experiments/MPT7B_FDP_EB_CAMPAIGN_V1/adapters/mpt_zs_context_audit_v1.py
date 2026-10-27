"""Input-only token-length audit for the frozen MPT 5-task zero-shot suite.

This does not load model weights, run inference, score answers, or alter the
frozen tasks. Dataset cache is temporary under /dev/shm and is removed by this
process when complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

TASKS = [
    "arc_challenge_localjsonl", "arc_easy_localjsonl", "hellaswag_localjsonl",
    "piqa_localjsonl", "winogrande_localjsonl",
]
YAML_SHAS = {
    "arc_challenge_localjsonl.yaml": "4f8304d6cd638072c8699a823dd8b71ead8ff820f1fcedc2215799a7ee41190d",
    "arc_easy_localjsonl.yaml": "28e5f31577e10a8973bfc5243262f50bd5a16ca5e29358a6c18c8df5e1401c15",
    "hellaswag_localjsonl.yaml": "ba3f3bac2ca1fe8147faa6867ab417d4c5e52a989727dd1650370764a3723815",
    "piqa_localjsonl.yaml": "1f2249ce2a482faf22e6b84558339d2e3f2cd8ec88f83920af405a51efb308b1",
    "winogrande_localjsonl.yaml": "7724e1ee315357732f125796126a1f0a99d2f0bf75eb6a63522d3eef9d88f999",
}
MODEL_DIR = Path("/diskF/jiaqizhang/models/新建文件夹")
INCLUDE_PATH = Path("/diskF/jiaqizhang/a1804298693/offline_datasets")
INPUT_BINDING = Path("/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/input_bindings/MPT7B_EVAL_INPUT_BINDING_V1.json")
OUT_PATH = Path("/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/audits/MPT7B_ZERO_SHOT_CONTEXT_AUDIT_V1.json")
MAX_CONTEXT = 2048


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def token_pair(tokenizer, context: str, continuation: str, prefix_id: int):
    # Match lm_eval.api.model.LM._encode_pair and HFLM.tok_encode defaults.
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    if context == "":
        context_tokens = [int(prefix_id)]
        continuation_tokens = tokenizer.encode(continuation, add_special_tokens=False)
    else:
        whole = tokenizer.encode(context + continuation, add_special_tokens=False)
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        continuation_tokens = whole[len(context_tokens):]
    return context_tokens, continuation_tokens


def main():
    # Force offline and keep derived HF datasets cache in RAM only.
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cache_dir = Path(tempfile.mkdtemp(prefix="mpt_zs_context_cache_", dir="/dev/shm"))
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

    try:
        from transformers import AutoTokenizer
        from lm_eval.tasks import TaskManager, get_task_dict

        binding = json.loads(INPUT_BINDING.read_text(encoding="utf-8"))
        if binding["accuracy_protocol"]["suite"] != TASKS:
            raise ValueError("frozen task suite differs from adapter task list")
        if binding["accuracy_protocol"]["num_fewshot"] != 0:
            raise ValueError("frozen suite is not zero-shot")
        for filename, expected in YAML_SHAS.items():
            actual = sha256_file(INCLUDE_PATH / filename)
            if actual != expected:
                raise ValueError(f"task YAML hash mismatch: {filename} {actual}")

        tok = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        if tok.__class__.__name__ != "GPTNeoXTokenizerFast":
            raise ValueError(f"tokenizer class changed: {tok.__class__.__name__}")
        prefix_id = tok.eos_token_id

        task_manager = TaskManager(include_path=str(INCLUDE_PATH))
        task_dict = get_task_dict(TASKS, task_manager)
        task_summaries = {}
        total_docs = 0
        total_requests = 0

        for task_name in TASKS:
            task = task_dict[task_name]
            task.build_all_requests(
                limit=None, rank=0, world_size=1, cache_requests=False,
                rewrite_requests_cache=False, system_instruction=None,
                apply_chat_template=False, fewshot_as_multiturn=False,
                tokenizer_name=tok.name_or_path,
            )
            per_doc = {}
            max_request = None
            over_requests = 0
            request_count = 0
            max_ctx = 0
            max_full = 0
            choice_counts = defaultdict(int)

            for instance in task.instances:
                if len(instance.args) != 2 or not all(isinstance(x, str) for x in instance.args):
                    raise ValueError(f"unexpected request schema in {task_name}: {type(instance.args)}")
                context, continuation = instance.args
                ctx_toks, cont_toks = token_pair(tok, context, continuation, prefix_id)
                full_len = len(ctx_toks) + len(cont_toks)
                doc_payload = json.dumps(instance.doc, sort_keys=True, ensure_ascii=False, default=str)
                doc_id = hashlib.sha256(doc_payload.encode("utf-8")).hexdigest()
                per_doc.setdefault(doc_id, {"overlength_choices": 0, "max_full_tokens": 0})
                per_doc[doc_id]["max_full_tokens"] = max(per_doc[doc_id]["max_full_tokens"], full_len)
                if full_len > MAX_CONTEXT:
                    over_requests += 1
                    per_doc[doc_id]["overlength_choices"] += 1
                request_count += 1
                choice_counts[doc_id] += 1
                max_ctx = max(max_ctx, len(ctx_toks))
                max_full = max(max_full, full_len)
                if max_request is None or full_len > max_request["full_tokens"]:
                    max_request = {
                        "document_sha256": doc_id,
                        "context_tokens": len(ctx_toks),
                        "continuation_tokens": len(cont_toks),
                        "full_tokens": full_len,
                    }

            over_docs = sum(1 for r in per_doc.values() if r["overlength_choices"] > 0)
            sample_count = len(per_doc)
            total_docs += sample_count
            total_requests += request_count
            task_summaries[task_name] = {
                "sample_count": sample_count,
                "loglikelihood_request_count": request_count,
                "over_context_request_count": over_requests,
                "samples_with_any_over_context_choice": over_docs,
                "max_context_tokens": max_ctx,
                "max_context_plus_continuation_tokens": max_full,
                "longest_request": max_request,
            }

        status = "PASS_NO_OVERCONTEXT_PROMPTS" if all(
            x["over_context_request_count"] == 0 for x in task_summaries.values()
        ) else "OVERCONTEXT_FOUND_REPORT_BEFORE_QUALITY"
        report = {
            "schema": "MPT7B_ZERO_SHOT_CONTEXT_AUDIT_V1",
            "status": status,
            "quality_run": False,
            "input_binding_sha256": sha256_file(INPUT_BINDING),
            "input_binding_internal_sha256": binding["manifest_sha256"],
            "model_content_fingerprint": binding["model_content_fingerprint"],
            "tokenizer": {
                "class": tok.__class__.__name__, "vocab_size": tok.vocab_size,
                "len": len(tok), "eos_token_id": tok.eos_token_id,
                "trust_remote_code": False,
            },
            "scorer_context_length": MAX_CONTEXT,
            "scorer_tokenization": "lm_eval HFLM default add_bos_token=False; exact _encode_pair whitespace handling",
            "harness_overlength_behavior_reference": "HFLM left-truncates (context+continuation) to max_length+1 then drops final token; audit only counts and does not truncate/run scoring",
            "frozen_expected_sample_count": binding["accuracy_protocol"]["suite_total_samples_from_frozen_Llama3_protocol"],
            "observed_sample_count": total_docs,
            "observed_request_count": total_requests,
            "task_yaml_sha256": YAML_SHAS,
            "per_task": task_summaries,
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
