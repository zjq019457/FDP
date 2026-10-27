# Unified LongBench Protocol v1.0

Protocol ID: `ULB-v1-052cfbbeaa2cc42f`

## Scope

This protocol freezes the shared data, prompting, generation, scoring, failure handling, identity, output, resume, and statistical rules for CC and DRM LongBench experiments. Model-specific pruning schedules and chat templates are explicit branches and do not make the protocol non-unified.

The protocol is for experiments and standardized result export only. It does not authorize paper/LaTeX edits or claims.

## Official task set

The ordered 17-task manifest contains 3,950 samples: 150 for `multifieldqa_en`, 500 each for `lcc` and `repobench-p`, and 200 for every other task. Dataset snapshot SHA256 is `7476580c04c3bcc47aeeb7cf5e5bb0c8b802d23096ab73ee5275147877edb469`. A run with a per-task cap of 200 is not this protocol.

Each task is scored independently with the frozen LongBench scorer. The final score is the simple arithmetic mean of 17 task scores. Pooling all samples into a weighted average is forbidden.

## Stable sample identity

Canonical sample IDs are `<task>:<zero_based_row_index>:<content_hash16>`. The full content SHA256 hashes canonical JSON over `task`, `context`, `input`, `question`, `answers`, `all_classes`, and `language`. JSON object key order and serialization whitespace do not affect the hash; parsed string values are preserved exactly.

## Prompt and generation

The exact dataset template is formatted first. If the raw tokenized prompt exceeds 131,072 tokens, the runner decodes the first and last 65,536 tokens and concatenates them. The model chat template is then applied except for `trec`, `triviaqa`, `samsum`, `lsht`, `lcc`, and `repobench-p`. Per-sample prompt SHA256 is computed over the exact final UTF-8 prompt before its final tokenizer call.

Generation is batch size 1, single-GPU sequential, greedy (`do_sample=False`, `num_beams=1`), with frozen task-specific `max_new_tokens`. `samsum` additionally accepts the final newline token as EOS, matching the historical runner.

## Model branches

- Qwen2.5-7B-Instruct: layers 9/18/26, K 192/96/64, M 10/5/3, retained tokens 12288/6144/4096.
- LLaMA-3.1-8B-Instruct: layers 10/20/30, K 128/64/32, M 6/3/2, retained tokens 8192/4096/2048; Add tokens 8576/4288/2176.
- Common: block size 64, token unit 8, gamma/swap threshold 0.9, local query window 4, fp16, FlashAttention2.

The requested LLaMA source state is currently `BLOCKED_SOURCE_ABSENT` on this machine. Its exact model/tokenizer/chat-template and method-code fingerprints must pass identity audit before LLaMA outputs can carry this protocol ID.

## DRM and fair comparisons

DRM is frozen at rescue ratio 0.05, alpha_diff 0.5, lambda_red 0.25, eta_rescue 0.25, eta_active 0.25, seed 0, and historical MMR. Add preserves 100% of A_K and adds exactly M valid blocks. Swap is strict one-in-one-out and remains K.

CC compares `cc_sliminfer_v1` with `sliminfer_original`. DRM compares Swap with Original at K, and DRM-Add with Larger/Random-Add at K+M. CC and DRM-Add are not a same-budget comparison.

## Status and output

Canonical statuses are `ok`, `oom`, `failed`, and `missing`. Missing numeric fields are `null`, never zero. Historical reconstruction is explicitly labeled in `field_sources`. OOM predictions remain `[OOM]`. Source file and line are always retained.

Raw full-set, global common-valid, pairwise common-valid, and offload-completed accuracy are distinct result concepts. Source predictions are immutable and are never silently merged across protocol identities.

## Current freeze boundary

Protocol semantics, the local 3,950-sample data snapshot, Qwen tokenizer/prompt rendering, scorer, schemas, and statistical policies are frozen. LLaMA source artifacts were not found at the requested path, so LLaMA result export, LLaMA identity PASS, smoke, and detached full launch remain blocked. No GPU task was started by this freeze.
