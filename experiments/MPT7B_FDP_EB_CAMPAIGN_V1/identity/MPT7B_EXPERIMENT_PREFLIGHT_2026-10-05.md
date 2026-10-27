# MPT-7B FDP-EB preflight — 2026-10-05

## Current state

`MODEL_CONTENT_BOUND / INPUTS_BOUND / RESOURCE_PROFILE_NOT_STARTED / QUALITY_NOT_STARTED`.
No MPT GPU experiment has been launched. No model or source-repository files were modified, and no D: writes were made.

## Checkpoint and transfer

- Uploaded to `/diskF/jiaqizhang/models/新建文件夹`.
- 24 files, 13,300,903,736 bytes; local and remote per-file SHA-256 inventories matched.
- Model content fingerprint: `3e6db8661db48e835566a784d5ef1d1495a38a629cc3f2f0207364e07c33270f`.
- Config: `MptConfig`, `model_type=mpt`, 32 layers, width 4096, 32 heads, context 2048, ALiBi.
- Tokenizer: `GPTNeoXTokenizerFast`, `vocab_size=50254`, `len(tokenizer)=50277`, EOS id 0. Loaded locally with `trust_remote_code=False`.
- CPU smoke: standard Transformers MPT-7B loaded with zero missing/unexpected/mismatched checkpoint tensors; a 6-token forward returned finite logits `[1,6,50432]`.
- The exact upstream repository revision is not recoverable from this directory; content is pinned by the fingerprint above. Do not infer revision from the parent folder name `l31`.

## Frozen evidence bindings prepared

- Llama-3 replication authority: `/diskE/jiaqizhang/ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_3COMP_5COMP_AUDIT/LLAMA3_3COMP_TRUE_TWO_PASS_METHOD_LOCK.json`, SHA-256 `8a8e95ae4df92203b5ad9afdbec4ed244004b57750e81a41b2f24bc5b880df4f`.
- MPT tokenizer-specific evaluation-input manifest: `/diskE/jiaqizhang/MPT7B_FDP_EB_CAMPAIGN_V1/input_bindings/MPT7B_EVAL_INPUT_BINDING_V1.json`, file SHA-256 `c28e98c14e082167f7c7389832a7f1807128003049a8ab2b2e3c076a3725da84`, canonical manifest digest `7aa095c9de75b3a5ddb5980f581f1835ca7ae09906b56efb0378bdc06bee98d0`.
- Input manifest contains 140 WikiText-2 test windows and 256 fixed C4-text windows after MPT-specific tokenization, each with an input-id hash.
- The five zero-shot task YAML SHA-256 values match the frozen Llama-3 suite exactly; total suite size is 16,695 examples. This binds task definitions, not completed MPT accuracy results.
- Local WikiText-2 parquet and selected C4-text source SHA-256 values are recorded in the input manifest; C4 source was read-only on D:.

## Adapter audit findings — blockers before formal quality

1. The official EigenAttention repository has MPT-native modules (`MptForCausalLM_EigenAttn`, `MptBlockEigenAttn`) and MPT-specific decomposition for 32 blocks, fused QKV and ALiBi.
2. The generic `derive_eb.py` default is five-component. It must be bypassed with the frozen `d1+d2+d5` definition: raw sum, then min-max across layers.
3. Current MPT decomposition defaults to threshold start 0.98 and has a rollback/floor loop different from the Llama-3 lock (1.00 down to 0.30 inclusive, 0.02 steps, first-failure stop). Do not use it unmodified for a formal schedule.
4. `ppl_instrumented.py` has no MPT compressed-model loader. The generic pipeline has an MPT route, but its default dataset/selection and precision path are not the pinned Llama-3 evaluation binding. Use a versioned MPT loader/scorer adapter; do not call the Llama fallback.
5. `MptConfig` has `max_seq_len=2048` and no `max_position_embeddings`. The generic `LMClass.max_length` fallback assumes `max_position_embeddings`; zero-shot evaluation needs an MPT-specific wrapper returning `max_seq_len`.
6. Llama-3 replication constants to preserve: `q=MinMax(d1+d2+d5)`; `lambda=1.2`; mean-normalized epsilon; layer 0 effective epsilon equals the base EB; rank floor 64; independent tiers `.011/.015/.019`; each tier gets its own Uniform pass/profile/q/schedule. Intended retention labels are not results—report actual `B_live` and retained ratio.

## Resource snapshot (read-only)

- GPU free VRAM: GPU0 25,519 MiB; GPU1 11,871 MiB; GPU2 17,413 MiB. All were at 100% utilization and below the 32-GiB-per-heavy-worker launch gate.
- E: 44,899,819,520 bytes free, above the 15-GiB reserve.
- F: 7,936,552,960 bytes free; uploaded model retained here, no further heavy output staged.
- D: 85,306,904,576 bytes free; read-only, no writes.
- Existing unrelated/foreign GPU jobs were left untouched. No MPT quality or basis job is waiting in a newly created scheduler.

## Decision

Transfer and identity checks passed. Formal resource profiling is not yet authorized because the MPT adapter does not yet reproduce the frozen Llama-3 rank-search/precision semantics, and no GPU currently meets the memory gate. Next safe work is to finish and hash the MPT-specific strict builder/evaluation wrappers, then launch the first resource-only Uniform/profile lane as soon as a safe GPU slot appears. No Wiki/C4/zero-shot score has been produced for MPT.
