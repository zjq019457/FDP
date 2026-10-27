# PaLu Table 1 vs Appendix F.3/Table 8 — Reproduction Audit

Audit date: 2026-09-29. Campaign root: `/diskE/jiaqizhang/PALU_OFFICIAL_TABLE1_REPRO_V1/`.

## Bottom line

The Table 1 G-LRD result (WikiText-2 6.01, C4 9.82) and Appendix F.3/Table 8's WikiText-2 Uniform result (7.36) are different experiments. G-LRD specifies decomposition/group granularity; Uniform/Automatic specifies rank allocation. Equal rank among the groups *within a projection* does not imply uniform allocation across all projections/layers. The official repository's default is `fisher_uniform`, and its README compression example uses `fisher_uniform` with G-LRD group size 4. This is strong evidence that the Table 1 number used Fisher-weighted allocation while Table 8's Uniform ablation did not. The paper/repository snapshot contains no archived Table-1-specific command/config, so the exact historical invocation cannot be proven from that artifact alone.

The requested no-Fisher run is therefore a useful official-code diagnostic, but it is **not an exact Table 1 reproduction**. No Fisher/Automatic allocation was run because the user explicitly prohibited it.

## Source-traced protocol

- Official PaLu source commit: `bb22666e2ef96707e8dd21d93fc00146c2e0d615`.
- Model: `meta-llama/Llama-2-7b-hf`; local checkpoint `/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/models/Llama-2-7b-hf` (the campaign used an isolated copy). Config: 32 layers, hidden size 4096, 32 attention heads and 32 KV heads, head dimension 128, max positions 4096. Local shard SHA-256s are in `audit/model_checkpoint_sha256.txt`.
- Paper Sec. 4.1: truncation-aware SVD from SVD-LLM; default G-LRD group size 4, equal rank size per group; Fisher rank calibration uses 2,048 WikiText-2 samples of length 1,024.
- Official `compress.py` default search method is `fisher_uniform`; the official README example specifies `fisher_uniform`, `head_group_size=4`, and whitening. No exact Table-1 launch command/config was found. The campaign diagnostic explicitly overrides the allocator to `uniform` per user instruction.
- In the official `fisher_uniform` implementation, the global target is `target_rank = total_rank * param_ratio_target` (0.5 here). Each K/V projection's per-group Fisher scores determine provisional rank in proportion to `target_rank * fisher_i / fisher_sum`; the code splits the allocated rank evenly across G-LRD subgroups and rounds ranks to the nearest legal block of 32. K and V use the same algorithm but their Fisher scores are computed from separate projection modules, so their actual ranks need not match. This exact allocation was **not run** in this campaign because of the no-Fisher constraint.
- PPL evaluator `run_ppl_eval.py::eval_ppl`: sequence length 2048, stride 2048 (non-overlapping windows), batch one sequence at a time. WikiText-2 test texts are joined with double newlines. C4 uses the first 1,100 examples from the pinned first validation shard, joins text with spaces, and truncates to 256 full windows. Tokenizer defaults are used, including BOS behavior from the model tokenizer.
- PPL runtime: FP16, official evaluation path, SDPA/automatic device map unless explicitly pinned to GPU0 for the engineering retry. Pinning placement did not alter model, data, precision, scorer, sequence length, or allocation.
- Frozen local inputs: WikiText-2 test SHA-256 `3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925`; train SHA-256 `49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701`; C4 revision `607bd4c8450a42878aa9ddc051a65a055450ef87`, shard SHA-256 `1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f`.
- Zero-shot task set from the official README/LM-Evaluation-Harness config: OpenBookQA (`acc`), HellaSwag (`acc_norm`), PIQA (`acc`), ARC-Easy (`acc`), ARC-Challenge (`acc_norm`), WinoGrande (`acc`). Harness version 0.4.2, batch size 8. The exact official YAMLs and task/split metadata are recorded in `audit/TABLE1_OFFICIAL_TRACE.json`.
- The executed user-requested diagnostic entry point was `python -u scripts/run_uniform_lrd_sweep.py`. It called the pinned official `compress.py::compress` with `param_ratio_target=0.5`, `search_method=uniform`, `decompose_method=whiten`, WikiText-2 calibration, group sizes 32/16/8/4/2/1, then official `run_ppl_eval.py::eval_ppl` at 2048/2048. That is an exact record of this diagnostic command path, **not** a recovered Table-1 command.

## Dense gate

| Dataset | Reproduced PPL | Paper target | Absolute error | Relative error | Classification |
|---|---:|---:|---:|---:|---|
| WikiText-2 | 5.4722633362 | 5.47 | 0.0022633362 | 0.04138% | CLOSE_NUMERICAL_REPRODUCTION |
| C4 | 7.2632622719 | 7.26 | 0.0032622719 | 0.04493% | CLOSE_NUMERICAL_REPRODUCTION |

The dense gate passed; compression diagnostics were allowed. Dense result file: `results/dense_baseline.json` (SHA-256 `830c7c7f69358c3b58f8ff81a76a62a77b4bcc5c7162f5c6fa23ac13437f1fe6`). This is close numerical agreement under the recovered evaluator, not a claim of bitwise-identical historical runtime.

## User-requested no-Fisher uniform PPL diagnostics

All six configurations used the official whiten/truncation-aware decomposition, 50% target compression, 32-rank legal quantum, WikiText-2 PPL at 2048/2048. Rank sum was 131,072 of 262,144 for every compressed cell. Factor bytes vary by decomposition granularity; prefill and decode cache bytes at sequence length 2048 are each 536,870,912 for these cells. Every result is explicitly labeled `PROTOCOL_MISMATCH_TO_LIKELY_OFFICIAL_TABLE1_ALLOCATION` because Fisher was disabled.

| Cell | WikiText-2 PPL | C4 PPL | Table 1 / Table 6 Wiki target | Wiki abs. / rel. error | Factor bytes | Result SHA-256 field |
|---|---:|---:|---:|---:|---:|---|
| J-LRD, gs=32, uniform | 6.2506790161 | 8.3918437958 | 5.62 / 5.62 | 0.630679016 / 11.22205% | 2,147,483,648 | `cae84c6fea4538aa61d8418fed156e0238a71f90ed12294e75d5f3a7e7cbba50` |
| G-LRD, gs=16, uniform | 6.5868253708 | — | 5.74 | 0.846825371 / 14.75306% | 1,610,612,736 | `c61590452367c50facf47b4355a7fc53ad252ffb50e67f46ba69d373d8ea45ed` |
| G-LRD, gs=8, uniform | 6.8781661987 | — | 5.88 | 0.998166199 / 16.97562% | 1,342,177,280 | `09bbb4f46cdf8f71dcf07e3570cbf85bf63f84f5627cea5e5a448994c14e1a99` |
| G-LRD, gs=4, uniform | 7.3562226295 | 12.4760093689 | 6.01 / 6.01 | 1.346222630 / 22.39971% | 1,207,959,552 | `93ecc5f90103e59f93e494ed3757be2d491e9b0d00d0351770956589abe37664` |
| G-LRD, gs=2, uniform | 7.8951859474 | — | 6.42 | 1.475185947 / 22.97797% | 1,140,850,688 | `93927540f6ebc8e043997de556753beeff9045e1657a40f7d79d71f8789fb289` |
| M-LRD, gs=1, uniform | 7.8281702995 | 15.0408191681 | 6.75 (Table 1); 6.81 (Table 6) | 1.078170300 / 15.97289% vs Table 1; 1.018170300 / 14.95111% vs Table 6 | 1,107,296,256 | `99a646c468679879be14851ec695d3a48d074effaae7fc8888420d5764b595ed` |

These are successful measurements of the explicitly requested **uniform allocator**. The differences from paper rows are protocol mismatches, not claims that official Table 1 failed to reproduce under its original allocator. The output JSONs also record allocation SHA, realization SHA, dataset hashes, exact resources, and result provenance under `uniform_diagnostics/<CELL>/`.

## Required 6.01 vs 7.36 diagnosis

1. Table 1 says G-LRD, group size 4, with equal rank size for each group. That describes how a projection is decomposed and how its rank is split among its subgroups.
2. Appendix F.3/Table 8 explicitly compares rank-search allocators: Uniform versus Automatic. The official source distinguishes `uniform` from `fisher_uniform`; under `fisher_uniform`, Fisher importance varies allocation across K/V projection targets, while rank remains equal among groups inside a target.
3. In this campaign, official-code G-LRD gs=4 with global Uniform produced Wiki PPL **7.3562226295**. The separate historical G.3-style local Uniform run produced **7.3563096850**; their absolute difference is **0.0000870554** (relative **0.00118%**). The G.3 rounded reference is 7.36 (absolute difference **0.0037773705**, relative **0.05132%**).
4. Therefore the observed 6.01-versus-7.36 gap is strongly explained by rank-allocation policy, not by an evaluator-only discrepancy. The dense baseline also closely matches 5.47/7.26 under this evaluator. Because no archived Table-1-specific command exists, the precise historical allocator attribution remains evidence-supported rather than directly proven.

The project keeps this separate lane under the requested local label `G3_RANK_ALLOCATION_REPRODUCTION`. In the published ICLR PDF, the corresponding rank-allocation ablation is Appendix F.3 / Table 8 (Appendix G is the experiment-details section). It uses WikiText-2, sequence length 2048, 50% compression, Uniform 7.36 / Automatic 6.02. The historical local Automatic result was 6.0204564227. None of these rank-allocation-ablation results are merged into the Table 1 outputs.

## Six zero-shot metrics

| Task | Paper reference | Metric | Reproduction status |
|---|---:|---|---|
| OpenBookQA | 44.20% | acc | TODO_BLOCKED |
| HellaSwag | 76.00% | acc_norm | TODO_BLOCKED |
| PIQA | 78.07% | acc | TODO_BLOCKED |
| ARC-Easy | 76.30% | acc | TODO_BLOCKED |
| ARC-Challenge | 46.42% | acc_norm | TODO_BLOCKED |
| WinoGrande | 69.30% | acc | TODO_BLOCKED |

No metric scoring began: fetching the first task (`openbookqa`) failed because the Hugging Face dataset Hub was unreachable. This is retryable after official dataset access/cache is restored. Scores are null; no substitutes or synthetic values were used. See `audit/ZERO_SHOT_TASK_BLOCKERS.json`.

## Runtime failure and repair history

- First pass: J-LRD and G-LRD gs=4 produced Wiki metrics (6.2511200905 and 7.3562226295) but failed before C4 scoring because a stale 1,194-byte C4 testloader cache lacked `data.pkl`. That corrupt cache was retained, not deleted: `audit/quarantine/corrupt_c4_testloader_20260929.cache`, SHA-256 `657b17891378d24edd8eb0839fc0f0d9cf4dbcfa8de4d1a9f23a65f62c80a89b`.
- Repair: quarantine only the invalid, uniquely named local C4 cache, allowing the unchanged official loader to regenerate it. The affected J-LRD and G-LRD gs=4 cells were retried; their completed results are above. Earlier Dense results remain valid.
- A retry initially failed because `device_map=auto` spread the model across GPU0 and GPU1 while the PPL input was pinned to `cuda:0`; no foreign task was interrupted. The two affected cells were retried with `CUDA_VISIBLE_DEVICES=0`, changing device placement only. Both then completed successfully. Historical failed-attempt JSONs and cache are preserved in the audit tree.
- `foreign_jobs_interrupted=0` throughout.

## Reproduction-level classification and remaining gate

- Dense Wiki/C4 PPL: `CLOSE_NUMERICAL_REPRODUCTION`.
- No-Fisher J/G/M and group-size cells: `PROTOCOL_MISMATCH` against the likely Table 1 Fisher-weighted allocation, although the requested uniform diagnostic itself completed and is reproducible from saved artifacts.
- Six zero-shot rows: `NOT_REPRODUCED` / `TODO_BLOCKED_EXTERNAL_DATASET_UNAVAILABLE`.
- Appendix G.3 Uniform cross-check: `CLOSE_NUMERICAL_REPRODUCTION` for Wiki PPL; it is not Table 1.
- Current V3 substrate: the separate audit reports G-LRD gs=4, official headwise low-rank module, same whitened/truncation-aware SVD, legal rank quantum 32, and exact resource census. This supports native-substrate compatibility, not reproduction of Table 1's allocator or zero-shot protocol; the V3 audit also flags non-bitwise parity with the legacy evaluator wrapper (C4 Fisher PPL drift 0.0002494314). V3 evidence is not merged into this campaign.

**No-Fisher constraint versus exact Table 1:** the request to disable Fisher conflicts with the strongest available evidence for Table 1's allocation procedure. Thus the requested no-Fisher work is complete as a diagnostic, but a full original Table 1 reproduction and its six zero-shot metrics remain open. The exact Table 1 allocator cannot be claimed without resolving this protocol conflict and restoring zero-shot dataset access.

## Provenance files

- `audit/TABLE1_OFFICIAL_TRACE.json`
- `audit/ZERO_SHOT_TASK_BLOCKERS.json`
- `results/dense_baseline.json`
- `state/UNIFORM_SWEEP_STATE.json`
- `uniform_diagnostics/<CELL>/allocation.json` and `result.json`
- `logs/uniform_lrd_sweep_20260929.log`, `logs/uniform_lrd_retry_after_cache_repair.log`, `logs/uniform_lrd_retry_gpu0_only.log`
