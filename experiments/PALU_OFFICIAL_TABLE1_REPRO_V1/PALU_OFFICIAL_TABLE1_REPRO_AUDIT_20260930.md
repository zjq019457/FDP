# PaLu official-paper reproduction audit (2026-09-30)

## Scope and guardrails

This is a clean reproduction lane at `/diskE/jiaqizhang/PALU_OFFICIAL_TABLE1_REPRO_V1/`. The official source is pinned at `bb22666e2ef96707e8dd21d93fc00146c2e0d615`; no official source files, frozen V1/V3 allocations, or earlier result files were modified. `foreign_jobs_interrupted=0`.

The question is not whether V3 beats the original paper's rounded Table 1 numbers. First establish which official decomposition/allocation/evaluation paths those numbers represent, then compare only protocol-aligned cells. The Appendix F.3 allocation ablation remains a separate result family.

## What the paper says, and what the repository implements

The paper describes three decomposition operators (J-LRD, G-LRD, M-LRD), states a default G-LRD group size of 4, and reports Fisher rank-search calibration using 2,048 WikiText-2 samples of length 1,024. It separately labels Appendix F.3 as Automatic vs Uniform rank allocation, with 50%-compression WikiText-2 PPL 6.02 vs 7.36. The official repository's default search method is `fisher_uniform`; its `fisher_uniform` branch itself requests 2,048 samples. These are distinct axes: decomposition granularity and rank allocation are not synonyms.

The paper's Table 1 reports Llama-2-7B at nominal 50% compression: Dense Wiki/C4 5.47/7.26; J-LRD 5.62/7.75; G-LRD 6.01/9.82; M-LRD 6.75/12.01. Table 6 reports WikiText-2 for the group-size sweep. The Table 1 prose does not provide a paper-specific launch command or a machine-readable allocation manifest. The `equal rank size for each group` sentence must therefore be interpreted together with the code and the separate F.3 ablation; it does not by itself prove that Table 1 used the globally Uniform allocator.

The two direct G-LRD/gs=4 diagnostics already separate the allocation schemes:

| Path | WikiText-2 PPL | C4 PPL | Interpretation |
|---|---:|---:|---|
| Official-code `fisher_uniform`, gs=4 | 6.020491 | 8.718500 | Appendix F.3 Automatic reproduction; Wiki matches F.3 6.02 closely. |
| Official-code `uniform`, gs=4 | 7.356223 | 12.476010 | Appendix F.3 Uniform reproduction; Wiki matches F.3 7.36 closely. |

The Automatic Wiki value is also numerically near Table 1 G-LRD Wiki 6.01; the Uniform value is not. This supports, but does not conclusively prove, that Table 1's G-LRD Wiki row used automatic/Fisher allocation. The C4 value from Automatic is 1.101500 lower than Table 1's 9.82, despite the Dense C4 reproduction being close. The Table 1 C4 discrepancy is unresolved; no cause is inferred without code/config evidence.

## Completed PPL evidence

| Cell | Wiki PPL | C4 PPL | Paper reference / error | Classification |
|---|---:|---:|---|---|
| Dense official evaluator | 5.472263 | 7.263262 | 5.47 / 7.26; relative errors 0.0414% / 0.0449% | Close numerical reproduction |
| J-LRD gs=32, Uniform | 6.250679 | 8.391844 | Table 1 5.62 / 7.75 | Not numerically reproduced |
| G-LRD gs=4, Uniform | 7.356223 | 12.476010 | Appendix F.3 Uniform Wiki 7.36; Table 1 C4 is only a diagnostic reference | F.3 Uniform reproduced on Wiki |
| M-LRD gs=1, Uniform | 7.828170 | 15.040820 | Table 1 6.75 / 12.01 | Not numerically reproduced |
| G-LRD gs=4, Automatic/Fisher | 6.020491 | 8.718500 | F.3 Automatic Wiki 6.02; Table 1 Wiki 6.01 / C4 9.82 | F.3 Automatic reproduced on Wiki; Table 1 C4 unresolved |

Uniform sweep rank totals are exact at 131,072 out of 262,144 native ranks, with 32-rank quantum. Official Fisher allocation uses its own per-unit rounding and can differ from the nominal aggregate target; actual ranks/bytes are recorded, not silently rebalanced.

## Current official group-size run

An additional, isolated `fisher_uniform` sweep is running on GPU0 for J-LRD gs=32, G-LRD gs=16/8/2, and M-LRD gs=1. G-LRD gs=4 is already recorded separately above. The current runner uses its own output/state/lock namespace and explicitly sets the wrapper sample-count field to 2,048; the official source branch also hardcodes 2,048 in `fisher_uniform`. No completed result is overwritten. At the last check the first J-LRD gs=32 cell was still running; its final PPL and the remaining group-size results are pending.

## Evaluator and missing evidence

The official PPL path uses sequence length and stride 2,048, non-overlapping windows, FP16 model weights, WikiText-2 test data, and the first 1,100 C4 validation documents (256 chunks). Dense reproduces both rounded Table 1 values closely, which supports—but does not fully identify—the evaluator path. The paper does not state every low-level Table 1 setting in enough detail to infer the exact original launch command from the PDF alone.

Zero-shot has not completed. The clean environment lacks the required Hub dataset artifacts and the network/cache path is unavailable; this is recorded as an external-data TODO, not a model result. No zero-shot score is claimed. Timing results are not attempted in this reproduction lane.

## Frozen V3 compatibility note

The existing read-only substrate audit reports official native representation equivalence PASS: G-LRD gs=4, same official materializer/module, legal rank quantum, frozen V3 allocation and exact resource census. It also records an evaluator-wrapper parity caveat for legacy C4 artifacts. The previously frozen V1-vs-V3 formal paired result is separate evidence: V1 J=1.9751725, V3 J=1.9655115, paired ΔJ=-0.0096610, 95% CI [-0.0102473,-0.0091206], exact resources. This audit does not reopen V3 search or replace the confirmatory paired result with unpaired Table 1 values.

## Provenance and next reporting gate

The remote reproduction root contains source trace, input/cache hashes, per-cell allocations, result JSON, and failure/blocker audits. The new group sweep must finish before Table 6 comparisons are classified. Zero-shot remains TODO until data is available. Table 1 C4 remains unresolved unless an exact official Table 1 invocation or matching original evaluation artifact is found.
