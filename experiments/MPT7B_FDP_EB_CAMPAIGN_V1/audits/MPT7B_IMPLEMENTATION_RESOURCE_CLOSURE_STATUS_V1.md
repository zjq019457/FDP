# MPT-7B FDP-EB implementation/resource closure status

Snapshot: 2026-10-05 09:56 PDT. This is a provenance/status record, not a quality result.

## Decision

Formal WikiText-2, C4, and zero-shot scoring remain **NOT STARTED**. No MPT profile, resource schedule, actual resource lock, or 7B compressed realization exists yet. The next work remains MPT-native profile/rank-search integration and resource closure. Do not treat synthetic smoke results or the dense-cache geometry below as MPT-7B formal evidence.

## A–L required closure report

| Item | Status | Evidence / blocker |
|---|---|---|
| A. Three-component profile | `ADAPTER_RULE_TESTED; MPT_PROFILE_PENDING` | Adapter computes raw `dZ+dS+dH`, then one layerwise min-max; frozen Llama-3 mapping uses λ=1.2 and mean normalization. 8/8 protocol tests passed in the actual `eigen` environment. No MPT 2×512 profile activations or MPT q have been generated. |
| B. Native rank-search parity | `SYNTHETIC_OPERATOR_SMOKE_PASS; CHECKPOINT_PARITY_PENDING` | Small synthetic MPT dense-vs-Eigen output and incremental-cache output each had max-abs=0, relative-L2=0; ALiBi/cache shapes and patched past-length handling passed. This is not a 7B-weight comparison. |
| C. Dense MPT B_live | `GEOMETRY_ONLY; ACTUAL_CENSUS_PENDING` | At batch=1, seq=2048, 32 layers, 32 heads, head-dim=128, BF16, K+V geometry is 1,073,741,824 B (1 GiB). This is a shape-derived expectation, not an observed runtime B_live and is not frozen as `B_dense_MPT`. |
| D. Uniform EB .011/.015/.019 | `NOT_STARTED` | No legal schedules, realized cache counts, or rank hashes. |
| E. R70/R60/R50 resource curve | `NOT_STARTED` | No actual-resource curve or selected Uniform/FDP candidates. EB values will not be assigned tier labels without measured retained ratios. |
| F. Matched pairs | `NONE` | No resource-locked Uniform/FDP pair exists. |
| G. Actual resource gaps | `NONE` | No actual MPT compressed B_live exists; no formal resource gap can be reported. |
| H. Rank evidence | `LATTICE_RULE_TESTED; SCHEDULE_PENDING` | 32-layer MPT schedule validation requires legal per-head ranks in multiples of 32, within floor 64 and full rank 4096. No layer ranks are selected. |
| I. Compressed PPL runner | `MPT_BRANCH_EXISTS; ACTUAL_CANARY_PENDING` | The official pipeline contains an MPT compressed-loader branch. Isolated adapter fixes two cache-length reads in a copy of the upstream MPT operator; only a synthetic CPU canary passed. No actual 7B compressed model was materialized. |
| J. Zero-shot context compatibility | `PASS_NO_OVERCONTEXT_PROMPTS` | Frozen 5-task suite: 16,695 unique samples and 60,566 requests; zero requests exceed 2,048 tokens. Longest request: PIQA 237 tokens. This audit counted lengths only and ran no model/scorer. |
| K. F-disk safety | `BLOCKED_BELOW_HEAVY_THRESHOLD` | F free=7,256,748,032 B; required floor=16,106,127,360 B (15 GiB); short by 8,849,379,328 B (8.24 GiB). Prior read-only cleanup audit found no safe, material regenerable candidate; nothing was deleted. |
| L. GPU availability | `NO MPT JOB LAUNCHED` | 09:56 snapshot: GPU0 free 39,577 MiB, util 98%; GPU1 free 39,577 MiB, util 100%; GPU2 free 17,413 MiB, util 100%. Active compute processes belong to unrelated/foreign jobs and an existing Llama-3 job; no MPT GPU process. Current F-space gate independently blocks heavy realization. No job was interrupted. |

## Zero-shot context audit details

| Frozen task | Samples | Requests | Over-2048 requests | Longest context+continuation |
|---|---:|---:|---:|---:|
| ARC-Challenge | 1,172 | 4,687 | 0 | 171 |
| ARC-Easy | 2,376 | 9,501 | 0 | 163 |
| HellaSwag | 10,042 | 40,168 | 0 | 164 |
| PIQA | 1,838 | 3,676 | 0 | 237 |
| WinoGrande | 1,267 | 2,534 | 0 | 39 |
| **Total** | **16,695** | **60,566** | **0** | **237** |

The audit used the frozen local YAMLs and tokenizer, `lm_eval` request construction, and exact context/continuation tokenization semantics; it performed no inference, scoring, or truncation. It binds the MPT tokenizer and 2048-token maximum, but does not itself authorize zero-shot quality before the MPT resource/method gates.

## Frozen identity and provenance

- Checkpoint content fingerprint: `3e6db8661db48e835566a784d5ef1d1495a38a629cc3f2f0207364e07c33270f`.
- Checkpoint content manifest SHA-256: `8fd59bc340a43f13b250c77d118cc8e479993ecdee1a839bcee5cd7bc1f51db9`.
- Evaluation input binding SHA-256: `c28e98c14e082167f7c7389832a7f1807128003049a8ab2b2e3c076a3725da84` (internal digest `7aa095c9de75b3a5ddb5980f581f1835ca7ae09906b56efb0378bdc06bee98d0`).
- Frozen Llama-3 three-component method lock SHA-256: `8a8e95ae4df92203b5ad9afdbec4ed244004b57750e81a41b2f24bc5b880df4f`.
- Upstream MPT operator SHA-256: `89ee4efbf17da3e2b0734ac74315ea0f9bc491e6ab1be5a48dfd1a909924b3fd`.
- Isolated patched MPT adapter SHA-256: `322357700e027b3bf2f9bdf5a680400e950be560669b7b24bb717c072c08d52a`.
- Protocol/rank adapter SHA-256: `8709466ea85c25cd0efec7b9fbeab373cfcedb50c07ad028f898f558aa80995e`.
- Synthetic operator smoke report SHA-256: `1875f93cd31732ff9cbdfe6135e359240483aff1a696ffd2e1a331df9ebb7112`.
- Zero-shot context audit report SHA-256: `8d8c8883b840ce66369cc784a00ae57fc4805f5a8a8edbd870f330893624521d`.
- Remote test environment: PyTorch `2.8.0+cu128`, Transformers `4.44.2`; protocol tests 8/8 passed.

## Next gate

Before any heavy realization, restore F free space to at least 15 GiB without deleting model, formal evidence, profiles, manifests, NLLs, hashes, or foreign artifacts. Then bind the actual MPT profile tensors to dense/reconstructed/rank-search operators, measure runtime dense B_live, generate quality-blind Uniform resource points and FDP schedules, and close actual cache/resource census plus operator/cache parity and deterministic 7B canary. Only after those gates may formal Wiki/C4 or zero-shot scoring begin.
