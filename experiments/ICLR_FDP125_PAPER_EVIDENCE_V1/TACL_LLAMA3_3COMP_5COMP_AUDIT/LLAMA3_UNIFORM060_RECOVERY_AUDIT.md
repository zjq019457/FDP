# Llama-3 Uniform-EB=.060 Pass-1 recovery and safe-space audit

Audit date: 2026-10-02 (remote host `ecs-bai-quad`)

## Decision

**`COMPARATOR_BINDING_BLOCKED`**. The historical profile points to the expected Pass-1 model path, and its layer rank pairs can be recovered from the profile JSON, but the model checkpoint itself is absent. The available EB=.060 checkpoints are different schedules: one is an Official/resource-favored branch and one is historical FDP using a per-layer adaptive basis. Neither is an exact substitute. No reconstruction or canary was attempted because the full binding is not established and storage is below the heavy-stage gate.

## Bounded lineage audit

### Expected Pass-1 lineage

1. Profile source: `/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb060.json`, 27,055 bytes, SHA-256 `d0d5d7653caae9c272d97ae38ef4dd0c3718573bde3250df650c028ca828dba4`.
   - Its `cmp` argument is `/diskE/jiaqizhang/llama3_8b_eigen_kv_eb060/final_model`.
   - It records 32 layer rank pairs and the 2×512 BF16/eager profile geometry; `prompts_file=null` resolves through the frozen two-prompt default path, with no explicit seed argument.
   - Profile generator `step1_diagnose_llama.py` SHA-256: `df5273a707fcc6006eaee00dfc0e7848bf3b29eee122a20115af7c74ba0ab75a`.
   - Canonical compact-JSON SHA-256 of the 32 rank pairs extracted from this profile: `c922f1af71e6f7393e7d85f14f794a058a7ebe116612dd0b62de279d259a0f1d` (rank sum 47,872). This is a **derived profile-rank signature**, not a recovered historical rank-config SHA.
2. Original materialization log: `/diskE/jiaqizhang/llama3_8b_eigen_kv_eb060/log_rank0_1777928806.txt`, SHA-256 `7fdf78b2da635080d7d0ccb8fd46725b1dfb882e05181ede0fe39ea9f1bed13a`.
   - Dated 2026-05-04; records base `/diskD/jiaqizhang/Llama-3`, `error_budget=0.06`, C4 calibration, `nsamples=64`, `seed=2`, BF16, eager, `avg_dim=4096`, and basis path `/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_eb/eigen_basis_llama3_8b_wikitext2_eb0p06.pt`.
   - The log says it saved to the expected E path and succeeded, but today that directory contains only the log: `final_model` is absent.
   - The basis file at that path currently exists (119,521,053 bytes), SHA-256 `1f38625bed2c9caa6efd4822ef96ec4151cd90173d7040e679e562d65b174dc8`, mtime 2026-05-01. The log records a path, not the basis digest. The adjacent report is a generic `per_layer_eb=false` report whose recorded budget is .058; it does not cryptographically bind the .06 basis contents to the missing checkpoint.
3. Base-model identity is recorded elsewhere as `meta-llama/Meta-Llama-3-8B`, content fingerprint `b6d5136c429a3df995f73537f05abb658bbc0a63203338930fdea9e8f64e7ae0`. This is the base checkpoint fingerprint, **not** a fingerprint of the missing compressed Pass-1 checkpoint.
4. The original materialization log does not record an operator/builder source hash, exact serialized rank-config SHA, checkpoint hash, or a materialization manifest. The present EigenAttn checkout is dirty; its current `main_eigen_attn.py` hash is `b6dc7eb0d001848121ffea1f28d152bf19b0efaeff2e3b26ad9dcb5e22aeadb7`, but that does not prove the source bytes used in May 2026.

### EB=.060 lookalikes checked and rejected as substitutes

| Candidate | Frozen identity found | Comparison to Pass-1 profile ranks | Verdict |
|---|---|---:|---|
| `/diskF/jiaqizhang/fdp_eb_orth_paper_v1_results/official_external_llama3_8b/resource_favored_eb0p06/final_model` | Official compatibility run; `error_budget=.06`, `nsamples=64`, seed 2, `avg_dim=32`; exact materialized schedule SHA `24e07af896053b25dd8a6f794b5ba68f25d288429488fff750d3747e1e26f4`; `B_live=200,114,176 B`; quality aggregates exist | 27/32 layers differ from the rank pairs embedded in the profile | Not the profile's exact Pass-1 schedule; do not substitute |
| `/diskF/jiaqizhang/fdp_eb_orth_paper_v1_results/historical_replay/eb060/final_model` | `materialize_meta.json` identifies a per-layer `frag_eb060` adaptive basis; rank SHA `4c2336e9a5f009e0babc5e078b48ebfca0fc91d58e219608a22ecc04a7ecf49a`; `B_live=198,377,472 B`; historical FDP Wiki/C4/ZS aggregates exist | 19/32 layers differ | Historical FDP, not Uniform Pass-1; do not substitute |

The resource-favored candidate's rank SHA and B_live are internally cross-checked against its materialized `low_rank_config`, but that validates only that candidate, not identity with the historical profile comparator.

## Cleanup dependency table

No item passed all deletion gates (quality frozen, provenance preserved, no pending dependency, and exact deterministic rematerialization source proven). **Nothing was deleted.** D was not written or modified.

| Path | Size (file bytes unless noted) | Model / artifact | Quality and provenance | Pending dependency / rematerializable? | Decision |
|---|---:|---|---|---|---|
| `/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/materialized_formal_checkpoints/medium/official_uniform/final_model` | 15,804,357,213 | Llama-3 Medium Official, EB=.0567 | Formal quality and resource lock are frozen; rank SHA `976e828d5a360663a6663181fb0afaae64bacfb22577bc3eaa5757325d4947bf` | Not a Uniform060 substitute; exact legacy basis source is not proven rematerializable | KEEP |
| `/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/materialized_formal_checkpoints/medium/fdp125_direct/final_model` | 15,803,766,353 | Llama-3 Medium FDP125, EB=.0567 | Formal quality and resource lock are frozen; rank SHA `f0c462759f53f77214eb7fdce65ea311b5890fd0cd7c6c3eaa061f4f07bf211b` | Not a Uniform060 substitute; exact legacy basis source is not proven rematerializable | KEEP |
| `/diskF/jiaqizhang/fdp_eb_orth_paper_v1_results/official_external_llama3_8b/resource_favored_eb0p06/final_model` | 15,801,346,179 tree bytes (`du -sb`) | Official/resource-favored EB=.060 | Wiki/C4/ZS quality aggregates and rank/B_live cross-check exist | Distinct rank schedule; exact source/basis binding does not prove deterministic recreation | KEEP; preserve near-match evidence |
| `/diskF/jiaqizhang/fdp_eb_orth_paper_v1_results/historical_replay/eb060/final_model` | 15,797,600,175 tree bytes (`du -sb`) | Historical FDP EB=.060 | Wiki/C4/ZS aggregates, rank SHA, and `materialize_meta.json` exist | Adaptive-basis FDP, not Pass-1; exact materializer provenance is insufficient for safe deletion | KEEP; preserve historical result |
| `/diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/LLAMA2_UNIFORM_EB059_STRICT_MATCH_V1/attempt5/final_model` | 13,510,073,587 | Llama-2 EB=.059 materialization | Materialization logs/config present; no bound completed formal quality aggregate found in the audited location | Belongs to active redesign evidence lane; not eligible for cleanup | KEEP |
| `/diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/llama2_e060_fullbasis_uniform_same_rank_audit_v1` | 16,699,346,296 total | Llama-2 EB=.060 basis/materialization/canary artifacts | Includes basis, schedule, parity/canary audits; not a disposable duplicate | Ongoing scientific/provenance dependency | KEEP |
| `/diskF/jiaqizhang/fdp125_tacl_llama2_exact_3comp_pass1_v1_20260930/llama2_e070_uniform_materialization` | 12,702,169,508 total | Llama-2 EB=.070 materialization/profile | No frozen full quality result identified at this path | Active multi-operating-point lane | KEEP |
| `/diskF/jiaqizhang/fdp125_tuning_stage2_llama2_20261001/LLAMA2_FDP_STAGE3_RUNS_V1/L1.2_G0.75_K8_C0.25pct` | 12,805,412,228 total | Llama-2 allocator-tuning candidate | Candidate materialization/logs; not a completed final result | Tuning artifact; do not delete | KEEP |

## Storage / execution snapshot

| Mount | Free bytes at audit | Free GiB (approx.) |
|---|---:|---:|
| `/diskE` | 14,185,197,568 | 13.21 |
| `/diskF` | 3,732,041,728 | 3.48 |
| `/diskD` | 6,276,386,816 | 5.85 |
| `/` | 15,367,651,328 | 14.31 |

Available GPU snapshot: 3× NVIDIA A100 80GB, each 81,153 MiB free, 0 MiB used, 0% utilization. No active FDP/EigenAttention Python worker was found. No new materialization, profile, PPL, or zero-shot job was launched. Space freed: **0 bytes**; the requested 25–35 GB reserve was not reached safely.

## Final gate

- Exact historical Uniform060 Pass-1 checkpoint: **missing**.
- Historical exact rank-config SHA / compressed checkpoint hash / frozen materialization manifest: **not found**. Profile rank signature is derivable but is not a substitute for a frozen rank artifact.
- Candidate basis file SHA is measurable, but the historical log did not record that digest; exact byte-level basis-to-checkpoint binding remains absent.
- Operator/builder version binding: **unresolved** for the historical materialization.
- Deterministic canary: **NOT RUN** (no exact bound checkpoint/materialization; storage also below the 15-GiB heavy-stage reserve).
- Final status: **`COMPARATOR_BINDING_BLOCKED`**.
- A/B/C independent profile stability, same-protocol structural-vs-functional analysis, and full FDP overhead: **do not proceed as formal evidence** until the exact comparator binding is recovered. Keep the q-only microbenchmark and cross-protocol correlations diagnostic-only.

No EB=.0567, .045, .090, .120, historical adaptive-FDP checkpoint, or resource-favored .060 checkpoint was substituted for the exact Pass-1 comparator.
