# SAFE_DISK_CLEANUP_AUDIT — Llama-3 FDP stability stage

Audit time: 2026-10-02 01:11 PDT (remote host)

## Space snapshot

| Mount | Free bytes before | Free after | Freed |
|---|---:|---:|---:|
| `/diskE` | 14,185,213,952 | 14,185,197,568 | 0 |
| `/diskF` | 3,732,041,728 | 3,732,041,728 | 0 |
| `/diskD` | 6,276,386,816 | 6,276,386,816 | 0 |
| `/` | 15,367,651,328 | 15,367,651,328 | 0 |

No deletion was performed. All proposed large-artifact removals either have an unresolved dependency, lack a verified rematerialization source, or fall under the existing D-drive read-only constraint. The two small audit reports written to E after the initial snapshot account for the 16,384-byte decrease in E free space. E now has about 13.21 GiB free, below the 15-GiB heavy-stage gate; F has about 3.48 GiB free.

## Large locations inventoried

- E: `FDP125_STRICT_FORMAL_CLOSURE_V1` about 30 GiB; `ICLR_FDP125_PAPER_EVIDENCE_V1` about 11 GiB; Qwen2.5-7B about 15 GiB.
- F: `FDP125_DIRECTION_CORRECTED_V2` about 28 GiB; `fdp_component_subset_search_v2` about 16 GiB; `fdp_clean_ablation_v1` about 2.6 GiB; Llama-3.1 about 15 GiB.
- D: original `/diskD/jiaqizhang/Llama-3` allocated 16,069,853,184 bytes; several historical basis and quality-result directories. D was not modified.

## Candidate-level dependency audit

| Path | Size | Experiment / quality state | Needed by current stability stage? | Rematerialization / metadata status | Decision |
|---|---:|---|---|---|---|
| `/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/materialized_formal_checkpoints/medium/official_uniform/final_model` | 15,804,357,213 file bytes; 15,804,407,808 allocated | Frozen Llama-3 Medium official-uniform cell, quality results complete | Not the required Uniform-EB=.060 comparator; current cell is EB=.0567 | Lock and hash metadata exist, but exact basis/build inputs are not proven safely reconstructable | KEEP; formal evidence dependency |
| `/diskE/jiaqizhang/FDP125_STRICT_FORMAL_CLOSURE_V1/materialized_formal_checkpoints/medium/fdp125_direct/final_model` | 15,803,766,353 file bytes; 15,803,822,080 allocated | Frozen Llama-3 Medium FDP125 cell, quality results complete | Not a substitute for the missing Uniform060 Pass-1 comparator | Lock and hash metadata exist; exact basis rematerialization source is not proven | KEEP; formal evidence dependency |
| `/diskD/jiaqizhang/Llama-3` | 16,069,853,184 allocated | Original Meta-Llama-3-8B checkpoint | Required as canonical base for any exact reconstruction | Original checkpoint; explicitly protected | KEEP; never delete |
| `/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_eb/eigen_basis_llama3_8b_wikitext2_eb0p06.pt` | 119,521,053 file bytes | Candidate Uniform basis; report found in directory records EB=.058 and `per_layer_eb=false`, not an exact bound to the historical Pass-1 checkpoint | Possible candidate only; identity/rank schedule not established | SHA-256 `1f38625bed2c9caa6efd4822ef96ec4151cd90173d7040e679e562d65b174dc8`; no matching frozen Uniform060 rank-schedule binding found | KEEP pending comparator audit |
| `/diskD/jiaqizhang/llama3_8b_eigen_basis_wikitext2_frag_eb060/eigen_basis_llama3_8b_wikitext2_frag_perlayer_eb.pt` | 120,196,953 file bytes | Adaptive/per-layer-EB basis, not Uniform060 | Historical profile provenance only; not a valid Uniform060 substitute | SHA-256 `f4a6755020313702c4f85d09b0a0a445e43dfd34d598489d23ca149413a0ad52` | KEEP; protocol differs |
| `/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb060.json` | 27,055 file bytes | Original d1/d2/d5 profile input | Required reference O | SHA-256 `d0d5d7653caae9c272d97ae38ef4dd0c3718573bde3250df650c028ca828dba4`; profile stores only per-layer aggregates | KEEP; evidence source |
| `/diskF/jiaqizhang/fdp_clean_ablation_v1/tiers/ultra/pass1_uniform` | 9,220,096 allocated | Failed Uniform-EB=.120 Pass-1 attempt; no model weight shards | Not the Uniform060 comparator; useful failure evidence | Log records `error_budget=0.12`, `nsamples=64`, seed 2, then RoPE device mismatch; config/log only | KEEP logs/metadata; not a deletion target |
| `/diskF/jiaqizhang/fdp_clean_ablation_v1` | about 2.6 GiB | Historical profile, rank-search, and failure artifacts | Contains the original profile and supporting audit trail | Mixed dependencies; cannot safely prune as a whole | KEEP |
| `/diskF/jiaqizhang/FDP125_DIRECTION_CORRECTED_V2/llama3_8b_medium_v2` (within the larger V2 tree) | Part of about 28 GiB V2 tree | Separate V2 diagnostic lane | Its report was used only for a clearly labeled structural diagnostic; not a canonical Uniform060 source | Mixed evidence and failed-attempt dependencies | KEEP pending closure |

## Cleanup verdict

- Deletion candidates verified safe: **0**.
- Deleted artifacts: **none**.
- Space freed: **0 bytes**.
- The current Uniform060 materialized checkpoint at `/diskE/jiaqizhang/llama3_8b_eigen_kv_eb060/final_model` is absent. The F-side `pass1_uniform` directory is an EB=.120 failed attempt, not a copy. The available D-side `.06` basis is not bound to the missing checkpoint/rank schedule. Consequently, no large model or basis is safe to remove before the comparator and profile dependencies are resolved.

This audit is inventory only; it does not authorize deleting any D-drive data and does not claim the profile-stability stage is complete.
