# FDP / FDP-EB — Functional Distortion Profiling for low-rank KV-cache compression (reproduction code)

Snapshot 2026-10-27 of all code, configs, frozen schedules and summary results.
**No model weights, compressed checkpoints, EigenAttention bases (*.pt > 20 MB) or HF caches are included.**
Original absolute server paths are preserved inside the tarballs (e.g. `diskF/jiaqizhang/...`), so every
config/log path can be matched 1:1 to the original run.

> **GitHub repository vs full backup.** This repository holds the *lean* tree: code, configs, preregistrations,
> frozen rank/budget schedules and summary / paired-bootstrap JSONs (files < 1 MB, paths with the server prefix
> `diskX/jiaqizhang/` mapped to `experiments/`). Per-window PPL JSONs, per-item zero-shot samples, logs, small eval data
> and the paper source are kept in the full offline archives (`tier1_*.tar.gz`, `tier2_*.tar.zst`) listed below,
> which are not in this repository.

## Contents

| Item | What |
|---|---|
| `tier1_code_configs_results.tar.gz` | All code, configs, preregistrations, frozen rank schedules (`low_rank_config.json`, `per_layer_eb.json`, `RESOURCE_PLAN.json`, `step1.json`), per-window PPL JSONs, summary/paired-bootstrap JSONs, logs, WikiText-2 parquet, C4 eval texts, offline zero-shot task data, QA-6 LongBench manifest/data |
| `tier2_zero_shot_per_item_samples.tar.zst` | Per-item lm-eval samples (`zero_shot_samples/`) — only needed to recompute item-level paired bootstraps |
| `env/ENVIRONMENT_FREEZE.txt` | `pip freeze` + `conda env export` for both envs, torch/CUDA/transformers versions, GPU, OS, git commits |
| `git_diffs/` | `git diff HEAD` of the EigenAttention repo (our modifications vs upstream commit) and git status of PaLu |
| `local_paper_and_scripts/paper_FDP_NAACL_20261011/` | LaTeX source, figure scripts, `RESULTS_PROVENANCE.md` (number → source file map), `REVISION_LOG.md`, PDFs |
| `SHA256SUMS.txt` | checksums of the archives |

Unpack (Linux): `tar -xzf tier1_code_configs_results.tar.gz` and `tar --zstd -xf tier2_zero_shot_per_item_samples.tar.zst`
(Windows: 7-Zip, or `python -m pip install zstandard`).

## Environments (see env/ENVIRONMENT_FREEZE.txt)

* `eigen` conda env: Python 3.9.21, torch 2.8.0+cu128, transformers 4.44.2 — EigenAttention build/eval, PaLu official runner.
* `.venv`: Python 3.12.3, torch 2.11.0+cu130, transformers 4.53.0 — MPT-7B / newer runners.
* Hardware used: NVIDIA A100 80GB PCIe (driver 580.82.07). BF16 everywhere.

Upstream repos (re-clone, then apply our files on top):
* EigenAttention: commit `8bd0d3fac373788aa38b9e7f646cc25109661cdd`; our changes = `git_diffs/EigenAttn_local_modifications.diff` + extra files under `diskD/jiaqizhang/home_moved/EigenAttn/EigenAttn/` in tier1.
* PaLu: `https://github.com/shadowpa0327/Palu.git` commit `bb22666e2ef96707e8dd21d93fc00146c2e0d615` (unmodified; copy also in tier1 under `diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu`, incl. `longbench_utils/scorer.py`).

Models to download again (HF): Meta-Llama-3-8B, Llama-2-7b-hf, mpt-7b. Datasets: WikiText-2 (parquet included), C4 eval texts (included), lm-eval tasks (local copies in `offline_datasets`), LongBench (QA-6 data/manifest in `FDP125_STRICT_FORMAL_CLOSURE_V1/longbench_qa6_v1`).

## Pipeline (Llama-3-8B FDP-EB)

1. **Parent EigenAttention basis / threshold curves**: `EigenAttn/main_eigen_attn.py` + `tools/build_eigen_basis_llama_error_budget_offline.py`
   (512 WikiText-2 seqs × 2048 tok, seed 2, avg_dim groups of 32, threshold step 0.02, svd_threshold_floor 0.30).
   Frozen curves: `diskF/jiaqizhang/llama3_full_resource_robustness_v1/ultra_aggressive_v1/envelope_attempt2/*perlayer_eb_report.json`.
2. **FDP profile** `EigenAttn/step1_diagnose_llama.py` (imports `EigenAttn_base/diagnose/step1.py`): d^Z + d^S + d^H, MinMax → q_l. Two default probe passages padded to 512. Output `step1.json` (frozen copies per tier in tier1).
3. **FDP-EB budgets**: eps_l = eps_base / (1 + 1.2 q_l), mean-normalized, layer 0 = eps_base (`realloc_rank_from_fragility.py`; builder flags `--freeze_layer0_eb --layer0_eb base`).
4. **Exact replay of ranks** on CPU: `FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1/code/c45lib.py` (`select`, `derive_eb`), builder `build_eigen_basis_llama_error_budget_offline_c45.py` (+ `.diff`). Gates require builder ranks == replay.
5. **Materialize**: `main_eigen_attn.py --compress_only --avg_dim 4096 --nsamples 64 --calib_dataset c4`.
6. **Evaluate**: `rebuttal_legacy_reproduction/runners/ppl_instrumented.py` (141 WikiText-2 / 256 C4 windows), `zero_shot_instrumented.py` (5 tasks, 16,695 items), `rebuttal_completion_pipeline/runners/wave2_longbench_eigen.py` (QA-6, N=1,150, max_len 8192). Protocol lock: `LEGACY_EVALUATION_PROTOCOL_LOCK.json`.
7. **Paired bootstrap**: 10,000 reps; PCG64 seeds 20261012 (WikiText-2), 20261013 (C4), 20261014 (items). Scripts: `c45_paired.py`, `c45_qa6.py`, `NAACL_PAPER_SUPPLEMENT_V1/code/*`.

## Paper element → server directory (inside tier1)

| Paper | Directory |
|---|---|
| Table main, no 45% floor (0.82/0.74/0.59/0.56×) + Spectral-tail | `FDP_OFFICIAL_NO45_V2` (`orch_v2.py`, `plan_lib.py`), `fdp125_final_campaign_v1`, `OLD_1003_SPECTRAL_TAIL_V1`, `SEVERE_V2_SPECTRAL_TAIL_V1`, `OLD_1003_SEVERE_REPAIR_V1` (0.56× + rank-trace audit), `NAACL_PAPER_SUPPLEMENT_V1/LLAMA3_UNIFIED_MAIN_TABLE.json` |
| Table main, with 45% floor (0.82/0.75/0.69×), QA-6 | `FDP_FROZEN_PROTOCOL_CONSTRAINED45_V1` (`PREREGISTRATION.json`, `tiers/*`, `PAIRED_*.json`), `NAACL_PAPER_SUPPLEMENT_V1/QA6_MEDIUM_MACRO_PAIRED_CI.json` |
| Rank-floor appendix | `OLD_1003_RANKFLOOR_EXTENSION_V1` |
| Ablation (components) | `fdp_clean_ablation_v1`, `ABL_WOC_MEDIUM_RM_V1`, `ICLR_FDP125_PAPER_EVIDENCE_V1/TACL_LLAMA3_3COMP_5COMP_AUDIT`, `TACL_LLAMA3_MEDIUM_NONFUNCTIONAL_CONTROLS_COMMON_BASIS_V1`, `TACL_LLAMA3_SPECTRAL_CROSS_TIER_V1` |
| MPT-7B PPL (0.83/0.72/0.62×) | `MPT7B_FDP_EB_CAMPAIGN_V1` (U011/U015/U019 `PPL_SUMMARY.json`, `PAIRED_BOOTSTRAP.json`), `DENSE_BASELINES_V1` |
| Llama-2-7B exchange | `LLAMA2_EXCHANGE_EXPANSION_V1` (`code/`, `schedules/`, `results/SUMMARY_PAIRED.json`, `EXCHANGE_TUNING_CURVES.json`, `FLOOR_UNIFORM_REPLAY.json`, `ABSOLUTE_METRICS_FOR_TABLE.json`), historical `13_MODEL_MATRIX_4X3/LLAMA2_E068_*` |
| PaLu-FDP | `PALU_OFFICIAL_PROTOCOL_REPRO_V1`, `PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1`, `PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3`, `PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1`, `PALU_OFFICIAL_TABLE1_REPRO_V1`, `PALU_PAPER_REPRO_AND_FDP_V2` (`palu_fixed_orchestrator.sh`, official repo). Whitening cache `official_whiten_factors_rank512.pt` (2.4 GB) NOT included — regenerate with the official PaLu script. |
| Cost (Sec 6.7, App. F/G) | `FDP_MEDIUM_LATENCY_V1/MEDIUM_LATENCY.csv`, `FDP_MEDIUM_OFFLINE_COST_V1` |
| Supplementary controls (not in paper) | `FLOOR45_CONTROLS_V1`, `FLOOR45_UNIFORM_AT_NOFLOOR_FDP_MEDIUM_V1`, `FDP_PROFILE_ROBUSTNESS_V1` (stopped) |
| Checkpoint deletion record | `CHECKPOINT_CLEANUP_20261016` (sizes + SHA256 of deleted compressed checkpoints) |
| Figures | `local_paper_and_scripts/paper_FDP_NAACL_20261011/paper/figures/make_*.py` |

Exact number-by-number provenance: `local_paper_and_scripts/paper_FDP_NAACL_20261011/RESULTS_PROVENANCE.md`.

## Known gaps
* 6,352 files under `TACL_LLAMA3_SPECTRAL_CROSS_TIER_V1/materialized_v2/...` were unreadable (permission denied, owned by another account) and are not in the bundle; their aggregated results are in the same directory's summary JSONs.
* Model weights, compressed checkpoints, EigenAttention basis tensors and PaLu whitening caches must be regenerated (deterministic: seeds and configs are in the bundle).
