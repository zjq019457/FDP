# MEDIUM_PROTOCOL_AUDIT — FDP_MEDIUM_OFFLINE_COST_V1

Date 2026-10-10. One real, cold, end-to-end Llama-3-8B Medium build. No PPL / zero-shot / LongBench was run.

## Source-level configuration (checked in code, not in protocol documents)

| Item | Source | Value used |
|---|---|---|
| Pipeline | `fdp125_final_campaign_v1/llama_fdp125_tier_worker.py` (sha faf753da…) | same stage order and arguments as the frozen Medium run |
| Builder | `EigenAttn/tools/build_eigen_basis_llama_error_budget_offline.py` (sha e2329411…) | unchanged |
| Calibration | builder sampler | WikiText-2 train, 512 × 2048, seed 2 |
| avg_dim / precision | builder defaults + args | 32 / bf16 |
| Threshold | builder | init 1.0, step 0.02, floor 0.30 |
| 45% official stop | — | not present in this builder (not enabled) |
| min_rank 64 | `--min_rank 64` is inactive because `--strict_original_rank=1` | no rank floor (effective lattice minimum 8) |
| FDP score | worker `derive_eb(step1, 0.060, (1,1,0,0,1))` | d1+d2+d5, MinMax, λ=1.2, mean-normalised; layer 0 = ρ 0.75 geometric then builder `--freeze_layer0_eb 0.060` |
| Step 1 | `step1_diagnose_llama.py --seqlen 512 --num_seqs 2` | original frozen profiling input size (not the 512-sample rank calibration) |
| Materialization | `main_eigen_attn.py --compress_only --avg_dim 4096 --nsamples 64 --calib_dataset c4` | unchanged |
| Profiling parent vs comparator | parent Uniform eb 0.060 (B 196,083,712); matched comparator eb 0.0567 (B 199,917,568) | both built |

## Integrity (all PASS)

| Artifact | New | Frozen | Result |
|---|---|---|---|
| Parent Uniform rank SHA | c922f1af… | c922f1af… | PASS |
| Parent Uniform basis file SHA | b7952632… | b7952632… | PASS (byte-identical) |
| Parent Uniform B_live | 196,083,712 | 196,083,712 | PASS |
| Step 1 profile | per-layer numbers identical (max abs diff 0.0) | | PASS (file SHA differs only in non-numeric metadata) |
| FDP EB vector | eb_sha 2a8f73b4…, max diff 0.0 | 2a8f73b4… | PASS |
| FDP rank SHA | f0c46275… | f0c46275… | PASS |
| FDP basis file SHA | 0e6ac0d4… | 0e6ac0d4… | PASS (byte-identical) |
| FDP B_live / per-layer ranks | 199,753,728 / equal | 199,753,728 | PASS |
| Uniform comparator low_rank_config file SHA | db207077… | db207077… | PASS |
| Uniform comparator B_live | 199,917,568 | 199,917,568 | PASS |

The instrumentation (runpy launcher + external NVML/psutil sampling) did not change execution.

## Measurement conditions (disclosure)

- Instrumentation only: `cost_launch.py` runs each frozen script unchanged via runpy and records torch peak allocated/reserved after `torch.cuda.synchronize()` at exit; `timed_stage.py` timestamps every stdout line and samples NVML per-process GPU memory (only this stage's process tree) and process-tree RSS every 0.5 s.
- Cold start: no basis, error curve or checkpoint was reused; no warm-up run.
- The FDP chain ran on GPU1 and the Uniform comparator on GPU2 at the same time (independent GPUs, shared host CPU and disk). From 11:17 to 12:12 an unrelated build of mine ran on GPU0, and from 12:45 a materialization and evaluation of mine ran on GPU0. Wall times may therefore be slightly pessimistic relative to an idle host; GPU-memory numbers are per-process and are not affected.
- Queue wait time: 0 (both GPUs were idle at launch).
- Disk: one checkpoint ≈ 14.7 GiB. The frozen Uniform-comparator procedure writes two copies (`--save_dir` and `output_dir/final_model`), 29.4 GiB.
