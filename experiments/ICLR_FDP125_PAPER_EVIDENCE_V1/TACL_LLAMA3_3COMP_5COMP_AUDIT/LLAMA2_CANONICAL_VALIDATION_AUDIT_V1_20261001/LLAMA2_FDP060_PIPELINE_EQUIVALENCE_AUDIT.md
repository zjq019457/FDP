# Llama-3 / Llama-2 FDP canonical pipeline validation

**Overall status: PIPELINE_NOT_YET_PROVEN.** This is a provenance and diagnostic audit. It does not change frozen method parameters, schedules, historical scores, or formal cell authorization.

## Executive finding

Both lanes share the intended method family: raw d1+d2+d5, layerwise min-max q, and TRUE_TWO_PASS_FDP_EB, with the same 2×512 BF16/eager profile settings and the same two prompt hashes. This supports method-family and profiler-setting similarity, not guaranteed end-to-end equivalence. Llama-3 replays q and its selected schedule from a historical raw profile whose Pass1 compressed checkpoint is absent; its audit says full Pass1-to-schedule replay was not reacquired. Llama-2 profiles a newly materialized Uniform(e=.060) model on a model-native full-capacity basis. Llama-2 also has an unresolved artifact-level L0 epsilon warning and a parity test whose scope conflates compression approximation with implementation parity.

The direct answer to whether the pipeline is guaranteed identical to Llama-3 is **no, not yet**. The method contract is shared, but provenance and realization gates are not all closed.

## Pipeline comparison

See LLAMA3_LLAMA2_FDP_PIPELINE_MAP.csv for per-stage evidence.
- Llama-3: meta-llama/Meta-Llama-3-8B, fingerprint b6d5136c429a3df995f73537f05abb658bbc0a63203338930fdea9e8f64e7ae0. This is Llama-3, not 3.1.
- Llama-2: meta-llama/Llama-2-7b-hf, fingerprint a627fb4d2200089590b109ad82c35098778aa5856660cc691f6626d25c2cf25197; tokenizer is bound in the Llama-2 input manifest.
- Shared method lock SHA: 8a8e95ae4df92203b5ad9afdbec4ed244004b57750e81a41b2f24bc5b880df4f.
- The Llama-3 three-component selection is post-hoc relative to the original five-component selection freeze; it is the current Llama-3 Medium schedule, not a preregistration.

## Llama-2 rank-lattice and allocation diagnostic

The candidate uses the saved 32-layer schedules and per-layer error trace. Both KQ and V ranks are in 64–4096 and multiples of 32 for all 32 layers. This is a candidate schedule check, not a formal resource lock.
- Spearman(q, −effective epsilon)=0.9304 (descriptive; monotone mapping is expected by construction).
- Spearman(q, ΔKQ rank)=0.4419; Spearman(q, ΔV rank)=0.4350.
- Spearman(q, rank-derived layer-byte delta)=0.4419; n=32 layers, descriptive only.
- Receivers: L4, L5, L31. Donors: L3, L21. Unchanged: 27 / 32.
- Highest-q layers: L31, L11, L5, L7, L4, L9, L12, L6, net rank-byte proxy +7,733,248 B. Lowest-q layers: L0, L1, L13, L2, L23, L21, L30, L20, net proxy -2,097,152 B.
- Only 3 / 8 top-q layers receive additional rank proxy; others are unchanged. Bottom-q has 1 donor(s), with the rest unchanged.
- Aggregate rank-derived proxy: Uniform 784,596,992 B; FDP 788,922,368 B; Δ +4,325,376 B (+0.5513%). This is a rank-derived proxy; the separate materialization audit reports realized FDP060 B_live=788,922,368 B.

The direction signal is positive but sparse: 3 receivers, 2 donors, 27 unchanged. It does not establish causality and no rank edits were made. L31 is a strong q outlier (q=1.000000) and receives +5,636,096 B in rank proxy. L0 effective epsilon is .060. The q artifact separately reports pre-override helper epsilon .06144907469380756, so the intermediate-to-effective L0 lineage needs one canonical ruling before formal resource-lock authorization.

## Resource and quality results

The Llama-2 Official candidate is 799,408,128 B. FDP060 realized B_live is 788,922,368 B: FDP−Official=−10,485,760 B (−1.31169%), outside the 0.15% match rule. Thus the Official-vs-FDP060 PPL/accuracy comparison is not resource matched.

The completed near-match diagnostic is Uniform EB=.059 vs FDP EB=.060:
- B_live: 789,577,728 vs 788,922,368 B; Uniform−FDP=+655,360 B (+0.0830703% relative to FDP), within the pair-specific 0–0.15% gate.
- This changes both allocator (Uniform vs FDP) and EB severity (.059 vs .060). It is diagnostic, not an Official-vs-FDP formal cell.
- Wiki PPL 7.9245101 vs 7.9725864; C4 PPL 9.6905768 vs 9.8114214; J 2.1705572 vs 2.1797781. Lower is better; this diagnostic favors Uniform .059 on all three.
- Five-task equal-task macro: Uniform .059 56.6271% vs FDP060 56.4681%; Δ Uniform−FDP=+0.1590 pp, N=16,695.
- Paired task-stratified bootstrap: 5,000 replicates, seed 20260921, 95% percentile CI [-0.1927, +0.5144] pp. It includes zero. Exact doc_id, target and document alignment passed for all paired examples.

| Task | N | Uniform .059 correct (accuracy) | FDP060 correct (accuracy) | Delta (pp), U−FDP | Discordant U-only/FDP-only |
|---|---:|---:|---:|---:|---:|
| piqa_localjsonl | 1,838 | 1,375 (74.810%) | 1,372 (74.646%) | +0.163 | 12/9 |
| winogrande_localjsonl | 1,267 | 794 (62.668%) | 792 (62.510%) | +0.158 | 33/31 |
| arc_easy_localjsonl | 2,376 | 1,470 (61.869%) | 1,468 (61.785%) | +0.084 | 29/27 |
| arc_challenge_localjsonl | 1,172 | 412 (35.154%) | 410 (34.983%) | +0.171 | 15/13 |
| hellaswag_localjsonl | 10,042 | 4,884 (48.636%) | 4,862 (48.417%) | +0.219 | 47/25 |

These scores and CI are diagnostic-only because the pair is not a resource-matched Official-vs-FDP cell. They must not be relabeled as authorized Llama-2 Medium results.

## Reconciliation issues

1. The L2 q artifact flags an audit because its helper's intermediate L0 epsilon is .06144907469380756, while the candidate schedule and imported L3 lock freeze effective L0 at .060. Keep both immutable; reconcile the stage semantics.
2. The L2 candidate records proxy-only forecast and pending resource lock. A separate materialization audit reports actual B_live. Bind them explicitly in a versioned lock rather than silently editing either artifact.
3. One L2 parity artifact fails at relative-L2 .00167549 comparing dense-projected with compressed outputs, which includes expected approximation error. A separate materialization/operator audit reports factor-load agreement and runtime-vs-composition relative-L2≈4.67e−7. The scopes differ; no runtime bug or unambiguous parity PASS is established.
4. The .059/FDP060 PPL and zero-shot runs completed but assert no formal cell authorization.

## Provenance anchors

| Evidence | SHA-256 |
|---|---|
| Llama-3 provenance | 48381029d32211b73ff987ab067b020878e7a162a2653a4305611630807588fd |
| Llama-3 replay | 9fa18d15596c4cb8d3047ed90adf8670383bc050721818d32d8a262f2b31b147 |
| Llama-3 three-component method lock | 8a8e95ae4df92203b5ad9afdbec4ed244004b57750e81a41b2f24bc5b880df4f |
| Llama-2 profile protocol binding | 99f19dd8a91e6833a3e5eebb5f92c871de68fb9c26c1fc78f31988a478369d1b |
| Llama-2 profile | dcd72dc1e7df1cfb80006a617a8fb15f55621fc4a15febf5083058838f87014c |
| Llama-2 q artifact | 435cd3c9fdab5b7ada048854bcd4dc51afdbc4bbcfc3d36f3843b1b10be4f97c |
| Llama-2 schedule candidate | 17d3a63e9a16af94c4279bfd3f56720fefcbe83b6e317980adf043386ce59025 |
| Uniform .059 zero-shot aggregate | ef3a0e6b96a3449c8a2980066830280e14d5009b90d04f6311d74107ea5c7018 |
| FDP060 zero-shot aggregate | c06eb28c091e5d8fc84f7dab854344e7633c8bf92fc44bcc0ca4bd199d90deea |

D drive is authorized up to 60 GB. The temporary diagnostic model currently occupies about 12.8 GB; no cleanup was performed. Heavy materialization still requires live capacity checks.
