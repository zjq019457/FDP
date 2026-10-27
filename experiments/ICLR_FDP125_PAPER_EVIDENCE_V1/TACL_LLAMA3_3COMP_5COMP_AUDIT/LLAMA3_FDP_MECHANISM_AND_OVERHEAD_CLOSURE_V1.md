# Llama-3 FDP mechanism and overhead evidence — status report

Status: **BLOCKED — `COMPARATOR_BINDING_BLOCKED`**. This is an interim evidence-status report, not a claim that the requested stability/structural/overhead closure passed. No PPL, C4, zero-shot, or downstream quality evaluation was run in this stage.

Primary identity: `meta-llama/Meta-Llama-3-8B`, local content fingerprint `b6d5136c429a3df995f73537f05abb658bbc0a63203338930fdea9e8f64e7ae0` (not Llama-3.1).

Frozen q source: `/diskF/jiaqizhang/fdp_clean_ablation_v1/raw_profiles/step1_llama3_8b_eb060.json`, SHA-256 `d0d5d7653caae9c272d97ae38ef4dd0c3718573bde3250df650c028ca828dba4`. The pinned profile code SHA is `df5273a707fcc6006eaee00dfc0e7848bf3b29eee122a20115af7c74ba0ab75a`. It records BF16/eager, `num_seqs=2`, `seqlen=512`, no explicit seed, and `prompts_file=null`; the code resolves this to the only two hard-coded `DEFAULT_PROMPTS` (Roman Empire and quantum mechanics), then truncates/pads each to 512 tokens. The source stores per-layer aggregates, not per-sequence component values or token/sample manifests.

## 1. Profile stability — BLOCKED

The original profile O is reproducible as a saved aggregate, but it is not a split-level observation: there are only two hard-coded prompt strings and no per-sequence raw metrics. Therefore O cannot be split into independent halves, and three disjoint A/B/C pairs cannot be drawn from that same source. Selecting a different corpus for A/B/C would introduce a calibration-domain change relative to O and needs an explicitly frozen protocol.

The expected Pass-1 model path `/diskE/jiaqizhang/llama3_8b_eigen_kv_eb060/final_model` is absent. The only F-side `pass1_uniform/final_model` found belongs to an EB=.120 run (`nsamples=64`, seed 2) that failed with a RoPE CPU/CUDA device mismatch; it has no model weight shards and is not Uniform060. Do not use it as the comparator.

No A/B/C manifests were frozen, and no GPU profiling was launched.

## 2. Structural vs functional — DIAGNOSTIC ONLY, not same-protocol evidence

An exploratory cross-artifact calculation on 32 layers compared q from the profile above with `layer_error` and selected rank sums in `llama3_8b_medium_v2_report.json` (report SHA-256 `6561ed10a2f3435ddf3c92ae154fbdcab354b1ab87e4f89fdb8355c07f9f1ce5`). That report is a separate Uniform EB=.0567 V2 basis/search lane using WikiText-2 train, 512 samples × 2048 tokens; q O uses two fixed 512-token prompts and an absent historical Uniform060 model.

| Descriptive metric | Value |
|---|---:|
| Spearman(q, V2 uniform `layer_error`) | 0.0539 |
| Spearman(q, V2 selected QK+V rank sum) | 0.1989 |
| Top-8 q / top-8 structural-error overlap | 2/8 (layers 2, 12) |
| Bottom-8 q / bottom-8 structural-error overlap | 1/8 (layer 29) |

These values must remain `DIAGNOSTIC_ONLY`: protocol, EB, sample geometry, basis, and comparator binding differ. They do not establish structural/functional independence and are not paper-ready.

## 3. Offline overhead — PARTIAL MICROBENCHMARK; full overhead BLOCKED

The in-memory q-only microbenchmark recomputed `MinMax(d1+d2+d5)` for 32 layers 20,000 times on an Intel Xeon Gold 6430 host (128 logical CPUs): median **13.09 μs**, P95 **13.23 μs** per q aggregation. This is the specified **q aggregation microbenchmark** only; it excludes input I/O, model forwards, functional profiling, and rank/resource search. It is not total FDP overhead.

No historical profile wall time, GPU-active time/GPU-hours, peak allocated/reserved VRAM, or CPU-memory log is bound to O. A trial replay against the available V2 threshold traces produced rank SHA `f077803eb276b2179d26d768df1e99be111b27bb7faa48ee03e38b8a1de9cedb` and `B_live=201,523,200 B`, which do not match the frozen Llama-3 Medium FDP schedule SHA `f0c462759f53f77214eb7fdce65ea311b5890fd0cd7c6c3eaa061f4f07bf211b` / `199,753,728 B`. It is invalid for timing the frozen schedule and is explicitly excluded. Canonical EigenAttention basis construction is not counted as FDP-incremental cost.

## 4. Paper-readiness classification and blocker

| Evidence | Classification |
|---|---|
| Original q profile source/hash and exact two-prompt geometry | `APPENDIX_READY` as provenance disclosure; not stability evidence |
| q aggregation CPU microbenchmark | `DIAGNOSTIC_ONLY` |
| Cross-artifact structural association above | `DIAGNOSTIC_ONLY` |
| O/A/B/C profile stability and rank stability | `BLOCKED` |
| Same-protocol q vs native reconstruction error | `BLOCKED` |
| Full incremental profile + mapping + rank-search overhead | `BLOCKED` |

The strict gate is the exact historical Uniform EB=.060 Pass-1 comparator: model/checkpoint fingerprint, canonical basis, frozen rank schedule SHA, B_live, and builder/operator binding. Current formal-cell models are a different EB=.0567 operating point; the D-side `.06`-named basis candidate is not bound to the missing Pass-1 checkpoint/rank schedule, while the per-layer `frag_eb060` basis is adaptive rather than uniform. Until the exact chain is restored or deterministically reconstructed and canaried, this report remains `COMPARATOR_BINDING_BLOCKED`.

## Disk cleanup status

See `SAFE_DISK_CLEANUP_AUDIT.md`. No artifacts were deleted; no safe deletion target was proven. At audit time E had 14,185,213,952 B free (below the 15-GiB heavy-stage gate), F 3,732,041,728 B, and D 6,276,386,816 B. Preserve the candidate bases, original profile, current formal checkpoints, logs, and source model until dependencies are resolved.
