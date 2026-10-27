# XKV_FIXED_RUNTIME_NATIVE_MARGINAL_V1

Independent mechanism-validation branch for the repaired xKV runtime.

- `RUNTIME_FIXED=true`
- `OLD_COLLAPSE_RESULTS_MODIFIED=false`
- `FORMAL_803X_RESULTS_MODIFIED=false`
- `foreign_jobs_interrupted=0`
- Native unit: 8 groups, each exactly 4 layers; no 32-layer claim.

## Provenance

Old profile classification: `LEGACY_BROKEN_RUNTIME_FRAGILITY`. old profile carries no proof of both repaired runtime fixes and predates/does not bind to the current fixed-runtime audit; excluded from current paper claim.
Repaired runtime: transpose fix=True, decode q_len=1 causal fix=True.

## Frozen native perturbation and resource formula

Control is K=384, V=576, W=4. Each probe changes only one group to K=352, V=528; the preflight gate verified requested/effective ranks before quality.
Official formula source: `/diskF/jiaqizhang/fdp_lowrank_operator_zoo_v1/repos/xkv_upstream/scripts/calc_kv_comp_rate.py` (SHA256 `a71c1b0fc833aff8837add29996f22a4925fc60dd9162bba31cf583256bd71f6`). The 65,536-token per-group factor bytes are 133,693,440 control and 122,552,320 perturbed; delta 11,141,120.

## Frozen q to group rule

q_group[g]=mean(q[4g:4g+4]); h_group=mean(1+1.2*q)=1+1.2*q_group; native projection retains 8 constant four-layer groups

## Per-group results

| group | layers | q_group | K/V control→perturb | factor bytes control→perturb | J control | J perturb | delta_J | phi_xkv |
|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|
| 0 | 0–3 | 0.412827 | 384/576 → 352/528 | 133,693,440 → 122,552,320 | 2.119690094 | 2.113563599 | -0.006126495 | -0.073517941 |
| 1 | 4–7 | 0.683174 | 384/576 → 352/528 | 133,693,440 → 122,552,320 | 2.119690094 | 2.126646770 | 0.006956677 | 0.083480122 |
| 2 | 8–11 | 0.628362 | 384/576 → 352/528 | 133,693,440 → 122,552,320 | 2.119690094 | 2.121648663 | 0.001958569 | 0.023502827 |
| 3 | 12–15 | 0.613195 | 384/576 → 352/528 | 133,693,440 → 122,552,320 | 2.119690094 | 2.119959108 | 0.000269015 | 0.003228174 |
| 4 | 16–19 | 0.438158 | 384/576 → 352/528 | 133,693,440 → 122,552,320 | 2.119690094 | 2.121231442 | 0.001541348 | 0.018496179 |
| 5 | 20–23 | 0.165089 | 384/576 → 352/528 | 133,693,440 → 122,552,320 | 2.119690094 | 2.120697103 | 0.001007009 | 0.012084108 |
| 6 | 24–27 | 0.168526 | 384/576 → 352/528 | 133,693,440 → 122,552,320 | 2.119690094 | 2.120670626 | 0.000980532 | 0.011766386 |
| 7 | 28–31 | 0.144037 | 384/576 → 352/528 | 133,693,440 → 122,552,320 | 2.119690094 | 2.121105930 | 0.001415837 | 0.016990041 |

## Alignment

Spearman rho = **0.452381**; Pearson r = **0.282237**.
q rank high→low: `[1, 2, 3, 4, 0, 6, 5, 7]`
native phi rank high→low: `[1, 2, 4, 7, 5, 6, 3, 0]`
top-2 overlap: `[1, 2]` (2/2); bottom-2 overlap: `[]` (0/2).

This audit does not change official xKV 8.03x or xKV+FDP allocation. Any disagreement is reported only as alignment evidence; no q re-search or NativePhi allocation is performed.
