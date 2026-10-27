# MPT-7B FDP / Uniform: independent three-band audit

## Decision

Audit gates did not pass. No formal freeze was created and no new formal zero-shot job was launched. Existing accuracy outputs remain diagnostic-only.

Only this new E-drive audit directory was written. Input hashes match before and after.

## Resource

| Band | Raw Uniform actual B_live | FDP actual B_live | Raw gap vs FDP | Relaxed Uniform estimate | Estimated gap | Actual relaxed census |
|---|---:|---:|---:|---:|---:|---|
| U011 | 851,705,856 B | 889,323,520 B | -37,617,664 B (-4.2299%) | 889,454,592 B | +131,072 B (+0.0147%) | pending |
| U015 | 717,094,912 B | 776,077,312 B | -58,982,400 B (-7.6001%) | 776,208,384 B | +131,072 B (+0.0169%) | pending |
| U019 | 608,043,008 B | 660,996,096 B | -52,953,088 B (-8.0111%) | 661,127,168 B | +131,072 B (+0.0198%) | pending |

Relaxed candidates were selected by resource-only rule. +131072 B is one aggregate rank-sum quantum, not one local rank edit: multiple layers move. The relaxed actual cache census is absent.

## PPL diagnostics

Delta is FDP minus relaxed Uniform. Window NLL/token reconstruction passed; J numerically equals half the Wiki and C4 mean NLL sum, but the bound implementation source was not found.

| Band | Wiki Uniform -> FDP | C4 Uniform -> FDP | Delta J | Stored paired 95% CI |
| U011 | 7.989396 -> 7.937105 | 9.847110 -> 9.784709 | -0.006462 | [-0.007657, -0.005391] |
| U015 | 8.354753 -> 8.260596 | 10.423579 -> 10.312049 | -0.011046 | [-0.012711, -0.009592] |
| U019 | 9.228396 -> 8.854795 | 11.661392 -> 11.262840 | -0.038050 | [-0.040791, -0.035488] |

These PPLs are marked formal_matched_quality=false. Each pair has 140 Wiki and 256 C4 windows. Bootstrap: 10000 paired window replicates, seed 2; independent U015/U019 CI bounds differ slightly from stored.

## Existing accuracy outputs

All 30 task outputs exist and ordered doc IDs pair. HellaSwag output metric is acc, while the frozen formal requirement is acc_norm; saved rows do not contain recoverable acc_norm. Thus these are not metric-compliant formal accuracy results. Macro CIs use paired within-task resampling then equal weighting, seed 20261007 (audit-only), 10000 replicates.

| Band | Uniform Avg-Acc | FDP Avg-Acc | Delta pp | Paired CI 95% pp |
| U011 | 58.1202% | 58.1795% | +0.0593 | [-0.3504, +0.4646] |
| U015 | 57.5188% | 56.9511% | -0.5678 | [-1.0298, -0.1191] |
| U019 | 56.0260% | 55.1824% | -0.8437 | [-1.3415, -0.3550] |

Raw target/prediction/log-likelihood evidence is absent; task-level McNemar counts and Holm corrections are in ACCURACY_TASK_DIAGNOSTICS.csv.

## Additional rank-provenance limitation

FDP and relaxed Uniform rank schedules were recovered, checked against the native rank lattice, and their rank SHA values recomputed. Raw Uniform runtime B_live values are present in locks, but this audit did not recover the serialized per-layer raw Uniform schedules matching the recorded hashes. Therefore raw Uniform rank-to-byte geometry was not independently reconstructed here; the recorded actual B_live remains lock-sourced.

## Gate ledger

| Gate | Status | Finding |
| G1 source/config integrity | **WARN** | Content fingerprint and tokenizer/task hashes pinned, but upstream revision and per-run evaluator invocation incomplete. |
| G2 byte accounting | **FAIL** | Relaxed Uniform candidates have predicted bytes only; actual runtime cache census is absent. |
| G3 native legality | **WARN** | Schedules satisfy 32-layer rank lattice. +131072 B is one aggregate rank-sum quantum; multiple layers move, not a single local edit; relaxed runtime shapes missing. |
| G4 sample identity | **PASS** | PPL window/input hashes and ordered zero-shot doc IDs match. |
| G5 PPL recomputation | **PASS** | NLL/token totals and PPL independently recomputed from per-window outputs. |
| G6 J recomputation | **WARN** | Numerical J identity matches half-sum of Wiki/C4 mean NLL; source implementation not located. |
| G7 paired bootstrap | **WARN** | Stored PPL uses paired 10000 resamples seed 2; independent U015/U019 interval bounds differ slightly. Accuracy CI uses audit-only seed 20261007. |
| G8 outcome-conditioned tuning | **PASS** | Selection locks say quality_used_to_select=false and resource-only selection. |
| G9 resource matching | **FAIL** | Raw pairs exceed 1%; relaxed candidate gaps are predictions, not actual resource match. |
| A1 frozen accuracy metric contract | **FAIL** | HellaSwag outputs record acc; frozen formal metric requested is acc_norm, unavailable in saved per-item rows. |

## Next required

1. Materialize each relaxed Uniform candidate and census actual cache tensor shapes and B_live.
2. Confirm the actual resource gap satisfies the required <=1.0% gate; <=0.15% remains preferred.
3. Resolve the HellaSwag metric mismatch without changing the frozen protocol.
4. Bind upstream checkpoint revision, evaluator invocation, J source, bootstrap seed and per-item outputs before any formal freeze/launch.
