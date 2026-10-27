# PALU FDP Native Residual Search V3 — Method Card

Status: frozen before V3 calibration. Independent branch `PALU_FDP_NATIVE_RESIDUAL_SEARCH_V3`; no V1/V2 file is modified. Search resolution changes; PaLu’s legal rank quantum does not.

## Fixed substrate and objective
- Official Llama-2-7B-hf model/tokenizer and PaLu native K/V group action space; ratio 0.5, 512 units, legal ranks 32..512 in multiples of 32.
- Phase 1 is the frozen V1 Native allocation, schedule SHA recorded in the manifest, with 10 accepted transfers. V3 never reselects or rewrites that starting point.
- Every accepted single/composite state must have rank sum 131072, factor bytes 1207959552, and prefill/decode cache each 536870912 bytes.
- `J=0.5*mean Wiki NLL + 0.5*mean C4 NLL`, official next-token CE semantics, 2048-token windows, finite scores only.

## Disjoint splits
`V3_SPLIT_BINDING.json` binds separate 8 Wiki + 8 C4 chunks for PROFILE, CALIBRATION-TUNE and CALIBRATION-LOCK. Wiki intervals are non-overlapping and disjoint from V1 exchange calibration; C4 line IDs are disjoint from V1 exchange and V2 profile. All calibration uses train splits only. FORMAL-WIKI/C4 reuses the frozen official V1 test/validation manifest and is inaccessible until after lock. RULER is deferred; before any RULER run a versioned task/scorer/sample manifest must be audited and frozen.

## Profile marginals and receiver rule
At V1-final, profile every legal unit: `phi_minus(u)=(J_PROFILE(remove32(u))-J_PROFILE(base))/(32/131072)` and `phi_plus(v)=(J_PROFILE(add32(v))-J_PROFILE(base))/(32/131072)`. For phi_plus stability, compare per-unit ranks from PROFILE chunks 1-4 vs 5-8. Use phi_plus only if >=95% of add-eligible units are finite in both halves and Spearman rho>=0.80; otherwise use the existing phi_minus receiver ordering and label `RECEIVER_RULE_CONTROLLED_VARIANT`. This gate is fixed before seeing quality results.

## Candidate ranking and TUNE
At each state select D lowest phi_minus donors and R best receiver candidates (lowest phi_plus, or frozen fallback). Enumerate all legal donor/receiver pairs, at most D*R<=256; rank by additive predicted delta then deterministic donor/receiver unit-name tie-break. D/R grid is 4/4, 8/8, 16/16. Rmax grid is 0.25%, 0.5%, 1%, 2%, where `R=0.5*L1(current-V1_start)/131072`; Rmax is a ceiling, never a target. Depth-2 arms also test beam widths 4 and 8.

For each TUNE iteration evaluate the full single-pair candidate pool; do not stop at the first improvement. A valid move requires exact resources, actual `DeltaJ<0`, and paired bootstrap 95% CI upper<0 (10,000 resamples). If multiple pass, take the most negative actual tune DeltaJ, then lexical donor/receiver tie-break. Rejected pair tabu is keyed by allocation SHA; after state change it can be revisited. B/B1-refresh arms reprofile affected modules after every accepted single and all units after every fourth accepted move. Non-B1 arms retain current scores except the same every-four global refresh.

## Frozen arms
A: broad single-pair search, no affected-only refresh. B: broad + B1 affected refresh. C: broad + depth-2 beam, no affected-only refresh. D: broad + B1 refresh + depth-2 beam. Optional E is `COMPOSITE_CYCLE_ABLATION`, exploratory only and never promoted by default.

Depth-2 starts only after a complete single-pair pool has zero strict-valid moves. Build up to beam-width predicted first moves, enumerate legal second moves, and evaluate the combined final allocation as one atomic move. The intermediate e1 state is never accepted/installed. The final composite must pass the same strict gate against its pre-composite state.

TUNE selects one configuration per arm using TUNE only. CALIBRATION-LOCK replays the frozen selected trajectory without re-optimizing; it passes only if the final V3-vs-V1 paired DeltaJ is negative with CI upper<0 and resource exact. If multiple arms pass, choose simplest, then smaller beam, lower Rmax, fewer composites—not the lowest lock J. If none pass, mark `NO_V3_GO_QUALITY` and do not run formal V3 PPL.

## Formal and downstream gates
Only after a separate LOCK pass, freeze the chosen trajectory and run the existing official formal Wiki/C4 protocol. Report V3 vs V1 and vs Fisher with paired CIs; never tune after formal. If accepted transfers>10 or R>V1 R, run 3 Random-Matched-V3 seeds matched on transfer count, rank quantum, approximate R and exact resources. RULER only after positive independent lock/formal signal and its own protocol manifest freeze. No V3 LongBench.

Never change rank quantum, lower acceptance threshold, use formal samples during search, or alter frozen V1/V2 results. Preserve failures/partials and mark split overlap, resource mismatch, nonfinite values or runtime failures as blocked.
