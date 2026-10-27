# FDP25 Existing-Result Audit and Execution Task Table

- generated_at: `2026-09-17T15:33:42`
- source of truth: `authority/FDP125_MODEL_AUTHORITIES.json` and `authority/FDP125_OPERATING_POINTS.json`
- scope: current `/diskF/jiaqizhang/fdp125_final_campaign_v1`; no experiment launched by this audit

## Authority gate

- formula: `{"components": ["d1", "d2", "d5"], "eta": 0, "lambda": 1.2, "rho": 0.75, "weights": [1, 1, 1]}`
- selection policy: resource-only; no quality-driven retuning
- storage policy: compact reports/results on diskF, heavy scratch on diskE, no new diskD writes
- code provenance: `4 files SHA256-bound; git_commit=UNAVAILABLE`

## Audited current results

| Model/tier | Method | B_live | Wiki PPL | C4 PPL | Windows | Validity | Resource gap vs D125 actual |
|---|---:|---:|---:|---:|---|---|---:|
| llama/aggressive | FDP25_D125 | 159088640 | 10.157443283195123 | 17.757530416687768 | wiki:141;c4:256 | VALID_COMPLETE | 0.000% |
| llama/aggressive | Official_EigenAttention_Uniform | 152371200 | n/a | n/a | wiki:None;c4:None | RESOURCE_ONLY_FAILURE | -4.222% |
| llama/aggressive | Official_EigenAttention_Uniform | 163512320 | 11.82206991111223 | n/a | wiki:141;c4:None | INCOMPLETE_RUNNING_METADATA | 2.781% |
| llama/aggressive | Official_EigenAttention_Uniform | 163512320 | 11.82206991111223 | 18.1513740274796 | wiki:141;c4:256 | VALID_COMPLETE_STALE_STATUS | 2.781% |
| llama/aggressive | Parent_5COMP | 158105600 | 10.318369519877693 | 18.069218416678243 | wiki:141;c4:256 | VALID_COMPLETE | -0.618% |
| llama/aggressive | Legacy_Uniform | 155090944 | 13.189824626309454 | 20.901409580429988 | wiki:141;c4:256 | VALID_COMPLETE | -2.513% |
| llama/dense | Dense | n/a | 6.138120287244493 | 8.59711311681018 | wiki:141;c4:256 | VALID_COMPLETE | n/a |
| llama/light | FDP25_D125 | 218923008 | 6.633935706456411 | 9.67960910323553 | wiki:141;c4:256 | VALID_COMPLETE | 0.000% |
| llama/light | Official_EigenAttention_Uniform | 216793088 | n/a | n/a | wiki:None;c4:None | RESOURCE_ONLY_FAILURE | -0.973% |
| llama/light | Official_EigenAttention_Uniform | 217219072 | n/a | n/a | wiki:None;c4:None | RESOURCE_ONLY_FAILURE | -0.778% |
| llama/light | Official_EigenAttention_Uniform | 219971584 | 6.693169316647977 | n/a | wiki:141;c4:None | INCOMPLETE_RUNNING_METADATA | 0.479% |
| llama/light | Official_EigenAttention_Uniform | 219971584 | 6.693169316647977 | 9.739063662997475 | wiki:141;c4:256 | VALID_COMPLETE_STALE_STATUS | 0.479% |
| llama/light | Parent_5COMP | 218562560 | 6.651764536184463 | 9.70315768089166 | wiki:141;c4:256 | VALID_COMPLETE | -0.165% |
| llama/light | Legacy_Uniform | 217219072 | 6.728769332634484 | 9.833558527467364 | wiki:141;c4:256 | VALID_COMPLETE | -0.778% |
| llama/medium | FDP25_D125 | 199753728 | 7.1594697724543135 | 11.99871051780785 | wiki:141;c4:256 | VALID_COMPLETE | 0.000% |
| llama/medium | Official_EigenAttention_Uniform | 200245248 | 7.377171453713626 | 13.381924273098633 | wiki:141;c4:256 | VALID_COMPLETE_STALE_STATUS | 0.246% |
| llama/medium | Official_EigenAttention_Uniform | n/a | n/a | n/a | wiki:None;c4:None | INCOMPLETE_RUNNING_METADATA | n/a |
| llama/medium | Official_EigenAttention_Uniform | n/a | 7.405433834542061 | 13.458814102228596 | wiki:141;c4:256 | QUALITY_COMPLETE_UNBOUND | n/a |
| llama/medium | Official_EigenAttention_Uniform | n/a | n/a | n/a | wiki:None;c4:None | INCOMPLETE_RUNNING_METADATA | n/a |
| llama/medium | Official_EigenAttention_Uniform | 202932224 | 7.1973638604816 | 10.727777142812508 | wiki:141;c4:256 | VALID_COMPLETE_STALE_STATUS | 1.591% |
| llama/medium | Official_EigenAttention_Uniform | n/a | n/a | n/a | wiki:None;c4:None | INCOMPLETE_OR_UNBOUND | n/a |
| llama/medium | Official_EigenAttention_Uniform | 202932224 | 7.1973638604816 | 10.727777142812508 | wiki:141;c4:256 | VALID_COMPLETE_STALE_STATUS | 1.591% |
| llama/medium | Parent_5COMP | 198377472 | 7.161075194201872 | 12.074390587688276 | wiki:141;c4:256 | VALID_COMPLETE | -0.689% |
| llama/medium | Legacy_Uniform | 196083712 | 7.591532568758708 | 13.107604796696982 | wiki:141;c4:256 | VALID_COMPLETE | -1.837% |
| qwen/dense | Dense | n/a | 6.850251410497807 | 10.37048328329531 | wiki:146;c4:256 | VALID_COMPLETE | n/a |

## Evidence conclusions

- Llama D125 is complete for Light/Medium/Aggressive: paired Wiki/C4, Wiki 141/141, C4 256/256, skipped=0, and common evaluator hash.
- Llama official Uniform has complete comparator artifacts for Light v4, Aggressive v4, Medium v5, and independent Medium v5 confirmation; these are official-path results. The older official variants with one dataset missing are not used as paired evidence.
- The official Uniform comparator is not uniformly strict-matched to the realized D125 bytes: the completed Medium v5 is above the realized Medium D125 by about 1.591%; the tighter 0.0565 cell is above by about 0.246% and is complete.
- Several `STATUS.json` files remain `RUNNING` even when `DONE.json` is `PASS`; this is stale metadata, not evidence of an active campaign worker and not a reason to rerun.
- Qwen currently has Dense only in this campaign; OPT has no bound operating point. Controls, ablations, long-context, external/system evidence and unified statistics remain open.
- CPU binding found 3 LongBench/long-context candidates, 8 external-baseline roots, and 20 OPT historical candidates; all remain binding-pending and were not relabeled as valid.
- `FDP25_AGGRESSIVE_UNIFORM_RESOURCE_LATTICE.csv` is a plan: only EB=0.085 is an existing measured state (2.781% above D125); all other entries are not measured and remain space-gated.
- Code provenance is bound by SHA256 for the four campaign-critical scripts; no usable Git commit was found in the source tree, so reproducibility is partial until a commit identity is supplied.

## Execution rule

- First execute only P0 CPU/read-only binding and report generation (done in this pass).
- GPU dispatch is held until E-space and strict-match gates are rechecked; foreign tasks are not interrupted.
- Do not repeat a cell with `VALID_COMPLETE` or `VALID_COMPLETE_STALE_STATUS`; preserve all v1 artifacts.

## Task table

| Priority | ID | Scope | Status | Next action | Launch |
|---|---|---|---|---|---|
| P0 | P0-A | Primary method authority | DONE | bind this manifest to all new records | YES |
| P0 | P0-B | Existing result registry + integrity gate | DONE | use registry as dispatch gate; do not duplicate valid cells | YES |
| P0 | P0-C | Stale status metadata reconciliation | PARTIAL | record stale metadata as audit issue; do not rerun or overwrite PPL | YES |
| P0 | P0-D | Code provenance closure | PARTIAL | retain immutable script hashes and record commit absence; obtain commit only if repository owner supplies it | YES |
| P1 | P1-A | Llama-3-8B D125 primary Wiki/C4 | DONE | freeze as development evidence; write claim bindings | YES |
| P1 | P1-B | Llama official Uniform comparator | WAITING_RESOURCE_GATE | automatic narrow resource-only enumeration; select nearest <=1% state, then quality only for selected state | YES |
| P2 | P2-A | Matched controls (random/depth/middle/spectral/reconstruction/sensitivity) | MANIFESTS_PREPARED | run resource lock/canary only after gate | NO |
| P2 | P2-B | Component ablation D125 minus d1/d2/d5 + Parent + Uniform | MANIFESTS_PREPARED | bind valid rows, then lock/run only D25/D15/D12 missing cells | NO |
| P2 | P2-C | Layer-0/rho/KQ-V/causal marginal audit | MISSING | CPU audit scripts and manifest binding first | YES |
| P3 | P3-A | Long-context + LongBench | PARTIAL | bind existing repaired campaign if provenance matches; otherwise queue minimum run | NO |
| P4 | P4-A | Qwen transfer | PARTIAL | audit binding and free/allocate scratch before materialization | NO |
| P4 | P4-B | OPT transfer | PARTIAL | complete checkpoint/bytes/ranks/protocol/scorer audit; do not GPU-launch | NO |
| P4 | P4-C | External baselines/generalization | PARTIAL | bind existing outputs; avoid duplicate experiments | NO |
| P5 | P5-A | System/hardware benchmark | MISSING | schedule after core/control gates | NO |
| P5 | P5-B | Statistics/reproducibility/claim-evidence | PARTIAL | generate claim matrix from registry; stats only for complete cells | YES |
