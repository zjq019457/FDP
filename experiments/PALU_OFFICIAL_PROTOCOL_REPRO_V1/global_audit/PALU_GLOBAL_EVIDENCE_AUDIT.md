# PALU GLOBAL SCIENTIFIC AUDIT

Generated: `2026-09-21T04:57:37.928646+00:00`  
Mode: `READ_ONLY_EVIDENCE_AUDIT`; no frozen artifact was modified and no new large experiment was started.

## Global status

- Evidence rows: **48** (18 Official, 30 Historical/diagnostic).
- Official Dense identity: **PASS**.
- Official Uniform 0.7/0.5/0.3: **PASS**.
- Official Automatic/Fisher 0.7/0.5/0.3: **COMPLETE**.
- Official action space and q→8-group mapping: **COMPLETE**.
- Official phi: **0.7 FROZEN; 0.5/0.3 PENDING**.
- Official Prior/Native formal PPL: **0.7 COMPLETE; 0.5/0.3 WAIT_PHI**.
- Official zero-shot: **0.7 Fisher/Prior COMPLETE; Native RUNNING at audit time; 0.5/0.3 WAIT_PHI**.
- `foreign_jobs_interrupted=0`; GPU1 excluded.

## Official 0.7 formal comparison

| Method | Wiki PPL | C4 PPL | J | ΔJ vs Fisher | 95% CI | Rank | Factor bytes | Cache bytes | Resource |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| PaLu Automatic (Fisher) | 5.616322681206661 | 7.589023149806249 | 1.8761900017060429 | 0.0 |  | 183808 | 1693974528 | 752877568 / 752877568 | REFERENCE_BASELINE |
| FDP Prior | 5.616322681206661 | 7.589023149806249 |  | 0.0 | [0.0, 0.0] | 183808 | 1693974528 | 752877568 / 752877568 | EXACT |
| FDP Native | 5.601589544767384 | 7.569297810607507 |  | -0.002614647509103798 | [-0.002763943499829396, -0.002463507365031415] | 183808 | 1693974528 | 752877568 / 752877568 | EXACT |

Interpretation: Fisher and Prior are equal in the recorded 0.7 formal outputs (`ΔJ=0`); the equivalence label is only promoted when the action audit proves no accepted transfer and identical rank signature. Native is the only 0.7 method currently showing a nonzero functional change.
The imported Fisher artifact in `official_formal_quality/ratio_0p7/fisher/result.json` reports Wiki `5.6163887622`, C4 `7.5890376247`, J `1.8761968383`; the matched Prior/Native control evaluator reports Wiki `5.6163226812`, C4 `7.5890231498`, J `1.8761900017`. This small aggregation/provenance discrepancy is preserved explicitly; the matched-control values are used for the official FDP delta, and no source file is overwritten.

## Official reproduction versus paper

| Retained ratio | Paper Uniform Wiki | Our Uniform Wiki | Δ Uniform | Paper Fisher Wiki | Our Fisher Wiki | Δ Fisher | Our Fisher C4 | Fisher rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 6.34 | 6.337619345019199 | -0.002380654980800756 | 5.62 | 5.616322681206661 | -0.0036773187933389906 | 7.589023149806249 | 183808 |
| 0.5 | 7.36 | 7.356309684959772 | -0.0036903150402283202 | 6.02 | 6.0204564227043535 | 0.0004564227043539759 | 8.718134429294908 | 131072 |
| 0.3 | 10.77 | 10.770384261708738 | 0.0003842617087386202 | 8.59 | 8.620355030908927 | 0.030355030908927105 | 15.014420078529556 | 78848 |

## Branch classification

### READY FOR PAPER

- Official Dense identity and Uniform reproduction as protocol validation.
- Official Automatic (Fisher) 0.7/0.5/0.3 as the faithful PaLu baseline.
- Official 0.7 FDP Native PPL: Wiki `5.6015895448`, C4 `7.5692978106`, J `1.8735753542`, ΔJ `-0.0026146475`, bootstrap 95% CI `[-0.0027639435,-0.0024635074]`, exact resource match.
- Official 0.7 Fisher/Prior zero-shot are comparable downstream rows; Prior is neutral/equivalent in the current evidence.

### VALID BUT SUPPLEMENTAL

- Historical DiskE LongBench raw task cells for ratio 0.5 Uniform and FDP Native (qasper, samsum, trec, triviaqa). They are task-level supplemental evidence only; aggregate scorer closure is pending.
- Historical system/profile and signal-audit artifacts are useful for mechanism/provenance context, not headline quality.

### HISTORICAL / DIAGNOSTIC

- All `PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1` quality, zero-shot, and random rows use the old 2-slot/module anchor schema and are quarantined as `HISTORICAL_ANCHOR_SCHEMA` / `DIAGNOSTIC_ONLY`.
- `PALU_PPL_GUIDED_MEDIUM_ANOMALOUS_NEGATIVE_FROZEN` remains diagnostic only: primary cause native-semantics mismatch; secondary allocation saturation.

### RUNNING

- Official ratio 0.7 FDP Native zero-shot was active at audit time; its final result file was not present. The audit records it as RUNNING and does not infer a score from logs.

### PENDING

- Official phi/profile, Prior/Native calibration, formal PPL, random controls, and zero-shot for ratios 0.5 and 0.3.
- Official 0.7 Native zero-shot final result and any downstream aggregate closure.
- LongBench aggregate/scorer closure for the historical supplemental cells.

### BLOCKED

- No current hard external-data blocker is required for the already bound six-task local zero-shot; remaining 0.5/0.3 work is dependency-gated on official phi, not a dataset claim.

### REDUNDANT / SHOULD SKIP

- The 0.7 action audit confirms `PRIOR_EQUIVALENT_TO_FISHER`: accepted transfer count `0` and exact realized selection equal to Fisher. Do not rerun Prior PPL/zero-shot/LongBench; cite Fisher once and retain Prior provenance as identity verification.
- Do not port old 32,640-rank anchor-schema zero-shot or phi into the official 8-group action space.

## Audit artifacts

- `PALU_GLOBAL_EVIDENCE_AUDIT.csv`: row-level evidence ledger; blanks mean unavailable, not zero.
- `PALU_GLOBAL_PROVENANCE.json`: input hashes, state snapshots, prior-equivalence audit, and live GPU/process snapshot.
- `PALU_GLOBAL_REQUIREMENT_MATRIX.md`: requirement-by-requirement closure status.

