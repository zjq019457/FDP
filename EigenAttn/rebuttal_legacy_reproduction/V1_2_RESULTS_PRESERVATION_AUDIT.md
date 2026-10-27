# V1.2 Results Preservation Audit

- generated_at: `2026-07-28T07:59:59.665176-07:00`
- policy tags:
  - `simplified_exact_byte_matched_protocol`
  - `not_submission_reproduction`
  - `negative_result`

## Read-only scope

- `EigenAttn` and `EigenAttn_base` are preserved in place.
- Historical outputs, logs, basis files, JSON artifacts, checkpoints and evaluation results are not overwritten by this orchestrator.
- New work is isolated under `rebuttal_legacy_reproduction/`.

## Discovered v1.2-like paths

- No explicit `rebuttal_fixed_fdp_v1_2` / `v1.2` path found under scanned roots.

## Current action

- All newly generated manifests, locks, logs and status files live only under `rebuttal_legacy_reproduction/`.
- If legacy parity is not proven, no result will be relabeled as submission reproduction.
