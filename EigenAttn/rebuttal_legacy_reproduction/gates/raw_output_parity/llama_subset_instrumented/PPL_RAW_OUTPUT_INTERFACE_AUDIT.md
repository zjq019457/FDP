# PPL Raw Output Interface Audit

- generated_at: `2026-07-28T08:37:15.369265-07:00`
- runner_hash: `db6638a202c09158150aec90019d0ed4fc03fa29728808ff082817ff1db0fd84`
- dataset: `c4`
- ppl: `6.202583101861317`
- used_windows: `2`
- token_count_sum: `4096`
- total_nll_sum: `7475.06005859375`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

