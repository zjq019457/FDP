# PPL Raw Output Interface Audit

- generated_at: `2026-07-28T08:38:14.392714-07:00`
- runner_hash: `db6638a202c09158150aec90019d0ed4fc03fa29728808ff082817ff1db0fd84`
- dataset: `c4`
- ppl: `8.59711311681018`
- used_windows: `256`
- token_count_sum: `524288`
- total_nll_sum: `1127967.0773353577`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

