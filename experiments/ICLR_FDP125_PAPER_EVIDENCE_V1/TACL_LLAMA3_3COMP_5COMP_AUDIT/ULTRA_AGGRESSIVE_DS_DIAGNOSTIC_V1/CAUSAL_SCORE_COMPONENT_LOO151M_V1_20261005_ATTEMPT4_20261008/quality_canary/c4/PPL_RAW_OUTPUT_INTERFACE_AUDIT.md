# PPL Raw Output Interface Audit

- generated_at: `2026-10-08T20:30:27.539515-07:00`
- runner_hash: `ffe4e52eda330ee54a408042bc1f9ca50e9c8efb1e6756579c2bbf96279d7852`
- dataset: `c4`
- ppl: `16.315925835689146`
- used_windows: `2`
- token_count_sum: `4096`
- total_nll_sum: `11436.6123046875`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

