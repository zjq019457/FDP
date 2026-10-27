# PPL Raw Output Interface Audit

- generated_at: `2026-10-10T20:13:01.374127-07:00`
- runner_hash: `ffe4e52eda330ee54a408042bc1f9ca50e9c8efb1e6756579c2bbf96279d7852`
- dataset: `c4`
- ppl: `21.69028020724712`
- used_windows: `256`
- token_count_sum: `524288`
- total_nll_sum: `1613163.0004882812`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

