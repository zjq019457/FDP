# PPL Raw Output Interface Audit

- generated_at: `2026-10-11T11:48:37.205087-07:00`
- runner_hash: `ffe4e52eda330ee54a408042bc1f9ca50e9c8efb1e6756579c2bbf96279d7852`
- dataset: `c4`
- ppl: `15.20630496912665`
- used_windows: `256`
- token_count_sum: `524288`
- total_nll_sum: `1426959.967163086`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

