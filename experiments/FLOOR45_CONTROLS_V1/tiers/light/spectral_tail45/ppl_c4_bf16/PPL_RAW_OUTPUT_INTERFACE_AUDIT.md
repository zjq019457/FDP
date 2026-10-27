# PPL Raw Output Interface Audit

- generated_at: `2026-10-12T15:35:46.145563-07:00`
- runner_hash: `ffe4e52eda330ee54a408042bc1f9ca50e9c8efb1e6756579c2bbf96279d7852`
- dataset: `c4`
- ppl: `9.811620558034264`
- used_windows: `256`
- token_count_sum: `524288`
- total_nll_sum: `1197247.0135498047`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

