# PPL Raw Output Interface Audit

- generated_at: `2026-10-11T11:47:37.071698-07:00`
- runner_hash: `ffe4e52eda330ee54a408042bc1f9ca50e9c8efb1e6756579c2bbf96279d7852`
- dataset: `wikitext2`
- ppl: `7.427668006772471`
- used_windows: `141`
- token_count_sum: `288768`
- total_nll_sum: `579041.0437011719`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

