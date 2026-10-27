# PPL Raw Output Interface Audit

- generated_at: `2026-09-10T14:17:52.084969-07:00`
- runner_hash: `ffe4e52eda330ee54a408042bc1f9ca50e9c8efb1e6756579c2bbf96279d7852`
- dataset: `wikitext2`
- ppl: `10.318369519877693`
- used_windows: `141`
- token_count_sum: `288768`
- total_nll_sum: `673963.0725097656`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

