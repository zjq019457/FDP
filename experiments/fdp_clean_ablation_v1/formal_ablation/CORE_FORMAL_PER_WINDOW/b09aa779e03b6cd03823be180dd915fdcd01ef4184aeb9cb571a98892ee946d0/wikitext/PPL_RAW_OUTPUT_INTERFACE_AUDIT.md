# PPL Raw Output Interface Audit

- generated_at: `2026-09-10T14:47:58.154639-07:00`
- runner_hash: `ffe4e52eda330ee54a408042bc1f9ca50e9c8efb1e6756579c2bbf96279d7852`
- dataset: `wikitext2`
- ppl: `161.30104284464625`
- used_windows: `141`
- token_count_sum: `288768`
- total_nll_sum: `1467886.4189453125`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

