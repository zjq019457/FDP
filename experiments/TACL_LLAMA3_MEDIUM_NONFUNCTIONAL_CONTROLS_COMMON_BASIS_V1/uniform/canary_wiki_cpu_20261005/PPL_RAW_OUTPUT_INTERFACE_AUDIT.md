# PPL Raw Output Interface Audit

- generated_at: `2026-10-05T14:02:30.929153-07:00`
- runner_hash: `ffe4e52eda330ee54a408042bc1f9ca50e9c8efb1e6756579c2bbf96279d7852`
- dataset: `wikitext2`
- ppl: `6.103181025587486`
- used_windows: `1`
- token_count_sum: `2048`
- total_nll_sum: `3704.443115234375`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

