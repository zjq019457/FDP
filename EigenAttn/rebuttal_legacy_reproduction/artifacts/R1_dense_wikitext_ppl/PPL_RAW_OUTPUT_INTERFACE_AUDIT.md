# PPL Raw Output Interface Audit

- generated_at: `2026-07-28T08:38:40.042718-07:00`
- runner_hash: `db6638a202c09158150aec90019d0ed4fc03fa29728808ff082817ff1db0fd84`
- dataset: `wikitext2`
- ppl: `6.138120287244493`
- used_windows: `141`
- token_count_sum: `288768`
- total_nll_sum: `523974.8935546875`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

