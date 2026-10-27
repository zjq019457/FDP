# PPL Raw Output Interface Audit

- generated_at: `2026-09-12T23:40:04.503800-07:00`
- runner_hash: `ffe4e52eda330ee54a408042bc1f9ca50e9c8efb1e6756579c2bbf96279d7852`
- dataset: `wikitext2`
- ppl: `6.850251410497807`
- used_windows: `146`
- token_count_sum: `299008`
- total_nll_sum: `575376.7150878906`

## Invariants

- Aggregate recomputable as `exp(sum(total_nll)/sum(token_count))`.
- Original EigenAttn scorer source files were not modified.
- Per-window atomic write: `.tmp` → fsync → rename → `.DONE`.

