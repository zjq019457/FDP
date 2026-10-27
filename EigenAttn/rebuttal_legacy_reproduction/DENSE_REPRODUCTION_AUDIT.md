# Dense Reproduction Audit

- generated_at: `2026-07-28T09:17:19.261184-07:00`
- wikitext2_ppl: `6.138120287244493`
- c4_ppl: `8.59711311681018`
- recompute_ok: `True`

## Zero-shot

- `hellaswag_localjsonl`: acc_from_correct=0.5846444931288588 lm_eval=0.5846444931288588 n=10042
- `arc_challenge_localjsonl`: acc_from_correct=0.5008532423208191 lm_eval=0.5008532423208191 n=1172
- `arc_easy_localjsonl`: acc_from_correct=0.7756734006734006 lm_eval=0.7756734006734006 n=2376
- `piqa_localjsonl`: acc_from_correct=0.794885745375408 lm_eval=0.794885745375408 n=1838
- `winogrande_localjsonl`: acc_from_correct=0.7324388318863457 lm_eval=0.7324388318863457 n=1267

## Historical comparison

- `{'task': 'hellaswag_localjsonl', 'current': 0.5846444931288588, 'historical': 0.5849432383987253, 'abs_diff': 0.0002987452698665072}`
- `{'task': 'arc_challenge_localjsonl', 'current': 0.5008532423208191, 'historical': 0.5017064846416383, 'abs_diff': 0.0008532423208191808}`
- `{'task': 'arc_easy_localjsonl', 'current': 0.7756734006734006, 'historical': 0.7756734006734006, 'abs_diff': 0.0}`
- `{'task': 'piqa_localjsonl', 'current': 0.794885745375408, 'historical': 0.7932535364526659, 'abs_diff': 0.0016322089227420733}`
- `{'task': 'winogrande_localjsonl', 'current': 0.7324388318863457, 'historical': 0.7324388318863457, 'abs_diff': 0.0}`
