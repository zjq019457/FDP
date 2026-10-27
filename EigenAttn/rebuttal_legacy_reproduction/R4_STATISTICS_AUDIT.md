# R4 Cross-Method Statistical Comparison

- generated_at: `2026-07-29T01:31:56.746963-07:00`

| Metric | Dense | Uniform | FGeb045 | Uni Δ | FDP Δ | FDP-Uni |
|--------|-------|---------|---------|-------|-------|---------|
| arc_challenge_localjsonl | 0.5009 | 0.4693 | 0.4863 | -0.0316 | -0.0145 | 0.0171 |
| arc_easy_localjsonl | 0.7757 | 0.7668 | 0.7685 | -0.0088 | -0.0072 | 0.0017 |
| c4_ppl | 8.5971 | 9.8356 | 9.6937 | 1.2385 | 1.0966 | -0.1419 |
| hellaswag_localjsonl | 0.5846 | 0.5614 | 0.5628 | -0.0232 | -0.0218 | 0.0014 |
| piqa_localjsonl | 0.7949 | 0.7851 | 0.7862 | -0.0098 | -0.0087 | 0.0011 |
| wiki_ppl | 6.1381 | 6.7295 | 6.6496 | 0.5914 | 0.5115 | -0.0799 |
| winogrande_localjsonl | 0.7324 | 0.7222 | 0.7230 | -0.0103 | -0.0095 | 0.0008 |
| kv_retained_ratio | 1.0000 | 0.8092 | — | — | — | — |

## Historical lm_eval cross-check

### dense_llama3
- hellaswag_localjsonl: 0.5849
- arc_challenge_localjsonl: 0.5017
- arc_easy_localjsonl: 0.7757
- winogrande_localjsonl: 0.7324
- piqa_localjsonl: 0.7933

### kv_eb043
- hellaswag_localjsonl: 0.5639
- arc_challenge_localjsonl: 0.4684
- arc_easy_localjsonl: 0.7719
- winogrande_localjsonl: 0.7301
- piqa_localjsonl: 0.7873

### kv_eb045
- hellaswag_localjsonl: 0.5609
- arc_challenge_localjsonl: 0.4659
- arc_easy_localjsonl: 0.7668
- winogrande_localjsonl: 0.7214
- piqa_localjsonl: 0.7851

### kv_frag_eb045
- hellaswag_localjsonl: 0.5625
- arc_challenge_localjsonl: 0.4846
- arc_easy_localjsonl: 0.7673
- winogrande_localjsonl: 0.7238
- piqa_localjsonl: 0.7867

