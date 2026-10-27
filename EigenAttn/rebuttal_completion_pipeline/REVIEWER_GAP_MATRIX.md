# Reviewer Gap Matrix

- generated_at: `2026-07-29T19:56:45.554223-07:00`

## Existing Evidence (reuse, do not rerun)

| Area | Status | Path |
|------|--------|------|
| DENSE_PARITY_PASS | PASS | `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/DENSE_PARITY_PASS.DONE` |
| UNIFORM_PARITY_PASS | PASS | `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/UNIFORM_PARITY_PASS.DONE` |
| LEGACY_FDP_PARITY_PASS | PASS | `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/LEGACY_FDP_PARITY_PASS.DONE` |
| R4_STATISTICS_PASS | PASS | `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/R4_STATISTICS_PASS.DONE` |
| R5_HISTORICAL_PARITY_PASS | PASS | `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/R5_HISTORICAL_PARITY_PASS.DONE` |
| R6_STATISTICS_PASS | PASS | `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/R6_STATISTICS_PASS.DONE` |
| R8_SYSTEM_METRICS_PASS | PASS | `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/R8_SYSTEM_METRICS_PASS.DONE` |
| R10_KQV_ABLATION_PASS | PASS | `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/R10_KQV_ABLATION_PASS.DONE` |
| R11_ROBUSTNESS_PASS | PASS | `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/R11_ROBUSTNESS_PASS.DONE` |
| SUPPLEMENTARY_EXPERIMENTS_COMPLETE | PASS | `/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/SUPPLEMENTARY_EXPERIMENTS_COMPLETE.DONE` |

## Confirmatory Results (R1-R3 instrumented)

### dense
- Wiki PPL: 6.138120287244493 (141 windows)
- C4 PPL: 8.59711311681018 (256 windows)
- Macro acc: 0.6776991426769664 (5 tasks: True)

### uniform_eb045
- Wiki PPL: 6.7294786479615984 (141 windows)
- C4 PPL: 9.83562032647886 (256 windows)
- Macro acc: 0.6609662206144898 (5 tasks: True)

### FGeb045
- Wiki PPL: 6.649582661740018 (141 windows)
- C4 PPL: 9.693687291910727 (256 windows)
- Macro acc: 0.6653702002059017 (5 tasks: True)

## Gaps Requiring New Work

| ID | Severity | Action |
|----|----------|--------|
| GAP_R4_MATCHED_CONTROLS | critical | Build matched-budget controls per operating point |
| GAP_BOOTSTRAP | critical | Compute from reused ppl_windows.jsonl and zero_shot samples |
| GAP_R10_KQV | high | Re-run with instrumented protocol matching R1 |
| GAP_R10_PPL_PROTOCOL | high | Discard R10 PPL for main claims; re-run with ppl_instrumented.py |
| GAP_R8_FIELDS | medium | Extend system metrics runner or document eager-only scope |
| GAP_OPERATING_POINTS | critical | Rebuild 3 operating points via legacy builder |
| GAP_COMPONENT_ABLATION | high | Run component and calibration domain ablations |
| GAP_QWEN | medium | Download from official HF |
| GAP_OPT30B | medium | Run fixed-schedule aggressive point if raw missing |
| GAP_EXTERNAL_BASELINE | medium | Attempt official implementations or document blocked |
