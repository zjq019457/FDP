# FDP125 QA-6 protocol audit V1

## Decision

- Status: **PASS**
- Classification: **MANIFEST_METADATA_COUNT_BUG_ONLY**
- Old manifest retained unchanged.
- Corrected candidate manifest declares **1150** samples.
- No predictions, sample IDs, task order, prompt, scorer, runtime, q, lambda, rho, allocation or formal 17-task result was modified.

## Count reconciliation

| task | frozen count |
|---|---:|
| narrativeqa | 200 |
| qasper | 200 |
| multifieldqa_en | 150 |
| hotpotqa | 200 |
| 2wikimqa | 200 |
| musique | 200 |

Sum: **1150**. The previous top-level `total_samples=1200` was inconsistent with the explicit task metadata. `MultiFieldQA-en=150` is the existing frozen formal count and is retained.

## Provenance checks

- NarrativeQA source rows: 200; unique `_id`: 200.
- Official NarrativeQA prediction rows: 200; source/prediction `_id` order identical: **True**.
- Existing five-task QA-6 copies are byte-identical to the frozen 17-task prediction artifacts for both Official and FDP125 Direct.
- Previous FDP125 NarrativeQA attempt observed progress through approximately 26/200, but no complete prediction file exists; it remains invalid for scoring and is preserved through its log.

## Hashes

- Old manifest SHA256: `271f4611374fc8756a5eea2d0c0c18b7ee9a236148f3d7223f91a52c6aefd132`
- Corrected manifest SHA256: `2d5b73583e52adea75e70521adb8811677e5c03114001abe21df7ccd13e2407e`
- Corrected payload SHA256: `09aa13759d8e8df6384be103be265e689206a28ed21467969949f926dd9fb847`

## Gate

The corrected manifest is the only candidate for the supplemental QA-6 scorer. Rebuild FDP125 NarrativeQA from 0/200 after a safe GPU becomes available; do not resume the incomplete partial attempt. Then score QA-6 as a supplemental/predefined QA subset. The main 17-task LongBench result remains frozen.
