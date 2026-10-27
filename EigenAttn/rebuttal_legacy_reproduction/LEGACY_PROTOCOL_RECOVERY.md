# Legacy Protocol Recovery

- generated_at: `2026-07-28T08:37:06.481977-07:00`
- This is a **legacy candidate protocol** frozen from source/artifact evidence.
- Paper numbers were NOT used to reverse-select eb045/058/060.

## Fields

### `lambda_eb`
- value: `1.2`
- provenance: `recovered_from_artifact`
- source: `step1_out/llama3_8b_wikitext2_frag_eb045.json meta.lambda_eb`

### `normalize`
- value: `mean`
- provenance: `recovered_from_artifact`
- source: `step1_out/...frag_eb045.json meta.normalize`

### `weights`
- value: `{'wz': 1.0, 'ws': 1.0, 'wv': 1.0, 'wa': 1.0, 'wh': 1.0}`
- provenance: `recovered_from_artifact`
- source: `frag_eb045.json meta.weights; also source defaults in per_layer_eb_from_step1.py`

### `freeze_layers`
- value: `[0]`
- provenance: `recovered_from_artifact`
- source: `frag_eb045.json meta.freeze_layers`

### `entropy_penalty`
- value: `max(0, -entropy_diff_A)`
- provenance: `recovered_from_source`
- source: `per_layer_eb_from_step1.py / realloc_rank_from_fragility.py`

### `fragility_formula`
- value: `s_l = wz*Z_rel + ws*Score_rel + wv*Value_rel + wa*AV_rel + wh*max(0,-entropy_diff_A)`
- provenance: `recovered_from_source`
- source: `realloc_rank_from_fragility.py`

### `per_layer_eb_map`
- value: `eb_l = eb_base / (1 + lambda_eb * s_tilde_l)`
- provenance: `recovered_from_source`
- source: `per_layer_eb_from_step1.py`

### `main_operating_point`
- value: `{'tag': 'FGeb045', 'uniform': 'eb045', 'compare': 'uniform_vs_FGeb045'}`
- provenance: `recovered_from_artifact`
- source: `llama3_8b_eb045_uniform_vs_FGeb045_distortion_summary.json`

### `builder_script_defaults_eb_base`
- value: `0.024`
- provenance: `recovered_from_source`
- source: `per_layer_eb_from_step1.py argparse default (NOT confirmed as submission operating point)`
- note: Do not confuse script default with FGeb045 artifact operating point.

### `calibration_nsamples_seqlen`
- value: `{'nsamples': 'unresolved_exact_submission', 'seqlen': 2048}`
- provenance: `inferred_only`
- source: `builder defaults nsamples=512 seqlen=2048; MANIFEST mentions 128; not hard-coded as truth`

### `score_term_S_definition`
- value: `score_rel_mse_causal (Score_rel in fragility sum)`
- provenance: `recovered_from_source`
- source: `realloc_rank_from_fragility.py / step1 diagnose fields`

## Unresolved

- `exact_submission_calibration_nsamples` (inferred_only): builder default 512 vs MANIFEST 128; not frozen as submission truth
- `optimized_shared_basis_equivalence` (pending_R4): Not yet verified; extensions must use legacy builder until R4 PASS
