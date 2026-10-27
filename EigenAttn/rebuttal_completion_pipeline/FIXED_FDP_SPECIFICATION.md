# Fixed FDP Schedule Specification

- generated_at: `2026-07-29T19:56:45.597242-07:00`
- protocol_sha256: `bf8bc0030f95918c425b8b3721331a27d1bfd1711b48692089c000a106b7fd9e`

## Confirmatory Config (FGeb045)

- eb_base: `0.045`
- lambda_eb: `1.2`
- normalize: `mean`
- weights: `{"wz": 1.0, "ws": 1.0, "wv": 1.0, "wa": 1.0, "wh": 1.0}`
- freeze_layers: `[0]`
- entropy_collapse: `max(0, -entropy_diff_A)`
- clipping_rails: `none`

## Fragility Formula

- Z_l_rel: Z_rel_mse from step1 per layer (query-key projection distortion)
- S_l_rel: score_rel_mse_causal from step1 (attention score distortion, causal mask)
- V_l_rel: value_only_Z_rel_mse from step1 (value projection distortion)
- A_l_rel: AV_rel or entropy-related attention value distortion from step1
- aggregation: per_head -> per_layer_mean -> normalize(mean) across layers
- s_l: `wz*Z_l_rel + ws*S_l_rel + wv*V_l_rel + wa*A_l_rel + wh*max(0,-entropy_diff_A)`
- eb_l: `eb_base / (1 + lambda_eb * s_tilde_l) where s_tilde_l is mean-normalized s_l`

## Operating Points (eb_base only varies)

- **mild**: target ratio `0.82`, eb_base `0.045`
- **moderate**: target ratio `0.739`, eb_base `0.06`
- **aggressive**: target ratio `0.589`, eb_base `0.09`

## Robustness Policy

- lambda=1.2 is confirmatory; lambda in {0.6, 2.0} and cal_size sweeps are sensitivity only.
- Best robustness result (e.g. lambda=2.0, cal=32, PPL=6.475) must NOT replace main config.

## KV Bytes

- sum_over_layers(2 * num_kv_heads * head_dim * (rank_kq + rank_v) * bytes_per_elem * seqlen)
- Controls must match FDP **actual** realized bytes, not nominal retained ratio.
