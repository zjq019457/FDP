# PaLu V3 — Final Evidence Bundle

Status: **PALU_V3_EVIDENCE_CLOSED_WITH_RANDOM_MATCHED_CONTROL**

## Frozen TUNE selection

- Grid complete: 72/72 configs across 4 arms.
- Selected by the frozen rule: Arm A / `armA_D16_R16_Rmax0.01_beam0`.
- TUNE: J=1.881027602; ΔJ vs V1 tune start=-0.009082094; paired 95% CI [-0.010273620, -0.007850654].
- Search trajectory: 26 accepted single exchanges, 5311 rejected proposals; frozen rank quantum 32.

## Independent Calibration-Lock

- Status: `V3_CALIBRATION_LOCK_FROZEN`; ΔJ vs V1=-0.007849477; paired 95% CI [-0.009224772, -0.006671324].
- Allocation SHA-256: `70e854eb862602f7cfb22faafba2acd1654b0356322486dfbc772b1de42cdf9c`.

## Formal V1 Native vs V3

- Formal manifest SHA-256: `858a57cad57042b1d1d5af3c1fabd62b9935794a3e0bca83460c625feab99010`.
- V1 Native: J=1.975172502; Wiki PPL=6.003692; C4 PPL=8.653557; N=166 Wiki / 256 C4 chunks.
- V3: J=1.965511483; Wiki PPL=5.951856; C4 PPL=8.561880; N=166 Wiki / 256 C4 chunks.
- V3−V1: ΔJ=-0.009661020; paired 95% CI [-0.010247311, -0.009120551].
- Fisher J=1.980298479; V3−Fisher ΔJ=-0.014786996; paired 95% CI [-0.016211521, -0.013513852].
- All three seed-specific V1 and V3 replays exactly matched their frozen formal chunk losses.

## Random-Matched formal results

| Seed | Wiki PPL | C4 PPL | J | Random−V1 ΔJ [95% CI] | V3−Random ΔJ [95% CI] |
|---:|---:|---:|---:|---:|---:|
| 0 | 6.000267 | 8.656231 | 1.975041677 | -0.000130826 [-0.000358938, 0.000099331] | -0.009530194 [-0.010205269, -0.008924634] |
| 1 | 6.008462 | 8.657585 | 1.975802295 | 0.000629793 [0.000369221, 0.000979906] | -0.010290812 [-0.011026463, -0.009643762] |
| 2 | 6.009558 | 8.675818 | 1.976945383 | 0.001772881 [0.001521846, 0.002050400] | -0.011433900 [-0.012175054, -0.010751529] |

- Random J mean=1.975929785; sample SD (ddof=1)=0.000958235; mean−V1 J=0.000757283.
- Wiki PPL mean=6.006095 (SD 0.005077); C4 PPL mean=8.663211 (SD 0.010939).
- Resource gate passed for all three seeds before quality evaluation; seeds were fixed at 0/1/2.

## Exact resource and movement

- All methods/seeds: total native rank 131072; FP16 factor bytes 1207959552; prefill cache 536870912 bytes; decode cache 536870912 bytes.
- V3 and each random control: 26 single exchanges × 32 rank; transferred rank 832; allocation L1 movement 1664; donor-side factor-byte movement 7667712 bytes.
- Spatial-support caveat: V3 changed 33 native units; each random seed changed 52. Movement and total resource match exactly, but the number/location pattern of changed units does not.

## Claim guard

- `PALU_V3_FORMAL_GO=true`.
- `PALU_V3_FUNCTIONAL_SIGNAL_SUPPORTED=true` under the frozen rule: all three V3−Random paired 95% CI upper bounds are below zero.
- Interpretation: Functional valuation contributes beyond generic budget-neutral non-uniform reallocation, conditional on this matched-control design.
- Scope: three random allocations only; preserve the spatial-support caveat above. Do not reopen V3 tuning.

## Provenance SHA-256

- V3 manifest: `0041bdf6f296c7ce1783e2fb45bc68089c68e8f6652a0924feb647c7dc941c93`
- V3 method card: `dacb622d47629fc4c55215e485f50c30b9064aa3c30df2b469c950d5c4a3b07f`
- V3 lock: `d3d17175e8df6a8b60a8874a388b7d703f72bbf68f3bd9cb7456896f308a6764`
- V3 formal result: `78d22423f47c34dad8cf46c3dcc62dc0eb63eb40dbbdaf29661c54ac115bcebf`
- Formal protocol manifest: `858a57cad57042b1d1d5af3c1fabd62b9935794a3e0bca83460c625feab99010`
- Random generation protocol: `00c492df523ea61832da1b56a74990282f5c78584d112692a11c590284b978d6`
- Random evaluator: `3e02294eef7ebcf069f926b94bec9cfe192b71a421b1171de8767fa3633033d0`
- This evidence JSON SHA-256: `4719179daa21bcf088baf2ad5beb3ed246b5a395129b667177d238edcbf70a33`
- Per-seed action-manifest, action-sequence, allocation and formal-result hashes are recorded in the adjacent evidence JSON.

foreign_jobs_interrupted=0
