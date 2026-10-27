# FDP Clean Ablation & Ultra-Compression Campaign: Master State Audit Report

**Audit Timestamp**: 2026-09-10T17:24:51.228830+00:00
**Workspace**: `/diskF/jiaqizhang/fdp_clean_ablation_v1`

## 1. Ultra Pipeline State (Section 8A)
- **Ultra Setup Completion**: **16/17 (94.1%)**
  - `U1`: **DONE_VALID** (Reusable: True)
  - `U2`: **DONE_VALID** (Reusable: True)
  - `U3`: **DONE_NEEDS_AUDIT** (Reusable: True)
  - `U4`: **DONE_NEEDS_AUDIT** (Reusable: True)
  - `U5`: **DONE_VALID** (Reusable: True)
  - `U6`: **DONE_VALID** (Reusable: True)
  - `U7`: **DONE_VALID** (Reusable: True)
  - `U8`: **DONE_VALID** (Reusable: True)
  - `U9`: **DONE_VALID** (Reusable: True)
  - `U10`: **DONE_VALID** (Reusable: True)
  - `U11`: **DONE_VALID** (Reusable: True)
  - `U12`: **DONE_VALID** (Reusable: True)
  - `U13`: **RUNNING** (Reusable: False)
  - `U14`: **DONE_VALID** (Reusable: True)
  - `U15`: **DONE_VALID** (Reusable: True)
  - `U16`: **DONE_VALID** (Reusable: True)
  - `U17`: **DONE_VALID** (Reusable: True)

## 2. Tier Base Authorities (Section 8B)
| Tier | Base EB | Total Rank | B_live (bytes) | Rank SHA (first 16) | Status |
|---|---|---|---|---|---|
| LIGHT | 0.045 | 53464 | 218988544 | `13727f5bfcafc699` | FROZEN |
| MEDIUM | 0.060 | 48432 | 198377472 | `4c2336e9a5f009e0` | FROZEN |
| AGGRESSIVE | 0.090 | 38600 | 158105600 | `af00927a4b682b8d` | FROZEN |
| ULTRA | 0.120 | 30328 | 124223488 | `6c5f695f4256c9c0` | FROZEN |

## 3. Development Data Independence (Section 8C)
- **DEV Data Status**: **VALID_AND_FROZEN**
- **DEV-WIKI**: 64 windows x 2048 tokens (`/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection/dev_wiki_tokens.pt`)
- **DEV-C4**: 64 windows x 2048 tokens (`/diskF/jiaqizhang/fdp_clean_ablation_v1/dev_selection/dev_c4_tokens.pt`)
- **Disjointness Audit**: Passed (Zero overlap with formal test / calibration sets)

## 4. Formula Search Families State (Sections 8D - 8H)
- **Family 1 (Rho Layer-0)**: Precomputed for all 4 tiers; ready for DEV PPL dispatch
- **Family 2 (Leave-One-Out)**: 5 historical components defined; derivation verified against raw tensors
- **Family 3 (Lambda)**: 5 candidates defined; CPU grid ready
- **Family 4 (Weights)**: 5 perturbation directions defined
- **Family 5 (Additions)**: Attention JS divergence and top-mass mismatch identified; queued post-rho

## 5. Nonfunctional Controls (Section 8I)
- `RANDOM_PERMUTED_PROFILE`: **RERUN_REQUIRED** (Found 0 instances)
- `SPECTRAL_TAIL`: **RERUN_REQUIRED** (Found 0 instances)
- `RECON_SLOPE`: **RERUN_REQUIRED** (Found 0 instances)
- `RECON_CURVATURE`: **RERUN_REQUIRED** (Found 0 instances)
- `MIDDLE_LAYER_PROTECTION`: **RERUN_REQUIRED** (Found 0 instances)
- `STATIC_PROFILE_TRANSFER`: **RERUN_REQUIRED** (Found 0 instances)
- `NO_LAYER0_FREEZE`: **RERUN_REQUIRED** (Found 0 instances)

## 6. Formal Contamination Audit (Section 8J)
- **Contamination Status**: **CLEAN (DEV_ONLY)**
- No development formula selection decisions have touched formal test sets.

## 7. Active Processes & Compute (Section 8K)
- **Active FDP Processes**: 1
  - PID 571016: `/home/bai-project/anaconda3/envs/eigen/bin/python -u /diskF/jiaqizhang/fdp_clean_ablation_`

