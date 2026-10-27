# PaLu V1 vs V2 trajectory audit — ratio 0.5

Status: `READ_ONLY_TRAJECTORY_RECONSTRUCTION_COMPLETE`. Source artifacts were read-only. The isolated outputs do not modify V1 or V2.

## Resource identity

- Native rank 131072; factor bytes 1207959552; prefill/decode cache 536870912/536870912.
- Rank quantum: 32. Exact resource was verified for every saved V1/V2 proposal.
- Fisher file SHA: `a7d8bd7b6bd41443119a8c5a08f56c5433fd4480952e32bea0db672ec8bd1a38`; semantic allocation SHA: `8a7e970f63191a22bff9b61f73f2f15596bf03ca23745b87a9bb3d43d3a2ef5b` (distinct hash domains).
- V1 final schedule SHA: `b14bd9d4c838f024eb885f541b06ba2c689a7ca03e05775d8cc9de82af8766cf`.

## Why V1 stopped

The frozen V1 order contained 16 pairs. It evaluated all 16 once, accepting 10 and rejecting 6; then the list was exhausted. This is not evidence that no beneficial exchange remained after the allocation changed, because V1 did not refresh/revisit candidates.
- Accepted proposal indices: 1, 2, 3, 4, 5, 6, 7, 8, 9, 13.
- Fisher-relative movement: 20 changed units; half-L1 moved rank 320; R=0.244140625%.

## Why B1 accepted six

B1 starts from Fisher, not V1 final. It refreshes after each accepted move and uses a 64-unit candidate universe. At the six-transfer state it reconstructed and evaluated 16 legal pairs; none passed `DeltaJ<0` plus paired-bootstrap upper CI `<0`, so its pass-level stopping rule ended the run. The end was not rank-bound saturation, resource mismatch, or a depth-2 result (B1 did not use depth-2).

V1 accepted proposals 7, 8, 9, and 13 each have donor and receiver outside B1’s 64-unit universe. Their attribution is `CANDIDATE_SET_EXCLUDED`, not calibration rejection or illegality. This only establishes lack of proposal coverage; it does not claim those moves would pass B1’s gate from its own endpoint.

## Counts and movement
| V1 | 10 / 16 | 6 | 0 | 20 | 320 | 0.244140625% |

| Method | Accepted / evaluated | Rejected | Refreshes | Changed units vs Fisher | half-L1 moved rank | R vs Fisher |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 6 / 56 | 50 | 6 | 10 | 192 | 0.146484375% |
| B2 | 3 / 22 | 19 | 1 | 6 | 96 | 0.073242188% |
| B4 | 3 / 16 | 13 | 0 | 6 | 96 | 0.073242188% |

The CSV has one row per evaluated proposal and includes current/candidate allocation hashes, donor/receiver phi, the explicitly derived `donor_phi - receiver_phi` proxy, actual calibration delta/CI, accept/reject reason, rank/resource status, candidate rank/set size, and refresh state.

## Provenance and caveat

Earlier generator attempt failed before creating this directory because it looked up `fisher_allocation_sha256` in `FROZEN_INPUTS.json`; that field is actually in `INITIAL_REFRESH_PHI.json`. The corrected audit separately validates Fisher file SHA, semantic allocation SHA, V1 decision chain, and exact V2 replay. No source result was invalidated or overwritten.

V2 proposals are reproduced from frozen `allocations/FDP_NATIVE_REFRESH_B*.json` decisions and profile snapshots; the replay matched each candidate pair and final schedule SHA. No V3 calibration/formal run has started.

Foreign jobs interrupted: 0.
