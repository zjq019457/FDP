# MPT native realization resource geometry

CPU/read-only audit of already measured legal breakpoints. No epsilon was retuned, no rank was edited, and no WikiText-2, C4, or zero-shot result was read.

## Resource criterion

MPT campaign locks and the protocol adapter do not contain a preregistered numeric resource-match tolerance. The figure ±0.15% is `NOT_PRE_REGISTERED_FOR_MPT`. It is only the Llama enumeration `STRICT_MATCH` label, kept here as a diagnostic reference. It is not an MPT pass rule.

`RESOURCE_MATCH_PASS` is therefore false wherever realized bytes differ from the Uniform census. They differ at every tested operating point.

## Realized native bytes

Census equals the rank prediction at every tier. Operator parity relative L2 is 0. Duplicate-process canary matches exact logits SHA and realization SHA.

| tier | Uniform B_live | FDP B_live | absolute gap | relative gap |
| --- | ---: | ---: | ---: | ---: |
| U011 | 851,705,856 | 889,323,520 | +37,617,664 | +4.4167% |
| U015 | 717,094,912 | 776,077,312 | +58,982,400 | +8.2252% |
| U019 | 608,043,008 | 660,996,096 | +52,953,088 | +8.7088% |

## What the staircase does

Legal ranks are multiples of 32 on the threshold grid 1.00 → 0.30, step 0.02. A layer stops at its first failure for the mapped epsilon, or at the 0.30 floor.

Measured adjacent rank-sum jumps run from 64 to 2272. The smallest step is 262,144 bytes: 0.031% of the U011 Uniform total, 0.037% of U015, and 0.043% of U019. The observed gaps are many steps wide.

The product of measured legal breakpoints contains each Uniform schedule, so the nearest reachable byte total has relative gap 0. The Uniform operating point is representable on the native lattice without any quality signal.

The frozen FDP map is also a deterministic quality-blind projection: `q` from `dZ+dS+dH`, `epsilon = epsilon_base / (1 + 1.2 q)`, mean normalization, layer 0 clamped to `epsilon_base`, then the first-failure legal rank. Its image is above Uniform at all three operating points.

The sign is not uniform across depth. Early layers realize fewer ranks than Uniform. Later layers realize more, and that surplus dominates:

| tier | ranks above Uniform | ranks below Uniform | net ranks | net bytes |
| --- | ---: | ---: | ---: | ---: |
| U011 | +13,312 | -4,128 | +9,184 | +37,617,664 |
| U015 | +19,200 | -4,800 | +14,400 | +58,982,400 |
| U019 | +17,664 | -4,736 | +12,928 | +52,953,088 |

Layer 0 matches Uniform at every tier, because its rank-search epsilon is `epsilon_base`.

## Projection

A nearest-byte choice among measured legal breakpoints would match Uniform exactly, because the Uniform schedule is one of those points. That choice uses the byte target. It is not the frozen epsilon map, and it was not executed.

The frozen map itself does not land on the Uniform byte total at `epsilon_base` 0.011, 0.015, or 0.019. Moving it onto that total inside the same formula would change the global epsilon scalar and therefore the operating point. That was not done.

`MPT_NATIVE_REALIZATION_RESOURCE_MISMATCH = true`

Formal quality was not started. U011 remains a valid MPT realization whose native resource differs from Uniform by 4.42%; the matched-resource quality comparison was not run.
