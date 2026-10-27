# FDP LAYER 0 PROTECTION INTERNALIZATION AUDIT

## 1. Executive Conclusion
- **LAYER0_PROTECTION_INTERNALIZED_BY_FORMULA**: `False`
- **Optimal rho (DEV Selected)**: `0.50`
- **Historical Anchor Necessity**: Retained - explicit Layer 0 hard/soft anchor remains empirically necessary

## 2. Empirical Rho Sweep on Promoted Fragility Formulation (PROP)
| rho | J_STRONG | Delta vs F_CURRENT | Delta J_LIGHT | Light Guard Pass |
|---|---|---|---|---|
| 0.00 | 3.354028 | +0.014393 | +0.000000 | True |
| 0.25 | 3.354028 | +0.014393 | +0.000000 | True |
| 0.50 | 3.339635 | +0.000000 | +0.000000 | True |
| 0.75 | 3.341908 | +0.002274 | +0.000000 | True |
| 1.00 | 3.341908 | +0.002274 | +0.000000 | True |

## 3. Physical Mechanism
Under standard FDP, Layer 0 exhibits high downstream causal propagation ($P_0$), because distortions introduced in early attention layers cascade through all downstream representations.
In propagation-aware fragility $H_l = (1-\eta)\tilde{S}_l + \eta\tilde{P}_l$, the high propagation $P_0$ naturally inflates the combined fragility $H_0$, driving $\epsilon_0$ lower (i.e. assigning higher rank preservation) automatically without requiring an ad-hoc geometric clamp $\rho$.
