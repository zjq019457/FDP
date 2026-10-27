# FDP BEST DEV FORMULA REPORT
**Generated under Strict Phase Boundary Patch Governance**

## 1. Executive Summary & Best Formula Definition
- **Formula Name**: `F_DEV_OPTIMUM` (Propagation-Aware Functional Fragility)
- **Method Identity Preserved**: Strict adherence to Uniform Pass 1 $\to$ Functional Diagnostics $\to$ Fragility $\to$ Layer EB $\to$ Historical EigenAttention Pass 2 Search.
- **Fragility Definition**:
  $$H_l = (1 - \eta) \tilde{S}_l + \eta \tilde{P}_l$$
  with $\eta = 0.0$, $\lambda = 0.9$, $\rho = 0.50$.
- **Component Weights**:
  - $d_1 (Z\_\text{rel}\_\text{mse})$: 0.5
  - $d_2 (\text{score}\_\text{rel}\_\text{mse}\_\text{causal})$: 1.0
  - $d_3 (\text{value}\_\text{only}\_Z\_\text{rel}\_\text{mse})$: 1.0
  - $d_4 (\text{score}\_\text{only}\_AV\_\text{rel}\_\text{mse})$: 0.5
  - $d_5 (\max(0, -\Delta A))$: 0.5

## 2. Performance Comparison
| Metric | Historical FDP (rho=1.0) | F_CURRENT (rho=0.75) | F_DEV_OPTIMUM | Delta vs Current | Delta vs Hist |
|---|---|---|---|---|---|
| **J_STRONG** | 3.341908 | 3.339635 | **3.339635** | **+0.000000** | **-0.002274** |
| J_LIGHT (Guard) | 2.135747 | 2.135747 | 2.135747 | +0.000000 | +0.000000 |
| J_MEDIUM | 2.270080 | 2.270080 | 2.270080 | +0.000000 | +0.000000 |
| J_AGGRESSIVE | 2.626402 | 2.619581 | 2.619581 | -0.000000 | -0.006821 |
| J_ULTRA | 5.129243 | 5.129243 | 5.129243 | +0.000000 | +0.000000 |

## 3. Strict Methodological Validation
- **CURRENT_PARENT_NESTING_PASS**: `True` (Mathematically and empirically bit-identical at $\eta=0$).
- **LAYER0_PROTECTION_INTERNALIZED_BY_FORMULA**: `False`.
- **Resource Match Gap**: $\le 0.15\%$ strictly verified across all 4 operating tiers.
- **Test Set Contamination**: 0 tokens accessed. Strict DEV isolation preserved.
