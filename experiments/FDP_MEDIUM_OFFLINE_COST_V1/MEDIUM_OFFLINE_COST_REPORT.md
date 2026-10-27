# MEDIUM_OFFLINE_COST_REPORT — Llama-3-8B Medium, FDP-EB vs Uniform EigenAttention

One real cold end-to-end measurement (2026-10-10). All numbers below are **measured** unless marked otherwise. All rebuilt artifacts are bit-identical to the frozen Medium results (see `MEDIUM_PROTOCOL_AUDIT.md`).

## Stage timings (measured)

| Stage | Wall time | Peak GPU (NVML, process) | Torch peak alloc / reserved | Peak CPU RSS |
|---|---|---|---|---|
| A+B Parent Uniform: load/tokenise (31 s) + calibration capture, basis, rank search (eb 0.060) | 44.08 min | 41.4 GiB | 39.0 / 40.9 GiB | 14.5 GiB |
| C Parent Uniform materialize/save | 2.97 min | 16.3 GiB | 15.7 / 15.8 GiB | 31.2 GiB |
| D FDP Step 1 profiling (Dense + compressed forwards, diagnostics) | 1.38 min | 16.2 GiB | 15.6 / 15.7 GiB | 8.9 GiB |
| E Score aggregation, MinMax, EB mapping (CPU) | 0.01 s | — | — | — |
| F FDP Pass-2 basis + adaptive rank search | 42.84 min | 41.3 GiB | 39.0 / 40.9 GiB | 14.5 GiB |
| G FDP checkpoint materialize/save | 2.85 min | 16.3 GiB | 15.7 / 15.8 GiB | 31.1 GiB |
| Uniform comparator: load/tokenise (31 s) + calibration, basis, rank search (eb 0.0567) | 42.34 min | 41.4 GiB | 39.0 / 40.9 GiB | 14.5 GiB |
| Uniform comparator materialize/save | 2.38 min | 16.3 GiB | 15.7 / 15.8 GiB | 31.2 GiB |

Inside each builder stage, model load + tokenisation takes ≈31–40 s and the first layer's search output appears at ≈49–60 s; the remainder (≈41–42 min) is per-layer decomposition plus rank search, which the frozen builder does not separate.

## Paper table (TACL)

| Metric | Uniform | FDP-EB |
|---|---|---|
| Total offline construction time | 44.7 min | 94.1 min |
| Additional functional profiling time | — | 4.3 min (parent save 3.0 + Step 1 1.4 + mapping <0.1) |
| Additional rank allocation/search time | — | 45.7 min (Pass-2 build/search 42.8 + save 2.8) |
| Peak GPU memory (NVML, process) | 41.4 GiB | 41.4 GiB |
| Peak CPU memory (RSS) | 31.2 GiB | 31.2 GiB |
| Offline overhead ratio | 1.00× | 2.10× |

- Absolute extra time: **+49.4 min** (94.1 − 44.7).
- Profiling share of FDP total: **4.6%** (C+D+E); Step 1 alone is **1.5%**.
- FDP adds no peak-memory cost: both peaks come from the shared builder (41.4 GiB GPU) and from materialization (31 GiB CPU).
- The extra time is dominated by the second full builder pass (F). In the frozen implementation that pass repeats the per-layer decomposition, which does not depend on the EB. A pipeline that reused the parent's decomposition would remove most of F, but this has **not been measured** and must not be reported as a measured number.

## Measurement status

| Number | Status |
|---|---|
| All stage times, GPU/CPU peaks, disk | MEASURED (one cold run) |
| Totals, ratio, shares | DERIVED from measured stages |
| Cost with reused decomposition | NOT MEASURED (estimate only; not reported as a result) |
| Historical stage timestamps (parent 61 min, FDP 50 min) | historical context; current run was faster (44 / 43 min) |

Single run only (no repetitions), on a shared server (see audit). Variance is not estimated.
