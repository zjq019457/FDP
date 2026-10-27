# Protocol Consistency Audit

- generated_at: `2026-07-29T19:56:45.562434-07:00`

## R10 vs R1 PPL
- R1 Dense Wiki PPL (instrumented): `6.138120287244493`
- R10 Dense Wiki PPL (sliding HF loss): `6.0465`
- Mismatch: `True` — R10 uses sliding-window HF loss; R1 uses instrumented non-overlapping chunks

## R11 Robustness
- Confirmatory λ=1.2 FDP PPL: `6.649582661740018`
- Best robustness (λ=2.0, cal=32): `6.475`
- Must NOT upgrade robustness best to main: `True`

## C4 Window Consistency
- All methods same 256 windows: `True`
{
  "dense": {
    "window_count": 256,
    "unique_input_hashes": 231,
    "first_three_hashes": [
      "032c3eb97b721d82a993d89d2220af34a24398de054d86f6e7b7a0150d29d0af",
      "0429eaa3e9b6b044f6755fac509d8ac8471404f556d43ac9d4b3600a032a8669",
      "05384c7d16fdfa1b8dde9bc611ce34453e952f8d8b48133dcdf6d09409467061"
    ]
  },
  "uniform_eb045": {
    "window_count": 256,
    "unique_input_hashes": 231,
    "first_three_hashes": [
      "032c3eb97b721d82a993d89d2220af34a24398de054d86f6e7b7a0150d29d0af",
      "0429eaa3e9b6b044f6755fac509d8ac8471404f556d43ac9d4b3600a032a8669",
      "05384c7d16fdfa1b8dde9bc611ce34453e952f8d8b48133dcdf6d09409467061"
    ]
  },
  "FGeb045": {
    "window_count": 256,
    "unique_input_hashes": 231,
    "first_three_hashes": [
      "032c3eb97b721d82a993d89d2220af34a24398de054d86f6e7b7a0150d29d0af",
      "0429eaa3e9b6b044f6755fac509d8ac8471404f556d43ac9d4b3600a032a8669",
      "05384c7d16fdfa1b8dde9bc611ce34453e952f8d8b48133dcdf6d09409467061"
    ]
  }
}

## R8 System Metrics Completeness
{
  "path": "/home/bai-project/jiaqizhang/EigenAttn/rebuttal_legacy_reproduction/R8_SYSTEM_METRICS.md",
  "fields_present": {
    "median": false,
    "p95": false,
    "peak allocated": false,
    "peak reserved": false,
    "TTFT": false,
    "prefill": true,
    "decode/token": false,
    "tokens/s": false,
    "offline cost": false
  },
  "has_median_ms": true,
  "has_p95": false,
  "has_peak_allocated": true,
  "has_offline_cost": false,
  "complete": false
}

## R4 Matched Controls
{
  "required_methods": [
    "Uniform",
    "FDP",
    "Random_seed0",
    "Random_seed1",
    "Random_seed2",
    "Spectral",
    "Middle",
    "Depth_front",
    "Depth_back",
    "Reconstruction_slope"
  ],
  "found_in_r4": [
    "Uniform",
    "FDP"
  ],
  "complete": false,
  "note": "R4 only has Dense/Uniform/FDP point comparison; matched controls missing"
}
