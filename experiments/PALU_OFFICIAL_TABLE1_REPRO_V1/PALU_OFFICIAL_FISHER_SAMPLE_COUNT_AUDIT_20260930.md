# PaLu official Fisher sample-count audit — 2026-09-30

## Finding

The interrupted `TABLE1_AUTOMATIC_LRD_GROUP_SWEEP` cell was not shown to have used a 32-example Fisher profile. Although its wrapper supplied `n_fisher_calib_samples=32`, the pinned official `fisher_uniform` branch in `palu/rank_search.py` constructs its loader with a literal `2048` and the requested calibration sequence length. It then calls `calib_fisher_info(..., args.use_cache)`. The cache-hit path loads the existing per-model Fisher tensor and does not use the wrapper's loader.

The existing cache was created by the preceding official `fisher_uniform` run: its log records `No cache_file`, `Create fisher info list`, and `Save the fisher info list` on 2026-09-29, using the same pinned official `fisher_uniform` branch. The subsequent gs=4 run and the interrupted group sweep both record a cache hit. This traces the effective profile to the official 2048-sample loader; the wrapper field `32` was misleading metadata, not the effective Fisher sample count.

## Interruption and recovery

The GPU0 worker was stopped during the first J-LRD/gs=32 cell while decomposing factors. It had not emitted a PPL metric or a completed result. Its log, state, active marker, and partial allocation were preserved under:

`/diskE/jiaqizhang/PALU_OFFICIAL_TABLE1_REPRO_V1/audit/attempt_history/FISHER32_PROTOCOL_MISMATCH_20260930/`

The earlier `INVALIDATED_PROTOCOL_MISMATCH` label was an overstatement and is superseded by this audit. No completed Dense, Uniform, or Appendix F.3 result was invalidated. The incomplete J-LRD/gs=32 quality cell is being resumed in a separately named run with the wrapper field explicitly set to 2048. No foreign process was interrupted (`foreign_jobs_interrupted=0`).

## Reproducibility implication

For every new result, retain both the effective loader source/hash and the Fisher-cache SHA. Do not infer the effective sample count from the wrapper field alone: in this official branch the literal `2048` in `rank_search.py` controls it. The paper-specific Table 1/Table 6 invocation is still not documented exactly in the repository, so this is an official-code reconstruction guided by the paper's stated Fisher profile, not proof of the authors' original launch command.
