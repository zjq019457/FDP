# Existing PaLu evidence audit

- `VALID_FORMAL` — `/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/quality/formal_protocol/FORMAL_QUALITY_PROTOCOL_MANIFEST.json` — Formal Wiki/C4 protocol manifest; declared SHA and observed SHA both retained in provenance.
- `VALID_REPRODUCTION` — `/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/locks/PALU_LLAMA2_OFFICIAL_REPRO_LOCK.json` — Official anchor schedules and ratio mapping.
- `VALID_CALIBRATION` — `/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/audit/SOURCE_PALU_NATIVE_MARGINAL_PER_UNIT.csv` — Immutable Llama-2 native marginal source; no phi/q recomputation.
- `VALID_CALIBRATION` — `/diskE/jiaqizhang/PALU_FDP_NATIVE_EXCHANGE_FORMAL_V1/exchange` — Common native exchange calibration schedules for FDP_PRIOR/FDP_NATIVE.
- `VALID_REPRODUCTION` — `/diskF/jiaqizhang/PALU_PAPER_REPRO_AND_FDP_V2/official/Palu` — bb22666e2ef96707e8dd21d93fc00146c2e0d615
- `UNKNOWN_PROVENANCE` — `/diskF/jiaqizhang/FDP_NATIVE_TRANSFER_V2_WORK/FINAL_FDP_CORE_V2/reports/SNAPKV_MASTER_STATUS.csv` — Not imported as PaLu evidence; central table records external status separately.
- `UNKNOWN_PROVENANCE` — `/diskF/jiaqizhang/xkv_native_runtime_fixed_v1/reports/XKV_FIXED_MEDIUM_RESULTS.csv` — Not imported as PaLu evidence; central table records external status separately.
- `VALID_DIAGNOSTIC` — `PALU_PPL_GUIDED_MEDIUM_ANOMALOUS_NEGATIVE_FROZEN` — Broken projector/native-semantics mismatch; not a current method result.
- `INVALID_PROTOCOL` — `current_wrapper_fisher_uniform_outputs` — 8 slots/module and 262144 capacity mismatch versus frozen 2-slot anchor schema.

Main formal matrix: 12/12 cells complete. Historical broken projector remains diagnostic only.
