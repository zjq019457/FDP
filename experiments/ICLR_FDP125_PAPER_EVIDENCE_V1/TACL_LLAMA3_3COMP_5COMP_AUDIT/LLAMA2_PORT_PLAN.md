# Llama-2 exact 3-component port plan

## Existing evidence audit

- Dense model fingerprint already recorded: `a627fb4d2200089590b109ad82c35098778aa5856660c691f6626d25c2cf25197`; tokenizer fingerprint `74d06083…a8ad300` (must rehash/verify exact tokenizer files at launch).
- The only located L2 d1/d2/d5 profile is `13_MODEL_MATRIX_4X3/unified_protocol_closure_v1/runs/LLAMA2_MEDIUM_NATIVE_PROFILE_V1_20260924T_ATTEMPT2/FDP125_MODEL_NATIVE_D1_D2_D5_PROFILE_V1.json`, SHA256 `3846296b86e6b8a6cf95d195d89a9d560a828f963a51dc4f3273efd0a0ad281d`. It uses 512 windows × 2048 tokens, seed 2, pooled numerator/denominator aggregation for d1/d2 and arithmetic mean entropy delta. This does **not** match the Llama-3 canonical Pass1 profile (`DEFAULT_PROMPTS[:2]`, 2 sequences × 512 tokens, eager/bf16; no explicit seed parameter), so it is `LEGACY_UNBOUND_FOR_THIS_PORT`. Do not reuse its q/ranks.
- Existing L2 V1/native-projection and Direction-Corrected V2 materialization/quality artifacts are separate methods; do not reuse their schedules as 3COMP TRUE_TWO_PASS.
- Llama-2 Official basis at e=.0567 is not a final 3COMP lock. A basis file is not B_live; its saved rank trace does not fully cover the e=.060 uniform Pass1 threshold for every layer.

## Exact port stages

1. Re-verify L2 checkpoint/tokenizer fingerprints and official operator/config identity.
2. Construct **Uniform EigenAttention e=.060** from L2-native basis with canonical builder/materializer; store all large artifacts under F and keep protocol hashes/compact census on E. Match the L3 Pass1 diagnostic protocol: same two fixed DEFAULT prompts, 512 tokens, eager attention, bf16, Dense-vs-Pass1. Preserve the no-explicit-seed policy unless the source code shows an RNG dependency.
3. Run the pinned `step1_diagnose_llama.py` path on Dense vs that exact Pass1 model. Hash raw d1/d2/d5 profile; compute `q=MinMax(d1+d2+d5)` with the locked code.
4. Apply the exact TRUE_TWO_PASS mapping from `LLAMA3_3COMP_TRUE_TWO_PASS_METHOD_LOCK.json`, using L2 legal ranks/head-group sizes only as architecture-specific constraints.
5. Produce L2 rank schedule, canonical rank/allocation hashes, actual tensor-byte realization and resource lock against Official.
6. Validate materialization (no random low-rank parameters), direct operator parity and deterministic reload canary.
7. Only after those gates, run matched Official/3COMP WikiText-2 and C4 PPL, compute `J=0.5*(ln Wiki + ln C4)`, then the frozen zero-shot suite.

## Current readiness and compute

- No valid L2 Uniform(e=.060) Pass1 checkpoint/profile pair was found in the searched evidence paths.
- At audit time GPU2 was idle (`81,153 MiB` free, `0 MiB` used, 0% util); GPU0/1 have unrelated active processes and must not be interrupted. F had about 66 GiB free; E about 14 GiB free. Heavy model/basis outputs belong on F; only compact lock/provenance artifacts on E.
- No Llama-2 quality run was started by this audit. Next runnable step is only the missing Uniform(.060) Pass1/profile construction after the exact port launcher is checked against the canonical materializer; quality remains gated.
