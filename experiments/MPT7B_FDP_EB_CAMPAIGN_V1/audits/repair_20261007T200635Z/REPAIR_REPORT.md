# MPT audit repair

RELAXED_UNIFORM_SEMANTICS = TRUE_UNIFORM_EB
HELLASWAG_METRIC_CONTRACT = CORRECTED_TO_ACC
SOURCE_BINDING = PARTIAL
J_SOURCE_BINDING = PASS
FORMAL_5TASK_REUSE_ALLOWED = NO
ACC_ARM_IDENTITY_U011 = RELAXED
MEASURED_RESOURCE_MATCH_U011 = PASS
ACC_ARM_IDENTITY_U015 = RELAXED
MEASURED_RESOURCE_MATCH_U015 = PASS
ACC_ARM_IDENTITY_U019 = RELAXED
MEASURED_RESOURCE_MATCH_U019 = PASS

## Findings and remaining evidence gaps

- All three relaxed schedules reproduce 32/32 ranks from their one shared epsilon and preserved native trial curves, using first-failure semantics. Thus their schedule semantics are TRUE_UNIFORM_EB; no layer-specific rank repair is required. This is an independent resource-only replay, not a claim that the original generation script was recovered. Historical generation source remains a source-binding gap. See COMMON_EPSILON_NATIVE_TRIAL_REPLAY.json and mpt_repair_uniform_replay.py. Native function authority: mpt_fdp_protocol_v1.py::select_last_passing_before_first_failure. Per-layer spectral thresholds are not global reconstruction epsilons.
- Launcher, loader, config and result rank SHA chains now agree on RELAXED for all three accuracy arms. Current runner/source hashes and model/task manifests are captured; historical execution-time source SHA is not embedded in outputs. Binding remains PARTIAL.
- Six actual batch1/seq2048/BF16 runtime caches were counted. Projection weights excluded and reported separately.
- Correct metric contract and 10k item-stratified bootstrap use only existing accuracy files. Independent paired-window PPL bootstrap stores historical intervals, seed/algorithm differences and endpoint differences.
- Real J source recovered in run_u011_diagnostic_ppl.py, main(), summary J field.
- No formal quality was rerun. No existing predictions or results were modified. No promotion and no unnecessary 15/30 task rerun: RAW/OTHER is not the remaining cause.
