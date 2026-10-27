# Stage 0 CPU audit — LLAMA_DRM_A100_4GPU_V1

- generated: 2026-07-28T17:32:11+00:00
- result: PASS
- checks: 214/214 passed, 0 failed, 0 warnings
- model weights loaded onto GPU: no

| Group | Check | Status | Detail |
| --- | --- | --- | --- |
| A_inventory | present:common.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/common.py |
| A_inventory | present:orchestrator.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/orchestrator.py |
| A_inventory | present:worker_entry.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/worker_entry.py |
| A_inventory | present:LLAMA_DRM_METHOD_SPEC.md | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/LLAMA_DRM_METHOD_SPEC.md |
| A_inventory | present:adapters/transformers_compat.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/adapters/transformers_compat.py |
| A_inventory | present:audit/stage0_cpu_audit.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/audit/stage0_cpu_audit.py |
| A_inventory | present:scheduler/gpu_coloc_scheduler.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/scheduler/gpu_coloc_scheduler.py |
| A_inventory | present:stages/stage7_analysis.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/stages/stage7_analysis.py |
| A_inventory | present:status_bootstrap/protocol_templates.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/status_bootstrap/protocol_templates.py |
| A_inventory | present:runtime/experiments/run_llama_sliminfer_variants.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/experiments/run_llama_sliminfer_variants.py |
| A_inventory | present:runtime/models/modeling_llama.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/models/modeling_llama.py |
| A_inventory | present:runtime/models/configuration_llama.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/models/configuration_llama.py |
| A_inventory | present:runtime/utils/cache_utils.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/utils/cache_utils.py |
| A_inventory | present:runtime/utils/processor.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/utils/processor.py |
| A_inventory | present:runtime/bridgekv/rescuekv_scoring.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/bridgekv/rescuekv_scoring.py |
| A_inventory | present:runtime/bridgekv/structured_rescue_scoring.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/bridgekv/structured_rescue_scoring.py |
| A_inventory | present:runtime/bridgekv/cc_topology_policies.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/bridgekv/cc_topology_policies.py |
| A_inventory | present:runtime/bridgekv/acc_sliminfer.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/bridgekv/acc_sliminfer.py |
| A_inventory | present:runtime/longbench_eval.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/longbench_eval.py |
| A_inventory | present:runtime/tools/qwen_pipeline_common.py | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/tools/qwen_pipeline_common.py |
| A_inventory | present:runtime/configs/dataset2prompt.json | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/configs/dataset2prompt.json |
| A_inventory | present:runtime/configs/dataset2maxlen.json | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/configs/dataset2maxlen.json |
| A_inventory | present:protocol_ulb_v1/task_manifest.json | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/protocol_ulb_v1/task_manifest.json |
| A_inventory | cache_utils_is_authoritative_size | PASS | 10846 lines |
| A_inventory | runner_syntax_ok | PASS | compiles |
| A_inventory | modeling_llama_syntax_ok | PASS | compiles |
| B_compat | torch_present | PASS | 2.9.1+cu128 |
| B_compat | transformers_present | PASS | 4.57.3 |
| B_compat | flash_attn_absent_so_backend_is_sdpa | PASS | flash_attn=MISSING; frozen backend is sdpa |
| B_compat | compat_probe_ran | PASS | PROBE_JSON:{"compat": {"protocol_id": "LLAMA_DRM_A100_4GPU_V1", "transformers_version": "4.57.3", "torch_version": "2.9.1+cu128", "flash_attn_available": false, "recommended_backend": "sdpa", "patches |
| B_compat | loss_kwargs_shim_effective | PASS | {'protocol_id': 'LLAMA_DRM_A100_4GPU_V1', 'transformers_version': '4.57.3', 'torch_version': '2.9.1+cu128', 'flash_attn_available': False, 'recommended_backend': 'sdpa', 'patches': {'transformers.util |
| B_compat | modeling_llama_imports | PASS |  |
| B_compat | sliminfer_drm_hooks_intact | PASS | SlimInferCache/HsCache/processor referenced |
| B_compat | cache_utils_classes_import | PASS |  |
| B_compat | audit_probe_saw_no_gpu | PASS | cuda_device_count=0 |
| B_compat | sdpa_branch_does_not_hardcode_is_causal_false | PASS | the sdpa branch must derive is_causal from mask presence, not pin it to False |
| B_compat | sdpa_prefill_uses_causal_when_mask_is_none | PASS | a None mask (prefill) must fall back to is_causal=True |
| B_compat | sdpa_length_mismatch_fails_loudly | PASS | unequal q_len/kv_len with no mask must raise rather than mis-align the causal mask |
| B_compat | prefill_still_leaves_mask_none_as_authoritative | PASS | upstream prefill behaviour retained; only the sdpa consumer was corrected |
| B_compat | sdpa_causality_probe_ran | PASS | PROBE_JSON:{"causal_flag_matches_explicit_mask": true, "bidirectional_differs_from_causal": true}  |
| B_compat | is_causal_true_equals_explicit_causal_mask | PASS | {'causal_flag_matches_explicit_mask': True, 'bidirectional_differs_from_causal': True} |
| B_compat | unmasked_sdpa_is_not_causal | PASS | {'causal_flag_matches_explicit_mask': True, 'bidirectional_differs_from_causal': True} |
| C_model | model_snapshot_found | PASS | searched ['/mnt/d/jiaqizhang/llama3', '/data/jiaqizhang/llama3', '/d/jiaqizhang/llama3', '/home/bai-project/jiaqizhang/llama3'] |
| C_model | architecture_is_llama | PASS | ['LlamaForCausalLM'] |
| C_model | max_position_embeddings_8192 | PASS | 8192 |
| C_model | num_hidden_layers_32 | PASS | 32 |
| C_model | prune_layers_within_depth | PASS | [10, 20, 30] |
| C_model | max_model_len_leaves_generation_headroom | PASS | 7500 + 512 <= 8192 |
| C_model | base_model_has_no_chat_template | PASS | build_chat must pass prompts through unchanged |
| C_model | revision_recorded_as_unknown | PASS | UNKNOWN_LOCAL_SNAPSHOT |
| C_model | weight_shards_present | PASS | ['model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors', 'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors'] |
| D_dataset | longbench_dir_found | PASS | searched ['/home/bai-project/jiaqizhang/EigenAttn/data/LongBench', '/home/bai-project/jiaqizhang/data/LongBench', '/home/bai-project/jiaqizhang/LongBench'] |
| D_dataset | manifest_has_17_tasks | PASS | 17 |
| D_dataset | manifest_total_3950 | PASS | 3950 |
| D_dataset | manifest_task_order_matches_protocol | PASS | ['passage_retrieval_en', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news', 'vcsum', 'passage_count', 'trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 're |
| D_dataset | pooled_sample_average_forbidden | PASS | True |
| D_dataset | file_present:passage_retrieval_en | PASS | /diskF/HeadKV/data/LongBench/passage_retrieval_en.jsonl |
| D_dataset | sample_count:passage_retrieval_en | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:passage_retrieval_en | PASS | observed 0f34d2955bfc9bd2..., manifest 0f34d2955bfc9bd2... |
| D_dataset | file_present:qasper | PASS | /diskF/HeadKV/data/LongBench/qasper.jsonl |
| D_dataset | sample_count:qasper | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:qasper | PASS | observed 29aa07d2a63f36f4..., manifest 29aa07d2a63f36f4... |
| D_dataset | file_present:multifieldqa_en | PASS | /diskF/HeadKV/data/LongBench/multifieldqa_en.jsonl |
| D_dataset | sample_count:multifieldqa_en | PASS | observed 150, manifest 150 |
| D_dataset | file_sha256_matches_manifest:multifieldqa_en | PASS | observed 0aac182fd317dcf6..., manifest 0aac182fd317dcf6... |
| D_dataset | file_present:hotpotqa | PASS | /diskF/HeadKV/data/LongBench/hotpotqa.jsonl |
| D_dataset | sample_count:hotpotqa | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:hotpotqa | PASS | observed a0005ab2a1bc2ac3..., manifest a0005ab2a1bc2ac3... |
| D_dataset | file_present:2wikimqa | PASS | /diskF/HeadKV/data/LongBench/2wikimqa.jsonl |
| D_dataset | sample_count:2wikimqa | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:2wikimqa | PASS | observed dda279cf93a99e1e..., manifest dda279cf93a99e1e... |
| D_dataset | file_present:musique | PASS | /diskF/HeadKV/data/LongBench/musique.jsonl |
| D_dataset | sample_count:musique | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:musique | PASS | observed 4ac69b91281c4ec6..., manifest 4ac69b91281c4ec6... |
| D_dataset | file_present:gov_report | PASS | /diskF/HeadKV/data/LongBench/gov_report.jsonl |
| D_dataset | sample_count:gov_report | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:gov_report | PASS | observed d28112beb3a9b41d..., manifest d28112beb3a9b41d... |
| D_dataset | file_present:qmsum | PASS | /diskF/HeadKV/data/LongBench/qmsum.jsonl |
| D_dataset | sample_count:qmsum | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:qmsum | PASS | observed e992f5157679b0c1..., manifest e992f5157679b0c1... |
| D_dataset | file_present:multi_news | PASS | /diskF/HeadKV/data/LongBench/multi_news.jsonl |
| D_dataset | sample_count:multi_news | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:multi_news | PASS | observed f8817c00af317c2d..., manifest f8817c00af317c2d... |
| D_dataset | file_present:vcsum | PASS | /diskF/HeadKV/data/LongBench/vcsum.jsonl |
| D_dataset | sample_count:vcsum | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:vcsum | PASS | observed d7412361c4187946..., manifest d7412361c4187946... |
| D_dataset | file_present:passage_count | PASS | /diskF/HeadKV/data/LongBench/passage_count.jsonl |
| D_dataset | sample_count:passage_count | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:passage_count | PASS | observed 0455a8f1e8e99d63..., manifest 0455a8f1e8e99d63... |
| D_dataset | file_present:trec | PASS | /diskF/HeadKV/data/LongBench/trec.jsonl |
| D_dataset | sample_count:trec | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:trec | PASS | observed 20a88e14f7442f58..., manifest 20a88e14f7442f58... |
| D_dataset | file_present:triviaqa | PASS | /diskF/HeadKV/data/LongBench/triviaqa.jsonl |
| D_dataset | sample_count:triviaqa | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:triviaqa | PASS | observed ed2529de2e10b12c..., manifest ed2529de2e10b12c... |
| D_dataset | file_present:samsum | PASS | /diskF/HeadKV/data/LongBench/samsum.jsonl |
| D_dataset | sample_count:samsum | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:samsum | PASS | observed d435a4c7a05d2006..., manifest d435a4c7a05d2006... |
| D_dataset | file_present:lsht | PASS | /diskF/HeadKV/data/LongBench/lsht.jsonl |
| D_dataset | sample_count:lsht | PASS | observed 200, manifest 200 |
| D_dataset | file_sha256_matches_manifest:lsht | PASS | observed b57834ed102ce3a6..., manifest b57834ed102ce3a6... |
| D_dataset | file_present:lcc | PASS | /diskF/HeadKV/data/LongBench/lcc.jsonl |
| D_dataset | sample_count:lcc | PASS | observed 500, manifest 500 |
| D_dataset | file_sha256_matches_manifest:lcc | PASS | observed a35809a88d899f61..., manifest a35809a88d899f61... |
| D_dataset | file_present:repobench-p | PASS | /diskF/HeadKV/data/LongBench/repobench-p.jsonl |
| D_dataset | sample_count:repobench-p | PASS | observed 500, manifest 500 |
| D_dataset | file_sha256_matches_manifest:repobench-p | PASS | observed 919a4439e2a84ebb..., manifest 919a4439e2a84ebb... |
| D_dataset | total_samples_3950 | PASS | observed 3950, expected 3950 |
| D_dataset | six_task_subset_is_17_task_subset | PASS | ['qasper', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'repobench-p'] |
| D_dataset | six_task_total_1500 | PASS | observed 1500, expected 1500 |
| D_dataset | runner_never_calls_load_dataset | PASS | datasets.load_dataset removed from load_longbench_samples |
| D_dataset | runner_raises_filenotfound_when_missing | PASS | offline hard error present |
| E_scorer | scorer_function_present | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime/longbench_eval.py |
| E_scorer | prompt_template_for_all_17 | PASS | [] |
| E_scorer | max_new_tokens_for_all_17 | PASS | [] |
| E_scorer | max_new_tokens_matches_manifest | PASS | [] |
| E_scorer | vendored_deps_dir_configured | PASS | pointer /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/VENDORED_DEPS.txt |
| E_scorer | vendored_deps_dir_exists | PASS | /diskF/jiaqizhang/llama_drm_a100_4gpu_v1_deps |
| E_scorer | vendored_deps_on_worker_pythonpath | PASS | /home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1:/home/bai-project/jiaqizhang/EigenAttn/tools/llama_drm_a100_4gpu_v1/runtime:/diskF/jiaqizhang/llama_drm_a100_4gpu_v1_deps |
| E_scorer | scorer_probe_ran | PASS | /diskF/jiaqizhang/llama_drm_a100_4gpu_v1_deps/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources pa |
| E_scorer | scorer_dependency_importable:jieba | PASS | 0.42.1 |
| E_scorer | scorer_dependency_importable:rouge | PASS | 1.0.1 |
| E_scorer | scorer_dependency_importable:fuzzywuzzy | PASS | 0.18.0 |
| E_scorer | longbench_eval_scorer_importable | PASS |  |
| E_scorer | metric_defined_for_all_17_tasks | PASS | [] |
| F_method_cli | sliminfer_original:method | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'none', 'rescue_score': 'delta_o'} |
| F_method_cli | sliminfer_original:rescue_policy | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'none', 'rescue_score': 'delta_o'} |
| F_method_cli | sliminfer_original:rescue_score | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'none', 'rescue_score': 'delta_o'} |
| F_method_cli | sliminfer_original:no_rescue_ratio_override | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'none', 'rescue_score': 'delta_o'} |
| F_method_cli | sliminfer_larger:method | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'sliminfer_larger', 'rescue_score': 'delta_o'} |
| F_method_cli | sliminfer_larger:rescue_policy | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'sliminfer_larger', 'rescue_score': 'delta_o'} |
| F_method_cli | sliminfer_larger:rescue_score | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'sliminfer_larger', 'rescue_score': 'delta_o'} |
| F_method_cli | sliminfer_larger:no_rescue_ratio_override | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'sliminfer_larger', 'rescue_score': 'delta_o'} |
| F_method_cli | random_add:method | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'random', 'rescue_ratio': 0.05, 'seed': 0} |
| F_method_cli | random_add:rescue_policy | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'random', 'rescue_ratio': 0.05, 'seed': 0} |
| F_method_cli | random_add:rescue_score | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'random', 'rescue_ratio': 0.05, 'seed': 0} |
| F_method_cli | random_add:rescue_ratio | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'random', 'rescue_ratio': 0.05, 'seed': 0} |
| F_method_cli | drm_add:method | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'drm', 'rescue_ratio': 0.05} |
| F_method_cli | drm_add:rescue_policy | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'drm', 'rescue_ratio': 0.05} |
| F_method_cli | drm_add:rescue_score | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'drm', 'rescue_ratio': 0.05} |
| F_method_cli | drm_add:rescue_ratio | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'drm', 'rescue_ratio': 0.05} |
| F_method_cli | drm_swap:method | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'budget_neutral_drm_swap', 'rescue_score': 'drm'} |
| F_method_cli | drm_swap:rescue_policy | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'budget_neutral_drm_swap', 'rescue_score': 'drm'} |
| F_method_cli | drm_swap:rescue_score | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'budget_neutral_drm_swap', 'rescue_score': 'drm'} |
| F_method_cli | drm_swap:no_rescue_ratio_override | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'budget_neutral_drm_swap', 'rescue_score': 'drm'} |
| F_method_cli | fullkv:method | PASS | {'method': 'fullkv'} |
| F_method_cli | fullkv:no_rescue_flags | PASS | {'method': 'fullkv'} |
| F_method_cli | only_fullkv_differs_in_method | PASS | {'sliminfer_original': 'sliminfer_original', 'sliminfer_larger': 'sliminfer_original', 'random_add': 'sliminfer_original', 'drm_add': 'sliminfer_original', 'drm_swap': 'sliminfer_original'} |
| F_method_cli | random_needs_explicit_seed | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'random', 'rescue_ratio': 0.05, 'seed': 0} |
| F_method_cli | argparse_probe_ran | PASS | , "max_model_len": 7500}, "random_add": {"method": "sliminfer_original", "rescue_policy": "add", "rescue_score": "random", "rescue_ratio": 0.05, "prune_layers": "10,20,30", "retain_tokens": "8192,4096 |
| F_method_cli | argparse_accepts_all_arms | PASS | {} |
| F_method_cli | resolved_frozen_config:sliminfer_original | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'none', 'rescue_score': 'delta_o', 'rescue_ratio': 0.02, 'prune_layers': '10,20,30', 'retain_tokens': '8192,4096,2048', 'dtype': 'bf16', 'backend': 's |
| F_method_cli | resolved_frozen_config:sliminfer_larger | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'sliminfer_larger', 'rescue_score': 'delta_o', 'rescue_ratio': 0.02, 'prune_layers': '10,20,30', 'retain_tokens': '8192,4096,2048', 'dtype': 'bf16', ' |
| F_method_cli | resolved_frozen_config:random_add | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'random', 'rescue_ratio': 0.05, 'prune_layers': '10,20,30', 'retain_tokens': '8192,4096,2048', 'dtype': 'bf16', 'backend': 'sdp |
| F_method_cli | resolved_frozen_config:drm_add | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'add', 'rescue_score': 'drm', 'rescue_ratio': 0.05, 'prune_layers': '10,20,30', 'retain_tokens': '8192,4096,2048', 'dtype': 'bf16', 'backend': 'sdpa', |
| F_method_cli | resolved_frozen_config:drm_swap | PASS | {'method': 'sliminfer_original', 'rescue_policy': 'budget_neutral_drm_swap', 'rescue_score': 'drm', 'rescue_ratio': 0.02, 'prune_layers': '10,20,30', 'retain_tokens': '8192,4096,2048', 'dtype': 'bf16' |
| G_random | random_vs_drm_differ_only_in_score_and_seed | PASS | ['rescue_score', 'seed'] |
| G_random | random_and_drm_share_rescue_policy | PASS | ['add', 'add'] |
| G_random | random_and_drm_share_rescue_ratio | PASS | [0.05, 0.05] |
| G_random | rescue_M_is_6_3_2_at_ratio_0p05 | PASS | {'pages': [128, 64, 32], 'derived_M': [6, 3, 2], 'frozen_M': [6, 3, 2]} |
| G_random | M_identical_across_add_arms | PASS | M depends only on (k_active, rescue_ratio), never on rescue_score |
| G_random | add_maps_to_additive_rescue | PASS | policy alias present |
| G_random | larger_shares_additive_budget | PASS | K+M budget for Larger |
| G_random | additive_rescue_reports_K_plus_M | PASS | additive arms are K+M |
| G_random | swap_reports_K_budget | PASS | swap arms stay at K |
| G_random | M_formula_present_in_source | PASS | budget arithmetic is score-independent |
| G_random | random_score_is_seeded | PASS | random rescue scores derive from cfg['seed'] via a CPU generator |
| G_random | random_uses_same_scoring_entrypoint | PASS | random is a branch of the shared rescue-score function, not a separate path |
| G_random | random_seeds_are_distinct | PASS | [0, 1, 2] |
| H_drm_rules | shared_A_K_active_set | PASS | the fixed-page swap path consumes the concrete SlimInfer A_K and inactive P set |
| H_drm_rules | sliminfer_larger_extra_budget_rule | PASS | Larger enlarges the retained budget by round(K*ratio) |
| H_drm_rules | add_policy_dispatch | PASS | additive rescue routed through the shared policy |
| H_drm_rules | budget_neutral_drm_swap_rule | PASS | swap policy recognised in the page path |
| H_drm_rules | drm_score_entrypoint | PASS | DRM S0 scoring invoked |
| H_drm_rules | drm_rescue_score_branch | PASS | drm rescue-score branch present |
| H_drm_rules | residual_aware_proxy_fields | PASS | residual-aware audit fields computed |
| H_drm_rules | block_size_64_constant | PASS | page size 64 frozen in the DRM cache |
| H_drm_rules | token_unit_default_8 | PASS | token unit 8 derived from page size |
| H_drm_rules | local_pruning_window_4 | PASS | local query window 4 |
| H_drm_rules | swap_threshold_default_0p8_overridden_to_0p9 | PASS | gamma is configurable; the run pins it to 0.9 via --swap_threshold |
| H_drm_rules | cache_utils_unmodified_authoritative | PASS | authoritative import block intact; CC/ACC satisfied by stub modules |
| I_scheduler | estimated_peak_rule | PASS | {'expected': {'10.0': 14.0, '30.0': 36.0, '100.0': 120.0}, 'observed': {'10.0': 14.0, '30.0': 36.0, '100.0': 120.0}} |
| I_scheduler | colocates_multiple_workers_per_gpu | PASS | {'placed': 8, 'per_gpu': {'GPU-MOCK-0': 2, 'GPU-MOCK-1': 2, 'GPU-MOCK-2': 2, 'GPU-MOCK-3': 2}, 'pending': 4} |
| I_scheduler | scheduling_is_not_exclusive_only | PASS | {'GPU-MOCK-0': 4, 'GPU-MOCK-1': 4, 'GPU-MOCK-2': 4, 'GPU-MOCK-3': 4} |
| I_scheduler | max_workers_per_gpu_cap | PASS | {'GPU-MOCK-0': 4, 'GPU-MOCK-1': 4, 'GPU-MOCK-2': 4, 'GPU-MOCK-3': 4} |
| I_scheduler | safety_reserve_blocks_overcommit | PASS | {'free_gib': 30.0, 'needed_gib': 35.0, 'placed': 0} |
| I_scheduler | projected_peak_prevents_launch_burst_oversubscription | PASS | {'placed': 2, 'reported_free_gib': 80.0, 'available_after_placement_gib': 28.0} |
| I_scheduler | exclusive_categories_get_whole_gpu | PASS | {'gov_gpu': 'GPU-MOCK-0', 'co_tenants': 0, 'placement': 'exclusive'} |
| I_scheduler | exclusive_category_list_complete | PASS | {'exclusive': ['gov_report', 'long_context_32k', 'fullkv', 'oom_fill', 'efficiency'], 'missing': []} |
| I_scheduler | canary_is_one_gpu_one_worker | PASS | {'workers': 1, 'gpus_used': 1} |
| I_scheduler | external_busy_yields_waiting_for_gpu_idle | PASS | {'state': 'WAITING_FOR_GPU_IDLE', 'placed': 0, 'blocked_only_by_external': True} |
| I_scheduler | no_code_path_signals_foreign_pids | PASS | {'killpg_call_sites': 1} |
| I_scheduler | oom_retries_exclusive_and_lowers_cap | PASS | {'cap_before': 4, 'cap_after': 3, 'retry_exclusive': True, 'retry_reuses_output_dir': True} |
| I_scheduler | oom_stops_after_max_attempts | PASS | {'queued': 0, 'failed': 1} |
| I_scheduler | gpu_pinned_by_uuid | PASS | workers are pinned by GPU UUID, not ordinal index |
| I_scheduler | persists_status_and_locks | PASS | {'files': {'GPU_STATUS.json': True, 'CURRENT_WORKERS.json': True, 'WORKER_REGISTRY.csv': True}, 'locks': ['gpu.GPU-MOCK-0.lock', 'gpu.GPU-MOCK-1.lock', 'scheduler.global.lock']} |
| I_scheduler | flock_is_mutually_exclusive | PASS | {'blocked_while_held': True, 'acquired_after_release': True} |
| J_resume | atomic_write_json_replaces_cleanly | PASS | {'first': {'a': 1}, 'second': {'a': 2, 'b': [1, 2, 3]}, 'temp_leftovers': []} |
| J_resume | atomic_write_never_leaves_partial_json | PASS | {'bytes': 200020} |
| J_resume | torn_jsonl_tail_is_ignored | PASS | {'records_counted': 2, 'note': 'torn final line ignored; earlier records kept'} |
| J_resume | resumes_newest_incomplete_result_dir | PASS | {'resumable': '/tmp/stage0_resume_tojdt6m8/results/llama_drm_a100_4gpu_v1_20260102_000000', 'older_complete': True, 'new_dir': 'llama_drm_a100_4gpu_v1_20260103_000000'} |
| J_resume | job_completion_requires_all_predictions | PASS | {'empty': False, 'at_199': False, 'at_200': True, 'progress_at_199': [199, 200]} |
| J_resume | stage_completion_is_a_durable_marker | PASS | stage markers are files |
| J_resume | single_orchestrator_instance_enforced | PASS | global flock makes the scheduler loop unique |
| J_resume | final_state_string_is_exact | PASS | ALL_REQUIRED_LLAMA_DRM_A100_EXPERIMENTS_COMPLETE |
| K_forbidden | forbidden_probe_ran | PASS | PROBE_JSON:{"cc_methods_empty": true, "acc_methods_empty": true, "cc_raises": true, "acc_raises": true}  |
| K_forbidden | cc_method_set_empty | PASS | {'cc_methods_empty': True, 'acc_methods_empty': True, 'cc_raises': True, 'acc_raises': True} |
| K_forbidden | acc_method_set_empty | PASS | {'cc_methods_empty': True, 'acc_methods_empty': True, 'cc_raises': True, 'acc_raises': True} |
| K_forbidden | cc_dispatch_raises | PASS | {'cc_methods_empty': True, 'acc_methods_empty': True, 'cc_raises': True, 'acc_raises': True} |
| K_forbidden | acc_dispatch_raises | PASS | {'cc_methods_empty': True, 'acc_methods_empty': True, 'cc_raises': True, 'acc_raises': True} |
| K_forbidden | no_cc_or_acc_method_scheduled | PASS | {'methods': ['fullkv', 'sliminfer_original'], 'offending': []} |
| L_storage | results_root_writable | PASS | /diskF/jiaqizhang/llama_drm_a100_4gpu_v1_results |
| L_storage | results_root_has_min_free_space | PASS | 60.6 GiB free at /diskF/jiaqizhang/llama_drm_a100_4gpu_v1_results; minimum 20.0 GiB |
| L_storage | tool_volume_not_critically_full | PASS | 20.81 GiB free on the volume holding the code; results are redirected via RESULTS_ROOT.txt |
