#!/usr/bin/env python3
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# REPO_ROOT is the runtime/ directory: it holds models/, utils/, bridgekv/ and tools/.
REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT.parent
for _root in (str(TOOL_ROOT), str(REPO_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

# Must run before models.modeling_llama is imported: restores transformers symbols that
# the authoritative slim export expects but transformers 4.57.3 no longer provides.
from adapters import transformers_compat as _transformers_compat  # noqa: E402

_transformers_compat.apply_compat()

import uuid
from bridgekv import build_blocks, build_bridge_graph, build_rescue_candidates, compute_bridge_utilities, compute_structural_scores, get_state, register_state
from bridgekv.metrics import write_bridgekv_summaries
from tools.qwen_pipeline_common import AtomicTaskTraceManager, canonical_sample_identity

LONG_BENCH_DATASETS = [
    "passage_retrieval_en",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "passage_count",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
]

TARGETS = {
    "fullkv_longbench_avg": 47.76,
    "sliminfer_longbench_avg": 47.38,
    "sliminfer_ttft_speedup_32k": 2.14,
    "sliminfer_e2e_speedup_32k": 1.84,
}

STRUCTURED_RESCUE_METHOD_SPECS = {
    "fixed_drm_add": {
        "method": "sliminfer_original",
        "block_strategy": "fixed",
        "rescue_policy": "add",
        "rescue_score": "drm",
        "rescue_ratio": 0.05,
    },
    "rh_pyramid_srm": {
        "method": "sliminfer_original",
        "block_strategy": "fixed",
        "rescue_policy": "add",
        "rescue_score": "rh_pyramid_srm",
        "rescue_ratio": 0.05,
        "rescue_granularity": "microblock",
        "microblock_size": 32,
    },
    "distilled_blockdrm": {
        "method": "sliminfer_original",
        "block_strategy": "fixed",
        "rescue_policy": "add",
        "rescue_score": "distilled_blockdrm",
        "rescue_ratio": 0.05,
        "microblock_size": 32,
    },
    "bridge_graph_srm": {
        "method": "sliminfer_original",
        "block_strategy": "fixed",
        "rescue_policy": "add",
        "rescue_score": "bridge_graph_srm",
        "rescue_ratio": 0.05,
    },
    "deferred_uncertainty_srm": {
        "method": "sliminfer_original",
        "block_strategy": "fixed",
        "rescue_policy": "add",
        "rescue_score": "deferred_uncertainty_srm",
        "rescue_ratio": 0.05,
        "deferred_mode": "audit",
    },
    "evidence_density_srm": {
        "method": "sliminfer_original",
        "block_strategy": "fixed",
        "rescue_policy": "add",
        "rescue_score": "evidence_density_srm",
        "rescue_ratio": 0.05,
        "rescue_granularity": "microblock",
        "microblock_size": 32,
    },
}


# Frozen per-family SlimInfer/DRM schedule defaults. Qwen2.5-7B-Instruct keeps the
# published reproduction schedule; Llama-3-8B (32 layers, mpe=8192) uses the
# LLAMA_DRM_A100_4GPU_V1 schedule. Only applied when the CLI leaves the flag unset.
MODEL_FAMILY_SCHEDULE_DEFAULTS: Dict[str, Dict[str, str]] = {
    "qwen2": {
        "prune_layers": "9,18,26",
        "retain_tokens": "12288,6144,4096",
        "dtype": "fp16",
    },
    "llama": {
        "prune_layers": "10,20,30",
        "retain_tokens": "8192,4096,2048",
        "dtype": "bf16",
    },
}


def flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLaMA/Qwen SlimInfer + DRM reproduction and variants.")
    parser.add_argument("--model_path", type=str, default="/home/bai-project/jiaqizhang/llama3")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--method", type=str, default="sliminfer_original",
                        choices=["fullkv", "sliminfer_original", "sp_sliminfer", "sp_sliminfer_v2",
                                 "cc_sliminfer", "cc_sliminfer_v1",
                                 "cc_gated_v0", "cc_spike_hybrid_v0", "cc_coverage_v0",
                                 "cc_diversify_v0", "cc_router_v0",
                                 "acc_sliminfer", "acc_sliminfer_v1", "drm_swap_sliminfer",
                                 "drm_acc_swap_sliminfer", "drm_acc_swap_sliminfer_v1", "drm_acc_add_sliminfer",
                                 "ud_sliminfer", "divcov_sliminfer",
                                 "rank_stability_sliminfer",
                                 "confidence_gated_sliminfer", "margin_stable_sliminfer",
                                 "multi_window_sliminfer", "adaptive_budget_sliminfer", "reliability_gated_sliminfer",
                                 "dc_sliminfer", "adaptive_dc_sliminfer_v3", "qdc_sliminfer", "mcu_sliminfer", "sliminfer_variant", "bridgekv", "fd_score", "fdp_schedule",
                                 "rescuekv", "kv_separated", "diffusion_ready", "residual_comp", "joint_token_rank"])
    parser.add_argument("--scoring", type=str, default="token_unit")
    parser.add_argument(
        "--fairness_base_scoring",
        type=str,
        default="",
        choices=["", "token_unit_qk", "delta_o"],
        help="Opt-in active-set base scoring override for fairness-closure runs.",
    )
    parser.add_argument("--model_family", type=str, default="llama", choices=["qwen2", "llama"])
    parser.add_argument("--backend", type=str, default="auto", choices=["flash_attention_2", "sdpa", "auto"])
    parser.add_argument(
        "--eval",
        type=str,
        default="all",
        choices=["longbench", "gate_audit", "latency", "ppl", "needle", "all"],
    )
    parser.add_argument("--longbench_dir", type=str,
                        default="/home/bai-project/jiaqizhang/EigenAttn/data/LongBench")
    parser.add_argument("--output_dir", type=str, default="results/qwen25_sliminfer_variants/exp")
    parser.add_argument("--context_lengths", type=str, default="8192,16384,24576,28672,32768")
    parser.add_argument("--decode_tokens", type=int, default=16)
    parser.add_argument("--one_context_per_process", action="store_true")
    parser.add_argument("--isolated_run_id", type=str, default="")
    # None means "use the frozen per-family default" resolved in normalize_args().
    parser.add_argument("--dtype", type=str, default=None, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--block_strategy", type=str, default="fixed",
                        choices=["fixed", "semantic", "semantic_v2", "semantic_v3", "attention_boundary", "query_dynamic", "query_dynamic_v2", "query_dynamic_v3",
                                 "hierarchical", "hybrid_attention_fixed", "hybrid_attention_semantic", "fixed_active_rescue"])
    parser.add_argument(
        "--selection_path",
        type=str,
        default="original_runtime",
        choices=["original_runtime", "canonical_fixed_page"],
        help="Explicit selection-path identity; canonical_fixed_page reproduces the frozen official Qwen runs without coupling behavior to tracing.",
    )
    parser.add_argument("--target_block_tokens", type=int, default=64)
    parser.add_argument("--min_block_tokens", type=int, default=32)
    parser.add_argument("--max_block_tokens", type=int, default=128)
    parser.add_argument("--block_overlap_tokens", type=int, default=0)
    parser.add_argument("--semantic_parser", type=str, default="task_aware", choices=["task_aware"])
    parser.add_argument("--semantic_target_block_tokens", type=int, default=64)
    parser.add_argument("--semantic_min_block_tokens", type=int, default=32)
    parser.add_argument("--semantic_max_block_tokens", type=int, default=128)
    parser.add_argument("--semantic_page_align", type=int, default=1)
    parser.add_argument("--max_page_expansion_overhead", type=float, default=2.0)
    parser.add_argument("--attention_boundary_layer", type=int, default=9)
    parser.add_argument("--boundary_percentile", type=float, default=85.0)
    parser.add_argument("--query_top_ratio", type=float, default=0.2)
    parser.add_argument("--query_neighbor_sentences", type=int, default=1)
    parser.add_argument("--query_high_max_block_tokens", type=int, default=64)
    parser.add_argument("--query_low_max_block_tokens", type=int, default=128)
    parser.add_argument("--query_score_mode", type=str, default="lexical_bm25", choices=["lexical_bm25"])
    parser.add_argument("--query_parser", type=str, default="task_aware", choices=["task_aware"])
    parser.add_argument("--coarse_block_tokens", type=int, default=256)
    parser.add_argument("--fine_block_tokens", type=int, default=64)
    parser.add_argument("--medium_block_tokens", type=int, default=128)
    parser.add_argument("--top_coarse_ratio", type=float, default=0.3)
    parser.add_argument("--budget_mode", type=str, default="block_count", choices=["block_count", "token_budget"])
    parser.add_argument("--target_token_budget_source", type=str, default="custom",
                        choices=["fixed_K", "fixed_KplusM_larger", "fixed_KplusM_drm", "attention_raw", "hierarchical_raw", "custom"])
    parser.add_argument("--target_token_budget", type=int, default=0)
    parser.add_argument("--token_budget_tolerance", type=float, default=0.05)
    parser.add_argument("--selection_cost", type=str, default="expanded_page_tokens", choices=["semantic_span_tokens", "expanded_page_tokens"])
    parser.add_argument("--score_density_mode", type=str, default="none", choices=["none", "sqrt_len", "len"])
    parser.add_argument("--allow_budget_overshoot", type=int, default=1)
    parser.add_argument("--soft_budget_tolerance", type=float, default=0.03)
    parser.add_argument("--hard_budget_tolerance", type=float, default=0.05)
    parser.add_argument("--active_budget_mode", type=str, default="block_count", choices=["block_count", "token_budget"])
    parser.add_argument("--rescue_budget_mode", type=str, default="block_count", choices=["block_count", "token_budget"])
    parser.add_argument("--target_active_tokens", type=int, default=0)
    parser.add_argument("--target_final_tokens", type=int, default=0)
    parser.add_argument("--target_final_tokens_source", type=str, default="custom",
                        choices=["fixed_KplusM_drm", "fixed_KplusM_larger", "custom"])
    parser.add_argument("--active_block_strategy", type=str, default="attention_boundary")
    parser.add_argument("--rescue_candidate_strategy", type=str, default="fixed")
    parser.add_argument("--hybrid_exclude_active_overlap", type=int, default=1)
    parser.add_argument("--hybrid_overlap_threshold", type=float, default=0.5)
    # A-F block-system diagnostic rescue-candidate params
    parser.add_argument("--rescue_overlap_window_tokens", type=int, default=64)
    parser.add_argument("--rescue_overlap_stride_tokens", type=int, default=32)
    parser.add_argument("--rescue_evidence_half_window_tokens", type=int, default=64)
    parser.add_argument("--rescue_evidence_top_ratio", type=float, default=0.2)
    parser.add_argument("--rescue_semantic_page_tolerance", type=int, default=16)
    parser.add_argument("--lambda_len", type=float, default=0.1)
    parser.add_argument("--beta_semantic", type=float, default=0.1)
    # Structural DRM-Add diagnostic knobs (directions 1-5)
    parser.add_argument("--rescue_selector", type=str, default="drm_mmr",
                        choices=["drm_mmr", "drm_only", "query_only", "damage_only", "coverage_aware", "functional_delta"])
    parser.add_argument("--rescue_gate_mode", type=str, default="none",
                        choices=["none", "binary", "dynamic_M"])
    parser.add_argument("--rescue_gate_threshold", type=float, default=0.4)
    parser.add_argument("--coverage_gamma", type=float, default=0.5)
    parser.add_argument("--noise_mu", type=float, default=0.2)
    parser.add_argument("--block_budget_mode", type=str, default="none",
                        choices=["none", "uniform", "query_weighted", "coverage_gap", "drm_peak"])
    parser.add_argument("--block_budget_block_tokens", type=int, default=256)
    parser.add_argument("--block_budget_log_weight", type=float, default=0.5)
    parser.add_argument("--verifier_mode", type=str, default="none",
                        choices=["none", "light", "attn", "oracle"])
    parser.add_argument("--verifier_topn_mult", type=float, default=3.0)
    parser.add_argument("--gamma_boundary", type=float, default=0.25)
    parser.add_argument("--boundary_weight_active_score", type=int, default=0)
    parser.add_argument("--gamma_active_boundary", type=float, default=0.1)
    parser.add_argument("--calibrated_budget_config", type=str, default="")
    parser.add_argument("--token_unit_size", type=int, default=8)
    parser.add_argument("--local_query_window", type=int, default=4)
    parser.add_argument("--swap_threshold", type=float, default=0.9)
    parser.add_argument("--cgs_confidence_threshold", type=float, default=0.35,
                        help="Confidence-Gated SlimInfer: below this, stabilize active set vs previous prune layer.")
    parser.add_argument("--cgs_min_swap_threshold", type=float, default=0.5,
                        help="Decode-phase floor swap threshold when pruning confidence is low.")
    parser.add_argument("--mss_swap_margin_ratio", type=float, default=0.08,
                        help="Margin-Stable SlimInfer: min relative score gap to swap a block.")
    parser.add_argument("--mw_windows", type=str, default="1,4,8",
                        help="Multi-Window SlimInfer: comma-separated local query windows.")
    parser.add_argument("--rgs_confidence_low", type=float, default=0.35)
    parser.add_argument("--rgs_confidence_high", type=float, default=0.65)
    parser.add_argument("--prune_layers", type=str, default=None)
    parser.add_argument("--retain_tokens", type=str, default=None)
    parser.add_argument("--preserve_first_block", action="store_true", default=True)
    parser.add_argument("--enable_cpu_offload", action="store_true", default=True)
    parser.add_argument("--enable_async_kv", action="store_true", default=True)
    parser.add_argument("--decode_policy", type=str, default="original",
                        choices=["reuse_prefill", "dynamic_refresh", "periodic_refresh", "original", "hysteresis"])
    parser.add_argument("--refresh_interval", type=int, default=8)
    parser.add_argument("--active_set_hysteresis", type=float, default=0.0)
    parser.add_argument("--active_set_momentum", type=float, default=0.0)
    parser.add_argument("--profile_io", action="store_true")
    parser.add_argument("--profile_memory", action="store_true")
    parser.add_argument("--profile_cuda_events", action="store_true")
    parser.add_argument("--profile_active_blocks", action="store_true")
    parser.add_argument("--profile_score_overhead", action="store_true")
    parser.add_argument("--save_metrics_json", action="store_true")
    parser.add_argument("--save_layer_stats", action="store_true")
    parser.add_argument("--save_block_scores", nargs="?", const="aggregate_only", default="none",
                        choices=["aggregate_only", "sampled", "none"])
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_selected_block_snippets", action="store_true")
    parser.add_argument("--max_snippets_per_sample", type=int, default=8)

    parser.add_argument("--fd_alpha_o", type=float, default=None)
    parser.add_argument("--fd_beta_h", type=float, default=None)
    parser.add_argument("--fd_gamma_s", type=float, default=None)
    parser.add_argument("--fd_beta_v", type=float, default=0.0)
    parser.add_argument("--fd_normalize", type=str, default="per_layer", choices=["per_layer", "global", "none"])
    parser.add_argument("--fd_score_clip", type=float, default=0.0)
    parser.add_argument("--fd_topk_mode", type=str, default="keep_high_damage",
                        choices=["keep_high_damage", "drop_low_damage"])
    parser.add_argument("--fd_compute_scope", type=str, default="prefill_only",
                        choices=["prefill_only", "decode_only", "prefill_and_decode"])
    parser.add_argument("--fd_approx", type=str, default="block_contrib",
                        choices=["block_contrib", "token_unit_contrib", "head_group_contrib"])
    parser.add_argument("--fd_use_entropy", action="store_true")
    parser.add_argument("--fd_use_score_proxy", action="store_true")
    parser.add_argument("--fd_use_value_norm", action="store_true")
    parser.add_argument("--fd_use_renorm", action="store_true", default=True)
    parser.add_argument("--hybrid_qk_weight", type=float, default=0.5)
    parser.add_argument("--hybrid_o_weight", type=float, default=0.5)
    parser.add_argument("--risk_ratio", type=float, default=0.10)

    parser.add_argument("--bridge_alpha", type=float, default=0.5)
    parser.add_argument("--bridge_beta", type=float, default=0.4)
    parser.add_argument("--bridge_gamma", type=float, default=0.0)
    parser.add_argument("--bridge_eta", type=float, default=0.1)
    parser.add_argument("--selection_policy", type=str, default="linear_fusion",
                        choices=["linear_fusion", "core_bridge_quota", "boundary_rerank", "agreement_gated_rescue", "evidence_first",
                                 "evidence_core_limited_rescue"])
    parser.add_argument("--base_score", type=str, default="delta_o", choices=["delta_o", "qk"])
    parser.add_argument("--quota_ratio", type=float, default=0.0)
    parser.add_argument("--core_ratio", type=float, default=0.90)
    parser.add_argument("--boundary_ratio", type=float, default=0.10)
    parser.add_argument("--replace_ratio", type=float, default=0.05)
    parser.add_argument("--rescue_ratio", type=float, default=0.02)
    parser.add_argument("--rescue_source", type=str, default="bridge",
                        choices=["bridge", "ppr", "path_coverage", "random_boundary", "qk_disagreement", "semantic_diversity", "delta_o_boundary"])
    parser.add_argument(
        "--rescue_policy",
        type=str,
        default="none",
        choices=[
            "none",
            "add",
            "swap",
            "conditional_swap",
            "sliminfer_larger",
            "budget_neutral_drm_swap",
            "gated_budget_neutral_drm_swap",
            "lssr_policy_b_middle_donor_late_receiver",
            "debug_recent_only",
            "debug_inverted",
        ],
    )
    parser.add_argument("--rescue_score", type=str, default="delta_o",
                        choices=["delta_o", "obd", "diffusion_gap", "redundancy", "path", "abc", "random", "sliminfer_next", "mmr", "drm",
                                 "query_drm", "stochastic_drm", "query_stochastic_drm", "boundary_drm",
                                 "rh_pyramid_srm", "distilled_blockdrm", "bridge_graph_srm", "deferred_uncertainty_srm",
                                 "evidence_density_srm"])
    parser.add_argument("--rescue_granularity", type=str, default="page", choices=["page", "microblock"])
    parser.add_argument("--microblock_size", type=int, default=32, choices=[16, 32, 64])
    parser.add_argument(
        "--microblock_parent_aggregation",
        type=str,
        default="max",
        choices=["max", "mean_top2", "sum_topk", "rank_weighted_sum"],
    )
    parser.add_argument("--rh_head_file", type=str, default="")
    parser.add_argument("--auto_retrieval_heads_top_frac", type=float, default=0.10)
    parser.add_argument("--rh_weight", type=float, default=0.35)
    parser.add_argument("--pyramid_weight", type=float, default=0.20)
    parser.add_argument("--low_layer_ratio", type=float, default=0.33)
    parser.add_argument("--mid_layer_ratio", type=float, default=0.34)
    parser.add_argument("--high_layer_ratio", type=float, default=0.33)
    parser.add_argument("--teacher_label_file", type=str, default="")
    parser.add_argument("--distilled_selector_file", type=str, default="")
    parser.add_argument("--teacher_align_weight", type=float, default=0.40)
    parser.add_argument("--bridge_weight", type=float, default=0.30)
    parser.add_argument("--bridge_top_candidates", type=int, default=256)
    parser.add_argument("--bridge_min_link", type=float, default=0.05)
    parser.add_argument("--deferred_mode", type=str, default="audit", choices=["audit", "dynamic"])
    parser.add_argument("--deferred_trigger_entropy", type=float, default=3.0)
    parser.add_argument("--deferred_trigger_margin", type=float, default=0.10)
    parser.add_argument("--deferred_topup_m", type=int, default=4)
    parser.add_argument("--deferred_max_triggers", type=int, default=3)
    parser.add_argument("--density_alpha_max", type=float, default=1.0)
    parser.add_argument("--density_alpha_count", type=float, default=0.6)
    parser.add_argument("--density_alpha_continuity", type=float, default=0.8)
    parser.add_argument("--density_alpha_coverage", type=float, default=0.5)
    parser.add_argument("--density_lambda_red", type=float, default=0.3)
    parser.add_argument(
        "--density_threshold_mode",
        type=str,
        default="mean_std",
        choices=["mean_std", "top_quantile"],
    )
    parser.add_argument("--density_top_quantile", type=float, default=0.30)
    parser.add_argument("--dump_candidate_scores", type=int, default=0)
    parser.add_argument("--dump_final_selection", type=int, default=0)
    parser.add_argument("--strict_structured_path_check", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--feature_dump_path", type=str, default="")
    parser.add_argument("--drm_trace_dir", type=str, default="")
    parser.add_argument("--rescue_audit_path", type=str, default="")
    parser.add_argument("--lssr_policy_b_plan_path", type=str, default="")
    parser.add_argument("--uasr_score_dump_path", type=str, default="")
    parser.add_argument("--sp_trajectory_dump_path", type=str, default="")
    parser.add_argument("--uasr_tail_width", type=int, default=8)
    parser.add_argument("--debug_used_block_audit_path", type=str, default="")
    parser.add_argument("--debug_used_block_include_full", type=str_to_bool, default=False)
    parser.add_argument("--debug_used_block_full_layers", type=str, default="")
    parser.add_argument("--block_layout_audit_path", type=str, default="")
    parser.add_argument("--dc_selection_audit_path", type=str, default="")
    parser.add_argument("--dc_activation_audit_path", type=str, default="")
    parser.add_argument("--cc_selection_audit_path", type=str, default="")
    parser.add_argument(
        "--active_block_trace",
        type=str,
        default="none",
        choices=["none", "summary", "full"],
        help="Behavior-neutral buffered trace. Unlike cc_selection_audit_path, this never selects the audit-only fixed-page path.",
    )
    parser.add_argument("--active_block_trace_dir", type=str, default="")
    parser.add_argument("--protocol_id", type=str, default="")
    parser.add_argument("--implementation_revision", type=str, default="")
    parser.add_argument("--source_config_hash", type=str, default="")
    parser.add_argument(
        "--selection_digest_dir",
        type=str,
        default="",
        help="Canary-only behavior-neutral selected-ID digest; does not activate full tracing or alter selection path.",
    )
    parser.add_argument(
        "--cc_gate_config_path",
        type=str,
        default=str(REPO_ROOT / "configs" / "cc_gate_v0.json"),
        help="JSON thresholds for CC topology gate/router (gold-free).",
    )
    parser.add_argument("--qdc_selection_audit_path", type=str, default="")
    parser.add_argument("--mcu_selection_audit_path", type=str, default="")
    parser.add_argument("--mcu_frozen_selection_path", type=str, default="")
    parser.add_argument("--dc_alpha", type=float, default=0.10)
    parser.add_argument("--dc_beta", type=float, default=0.05)
    parser.add_argument("--dc_lambda", type=float, default=0.15)
    parser.add_argument("--dc_gamma", type=float, default=0.05)
    parser.add_argument("--adaptive_dc_v3_gate_json", type=str, default="")
    parser.add_argument("--debug_generation_jsonl", type=str, default="")
    parser.add_argument("--debug_first_token_logits_path", type=str, default="")
    parser.add_argument("--gate_mode", type=str, default="shadow", choices=["shadow", "apply"])
    parser.add_argument("--gate_policy", type=str, default="conservative_v1",
                        choices=["conservative_v1", "gate_a_removal_risk"])
    parser.add_argument("--gate_default_policy", type=str, default="original",
                        choices=["original"])
    parser.add_argument("--gate_rescue_margin_min", type=float, default=0.25)
    parser.add_argument("--gate_coverage_gap_min", type=float, default=0.0)
    parser.add_argument("--gate_deletion_risk_max", type=float, default=0.20)
    parser.add_argument(
        "--gate_feature_version",
        type=str,
        default="selector_native_v2",
        choices=[
            "selector_native_v2",
            "selector_native_v3",
            "selector_native_v3_1",
        ],
    )
    parser.add_argument(
        "--gate_sentence_jaccard_novel_max", type=float, default=0.35
    )
    parser.add_argument("--swap_drop_score", type=str, default="sliminfer", choices=["sliminfer", "obd"])
    parser.add_argument("--alpha_diff", type=float, default=0.5)
    parser.add_argument("--lambda_red", type=float, default=0.25)
    parser.add_argument("--eta_rescue", type=float, default=0.25)
    parser.add_argument("--eta_active", type=float, default=0.25)
    parser.add_argument("--beta_path", type=float, default=0.25)
    parser.add_argument("--beta_query", type=float, default=0.5)
    parser.add_argument("--stochastic_mode", type=str, default="gumbel", choices=["gumbel", "pool"])
    parser.add_argument("--gumbel_tau", type=float, default=0.05)
    parser.add_argument("--candidate_pool_ratio", type=float, default=0.3)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--swap_tau", type=float, default=0.0)
    parser.add_argument("--agreement_top_p", type=float, default=0.10)
    parser.add_argument("--evidence_ratio", type=float, default=0.90)
    parser.add_argument("--bridge_ratio", type=float, default=0.10)
    parser.add_argument("--control_type", type=str, default="none",
                        choices=["none", "random_graph", "random_anchor", "shuffled_bridge_scores", "random_boundary_replacement"])
    parser.add_argument("--agreement_signals", type=str, default="delta_o,qk,lexical_overlap,structural")
    parser.add_argument("--bridge_graph_edges", type=str, default="query_semantic_pos",
                        choices=["query_only", "query_semantic", "query_semantic_pos", "query_semantic_pos_attention", "random_graph", "random_graph_control"])
    parser.add_argument("--bridge_score", type=str, default="ppr",
                        choices=["ppr", "ppr_product", "degree_centrality", "random", "shuffled_bridge_scores_control", "path_coverage"])
    parser.add_argument("--bridge_ppr_alpha", type=float, default=0.85)
    parser.add_argument("--bridge_pseudo_evidence_ratio", type=float, default=0.10)
    parser.add_argument("--bridge_anchor_mode", type=str, default="union_weighted",
                        choices=["delta_o_top", "qk_top", "lexical_overlap_top", "intersection_delta_o_qk", "union_weighted", "retrieval_struct_top", "random_anchor_control"])
    parser.add_argument("--bridge_anchor_ratio", type=float, default=0.10)
    parser.add_argument("--bridge_pos_edge_weight", type=float, default=0.05)
    parser.add_argument("--bridge_topk_sem_edges", type=int, default=16)
    parser.add_argument("--bridge_query_edge_weight", type=float, default=1.0)
    parser.add_argument("--bridge_edge_weighting", type=str, default="raw", choices=["raw", "idf"])
    parser.add_argument("--bridge_hub_penalty", type=str_to_bool, default=False)
    parser.add_argument("--bridge_hub_lambda", type=float, default=0.0)
    parser.add_argument("--bridge_idf_node", type=str_to_bool, default=False)
    parser.add_argument("--bridge_semantic_threshold", type=float, default=0.0)
    parser.add_argument("--num_evidence_anchors", type=int, default=16)
    parser.add_argument("--max_paths_per_anchor", type=int, default=1)
    parser.add_argument("--bridge_norm", type=str, default="per_layer", choices=["per_layer", "minmax", "zscore", "none"])
    parser.add_argument("--local_score", type=str, default="delta_o", choices=["delta_o", "qk", "none"])
    parser.add_argument("--bridge_structural_risk", type=str_to_bool, default=True)
    parser.add_argument("--uncertainty_tau", type=float, default=0.5)
    parser.add_argument("--reserve_ratio", type=float, default=0.02)

    parser.add_argument("--schedule_mode", type=str, default="fixed", choices=["fixed", "fdp", "diffusion", "hybrid"])
    parser.add_argument("--fdp_mode", type=str, default="mild", choices=["mild", "aggressive", "matched"])
    parser.add_argument("--target_budget", type=str, default="matched_original",
                        choices=["original", "matched_original", "lower_than_original", "latency_matched"])
    parser.add_argument("--calib_samples", type=int, default=64)
    parser.add_argument("--calib_seqlen", type=int, default=8192)
    parser.add_argument("--calib_tasks", type=str, default="qasper,hotpotqa,2wikimqa,passage_retrieval_en")
    parser.add_argument("--max_prune_points", type=int, default=3)
    parser.add_argument("--fdp_profile_path", type=str, default="")
    parser.add_argument("--save_fdp_profile", action="store_true")
    parser.add_argument("--diffusion_metric", type=str, default="delta_o", choices=["delta_o", "delta_oh", "logit_proxy"])
    parser.add_argument("--kv_mode", type=str, default="none", choices=["none", "guardrail", "weighted", "true_separated_if_supported"])
    parser.add_argument("--kv_topk_risk_ratio", type=float, default=0.10)
    parser.add_argument("--lambda_k", type=float, default=0.5)
    parser.add_argument("--lambda_v", type=float, default=0.5)
    parser.add_argument("--k_high_risk_action", type=str, default="keep_full",
                        choices=["keep_full", "keep_k_only", "boost_score"])
    parser.add_argument("--v_high_risk_action", type=str, default="keep_full",
                        choices=["keep_full", "keep_v_only", "boost_score"])
    parser.add_argument("--diffusion_tau", type=float, default=0.6)
    parser.add_argument("--residual_type", type=str, default="none", choices=["none", "mean", "attn", "value_contrib"])
    parser.add_argument("--residual_slots", type=int, default=8)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--residual_grouping", type=str, default="global",
                        choices=["global", "per_layer", "per_head_group"])
    parser.add_argument("--residual_insert_mode", type=str, default="attention_only",
                        choices=["attention_only", "hidden_state_token"])
    parser.add_argument("--head_group_mode", type=str, default="none",
                        choices=["none", "grouped", "per_head_debug"])
    parser.add_argument("--num_head_groups", type=int, default=4)
    parser.add_argument("--head_agg", type=str, default="mean", choices=["mean", "max", "top2mean", "risk_weighted"])
    parser.add_argument("--token_schedule", type=str, default="sliminfer")
    parser.add_argument("--rank_schedule", type=str, default="none")
    parser.add_argument("--eigen_basis_path", type=str, default="none")
    parser.add_argument("--per_layer_eb_path", type=str, default="none")

    parser.add_argument("--longbench_datasets", type=str, default=",".join(LONG_BENCH_DATASETS))
    parser.add_argument("--longbench_tasks", type=str, default="")
    parser.add_argument("--longbench_max_samples", type=int, default=0)
    parser.add_argument("--longbench_sample_indices_by_task", type=str, default="")
    parser.add_argument("--max_samples_per_task", type=int, default=0)
    parser.add_argument("--longbench_subset_name", type=str, default="")
    parser.add_argument("--generation_max_new_tokens", type=int, default=0)
    parser.add_argument("--min_free_gpu_gb", type=float, default=0.0)
    parser.add_argument("--smoke_mode", type=str_to_bool, default=False)
    parser.add_argument("--max_smoke_samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=131072)
    parser.add_argument("--ppl_dataset", type=str, default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--ppl_samples", type=int, default=0)
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--needle_context_lengths", type=str, default="")
    parser.add_argument("--needle_depths", type=str, default="")
    parser.add_argument("--needle_samples", type=int, default=0)
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    if value is None or value == "":
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_sample_indices_by_task(value: str) -> Dict[str, List[int]]:
    value = str(value or "").strip()
    if not value:
        return {}
    parsed: Dict[str, List[int]] = {}
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                "--longbench_sample_indices_by_task must use task:i,j;task:k format"
            )
        task, indices = chunk.split(":", 1)
        task = task.strip()
        if not task:
            raise ValueError("--longbench_sample_indices_by_task has empty task")
        selected: List[int] = []
        seen = set()
        for part in indices.split(","):
            part = part.strip()
            if not part:
                continue
            index = int(part)
            if index < 0:
                raise ValueError(
                    f"--longbench_sample_indices_by_task has negative index: {index}"
                )
            if index not in seen:
                selected.append(index)
                seen.add(index)
        if not selected:
            raise ValueError(
                f"--longbench_sample_indices_by_task has no indices for {task}"
            )
        parsed[task] = selected
    return parsed


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    family_defaults = MODEL_FAMILY_SCHEDULE_DEFAULTS.get(
        getattr(args, "model_family", "llama"), MODEL_FAMILY_SCHEDULE_DEFAULTS["llama"]
    )
    for flag, value in family_defaults.items():
        if getattr(args, flag, None) in (None, ""):
            setattr(args, flag, value)
    if args.longbench_tasks:
        args.longbench_datasets = args.longbench_tasks
    if args.max_samples_per_task > 0:
        args.longbench_max_samples = args.max_samples_per_task
    args.longbench_sample_indices_by_task_map = parse_sample_indices_by_task(
        args.longbench_sample_indices_by_task
    )
    if args.smoke_mode and args.max_smoke_samples > 0 and args.longbench_max_samples <= 0:
        args.longbench_max_samples = args.max_smoke_samples
    if args.scoring == "token_unit":
        args.scoring = "token_unit_qk"
    if args.method == "rescuekv":
        args.method = "bridgekv"
        if args.selection_policy == "evidence_core_limited_rescue":
            args.selection_policy = "core_bridge_quota"
    if args.method == "bridgekv":
        args.scoring = "bridgekv"
    if args.bridge_graph_edges == "random_graph":
        args.bridge_graph_edges = "random_graph_control"
        if args.control_type == "none":
            args.control_type = "random_graph"
    if args.bridge_score == "random" and args.control_type == "none":
        args.control_type = "random_boundary_replacement"
    if args.bridge_score == "shuffled_bridge_scores_control" and args.control_type == "none":
        args.control_type = "shuffled_bridge_scores"
    if args.method == "fd_score":
        args.method = "sliminfer_variant"
    if args.method == "kv_separated":
        args.method = "sliminfer_variant"
        if args.kv_mode == "none":
            args.kv_mode = "guardrail"
    if args.schedule_mode == "fdp":
        args.method = "fdp_schedule"
    elif args.schedule_mode == "diffusion":
        args.method = "diffusion_ready"
    if args.backend == "auto":
        # flash_attn is not installed in the llamaqkv env, so auto must resolve to sdpa
        # rather than paying a load failure + retry on every worker.
        args.backend = "flash_attention_2" if flash_attention_available() else "sdpa"
    elif args.backend == "flash_attention_2" and not flash_attention_available():
        args.backend = "sdpa"
    if args.save_block_scores is True:
        args.save_block_scores = "aggregate_only"
    if args.save_block_scores is False:
        args.save_block_scores = "none"
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_min_free_gpu_gb(min_free_gpu_gb: float, *, dataset: str, sample_index: int) -> None:
    if float(min_free_gpu_gb or 0.0) <= 0.0:
        return
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except Exception as exc:
        raise RuntimeError(
            f"nvidia-smi preflight failed before {dataset}:{sample_index}: {exc}"
        ) from exc
    free_mib = [float(line.strip()) for line in out if line.strip()]
    if not free_mib:
        raise RuntimeError(f"nvidia-smi returned no GPU memory rows before {dataset}:{sample_index}")
    free_gb = max(free_mib) / 1024.0
    if free_gb < float(min_free_gpu_gb):
        raise RuntimeError(
            f"free GPU memory {free_gb:.2f} GiB < required {float(min_free_gpu_gb):.2f} GiB "
            f"before {dataset}:{sample_index}"
        )


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_structured_rescue_cmd(
    spec_name: str,
    *,
    output_dir: str,
    longbench_dir: str,
    model_path: str = "/diskE/models/Qwen2.5-7B-Instruct",
    samples_per_task: Optional[int] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> str:
    spec = dict(STRUCTURED_RESCUE_METHOD_SPECS.get(spec_name, {}))
    if not spec:
        raise KeyError(f"Unknown structured rescue spec: {spec_name}")
    if extra_args:
        spec.update(extra_args)
    method = spec.pop("method", "sliminfer_original")
    parts = [
        f"scripts/run_with_silm.sh experiments/run_qwen25_sliminfer_variants.py",
        f"--method {method}",
        f"--eval longbench",
        f"--model_path {model_path}",
        f"--longbench_dir {longbench_dir}",
        f"--output_dir {output_dir}/{spec_name}",
    ]
    for key, value in sorted(spec.items()):
        flag = f"--{key}"
        if isinstance(value, bool):
            parts.append(f"{flag} {1 if value else 0}")
        else:
            parts.append(f"{flag} {value}")
    parts.append(f"--rescue_audit_path {output_dir}/{spec_name}/rescue_audit.jsonl")
    if samples_per_task:
        parts.append(f"--longbench_max_samples {samples_per_task}")
    return " ".join(parts)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def register_bridgekv_state(state: Dict[str, Any]) -> str:
    state_id = str(uuid.uuid4())
    import bridgekv as bridgekv_module
    if not hasattr(bridgekv_module, "_state_cache"):
        bridgekv_module._state_cache = {}
    bridgekv_module._state_cache[state_id] = state
    return state_id


def get_bridgekv_state(state_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not state_id:
        return None
    import bridgekv as bridgekv_module
    return getattr(bridgekv_module, "_state_cache", {}).get(state_id)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def initialize_bridgekv_state(args: argparse.Namespace, tokenizer: AutoTokenizer, input_ids: torch.Tensor, sliminfer_config: Dict[str, Any]) -> None:
    if args.method != "bridgekv" or sliminfer_config is None:
        return
    block_texts = []
    seq_len = input_ids.shape[1]
    for start in range(0, seq_len, args.block_size):
        block_texts.append(
            tokenizer.decode(
                input_ids[0, start:min(start + args.block_size, seq_len)].tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    query_text = block_texts[-1] if block_texts else ""
    graph_state = build_bridge_graph(
        block_texts=block_texts,
        query_text=query_text,
        bridge_graph_edges=args.bridge_graph_edges,
        bridge_pos_edge_weight=args.bridge_pos_edge_weight,
        bridge_topk_sem_edges=args.bridge_topk_sem_edges,
        qk_scores=None,
        bridge_query_edge_weight=args.bridge_query_edge_weight,
        bridge_edge_weighting=args.bridge_edge_weighting,
        bridge_hub_penalty=args.bridge_hub_penalty,
        bridge_semantic_threshold=args.bridge_semantic_threshold,
    )
    graph_state["structural_scores"] = compute_structural_scores(block_texts, query_text) if args.bridge_structural_risk else [0.0 for _ in block_texts]
    graph_state["bridge_alpha"] = args.bridge_alpha
    graph_state["bridge_beta"] = args.bridge_beta
    graph_state["bridge_gamma"] = args.bridge_gamma
    graph_state["bridge_eta"] = args.bridge_eta
    graph_state["bridge_structural_risk"] = bool(args.bridge_structural_risk)
    graph_state["control_type"] = args.control_type
    graph_state["seed"] = args.seed
    state_id = register_state(graph_state)
    sliminfer_config["bridgekv_state_id"] = state_id


def get_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception as exc:
        info["transformers_error"] = repr(exc)
    try:
        import flash_attn
        info["flash_attn"] = getattr(flash_attn, "__version__", "unknown")
    except Exception as exc:
        info["flash_attn"] = f"unavailable: {exc}"
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        info.update({
            "gpu_name": props.name,
            "gpu_total_memory_GB": round(props.total_memory / 1e9, 3),
            "cuda_device_count": torch.cuda.device_count(),
        })
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip().splitlines()
            if smi:
                info["nvidia_driver"] = smi[0].strip()
        except Exception as exc:
            info["nvidia_driver_error"] = repr(exc)
    try:
        nvcc = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL)
        info["nvcc"] = nvcc.strip().splitlines()[-1]
    except Exception:
        info["nvcc"] = "unavailable"
    return info


def align_retain_tokens(retain_tokens: List[int], block_size: int) -> Tuple[List[int], List[str]]:
    notes = []
    aligned = []
    for tokens in retain_tokens:
        new_value = int(math.ceil(tokens / block_size) * block_size)
        if new_value != tokens:
            notes.append(f"retain_tokens {tokens} rounded to {new_value} for block_size={block_size}")
        aligned.append(new_value)
    return aligned, notes


def build_fixed_token_config(args: argparse.Namespace, prune_layers: List[int], retain_tokens: List[int]) -> Dict[str, Any]:
    pruning_config: Dict[int, Dict[str, Any]] = {}
    for layer, tokens in zip(prune_layers, retain_tokens):
        pruning_config[int(layer)] = {
            "fixed_keep_tokens": float(tokens) / 1024.0,
            "fixed_keep_tokens_raw": int(tokens),
            "always_keep_sink_recent_token": bool(args.preserve_first_block),
            "always_keep_recent_block": True,
        }
    return {
        "block_size": args.block_size,
        "semantic_kernel_size": args.token_unit_size,
        "swap_threshold": args.swap_threshold,
        "pruning_window_len": args.local_query_window,
        "pruning_method": "fixed_tokens",
        "pruning_config": pruning_config,
        "enable_cpu_offload": bool(args.enable_cpu_offload),
        "enable_async_kv": bool(args.enable_async_kv),
    }


def default_fd_weights(scoring: str, args: argparse.Namespace) -> Tuple[float, float, float]:
    defaults = {
        "delta_o": (1.0, 0.0, 0.0),
        "delta_oh": (1.0, 0.1, 0.0),
        "delta_ov": (1.0, 0.0, 0.0),
        "delta_qk_o_hybrid": (1.0, 0.0, 0.0),
        "entropy_guarded_qk": (1.0, 0.1, 0.0),
        "low_cost_delta_o": (1.0, 0.0, 0.0),
        "delta_soh": (1.0, 0.1, 0.1),
    }.get(scoring, (1.0, 0.0, 0.0))
    return (
        defaults[0] if args.fd_alpha_o is None else args.fd_alpha_o,
        defaults[1] if args.fd_beta_h is None else args.fd_beta_h,
        defaults[2] if args.fd_gamma_s is None else args.fd_gamma_s,
    )


def heuristic_fragility(num_layers: int, mode: str) -> List[float]:
    values = []
    denom = max(1, num_layers - 1)
    for layer in range(num_layers):
        x = layer / denom
        early = math.exp(-((x - 0.18) ** 2) / 0.020)
        late = 0.65 * math.exp(-((x - 0.82) ** 2) / 0.035)
        middle_stability = 0.25 * math.cos(math.pi * min(1.0, x)) + 0.25
        value = 0.15 + early + late + middle_stability
        if mode == "aggressive":
            value *= 0.92
        values.append(value)
    mn, mx = min(values), max(values)
    return [(v - mn) / max(mx - mn, 1e-9) for v in values]


def generate_fdp_schedule(num_layers: int, args: argparse.Namespace, out_dir: Path) -> Tuple[List[int], List[int], Dict[str, Any]]:
    fragility = heuristic_fragility(num_layers, args.fdp_mode)
    candidates = list(range(4, max(4, num_layers - 1)))
    stable_candidates = sorted(candidates, key=lambda idx: (fragility[idx], idx))
    selected = sorted(stable_candidates[: max(1, args.max_prune_points)])

    if args.fdp_mode == "aggressive" or args.target_budget == "lower_than_original":
        min_retain, max_retain = 3072, 10240
    elif args.fdp_mode == "matched" or args.target_budget == "matched_original":
        min_retain, max_retain = 4096, 12288
    else:
        min_retain, max_retain = 6144, 14336

    retain = []
    for layer in selected:
        tokens = min_retain + (max_retain - min_retain) * fragility[layer]
        retain.append(int(math.ceil(tokens / args.block_size) * args.block_size))

    profile = {
        "profile_type": "fdp_schedule_proxy",
        "note": "Lightweight layer fragility proxy; avoids per-block forward reruns during experiment orchestration.",
        "calib_samples_requested": args.calib_samples,
        "calib_seqlen_requested": args.calib_seqlen,
        "num_layers": num_layers,
        "layer_fragility": {str(i): fragility[i] for i in range(num_layers)},
        "selected_prune_layers": selected,
        "selected_retain_tokens": retain,
    }
    write_json(out_dir.parent / "fdp_profile_qwen25.json", profile)
    return selected, retain, profile


def generate_diffusion_schedule(num_layers: int, args: argparse.Namespace, out_dir: Path) -> Tuple[List[int], List[int], Dict[str, Any]]:
    readiness: Dict[int, float] = {}
    for layer in range(num_layers):
        x = layer / max(1, num_layers - 1)
        readiness[layer] = 1.0 / (1.0 + math.exp(-10.0 * (x - 0.28)))

    first = next((layer for layer, value in readiness.items() if value >= args.diffusion_tau), 9)
    prune_layers = [layer for layer in [first, first + 8, first + 16] if layer < num_layers]
    retain = parse_int_list(args.retain_tokens)[: len(prune_layers)]
    if not retain:
        retain = [12288, 6144, 4096][: len(prune_layers)]
    profile = {
        "profile_type": "diffusion_readiness_proxy",
        "note": "Readiness is a monotonic proxy over layer depth; selected layers are materialized in config for reproducible runs.",
        "tau": args.diffusion_tau,
        "readiness": {str(k): v for k, v in readiness.items()},
        "selected_prune_layers": prune_layers,
        "selected_retain_tokens": retain,
    }
    write_json(out_dir / "diffusion_profile.json", profile)
    return prune_layers, retain, profile


def rank_code_available() -> bool:
    patterns = ("EigenAttention", "FDP-EB", "eigen_basis", "low_rank", "low-rank", "rank_schedule")
    for path in REPO_ROOT.rglob("*"):
        if "experiments" in path.parts or "scripts" in path.parts:
            continue
        if path.is_file() and path.suffix in {".py", ".cu", ".cpp", ".h", ".md", ".yaml", ".json"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if any(pattern in text for pattern in patterns):
                return True
    return False


def _load_mcu_frozen_selections(path: str) -> Dict[str, Dict[str, Any]]:
    frozen_path = Path(path)
    if not frozen_path.exists():
        raise RuntimeError(f"mcu frozen selection file missing: {frozen_path}")
    if frozen_path.suffix == ".csv":
        from tools.mcu_sliminfer_selection import load_frozen_selections_from_audit_csv

        return load_frozen_selections_from_audit_csv(frozen_path)
    from tools.mcu_sliminfer_selection import frozen_row_to_cached_result

    selections: Dict[str, Dict[str, Any]] = {}
    for line in frozen_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = str(row.get("task", ""))
        sample_index = row.get("sample_index")
        if not task or sample_index is None:
            key = str(row.get("sample_key", ""))
            if ":" not in key:
                continue
            task, sample_index = key.split(":", 1)
        sample_key = f"{task}:{int(sample_index)}"
        cached = frozen_row_to_cached_result(row)
        cached["task"] = task
        cached["sample_index"] = int(sample_index)
        selections[sample_key] = cached
    return selections


def build_sliminfer_config(args: argparse.Namespace, out_dir: Path, num_layers: int) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    notes: List[str] = []
    prune_layers = parse_int_list(args.prune_layers)
    retain_tokens = parse_int_list(args.retain_tokens)

    profile: Dict[str, Any] = {}
    if args.method == "fdp_schedule":
        prune_layers, retain_tokens, profile = generate_fdp_schedule(num_layers, args, out_dir)
    elif args.method == "diffusion_ready":
        prune_layers, retain_tokens, profile = generate_diffusion_schedule(num_layers, args, out_dir)

    retain_tokens, align_notes = align_retain_tokens(retain_tokens, args.block_size)
    notes.extend(align_notes)
    cfg = build_fixed_token_config(args, prune_layers, retain_tokens)
    cfg["num_layers"] = num_layers
    cfg["variant_method"] = args.method
    cfg["score_kind"] = "token_unit_qk"
    cfg["scoring"] = args.scoring
    cfg["fairness_base_scoring"] = args.fairness_base_scoring
    cfg["decode_policy"] = args.decode_policy
    cfg["refresh_interval"] = args.refresh_interval
    cfg["active_set_hysteresis"] = args.active_set_hysteresis
    cfg["active_set_momentum"] = args.active_set_momentum
    cfg["stats_mode"] = "aggregate"
    cfg["block_scores_mode"] = "aggregate" if args.save_block_scores == "aggregate_only" else args.save_block_scores
    cfg["runtime_stats"] = {
        "score_overhead_ms": 0.0,
        "score_calls": 0,
        "prefill_score_calls": 0,
        "decode_score_calls": 0,
        "decode_reuse_calls": 0,
        "swap_overlap_sum": 0.0,
        "swap_overlap_count": 0,
        "jaccard_vs_original_sum": 0.0,
        "jaccard_vs_original_count": 0,
    }
    cfg["rescuekv_variant"] = {
        "rescue_policy": args.rescue_policy,
        "rescue_score": args.rescue_score,
        "rescue_ratio": args.rescue_ratio,
        "swap_drop_score": args.swap_drop_score,
        "alpha_diff": args.alpha_diff,
        "lambda_red": args.lambda_red,
        "eta_rescue": args.eta_rescue,
        "eta_active": args.eta_active,
        "beta_path": args.beta_path,
        "swap_tau": args.swap_tau,
        "note": "Optional RescueKV Add/DRM smoke parameters wired into SlimInfer cache path.",
    }
    cfg["rescue_policy"] = args.rescue_policy
    cfg["rescue_score"] = args.rescue_score
    cfg["rescue_ratio"] = args.rescue_ratio
    cfg["lssr_policy_b_plan_path"] = args.lssr_policy_b_plan_path
    cfg["dc_alpha"] = args.dc_alpha
    cfg["dc_beta"] = args.dc_beta
    cfg["dc_lambda"] = args.dc_lambda
    cfg["dc_gamma"] = args.dc_gamma
    cfg["dc_selection_audit_path"] = args.dc_selection_audit_path
    cfg["dc_activation_audit_path"] = args.dc_activation_audit_path
    cfg["cc_selection_audit_path"] = args.cc_selection_audit_path
    cfg["active_block_trace_level"] = args.active_block_trace
    cfg["active_block_trace_dir"] = args.active_block_trace_dir
    cfg["protocol_id"] = args.protocol_id
    cfg["implementation_revision"] = args.implementation_revision
    cfg["model_id"] = args.model_path
    cfg["selection_digest_dir"] = args.selection_digest_dir
    cfg["drm_trace_dir"] = args.drm_trace_dir
    cfg["cc_gate_config_path"] = getattr(args, "cc_gate_config_path", "") or ""
    cfg["qdc_selection_audit_path"] = args.qdc_selection_audit_path
    cfg["gate_mode"] = args.gate_mode
    cfg["gate_policy"] = args.gate_policy
    cfg["gate_default_policy"] = args.gate_default_policy
    cfg["gate_rescue_margin_min"] = args.gate_rescue_margin_min
    cfg["gate_coverage_gap_min"] = args.gate_coverage_gap_min
    cfg["gate_deletion_risk_max"] = args.gate_deletion_risk_max
    cfg["gate_feature_version"] = args.gate_feature_version
    cfg["gate_sentence_jaccard_novel_max"] = (
        args.gate_sentence_jaccard_novel_max
    )
    cfg["gate_apply_layer"] = max(prune_layers) if prune_layers else -1
    cfg["block_strategy"] = args.block_strategy
    cfg["selection_path"] = args.selection_path
    cfg["target_block_tokens"] = args.target_block_tokens
    cfg["min_block_tokens"] = args.min_block_tokens
    cfg["max_block_tokens"] = args.max_block_tokens
    cfg["block_overlap_tokens"] = args.block_overlap_tokens
    cfg["semantic_parser"] = args.semantic_parser
    cfg["semantic_target_block_tokens"] = args.semantic_target_block_tokens
    cfg["semantic_min_block_tokens"] = args.semantic_min_block_tokens
    cfg["semantic_max_block_tokens"] = args.semantic_max_block_tokens
    cfg["semantic_page_align"] = args.semantic_page_align
    cfg["max_page_expansion_overhead"] = args.max_page_expansion_overhead
    cfg["attention_boundary_layer"] = args.attention_boundary_layer
    cfg["boundary_percentile"] = args.boundary_percentile
    cfg["query_top_ratio"] = args.query_top_ratio
    cfg["query_neighbor_sentences"] = args.query_neighbor_sentences
    cfg["query_high_max_block_tokens"] = args.query_high_max_block_tokens
    cfg["query_low_max_block_tokens"] = args.query_low_max_block_tokens
    cfg["query_score_mode"] = args.query_score_mode
    cfg["query_parser"] = args.query_parser
    cfg["coarse_block_tokens"] = args.coarse_block_tokens
    cfg["fine_block_tokens"] = args.fine_block_tokens
    cfg["medium_block_tokens"] = args.medium_block_tokens
    cfg["top_coarse_ratio"] = args.top_coarse_ratio
    cfg["budget_mode"] = args.budget_mode
    cfg["target_token_budget_source"] = args.target_token_budget_source
    cfg["target_token_budget"] = args.target_token_budget
    cfg["token_budget_tolerance"] = args.token_budget_tolerance
    cfg["selection_cost"] = args.selection_cost
    cfg["score_density_mode"] = args.score_density_mode
    cfg["allow_budget_overshoot"] = args.allow_budget_overshoot
    cfg["soft_budget_tolerance"] = args.soft_budget_tolerance
    cfg["hard_budget_tolerance"] = args.hard_budget_tolerance
    cfg["active_budget_mode"] = args.active_budget_mode
    cfg["rescue_budget_mode"] = args.rescue_budget_mode
    cfg["target_active_tokens"] = args.target_active_tokens
    cfg["target_final_tokens"] = args.target_final_tokens
    cfg["target_final_tokens_source"] = args.target_final_tokens_source
    cfg["active_block_strategy"] = args.active_block_strategy
    cfg["rescue_candidate_strategy"] = args.rescue_candidate_strategy
    cfg["hybrid_exclude_active_overlap"] = args.hybrid_exclude_active_overlap
    cfg["hybrid_overlap_threshold"] = args.hybrid_overlap_threshold
    cfg["gamma_boundary"] = args.gamma_boundary
    cfg["boundary_weight_active_score"] = args.boundary_weight_active_score
    cfg["gamma_active_boundary"] = args.gamma_active_boundary
    cfg["calibrated_budget_config_path"] = args.calibrated_budget_config
    if args.calibrated_budget_config:
        try:
            cfg["calibrated_budget_config"] = load_json(Path(args.calibrated_budget_config))
        except Exception as exc:
            notes.append(f"Could not load calibrated budget config {args.calibrated_budget_config}: {exc}")
    cfg["_block_layout_written"] = []
    cfg["seed"] = args.seed
    cfg["swap_drop_score"] = args.swap_drop_score
    cfg["alpha_diff"] = args.alpha_diff
    cfg["lambda_red"] = args.lambda_red
    cfg["eta_rescue"] = args.eta_rescue
    cfg["eta_active"] = args.eta_active
    cfg["beta_path"] = args.beta_path
    cfg["beta_query"] = args.beta_query
    cfg["beta_semantic"] = args.beta_semantic
    cfg["lambda_len"] = args.lambda_len
    cfg["rescue_overlap_window_tokens"] = args.rescue_overlap_window_tokens
    cfg["rescue_overlap_stride_tokens"] = args.rescue_overlap_stride_tokens
    cfg["rescue_evidence_half_window_tokens"] = args.rescue_evidence_half_window_tokens
    cfg["rescue_evidence_top_ratio"] = args.rescue_evidence_top_ratio
    cfg["rescue_semantic_page_tolerance"] = args.rescue_semantic_page_tolerance
    cfg["rescue_selector"] = args.rescue_selector
    cfg["rescue_gate_mode"] = args.rescue_gate_mode
    cfg["rescue_gate_threshold"] = args.rescue_gate_threshold
    cfg["coverage_gamma"] = args.coverage_gamma
    cfg["noise_mu"] = args.noise_mu
    cfg["block_budget_mode"] = args.block_budget_mode
    cfg["block_budget_block_tokens"] = args.block_budget_block_tokens
    cfg["block_budget_log_weight"] = args.block_budget_log_weight
    cfg["verifier_mode"] = args.verifier_mode
    cfg["verifier_topn_mult"] = args.verifier_topn_mult
    cfg["stochastic_mode"] = args.stochastic_mode
    cfg["gumbel_tau"] = args.gumbel_tau
    cfg["candidate_pool_ratio"] = args.candidate_pool_ratio
    cfg["sampling_temperature"] = args.sampling_temperature
    cfg["swap_tau"] = args.swap_tau
    cfg["rescue_granularity"] = args.rescue_granularity
    cfg["microblock_size"] = args.microblock_size
    cfg["microblock_parent_aggregation"] = args.microblock_parent_aggregation
    cfg["rh_head_file"] = args.rh_head_file
    cfg["auto_retrieval_heads_top_frac"] = args.auto_retrieval_heads_top_frac
    cfg["rh_weight"] = args.rh_weight
    cfg["pyramid_weight"] = args.pyramid_weight
    cfg["low_layer_ratio"] = args.low_layer_ratio
    cfg["mid_layer_ratio"] = args.mid_layer_ratio
    cfg["high_layer_ratio"] = args.high_layer_ratio
    cfg["teacher_label_file"] = args.teacher_label_file
    cfg["distilled_selector_file"] = args.distilled_selector_file
    cfg["teacher_align_weight"] = args.teacher_align_weight
    cfg["bridge_weight"] = args.bridge_weight
    cfg["bridge_top_candidates"] = args.bridge_top_candidates
    cfg["bridge_min_link"] = args.bridge_min_link
    cfg["deferred_mode"] = args.deferred_mode
    cfg["deferred_trigger_entropy"] = args.deferred_trigger_entropy
    cfg["deferred_trigger_margin"] = args.deferred_trigger_margin
    cfg["deferred_topup_m"] = args.deferred_topup_m
    cfg["deferred_max_triggers"] = args.deferred_max_triggers
    cfg["density_alpha_max"] = args.density_alpha_max
    cfg["density_alpha_count"] = args.density_alpha_count
    cfg["density_alpha_continuity"] = args.density_alpha_continuity
    cfg["density_alpha_coverage"] = args.density_alpha_coverage
    cfg["density_lambda_red"] = args.density_lambda_red
    cfg["density_threshold_mode"] = args.density_threshold_mode
    cfg["density_top_quantile"] = args.density_top_quantile
    cfg["strict_structured_path_check"] = int(str(args.strict_structured_path_check).lower() == "true")
    if cfg["strict_structured_path_check"]:
        args.dump_candidate_scores = max(int(args.dump_candidate_scores), 1)
        args.dump_final_selection = max(int(args.dump_final_selection), 1)
    cfg["dump_candidate_scores"] = int(args.dump_candidate_scores)
    cfg["dump_final_selection"] = int(args.dump_final_selection)
    audit_parent = Path(cfg.get("rescue_audit_path", out_dir / "rescue_audit.jsonl")).parent
    if args.dump_candidate_scores or args.dump_final_selection or cfg["strict_structured_path_check"]:
        cfg["candidate_score_dump_path"] = str(audit_parent / "candidate_score_dump.jsonl")
        cfg["final_selection_dump_path"] = str(audit_parent / "final_selection_dump.jsonl")
        cfg["path_marker_dump_path"] = str(audit_parent / "path_marker.jsonl")
    if args.feature_dump_path:
        cfg["feature_dump_path"] = args.feature_dump_path
    if args.drm_trace_dir:
        drm_trace_dir = Path(args.drm_trace_dir)
        drm_trace_dir.mkdir(parents=True, exist_ok=True)
        cfg["drm_trace_dir"] = str(drm_trace_dir)
    if args.rh_head_file:
        try:
            cfg["rh_head_profile"] = load_json(Path(args.rh_head_file))
        except Exception as exc:
            notes.append(f"Could not load rh_head_file {args.rh_head_file}: {exc}")
    if args.rescue_audit_path:
        audit_path = Path(args.rescue_audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("", encoding="utf-8")
        cfg["rescue_audit_path"] = str(audit_path)
    elif args.rescue_policy not in {"none", ""}:
        audit_path = Path(args.rescue_audit_path) if args.rescue_audit_path else out_dir / "rescue_policy_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("", encoding="utf-8")
        cfg["rescue_audit_path"] = str(audit_path)
    if args.block_layout_audit_path:
        layout_path = Path(args.block_layout_audit_path)
        layout_path.parent.mkdir(parents=True, exist_ok=True)
        layout_path.write_text("", encoding="utf-8")
        cfg["block_layout_audit_path"] = str(layout_path)
    if args.uasr_score_dump_path:
        uasr_path = Path(args.uasr_score_dump_path)
        uasr_path.parent.mkdir(parents=True, exist_ok=True)
        uasr_path.write_text("", encoding="utf-8")
        cfg["uasr_score_dump_path"] = str(uasr_path)
        cfg["uasr_tail_width"] = int(args.uasr_tail_width)
    if args.sp_trajectory_dump_path:
        traj_path = Path(args.sp_trajectory_dump_path)
        traj_path.mkdir(parents=True, exist_ok=True)
        cfg["sp_trajectory_dump_path"] = str(traj_path)

    if args.method == "sliminfer_original":
        cfg["variant_method"] = "sliminfer_original"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        notes.append(f"SlimInfer original uses decode_policy={args.decode_policy}; original means dynamic score refresh with overlap-aware KV swapping.")
    elif args.method == "confidence_gated_sliminfer":
        cfg["variant_method"] = "confidence_gated_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["cgs_confidence_threshold"] = float(args.cgs_confidence_threshold)
        cfg["cgs_min_swap_threshold"] = float(args.cgs_min_swap_threshold)
        notes.append(
            "Confidence-Gated SlimInfer keeps token_unit_qk scoring; when block-score confidence is low, "
            "active-set updates are stabilized against the previous prune layer (no gold / no task routing)."
        )
    elif args.method == "margin_stable_sliminfer":
        cfg["variant_method"] = "margin_stable_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["mss_swap_margin_ratio"] = float(args.mss_swap_margin_ratio)
        notes.append("Margin-Stable SlimInfer: swap active blocks only when score margin is significant.")
    elif args.method == "sp_sliminfer":
        cfg["variant_method"] = "sp_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["abs_prune_layers"] = prune_layers
        notes.append(
            "SP-SlimInfer: semantic propagation score adjustment on token_unit_qk; "
            "same KV budget as sliminfer_original (no gold, no RoleFlow)."
        )
    elif args.method == "sp_sliminfer_v2":
        cfg["variant_method"] = "sp_sliminfer_v2"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["abs_prune_layers"] = prune_layers
        cfg["sp_v2_policy"] = "transport_residual"
        cfg["sp_v2_tau"] = 0.10
        cfg["sp_v2_top_r"] = 5
        cfg["sp_v2_replacement_cap"] = 0.05
        cfg["local_query_window"] = int(getattr(args, "local_query_window", 4) or 4)
        notes.append(
            "SP-SlimInfer v2: cross-layer semantic transport (transport_residual); "
            "L9 original SlimInfer; L18/L26 bounded 5% replacement; frozen hyperparams."
        )
    elif args.method in ("cc_sliminfer", "cc_sliminfer_v1"):
        # Frozen alias: cc_sliminfer_v1 == cc_sliminfer (no retuning).
        cfg["variant_method"] = "cc_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["abs_prune_layers"] = prune_layers
        cfg["cc_alpha"] = 0.25
        cfg["cc_min_cluster_len"] = 2
        cfg["cc_max_neighbors"] = 2
        cfg["cc_replacement_cap"] = 0.05
        cfg["cc_version"] = "cc_sliminfer_v1"
        notes.append(
            "cc_sliminfer_v1 (frozen Cluster-Continuity): alpha=0.25, min_cluster_len=2, "
            "max_neighbors=2, replacement_cap=0.05; preserve evidence clusters within budget."
        )
    elif args.method in (
        "cc_gated_v0",
        "cc_spike_hybrid_v0",
        "cc_coverage_v0",
        "cc_diversify_v0",
        "cc_router_v0",
    ):
        # Topology wrappers around frozen CC operator (no retune of alpha/neighbors/cap).
        cfg["variant_method"] = args.method
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["abs_prune_layers"] = prune_layers
        cfg["cc_alpha"] = 0.25
        cfg["cc_min_cluster_len"] = 2
        cfg["cc_max_neighbors"] = 2
        cfg["cc_replacement_cap"] = 0.05
        cfg["cc_version"] = args.method
        cfg["cc_gate_config_path"] = getattr(args, "cc_gate_config_path", "") or str(
            REPO_ROOT / "configs" / "cc_gate_v0.json"
        )
        notes.append(
            f"{args.method}: topology policy around frozen CC operator; "
            f"gate_config={cfg['cc_gate_config_path']}"
        )
    elif args.method in (
        "acc_sliminfer",
        "acc_sliminfer_v1",
        "drm_swap_sliminfer",
        "drm_acc_swap_sliminfer",
        "drm_acc_swap_sliminfer_v1",
        "drm_acc_add_sliminfer",
    ):
        cfg["variant_method"] = args.method
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["abs_prune_layers"] = prune_layers
        cfg["cc_alpha"] = 0.25
        cfg["cc_min_cluster_len"] = 2
        cfg["cc_max_neighbors"] = 2
        cfg["cc_replacement_cap"] = 0.05
        cfg["rescue_ratio"] = float(getattr(args, "rescue_ratio", 0.05) or 0.05)
        cfg["cc_version"] = args.method
        cfg["acc_drm_version"] = "acc_drm_v1"
        # Frozen Phase-1B spike joint thresholds (global; not task-specific).
        cfg["tau_spike"] = 0.28
        cfg["tau_margin"] = 0.20
        cfg["tau_spike_nei_max"] = 0.18
        # Keep optional rescue none — ACC/DRM methods self-contain selection.
        if args.method != "drm_acc_add_sliminfer":
            notes.append(
                f"{args.method}: K-budget ACC/DRM-ACC v1 path "
                f"(spike joint tau_spike=0.28/tau_margin=0.20/tau_spike_nei_max=0.18)."
            )
        else:
            notes.append(
                "drm_acc_add_sliminfer: K+M TRADEOFF only — not fair vs K-budget methods."
            )
    elif args.method == "ud_sliminfer":
        cfg["variant_method"] = "ud_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["abs_prune_layers"] = prune_layers
        cfg["ud_w_max"] = 0.50
        cfg["ud_w_density"] = 0.35
        cfg["ud_w_spike"] = 0.15
        notes.append("Unit-Density SlimInfer: penalize single-unit spike blocks; reward internal unit agreement.")
    elif args.method == "divcov_sliminfer":
        cfg["variant_method"] = "divcov_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["abs_prune_layers"] = prune_layers
        cfg["divcov_mmr_lambda"] = 0.15
        notes.append("Diversity-Coverage SlimInfer: MMR selection on SlimInfer relevance with key-rep redundancy penalty.")
    elif args.method == "rank_stability_sliminfer":
        cfg["variant_method"] = "rank_stability_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["abs_prune_layers"] = prune_layers
        cfg["rss_weight"] = float(getattr(args, "rss_weight", 0.12))
        notes.append(
            "Rank-Stability SlimInfer: boost blocks with stable SILM ranks across prune layers."
        )
    elif args.method == "multi_window_sliminfer":
        cfg["variant_method"] = "multi_window_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["mw_windows"] = parse_int_list(args.mw_windows) or [1, 4, 8]
        notes.append("Multi-Window SlimInfer: consensus block scores from local query windows w=1,4,8.")
    elif args.method == "adaptive_budget_sliminfer":
        cfg["variant_method"] = "adaptive_budget_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["abs_prune_layers"] = prune_layers
        cfg["abs_easy_schedule"] = (12, 6, 4)
        cfg["abs_hard_schedule"] = (12, 8, 6)
        notes.append(
            "Adaptive-Budget SlimInfer: sample difficulty (gold-free) selects easy 12/6/4k vs hard 12/8/6k retention."
        )
    elif args.method == "reliability_gated_sliminfer":
        cfg["variant_method"] = "reliability_gated_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["abs_prune_layers"] = prune_layers
        cfg["rgs_confidence_low"] = float(args.rgs_confidence_low)
        cfg["rgs_confidence_high"] = float(args.rgs_confidence_high)
        cfg["cgs_confidence_threshold"] = float(args.cgs_confidence_threshold)
        cfg["cgs_min_swap_threshold"] = float(args.cgs_min_swap_threshold)
        notes.append(
            "Reliability-Gated SlimInfer: sample-level policy routes to original vs conservative pruning."
        )
    elif args.method == "dc_sliminfer":
        cfg["variant_method"] = "dc_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["reuse_decode_blocks"] = args.decode_policy == "reuse_prefill"
        cfg["stats_mode"] = "aggregate"
        cfg["block_scores_mode"] = "aggregate" if args.save_block_scores == "aggregate_only" else args.save_block_scores
        notes.append(
            "DC-SlimInfer keeps SlimInfer token-unit saliency as the base score and replaces per-layer top-K with fixed-parameter diversity-calibrated iterative selection."
        )
    elif args.method == "adaptive_dc_sliminfer_v3":
        cfg["variant_method"] = "adaptive_dc_sliminfer_v3"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["reuse_decode_blocks"] = args.decode_policy == "reuse_prefill"
        cfg["stats_mode"] = "aggregate"
        cfg["block_scores_mode"] = "aggregate" if args.save_block_scores == "aggregate_only" else args.save_block_scores
        adaptive_gate_json = getattr(args, "adaptive_dc_v3_gate_json", "")
        if adaptive_gate_json:
            cfg["adaptive_dc_v3_gate"] = json.loads(Path(adaptive_gate_json).read_text(encoding="utf-8")) if Path(adaptive_gate_json).exists() else json.loads(adaptive_gate_json)
        notes.append(
            "Adaptive DC-SlimInfer v3 uses SlimInfer original as default selector; "
            "DC repair is activated only when the runtime dispersed-evidence gate triggers."
        )
    elif args.method == "qdc_sliminfer":
        cfg["variant_method"] = "qdc_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["reuse_decode_blocks"] = args.decode_policy == "reuse_prefill"
        cfg["stats_mode"] = "aggregate"
        cfg["block_scores_mode"] = "aggregate" if args.save_block_scores == "aggregate_only" else args.save_block_scores
        cfg["qdc_gate_apply_layer"] = max(prune_layers) if prune_layers else 26
        notes.append(
            "QDC-SlimInfer keeps SlimInfer original selection as the base and applies "
            "budget-neutral query-decomposed coverage repair at the final prune layer."
        )
    elif args.method == "mcu_sliminfer":
        cfg["variant_method"] = "mcu_sliminfer"
        cfg["score_kind"] = "token_unit_qk"
        cfg["decode_policy"] = args.decode_policy
        cfg["reuse_decode_blocks"] = args.decode_policy == "reuse_prefill"
        cfg["stats_mode"] = "aggregate"
        cfg["block_scores_mode"] = "aggregate" if args.save_block_scores == "aggregate_only" else args.save_block_scores
        cfg["mcu_gate_apply_layer"] = max(prune_layers) if prune_layers else 26
        cfg["mcu_selection_audit_path"] = args.mcu_selection_audit_path
        frozen_path = getattr(args, "mcu_frozen_selection_path", "")
        if frozen_path:
            cfg["mcu_frozen_selections"] = _load_mcu_frozen_selections(frozen_path)
        notes.append(
            "MCU-SlimInfer keeps SlimInfer original budget and applies budget-neutral "
            "counterfactual-utility block swaps from prefill probes at the final prune layer."
        )
    elif args.method == "sliminfer_variant":
        alpha, beta, gamma = default_fd_weights(args.scoring, args)
        cfg.update({
            "score_kind": args.scoring,
            "fd_alpha_o": alpha,
            "fd_beta_h": beta,
            "fd_gamma_s": gamma,
            "fd_beta_v": args.fd_beta_v,
            "fd_normalize": args.fd_normalize,
            "fd_score_clip": args.fd_score_clip,
            "fd_topk_mode": args.fd_topk_mode,
            "fd_compute_scope": args.fd_compute_scope,
            "fd_approx": args.fd_approx,
            "fd_use_entropy": bool(args.fd_use_entropy),
            "fd_use_score_proxy": bool(args.fd_use_score_proxy),
            "fd_use_value_norm": bool(args.fd_use_value_norm),
            "fd_use_renorm": bool(args.fd_use_renorm),
            "hybrid_qk_weight": args.hybrid_qk_weight,
            "hybrid_o_weight": args.hybrid_o_weight,
            "risk_ratio": args.risk_ratio,
            "kv_mode": args.kv_mode,
            "kv_topk_risk_ratio": args.kv_topk_risk_ratio,
            "lambda_k": args.lambda_k,
            "lambda_v": args.lambda_v,
            "head_group_mode": args.head_group_mode,
            "num_head_groups": args.num_head_groups,
            "head_agg": args.head_agg,
            "reuse_decode_blocks": args.decode_policy == "reuse_prefill",
            "stats_mode": "aggregate",
            "block_scores_mode": "aggregate" if args.save_block_scores == "aggregate_only" else args.save_block_scores,
        })
        if args.decode_policy == "reuse_prefill":
            notes.append("sliminfer_variant reuses prefill active blocks during decode to avoid repeated scoring and KV swap churn.")
    elif args.method == "bridgekv":
        cfg.update({
            "score_kind": "bridgekv",
            "selection_policy": args.selection_policy,
            "base_score": args.base_score,
            "quota_ratio": args.quota_ratio,
            "core_ratio": args.core_ratio,
            "boundary_ratio": args.boundary_ratio,
            "replace_ratio": args.replace_ratio,
            "rescue_ratio": args.rescue_ratio,
            "rescue_source": args.rescue_source,
            "agreement_top_p": args.agreement_top_p,
            "agreement_signals": args.agreement_signals,
            "evidence_ratio": args.evidence_ratio,
            "bridge_ratio": args.bridge_ratio,
            "control_type": args.control_type,
            "seed": args.seed,
            "bridge_alpha": args.bridge_alpha,
            "bridge_beta": args.bridge_beta,
            "bridge_gamma": args.bridge_gamma,
            "bridge_eta": args.bridge_eta,
            "bridge_graph_edges": args.bridge_graph_edges,
            "bridge_score": args.bridge_score,
            "bridge_ppr_alpha": args.bridge_ppr_alpha,
            "bridge_pseudo_evidence_ratio": args.bridge_pseudo_evidence_ratio,
            "bridge_anchor_mode": args.bridge_anchor_mode,
            "bridge_anchor_ratio": args.bridge_anchor_ratio,
            "bridge_pos_edge_weight": args.bridge_pos_edge_weight,
            "bridge_topk_sem_edges": args.bridge_topk_sem_edges,
            "bridge_query_edge_weight": args.bridge_query_edge_weight,
            "bridge_edge_weighting": args.bridge_edge_weighting,
            "bridge_hub_penalty": bool(args.bridge_hub_penalty),
            "bridge_hub_lambda": args.bridge_hub_lambda,
            "bridge_idf_node": bool(args.bridge_idf_node),
            "bridge_semantic_threshold": args.bridge_semantic_threshold,
            "num_evidence_anchors": args.num_evidence_anchors,
            "max_paths_per_anchor": args.max_paths_per_anchor,
            "bridge_norm": args.bridge_norm,
            "bridge_structural_risk": bool(args.bridge_structural_risk),
            "local_score": args.local_score,
            "reuse_decode_blocks": args.decode_policy == "reuse_prefill",
            "stats_mode": "aggregate",
            "block_scores_mode": "aggregate" if args.save_block_scores == "aggregate_only" else args.save_block_scores,
        })
        notes.append("BridgeKV uses reasoning-path-aware graph utility combined with local damage and structural signals.")
    elif args.method == "fdp_schedule":
        cfg.update({"score_kind": "delta_oh", "fd_alpha_o": 1.0, "fd_beta_h": 0.1, "fd_gamma_s": 0.0})
    elif args.method == "kv_separated":
        cfg.update({
            "score_kind": "delta_soh",
            "kv_mode": args.kv_mode,
            "kv_topk_risk_ratio": args.kv_topk_risk_ratio,
            "lambda_k": args.lambda_k,
            "lambda_v": args.lambda_v,
        })
    elif args.method == "diffusion_ready":
        cfg["score_kind"] = "token_unit"
    elif args.method == "residual_comp":
        cfg.update({
            "residual_type": args.residual_type,
            "residual_slots": args.residual_slots,
            "residual_scale": args.residual_scale,
            "residual_runtime_applied": False,
        })
        notes.append("residual_comp uses the SlimInfer token path with residual metadata recorded; compressed residual KV injection is not enabled in this repository path.")

    if args.save_layer_stats:
        layer_stats_path = out_dir / "layer_stats.jsonl"
        layer_stats_path.write_text("", encoding="utf-8")
        cfg["layer_stats_path"] = str(layer_stats_path)
    if args.save_block_scores != "none":
        block_scores_path = out_dir / "block_scores.jsonl"
        block_scores_path.write_text("", encoding="utf-8")
        cfg["block_scores_path"] = str(block_scores_path)
    if args.uasr_score_dump_path:
        cfg["uasr_score_dump_path"] = str(Path(args.uasr_score_dump_path))
        cfg["uasr_tail_width"] = int(args.uasr_tail_width)
    if args.sp_trajectory_dump_path:
        cfg["sp_trajectory_dump_path"] = str(Path(args.sp_trajectory_dump_path))

    derived = {
        "prune_layers": prune_layers,
        "retain_tokens": retain_tokens,
        "profile": profile,
    }
    return cfg, derived, notes


def load_model_and_tokenizer(args: argparse.Namespace, attn_impl: str = "sdpa"):
    tokenizer_path = args.tokenizer_path or args.model_path
    dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if attn_impl in ("flash_attention_2", "auto") and not flash_attention_available():
        attn_impl = "sdpa"

    family = getattr(args, "model_family", "llama")
    if args.method == "fullkv":
        # Full-KV is the unmodified-baseline arm: load stock transformers weights so no
        # DRM/SlimInfer hook can influence it.
        model_cls = AutoModelForCausalLM
        cache_classes = (None, None)
    elif family == "llama":
        from models.modeling_llama import LlamaForCausalLM
        from utils.cache_utils import HsCache, SlimInferCache
        model_cls = LlamaForCausalLM
        cache_classes = (SlimInferCache, HsCache)
    else:
        from models.modeling_qwen2 import Qwen2ForCausalLM
        from utils.cache_utils import HsCache, SlimInferCache
        model_cls = Qwen2ForCausalLM
        cache_classes = (SlimInferCache, HsCache)

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "attn_implementation": attn_impl,
    }
    try:
        model = model_cls.from_pretrained(args.model_path, **load_kwargs)
        actual_attn_impl = attn_impl
    except Exception as exc:
        if attn_impl != "sdpa":
            print(f"Warning: failed to load with {attn_impl}: {exc}. Retrying with sdpa.")
            load_kwargs["attn_implementation"] = "sdpa"
            model = model_cls.from_pretrained(args.model_path, **load_kwargs)
            actual_attn_impl = "sdpa"
        else:
            raise

    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer, cache_classes, actual_attn_impl


def make_benchmark_text(tokenizer: AutoTokenizer, target_tokens: int) -> str:
    source = "Long context benchmark sentence with varied factual detail, dates, names, and short clauses. "
    path = REPO_ROOT / "data" / "example_data.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data and "prompt" in data[0]:
                source = data[0]["prompt"] + "\n"
        except Exception:
            pass
    text = source
    while True:
        n_tokens = len(tokenizer(text, add_special_tokens=False).input_ids)
        if n_tokens >= target_tokens + 128:
            return text
        multiplier = max(2, math.ceil((target_tokens + 128) / max(1, n_tokens)))
        text = text * min(multiplier, 8)


def estimate_full_kv_bytes(model: torch.nn.Module, prompt_len: int, dtype: torch.dtype) -> int:
    cfg = model.config
    n_layers = int(getattr(cfg, "num_hidden_layers", 0))
    n_kv = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 0)))
    head_dim = int(getattr(cfg, "head_dim", getattr(cfg, "hidden_size", 0) // max(1, getattr(cfg, "num_attention_heads", 1))))
    element_size = torch.tensor([], dtype=dtype).element_size()
    return int(n_layers * 2 * prompt_len * n_kv * head_dim * element_size)


def cache_gpu_bytes(cache: Any) -> int:
    if cache is None:
        return 0
    total = 0
    for attr in ["prompt_key_cache", "prompt_value_cache", "decode_key_cache", "decode_value_cache"]:
        values = getattr(cache, attr, [])
        for tensor in values:
            if tensor is not None and tensor.device.type == "cuda":
                total += tensor.numel() * tensor.element_size()
    return int(total)


def reset_runtime_stats(sliminfer_config: Optional[Dict[str, Any]]) -> None:
    if not sliminfer_config:
        return
    if sliminfer_config.get("sp_trajectory_dump_path"):
        try:
            from bridgekv.sp_trajectory import reset_trajectory_sample_state

            reset_trajectory_sample_state(sliminfer_config)
        except Exception:
            pass
    if sliminfer_config.get("variant_method") == "sp_sliminfer_v2":
        try:
            from bridgekv.sp_semantic_transport import reset_sp_v2_runtime_state

            reset_sp_v2_runtime_state(sliminfer_config)
        except Exception:
            pass
    for key in (
        "_rss_layer_ranks",
        "_sp_traj_prev_selected",
        "_sp_layer_trace",
        "_cgs_prev_active_block_ids",
        "_mss_prev_active_block_ids",
        "_rgs_prev_active_block_ids",
    ):
        sliminfer_config.pop(key, None)
    stats = sliminfer_config.get("runtime_stats")
    if not isinstance(stats, dict):
        return
    for key in list(stats.keys()):
        stats[key] = 0.0 if isinstance(stats[key], float) else 0


def summarize_runtime_stats(sliminfer_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    stats = dict(sliminfer_config.get("runtime_stats", {})) if sliminfer_config else {}
    score_calls = int(stats.get("score_calls", 0) or 0)
    j_count = int(stats.get("jaccard_vs_original_count", 0) or 0)
    o_count = int(stats.get("swap_overlap_count", 0) or 0)
    if j_count > 0:
        stats["jaccard_vs_original"] = float(stats.get("jaccard_vs_original_sum", 0.0)) / j_count
    if o_count > 0:
        stats["active_block_overlap"] = float(stats.get("swap_overlap_sum", 0.0)) / o_count
        stats["active_set_churn"] = 1.0 - float(stats["active_block_overlap"])
    if score_calls > 0:
        stats["score_overhead_per_call_ms"] = float(stats.get("score_overhead_ms", 0.0)) / score_calls
    return stats


def generate_once(model, tokenizer, input_ids, attention_mask, max_new_tokens: int,
                  args: argparse.Namespace, cache_classes, sliminfer_config: Optional[Dict[str, Any]]) -> Tuple[float, Any]:
    kwargs: Dict[str, Any] = {}
    cache = None
    if args.method != "fullkv":
        SlimInferCache, HsCache = cache_classes
        cache = SlimInferCache(model.device)
        kwargs.update({
            "past_key_values": cache,
            "hs_cache": HsCache(),
            "sliminfer_config": sliminfer_config,
            "all_original_prompt_indices_abs": list(range(input_ids.shape[1])),
        })
        if args.method == "bridgekv":
            initialize_bridgekv_state(args, tokenizer, input_ids, sliminfer_config)

    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    do_sample = bool(args.temperature > 0.0 and (args.temperature != 1.0 or args.top_p != 1.0))
    generation_kwargs: Dict[str, Any] = {}
    if do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=1,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
            **kwargs,
        )
    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start, cache


def run_latency(model, tokenizer, cache_classes, args: argparse.Namespace,
                sliminfer_config: Optional[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    context_lengths = parse_int_list(args.context_lengths)
    dtype = dtype_from_name(args.dtype)
    max_context = max(context_lengths) if context_lengths else 32768
    benchmark_text = make_benchmark_text(tokenizer, max_context)

    warm = tokenizer("warmup " * 256, return_tensors="pt").to(model.device)
    try:
        reset_runtime_stats(sliminfer_config)
        _ = generate_once(model, tokenizer, warm.input_ids, warm.attention_mask, 1, args, cache_classes, sliminfer_config)
    except Exception as exc:
        print(f"Warning: latency warmup failed: {exc}")

    for target_len in context_lengths:
        if torch.cuda.is_available() and model.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        inputs = tokenizer(benchmark_text, return_tensors="pt", truncation=True, max_length=target_len).to(model.device)
        actual_len = int(inputs.input_ids.shape[1])

        reset_runtime_stats(sliminfer_config)
        ttft_s, ttft_cache = generate_once(
            model, tokenizer, inputs.input_ids, inputs.attention_mask, 1, args, cache_classes, sliminfer_config
        )
        ttft_runtime = summarize_runtime_stats(sliminfer_config)
        peak_after_ttft = torch.cuda.max_memory_allocated() if torch.cuda.is_available() and model.device.type == "cuda" else 0

        if torch.cuda.is_available() and model.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        reset_runtime_stats(sliminfer_config)
        e2e_s, e2e_cache = generate_once(
            model, tokenizer, inputs.input_ids, inputs.attention_mask, args.decode_tokens, args, cache_classes, sliminfer_config
        )
        e2e_runtime = summarize_runtime_stats(sliminfer_config)
        peak_after_e2e = torch.cuda.max_memory_allocated() if torch.cuda.is_available() and model.device.type == "cuda" else 0

        cache_for_stats = e2e_cache or ttft_cache
        if args.method == "fullkv":
            prompt_kv_bytes = estimate_full_kv_bytes(model, actual_len, dtype)
            io_stats = {"h2d_bytes": 0, "d2h_bytes": 0, "async": False}
        else:
            prompt_kv_bytes = cache_gpu_bytes(cache_for_stats)
            io_stats = getattr(cache_for_stats, "io_stats", {"h2d_bytes": 0, "d2h_bytes": 0, "async": False})

        row = {
            "context_length": target_len,
            "actual_prompt_len": actual_len,
            "decode_tokens": args.decode_tokens,
            "TTFT_s": ttft_s,
            "E2E_s": e2e_s,
            "peak_gpu_mem_GB": max(peak_after_ttft, peak_after_e2e) / 1e9,
            "prompt_kv_mem_GB": prompt_kv_bytes / 1e9,
            "h2d_GB": int(io_stats.get("h2d_bytes", 0)) / 1e9,
            "d2h_GB": int(io_stats.get("d2h_bytes", 0)) / 1e9,
            "async_kv": bool(io_stats.get("async", False)),
            "swap_in_blocks": int(io_stats.get("swap_in_blocks", 0) or 0),
            "swap_out_blocks": int(io_stats.get("swap_out_blocks", 0) or 0),
            "swap_count": int(io_stats.get("swap_in_blocks", 0) or 0) + int(io_stats.get("swap_out_blocks", 0) or 0),
            "score_overhead_ms": float(e2e_runtime.get("score_overhead_ms", 0.0)),
            "score_calls": int(e2e_runtime.get("score_calls", 0) or 0),
            "active_set_churn": e2e_runtime.get("active_set_churn", ""),
            "jaccard_vs_original": e2e_runtime.get("jaccard_vs_original", ""),
        }
        print(f"Latency {target_len}: TTFT={ttft_s:.4f}s E2E={e2e_s:.4f}s prompt={actual_len}")
        rows.append(row)

    write_csv(out_dir / "latency.csv", rows, [
        "context_length", "actual_prompt_len", "decode_tokens", "TTFT_s", "E2E_s",
        "peak_gpu_mem_GB", "prompt_kv_mem_GB", "h2d_GB", "d2h_GB", "async_kv",
        "swap_in_blocks", "swap_out_blocks", "swap_count",
        "score_overhead_ms", "score_calls", "active_set_churn", "jaccard_vs_original",
    ])
    return {"rows": rows}


def build_chat(tokenizer, prompt: str) -> str:
    # Llama-3-8B base has no chat_template. LongBench prompts are already fully
    # formatted, so the base-model arm must be fed the raw prompt instead of crashing
    # or inventing a template (which would silently change the evaluated inputs).
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def load_longbench_samples(longbench_dir: Path, dataset: str) -> List[Dict[str, Any]]:
    candidates = [
        longbench_dir / f"{dataset}.jsonl",
        longbench_dir / f"{dataset}.json",
        longbench_dir / dataset / "test.jsonl",
        longbench_dir / dataset / "test.json",
        longbench_dir / dataset / f"{dataset}.jsonl",
        longbench_dir / dataset / f"{dataset}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]

    raise FileNotFoundError(
        f"LongBench task {dataset!r} not found under {longbench_dir}. "
        "This pipeline runs fully offline: downloading via datasets.load_dataset is "
        "forbidden because it would silently substitute an unverified dataset revision. "
        f"Searched: {[str(p) for p in candidates]}"
    )


def _sample_query_text(sample: Dict[str, Any]) -> str:
    for key in ("question", "query", "input", "instruction"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


_SOURCE_TITLE_LABEL_PATTERN = re.compile(
    r"^(?:passage|document|article|chunk)\s*\d+\s*:?\s*(.*)$",
    flags=re.IGNORECASE,
)


def _sample_title_heading_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only source-provided title-like fields/lines; never infer a title."""
    titles: List[str] = []
    for key in (
        "title",
        "document_title",
        "article_title",
        "chunk_title",
        "heading",
        "headings",
    ):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            titles.append(value.strip())
        elif isinstance(value, list):
            titles.extend(
                str(item).strip()
                for item in value
                if isinstance(item, str) and item.strip()
            )

    context = sample.get("context")
    if isinstance(context, str) and context.strip():
        lines = context.splitlines()
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            match = _SOURCE_TITLE_LABEL_PATTERN.match(line)
            if not match:
                continue
            inline = match.group(1).strip()
            if inline:
                titles.append(inline)
                continue
            for following in lines[index + 1:]:
                candidate = following.strip()
                if not candidate:
                    continue
                if _SOURCE_TITLE_LABEL_PATTERN.match(candidate):
                    break
                titles.append(candidate)
                break

    deduplicated: List[str] = []
    seen = set()
    for title in titles:
        normalized = " ".join(title.split())
        key = normalized.lower()
        if (
            not normalized
            or len(normalized) > 200
            or key in seen
        ):
            continue
        seen.add(key)
        deduplicated.append(normalized)
    return {
        "source_title_units": deduplicated,
        "title_or_heading_available": bool(deduplicated),
    }


def build_prompt_block_texts(tokenizer: AutoTokenizer, input_ids: torch.Tensor, block_size: int) -> List[str]:
    seq_len = int(input_ids.shape[-1])
    block_texts: List[str] = []
    for start in range(0, seq_len, block_size):
        block_texts.append(
            tokenizer.decode(
                input_ids[..., start:min(start + block_size, seq_len)].reshape(-1).tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return block_texts


def _snippet_features(snippet: str, query_text: str) -> Dict[str, Any]:
    query_words = set(re.findall(r"[A-Za-z0-9_\\-]+", query_text.lower()))
    snippet_words = set(re.findall(r"[A-Za-z0-9_\\-]+", snippet.lower()))
    overlap = len(query_words & snippet_words) / max(1, len(query_words)) if query_words else 0.0
    return {
        "question_overlap": round(overlap, 4),
        "has_number": bool(re.search(r"\d", snippet)),
        "entity_like_count": len(re.findall(r"\b[A-Z][A-Za-z0-9_\\-]{2,}\b", snippet)),
        "code_symbol_count": len(re.findall(r"[{}()[\\]._:=;|/\\\\-]", snippet)),
    }


def maybe_write_rescue_block_record(
    handle: Any,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model_input: Any,
    sliminfer_config: Optional[Dict[str, Any]],
    dataset: str,
    sample_idx: int,
    sample_id: Any,
    sample: Dict[str, Any],
) -> None:
    if handle is None or args.method != "bridgekv" or sliminfer_config is None:
        return
    state = get_bridgekv_state(sliminfer_config.get("bridgekv_state_id"))
    if state is None:
        return
    selection = state.get("selection_summary", {}) or {}
    block_ids = selection.get("replaced_block_ids_sampled", []) or []
    base_ranks = selection.get("replaced_block_base_ranks_sampled", []) or []
    bridge_ranks = selection.get("replaced_block_bridge_ranks_sampled", []) or []
    query_text = _sample_query_text(sample)
    max_snips = max(0, int(args.max_snippets_per_sample))
    input_ids = model_input.input_ids[0]
    seq_len = int(input_ids.shape[0])
    snippets: List[Dict[str, Any]] = []
    for idx, raw_block_id in enumerate(block_ids[:max_snips]):
        try:
            block_id = int(raw_block_id)
        except Exception:
            continue
        start = max(0, block_id * int(args.block_size))
        end = min(seq_len, start + int(args.block_size))
        if start >= end:
            continue
        token_slice = input_ids[start:end].detach().cpu().tolist()
        snippet = tokenizer.decode(
            token_slice,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        item: Dict[str, Any] = {
            "block_id": block_id,
            "token_start": start,
            "token_end": end,
            "base_rank": base_ranks[idx] if idx < len(base_ranks) else "",
            "rescue_rank": bridge_ranks[idx] if idx < len(bridge_ranks) else "",
            "text": snippet[:800],
        }
        item.update(_snippet_features(snippet, query_text))
        snippets.append(item)
    handle.write(json.dumps({
        "dataset": dataset,
        "sample_index": sample_idx,
        "sample_id": sample_id,
        "selection_policy": selection.get("selection_policy", ""),
        "rescue_source": selection.get("rescue_source", ""),
        "replacement_source": selection.get("replacement_source", ""),
        "selected_by_rescue_count": selection.get("selected_by_rescue_count", ""),
        "replaced_blocks_count": selection.get("replaced_blocks_count", ""),
        "jaccard_vs_base": selection.get("jaccard_vs_base", ""),
        "rescue_blocks": snippets,
    }, ensure_ascii=False) + "\n")


def run_gate_audit(
    model,
    tokenizer,
    cache_classes,
    args: argparse.Namespace,
    sliminfer_config: Optional[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    """Run prefill only and emit selector/gate audit; never decode or score."""
    if args.method == "fullkv" or sliminfer_config is None:
        raise RuntimeError("gate_audit requires a SlimInfer method/config")
    uasr_score_audit = bool(getattr(args, "uasr_score_dump_path", ""))
    drm_trace_audit = bool(getattr(args, "drm_trace_dir", ""))
    dc_selection_audit = bool(getattr(args, "dc_selection_audit_path", ""))
    cc_selection_audit = bool(getattr(args, "cc_selection_audit_path", ""))
    if (
        args.rescue_policy != "gated_budget_neutral_drm_swap"
        and not uasr_score_audit
        and not drm_trace_audit
        and not dc_selection_audit
        and not cc_selection_audit
    ):
        raise RuntimeError(
            "gate_audit is restricted to gated_budget_neutral_drm_swap "
            "unless --uasr_score_dump_path, --drm_trace_dir, "
            "--dc_selection_audit_path, or --cc_selection_audit_path is set "
            "for audit-only selector instrumentation"
        )
    if args.rescue_policy == "gated_budget_neutral_drm_swap" and args.gate_mode != "shadow":
        raise RuntimeError("gate_audit is shadow-only")

    dataset2prompt = load_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    datasets = [x for x in args.longbench_datasets.split(",") if x]
    manifest_path = out_dir / "gate_audit_manifest.jsonl"
    existing: List[Dict[str, Any]] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
    existing_keys = {
        (str(row.get("task")), int(row.get("sample_index", -1)))
        for row in existing
    }
    rows = list(existing)
    SlimInferCache, HsCache = cache_classes
    selected_indices_by_task = getattr(
        args, "longbench_sample_indices_by_task_map", {}
    )
    for dataset in datasets:
        samples = load_longbench_samples(Path(args.longbench_dir), dataset)
        explicit_indices = list(selected_indices_by_task.get(dataset, []))
        if explicit_indices:
            max_index = max(explicit_indices)
            if max_index >= len(samples):
                raise RuntimeError(
                    f"sample index {max_index} out of range for {dataset}: "
                    f"rows={len(samples)}"
                )
            indexed_samples = [
                (int(index), samples[int(index)]) for index in explicit_indices
            ]
        else:
            if args.longbench_max_samples > 0:
                samples = samples[: args.longbench_max_samples]
            indexed_samples = list(enumerate(samples))
        prompt_format = dataset2prompt[dataset]
        for sample_idx, sample in indexed_samples:
            if (dataset, sample_idx) in existing_keys:
                continue
            sample_id = sample.get("_id", sample.get("id", sample_idx))
            prompt = prompt_format.format(**sample)
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt"
            ).input_ids[0]
            if len(tokenized_prompt) > args.max_model_len:
                half = args.max_model_len // 2
                prompt = (
                    tokenizer.decode(
                        tokenized_prompt[:half], skip_special_tokens=True
                    )
                    + tokenizer.decode(
                        tokenized_prompt[-half:], skip_special_tokens=True
                    )
                )
            if dataset not in [
                "trec",
                "triviaqa",
                "samsum",
                "lsht",
                "lcc",
                "repobench-p",
            ]:
                prompt = build_chat(tokenizer, prompt)
            model_input = tokenizer(
                prompt, truncation=False, return_tensors="pt"
            ).to(model.device)
            context_length = int(model_input.input_ids.shape[1])
            query_text = _sample_query_text(sample)
            title_heading_metadata = _sample_title_heading_metadata(sample)
            block_texts = build_prompt_block_texts(
                tokenizer, model_input.input_ids, args.block_size
            )
            block_layout = build_blocks(
                model_input.input_ids[0],
                tokenizer,
                sample,
                prompt_text=prompt,
                task_name=dataset,
                block_strategy=args.block_strategy,
                target_block_tokens=args.target_block_tokens,
                min_block_tokens=args.min_block_tokens,
                max_block_tokens=args.max_block_tokens,
                query_text=query_text,
                config={
                    "block_size": args.block_size,
                    "fixed_page_tokens": args.block_size,
                    "semantic_parser": args.semantic_parser,
                    "semantic_target_block_tokens": args.semantic_target_block_tokens,
                    "semantic_min_block_tokens": args.semantic_min_block_tokens,
                    "semantic_max_block_tokens": args.semantic_max_block_tokens,
                    "semantic_page_align": args.semantic_page_align,
                    "max_page_expansion_overhead": args.max_page_expansion_overhead,
                    "query_top_ratio": args.query_top_ratio,
                    "query_neighbor_sentences": args.query_neighbor_sentences,
                    "query_high_max_block_tokens": args.query_high_max_block_tokens,
                    "query_low_max_block_tokens": args.query_low_max_block_tokens,
                    "query_score_mode": args.query_score_mode,
                    "query_parser": args.query_parser,
                },
            ).to_dict()
            sliminfer_config["audit_context"] = {
                "task": dataset,
                "sample_id": sample_id,
                "sample_index": sample_idx,
                "context_length": context_length,
                "method": out_dir.name,
                "query_text": query_text,
                "block_texts": block_texts,
                **title_heading_metadata,
                "prompt_text_preview": prompt[:512],
            }
            sliminfer_config["query_text"] = query_text
            sliminfer_config["block_texts"] = block_texts
            sliminfer_config.update(title_heading_metadata)
            sliminfer_config["block_layout"] = block_layout
            slim_cache = SlimInferCache(model.device)
            if args.debug_used_block_audit_path:
                slim_cache.debug_used_block_audit_path = (
                    args.debug_used_block_audit_path
                )
                slim_cache.debug_used_block_include_full = bool(
                    args.debug_used_block_include_full
                )
                slim_cache.debug_used_block_full_layers = {
                    int(part.strip())
                    for part in str(args.debug_used_block_full_layers).split(",")
                    if part.strip()
                }
                slim_cache.debug_context = {
                    "task": dataset,
                    "sample_id": sample_id,
                    "sample_index": sample_idx,
                    "method": out_dir.name,
                    "block_strategy": args.block_strategy,
                    "rescue_policy": args.rescue_policy,
                    "rescue_score": args.rescue_score,
                    "seed": args.seed,
                }
            kwargs = {
                "past_key_values": slim_cache,
                "hs_cache": HsCache(),
                "sliminfer_config": sliminfer_config,
                "all_original_prompt_indices_abs": list(range(context_length)),
            }
            started = time.perf_counter()
            with torch.no_grad():
                reset_runtime_stats(sliminfer_config)
                outputs = model(
                    input_ids=model_input.input_ids,
                    attention_mask=model_input.attention_mask,
                    use_cache=True,
                    return_dict=True,
                    **kwargs,
                )
            elapsed = time.perf_counter() - started
            del outputs
            if sliminfer_config.get("sp_trajectory_dump_path"):
                try:
                    from bridgekv.sp_trajectory import finalize_trajectory_sample

                    finalize_trajectory_sample(sliminfer_config)
                except Exception:
                    pass
            if sliminfer_config.get("variant_method") == "sp_sliminfer_v2":
                sliminfer_config.pop("_trajectory_hs_tokens", None)
                sliminfer_config.pop("_sp_v2_layer_history", None)
                sliminfer_config.pop("_sp_v2_scalar_rows", None)
            record = {
                "task": dataset,
                "sample_id": sample_id,
                "sample_index": sample_idx,
                "prompt_hash": hashlib.sha256(
                    prompt.encode("utf-8", errors="ignore")
                ).hexdigest(),
                "prompt_token_count": context_length,
                "gate_mode": args.gate_mode,
                "gate_policy": args.gate_policy,
                "quality_generation_performed": False,
                "scorer_invoked": False,
                "prefill_elapsed_s": elapsed,
            }
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append(record)
    write_json(
        out_dir / "gate_audit_status.json",
        {
            "status": "ok",
            "mode": "gate_audit",
            "quality_generation_performed": False,
            "scorer_invoked": False,
            "samples": len(rows),
        },
    )
    return {
        "status": "ok",
        "samples": len(rows),
        "quality_generation_performed": False,
        "scorer_invoked": False,
    }


def run_longbench(model, tokenizer, cache_classes, args: argparse.Namespace,
                  sliminfer_config: Optional[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    from longbench_eval import scorer

    dataset2prompt = load_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    dataset2maxlen = load_json(REPO_ROOT / "configs" / "dataset2maxlen.json")
    datasets = [x for x in args.longbench_datasets.split(",") if x]
    rows: List[Dict[str, Any]] = []
    sample_id_manifest: Dict[str, List[Any]] = {}
    pred_dir = out_dir / "longbench_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    resource_rows: List[Dict[str, Any]] = []
    failed_samples: List[Dict[str, Any]] = []
    cuda_available = torch.cuda.is_available()
    runtime_model_dtype = str(next(model.parameters()).dtype)
    runtime_attention_implementation = str(
        getattr(model.config, "_attn_implementation", "") or ""
    )
    rescue_record_handle = None
    trace_manager: Optional[AtomicTaskTraceManager] = None
    selection_digest_manager: Optional[AtomicTaskTraceManager] = None
    if args.active_block_trace != "none":
        if not args.active_block_trace_dir:
            raise RuntimeError("--active_block_trace requires --active_block_trace_dir")
        trace_manager = AtomicTaskTraceManager(
            Path(args.active_block_trace_dir),
            buffer_events=512,
        )
    if args.selection_digest_dir:
        selection_digest_manager = AtomicTaskTraceManager(
            Path(args.selection_digest_dir),
            buffer_events=512,
        )
    if args.save_selected_block_snippets:
        rescue_record_handle = (out_dir / "rescue_block_records.jsonl").open("w", encoding="utf-8")

    def load_existing_predictions(path: Path, expected: int) -> Tuple[List[str], List[Any], List[int], List[Any], List[Any]]:
        predictions: List[str] = []
        answers: List[Any] = []
        lengths: List[int] = []
        sample_ids: List[Any] = []
        all_classes: List[Any] = []
        if not path.exists():
            return predictions, answers, lengths, sample_ids, all_classes
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(predictions) >= expected:
                    break
                try:
                    row = json.loads(line)
                except Exception:
                    break
                predictions.append(str(row.get("pred", "")))
                answers.append(row.get("answers", []))
                lengths.append(int(row.get("length", 0) or 0))
                sample_ids.append(row.get("sample_id", row.get("sample_index", len(sample_ids))))
                if row.get("all_classes"):
                    all_classes = row.get("all_classes", all_classes)
        return predictions, answers, lengths, sample_ids, all_classes

    def prediction_record(
        *,
        dataset: str,
        sample_idx: int,
        sample: Dict[str, Any],
        source_sample_id: Any,
        canonical_sample_id: str,
        content_sha256: str,
        prompt: str,
        input_ids: torch.Tensor,
        prediction: str,
        status: str,
        output_token_count: Optional[int],
        context_length: int,
        max_new_tokens: int,
        eos_token_id: Any,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "sample_id": canonical_sample_id if args.protocol_id else source_sample_id,
            "sample_index": sample_idx,
            "pred": prediction,
            "answers": sample["answers"],
            "all_classes": sample.get("all_classes", []),
            "length": context_length,
        }
        if not args.protocol_id:
            if status != "ok":
                row.update({"status": status, "error": error})
            return row
        input_id_values = [int(value) for value in input_ids.detach().cpu().reshape(-1).tolist()]
        template = dataset2prompt[dataset]
        chat_template = tokenizer.chat_template or ""
        generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "temperature_effective": None,
            "top_p_effective": None,
        }
        row.update({
            "protocol_id": args.protocol_id,
            "implementation_revision": args.implementation_revision,
            "model_id": args.model_path,
            "method": args.method,
            "task": dataset,
            "source_sample_id": str(source_sample_id),
            "content_sha256": content_sha256,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
            "input_token_ids_sha256": hashlib.sha256(
                ",".join(str(value) for value in input_id_values).encode("ascii")
            ).hexdigest(),
            "status": status,
            "prediction": prediction,
            "input_tokens": context_length,
            "output_tokens": output_token_count,
            "max_new_tokens": int(max_new_tokens),
            "generation_config": generation_config,
            "generation_config_sha256": hashlib.sha256(
                json.dumps(
                    generation_config, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "runtime_dtype": runtime_model_dtype,
            "attention_implementation": runtime_attention_implementation,
            "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
            "tokenizer_id": args.tokenizer_path or args.model_path,
            "source_config_hash": args.source_config_hash,
            "field_sources": {
                "sample_id": "runtime",
                "prompt_sha256": "runtime",
                "status": "runtime",
                "input_tokens": "runtime",
                "output_tokens": "runtime" if output_token_count is not None else "unavailable",
            },
            "source_file": str(Path(args.longbench_dir) / f"{dataset}.jsonl"),
            "source_line": sample_idx + 1,
        })
        if error:
            row["error"] = error
        return row

    try:
        selected_indices_by_task = getattr(
            args, "longbench_sample_indices_by_task_map", {}
        )
        for dataset in datasets:
            try:
                samples_all = load_longbench_samples(Path(args.longbench_dir), dataset)
            except Exception as exc:
                rows.append({"dataset": dataset, "score": "", "num_samples": 0, "status": "failed", "error": repr(exc)})
                continue
            explicit_indices = list(selected_indices_by_task.get(dataset, []))
            if explicit_indices:
                max_index = max(explicit_indices)
                if max_index >= len(samples_all):
                    rows.append({
                        "dataset": dataset,
                        "score": "",
                        "num_samples": 0,
                        "status": "failed",
                        "error": (
                            f"sample index {max_index} out of range for "
                            f"{dataset}: rows={len(samples_all)}"
                        ),
                    })
                    continue
                indexed_samples = [
                    (int(index), samples_all[int(index)])
                    for index in explicit_indices
                ]
            else:
                samples = samples_all
                if args.longbench_max_samples > 0:
                    samples = samples[: args.longbench_max_samples]
                indexed_samples = list(enumerate(samples))

            prompt_format = dataset2prompt[dataset]
            max_gen = int(args.generation_max_new_tokens or dataset2maxlen[dataset])
            pred_path = pred_dir / f"{dataset}.jsonl"
            predictions, answers, lengths, sample_ids, all_classes = load_existing_predictions(pred_path, len(indexed_samples))
            start_idx = len(predictions)
            if start_idx >= len(indexed_samples):
                if trace_manager is not None:
                    completed_trace = Path(args.active_block_trace_dir) / f"{dataset}.jsonl.gz"
                    if not completed_trace.exists():
                        raise RuntimeError(
                            f"Predictions for {dataset} are complete but atomic trace is missing; "
                            "refusing an identity-incomplete resume"
                        )
                if selection_digest_manager is not None:
                    completed_digest = Path(args.selection_digest_dir) / f"{dataset}.jsonl.gz"
                    if not completed_digest.exists():
                        raise RuntimeError(
                            f"Predictions for {dataset} are complete but the atomic selection digest "
                            "is missing; refusing an identity-incomplete resume"
                        )
                score = scorer(dataset, predictions[: len(indexed_samples)], answers[: len(indexed_samples)], lengths[: len(indexed_samples)], all_classes)
                sample_id_manifest[dataset] = sample_ids[: len(indexed_samples)]
                rows.append({"dataset": dataset, "score": score, "num_samples": len(indexed_samples), "status": "ok", "error": ""})
                print(f"LongBench {dataset}: {score} (resumed)")
                continue
            if trace_manager is not None:
                if start_idx > 0:
                    raise RuntimeError(
                        f"{dataset} has {start_idx} resumed predictions but no completed atomic trace; "
                        "restart this incomplete task in a fresh task worker"
                    )
                trace_manager.start_task(dataset)
                if sliminfer_config is not None:
                    sliminfer_config["_active_block_trace_emit"] = trace_manager.write
            if selection_digest_manager is not None:
                if start_idx > 0:
                    raise RuntimeError(
                        f"{dataset} has {start_idx} resumed predictions but no completed atomic "
                        "selection digest; restart this incomplete task in a fresh task worker"
                    )
                selection_digest_manager.start_task(dataset)
                if sliminfer_config is not None:
                    sliminfer_config["_selection_digest_emit"] = selection_digest_manager.write
            file_mode = "a" if start_idx > 0 else "w"
            with pred_path.open(file_mode, encoding="utf-8") as f:
                for sample_pos in range(start_idx, len(indexed_samples)):
                    sample_idx, sample = indexed_samples[sample_pos]
                    require_min_free_gpu_gb(
                        args.min_free_gpu_gb,
                        dataset=dataset,
                        sample_index=sample_idx,
                    )
                    source_sample_id = sample.get("_id", sample.get("id", sample_idx)) if isinstance(sample, dict) else sample_idx
                    canonical_id, content_sha256 = canonical_sample_identity(dataset, sample_idx, sample)
                    sample_id = canonical_id if args.protocol_id else source_sample_id
                    sample_ids.append(sample_id)
                    prompt = prompt_format.format(**sample)
                    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                    if len(tokenized_prompt) > args.max_model_len:
                        half = args.max_model_len // 2
                        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                        prompt = build_chat(tokenizer, prompt)

                    model_input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
                    context_length = int(model_input.input_ids.shape[1])
                    query_text = _sample_query_text(sample if isinstance(sample, dict) else {})
                    title_heading_metadata = _sample_title_heading_metadata(
                        sample if isinstance(sample, dict) else {}
                    )
                    block_texts = build_prompt_block_texts(tokenizer, model_input.input_ids, args.block_size)
                    block_layout = None
                    if args.method != "fullkv":
                        block_layout = build_blocks(
                            model_input.input_ids[0],
                            tokenizer,
                            sample if isinstance(sample, dict) else {},
                            prompt_text=prompt,
                            task_name=dataset,
                            block_strategy=args.block_strategy,
                            target_block_tokens=args.target_block_tokens,
                            min_block_tokens=args.min_block_tokens,
                            max_block_tokens=args.max_block_tokens,
                            query_text=query_text,
                            config={
                                "block_size": args.block_size,
                                "fixed_page_tokens": args.block_size,
                                "semantic_parser": args.semantic_parser,
                                "semantic_target_block_tokens": args.semantic_target_block_tokens,
                                "semantic_min_block_tokens": args.semantic_min_block_tokens,
                                "semantic_max_block_tokens": args.semantic_max_block_tokens,
                                "semantic_page_align": args.semantic_page_align,
                                "max_page_expansion_overhead": args.max_page_expansion_overhead,
                                "query_top_ratio": args.query_top_ratio,
                                "query_neighbor_sentences": args.query_neighbor_sentences,
                                "query_high_max_block_tokens": args.query_high_max_block_tokens,
                                "query_low_max_block_tokens": args.query_low_max_block_tokens,
                                "query_score_mode": args.query_score_mode,
                                "query_parser": args.query_parser,
                            },
                        ).to_dict()
                    kwargs: Dict[str, Any] = {}
                    if args.method != "fullkv":
                        SlimInferCache, HsCache = cache_classes
                        if sliminfer_config is not None:
                            for reset_key in (
                                "_qdc_cached_result",
                                "_qdc_last",
                                "_qdc_block_token_lengths",
                                "_adaptive_dc_v3_gate_cached",
                                "_adaptive_dc_v3_gate_last",
                                "_mcu_cached_result",
                                "_mcu_last",
                                "_mcu_audit_written_keys",
                                "_mcu_runtime_selected_block_ids",
                            ):
                                sliminfer_config.pop(reset_key, None)
                            sliminfer_config["audit_context"] = {
                                "task": dataset,
                                "sample_id": sample_id,
                                "sample_index": sample_idx,
                                "context_length": context_length,
                                "method": args.method if args.protocol_id else out_dir.name,
                                "query_text": query_text,
                                "block_texts": block_texts,
                                **title_heading_metadata,
                                "prompt_text_preview": prompt[:512],
                            }
                            sliminfer_config["query_text"] = query_text
                            sliminfer_config["block_texts"] = block_texts
                            sliminfer_config.update(title_heading_metadata)
                            sliminfer_config["block_layout"] = block_layout
                            if args.method == "mcu_sliminfer" and sliminfer_config is not None:
                                sample_key = f"{dataset}:{sample_idx}"
                                frozen = (sliminfer_config.get("mcu_frozen_selections") or {}).get(sample_key)
                                if frozen:
                                    sliminfer_config["_mcu_cached_result"] = dict(frozen)
                                else:
                                    sliminfer_config.pop("_mcu_cached_result", None)
                            if args.block_strategy == "fixed_active_rescue":
                                try:
                                    rescue_set = build_rescue_candidates(
                                        model_input.input_ids[0],
                                        tokenizer,
                                        sample if isinstance(sample, dict) else {},
                                        task_name=dataset,
                                        fixed_active_pages=None,
                                        block_layout_fixed=None,
                                        strategy=args.rescue_candidate_strategy,
                                        query_text=query_text,
                                        config={
                                            "page_tokens": args.block_size,
                                            "prompt_text": prompt,
                                            "answer_text": (" ".join(str(a) for a in sample.get("answers", [])) if isinstance(sample, dict) and isinstance(sample.get("answers"), list) else str(sample.get("answers", "")) if isinstance(sample, dict) else ""),
                                            "overlap_window_tokens": args.rescue_overlap_window_tokens,
                                            "overlap_stride_tokens": args.rescue_overlap_stride_tokens,
                                            "evidence_half_window_tokens": args.rescue_evidence_half_window_tokens,
                                            "evidence_top_ratio": args.rescue_evidence_top_ratio,
                                            "semantic_page_tolerance": args.rescue_semantic_page_tolerance,
                                            "semantic_target_block_tokens": args.semantic_target_block_tokens,
                                            "semantic_min_block_tokens": args.semantic_min_block_tokens,
                                            "semantic_max_block_tokens": args.semantic_max_block_tokens,
                                            "coarse_block_tokens": args.coarse_block_tokens,
                                            "fine_block_tokens": args.fine_block_tokens,
                                            "medium_block_tokens": args.medium_block_tokens,
                                        },
                                    )
                                    sliminfer_config["rescue_candidate_layout"] = rescue_set.to_dict()
                                except Exception as exc:
                                    sliminfer_config["rescue_candidate_layout"] = {}
                                    print(f"[rescue_candidate_builder] failed: {exc}", flush=True)
                        slim_cache = SlimInferCache(model.device)
                        if args.debug_used_block_audit_path:
                            slim_cache.debug_used_block_audit_path = args.debug_used_block_audit_path
                            slim_cache.debug_used_block_include_full = bool(args.debug_used_block_include_full)
                            slim_cache.debug_used_block_full_layers = {
                                int(part.strip())
                                for part in str(args.debug_used_block_full_layers).split(",")
                                if part.strip()
                            }
                            slim_cache.debug_context = {
                                "task": dataset,
                                "sample_id": sample_id,
                                "sample_index": sample_idx,
                                "method": out_dir.name,
                                "block_strategy": args.block_strategy,
                                "rescue_policy": args.rescue_policy,
                                "rescue_score": args.rescue_score,
                                "seed": args.seed,
                            }
                        kwargs = {
                            "past_key_values": slim_cache,
                            "hs_cache": HsCache(),
                            "sliminfer_config": sliminfer_config,
                            "all_original_prompt_indices_abs": list(range(context_length)),
                        }
                        if args.method == "bridgekv":
                            initialize_bridgekv_state(args, tokenizer, model_input.input_ids, sliminfer_config)

                    eos_token_id = tokenizer.eos_token_id
                    if dataset == "samsum":
                        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
                        eos_token_id = [tokenizer.eos_token_id, newline_ids[-1]] if newline_ids else tokenizer.eos_token_id
                    do_sample = bool(args.temperature > 0.0 and (args.temperature != 1.0 or args.top_p != 1.0))
                    generation_kwargs: Dict[str, Any] = {}
                    if do_sample:
                        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
                    debug_logits = bool(args.debug_first_token_logits_path)
                    sample_started = time.perf_counter()
                    if cuda_available:
                        torch.cuda.reset_peak_memory_stats()

                    def _refresh_kv_kwargs() -> None:
                        """Rebuild KV/HS caches after OOM so a half-written cache cannot poison retry/next sample."""
                        if args.method == "fullkv" or not cache_classes or cache_classes[0] is None:
                            return
                        SlimInferCache, HsCache = cache_classes
                        kwargs["past_key_values"] = SlimInferCache(model.device)
                        kwargs["hs_cache"] = HsCache()
                        if sliminfer_config is not None:
                            reset_runtime_stats(sliminfer_config)
                    if sliminfer_config is not None:
                        sliminfer_config["_active_block_trace_event_index"] = 0

                    def _generate_once():
                        with torch.no_grad():
                            reset_runtime_stats(sliminfer_config)
                            return model.generate(
                                model_input.input_ids,
                                attention_mask=model_input.attention_mask,
                                max_new_tokens=max_gen,
                                num_beams=1,
                                do_sample=do_sample,
                                use_cache=True,
                                eos_token_id=eos_token_id,
                                return_dict_in_generate=debug_logits,
                                output_scores=debug_logits,
                                **generation_kwargs,
                                **kwargs,
                            )

                    generated = None
                    try:
                        generated = _generate_once()
                    except torch.cuda.OutOfMemoryError as oom_exc:
                        if cuda_available:
                            torch.cuda.empty_cache()
                        gc.collect()
                        if cuda_available:
                            torch.cuda.empty_cache()
                        _refresh_kv_kwargs()
                        try:
                            print(
                                f"[OOM-retry] {dataset} sample_index={sample_idx} "
                                f"context_length={context_length}: {oom_exc}",
                                flush=True,
                            )
                            generated = _generate_once()
                        except torch.cuda.OutOfMemoryError as oom_exc2:
                            # Mark OOM and continue; do not abort the whole run.
                            failed_samples.append({
                                "task": dataset,
                                "sample_id": sample_id,
                                "sample_index": sample_idx,
                                "error": "CUDA OOM",
                                "detail": repr(oom_exc2),
                                "context_length": context_length,
                            })
                            pred = "[OOM]"
                            predictions.append(pred)
                            answers.append(sample["answers"])
                            all_classes = sample.get("all_classes", all_classes)
                            lengths.append(context_length)
                            f.write(json.dumps(prediction_record(
                                dataset=dataset,
                                sample_idx=sample_idx,
                                sample=sample,
                                source_sample_id=source_sample_id,
                                canonical_sample_id=canonical_id,
                                content_sha256=content_sha256,
                                prompt=prompt,
                                input_ids=model_input.input_ids,
                                prediction=pred,
                                status="oom",
                                output_token_count=None,
                                context_length=context_length,
                                max_new_tokens=max_gen,
                                eos_token_id=eos_token_id,
                                error="CUDA OOM",
                            ), ensure_ascii=False) + "\n")
                            f.flush()
                            print(
                                f"[OOM-skip] {dataset} sample_index={sample_idx} "
                                f"context_length={context_length}: marked and continuing",
                                flush=True,
                            )
                            if cuda_available:
                                torch.cuda.empty_cache()
                            gc.collect()
                            if cuda_available:
                                torch.cuda.empty_cache()
                            continue
                    sample_elapsed = time.perf_counter() - sample_started
                    peak_alloc = int(torch.cuda.max_memory_allocated()) if cuda_available else 0
                    peak_reserved = int(torch.cuda.max_memory_reserved()) if cuda_available else 0
                    resource_rows.append({
                        "task": dataset,
                        "sample_id": sample_id,
                        "sample_index": sample_idx,
                        "wall_clock_s": round(sample_elapsed, 6),
                        "peak_memory_allocated_bytes": peak_alloc,
                        "peak_memory_reserved_bytes": peak_reserved,
                        "peak_memory_allocated_gb": round(peak_alloc / (1024 ** 3), 6) if peak_alloc else 0.0,
                        "peak_memory_reserved_gb": round(peak_reserved / (1024 ** 3), 6) if peak_reserved else 0.0,
                        "context_length": context_length,
                        "method": args.method,
                    })
                    if debug_logits:
                        output = generated.sequences[0]
                        try:
                            first_scores = generated.scores[0][0].detach().float().cpu()
                            top_k = min(20, int(first_scores.numel()))
                            top_vals, top_ids = torch.topk(first_scores, k=top_k)
                            top_probs = torch.softmax(top_vals, dim=-1)
                            chosen_id = int(output[context_length].detach().cpu()) if int(output.shape[0]) > context_length else -1
                            logits_record = {
                                "task": dataset,
                                "sample_id": sample_id,
                                "sample_index": sample_idx,
                                "method": out_dir.name,
                                "rescue_policy": args.rescue_policy,
                                "rescue_score": args.rescue_score,
                                "seed": args.seed,
                                "top20_token_ids": [int(x) for x in top_ids.tolist()],
                                "top20_logits": [float(x) for x in top_vals.tolist()],
                                "top20_probs": [float(x) for x in top_probs.tolist()],
                                "chosen_token_id": chosen_id,
                                "chosen_token_text": tokenizer.decode([chosen_id], skip_special_tokens=False) if chosen_id >= 0 else "",
                            }
                            logits_path = Path(args.debug_first_token_logits_path)
                            logits_path.parent.mkdir(parents=True, exist_ok=True)
                            with logits_path.open("a", encoding="utf-8") as lf:
                                lf.write(json.dumps(logits_record, ensure_ascii=False) + "\n")
                        except Exception as exc:
                            logits_path = Path(args.debug_first_token_logits_path)
                            logits_path.parent.mkdir(parents=True, exist_ok=True)
                            with logits_path.open("a", encoding="utf-8") as lf:
                                lf.write(json.dumps({
                                    "task": dataset,
                                    "sample_id": sample_id,
                                    "sample_index": sample_idx,
                                    "method": out_dir.name,
                                    "error": repr(exc),
                                }, ensure_ascii=False) + "\n")
                    else:
                        output = generated[0]
                    maybe_write_rescue_block_record(
                        rescue_record_handle,
                        args,
                        tokenizer,
                        model_input,
                        sliminfer_config,
                        dataset,
                        sample_idx,
                        sample_id,
                        sample,
                    )
                    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
                    if args.debug_generation_jsonl:
                        gen_path = Path(args.debug_generation_jsonl)
                        gen_path.parent.mkdir(parents=True, exist_ok=True)
                        prompt_hash = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()
                        output_hash = hashlib.sha256(pred.encode("utf-8", errors="ignore")).hexdigest()
                        with gen_path.open("a", encoding="utf-8") as gf:
                            gf.write(json.dumps({
                                "task": dataset,
                                "sample_id": sample_id,
                                "sample_index": sample_idx,
                                "method": out_dir.name,
                                "block_strategy": args.block_strategy,
                                "rescue_policy": args.rescue_policy,
                                "rescue_score": args.rescue_score,
                                "seed": args.seed,
                                "prompt_hash": prompt_hash,
                                "output_text": pred,
                                "output_hash": output_hash,
                                "metric_score": "",
                                "output_same_as_fixed_sliminfer": "",
                                "output_same_as_same_strategy_sliminfer": "",
                            }, ensure_ascii=False) + "\n")
                    predictions.append(pred)
                    answers.append(sample["answers"])
                    all_classes = sample.get("all_classes", all_classes)
                    lengths.append(context_length)
                    f.write(json.dumps(prediction_record(
                        dataset=dataset,
                        sample_idx=sample_idx,
                        sample=sample,
                        source_sample_id=source_sample_id,
                        canonical_sample_id=canonical_id,
                        content_sha256=content_sha256,
                        prompt=prompt,
                        input_ids=model_input.input_ids,
                        prediction=pred,
                        status="ok",
                        output_token_count=max(0, int(output.shape[0]) - context_length),
                        context_length=context_length,
                        max_new_tokens=max_gen,
                        eos_token_id=eos_token_id,
                    ), ensure_ascii=False) + "\n")
                    f.flush()
                    # Release per-sample tensors so summarization tasks don't fragment VRAM.
                    del generated, output, model_input
                    if "past_key_values" in kwargs:
                        kwargs.pop("past_key_values", None)
                    if cuda_available:
                        torch.cuda.empty_cache()

            score = scorer(dataset, predictions, answers, lengths, all_classes)
            sample_id_manifest[dataset] = sample_ids
            rows.append({"dataset": dataset, "score": score, "num_samples": len(indexed_samples), "status": "ok", "error": ""})
            if trace_manager is not None:
                trace_manager.finish_task(dataset)
                if sliminfer_config is not None:
                    sliminfer_config.pop("_active_block_trace_emit", None)
            if selection_digest_manager is not None:
                selection_digest_manager.finish_task(dataset)
                if sliminfer_config is not None:
                    sliminfer_config.pop("_selection_digest_emit", None)
            print(f"LongBench {dataset}: {score}")
    finally:
        if trace_manager is not None:
            trace_manager.abort_open_task()
        if selection_digest_manager is not None:
            selection_digest_manager.abort_open_task()
        if sliminfer_config is not None:
            sliminfer_config.pop("_active_block_trace_emit", None)
            sliminfer_config.pop("_selection_digest_emit", None)
        if rescue_record_handle is not None:
            rescue_record_handle.close()

    numeric_scores = [float(row["score"]) for row in rows if row.get("score") not in ("", None)]
    avg = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None
    write_csv(out_dir / "longbench_scores.csv", rows, ["dataset", "score", "num_samples", "status", "error"])
    write_json(out_dir / "longbench_sample_ids.json", sample_id_manifest)
    write_json(out_dir / "sample_ids.json", sample_id_manifest)
    if resource_rows:
        write_csv(
            out_dir / "runtime_resource_stats.csv",
            resource_rows,
            [
                "task", "sample_id", "sample_index", "wall_clock_s", "peak_memory_allocated_bytes",
                "peak_memory_reserved_bytes", "peak_memory_allocated_gb", "peak_memory_reserved_gb",
                "context_length", "method",
            ],
        )
    if failed_samples:
        fail_path = out_dir / "failed_samples.jsonl"
        with fail_path.open("w", encoding="utf-8") as ff:
            for rec in failed_samples:
                ff.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"rows": rows, "avg": avg, "resource_rows": resource_rows, "failed_samples": failed_samples}


def write_empty_outputs(out_dir: Path) -> None:
    if not (out_dir / "latency.csv").exists():
        write_csv(out_dir / "latency.csv", [], [
            "context_length", "actual_prompt_len", "decode_tokens", "TTFT_s", "E2E_s",
            "peak_gpu_mem_GB", "prompt_kv_mem_GB", "h2d_GB", "d2h_GB", "async_kv",
            "swap_in_blocks", "swap_out_blocks", "swap_count",
            "score_overhead_ms", "score_calls", "active_set_churn", "jaccard_vs_original",
        ])
    if not (out_dir / "longbench_scores.csv").exists():
        write_csv(out_dir / "longbench_scores.csv", [], ["dataset", "score", "num_samples", "status", "error"])


def maybe_collect(root: Path) -> None:
    try:
        from experiments.collect_qwen25_results import collect_results
        collect_results(root, root / "qwen25_summary.csv", root / "qwen25_summary.md")
    except Exception as exc:
        print(f"Warning: result collection skipped: {exc}")


def write_profile_artifacts(out_dir: Path, metrics: Dict[str, Any], sliminfer_config: Optional[Dict[str, Any]]) -> None:
    latency_rows = metrics.get("latency", {}).get("rows", [])
    memory_rows = [
        {
            "context_length": row.get("context_length"),
            "peak_gpu_mem_GB": row.get("peak_gpu_mem_GB"),
            "prompt_kv_mem_GB": row.get("prompt_kv_mem_GB"),
        }
        for row in latency_rows
    ]
    write_json(out_dir / "memory_profile.json", {"rows": memory_rows})
    io_rows = [
        {
            "context_length": row.get("context_length"),
            "h2d_GB": row.get("h2d_GB"),
            "d2h_GB": row.get("d2h_GB"),
            "async_kv": row.get("async_kv"),
        }
        for row in latency_rows
    ]
    write_json(out_dir / "io_profile.json", {"rows": io_rows})

    runtime = summarize_runtime_stats(sliminfer_config)
    write_json(out_dir / "block_score_summary.json", runtime)
    write_json(out_dir / "layer_stats_summary.json", runtime)
    write_csv(out_dir / "active_set_stats.csv", [runtime], [
        "score_overhead_ms",
        "score_calls",
        "prefill_score_calls",
        "decode_score_calls",
        "decode_reuse_calls",
        "active_block_overlap",
        "active_set_churn",
        "jaccard_vs_original",
    ])

    bridge_summary = {"graph_summary": {}, "score_summary": {}, "selection_summary": {}}
    if sliminfer_config is not None and sliminfer_config.get("bridgekv_state_id"):
        state = get_bridgekv_state(sliminfer_config.get("bridgekv_state_id"))
        if state is not None:
            bridge_summary = {
                "graph_summary": state.get("graph_summary", {}),
                "score_summary": state.get("bridge_score_summary", {}),
                "selection_summary": state.get("selection_summary", {}),
            }
            graph_summary = bridge_summary["graph_summary"]
            score_summary = bridge_summary["score_summary"]
            selection_summary = bridge_summary["selection_summary"]
            for key in [
                "graph_build_time_ms",
                "ppr_time_ms",
                "bridge_overhead_ms",
                "num_edges_total",
                "avg_degree",
                "max_degree",
                "num_query_edges",
                "num_semantic_edges",
                "num_position_edges",
                "num_attention_edges",
                "bridge_edge_weighting",
                "bridge_hub_penalty",
                "bridge_semantic_threshold",
                "anchor_mode",
                "anchor_ratio",
                "control_type",
            ]:
                if key in graph_summary:
                    metrics[key] = graph_summary.get(key)
            for key in ["bridge_score_type", "bridge_ppr_time_ms", "path_coverage_time_ms", "path_coverage_blocks_count"]:
                if key in score_summary:
                    metrics[key] = score_summary.get(key)
            for key in [
                "jaccard_vs_qk_reuse",
                "jaccard_vs_delta_o_reuse",
                "selected_by_bridge_count",
                "selected_by_local_count",
                "selected_by_structural_count",
                "selection_policy",
                "base_score",
                "bridge_quota_blocks_count",
                "selected_by_rescue_count",
                "replaced_blocks_count",
                "jaccard_vs_base",
                "bridge_score",
                "control_type",
                "replacement_source",
                "rescue_source",
                "replaced_block_base_ranks_sampled",
                "replaced_block_bridge_ranks_sampled",
            ]:
                if key in selection_summary:
                    metrics[key] = selection_summary.get(key)
    write_bridgekv_summaries(out_dir, bridge_summary)
    write_experiment_diagnosis(out_dir, metrics)


def write_experiment_diagnosis(out_dir: Path, metrics: Dict[str, Any]) -> None:
    method = metrics.get("method", "")
    scoring = metrics.get("scoring", "")
    lines = [
        "# Experiment Diagnosis",
        "",
        f"- Status: {metrics.get('status', 'unknown')}",
        f"- Method: {method}",
        f"- Scoring: {scoring}",
        f"- Backend: {metrics.get('attn_implementation', '')}",
    ]
    notes = metrics.get("notes", [])
    if notes:
        lines.append(f"- Notes: {'; '.join(str(note) for note in notes)}")
    if method == "bridgekv":
        lines.extend([
            "",
            "BridgeKV is a reasoning-path-aware compression prototype. It augments local importance scores such as qk relevance or delta_o with graph-based bridge utility, aiming to preserve not only evidence blocks but also intermediate bridge blocks that maintain query-to-answer information flow.",
            "",
            "BridgeKV 的核心不是局部 output damage，而是 local damage + reasoning path bridge utility。",
            f"- Graph build time ms: {metrics.get('graph_build_time_ms', '')}",
            f"- PPR time ms: {metrics.get('ppr_time_ms', metrics.get('bridge_ppr_time_ms', ''))}",
            f"- Edges total: {metrics.get('num_edges_total', '')}",
            f"- Jaccard vs qk reuse: {metrics.get('jaccard_vs_qk_reuse', '')}",
            f"- Jaccard vs delta_o reuse: {metrics.get('jaccard_vs_delta_o_reuse', '')}",
        ])
    (out_dir / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = normalize_args(parse_args())
    seed_everything(args.seed)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")

    status = "ok"
    notes: List[str] = []
    metrics: Dict[str, Any] = {
        "exp_name": out_dir.name,
        "method": args.method,
        "scoring": args.scoring,
        "targets": TARGETS,
        "status": status,
        "notes": notes,
        "env": get_env_info(),
    }

    if args.method == "joint_token_rank" and not rank_code_available():
        status = "skipped"
        notes.append("No EigenAttention/FDP-EB/low-rank KV compression code was found in this repository; interface only.")
        metrics.update({"status": status, "skipped": True})
        write_json(out_dir / "config.json", {"args": vars(args), "sliminfer_config": None, "derived": {}, "env": metrics["env"]})
        write_empty_outputs(out_dir)
        write_json(out_dir / "metrics.json", metrics)
        maybe_collect(out_dir.parent)
        return

    model = tokenizer = None
    sliminfer_config: Optional[Dict[str, Any]] = None
    derived: Dict[str, Any] = {}
    cache_classes = (None, None)
    actual_attn_impl = "unknown"
    try:
        model, tokenizer, cache_classes, actual_attn_impl = load_model_and_tokenizer(args, attn_impl=args.backend)
        if args.backend == "flash_attention_2" and actual_attn_impl != "flash_attention_2":
            notes.append(f"warning: requested flash_attention_2 but actual backend is {actual_attn_impl}")
        if actual_attn_impl == "sdpa" and not flash_attention_available():
            notes.append("flash_attn is not installed in this environment; sdpa is the frozen backend")
        num_layers = int(getattr(model.config, "num_hidden_layers", 32))
        if args.method != "fullkv":
            sliminfer_config, derived, config_notes = build_sliminfer_config(args, out_dir, num_layers)
            notes.extend(config_notes)
            aligned = all(tokens % args.block_size == 0 for tokens in derived.get("retain_tokens", []))
            metrics["sanity_checks"] = {
                "generated_tokens_retained_by_decode_cache": True,
                "preserve_first_block": bool(args.preserve_first_block),
                "retain_tokens_block_aligned": aligned,
                "active_blocks_nonempty_guard": True,
                "cpu_offload_fetch_on_reactivate": bool(args.enable_cpu_offload),
                "async_on_off_equivalence": "not_run",
            }
            if args.method == "residual_comp":
                metrics["residual"] = {
                    "type": args.residual_type,
                    "slots": args.residual_slots,
                    "scale": args.residual_scale,
                    "runtime_applied": False,
                    "residual_token_overhead": args.residual_slots,
                }
        else:
            derived = {
                "prune_layers": [],
                "retain_tokens": [],
                "profile": {},
            }
            metrics["sanity_checks"] = {
                "fullkv_no_hidden_state_pruning": True,
                "fullkv_no_cpu_offload": True,
            }
        metrics["attn_implementation"] = actual_attn_impl
        config_payload = {
            "args": vars(args),
            "sliminfer_config": sliminfer_config,
            "derived": derived,
            "env": metrics["env"],
            "attn_implementation": actual_attn_impl,
        }
        write_json(out_dir / "config.json", config_payload)

        if args.eval in {"latency", "all"}:
            metrics["latency"] = run_latency(model, tokenizer, cache_classes, args, sliminfer_config, out_dir)
        if args.eval == "gate_audit":
            metrics["gate_audit"] = run_gate_audit(
                model,
                tokenizer,
                cache_classes,
                args,
                sliminfer_config,
                out_dir,
            )
        if args.eval in {"longbench", "all"}:
            metrics["longbench"] = run_longbench(model, tokenizer, cache_classes, args, sliminfer_config, out_dir)
        if args.eval == "ppl":
            metrics["ppl"] = {"status": "skipped", "reason": "No repository PPL entrypoint is present; interface reserved."}
        if args.eval == "needle":
            metrics["needle"] = {"status": "skipped", "reason": "Use needle_in_haystack.py for dedicated NIAH sweeps; unified hook reserved."}

        if "latency" in metrics and metrics["latency"]["rows"]:
            last = metrics["latency"]["rows"][-1]
            metrics["peak_gpu_mem_GB"] = max(float(row["peak_gpu_mem_GB"]) for row in metrics["latency"]["rows"])
            metrics["prompt_kv_mem_GB"] = max(float(row["prompt_kv_mem_GB"]) for row in metrics["latency"]["rows"])
            metrics["h2d_GB"] = sum(float(row["h2d_GB"]) for row in metrics["latency"]["rows"])
            metrics["d2h_GB"] = sum(float(row["d2h_GB"]) for row in metrics["latency"]["rows"])
            metrics["swap_count"] = sum(int(row.get("swap_count", 0) or 0) for row in metrics["latency"]["rows"])
            metrics["score_overhead_ms"] = max(float(row.get("score_overhead_ms", 0.0) or 0.0) for row in metrics["latency"]["rows"])
            metrics["score_calls"] = sum(int(row.get("score_calls", 0) or 0) for row in metrics["latency"]["rows"])
            churn_values = [float(row["active_set_churn"]) for row in metrics["latency"]["rows"] if row.get("active_set_churn") not in ("", None)]
            jaccard_values = [float(row["jaccard_vs_original"]) for row in metrics["latency"]["rows"] if row.get("jaccard_vs_original") not in ("", None)]
            if churn_values:
                metrics["active_set_churn"] = sum(churn_values) / len(churn_values)
            if jaccard_values:
                metrics["jaccard_vs_original"] = sum(jaccard_values) / len(jaccard_values)
            metrics["async_kv"] = bool(last.get("async_kv", False))
            if args.method != "fullkv" and args.enable_async_kv and not metrics["async_kv"]:
                notes.append("unsupported async KV; using synchronous fallback")
        if "longbench" in metrics:
            metrics["LongBench_avg"] = metrics["longbench"].get("avg")

    except torch.cuda.OutOfMemoryError as exc:
        status = "OOM"
        metrics["error"] = repr(exc)
        raise
    except Exception as exc:
        status = "failed"
        metrics["error"] = repr(exc)
        raise
    finally:
        metrics["status"] = status
        metrics["notes"] = notes
        if args.eval == "gate_audit":
            # Keep this path semantically audit-only: no LongBench placeholders,
            # aggregate method tables, or generic quality/latency profiles.
            metrics.pop("targets", None)
        if not (out_dir / "config.json").exists():
            write_json(out_dir / "config.json", {
                "args": vars(args),
                "sliminfer_config": sliminfer_config,
                "derived": derived,
                "env": metrics.get("env", {}),
                "attn_implementation": actual_attn_impl,
            })
        if args.eval != "gate_audit":
            write_empty_outputs(out_dir)
            write_profile_artifacts(out_dir, metrics, sliminfer_config)
        write_json(out_dir / "metrics.json", metrics)
        if args.eval != "gate_audit":
            maybe_collect(out_dir.parent)
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
