from typing import  Optional, Dict, List, Any, Set, Tuple

import torch

from utils.cache_utils import (
    from_abs_indices_to_block_ids, from_blocks_to_abs_indices, prune_from_list, 
    from_abs_indices_to_block_ids_list, from_blocks_to_abs_indices_list, 
    _calculate_pruning_with_rep_keys
)
from utils.cache_utils import SlimInferCache, HsCache
import utils.cache_utils as sp_utils

DEFAULT_LAYER_PRUNING_CONFIG = {
    "pruning_ratio": 0.0,             
    "pruning_threshold": 0.0,         
    "min_keep_tokens": 2048,          
    "min_keep_ratio": 0.3,            
    "always_keep_sink_recent_token": True,
}

class SlimInferInferenceState:
    def __init__(self,
                 all_original_prompt_indices_abs: List[int],
                 is_decode_phase: bool,
                 device: torch.device,
                 bsz: int,
                 decoded_token_abs_idx: Optional[int] = None):
        # --- static configs ---
        self.all_original_prompt_indices_abs = all_original_prompt_indices_abs
        self.original_prompt_len = len(all_original_prompt_indices_abs)
        self.is_decode_phase = is_decode_phase
        self.device = device
        self.bsz = bsz
        self.decoded_token_abs_idx = decoded_token_abs_idx

        # --- dynamic states ---
        self.current_APIA_set: Set[int] = set(all_original_prompt_indices_abs)
        self.current_APIA_block_set: Set[int] = from_abs_indices_to_block_ids(self.current_APIA_set)

        self.current_attention_mask: Optional[torch.Tensor] = None
        self.current_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # --- flag && data ---
        self.hidden_states_prev_layer: Optional[torch.Tensor] = None
        self.hs_prev_mlp_output_abs_indices_list: Optional[List[int]] = None
        self.prev_mlp_abs_to_relative_idx_map: Dict[int, int] = {}
        self.prev_layer_was_pruned_flag: bool = False
        self.need_to_swap_kv_cache_flag: bool = False
        self.need_to_rebuild_attention_mask_flag: bool = False
        self.need_to_rebuild_position_embeddings_flag: bool = False
        self.query_block_id_list : List[int] = []

class SlimInferLayerProcessor:
    def prepare_layer_inputs(
        self,
        layer_idx: int,
        state: SlimInferInferenceState,
        initial_inputs: Dict[str, Any],
        slim_kv_cache: SlimInferCache,
        hs_cache: HsCache,
    ) -> Dict[str, Any]:
        if layer_idx == 0:
            return {
                "hs_for_query": initial_inputs["inputs_embeds"],
                "query_abs_indices_list": initial_inputs["current_input_abs_indices_list"],
                "query_abs_to_relative_map": initial_inputs["input_abs_to_batch_idx_map"],
                "restored_position_ids": None,
                "blocks_fetched_from_hs_cache": [],
            }
        
        if not state.prev_layer_was_pruned_flag:
            return {
                "hs_for_query": state.hidden_states_prev_layer,
                "query_abs_indices_list": state.hs_prev_mlp_output_abs_indices_list,
                "query_abs_to_relative_map": state.prev_mlp_abs_to_relative_idx_map,
                "restored_position_ids": None,
                "blocks_fetched_from_hs_cache": [],
            }
        
        # check which blocks need to be fetched from HsCache
        hs_from_prev_layer = state.hidden_states_prev_layer
        current_query_indices = state.hs_prev_mlp_output_abs_indices_list or []
        
        blocks_to_fetch = []
        for block_id in state.current_APIA_block_set:
            if slim_kv_cache.has_block_key_value(layer_idx, block_id):
                continue
            if hs_cache.has_hidden_state_block(layer_idx - 1, block_id):
                # check if already present
                is_block_already_present = block_id in state.query_block_id_list
                if not is_block_already_present:
                    blocks_to_fetch.append(block_id)

        if not blocks_to_fetch:
            return {
                "hs_for_query": hs_from_prev_layer,
                "query_abs_indices_list": current_query_indices,
                "query_abs_to_relative_map": state.prev_mlp_abs_to_relative_idx_map,
                "restored_position_ids": None,
                "blocks_fetched_from_hs_cache": [],
            }

        # get hidden states from HsCache and concatenate
        state.need_to_rebuild_attention_mask_flag = True
        state.query_block_id_list.extend(blocks_to_fetch)
        hs_from_cache = hs_cache.get_hidden_states(layer_idx - 1, blocks_to_fetch)
        if hs_from_cache is None or hs_from_prev_layer is None:
            raise KeyError("hs_from_cache or hs_from_prev_layer is None!")
        hs_for_query = torch.cat([hs_from_prev_layer, hs_from_cache], dim=1)
        
        restored_indices = from_blocks_to_abs_indices_list(blocks_to_fetch, state.original_prompt_len)
        query_abs_indices_list = current_query_indices + restored_indices
        query_abs_to_relative_map = {idx: i for i, idx in enumerate(query_abs_indices_list)}

        return {
            "hs_for_query": hs_for_query,
            "query_abs_indices_list": query_abs_indices_list,
            "query_abs_to_relative_map": query_abs_to_relative_map,
            "restored_position_ids": torch.tensor(restored_indices, device=state.device, dtype=torch.long).unsqueeze(0),
            "blocks_fetched_from_hs_cache": blocks_to_fetch,
        }

    def plan_next_layer_state(
        self,
        layer_idx: int,
        state: SlimInferInferenceState,
        q_squery: torch.Tensor,
        num_key_value_groups: int,
        layer_inputs: Dict[str, Any],
        sliminfer_config: Dict[str, Any],
        slim_kv_cache: SlimInferCache,
    ) -> Tuple[Set[int], Set[int]]:
        """
        prune according to attention scores, update cache and inference state (state object)
        returns the next layer's APIA set, does not modify current layer's hidden_states
        """
        query_abs_indices_list = layer_inputs["query_abs_indices_list"]
        query_abs_to_relative_map = layer_inputs["query_abs_to_relative_map"]

        # get APIA for next layer
        next_APIA_set = state.current_APIA_set
        next_APIA_block_set = state.current_APIA_block_set
        state.prev_layer_was_pruned_flag = False
                
        layer_pruning_cfg = DEFAULT_LAYER_PRUNING_CONFIG.copy()
        pruning_config = sliminfer_config.get("pruning_config", {})
        if pruning_config and layer_idx in pruning_config:
            layer_pruning_cfg.update(pruning_config[layer_idx])

        method = sliminfer_config.get("pruning_method")
        layer_pruning_cfg.update({"pruning_method": method})
        for key in [
            "variant_method",
            "score_kind",
            "scoring",
            "fd_alpha_o",
            "fd_beta_h",
            "fd_gamma_s",
            "fd_beta_v",
            "fd_normalize",
            "fd_score_clip",
            "fd_topk_mode",
            "fd_compute_scope",
            "fd_approx",
            "fd_use_entropy",
            "fd_use_score_proxy",
            "fd_use_value_norm",
            "fd_use_renorm",
            "hybrid_qk_weight",
            "hybrid_o_weight",
            "risk_ratio",
            "kv_mode",
            "kv_topk_risk_ratio",
            "lambda_k",
            "lambda_v",
            "head_group_mode",
            "num_head_groups",
            "head_agg",
            "layer_stats_path",
            "block_scores_path",
            "reuse_decode_blocks",
            "decode_policy",
            "refresh_interval",
            "active_set_hysteresis",
            "active_set_momentum",
            "bridgekv_state_id",
            "selection_policy",
            "base_score",
            "quota_ratio",
            "core_ratio",
            "boundary_ratio",
            "replace_ratio",
            "rescue_ratio",
            "rescue_policy",
            "rescue_score",
            "seed",
            "rescue_audit_path",
            "swap_drop_score",
            "alpha_diff",
            "lambda_red",
            "eta_rescue",
            "eta_active",
            "beta_path",
            "beta_query",
            "stochastic_mode",
            "gumbel_tau",
            "candidate_pool_ratio",
            "sampling_temperature",
            "block_texts",
            "query_text",
            "swap_tau",
            "audit_context",
            "agreement_top_p",
            "evidence_ratio",
            "bridge_alpha",
            "bridge_beta",
            "bridge_gamma",
            "bridge_eta",
            "bridge_graph_edges",
            "bridge_score",
            "bridge_ppr_alpha",
            "bridge_pseudo_evidence_ratio",
            "bridge_anchor_mode",
            "bridge_anchor_ratio",
            "bridge_pos_edge_weight",
            "bridge_topk_sem_edges",
            "bridge_query_edge_weight",
            "bridge_edge_weighting",
            "bridge_hub_penalty",
            "bridge_hub_lambda",
            "bridge_idf_node",
            "bridge_semantic_threshold",
            "num_evidence_anchors",
            "max_paths_per_anchor",
            "bridge_norm",
            "bridge_structural_risk",
            "local_score",
            "stats_mode",
            "block_scores_mode",
            "runtime_stats",
            "block_strategy",
            "target_block_tokens",
            "min_block_tokens",
            "max_block_tokens",
            "block_overlap_tokens",
            "semantic_parser",
            "semantic_target_block_tokens",
            "semantic_min_block_tokens",
            "semantic_max_block_tokens",
            "semantic_page_align",
            "max_page_expansion_overhead",
            "attention_boundary_layer",
            "boundary_percentile",
            "query_top_ratio",
            "query_neighbor_sentences",
            "query_high_max_block_tokens",
            "query_low_max_block_tokens",
            "query_score_mode",
            "query_parser",
            "coarse_block_tokens",
            "fine_block_tokens",
            "medium_block_tokens",
            "top_coarse_ratio",
            "budget_mode",
            "target_token_budget_source",
            "target_token_budget",
            "token_budget_tolerance",
            "selection_cost",
            "score_density_mode",
            "allow_budget_overshoot",
            "soft_budget_tolerance",
            "hard_budget_tolerance",
            "active_budget_mode",
            "rescue_budget_mode",
            "target_active_tokens",
            "target_final_tokens",
            "target_final_tokens_source",
            "active_block_strategy",
            "rescue_candidate_strategy",
            "rescue_candidate_layout",
            "lambda_len",
            "beta_semantic",
            "rescue_overlap_window_tokens",
            "rescue_overlap_stride_tokens",
            "rescue_evidence_half_window_tokens",
            "rescue_evidence_top_ratio",
            "rescue_semantic_page_tolerance",
            "rescue_selector",
            "rescue_gate_mode",
            "rescue_gate_threshold",
            "coverage_gamma",
            "noise_mu",
            "block_budget_mode",
            "block_budget_block_tokens",
            "block_budget_log_weight",
            "verifier_mode",
            "verifier_topn_mult",
            "feature_dump_path",
            "forced_rescue_pages",
            "page_utility_map",
            "guarded_swap",
            "active_core_ratio",
            "max_swaps",
            "swap_tau",
            "swap_utility_source",
            "hybrid_exclude_active_overlap",
            "hybrid_overlap_threshold",
            "gamma_boundary",
            "boundary_weight_active_score",
            "gamma_active_boundary",
            "calibrated_budget_config",
            "calibrated_budget_config_path",
            "block_layout",
            "block_layout_audit_path",
            "_block_layout_written",
        ]:
            if key in sliminfer_config:
                layer_pruning_cfg[key] = sliminfer_config[key]
        layer_pruning_cfg["layer_idx"] = layer_idx
        layer_pruning_cfg["phase"] = "decode" if state.is_decode_phase else "prefill"
        layer_pruning_cfg["decoded_token_abs_idx"] = state.decoded_token_abs_idx
        layer_pruning_cfg["original_prompt_len"] = state.original_prompt_len
        ratio = layer_pruning_cfg.get("pruning_ratio", 0.0)
        threshold = layer_pruning_cfg.get("pruning_threshold", 0.0)
        fixed_keep_tokens = layer_pruning_cfg.get("fixed_keep_tokens", 0.0)
        should_trigger_pruning = (method == 'threshold' and threshold > 0.0) or \
                                (method == 'ratio' and ratio > 0.0) or \
                                (method == 'fixed_tokens' and fixed_keep_tokens > 0.0)

        if should_trigger_pruning:
            decode_policy = layer_pruning_cfg.get("decode_policy", "original")
            should_reuse_decode = bool(layer_pruning_cfg.get("reuse_decode_blocks", False))
            if state.is_decode_phase and decode_policy == "periodic_refresh":
                refresh_interval = max(1, int(layer_pruning_cfg.get("refresh_interval", 8)))
                decode_step = 0
                if state.decoded_token_abs_idx is not None:
                    decode_step = max(0, int(state.decoded_token_abs_idx) - int(state.original_prompt_len))
                should_reuse_decode = (decode_step % refresh_interval) != 0
            elif state.is_decode_phase and decode_policy == "dynamic_refresh":
                should_reuse_decode = False
            elif state.is_decode_phase and decode_policy == "reuse_prefill":
                should_reuse_decode = True

            if state.is_decode_phase and should_reuse_decode:
                next_layer_apia_blocks = slim_kv_cache.get_activate_prompt_block_ids(layer_idx + 1)
                if next_layer_apia_blocks is not None:
                    runtime_stats = layer_pruning_cfg.get("runtime_stats")
                    if isinstance(runtime_stats, dict):
                        runtime_stats["decode_reuse_calls"] = int(runtime_stats.get("decode_reuse_calls", 0)) + 1
                    state.prev_layer_was_pruned_flag = True
                    next_APIA_block_set = set(next_layer_apia_blocks)
                    next_APIA_set = from_blocks_to_abs_indices(next_APIA_block_set, state.original_prompt_len)
                    state.need_to_swap_kv_cache_flag = False
                    return next_APIA_set, next_APIA_block_set

            last_query_abs_idx = max(query_abs_indices_list)
            last_query_relative_idx = query_abs_to_relative_map[last_query_abs_idx]
            if not state.is_decode_phase:
                # Prefill
                new_q = q_squery
            else:
                # Decode
                assert(last_query_relative_idx == 0)
                new_q = q_squery[:, :, 0:1, :]
            slim_kv_cache.update_pruning_query(layer_idx, new_q)
            q_for_pruning = slim_kv_cache.pruning_query_cache[layer_idx]
            if q_for_pruning is None:
                raise KeyError("q_for_pruning is None!")

            active_prompt_block_ids = sorted(list(state.current_APIA_block_set))
            represent_keys = slim_kv_cache.get_represent_keys(layer_idx, active_prompt_block_ids)
            represent_values = slim_kv_cache.get_represent_values(layer_idx, active_prompt_block_ids)
            kept_block_ids = _calculate_pruning_with_rep_keys(
                q_for_pruning=q_for_pruning,
                represent_keys=represent_keys,
                original_block_ids=active_prompt_block_ids,
                cfg=layer_pruning_cfg,
                num_key_value_groups=num_key_value_groups,
                block_to_kernel_map=slim_kv_cache.represent_key_block_map[layer_idx],
                represent_values=represent_values,
            )
            
            next_APIA_block_set = set(kept_block_ids)
            state.prev_layer_was_pruned_flag = True

            if not state.is_decode_phase and next_APIA_block_set != state.current_APIA_block_set:
                if layer_idx + 1 < len(slim_kv_cache.prompt_key_cache) and slim_kv_cache.prompt_key_cache[layer_idx + 1] is not None:
                    state.need_to_swap_kv_cache_flag = True

            if state.is_decode_phase:
                next_layer_apia_blocks = slim_kv_cache.get_activate_prompt_block_ids(layer_idx + 1)
                if next_layer_apia_blocks is not None:
                    # calculate overlap ratio
                    overlap = len(next_APIA_block_set.intersection(next_layer_apia_blocks)) / len(next_APIA_block_set)
                    runtime_stats = layer_pruning_cfg.get("runtime_stats")
                    if isinstance(runtime_stats, dict):
                        runtime_stats["swap_overlap_sum"] = float(runtime_stats.get("swap_overlap_sum", 0.0)) + float(overlap)
                        runtime_stats["swap_overlap_count"] = int(runtime_stats.get("swap_overlap_count", 0)) + 1
                    if overlap < sp_utils._SWAP_THRESHOLD:
                        state.need_to_swap_kv_cache_flag = True
                        next_APIA_block_set = set(kept_block_ids)
                    else: 
                        # reuse next layer's APIA to avoid swapping
                        next_APIA_block_set = next_layer_apia_blocks
                        state.need_to_swap_kv_cache_flag = False
            
            next_APIA_set = from_blocks_to_abs_indices(next_APIA_block_set, state.original_prompt_len)

        return next_APIA_set, next_APIA_block_set


    def apply_pruning_and_cache_hs(
        self,
        layer_idx: int,
        state: SlimInferInferenceState,
        layer_output_hs: torch.Tensor,
        layer_inputs: Dict[str, Any],
        hs_cache: HsCache,
        kv_pool_abs_indices_list: List[int],
        next_APIA_set: Set[int],
        next_APIA_block_set: Set[int],
    ) -> None:
        """
        prune the hidden states after MLP according to the planned APIA set
        cache the pruned hidden states, and update the inference state accordingly
        """
        query_abs_indices_list = layer_inputs["query_abs_indices_list"]
        query_abs_to_relative_map = layer_inputs["query_abs_to_relative_map"]
        # 1. restore hidden states for pruned tokens
        pruned_indices_abs = state.current_APIA_set - next_APIA_set
        # prefill phase does not need to restore hidden states here
        if state.is_decode_phase and len(pruned_indices_abs) > 0:
            if state.current_attention_mask is not None and state.is_decode_phase and state.decoded_token_abs_idx is not None:
                next_kv_abs_indices_set = set(next_APIA_set)
                next_kv_abs_indices_set.update(range(state.original_prompt_len, state.decoded_token_abs_idx + 1))
                # prune attention mask if possible
                current_kv_pool_set = set(kv_pool_abs_indices_list)
                if not next_kv_abs_indices_set.issubset(current_kv_pool_set):
                    # set rebuild flag to True if not all next_kv tokens are in current kv pool
                    state.need_to_rebuild_attention_mask_flag = True
                    state.need_to_rebuild_position_embeddings_flag = True
                else:
                    key_keep_mask = [idx in next_kv_abs_indices_set for idx in kv_pool_abs_indices_list]
                    key_keep_mask = torch.tensor(key_keep_mask, device=state.device, dtype=torch.bool)
                    state.current_attention_mask = state.current_attention_mask[:, :, :, key_keep_mask]
                    state.need_to_rebuild_attention_mask_flag = False
                    
        pruned_indices_in_query = pruned_indices_abs.intersection(set(query_abs_indices_list))
        
        final_hs = layer_output_hs
        final_query_indices_list = query_abs_indices_list
        final_query_abs_to_relative_map = query_abs_to_relative_map

        # 2. cache pruned hidden states and update
        if pruned_indices_in_query:
            relative_indices_to_prune = [query_abs_to_relative_map[p_idx] for p_idx in pruned_indices_in_query]
            
            hs_to_cache = final_hs[:, relative_indices_to_prune, :].detach()
            pruned_block_ids = from_abs_indices_to_block_ids_list(pruned_indices_in_query)
            hs_cache.update(hs_to_cache, layer_idx, pruned_block_ids)
            state.query_block_id_list = prune_from_list(state.query_block_id_list, pruned_block_ids)
            
            # create a mask to keep unpruned hidden states
            keep_mask = torch.ones(final_hs.shape[1], dtype=torch.bool, device=final_hs.device)
            keep_mask[relative_indices_to_prune] = False
            
            final_hs = final_hs[:, keep_mask, :]
            final_query_indices_list = [idx for idx in query_abs_indices_list if idx not in pruned_indices_in_query]
            final_query_abs_to_relative_map = {idx: i for i, idx in enumerate(final_query_indices_list)}
            
            # prune attention mask and position embeddings
            if state.current_attention_mask is not None:
                if not state.is_decode_phase:
                    state.current_attention_mask = state.current_attention_mask[:, :, keep_mask, :][:, :, :, keep_mask]
                else:
                    state.current_attention_mask = state.current_attention_mask[:, :, keep_mask, :]

            if state.current_position_embeddings is not None:
                cos, sin = state.current_position_embeddings
                state.current_position_embeddings = (cos[:, keep_mask, :], sin[:, keep_mask, :])
        
        # 3. update state object for next layer
        state.current_APIA_set = next_APIA_set
        state.current_APIA_block_set = next_APIA_block_set
        state.hidden_states_prev_layer = final_hs
        state.hs_prev_mlp_output_abs_indices_list = final_query_indices_list
        state.prev_mlp_abs_to_relative_idx_map = final_query_abs_to_relative_map
        return
