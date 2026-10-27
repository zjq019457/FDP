import torch
from transformers.cache_utils import Cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, List, Any, Union, Set, Iterable, Sequence
from collections import Counter
import math
import threading
import json
import hashlib
import re
import time

from bridgekv import (
    apply_gumbel_noise,
    apply_rescue_policy,
    block_obd_saliency,
    combined_abc_score,
    compute_bridge_utilities,
    diffusion_gap_score,
    diffusion_gap_values,
    drm_s0_score,
    get_state as get_bridgekv_state,
    mmr_select_rescue_ids,
    normalize_scores,
    path_aware_score,
    pool_mmr_select_rescue_ids,
    query_drm_s0_score,
    query_evidence_score,
    redundancy_aware_score,
)
from bridgekv.structured_rescue_scoring import (
    append_path_marker,
    append_selection_dumps,
    build_structured_final_selection_row,
    enrich_structured_audit_fields,
    enforce_strict_structured_path_check,
    score_bridge_graph_srm,
    score_deferred_uncertainty_srm,
    score_distilled_blockdrm,
    score_evidence_density_srm,
    score_rh_pyramid_srm,
    select_structured_rescue_via_microblocks,
)
from bridgekv.confidence_gated_sliminfer import (
    confidence_gated_active_ids,
    compute_pruning_confidence,
    effective_swap_threshold,
)
from bridgekv.reliability_sliminfer import (
    adaptive_retain_tokens_k,
    compute_multi_window_block_scores,
    margin_stable_active_ids,
    rank_stability_sliminfer_active_ids,
    reliability_gated_active_ids,
    reliability_policy_from_confidence,
)
from bridgekv.sp_sliminfer import sp_sliminfer_active_ids
from bridgekv.sp_semantic_transport import capture_sp_v2_runtime_layer_state, sp_v2_active_ids_runtime
from bridgekv.structural_sliminfer import (
    cluster_continuity_active_ids,
    compute_unit_density_block_scores,
    divcov_active_ids,
)
from bridgekv.cc_topology_policies import (
    CC_TOPOLOGY_METHODS,
    dispatch_cc_topology_policy,
    load_cc_gate_config,
)
from bridgekv.acc_sliminfer import (
    ACC_DRM_METHODS,
    dispatch_acc_drm_policy,
)
from bridgekv.roleflow_kv import apply_roleflow_score_adjustment, select_roleflow_kv_ids

_STRUCTURED_RESCUE_SCORES = {
    "rh_pyramid_srm",
    "distilled_blockdrm",
    "bridge_graph_srm",
    "deferred_uncertainty_srm",
    "evidence_density_srm",
}

# --- Global Paging Constant ---
_BLOCK_SIZE = 64
_SWAP_THRESHOLD = 0.8
_PRUNING_WINDOW_LEN = 4
_SEMANTIC_KERNEL_SIZE = _BLOCK_SIZE // 8
_LSSR_POLICY_B_PLAN_CACHE: Dict[str, Dict[str, Any]] = {}
_LSSR_POLICY_B_RUNTIME_STATE: Dict[str, Dict[str, Any]] = {}

import heapq

def _tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())

def from_block_id_to_abs_index(block_id: int, max_len: int) -> List[int]:
    start_idx = block_id * _BLOCK_SIZE
    end_idx = min((block_id + 1) * _BLOCK_SIZE, max_len)
    return list(range(start_idx, end_idx))

def from_blocks_to_abs_indices(block_ids: Set[int], max_len: int) -> Set[int]:
    abs_indices = set()
    for block_id in block_ids:
        abs_indices.update(from_block_id_to_abs_index(block_id, max_len))
    return abs_indices

def from_blocks_to_abs_indices_list(block_ids: List[int], max_len: int) -> List[int]:
    abs_indices = []
    for block_id in block_ids:
        abs_indices.extend(from_block_id_to_abs_index(block_id, max_len))
    return abs_indices

def from_abs_indices_to_block_ids_list(abs_indices: Set[int]) -> List[int]:
    sorted_abs_indices = sorted(abs_indices)
    result_block_ids: List[int] = []
    last_processed_block_id = -1 
    for abs_idx in sorted_abs_indices:
        current_block_id = abs_idx // _BLOCK_SIZE
        if current_block_id != last_processed_block_id:
            result_block_ids.append(current_block_id)
            last_processed_block_id = current_block_id
    return result_block_ids

def from_abs_indices_to_block_ids(abs_indices: Set[int]) -> Set[int]:
    """将一组绝对索引转换为块ID集合"""
    block_ids = set()
    for abs_idx in abs_indices:
        block_id = abs_idx // _BLOCK_SIZE
        block_ids.add(block_id)
    return block_ids
    

def prune_from_list(current_list: List[int], 
                    indices_to_prune: List[int]) -> List[int]:
    indices_to_prune_set = set(indices_to_prune)
    return [item for item in current_list if item not in indices_to_prune_set]

class SlimInferCache(Cache):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.prompt_total_len: int = 0
        self.seq_len: int = 0
        self.async_supported = torch.cuda.is_available() and torch.device(device).type == "cuda"
        self.enable_async_swap = self.async_supported
        self.enable_cpu_offload = True
        self.io_stats: Dict[str, Union[int, bool]] = {
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "swap_in_blocks": 0,
            "swap_out_blocks": 0,
            "async": bool(self.enable_async_swap),
        }
        self.prompt_block_max_id: int = -1
        self.prompt_last_block_len: int = -1
        self.prompt_key_cache: List[Optional[torch.Tensor]] = []
        self.prompt_value_cache: List[Optional[torch.Tensor]] = []
        self.represent_keys_cache: List[Optional[torch.Tensor]] = []
        self.represent_values_cache: List[Optional[torch.Tensor]] = []
        self.pruning_query_cache: List[Optional[torch.Tensor]] = []
        
        # record layer-wise block ID to physical index mapping
        self.prompt_layer_block_map: List[Dict[int, int]] = [] 
        self.represent_key_block_map: List[Dict[int, Tuple[int, int]]] = []
        # record layer-wise cached block IDs in physical order
        self.prompt_layer_cached_block_ids: List[List[int]] = []
        self.active_prompt_block_ids_per_layer: List[Set[int]] = []
        
        #  --- RAM-resident Tensors (Offloaded Cache) ---
        self.offloaded_prompt_key_cache: List[Optional[torch.Tensor]] = []
        self.offloaded_prompt_value_cache: List[Optional[torch.Tensor]] = []
        self.offloaded_prompt_block_ids: List[List[int]] = []
        self.offloaded_layer_block_map: List[Dict[int, int]] = []
        self.swap_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream() if self.async_supported else None
        self.thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.d2d_events: List[Optional[torch.cuda.Event]] = [] 
        self.swap_out_events: List[Optional[torch.cuda.Event]] = []
        self.cpu_cat_events = [] 
        
        # Decode Token cache structures (single-token appends, no pruning, no block management)
        self.decode_key_cache: List[Optional[torch.Tensor]] = [] 
        self.decode_value_cache: List[Optional[torch.Tensor]] = []
        self.decode_sequence_length_per_layer: List[int] = []
        self.decode_base_abs_idx_per_layer: List[Optional[int]] = []
        self.debug_used_block_audit_path: str = ""
        self.debug_context: Dict[str, Any] = {}
        self.debug_used_block_max_records: int = 4096
        self.debug_used_block_include_full: bool = False
        self.debug_used_block_full_layers: Set[int] = set()
        self._debug_used_block_records_written: int = 0

    def _ensure_layer_capacity(self, layer_idx: int):
        while len(self.prompt_key_cache) <= layer_idx:
            self.prompt_key_cache.append(None)
            self.prompt_value_cache.append(None)
            self.represent_keys_cache.append(None)
            self.represent_values_cache.append(None)
            self.pruning_query_cache.append(None)
            self.prompt_layer_block_map.append({})
            self.represent_key_block_map.append({})
            self.prompt_layer_cached_block_ids.append([])
            self.active_prompt_block_ids_per_layer.append(set())
            
            self.offloaded_prompt_key_cache.append(None)
            self.offloaded_prompt_value_cache.append(None)
            self.offloaded_prompt_block_ids.append([])
            self.offloaded_layer_block_map.append({})
            self.d2d_events.append(torch.cuda.Event() if self.async_supported else None)
            self.swap_out_events.append(torch.cuda.Event() if self.async_supported else None)
            self.cpu_cat_events.append(threading.Event())
            
            self.decode_key_cache.append(None)
            self.decode_value_cache.append(None)
            self.decode_sequence_length_per_layer.append(0)
            self.decode_base_abs_idx_per_layer.append(None)
            
    def _get_prompt_block_len(self, block_id: int) -> int:
        if block_id == self.prompt_block_max_id:
            return self.prompt_last_block_len
        else:
            return _BLOCK_SIZE
        
    def update_pruning_query(self, layer_idx: int, new_q: torch.Tensor):
        self._ensure_layer_capacity(layer_idx)

        cached_q = self.pruning_query_cache[layer_idx]
        if new_q.shape[2] > _PRUNING_WINDOW_LEN:
            new_q = new_q[:, :, -_PRUNING_WINDOW_LEN:, :]
        
        if cached_q is None:
            updated_q = new_q
        else:
            updated_q = torch.cat([cached_q, new_q], dim=2)
        
        if updated_q.shape[2] > _PRUNING_WINDOW_LEN:
            updated_q = updated_q[:, :, -_PRUNING_WINDOW_LEN:, :]
        self.pruning_query_cache[layer_idx] = updated_q.detach()
    
    def update_runtime_options(self, enable_cpu_offload: bool = True, enable_async_kv: bool = True) -> None:
        self.enable_cpu_offload = bool(enable_cpu_offload)
        self.enable_async_swap = bool(enable_async_kv) and self.async_supported
        self.io_stats["async"] = bool(self.enable_async_swap)

    def update_represent_keys(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, block_ids: List[int]):
        """
        update new representative keys/values. If cache does not exist, create it; otherwise, append to existing tensor.
        """
        if not block_ids:
            return
        
        bsz, num_kv_heads, num_tokens, head_dim = key_states.shape
        pooled_kernels_for_update = []
        pooled_values_for_update = []
        last_block_len = self._get_prompt_block_len(block_ids[-1])
        num_blocks_in_update = len(block_ids)
        has_partial_block = (last_block_len != _BLOCK_SIZE)
        num_full_blocks = num_blocks_in_update - 1 if has_partial_block else num_blocks_in_update
        
        if num_full_blocks > 0:
            num_tokens_in_full_blocks = num_full_blocks * _BLOCK_SIZE
            full_blocks_tensor = key_states[:, :, :num_tokens_in_full_blocks, :]
            num_kernels_per_block = _BLOCK_SIZE // _SEMANTIC_KERNEL_SIZE
            pooled_full_kernels = (
                full_blocks_tensor.view(
                    bsz, num_kv_heads, num_full_blocks * num_kernels_per_block, _SEMANTIC_KERNEL_SIZE, head_dim
                )
                .mean(dim=3)
            )
            full_value_blocks_tensor = value_states[:, :, :num_tokens_in_full_blocks, :]
            pooled_full_values = (
                full_value_blocks_tensor.view(
                    bsz, num_kv_heads, num_full_blocks * num_kernels_per_block, _SEMANTIC_KERNEL_SIZE, head_dim
                )
                .mean(dim=3)
            )
            pooled_kernels_for_update.append(pooled_full_kernels)
            pooled_values_for_update.append(pooled_full_values)

        # --- 2. Process the last, partial block if it exists ---
        if has_partial_block:
            num_tokens_in_full_blocks = num_full_blocks * _BLOCK_SIZE
            partial_block_tensor = key_states[:, :, num_tokens_in_full_blocks:, :]
            kernel_tensors = torch.split(partial_block_tensor, _SEMANTIC_KERNEL_SIZE, dim=2)
            pooled_partial_kernels = [torch.mean(kernel, dim=2, keepdim=True) for kernel in kernel_tensors]
            pooled_kernels_for_update.append(torch.cat(pooled_partial_kernels, dim=2))
            partial_value_tensor = value_states[:, :, num_tokens_in_full_blocks:, :]
            value_tensors = torch.split(partial_value_tensor, _SEMANTIC_KERNEL_SIZE, dim=2)
            pooled_partial_values = [torch.mean(kernel, dim=2, keepdim=True) for kernel in value_tensors]
            pooled_values_for_update.append(torch.cat(pooled_partial_values, dim=2))

        # --- 3. Concatenate all new kernel representations and update cache ---
        new_kernels_tensor = torch.cat(pooled_kernels_for_update, dim=2).detach()
        new_values_tensor = torch.cat(pooled_values_for_update, dim=2).detach()

        current_total_kernels = 0
        if self.represent_keys_cache[layer_idx] is not None:
            current_total_kernels = self.represent_keys_cache[layer_idx].shape[2]
            self.represent_keys_cache[layer_idx] = torch.cat(
                [self.represent_keys_cache[layer_idx], new_kernels_tensor], dim=2
            )
            self.represent_values_cache[layer_idx] = torch.cat(
                [self.represent_values_cache[layer_idx], new_values_tensor], dim=2
            )
        else:
            self.represent_keys_cache[layer_idx] = new_kernels_tensor
            self.represent_values_cache[layer_idx] = new_values_tensor
        
        # --- 4. Update the block-to-kernel metadata map (this still requires a loop) ---
        num_kernels_per_full_block = _BLOCK_SIZE // _SEMANTIC_KERNEL_SIZE
        
        for i in range(num_full_blocks):
            block_id = block_ids[i]
            self.represent_key_block_map[layer_idx][block_id] = (current_total_kernels, num_kernels_per_full_block)
            current_total_kernels += num_kernels_per_full_block
            
        if has_partial_block:
            block_id = block_ids[-1]
            num_kernels_in_partial = math.ceil(last_block_len / _SEMANTIC_KERNEL_SIZE)
            self.represent_key_block_map[layer_idx][block_id] = (current_total_kernels, num_kernels_in_partial)
    
    def get_represent_keys(self, layer_idx: int, block_ids_to_get: List[int]) -> torch.Tensor:
        all_kernels_tensor = self.represent_keys_cache[layer_idx]            
        kernel_indices_to_select = []
        layer_kernel_map = self.represent_key_block_map[layer_idx]

        for b_id in block_ids_to_get:
            if b_id in layer_kernel_map:
                start_kernel_idx, num_kernels = layer_kernel_map[b_id]
                kernel_indices_to_select.extend(range(start_kernel_idx, start_kernel_idx + num_kernels))
            else:
                pass 
        
        if not kernel_indices_to_select:
            raise KeyError("indices empty")

        indices_tensor = torch.tensor(kernel_indices_to_select, device=self.device, dtype=torch.long)
        return torch.index_select(all_kernels_tensor, dim=2, index=indices_tensor)

    def get_represent_values(self, layer_idx: int, block_ids_to_get: List[int]) -> Optional[torch.Tensor]:
        all_values_tensor = self.represent_values_cache[layer_idx]
        if all_values_tensor is None:
            return None
        kernel_indices_to_select = []
        layer_kernel_map = self.represent_key_block_map[layer_idx]

        for b_id in block_ids_to_get:
            if b_id in layer_kernel_map:
                start_kernel_idx, num_kernels = layer_kernel_map[b_id]
                kernel_indices_to_select.extend(range(start_kernel_idx, start_kernel_idx + num_kernels))
        
        if not kernel_indices_to_select:
            return None

        indices_tensor = torch.tensor(kernel_indices_to_select, device=self.device, dtype=torch.long)
        return torch.index_select(all_values_tensor, dim=2, index=indices_tensor)
           
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        is_prompt_token: bool,
        abs_or_block_ids: Union[int, List[int]],
        prompt_seq_len: Optional[int] = None, 
    ) -> None:
        if key_states.shape[2] == 0:
            return

        self._ensure_layer_capacity(layer_idx)

        if is_prompt_token and isinstance(abs_or_block_ids, list):
            if prompt_seq_len is not None and self.prompt_total_len == 0:
                self.prompt_total_len = prompt_seq_len
                self.seq_len = prompt_seq_len
                self.prompt_block_max_id = (prompt_seq_len - 1) // _BLOCK_SIZE
                self.prompt_last_block_len = (prompt_seq_len - 1) % _BLOCK_SIZE + 1
            block_ids_to_add = abs_or_block_ids
            
            current_kv_len = self.prompt_key_cache[layer_idx].shape[2] if self.prompt_key_cache[layer_idx] is not None else 0
            self.prompt_key_cache[layer_idx] = torch.cat([self.prompt_key_cache[layer_idx], key_states], dim=-2) \
                if self.prompt_key_cache[layer_idx] is not None else key_states
            self.prompt_value_cache[layer_idx] = torch.cat([self.prompt_value_cache[layer_idx], value_states], dim=-2) \
                if self.prompt_value_cache[layer_idx] is not None else value_states

            processed_tokens = 0
            for block_id in block_ids_to_add:
                if block_id not in self.prompt_layer_block_map[layer_idx]:
                    block_len = self._get_prompt_block_len(block_id)
                    
                    self.prompt_layer_block_map[layer_idx][block_id] = current_kv_len + processed_tokens
                    self.prompt_layer_cached_block_ids[layer_idx].append(block_id)
                    processed_tokens += block_len
            self.active_prompt_block_ids_per_layer[layer_idx].update(block_ids_to_add)
            self.update_represent_keys(key_states, value_states, layer_idx, block_ids_to_add)
        else:
            decode_abs_idx = abs_or_block_ids

            current_k_cache = self.decode_key_cache[layer_idx]
            current_v_cache = self.decode_value_cache[layer_idx]

            self.decode_key_cache[layer_idx] = torch.cat([current_k_cache, key_states], dim=-2) if current_k_cache is not None else key_states
            self.decode_value_cache[layer_idx] = torch.cat([current_v_cache, value_states], dim=-2) if current_v_cache is not None else value_states
            
            if self.decode_base_abs_idx_per_layer[layer_idx] is None:
                self.decode_base_abs_idx_per_layer[layer_idx] = decode_abs_idx
            self.decode_sequence_length_per_layer[layer_idx] += 1
    
    def update_active_prompt_block_ids(self, layer_idx: int, new_active_blocks: Set[int]) -> None:
        self.active_prompt_block_ids_per_layer[layer_idx] = new_active_blocks
        
    def get_activate_prompt_block_ids(self, layer_idx: int) -> Optional[Set[int]]:
        if layer_idx >= len(self.active_prompt_block_ids_per_layer):
            return None
        return self.active_prompt_block_ids_per_layer[layer_idx]
    
    def kv_swap_out_operation(self, layer_idx, d2d_event, swap_out_event, cpu_cat_event, k_to_offload, v_to_offload):
        try:
            if self.swap_stream is None or d2d_event is None or swap_out_event is None:
                k_to_offload_cpu = k_to_offload.to("cpu")
                v_to_offload_cpu = v_to_offload.to("cpu")
                self.io_stats["d2h_bytes"] = int(self.io_stats["d2h_bytes"]) + _tensor_nbytes(k_to_offload) + _tensor_nbytes(v_to_offload)
            else:
                with torch.cuda.stream(self.swap_stream):
                    torch.cuda.current_stream().wait_event(d2d_event)
                    k_to_offload_cpu = k_to_offload.to("cpu", non_blocking=True)
                    v_to_offload_cpu = v_to_offload.to("cpu", non_blocking=True)
                    self.io_stats["d2h_bytes"] = int(self.io_stats["d2h_bytes"]) + _tensor_nbytes(k_to_offload) + _tensor_nbytes(v_to_offload)
                    swap_out_event.record()

                    torch.cuda.current_stream().synchronize()

                swap_out_event.wait()
            self.offloaded_prompt_key_cache[layer_idx] = torch.cat([self.offloaded_prompt_key_cache[layer_idx], k_to_offload_cpu], dim=-2) \
                if self.offloaded_prompt_key_cache[layer_idx] is not None else k_to_offload_cpu
            self.offloaded_prompt_value_cache[layer_idx] = torch.cat([self.offloaded_prompt_value_cache[layer_idx], v_to_offload_cpu], dim=-2) \
                if self.offloaded_prompt_value_cache[layer_idx] is not None else v_to_offload_cpu
            del k_to_offload, v_to_offload
        finally: 
            cpu_cat_event.set()
        
    def swap_prompt_cache(self, layer_idx: int, target_active_block_ids: Set[int]) -> None:
        self._ensure_layer_capacity(layer_idx)
        current_active_ids = self.active_prompt_block_ids_per_layer[layer_idx]
        if current_active_ids == target_active_block_ids:
            return
        if not self.enable_cpu_offload and target_active_block_ids - current_active_ids:
            raise RuntimeError("CPU offload is disabled, but a later layer requested prompt blocks that are no longer resident on GPU.")
        
        # 1. prepare block sets
        current_k = self.prompt_key_cache[layer_idx]
        current_v = self.prompt_value_cache[layer_idx]
        if self.prompt_key_cache[layer_idx] is None or self.prompt_key_cache[layer_idx] is None:
            raise KeyError("{layer_idx} prompt cache empty")
        
        current_cpu_block_set = set(self.offloaded_prompt_block_ids[layer_idx])
        current_gpu_block_set = set(self.prompt_layer_cached_block_ids[layer_idx])
        blocks_to_keep = current_gpu_block_set.intersection(target_active_block_ids)
        blocks_to_move_out = current_gpu_block_set - target_active_block_ids
        blocks_to_offload = blocks_to_move_out - current_cpu_block_set
        blocks_to_load_from_cpu = (target_active_block_ids - current_gpu_block_set).intersection(current_cpu_block_set)
        
        # --- 2. offload prompt tokens ---
        num_tokens_to_offload = len(blocks_to_offload) * _BLOCK_SIZE

        if num_tokens_to_offload > 0:
            offload_token_indices = []
            for block_id in blocks_to_offload:
                start_idx = self.prompt_layer_block_map[layer_idx][block_id]
                offload_token_indices.extend(range(start_idx, start_idx + _BLOCK_SIZE))
            
            if offload_token_indices:
                offload_indices_tensor = torch.tensor(offload_token_indices, device=self.device, dtype=torch.long)
                k_to_offload = torch.index_select(current_k, dim=-2, index=offload_indices_tensor)
                v_to_offload = torch.index_select(current_v, dim=-2, index=offload_indices_tensor)
            self.io_stats["swap_out_blocks"] = int(self.io_stats["swap_out_blocks"]) + len(blocks_to_offload)
            if self.enable_async_swap:
                k_clone = k_to_offload.clone()
                v_clone = v_to_offload.clone()
                d2d_event = self.d2d_events[layer_idx]
                if d2d_event is not None:
                    d2d_event.record()
                self.thread_pool.submit(
                    self.kv_swap_out_operation, 
                    layer_idx, 
                    d2d_event, 
                    self.swap_out_events[layer_idx], 
                    self.cpu_cat_events[layer_idx],
                    k_clone, v_clone
                )
            elif self.enable_cpu_offload:
                k_to_offload_cpu = k_to_offload.to("cpu")
                v_to_offload_cpu = v_to_offload.to("cpu")
                self.io_stats["d2h_bytes"] = int(self.io_stats["d2h_bytes"]) + _tensor_nbytes(k_to_offload) + _tensor_nbytes(v_to_offload)

                self.offloaded_prompt_key_cache[layer_idx] = torch.cat([self.offloaded_prompt_key_cache[layer_idx], k_to_offload_cpu], dim=-2) \
                    if self.offloaded_prompt_key_cache[layer_idx] is not None else k_to_offload_cpu
                self.offloaded_prompt_value_cache[layer_idx] = torch.cat([self.offloaded_prompt_value_cache[layer_idx], v_to_offload_cpu], dim=-2) \
                    if self.offloaded_prompt_value_cache[layer_idx] is not None else v_to_offload_cpu
            
            current_cpu_offset = len(self.offloaded_prompt_block_ids[layer_idx]) * _BLOCK_SIZE
            for block_id in blocks_to_offload:
                self.offloaded_layer_block_map[layer_idx][block_id] = current_cpu_offset
                current_cpu_offset += _BLOCK_SIZE
            self.offloaded_prompt_block_ids[layer_idx].extend(blocks_to_offload)
            
        # get the kept part in VRAM
        keep_token_indices = []
        new_physical_block_order = sorted(list(blocks_to_keep))
        for block_id in new_physical_block_order:
            start_idx = self.prompt_layer_block_map[layer_idx][block_id]
            block_len = self._get_prompt_block_len(block_id)
            keep_token_indices.extend(range(start_idx, start_idx + block_len))
        
        keep_indices_tensor = torch.tensor(keep_token_indices, device=self.device, dtype=torch.long)
        k_kept_in_vram = torch.index_select(current_k, dim=-2, index=keep_indices_tensor)
        v_kept_in_vram = torch.index_select(current_v, dim=-2, index=keep_indices_tensor)
        
        del current_k, current_v
        
        # 3. load from CPU
        loaded_block_ids_ordered = sorted(list(blocks_to_load_from_cpu))
        k_loaded_from_cpu, v_loaded_from_cpu = None, None
        if blocks_to_load_from_cpu:
            load_token_indices_cpu = []
            for block_id in blocks_to_load_from_cpu:
                if block_id not in self.offloaded_layer_block_map[layer_idx]:
                     raise KeyError(f"L{layer_idx}: Attempted to load block {block_id}, but it was not found in the offloaded cache map.")
                start_idx = self.offloaded_layer_block_map[layer_idx][block_id]
                load_token_indices_cpu.extend(range(start_idx, start_idx + _BLOCK_SIZE))
            
            if self.enable_async_swap:
                self.cpu_cat_events[layer_idx].wait()
            indices_tensor_cpu = torch.tensor(load_token_indices_cpu, device="cpu", dtype=torch.long)
            k_loaded_from_cpu = torch.index_select(self.offloaded_prompt_key_cache[layer_idx], dim=-2, index=indices_tensor_cpu).to(self.device, non_blocking=True)
            v_loaded_from_cpu = torch.index_select(self.offloaded_prompt_value_cache[layer_idx], dim=-2, index=indices_tensor_cpu).to(self.device, non_blocking=True)
            self.io_stats["h2d_bytes"] = int(self.io_stats["h2d_bytes"]) + _tensor_nbytes(k_loaded_from_cpu) + _tensor_nbytes(v_loaded_from_cpu)
            self.io_stats["swap_in_blocks"] = int(self.io_stats["swap_in_blocks"]) + len(blocks_to_load_from_cpu)
        
        # 4. concatenate kept and loaded parts
        if k_loaded_from_cpu is not None and v_loaded_from_cpu is not None:
            self.prompt_key_cache[layer_idx] = torch.cat([k_kept_in_vram, k_loaded_from_cpu], dim=-2)
            self.prompt_value_cache[layer_idx] = torch.cat([v_kept_in_vram, v_loaded_from_cpu], dim=-2)
        else:
            self.prompt_key_cache[layer_idx] = k_kept_in_vram
            self.prompt_value_cache[layer_idx] = v_kept_in_vram
        
        # 5. update metadata
        final_physical_order = new_physical_block_order + loaded_block_ids_ordered
        self.prompt_layer_cached_block_ids[layer_idx] = final_physical_order
        self.active_prompt_block_ids_per_layer[layer_idx] = set(final_physical_order)
        # rebuild block to physical index map
        new_block_map = {}
        current_offset = 0
        for block_id in final_physical_order:
            block_len = self._get_prompt_block_len(block_id)
            new_block_map[block_id] = current_offset
            current_offset += block_len
        self.prompt_layer_block_map[layer_idx] = new_block_map
    
    def get_kv_pool_for_attention(
        self, 
        layer_idx: int, 
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[int]]:
        
        self._ensure_layer_capacity(layer_idx)

        combined_k = None
        combined_v = None
        final_kv_abs_indices = []

        prompt_k_full = self.prompt_key_cache[layer_idx]
        prompt_v_full = self.prompt_value_cache[layer_idx]
        prompt_cached_block_ids_ordered = self.prompt_layer_cached_block_ids[layer_idx]
        
        decode_k = self.decode_key_cache[layer_idx]
        decode_v = self.decode_value_cache[layer_idx]
        decode_base_abs_idx = self.decode_base_abs_idx_per_layer[layer_idx]
        
        # Preifll phase
        if decode_base_abs_idx is None:
            prompt_abs_indices = from_blocks_to_abs_indices_list(prompt_cached_block_ids_ordered, self.prompt_total_len)
            self._write_used_block_audit(layer_idx, "prefill", prompt_cached_block_ids_ordered, prompt_abs_indices)
            return prompt_k_full, prompt_v_full, prompt_abs_indices
        
        # Decode phase
        if prompt_k_full is not None and prompt_v_full is not None:
            combined_k = prompt_k_full
            combined_v = prompt_v_full
            final_kv_abs_indices.extend(from_blocks_to_abs_indices_list(prompt_cached_block_ids_ordered, self.prompt_total_len))
            
        if decode_k is not None and self.decode_sequence_length_per_layer[layer_idx] > 0:
            base_idx = self.decode_base_abs_idx_per_layer[layer_idx]
            seq_len = self.decode_sequence_length_per_layer[layer_idx]
            current_decode_abs_indices = list(range(base_idx, base_idx + seq_len))

            combined_k = torch.cat([combined_k, decode_k], dim=-2)
            combined_v = torch.cat([combined_v, decode_v], dim=-2)
            final_kv_abs_indices.extend(current_decode_abs_indices)
            
        self._write_used_block_audit(layer_idx, "decode", prompt_cached_block_ids_ordered, final_kv_abs_indices)
        return combined_k, combined_v, final_kv_abs_indices

    def _write_used_block_audit(
        self,
        layer_idx: int,
        phase: str,
        prompt_block_ids: List[int],
        kv_abs_indices: List[int],
    ) -> None:
        audit_path = getattr(self, "debug_used_block_audit_path", "") or ""
        if not audit_path:
            return
        if self._debug_used_block_records_written >= int(getattr(self, "debug_used_block_max_records", 4096)):
            return
        context = getattr(self, "debug_context", {}) if isinstance(getattr(self, "debug_context", {}), dict) else {}
        record = {
            "task": context.get("task", ""),
            "sample_id": context.get("sample_id", ""),
            "sample_index": context.get("sample_index", ""),
            "method": context.get("method", ""),
            "rescue_policy": context.get("rescue_policy", ""),
            "rescue_score": context.get("rescue_score", ""),
            "seed": context.get("seed", ""),
            "layer_id": int(layer_idx),
            "phase": phase,
            "final_block_count_used_by_attention": int(len(prompt_block_ids)),
            "used_block_ids_first100": [int(x) for x in list(prompt_block_ids)[:100]],
            "kv_abs_indices_count": int(len(kv_abs_indices)),
            "kv_abs_indices_first100": [int(x) for x in list(kv_abs_indices)[:100]],
            "prompt_token_count": int(self.prompt_total_len),
            "prompt_block_count": int(math.ceil(self.prompt_total_len / _BLOCK_SIZE)),
        }
        full_layers = set(getattr(self, "debug_used_block_full_layers", set()) or set())
        include_full = (
            bool(getattr(self, "debug_used_block_include_full", False))
            and phase == "prefill"
            and (not full_layers or int(layer_idx) in full_layers)
        )
        if include_full:
            record["used_block_ids"] = [int(x) for x in prompt_block_ids]
            record["used_tokens"] = [int(x) for x in kv_abs_indices]
        try:
            with open(str(audit_path), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._debug_used_block_records_written += 1
        except Exception:
            pass
    

    def has_block_key_value(self, layer_idx: int, block_idx_abs: int) -> bool:
        if layer_idx < len(self.prompt_layer_block_map):
            return block_idx_abs in self.prompt_layer_block_map[layer_idx]
        return False

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.seq_len
    
    def plus_seq_length(self) -> None:
        self.seq_len += 1

    def get_max_length(self) -> Optional[int]:
        return None

    @classmethod
    def from_legacy_cache(cls) -> "SlimInferCache":
        import warnings
        warnings.warn("return an empty SlimInferCache instance.")
        cache = cls()
        return cache

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise KeyError(f"layer {layer_idx} has no data in cache or not fully initialized for this access way. Please use get_kv_for_layer.")

    def __len__(self):
        return len(self.prompt_key_cache)

    def clear(self):
        self.prompt_key_cache.clear()
        self.prompt_value_cache.clear()
        self.represent_keys_cache.clear()
        self.represent_values_cache.clear()
        self.pruning_query_cache.clear()
        self.represent_key_block_map.clear()
        self.prompt_layer_block_map.clear()
        self.prompt_layer_cached_block_ids.clear()
        self.active_prompt_block_ids_per_layer.clear()
        self.offloaded_layer_block_map.clear()
        self.offloaded_prompt_key_cache.clear()
        self.offloaded_prompt_value_cache.clear()
        self.offloaded_prompt_block_ids.clear()
        self.decode_key_cache.clear()
        self.decode_value_cache.clear()
        self.decode_sequence_length_per_layer.clear()
        self.decode_base_abs_idx_per_layer.clear()
        print("SlimInferCache has been cleared.")


class HsCache:
    def __init__(self) -> None:
        super().__init__()
        self.hidden_states_cache: List[Optional[torch.Tensor]] = []
        self.layer_block_map: List[Dict[int, int]] = [] # block_id -> start_idx_in_layer_tensor
        self.layer_cached_block_ids: List[List[int]] = []
    
    def _ensure_layer_capacity(self, layer_idx: int):
        while len(self.hidden_states_cache) <= layer_idx:
            self.hidden_states_cache.append(None)
            self.layer_block_map.append({})
            self.layer_cached_block_ids.append([])
            
    def update(
        self,
        hidden_states_blocks: torch.Tensor, # [bsz, num_blocks * _BLOCK_SIZE, hidden_dim]
        layer_idx: int,
        block_ids: List[int],
    ) -> None:
        if hidden_states_blocks.shape[1] == 0:
            return

        self._ensure_layer_capacity(layer_idx)
        current_hs_cache = self.hidden_states_cache[layer_idx]
        self.hidden_states_cache[layer_idx] = torch.cat([current_hs_cache, hidden_states_blocks], dim=1) \
                if current_hs_cache is not None else hidden_states_blocks

        current_len = current_hs_cache.shape[1] if current_hs_cache is not None else 0
        for i, block_id in enumerate(block_ids):
            if block_id not in self.layer_block_map[layer_idx]:
                self.layer_block_map[layer_idx][block_id] = current_len + i * _BLOCK_SIZE
                self.layer_cached_block_ids[layer_idx].append(block_id)

    def get_hidden_states(self, layer_idx: int, target_block_ids: List[int]) -> Optional[torch.Tensor]:
        if not target_block_ids:
            return None

        self._ensure_layer_capacity(layer_idx)
        if self.hidden_states_cache[layer_idx] is None or not self.layer_cached_block_ids[layer_idx]:
            return None
        
        full_hs_cache = self.hidden_states_cache[layer_idx]
        indices_to_select_relative = []
        
        # create temporary block_id -> (start_idx, end_idx) mapping
        temp_block_ranges = {}
        for block_id_in_cache in self.layer_cached_block_ids[layer_idx]:
            if block_id_in_cache in target_block_ids:
                start_idx = self.layer_block_map[layer_idx][block_id_in_cache]
                temp_block_ranges[block_id_in_cache] = (start_idx, start_idx + _BLOCK_SIZE)
        
        for block_id in target_block_ids:
            if block_id in temp_block_ranges:
                start_idx, end_idx = temp_block_ranges[block_id]
                indices_to_select_relative.extend(range(start_idx, end_idx))

        indices_tensor = torch.tensor(indices_to_select_relative, device=full_hs_cache.device, dtype=torch.long)
        return torch.index_select(full_hs_cache, dim=1, index=indices_tensor)

    def has_hidden_state_block(self, layer_idx: int, block_id: int) -> bool:
        self._ensure_layer_capacity(layer_idx)
        return block_id in self.layer_block_map[layer_idx]

    def clear(self):
        self.hidden_states_cache.clear()
        self.layer_block_map.clear()
        self.layer_cached_block_ids.clear()
        print("HsCache has been cleared.")
        
    def __len__(self):
        return len(self.hidden_states_cache)

from transformers.models.llama.modeling_llama import repeat_kv

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _calibration_entry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    calib = cfg.get("calibrated_budget_config")
    if not isinstance(calib, dict):
        return {}
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    method = str(context.get("method", cfg.get("variant_method", "")) or "")
    task = str(context.get("task", "") or "")
    methods = calib.get("methods", {}) if isinstance(calib.get("methods"), dict) else {}
    method_entry = methods.get(method, {}) if method else {}
    if not method_entry and method and isinstance(calib.get(method), dict):
        method_entry = calib.get(method, {})
    if not isinstance(method_entry, dict):
        return {}
    tasks = method_entry.get("tasks", {}) if isinstance(method_entry.get("tasks"), dict) else {}
    task_entry = tasks.get(task, {}) if task else {}
    if isinstance(task_entry, dict) and task_entry:
        merged = dict(method_entry)
        merged.pop("tasks", None)
        merged.update(task_entry)
        return merged
    return {k: v for k, v in method_entry.items() if k != "tasks"}

def _calibrated_budget(
    cfg: Dict[str, Any],
    kind: str,
    target: int,
    *,
    min_reachable: int = 0,
    max_reachable: int = 0,
) -> int:
    target = max(0, int(target or 0))
    entry = _calibration_entry(cfg)
    if not entry or target <= 0:
        return target
    calibration_target = max(1, int(round(_safe_float(entry.get("target_tokens", target), target))))
    if kind == "active":
        override = entry.get("controller_target_active_tokens")
        scale = _safe_float(entry.get("active_budget_scale", entry.get("budget_scale", 1.0)), 1.0)
    elif kind == "final":
        override = entry.get("controller_target_final_tokens")
        scale = _safe_float(entry.get("final_budget_scale", entry.get("budget_scale", 1.0)), 1.0)
    else:
        override = entry.get("controller_target_token_budget")
        scale = _safe_float(entry.get("budget_scale", 1.0), 1.0)
    controller_target = 0
    if override not in (None, ""):
        try:
            controller_target = max(1, int(round(float(override))))
        except Exception:
            pass
    if controller_target <= 0:
        controller_target = max(1, int(round(calibration_target * scale)))

    uncapped_target = int(controller_target)
    cap_reasons: List[str] = []
    cap_to_reachable = bool(entry.get("cap_to_reachable_tokens", False))
    if cap_to_reachable and min_reachable > 0 and controller_target < int(min_reachable):
        controller_target = int(min_reachable)
        cap_reasons.append("protected_token_floor")
    if cap_to_reachable and max_reachable > 0 and controller_target > int(max_reachable):
        controller_target = int(max_reachable)
        cap_reasons.append("candidate_token_ceiling")

    cfg[f"_requested_target_{kind}_tokens"] = int(target)
    cfg[f"_calibration_target_{kind}_tokens"] = int(calibration_target)
    cfg[f"_uncapped_controller_target_{kind}_tokens"] = int(uncapped_target)
    cfg[f"_min_reachable_{kind}_tokens"] = int(min_reachable)
    cfg[f"_max_reachable_{kind}_tokens"] = int(max_reachable)
    cfg[f"_target_cap_reason_{kind}"] = ",".join(cap_reasons)
    return max(1, int(controller_target))

def _calibration_status(cfg: Dict[str, Any]) -> Tuple[bool, bool]:
    loaded = isinstance(cfg.get("calibrated_budget_config"), dict)
    return loaded, bool(_calibration_entry(cfg))

def _layout_pages_for_span(start: int, end: int, prompt_len: int) -> List[int]:
    if end <= start:
        return []
    first = max(0, int(start)) // _BLOCK_SIZE
    last = max(0, int(end) - 1) // _BLOCK_SIZE
    max_page = max(0, (max(1, int(prompt_len)) - 1) // _BLOCK_SIZE)
    return [int(x) for x in range(first, min(last, max_page) + 1)]

def _span_for_pages(pages: List[int], prompt_len: int) -> Tuple[int, int]:
    if not pages:
        return (0, 0)
    start = max(0, min(int(p) for p in pages) * _BLOCK_SIZE)
    end = min(int(prompt_len), (max(int(p) for p in pages) + 1) * _BLOCK_SIZE)
    return start, end

def _block_length_stats(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    tensor = torch.tensor([float(x) for x in lengths], dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }

def _virtual_layout_from_precomputed(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prompt_len = int(cfg.get("original_prompt_len", cfg.get("audit_context", {}).get("context_length", 0)) or 0)
    layout = cfg.get("block_layout") if isinstance(cfg.get("block_layout"), dict) else {}
    active_pages = {int(x) for x in original_block_ids}
    spans = layout.get("token_spans") or []
    mapping = layout.get("mapping_to_fixed_pages") or []
    lengths = layout.get("block_lengths") or []
    semantic_tokens = layout.get("semantic_span_tokens") or lengths
    expanded_tokens = layout.get("expanded_page_tokens") or []
    overhead = layout.get("page_expansion_overhead") or []
    boundary_conf = layout.get("boundary_confidence") or []
    boundary_left = layout.get("boundary_score_left") or []
    boundary_right = layout.get("boundary_score_right") or []
    char_spans = layout.get("char_spans") or []
    previews = layout.get("block_text_preview") or []
    sources = layout.get("source_types") or []
    meta = dict(layout.get("metadata") or {})
    for key in [
        "fallback_used",
        "fallback_reason",
        "fallback_reason_counts",
        "coverage_ok",
        "no_gap",
        "no_overlap",
        "ordered_spans",
    ]:
        if key in layout:
            meta[key] = layout.get(key)
    virtual_blocks: List[Dict[str, Any]] = []
    for idx, raw_span in enumerate(spans):
        if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
            continue
        start, end = int(raw_span[0]), int(raw_span[1])
        pages = [int(x) for x in (mapping[idx] if idx < len(mapping) else _layout_pages_for_span(start, end, prompt_len))]
        active = sorted(active_pages.intersection(pages))
        if not active:
            continue
        virtual_blocks.append({
            "virtual_id": len(virtual_blocks),
            "source_layout_id": int(idx),
            "token_start": start,
            "token_end": end,
            "token_len": int(lengths[idx]) if idx < len(lengths) else max(0, end - start),
            "source_type": str(sources[idx]) if idx < len(sources) else str(layout.get("strategy_name", "")),
            "text_preview": str(previews[idx])[:160] if idx < len(previews) else "",
            "fixed_page_ids": active,
            "all_fixed_page_ids": pages,
            "char_span": char_spans[idx] if idx < len(char_spans) else [0, 0],
            "semantic_span_tokens": int(semantic_tokens[idx]) if idx < len(semantic_tokens) else max(0, end - start),
            "expanded_page_tokens": int(expanded_tokens[idx]) if idx < len(expanded_tokens) else int(len(pages) * _BLOCK_SIZE),
            "page_expansion_overhead": float(overhead[idx]) if idx < len(overhead) else float((len(pages) * _BLOCK_SIZE) / max(1, end - start)),
            "boundary_confidence": float(boundary_conf[idx]) if idx < len(boundary_conf) else 0.0,
            "boundary_score_left": float(boundary_left[idx]) if idx < len(boundary_left) else 0.0,
            "boundary_score_right": float(boundary_right[idx]) if idx < len(boundary_right) else 0.0,
        })
    meta.setdefault("fallback_used", False)
    meta.setdefault("fallback_reason", "")
    meta.setdefault("fallback_reason_counts", {})
    meta.setdefault("coverage_ok", True)
    meta.setdefault("no_gap", True)
    meta.setdefault("no_overlap", True)
    meta.setdefault("ordered_spans", True)
    return virtual_blocks, meta

def _fixed_virtual_blocks(original_block_ids: List[int], prompt_len: int, source_type: str = "fixed") -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for page in original_block_ids:
        start, end = _span_for_pages([int(page)], prompt_len)
        blocks.append({
            "virtual_id": len(blocks),
            "source_layout_id": int(page),
            "token_start": start,
            "token_end": end,
            "token_len": max(0, end - start),
            "source_type": source_type,
            "text_preview": "",
            "fixed_page_ids": [int(page)],
            "all_fixed_page_ids": [int(page)],
            "char_span": [0, 0],
            "semantic_span_tokens": max(0, end - start),
            "expanded_page_tokens": _BLOCK_SIZE,
            "page_expansion_overhead": float(_BLOCK_SIZE / max(1, end - start)),
            "boundary_confidence": 0.0,
            "boundary_score_left": 0.0,
            "boundary_score_right": 0.0,
        })
    return blocks

def _page_representations_global(
    represent_values: Optional[torch.Tensor],
    num_key_value_groups: int,
    block_slices: List[Tuple[int, int]],
    original_block_ids: List[int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    return _block_value_representations_global(
        represent_values=represent_values,
        num_key_value_groups=num_key_value_groups,
        block_slices=block_slices,
        original_block_ids=original_block_ids,
        device=device,
    )

def _attention_boundary_virtual_blocks(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    page_reps: Optional[torch.Tensor],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prompt_len = int(cfg.get("original_prompt_len", cfg.get("audit_context", {}).get("context_length", 0)) or 0)
    pages = [int(x) for x in original_block_ids]
    if not pages:
        return [], {"fallback_used": True, "fallback_reason": "no_active_fixed_pages"}
    if page_reps is None or page_reps.numel() == 0:
        return _fixed_virtual_blocks(pages, prompt_len, "fallback_fixed"), {
            "fallback_used": True,
            "fallback_reason": "missing_key_or_hidden_representations",
        }
    distances: List[float] = []
    for left, right in zip(pages[:-1], pages[1:]):
        if left >= page_reps.shape[0] or right >= page_reps.shape[0]:
            distances.append(0.0)
            continue
        l = torch.nn.functional.normalize(page_reps[left].float().reshape(1, -1), dim=-1, eps=1e-6)
        r = torch.nn.functional.normalize(page_reps[right].float().reshape(1, -1), dim=-1, eps=1e-6)
        distances.append(float((1.0 - torch.nn.functional.cosine_similarity(l, r, dim=-1)).item()))
    if distances:
        dist_t = torch.tensor(distances, dtype=torch.float32)
        percentile = max(0.0, min(100.0, _safe_float(cfg.get("boundary_percentile", 85.0), 85.0))) / 100.0
        threshold = float(torch.quantile(dist_t, percentile).item())
    else:
        threshold = float("inf")
    min_pages = max(1, int(math.floor(_safe_float(cfg.get("min_block_tokens", 32), 32) / _BLOCK_SIZE)))
    max_pages = max(1, int(math.ceil(_safe_float(cfg.get("max_block_tokens", 128), 128) / _BLOCK_SIZE)))
    groups: List[List[int]] = []
    cur = [pages[0]]
    for pos, page in enumerate(pages[1:]):
        boundary = distances[pos] >= threshold if pos < len(distances) else False
        if len(cur) >= max_pages or (boundary and len(cur) >= min_pages):
            groups.append(cur)
            cur = [page]
        else:
            cur.append(page)
    if cur:
        groups.append(cur)
    page_pos = {int(page): idx for idx, page in enumerate(pages)}
    blocks = []
    for group in groups:
        start, end = _span_for_pages(group, prompt_len)
        first_pos = page_pos.get(int(group[0]), 0)
        last_pos = page_pos.get(int(group[-1]), 0)
        left_score = float(distances[first_pos - 1]) if first_pos > 0 and first_pos - 1 < len(distances) else 0.0
        right_score = float(distances[last_pos]) if last_pos < len(distances) else 0.0
        confidence = (left_score + right_score) / 2.0 if (left_score or right_score) else 0.0
        blocks.append({
            "virtual_id": len(blocks),
            "source_layout_id": len(blocks),
            "token_start": start,
            "token_end": end,
            "token_len": max(0, end - start),
            "source_type": "attention_boundary",
            "text_preview": "",
            "fixed_page_ids": [int(x) for x in group],
            "all_fixed_page_ids": [int(x) for x in group],
            "char_span": [0, 0],
            "semantic_span_tokens": max(0, end - start),
            "expanded_page_tokens": int(len(group) * _BLOCK_SIZE),
            "page_expansion_overhead": float((len(group) * _BLOCK_SIZE) / max(1, end - start)),
            "boundary_confidence": float(confidence),
            "boundary_score_left": float(left_score),
            "boundary_score_right": float(right_score),
        })
    meta = {
        "fallback_used": False,
        "fallback_reason": "",
        "representation": "pooled_fixed_page_value_reps",
        "boundary_percentile": _safe_float(cfg.get("boundary_percentile", 85.0), 85.0),
        "boundary_score_mean": float(torch.tensor(distances).mean().item()) if distances else 0.0,
        "boundary_score_std": float(torch.tensor(distances).std(unbiased=False).item()) if len(distances) > 1 else 0.0,
        "boundary_confidence_mean": float(torch.tensor([float(b.get("boundary_confidence", 0.0)) for b in blocks]).mean().item()) if blocks else 0.0,
        "boundary_confidence_std": float(torch.tensor([float(b.get("boundary_confidence", 0.0)) for b in blocks]).std(unbiased=False).item()) if len(blocks) > 1 else 0.0,
        "number_of_blocks": int(len(blocks)),
    }
    return blocks, meta

def _hierarchical_virtual_blocks(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    page_scores: torch.Tensor,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prompt_len = int(cfg.get("original_prompt_len", cfg.get("audit_context", {}).get("context_length", 0)) or 0)
    pages = [int(x) for x in original_block_ids]
    if not pages:
        return [], {"fallback_used": True, "fallback_reason": "no_active_fixed_pages"}
    coarse_pages = max(1, int(round(_safe_float(cfg.get("coarse_block_tokens", 256), 256) / _BLOCK_SIZE)))
    fine_pages = max(1, int(round(_safe_float(cfg.get("fine_block_tokens", 64), 64) / _BLOCK_SIZE)))
    medium_pages = max(1, int(round(_safe_float(cfg.get("medium_block_tokens", 128), 128) / _BLOCK_SIZE)))
    coarse_groups = [pages[i:i + coarse_pages] for i in range(0, len(pages), coarse_pages)]
    local_score = {int(page): float(page_scores[pos]) for pos, page in enumerate(pages[: int(page_scores.numel())])}
    coarse_scores = [max(local_score.get(int(p), 0.0) for p in group) for group in coarse_groups]
    top_n = max(1, int(math.ceil(len(coarse_groups) * _safe_float(cfg.get("top_coarse_ratio", 0.3), 0.3))))
    top_coarse = set(sorted(range(len(coarse_groups)), key=lambda i: coarse_scores[i], reverse=True)[:top_n])
    blocks: List[Dict[str, Any]] = []
    fine_count = 0
    medium_count = 0
    for idx, group in enumerate(coarse_groups):
        chunk_size = fine_pages if idx in top_coarse else medium_pages
        source = "hierarchical_fine" if idx in top_coarse else "hierarchical_medium"
        for start_i in range(0, len(group), chunk_size):
            sub = group[start_i:start_i + chunk_size]
            start, end = _span_for_pages(sub, prompt_len)
            blocks.append({
                "virtual_id": len(blocks),
                "source_layout_id": len(blocks),
                "token_start": start,
                "token_end": end,
                "token_len": max(0, end - start),
                "source_type": source,
                "text_preview": "",
                "fixed_page_ids": [int(x) for x in sub],
                "all_fixed_page_ids": [int(x) for x in sub],
                "char_span": [0, 0],
                "semantic_span_tokens": max(0, end - start),
                "expanded_page_tokens": int(len(sub) * _BLOCK_SIZE),
                "page_expansion_overhead": float((len(sub) * _BLOCK_SIZE) / max(1, end - start)),
                "boundary_confidence": 0.0,
                "boundary_score_left": 0.0,
                "boundary_score_right": 0.0,
            })
            if idx in top_coarse:
                fine_count += 1
            else:
                medium_count += 1
    meta = {
        "fallback_used": False,
        "fallback_reason": "",
        "coarse_block_count": int(len(coarse_groups)),
        "fine_block_count": int(fine_count),
        "medium_block_count": int(medium_count),
        "top_coarse_regions": [int(x) for x in sorted(top_coarse)[:64]],
        "final_block_count": int(len(blocks)),
        "coarse_block_tokens": int(coarse_pages * _BLOCK_SIZE),
        "fine_block_tokens": int(fine_pages * _BLOCK_SIZE),
        "medium_block_tokens": int(medium_pages * _BLOCK_SIZE),
        "top_coarse_ratio": _safe_float(cfg.get("top_coarse_ratio", 0.3), 0.3),
    }
    return blocks, meta

def _virtualize_layout(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    block_scores: torch.Tensor,
    page_reps: Optional[torch.Tensor],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    strategy = str(cfg.get("block_strategy", "fixed") or "fixed")
    prompt_len = int(cfg.get("original_prompt_len", cfg.get("audit_context", {}).get("context_length", 0)) or 0)
    if strategy in {"semantic", "semantic_v2", "semantic_v3", "query_dynamic", "query_dynamic_v2", "query_dynamic_v3"}:
        blocks, meta = _virtual_layout_from_precomputed(cfg, original_block_ids)
        if blocks:
            return blocks, meta
        return _fixed_virtual_blocks(original_block_ids, prompt_len, "fallback_fixed"), {
            "fallback_used": True,
            "fallback_reason": "missing_or_empty_precomputed_block_layout",
        }
    if strategy == "attention_boundary":
        return _attention_boundary_virtual_blocks(cfg, original_block_ids, page_reps)
    if strategy == "hierarchical":
        return _hierarchical_virtual_blocks(cfg, original_block_ids, block_scores.detach().float())
    if strategy in {"hybrid_attention_fixed", "hybrid_attention_semantic"}:
        return _attention_boundary_virtual_blocks(cfg, original_block_ids, page_reps)
    if strategy == "fixed_active_rescue":
        # A-F diagnostic: active set stays the fixed 64-token pages (unchanged from
        # fixed SlimInfer). Only the rescue candidate pool varies.
        return _fixed_virtual_blocks(original_block_ids, prompt_len, "fixed"), {"fallback_used": False, "fallback_reason": ""}
    return _fixed_virtual_blocks(original_block_ids, prompt_len, "fixed"), {"fallback_used": False, "fallback_reason": ""}

def _aggregate_virtual_scores(
    virtual_blocks: List[Dict[str, Any]],
    original_block_ids: List[int],
    scores: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    out = torch.zeros(len(virtual_blocks), dtype=torch.float32, device=device)
    if scores is None or scores.numel() == 0:
        return out
    local_idx = {int(block_id): idx for idx, block_id in enumerate(original_block_ids)}
    for vid, block in enumerate(virtual_blocks):
        vals = []
        for page in block.get("fixed_page_ids", []):
            idx = local_idx.get(int(page))
            if idx is not None and idx < scores.numel():
                vals.append(scores[idx].detach().float())
        if vals:
            out[vid] = torch.stack(vals).max()
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _head_page_mass_from_kernel_scores(
    kernel_scores: torch.Tensor,
    block_slices: List[Tuple[int, int]],
) -> Optional[torch.Tensor]:
    """Collapse semantic kernels but retain attention-head identity.

    Returns a [num_heads, num_pages] softmax mass tensor in the local
    ``original_block_ids`` order.
    """
    if kernel_scores is None or kernel_scores.numel() == 0 or not block_slices:
        return None
    per_page: List[torch.Tensor] = []
    for start, num_kernels in block_slices:
        if num_kernels <= 0:
            per_page.append(torch.full(
                (kernel_scores.shape[1],),
                -1e4,
                dtype=torch.float32,
                device=kernel_scores.device,
            ))
            continue
        scores = kernel_scores[:, :, :, start:start + num_kernels].detach().float()
        per_page.append(scores.max(dim=-1).values.mean(dim=(0, 2)))
    if not per_page:
        return None
    logits = torch.stack(per_page, dim=1)
    logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    return torch.softmax(logits, dim=1)


def _aggregate_virtual_head_mass(
    virtual_blocks: List[Dict[str, Any]],
    original_block_ids: List[int],
    head_page_mass: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if head_page_mass is None or head_page_mass.numel() == 0:
        return None
    out = torch.zeros(
        (head_page_mass.shape[0], len(virtual_blocks)),
        dtype=torch.float32,
        device=device,
    )
    local_idx = {int(block_id): idx for idx, block_id in enumerate(original_block_ids)}
    for vid, block in enumerate(virtual_blocks):
        idx = [
            local_idx[int(page)]
            for page in block.get("fixed_page_ids", [])
            if int(page) in local_idx
        ]
        if idx:
            out[:, vid] = head_page_mass[:, idx].sum(dim=1)
    return out


def _aggregate_virtual_reps(
    virtual_blocks: List[Dict[str, Any]],
    page_reps: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if page_reps is None or page_reps.numel() == 0:
        return None
    dim = int(page_reps.shape[-1])
    reps = torch.zeros(len(virtual_blocks), dim, dtype=torch.float32, device=device)
    for vid, block in enumerate(virtual_blocks):
        valid = [int(p) for p in block.get("fixed_page_ids", []) if 0 <= int(p) < page_reps.shape[0]]
        if valid:
            reps[vid] = page_reps[torch.tensor(valid, dtype=torch.long, device=device)].float().mean(dim=0)
    return reps

def _expand_virtual_to_fixed_pages(virtual_blocks: List[Dict[str, Any]], virtual_ids: Iterable[int]) -> Set[int]:
    out: Set[int] = set()
    for vid in virtual_ids:
        if 0 <= int(vid) < len(virtual_blocks):
            out.update(int(p) for p in virtual_blocks[int(vid)].get("fixed_page_ids", []))
    return out

def _selected_virtual_tokens(virtual_blocks: List[Dict[str, Any]], virtual_ids: Iterable[int]) -> int:
    return int(sum(int(virtual_blocks[int(vid)].get("token_len", 0)) for vid in virtual_ids if 0 <= int(vid) < len(virtual_blocks)))

def _virtual_must_keep_ids(virtual_blocks: List[Dict[str, Any]], must_keep_pages: Set[int]) -> Set[int]:
    out: Set[int] = set()
    for block in virtual_blocks:
        if must_keep_pages.intersection({int(x) for x in block.get("fixed_page_ids", [])}):
            out.add(int(block["virtual_id"]))
    return out

def _top_virtual_ids(scores: torch.Tensor, k: int, exclude: Optional[Set[int]] = None) -> List[int]:
    exclude = exclude or set()
    ordered = torch.argsort(scores.detach().float(), descending=True).detach().cpu().tolist()
    out = []
    for idx in ordered:
        idx = int(idx)
        if idx in exclude:
            continue
        out.append(idx)
        if len(out) >= k:
            break
    return out

def _virtual_block_cost(block: Dict[str, Any], mode: str = "expanded_page_tokens") -> int:
    if mode == "semantic_span_tokens":
        return max(1, int(block.get("semantic_span_tokens", block.get("token_len", 0)) or block.get("token_len", 0) or 1))
    return max(1, int(block.get("expanded_page_tokens", len(block.get("fixed_page_ids", [])) * _BLOCK_SIZE) or 1))

def _virtual_selected_cost(virtual_blocks: List[Dict[str, Any]], virtual_ids: Iterable[int], mode: str = "expanded_page_tokens") -> int:
    if mode == "expanded_page_tokens":
        pages: Set[int] = set()
        for vid in virtual_ids:
            if 0 <= int(vid) < len(virtual_blocks):
                pages.update(int(p) for p in virtual_blocks[int(vid)].get("fixed_page_ids", []))
        return int(len(pages) * _BLOCK_SIZE)
    return int(sum(_virtual_block_cost(virtual_blocks[int(vid)], mode) for vid in virtual_ids if 0 <= int(vid) < len(virtual_blocks)))

def _virtual_incremental_cost(
    virtual_blocks: List[Dict[str, Any]],
    selected_ids: Iterable[int],
    candidate_id: int,
    mode: str = "expanded_page_tokens",
) -> int:
    candidate_id = int(candidate_id)
    if candidate_id < 0 or candidate_id >= len(virtual_blocks):
        return 0
    if mode == "expanded_page_tokens":
        selected_pages = _expand_virtual_to_fixed_pages(virtual_blocks, selected_ids)
        candidate_pages = {int(p) for p in virtual_blocks[candidate_id].get("fixed_page_ids", [])}
        return int(len(candidate_pages.difference(selected_pages)) * _BLOCK_SIZE)
    return _virtual_block_cost(virtual_blocks[candidate_id], mode)

def _density_adjusted_scores(scores: torch.Tensor, virtual_blocks: List[Dict[str, Any]], cost_mode: str, density_mode: str) -> torch.Tensor:
    adjusted = scores.detach().float().clone()
    if density_mode == "none":
        return torch.nan_to_num(adjusted, nan=0.0, posinf=0.0, neginf=0.0)
    costs = torch.tensor([float(_virtual_block_cost(block, cost_mode)) for block in virtual_blocks], dtype=torch.float32, device=adjusted.device)
    if density_mode == "sqrt_len":
        adjusted = adjusted / torch.sqrt(torch.clamp(costs, min=1.0))
    elif density_mode == "len":
        adjusted = adjusted / torch.clamp(costs, min=1.0)
    return torch.nan_to_num(adjusted, nan=0.0, posinf=0.0, neginf=0.0)

def _select_virtual_token_budget(
    scores: torch.Tensor,
    virtual_blocks: List[Dict[str, Any]],
    target_tokens: int,
    *,
    cost_mode: str = "expanded_page_tokens",
    density_mode: str = "sqrt_len",
    exclude: Optional[Set[int]] = None,
    must_keep: Optional[Set[int]] = None,
    allow_overshoot: bool = True,
    soft_tolerance: float = 0.03,
    hard_tolerance: float = 0.05,
) -> Set[int]:
    exclude = exclude or set()
    must_keep_ids: Set[int] = set(int(x) for x in (must_keep or set()) if 0 <= int(x) < len(virtual_blocks))
    selected: Set[int] = set(must_keep_ids)
    target = max(1, int(target_tokens))
    hard_limit = int(math.floor(target * (1.0 + max(0.0, float(hard_tolerance)))))
    adjusted = _density_adjusted_scores(scores, virtual_blocks, cost_mode, density_mode)
    ordered = [
        int(idx)
        for idx in torch.argsort(adjusted, descending=True).detach().cpu().tolist()
        if int(idx) not in exclude and int(idx) not in must_keep_ids
    ]
    expanded_candidates = set(selected).union(ordered)
    max_candidate_cost = _virtual_selected_cost(virtual_blocks, expanded_candidates, cost_mode)
    if max_candidate_cost <= target:
        return expanded_candidates

    best_selected = set(selected)
    best_cost = _virtual_selected_cost(virtual_blocks, best_selected, cost_mode)
    best_score = float(sum(float(adjusted[idx].item()) for idx in best_selected))

    def consider(candidate: Set[int]) -> bool:
        nonlocal best_selected, best_cost, best_score
        cost = _virtual_selected_cost(virtual_blocks, candidate, cost_mode)
        if not allow_overshoot and cost > target:
            return False
        if allow_overshoot and cost > hard_limit and cost > best_cost:
            return False
        score = float(sum(float(adjusted[idx].item()) for idx in candidate))
        key = (abs(cost - target), int(cost > target), -score)
        best_key = (abs(best_cost - target), int(best_cost > target), -best_score)
        if key < best_key:
            best_selected = set(candidate)
            best_cost = int(cost)
            best_score = score
            return True
        return False

    prefix = set(selected)
    for idx in ordered:
        prefix.add(idx)
        consider(prefix)
        prefix_cost = _virtual_selected_cost(virtual_blocks, prefix, cost_mode)
        if prefix_cost > hard_limit:
            break

    relative_error = abs(best_cost - target) / max(1, target)
    if relative_error > max(0.0, float(soft_tolerance)):
        # Expand beyond score prefixes. Single add/remove checks cover all
        # candidates; bounded swaps repair one-page/two-page layout granularity.
        for _ in range(2):
            snapshot = set(best_selected)
            for idx in ordered:
                if idx not in snapshot:
                    consider(snapshot.union({idx}))
            for idx in list(snapshot):
                if idx not in must_keep_ids:
                    consider(snapshot.difference({idx}))

            snapshot = set(best_selected)
            removable = sorted(
                (idx for idx in snapshot if idx not in must_keep_ids),
                key=lambda idx: float(adjusted[idx].item()),
            )[:64]
            addable = [idx for idx in ordered if idx not in snapshot][:128]
            for old_idx in removable:
                without_old = snapshot.difference({old_idx})
                for new_idx in addable:
                    consider(without_old.union({new_idx}))
            if abs(best_cost - target) / max(1, target) <= max(0.0, float(soft_tolerance)):
                break

    if not best_selected and ordered:
        best_selected = {ordered[0]}
    return best_selected

def _trim_to_token_budget(
    selected: Set[int],
    scores: torch.Tensor,
    virtual_blocks: List[Dict[str, Any]],
    target_tokens: int,
    must_keep: Set[int],
    cost_mode: str,
) -> Set[int]:
    target = max(1, int(target_tokens))
    while _virtual_selected_cost(virtual_blocks, selected, cost_mode) > target and len(selected) > len(must_keep):
        removable = [vid for vid in selected if vid not in must_keep]
        if not removable:
            break
        worst = min(removable, key=lambda vid: float(scores[int(vid)].detach().float().item()))
        selected.discard(int(worst))
    return selected

def _boundary_confidence_tensor(virtual_blocks: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor([float(block.get("boundary_confidence", 0.0)) for block in virtual_blocks], dtype=torch.float32, device=device)

def _candidate_blocks_for_hybrid(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    block_scores: torch.Tensor,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prompt_len = int(cfg.get("original_prompt_len", cfg.get("audit_context", {}).get("context_length", 0)) or 0)
    candidate_strategy = str(cfg.get("rescue_candidate_strategy", "fixed") or "fixed")
    if candidate_strategy == "semantic_v2":
        blocks, meta = _virtual_layout_from_precomputed(cfg, original_block_ids)
        if blocks:
            meta = dict(meta)
            meta["rescue_candidate_strategy"] = "semantic_v2"
            return blocks, meta
    return _fixed_virtual_blocks(original_block_ids, prompt_len, "hybrid_fixed_candidate"), {
        "fallback_used": False,
        "fallback_reason": "",
        "rescue_candidate_strategy": "fixed",
    }

def _candidate_blocks_from_rescue_layout(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build rescue candidate virtual blocks from a precomputed rescue-candidate
    layout (see bridgekv.rescue_candidate_builder). Real attention unit stays the
    fixed page; candidates only restrict to pages still available at this layer."""
    prompt_len = int(cfg.get("original_prompt_len", cfg.get("audit_context", {}).get("context_length", 0)) or 0)
    layout = cfg.get("rescue_candidate_layout") if isinstance(cfg.get("rescue_candidate_layout"), dict) else {}
    available_pages = {int(x) for x in original_block_ids}
    candidate_strategy = str(cfg.get("rescue_candidate_strategy", "fixed") or "fixed")
    spans = layout.get("token_spans") or []
    mapping = layout.get("mapping_to_fixed_pages") or []
    sources = layout.get("source_types") or []
    previews = layout.get("block_text_preview") or []
    expanded = layout.get("expanded_page_tokens") or []
    semantic_tokens = layout.get("semantic_span_tokens") or []
    query_scores = layout.get("query_score") or []
    semantic_sources = layout.get("semantic_source") or []
    coarse_ids = layout.get("coarse_region_id") or []
    neighbor_pages = layout.get("neighbor_source_page") or []
    query_covered = layout.get("query_covered_idx") or []
    answer_scores = layout.get("answer_score") or []
    if not spans:
        # no precomputed layout -> fall back to fixed pages so rescue still runs
        return _fixed_virtual_blocks(original_block_ids, prompt_len, "rescue_fixed_fallback"), {
            "fallback_used": True,
            "fallback_reason": "missing_rescue_candidate_layout",
            "rescue_candidate_strategy": candidate_strategy,
        }
    blocks: List[Dict[str, Any]] = []
    for idx, raw_span in enumerate(spans):
        if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
            continue
        start, end = int(raw_span[0]), int(raw_span[1])
        pages = [int(x) for x in (mapping[idx] if idx < len(mapping) else _layout_pages_for_span(start, end, prompt_len))]
        pages = [p for p in pages if p in available_pages]
        if not pages:
            continue
        token_len = max(0, end - start)
        sem_span = int(semantic_tokens[idx]) if idx < len(semantic_tokens) else token_len
        blocks.append({
            "virtual_id": len(blocks),
            "source_layout_id": int(idx),
            "token_start": start,
            "token_end": end,
            "token_len": token_len,
            "source_type": str(sources[idx]) if idx < len(sources) else candidate_strategy,
            "text_preview": str(previews[idx])[:160] if idx < len(previews) else "",
            "fixed_page_ids": pages,
            "all_fixed_page_ids": pages,
            "char_span": [0, 0],
            "semantic_span_tokens": int(sem_span),
            "expanded_page_tokens": int(expanded[idx]) if idx < len(expanded) else int(len(pages) * _BLOCK_SIZE),
            "page_expansion_overhead": float((len(pages) * _BLOCK_SIZE) / max(1, sem_span)),
            "boundary_confidence": 0.0,
            "boundary_score_left": 0.0,
            "boundary_score_right": 0.0,
            "rescue_query_score": float(query_scores[idx]) if idx < len(query_scores) else 0.0,
            "rescue_semantic_source": str(semantic_sources[idx]) if idx < len(semantic_sources) else "",
            "rescue_coarse_region_id": int(coarse_ids[idx]) if idx < len(coarse_ids) else -1,
            "rescue_neighbor_source_page": int(neighbor_pages[idx]) if idx < len(neighbor_pages) else -1,
            "rescue_query_covered_idx": [int(x) for x in (query_covered[idx] if idx < len(query_covered) else [])],
            "rescue_answer_score": float(answer_scores[idx]) if idx < len(answer_scores) else 0.0,
        })
    if not blocks:
        return _fixed_virtual_blocks(original_block_ids, prompt_len, "rescue_fixed_fallback"), {
            "fallback_used": True,
            "fallback_reason": "empty_rescue_candidate_layout",
            "rescue_candidate_strategy": candidate_strategy,
        }
    meta = {
        "fallback_used": bool(layout.get("fallback_used", False)),
        "fallback_reason": str(layout.get("fallback_reason", "")),
        "fallback_reason_counts": layout.get("fallback_reason_counts", {}),
        "rescue_candidate_strategy": candidate_strategy,
        "rescue_candidate_count": int(len(blocks)),
        "rescue_source_counts": dict((layout.get("metadata") or {}).get("source_counts", {})),
        "query_vocab_size": int(layout.get("query_vocab_size", 0) or 0),
    }
    return blocks, meta

def _greedy_mmr_token_budget(
    *,
    inactive_ids: List[int],
    s0_scores: torch.Tensor,
    virtual_blocks: List[Dict[str, Any]],
    block_reps: Optional[torch.Tensor],
    active_ids: List[int],
    initial_selected_ids: Set[int],
    target_tokens: int,
    cost_mode: str,
    density_mode: str,
    eta_rescue: float,
    eta_active: float,
    allow_overshoot: bool = True,
    hard_tolerance: float = 0.05,
) -> List[int]:
    selected_rescue: List[int] = []
    selected_all = set(int(x) for x in initial_selected_ids)
    target = max(1, int(target_tokens))
    hard_limit = int(math.floor(target * (1.0 + max(0.0, float(hard_tolerance)))))
    best_rescue: List[int] = list(selected_rescue)
    best_error = abs(_virtual_selected_cost(virtual_blocks, selected_all, cost_mode) - target)
    while _virtual_selected_cost(virtual_blocks, selected_all, cost_mode) < target:
        best_id: Optional[int] = None
        best_score = -float("inf")
        for vid in inactive_ids:
            vid = int(vid)
            if vid in selected_all:
                continue
            score = float(s0_scores[vid].detach().float().item()) if vid < s0_scores.numel() else 0.0
            cost = float(max(1, _virtual_incremental_cost(virtual_blocks, selected_all, vid, cost_mode)))
            if density_mode == "sqrt_len":
                score = score / math.sqrt(max(1.0, cost))
            elif density_mode == "len":
                score = score / max(1.0, cost)
            if block_reps is not None and block_reps.numel() > 0 and vid < block_reps.shape[0]:
                cur = torch.nn.functional.normalize(block_reps[vid].float().reshape(1, -1), dim=-1, eps=1e-6)
                if selected_rescue:
                    r_ids = torch.tensor([x for x in selected_rescue if 0 <= x < block_reps.shape[0]], dtype=torch.long, device=block_reps.device)
                    if r_ids.numel() > 0:
                        r = torch.nn.functional.normalize(block_reps[r_ids].float(), dim=-1, eps=1e-6)
                        score -= float(eta_rescue) * float(torch.matmul(cur, r.T).max().item())
                if active_ids:
                    a_ids = torch.tensor([x for x in active_ids if 0 <= x < block_reps.shape[0]], dtype=torch.long, device=block_reps.device)
                    if a_ids.numel() > 0:
                        a = torch.nn.functional.normalize(block_reps[a_ids].float(), dim=-1, eps=1e-6)
                        score -= float(eta_active) * float(torch.matmul(cur, a.T).max().item())
            if score > best_score:
                best_score = score
                best_id = vid
        if best_id is None:
            break
        candidate_all = set(selected_all)
        candidate_all.add(int(best_id))
        candidate_cost = _virtual_selected_cost(virtual_blocks, candidate_all, cost_mode)
        if allow_overshoot and selected_all and candidate_cost > hard_limit:
            candidate_rescue = list(selected_rescue) + [int(best_id)]
            candidate_error = abs(candidate_cost - target)
            if candidate_error < best_error:
                best_rescue = candidate_rescue
                best_error = candidate_error
            break
        selected_rescue.append(int(best_id))
        selected_all.add(int(best_id))
        current_error = abs(_virtual_selected_cost(virtual_blocks, selected_all, cost_mode) - target)
        if current_error < best_error:
            best_rescue = list(selected_rescue)
            best_error = current_error
        if len(selected_all) >= len(virtual_blocks):
            break
    return best_rescue

def _write_block_layout_record(
    cfg: Dict[str, Any],
    virtual_blocks: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> None:
    path = cfg.get("block_layout_audit_path")
    if not path or cfg.get("phase") != "prefill":
        return
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    key = f"{context.get('task', '')}::{context.get('sample_id', '')}::{cfg.get('block_strategy', 'fixed')}"
    written = cfg.get("_block_layout_written")
    if not isinstance(written, list):
        written = []
        cfg["_block_layout_written"] = written
    if key in written:
        return
    written.append(key)
    lengths = [int(b.get("token_len", 0)) for b in virtual_blocks]
    overhead = [float(b.get("page_expansion_overhead", 0.0)) for b in virtual_blocks]
    boundary_conf = [float(b.get("boundary_confidence", 0.0)) for b in virtual_blocks]
    record = {
        "sample_id": context.get("sample_id", ""),
        "task": context.get("task", ""),
        "block_strategy": cfg.get("block_strategy", "fixed"),
        "num_blocks": int(len(virtual_blocks)),
        "block_lengths": lengths,
        "token_spans_first30": [[int(b.get("token_start", 0)), int(b.get("token_end", 0))] for b in virtual_blocks[:30]],
        "token_spans_first50": [[int(b.get("token_start", 0)), int(b.get("token_end", 0))] for b in virtual_blocks[:50]],
        "char_spans_first50": [b.get("char_span", [0, 0]) for b in virtual_blocks[:50]],
        "block_text_preview_first20": [str(b.get("text_preview", ""))[:160] for b in virtual_blocks[:20]],
        "strategy_metadata": meta,
        "fallback_used": bool(meta.get("fallback_used", False)),
        "fallback_reason": str(meta.get("fallback_reason", "")),
        "fallback_reason_counts": meta.get("fallback_reason_counts", {}),
        "page_expansion_overhead": overhead,
        "page_expansion_overhead_mean": float(sum(overhead) / max(1, len(overhead))) if overhead else 0.0,
        "boundary_confidence": boundary_conf,
        "boundary_confidence_mean": float(sum(boundary_conf) / max(1, len(boundary_conf))) if boundary_conf else 0.0,
    }
    try:
        with open(str(path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _write_generic_pruning_audit(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    final_ids: Set[int],
    block_scores: torch.Tensor,
    num_blocks_to_keep: int,
    final_budget_type: str,
    layout_meta: Optional[Dict[str, Any]] = None,
    virtual_blocks: Optional[List[Dict[str, Any]]] = None,
    active_virtual_ids: Optional[Set[int]] = None,
    final_virtual_ids: Optional[Set[int]] = None,
    rescued_virtual_ids: Optional[List[int]] = None,
    score_meta: Optional[Dict[str, Any]] = None,
) -> None:
    audit_path = cfg.get("rescue_audit_path")
    if not audit_path:
        return
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    prompt_len = int(cfg.get("original_prompt_len", context.get("context_length", 0)) or 0)
    fixed_abs = from_blocks_to_abs_indices_list(sorted(final_ids), prompt_len)
    layout_meta = layout_meta or {}
    virtual_blocks = virtual_blocks or []
    final_virtual_ids = final_virtual_ids or set()
    active_virtual_ids = active_virtual_ids or set()
    rescued_virtual_ids = rescued_virtual_ids or []
    score_meta = score_meta or {}
    calibrated_config_loaded, calibration_entry_found = _calibration_status(cfg)
    calibration_entry = _calibration_entry(cfg)
    lengths = [int(b.get("token_len", 0)) for b in virtual_blocks] or [
        max(0, min(prompt_len, (int(b) + 1) * _BLOCK_SIZE) - int(b) * _BLOCK_SIZE) for b in original_block_ids
    ]
    selected_virtual_token_count = _selected_virtual_tokens(virtual_blocks, final_virtual_ids) if virtual_blocks else int(len(fixed_abs))
    selected_token_count = int(len(fixed_abs))
    selected_block_objs = [virtual_blocks[int(vid)] for vid in final_virtual_ids if virtual_blocks and 0 <= int(vid) < len(virtual_blocks)]
    semantic_span_tokens = int(sum(int(b.get("semantic_span_tokens", b.get("token_len", 0)) or 0) for b in selected_block_objs)) if selected_block_objs else selected_virtual_token_count
    expanded_page_tokens = _virtual_selected_cost(virtual_blocks, final_virtual_ids, "expanded_page_tokens") if selected_block_objs else selected_token_count
    overhead_values = [float(b.get("page_expansion_overhead", 0.0)) for b in selected_block_objs]
    boundary_values = [float(b.get("boundary_confidence", 0.0)) for b in virtual_blocks]
    controller_kind = "final" if int(cfg.get("_controller_target_final_tokens") or 0) > 0 else "active"
    target_budget = int(
        cfg.get(f"_controller_target_{controller_kind}_tokens")
        or cfg.get("target_final_tokens")
        or cfg.get("target_token_budget")
        or 0
    )
    uncapped_target = int(cfg.get(f"_uncapped_controller_target_{controller_kind}_tokens") or target_budget)
    requested_target = int(cfg.get(f"_requested_target_{controller_kind}_tokens") or target_budget)
    calibration_target = int(cfg.get(f"_calibration_target_{controller_kind}_tokens") or target_budget)
    min_reachable = int(cfg.get(f"_min_reachable_{controller_kind}_tokens") or 0)
    max_reachable = int(cfg.get(f"_max_reachable_{controller_kind}_tokens") or 0)
    cap_reasons = [
        str(calibration_entry.get("target_cap_reason", "") or ""),
        str(cfg.get(f"_target_cap_reason_{controller_kind}", "") or ""),
    ]
    target_cap_reason = ",".join(reason for reason in cap_reasons if reason)
    token_error = float((selected_token_count - target_budget) / max(1, target_budget)) if target_budget > 0 else ""
    fixed_sliminfer_ids = set(_top_virtual_ids(block_scores, min(max(1, int(num_blocks_to_keep)), int(block_scores.numel())))) if block_scores.numel() else set()
    record = {
        "task": context.get("task", ""),
        "sample_id": context.get("sample_id", ""),
        "sample_index": context.get("sample_index", ""),
        "method": context.get("method", cfg.get("variant_method", "")),
        "block_strategy": cfg.get("block_strategy", "fixed"),
        "active_block_strategy": cfg.get("active_block_strategy", "") if (str(cfg.get("block_strategy", "")).startswith("hybrid") or str(cfg.get("block_strategy", "")) == "fixed_active_rescue") else "",
        "rescue_candidate_strategy": cfg.get("rescue_candidate_strategy", "") if (str(cfg.get("block_strategy", "")).startswith("hybrid") or str(cfg.get("block_strategy", "")) == "fixed_active_rescue") else "",
        "budget_mode": cfg.get("budget_mode", "block_count"),
        "active_budget_mode": cfg.get("active_budget_mode", cfg.get("budget_mode", "block_count")),
        "rescue_budget_mode": cfg.get("rescue_budget_mode", "block_count"),
        "context_length": prompt_len,
        "layer_id": cfg.get("layer_idx", ""),
        "phase": cfg.get("phase", ""),
        "rescue_policy": cfg.get("rescue_policy", "none"),
        "rescue_score": cfg.get("rescue_score", "delta_o"),
        "target_token_budget": int(cfg.get("target_token_budget") or 0),
        "target_final_tokens": int(cfg.get("target_final_tokens") or 0),
        "requested_target_tokens": requested_target,
        "calibration_target_tokens": calibration_target,
        "uncapped_target_tokens": uncapped_target,
        "effective_target_tokens": target_budget,
        "min_reachable_tokens": min_reachable,
        "max_reachable_tokens": max_reachable,
        "target_cap_reason": target_cap_reason,
        "controller_target_active_tokens": int(cfg.get("_controller_target_active_tokens") or 0),
        "controller_target_final_tokens": int(cfg.get("_controller_target_final_tokens") or 0),
        "calibrated_budget_config_loaded": calibrated_config_loaded,
        "calibration_entry_found": calibration_entry_found,
        "target_tokens_used_by_cache": target_budget,
        "observed_final_selected_tokens_used_by_attention": selected_token_count,
        "budget_error_vs_effective_target_pct": token_error,
        "overshoot_used": bool(target_budget > 0 and selected_token_count > target_budget),
        "selected_pages": int(len(final_ids)),
        "K": int(len(active_virtual_ids) if active_virtual_ids else num_blocks_to_keep),
        "M": int(len(rescued_virtual_ids)),
        "final_block_count_policy": int(len(final_virtual_ids) if final_virtual_ids else len(final_ids)),
        "final_block_count_used_by_attention": int(len(final_ids)),
        "selected_token_count": selected_token_count,
        "selected_token_error_pct": token_error,
        "selected_virtual_token_count": int(selected_virtual_token_count),
        "selected_token_ratio": float(selected_token_count / max(1, prompt_len)),
        "expanded_page_tokens": expanded_page_tokens,
        "semantic_span_tokens": semantic_span_tokens,
        "page_expansion_overhead_mean": float(sum(overhead_values) / max(1, len(overhead_values))) if overhead_values else 0.0,
        "page_expansion_overhead_max": float(max(overhead_values)) if overhead_values else 0.0,
        "avg_block_tokens": float(sum(lengths) / max(1, len(lengths))),
        "block_count_total": int(len(virtual_blocks) if virtual_blocks else len(original_block_ids)),
        "selected_block_count": int(len(final_virtual_ids) if final_virtual_ids else len(final_ids)),
        "block_len_mean": _block_length_stats(lengths)["mean"],
        "block_len_std": _block_length_stats(lengths)["std"],
        "block_len_min": _block_length_stats(lengths)["min"],
        "block_len_max": _block_length_stats(lengths)["max"],
        "rescued_count": int(len(rescued_virtual_ids)),
        "duplicate_block_ids": bool(len(final_ids) != len(list(final_ids))),
        "duplicate_token_spans": False,
        "protected_block_violation": False,
        "fallback_used": bool(layout_meta.get("fallback_used", False)),
        "fallback_reason": str(layout_meta.get("fallback_reason", "")),
        "fallback_reason_counts": layout_meta.get("fallback_reason_counts", {}),
        "coverage_ok": bool(layout_meta.get("coverage_ok", True)),
        "no_gap": bool(layout_meta.get("no_gap", True)),
        "no_overlap": bool(layout_meta.get("no_overlap", True)),
        "offset_alignment_success_rate": (layout_meta.get("offset_alignment_success_rate", "") if isinstance(layout_meta, dict) else ""),
        "D_base_stats": score_meta.get("D_base_stats", {}),
        "G_diffusion_stats": score_meta.get("G_diffusion_stats", {}),
        "R_active_stats": score_meta.get("R_active_stats", {}),
        "rescue_selector": score_meta.get("rescue_selector", ""),
        "rescue_gate_mode": score_meta.get("rescue_gate_mode", ""),
        "rescue_need_score": score_meta.get("rescue_need_score", ""),
        "rescue_active_cov_frac": score_meta.get("rescue_active_cov_frac", ""),
        "rescue_m_pages": score_meta.get("rescue_m_pages", ""),
        "rescue_m_default": score_meta.get("rescue_m_default", ""),
        "block_budget_mode_used": score_meta.get("block_budget_mode", ""),
        "verifier_mode_used": score_meta.get("verifier_mode", ""),
        "swap_count": score_meta.get("swap_count", 0),
        "swapped_out_pages": score_meta.get("swapped_out_pages", []),
        "swapped_in_pages": score_meta.get("swapped_in_pages", []),
        "active_core_size": score_meta.get("active_core_size", ""),
        "active_tail_size": score_meta.get("active_tail_size", ""),
        "swap_utility_margin": score_meta.get("swap_utility_margin", ""),
        "swap_utility_source": score_meta.get("swap_utility_source", ""),
        "rescue_candidate_pool_size": score_meta.get("rescue_candidate_count", ""),
        "rh_mode": score_meta.get("rh_mode", ""),
        "rh_profile_heads_retrieval": score_meta.get("rh_profile_heads_retrieval", ""),
        "rh_profile_heads_reasoning": score_meta.get("rh_profile_heads_reasoning", ""),
        "rh_profile_heads_query": score_meta.get("rh_profile_heads_query", ""),
        "active_fixed_page_ids": score_meta.get("active_fixed_page_ids", []),
        "rescued_fixed_page_ids": score_meta.get("rescued_fixed_page_ids", []),
        "D_base_std_mean": (score_meta.get("D_base_stats", {}) or {}).get("std", ""),
        "G_std_mean": (score_meta.get("G_diffusion_stats", {}) or {}).get("std", ""),
        "R_std_mean": (score_meta.get("R_active_stats", {}) or {}).get("std", ""),
        "boundary_confidence_mean": float(sum(boundary_values) / max(1, len(boundary_values))) if boundary_values else 0.0,
        "boundary_confidence_std": float(torch.tensor(boundary_values, dtype=torch.float32).std(unbiased=False).item()) if len(boundary_values) > 1 else 0.0,
        "jaccard_vs_fixed_sliminfer": _jaccard(final_ids, fixed_sliminfer_ids) if fixed_sliminfer_ids else "",
        "jaccard_vs_same_strategy_sliminfer": _jaccard(set(final_virtual_ids), set(active_virtual_ids)) if final_virtual_ids and active_virtual_ids else "",
        "output_same_as_fixed_sliminfer": "",
        "final_budget_type": final_budget_type,
        "num_original_blocks": int(len(original_block_ids)),
        "final_block_ids_first100": [int(x) for x in sorted(final_ids)[:100]],
        "final_virtual_ids_first100": [int(x) for x in sorted(final_virtual_ids)[:100]] if final_virtual_ids else [],
        "rescued_virtual_ids_first50": [int(x) for x in rescued_virtual_ids[:50]],
    }
    if str(cfg.get("rescue_score", "")) in _STRUCTURED_RESCUE_SCORES:
        enrich_structured_audit_fields(
            record, cfg, score_meta or {},
            list(active_virtual_ids) if active_virtual_ids else [],
            rescued_virtual_ids or [],
            list(final_ids),
        )
    try:
        with open(str(audit_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _zscore_over(values: torch.Tensor, ids: List[int], device: torch.device) -> Dict[int, float]:
    if not ids:
        return {}
    vals = torch.tensor([float(values[i].item()) if 0 <= i < values.numel() else 0.0 for i in ids], dtype=torch.float32)
    mean = float(vals.mean().item())
    std = float(vals.std(unbiased=False).item())
    if std < 1e-9:
        return {int(i): 0.0 for i in ids}
    return {int(i): (float(vals[k].item()) - mean) / std for k, i in enumerate(ids)}


def _maxcos(reps: Optional[torch.Tensor], cid: int, others: List[int]) -> float:
    if reps is None or reps.numel() == 0 or not others or cid >= reps.shape[0]:
        return 0.0
    cur = torch.nn.functional.normalize(reps[cid].float().reshape(1, -1), dim=-1, eps=1e-6)
    idx = torch.tensor([o for o in others if 0 <= o < reps.shape[0]], dtype=torch.long, device=reps.device)
    if idx.numel() == 0:
        return 0.0
    oth = torch.nn.functional.normalize(reps[idx].float(), dim=-1, eps=1e-6)
    return float(torch.matmul(cur, oth.T).max().item())


def _ac_active_tail_score(virtual_scores: torch.Tensor, sel_active: List[int]) -> float:
    if not sel_active:
        return 0.0
    vals = [float(virtual_scores[v].item()) if v < virtual_scores.numel() else 0.0 for v in sel_active]
    vals.sort()
    n_tail = max(1, int(math.ceil(0.2 * len(vals))))
    return float(sum(vals[:n_tail]) / n_tail)


def _ac_compute_need_score(
    *,
    cfg: Dict[str, Any],
    active_cov_frac: float,
    conc: float,
    flat_norm: float,
    fup_peak: float,
    top_margin: float,
    task: str,
) -> float:
    """Composite rescue-need score in [0,1] (higher => more need to rescue)."""
    active_redundancy = 1.0 - flat_norm
    task = str(task or "")
    task_risk = 0.15 if task in ("passage_retrieval_en", "passage_count") else 0.0
    coverage_gap = max(0.0, 1.0 - active_cov_frac)
    margin_signal = max(0.0, min(1.0, top_margin / 2.0))
    fup_signal = max(0.0, min(1.0, fup_peak))
    w_cov, w_red, w_margin, w_fup, w_task = 0.30, 0.15, 0.25, 0.20, 0.10
    raw = (w_cov * coverage_gap + w_red * active_redundancy + w_margin * margin_signal
           + w_fup * fup_signal + w_task * (1.0 - task_risk))
    return float(max(0.0, min(1.0, raw)))


def _ac_gate_m_pages(
    *,
    cfg: Dict[str, Any],
    ac_gate_type: str,
    m_default: int,
    need_score: float,
    top_margin: float,
    task: str,
) -> Tuple[int, float, str]:
    """Return (m_pages, threshold_used, gate_type)."""
    gt = str(ac_gate_type or "no_gate")
    if gt == "no_gate":
        return int(m_default), 0.0, gt
    task_thresh = {
        "passage_retrieval_en": float(cfg.get("ac_gate_thresh_retrieval", 0.52)),
        "passage_count": float(cfg.get("ac_gate_thresh_count", 0.48)),
        "musique": float(cfg.get("ac_gate_thresh_multihop", 0.36)),
        "hotpotqa": float(cfg.get("ac_gate_thresh_multihop", 0.36)),
        "qasper": float(cfg.get("ac_gate_thresh_qasper", 0.40)),
    }
    if gt == "gate_score_threshold":
        th = float(cfg.get("ac_gate_score_threshold", 0.42))
        return (int(m_default) if need_score >= th else 0), th, gt
    if gt == "gate_top_margin":
        th = float(cfg.get("ac_gate_margin_threshold", 0.12))
        return (int(m_default) if top_margin >= th else 0), th, gt
    if gt == "gate_task_aware":
        th = float(task_thresh.get(str(task), float(cfg.get("ac_gate_score_threshold", 0.42))))
        return (int(m_default) if need_score >= th else 0), th, gt
    return int(m_default), 0.0, gt


def _rescue_page_feature_maps(rescue_blocks: List[Dict[str, Any]]) -> Tuple[Dict[int, float], Dict[int, float]]:
    query: Dict[int, float] = {}
    answer: Dict[int, float] = {}
    for block in rescue_blocks:
        q = float(block.get("rescue_query_score", 0.0) or 0.0)
        a = float(block.get("rescue_answer_score", 0.0) or 0.0)
        for raw_page in block.get("fixed_page_ids", []):
            page = int(raw_page)
            query[page] = max(query.get(page, 0.0), q)
            answer[page] = max(answer.get(page, 0.0), a)
    return query, answer


def _maybe_dump_head_profile_features(
    *,
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    head_page_mass: Optional[torch.Tensor],
    active_pages: Set[int],
    rescue_blocks: List[Dict[str, Any]],
) -> None:
    path = cfg.get("head_profile_dump_path")
    if not path or str(cfg.get("phase", "")) == "decode":
        return
    if head_page_mass is None or head_page_mass.numel() == 0:
        return
    ctx = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    task = str(ctx.get("task", ""))
    query_map, answer_map = _rescue_page_feature_maps(rescue_blocks)
    pages = [int(x) for x in original_block_ids]
    local_idx = {p: i for i, p in enumerate(pages)}
    query_pages = {p for p in pages if query_map.get(p, 0.0) > 0.0}
    evidence_pages = {p for p in pages if answer_map.get(p, 0.0) > 0.0}
    reasoning_pages = (
        {p for p in query_pages if p not in evidence_pages}
        if task in {"musique", "hotpotqa"} else set()
    )
    n_local = max(1, int(math.ceil(len(pages) * 0.10)))
    local_pages = set(pages[-n_local:])

    def mass_for(head: int, selected_pages: Set[int]) -> float:
        idx = [local_idx[p] for p in selected_pages if p in local_idx]
        return float(head_page_mass[head, idx].sum().item()) if idx else 0.0

    rows = []
    for head in range(int(head_page_mass.shape[0])):
        rows.append({
            "task": task,
            "sample_id": str(ctx.get("sample_id", "")),
            "layer_id": int(cfg.get("layer_idx", 0) or 0),
            "head_id": int(head),
            "evidence_attention_mass": round(mass_for(head, evidence_pages), 7),
            "query_attention_mass": round(mass_for(head, query_pages), 7),
            "reasoning_attention_mass": round(mass_for(head, reasoning_pages), 7),
            "local_attention_mass": round(mass_for(head, local_pages), 7),
            "active_coverage_score": round(mass_for(head, active_pages), 7),
            "calibration_support": int(bool(evidence_pages or query_pages)),
            "evidence_page_fraction": round(len(evidence_pages) / max(1, len(pages)), 7),
            "query_page_fraction": round(len(query_pages) / max(1, len(pages)), 7),
            "reasoning_page_fraction": round(len(reasoning_pages) / max(1, len(pages)), 7),
            "local_page_fraction": round(len(local_pages) / max(1, len(pages)), 7),
        })
    try:
        import os as _os
        _os.makedirs(_os.path.dirname(_os.path.abspath(str(path))), exist_ok=True)
        with open(str(path), "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _rh_profile_weights(
    cfg: Dict[str, Any],
    head_type: str,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    profile = cfg.get("rh_head_profile") if isinstance(cfg.get("rh_head_profile"), dict) else {}
    heads = profile.get("heads", []) if isinstance(profile, dict) else []
    layer_id = int(cfg.get("layer_idx", 0) or 0)
    weights = torch.zeros(num_heads, dtype=torch.float32, device=device)
    score_key = {
        "retrieval": "retrieval_score",
        "reasoning": "reasoning_score",
        "query_focused": "query_focused_score",
    }.get(head_type, "")
    for row in heads:
        if not isinstance(row, dict):
            continue
        if int(row.get("layer_id", -1)) != layer_id or str(row.get("head_type", "")) != head_type:
            continue
        head_id = int(row.get("head_id", -1))
        if 0 <= head_id < num_heads:
            weights[head_id] = max(1e-6, float(row.get(score_key, 1.0) or 1.0))
    return weights


def _rh_weighted_mean(vec: torch.Tensor, weights: torch.Tensor) -> float:
    denom = float(weights.sum().item())
    if denom <= 1e-9:
        return 0.0
    return float((vec * weights).sum().item() / denom)


def _rh_dict_zscore(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    keys = list(values)
    vals = torch.tensor([float(values[k]) for k in keys], dtype=torch.float32)
    std = float(vals.std(unbiased=False).item())
    if std <= 1e-9:
        return {k: 0.0 for k in keys}
    mean = float(vals.mean().item())
    return {k: (float(values[k]) - mean) / std for k in keys}


def _rh_page_utility_map(cfg: Dict[str, Any]) -> Dict[int, float]:
    raw = cfg.get("rh_page_utility_map") or {}
    if not isinstance(raw, dict):
        return {}
    layer_key = str(int(cfg.get("layer_idx", 0) or 0))
    if layer_key in raw and isinstance(raw[layer_key], dict):
        raw = raw[layer_key]
    elif int(cfg.get("layer_idx", 0) or 0) in raw and isinstance(raw[int(cfg.get("layer_idx", 0) or 0)], dict):
        raw = raw[int(cfg.get("layer_idx", 0) or 0)]
    out: Dict[int, float] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = float(value)
        except Exception:
            continue
    return out


def _rh_structural_rescue_select(
    *,
    cfg: Dict[str, Any],
    combined: List[Dict[str, Any]],
    candidate_ids: List[int],
    active_ids: List[int],
    original_block_ids: List[int],
    head_page_mass: Optional[torch.Tensor],
    combined_delta: torch.Tensor,
    g_tensor: torch.Tensor,
    r_tensor: torch.Tensor,
    combined_reps: Optional[torch.Tensor],
    offset: int,
    rescue_blocks: List[Dict[str, Any]],
    target_final: int,
    cost_mode: str,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any]]:
    mode = str(cfg.get("rh_mode", cfg.get("rescue_selector", "rh_static")) or "rh_static")
    alpha = float(cfg.get("rh_alpha_drm", 1.0) or 1.0)
    beta_r = float(cfg.get("rh_beta_retrieval", 0.8) or 0.8)
    beta_q = float(cfg.get("rh_beta_query", 0.6) or 0.6)
    beta_reason = float(cfg.get("rh_beta_reasoning", 0.4) or 0.4)
    lambda_h = float(cfg.get("rh_lambda_head", 0.25) or 0.25)
    lambda_p = float(cfg.get("rh_lambda_page", 0.25) or 0.25)
    cost_adjusted = bool(int(cfg.get("rh_cost_adjusted", 1) or 0))
    lam_red = float(cfg.get("lambda_red", 0.25) or 0.25)
    z_d = _zscore_over(combined_delta, candidate_ids, device)
    z_g = _zscore_over(g_tensor, candidate_ids, device)
    z_r = _zscore_over(r_tensor, candidate_ids, device)

    def base_drm(cid: int) -> float:
        return z_d.get(cid, 0.0) + float(cfg.get("alpha_diff", 0.5)) * z_g.get(cid, 0.0) - lam_red * z_r.get(cid, 0.0)

    combined_head = _aggregate_virtual_head_mass(combined, original_block_ids, head_page_mass, device)
    num_heads = int(combined_head.shape[0]) if combined_head is not None else 0
    wr = _rh_profile_weights(cfg, "retrieval", num_heads, device) if num_heads else torch.zeros(0, device=device)
    wreason = _rh_profile_weights(cfg, "reasoning", num_heads, device) if num_heads else torch.zeros(0, device=device)
    wq_static = _rh_profile_weights(cfg, "query_focused", num_heads, device) if num_heads else torch.zeros(0, device=device)
    if num_heads and float(wr.sum().item()) <= 1e-9:
        wr = torch.ones(num_heads, dtype=torch.float32, device=device)

    query_map, answer_map = _rescue_page_feature_maps(rescue_blocks)
    q_by_candidate = {
        cid: max(
            (query_map.get(int(p), 0.0) for p in combined[cid].get("fixed_page_ids", [])),
            default=0.0,
        )
        for cid in candidate_ids
    }
    answer_by_candidate = {
        cid: max(
            (answer_map.get(int(p), 0.0) for p in combined[cid].get("fixed_page_ids", [])),
            default=0.0,
        )
        for cid in candidate_ids
    }
    if num_heads and head_page_mass is not None:
        local_idx = {int(page): idx for idx, page in enumerate(original_block_ids)}
        qvec = torch.tensor(
            [float(query_map.get(int(page), 0.0)) for page in original_block_ids],
            dtype=torch.float32,
            device=device,
        )
        query_dynamic = torch.matmul(head_page_mass, qvec)
        if float(query_dynamic.sum().item()) > 1e-9:
            keep = max(2, int(math.ceil(num_heads * 0.20)))
            top_idx = torch.topk(query_dynamic, k=min(keep, num_heads)).indices
            masked = torch.zeros_like(query_dynamic)
            masked[top_idx] = query_dynamic[top_idx]
            query_dynamic = masked / masked.sum().clamp_min(1e-9)
    else:
        query_dynamic = torch.zeros(num_heads, dtype=torch.float32, device=device)
    if mode == "rh_query":
        wq = wq_static + query_dynamic
    else:
        wq = wq_static
    if num_heads and float(wq.sum().item()) <= 1e-9:
        wq = query_dynamic if float(query_dynamic.sum().item()) > 1e-9 else wr.clone()

    learned_map = _rh_page_utility_map(cfg)
    selected_all = set(active_ids)
    rescued: List[int] = []
    coverage = (
        combined_head[:, active_ids].sum(dim=1).clamp(0.0, 1.0)
        if combined_head is not None and active_ids else torch.zeros(num_heads, dtype=torch.float32, device=device)
    )
    active_head_vec = (
        combined_head[:, active_ids].sum(dim=1)
        if combined_head is not None and active_ids else torch.zeros(num_heads, dtype=torch.float32, device=device)
    )
    rescue_head_vec = torch.zeros(num_heads, dtype=torch.float32, device=device)
    feature_rows: Dict[int, Dict[str, Any]] = {}
    target = max(1, int(target_final))
    hard = int(math.floor(target * 1.01))
    best_rescue: List[int] = []
    best_err = abs(_virtual_selected_cost(combined, selected_all, cost_mode) - target)

    def head_red(vec: torch.Tensor, other: torch.Tensor) -> float:
        if vec.numel() == 0 or other.numel() == 0 or float(other.norm().item()) <= 1e-9:
            return 0.0
        return float(torch.nn.functional.cosine_similarity(vec.reshape(1, -1), other.reshape(1, -1)).item())

    guard = 0
    while _virtual_selected_cost(combined, selected_all, cost_mode) < target and guard < len(candidate_ids) + 2:
        guard += 1
        remaining = [cid for cid in candidate_ids if cid not in selected_all]
        if not remaining:
            break
        raw_r: Dict[int, float] = {}
        raw_q: Dict[int, float] = {}
        raw_reason: Dict[int, float] = {}
        raw_hr: Dict[int, float] = {}
        raw_pr: Dict[int, float] = {}
        raw_div: Dict[int, float] = {}
        for cid in remaining:
            vec = combined_head[:, cid] if combined_head is not None else torch.zeros(num_heads, device=device)
            deficit = (1.0 - coverage).clamp(0.0, 1.0)
            raw_r[cid] = _rh_weighted_mean(vec * deficit, wr)
            raw_q[cid] = _rh_weighted_mean(vec * deficit, wq)
            raw_reason[cid] = _rh_weighted_mean(vec * deficit, wreason)
            raw_hr[cid] = max(head_red(vec, active_head_vec), head_red(vec, rescue_head_vec))
            raw_pr[cid] = _maxcos(combined_reps, cid, list(active_ids) + rescued)
            raw_div[cid] = max(0.0, 1.0 - raw_hr[cid])
        zr_gain = _rh_dict_zscore(raw_r)
        zq_gain = _rh_dict_zscore(raw_q)
        zreason_gain = _rh_dict_zscore(raw_reason)
        zh_red = _rh_dict_zscore(raw_hr)
        zp_red = _rh_dict_zscore(raw_pr)
        best_id: Optional[int] = None
        best_score = -float("inf")
        for cid in remaining:
            pages = [int(p) for p in combined[cid].get("fixed_page_ids", [])]
            primary = min(pages) if pages else -1
            if mode == "rh_learned":
                utility = max((learned_map.get(p, float("-inf")) for p in pages), default=float("-inf"))
                if utility == float("-inf"):
                    utility = base_drm(cid)
                utility -= lambda_h * zh_red.get(cid, 0.0) + lambda_p * zp_red.get(cid, 0.0)
            elif mode == "rh_oracle":
                utility = 2.0 * answer_by_candidate.get(cid, 0.0) + 0.25 * base_drm(cid) - lambda_p * zp_red.get(cid, 0.0)
            else:
                utility = (
                    alpha * base_drm(cid)
                    + beta_r * zr_gain.get(cid, 0.0)
                    + beta_q * zq_gain.get(cid, 0.0)
                    + beta_reason * zreason_gain.get(cid, 0.0)
                    - lambda_h * zh_red.get(cid, 0.0)
                    - lambda_p * zp_red.get(cid, 0.0)
                )
            incremental_cost = max(1, _virtual_incremental_cost(combined, selected_all, cid, cost_mode))
            score = utility / math.sqrt(float(incremental_cost)) if cost_adjusted else utility
            vec = combined_head[:, cid] if combined_head is not None else torch.zeros(num_heads, device=device)
            feature_rows[cid] = {
                "task": str((cfg.get("audit_context") or {}).get("task", "")),
                "sample_id": str((cfg.get("audit_context") or {}).get("sample_id", "")),
                "method": str((cfg.get("audit_context") or {}).get("method", "")),
                "layer_id": int(cfg.get("layer_idx", 0) or 0),
                "page_id": primary,
                "candidate_source": str(combined[cid].get("source_type", combined[cid].get("strategy_source", "fixed"))),
                "token_cost": int(incremental_cost),
                "drm_score": round(base_drm(cid), 7),
                "query_similarity": round(q_by_candidate.get(cid, 0.0), 7),
                "retrieval_head_mass": round(_rh_weighted_mean(vec, wr), 7),
                "reasoning_head_mass": round(_rh_weighted_mean(vec, wreason), 7),
                "query_focused_head_mass": round(_rh_weighted_mean(vec, wq), 7),
                "active_head_coverage": round(_rh_weighted_mean(coverage, wr), 7),
                "active_head_deficit": round(_rh_weighted_mean((1.0 - coverage).clamp(0.0, 1.0), wr), 7),
                "head_coverage_gain": round(raw_r.get(cid, 0.0), 7),
                "retrieval_head_gain": round(raw_r.get(cid, 0.0), 7),
                "reasoning_head_gain": round(raw_reason.get(cid, 0.0), 7),
                "query_focused_head_gain": round(raw_q.get(cid, 0.0), 7),
                "head_diversity_gain": round(raw_div.get(cid, 0.0), 7),
                "head_redundancy_with_active": round(head_red(vec, active_head_vec), 7),
                "head_redundancy_with_rescue": round(head_red(vec, rescue_head_vec), 7),
                "page_redundancy": round(raw_pr.get(cid, 0.0), 7),
                "final_rh_utility": round(float(utility), 7),
                "selected": 0,
                "rh_mode": mode,
                "answer_overlap_oracle_diag": round(answer_by_candidate.get(cid, 0.0), 7),
            }
            if score > best_score:
                best_score = score
                best_id = cid
        if best_id is None:
            break
        candidate_all = set(selected_all)
        candidate_all.add(best_id)
        candidate_cost = _virtual_selected_cost(combined, candidate_all, cost_mode)
        if selected_all != set(active_ids) and candidate_cost > hard:
            err = abs(candidate_cost - target)
            if err < best_err:
                best_rescue = rescued + [best_id]
                best_err = err
            break
        rescued.append(best_id)
        selected_all.add(best_id)
        feature_rows[best_id]["selected"] = 1
        if combined_head is not None:
            vec = combined_head[:, best_id]
            coverage = (coverage + vec).clamp(0.0, 1.0)
            rescue_head_vec = rescue_head_vec + vec
        err = abs(_virtual_selected_cost(combined, selected_all, cost_mode) - target)
        if err < best_err:
            best_rescue = list(rescued)
            best_err = err
    rescued = best_rescue if best_rescue else rescued
    selected_set = set(rescued)
    for cid, row in feature_rows.items():
        row["selected"] = 1 if cid in selected_set else 0
    dump_path = cfg.get("rh_candidate_dump_path")
    if dump_path and str(cfg.get("phase", "")) != "decode":
        try:
            import os as _os
            _os.makedirs(_os.path.dirname(_os.path.abspath(str(dump_path))), exist_ok=True)
            with open(str(dump_path), "a", encoding="utf-8") as f:
                for cid in candidate_ids:
                    if cid in feature_rows:
                        f.write(json.dumps(feature_rows[cid], ensure_ascii=False) + "\n")
        except Exception:
            pass
    return rescued, {
        "rescue_selector": mode,
        "rh_mode": mode,
        "rescued_count": len(rescued),
        "candidate_pool_size": len(candidate_ids),
        "rh_profile_heads_retrieval": int((wr > 0).sum().item()) if wr.numel() else 0,
        "rh_profile_heads_reasoning": int((wreason > 0).sum().item()) if wreason.numel() else 0,
        "rh_profile_heads_query": int((wq > 0).sum().item()) if wq.numel() else 0,
    }


def _structural_rescue_select(
    *,
    cfg: Dict[str, Any],
    combined: List[Dict[str, Any]],
    candidate_ids: List[int],
    active_ids: List[int],
    combined_delta: torch.Tensor,
    g_tensor: torch.Tensor,
    r_tensor: torch.Tensor,
    combined_reps: Optional[torch.Tensor],
    offset: int,
    rescue_blocks: List[Dict[str, Any]],
    target_final: int,
    m_pages: int,
    active_query_covered: set,
    query_vocab_size: int,
    cost_mode: str,
    device: torch.device,
    original_block_ids: Optional[List[int]] = None,
    head_page_mass: Optional[torch.Tensor] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Unified structural rescue selector covering directions 1-5.

    All variants keep the active set fixed and only differ in how rescue
    candidates are scored / gated / budget-allocated / verified. Selection is
    always to a matched K+M token budget using incremental fixed-page cost.
    """
    selector = str(cfg.get("rescue_selector", "drm_mmr") or "drm_mmr")
    if selector.startswith("rh_"):
        return _rh_structural_rescue_select(
            cfg=cfg,
            combined=combined,
            candidate_ids=candidate_ids,
            active_ids=active_ids,
            original_block_ids=list(original_block_ids or []),
            head_page_mass=head_page_mass,
            combined_delta=combined_delta,
            g_tensor=g_tensor,
            r_tensor=r_tensor,
            combined_reps=combined_reps,
            offset=offset,
            rescue_blocks=rescue_blocks,
            target_final=target_final,
            cost_mode=cost_mode,
            device=device,
        )
    page_utility_map = cfg.get("page_utility_map") or {}
    ac_page_utility_map = cfg.get("ac_page_utility_map") or {}
    if page_utility_map and not isinstance(next(iter(page_utility_map)), int):
        page_utility_map = {int(k): float(v) for k, v in page_utility_map.items()}
    if ac_page_utility_map and not isinstance(next(iter(ac_page_utility_map)), int):
        ac_page_utility_map = {int(k): float(v) for k, v in ac_page_utility_map.items()}
    ac_cost_adj = bool(int(cfg.get("ac_cost_adjusted", 1) or 0))
    gate_mode = str(cfg.get("rescue_gate_mode", "none") or "none")
    block_budget_mode = str(cfg.get("block_budget_mode", "none") or "none")
    verifier_mode = str(cfg.get("verifier_mode", "none") or "none")
    alpha = float(cfg.get("alpha_diff", 0.5))
    lam_red = float(cfg.get("lambda_red", 0.25))
    lam_len = float(cfg.get("lambda_len", 0.1))
    beta_q = float(cfg.get("beta_query", 0.2))
    gamma_cov = float(cfg.get("coverage_gamma", 0.5))
    mu_noise = float(cfg.get("noise_mu", 0.2))
    eta_rescue = float(cfg.get("eta_rescue", 0.25))
    eta_active = float(cfg.get("eta_active", 0.25))
    vocab = max(1, int(query_vocab_size))

    z_d = _zscore_over(combined_delta, candidate_ids, device)
    z_g = _zscore_over(g_tensor, candidate_ids, device)
    z_r = _zscore_over(r_tensor, candidate_ids, device)
    qrel = {cid: float(rescue_blocks[cid - offset].get("rescue_query_score", 0.0)) for cid in candidate_ids}
    z_q = _zscore_over(torch.tensor([qrel[c] for c in candidate_ids], dtype=torch.float32) if candidate_ids else torch.zeros(0),
                       list(range(len(candidate_ids))), device)
    z_q = {candidate_ids[k]: v for k, v in z_q.items()}
    answer = {cid: float(rescue_blocks[cid - offset].get("rescue_answer_score", 0.0)) for cid in candidate_ids}
    covered_map = {cid: set(int(x) for x in rescue_blocks[cid - offset].get("rescue_query_covered_idx", [])) for cid in candidate_ids}
    len_pen = {cid: math.log(1.0 + float(combined[cid].get("expanded_page_tokens", _BLOCK_SIZE)) / _BLOCK_SIZE) for cid in candidate_ids}
    repnorm_raw = {cid: (float(torch.norm(combined_reps[cid].float()).item()) if combined_reps is not None and cid < combined_reps.shape[0] else 0.0) for cid in candidate_ids}
    z_rn = _zscore_over(torch.tensor([repnorm_raw[c] for c in candidate_ids], dtype=torch.float32) if candidate_ids else torch.zeros(0),
                        list(range(len(candidate_ids))), device)
    z_rn = {candidate_ids[k]: v for k, v in z_rn.items()}

    def base_drm(cid: int) -> float:
        return z_d.get(cid, 0.0) + alpha * z_g.get(cid, 0.0) - lam_red * z_r.get(cid, 0.0)

    def noise_risk(cid: int) -> float:
        # high functional score but no query relevance -> likely off-topic high-norm page
        return max(0.0, z_d.get(cid, 0.0)) * (1.0 - min(1.0, qrel.get(cid, 0.0)))

    # ---- block budget soft reweighting (direction 4) ----
    block_tokens = max(_BLOCK_SIZE, int(cfg.get("block_budget_block_tokens", 256) or 256))
    block_pages = max(1, block_tokens // _BLOCK_SIZE)
    block_log_w = float(cfg.get("block_budget_log_weight", 0.5))
    block_bonus = {cid: 0.0 for cid in candidate_ids}
    block_meta: Dict[str, Any] = {}
    if block_budget_mode not in ("none", ""):
        block_of: Dict[int, int] = {}
        for cid in candidate_ids:
            pages = combined[cid].get("fixed_page_ids", [])
            center = (sum(int(p) for p in pages) / max(1, len(pages))) if pages else 0
            block_of[cid] = int(center // block_pages)
        blocks_seen = sorted(set(block_of.values()))
        weight: Dict[int, float] = {}
        active_pages_set = set()
        for aid in active_ids:
            active_pages_set.update(int(p) for p in combined[aid].get("fixed_page_ids", []))
        for b in blocks_seen:
            cids = [c for c in candidate_ids if block_of[c] == b]
            if block_budget_mode == "uniform":
                weight[b] = 1.0
            elif block_budget_mode == "query_weighted":
                weight[b] = max(qrel[c] for c in cids) if cids else 0.0
            elif block_budget_mode == "drm_peak":
                weight[b] = max(base_drm(c) for c in cids) if cids else 0.0
            elif block_budget_mode == "coverage_gap":
                block_pages_ids = set()
                for c in cids:
                    block_pages_ids.update(int(p) for p in combined[c].get("fixed_page_ids", []))
                cov_frac = len(block_pages_ids & active_pages_set) / max(1, len(block_pages_ids))
                weight[b] = 1.0 - cov_frac
            else:
                weight[b] = 1.0
        wmin = min(weight.values()) if weight else 0.0
        for cid in candidate_ids:
            w = weight.get(block_of[cid], 0.0) - wmin
            block_bonus[cid] = block_log_w * math.log1p(max(0.0, w))
        block_meta = {"block_budget_mode": block_budget_mode, "num_blocks": len(blocks_seen)}

    # ---- verifier rerank: restrict pool to Stage-A top-N by DRM (direction 5) ----
    pool = list(candidate_ids)
    verifier_meta: Dict[str, Any] = {}
    if verifier_mode not in ("none", "") and pool:
        topn = max(m_pages, int(round(m_pages * float(cfg.get("verifier_topn_mult", 3)))))
        pool = sorted(pool, key=lambda c: base_drm(c), reverse=True)[:max(1, topn)]
        verifier_meta = {"verifier_mode": verifier_mode, "stageA_topn": len(pool)}

    # ---- per-step score function per selector ----
    def selector_score(cid: int, rescued: List[int], covered: set) -> float:
        bonus = block_bonus.get(cid, 0.0)
        if verifier_mode == "light":
            cov_gain = len(covered_map[cid] - covered) / vocab
            red = _maxcos(combined_reps, cid, list(active_ids) + rescued)
            return beta_q * z_q.get(cid, 0.0) + gamma_cov * cov_gain - lam_red * red - mu_noise * noise_risk(cid) + bonus
        if verifier_mode == "attn":
            cov_gain = len(covered_map[cid] - covered) / vocab
            red = _maxcos(combined_reps, cid, list(active_ids) + rescued)
            return 0.5 * z_rn.get(cid, 0.0) + beta_q * z_q.get(cid, 0.0) + gamma_cov * cov_gain - lam_red * red - mu_noise * noise_risk(cid) + bonus
        if verifier_mode == "oracle":
            cov_gain = len(covered_map[cid] - covered) / vocab
            return 2.0 * answer.get(cid, 0.0) + 0.5 * cov_gain - 0.2 * _maxcos(combined_reps, cid, list(active_ids) + rescued) + bonus
        if selector == "fup":
            # injected per-page utility (predicted functional utility / proxy)
            pages = combined[cid].get("fixed_page_ids", [])
            util = max((page_utility_map.get(int(p), float("-inf")) for p in pages), default=0.0)
            if util == float("-inf"):
                util = 0.0
            red = _maxcos(combined_reps, cid, list(active_ids) + rescued)
            return util - lam_red * red + bonus
        if selector == "ac_util":
            pages = combined[cid].get("fixed_page_ids", [])
            util = max((ac_page_utility_map.get(int(p), float("-inf")) for p in pages), default=0.0)
            if util == float("-inf"):
                util = base_drm(cid)
            red_r = _maxcos(combined_reps, cid, rescued)
            sc = float(util) - eta_rescue * red_r + bonus
            if ac_cost_adj:
                sc = sc / max(1e-6, len_pen.get(cid, 1.0))
            return sc
        if selector == "drm_only":
            return base_drm(cid) + bonus
        if selector == "query_only":
            return z_q.get(cid, 0.0) + bonus
        if selector == "damage_only":
            return z_d.get(cid, 0.0) + bonus
        if selector == "functional_delta":
            return z_d.get(cid, 0.0) + z_g.get(cid, 0.0) + bonus
        if selector == "coverage_aware":
            cov_gain = len(covered_map[cid] - covered) / vocab
            red = _maxcos(combined_reps, cid, list(active_ids) + rescued)
            return (alpha * base_drm(cid) + beta_q * z_q.get(cid, 0.0) + gamma_cov * cov_gain
                    - lam_red * red - mu_noise * noise_risk(cid) - lam_len * len_pen[cid] + bonus)
        # default drm_mmr
        red_r = _maxcos(combined_reps, cid, rescued)
        red_a = _maxcos(combined_reps, cid, list(active_ids))
        return base_drm(cid) - lam_len * len_pen[cid] - eta_rescue * red_r - eta_active * red_a + bonus

    # ---- greedy to matched token budget ----
    selected_all = set(active_ids)
    rescued: List[int] = []
    covered = set(active_query_covered)
    target = max(1, int(target_final))
    hard = int(math.floor(target * (1.0 + 0.01)))  # gate requires <1% error; keep tight
    best_rescue: List[int] = []
    best_err = abs(_virtual_selected_cost(combined, selected_all, cost_mode) - target)
    guard = 0
    while _virtual_selected_cost(combined, selected_all, cost_mode) < target and guard < len(pool) + 2:
        guard += 1
        best_id = None
        best_sc = -float("inf")
        for cid in pool:
            if cid in selected_all:
                continue
            sc = selector_score(cid, rescued, covered)
            if sc > best_sc:
                best_sc = sc
                best_id = cid
        if best_id is None:
            break
        cand_all = set(selected_all)
        cand_all.add(best_id)
        cc = _virtual_selected_cost(combined, cand_all, cost_mode)
        if selected_all != set(active_ids) and cc > hard:
            err = abs(cc - target)
            if err < best_err:
                best_rescue = rescued + [best_id]
                best_err = err
            break
        rescued.append(best_id)
        selected_all.add(best_id)
        covered |= covered_map.get(best_id, set())
        err = abs(_virtual_selected_cost(combined, selected_all, cost_mode) - target)
        if err < best_err:
            best_rescue = list(rescued)
            best_err = err
    rescued = best_rescue if best_rescue else rescued
    sel_meta = {
        "rescue_selector": selector,
        "rescue_gate_mode": gate_mode,
        "rescued_count": len(rescued),
        "candidate_pool_size": len(candidate_ids),
        **block_meta,
        **verifier_meta,
    }
    return rescued, sel_meta


def _maybe_dump_fup_features(
    *, cfg: Dict[str, Any], combined: List[Dict[str, Any]], candidate_ids: List[int], offset: int,
    sel_active: List[int], active_pages: Set[int], virtual_scores: torch.Tensor,
    combined_delta: torch.Tensor, g_tensor: torch.Tensor, r_tensor: torch.Tensor,
    rescue_blocks: List[Dict[str, Any]], qk_block_scores: torch.Tensor, original_block_ids: List[int],
    vocab: int, m_default: int,
) -> None:
    """Dump per-candidate features once per sample for the FUP pipeline (direction 1)."""
    path = cfg.get("feature_dump_path")
    if not path:
        return
    ctx = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    task = str(ctx.get("task", ""))
    sample_id = str(ctx.get("sample_id", ""))
    if str(cfg.get("phase", "")) == "decode":
        return
    layer_id = int(cfg.get("layer_idx", 0) or 0)
    num_pages = max(1, int((cfg.get("rescue_candidate_layout") or {}).get("metadata", {}).get("num_pages", len(original_block_ids)) or len(original_block_ids)))
    # z-scores over candidate pool (consistent with selector)
    all_ids = list(range(len(combined)))
    zd = _zscore_over(combined_delta, all_ids, combined_delta.device)
    zg = _zscore_over(g_tensor, all_ids, g_tensor.device)
    zr = _zscore_over(r_tensor, all_ids, r_tensor.device)
    qk_per_page = {int(original_block_ids[i]): float(qk_block_scores[i].item())
                   for i in range(min(len(original_block_ids), qk_block_scores.numel()))}
    active_set = set(int(p) for p in active_pages)
    rows = []
    for vid in range(len(combined)):
        blk = combined[vid]
        pages = [int(p) for p in blk.get("fixed_page_ids", [])]
        if not pages:
            continue
        rb = rescue_blocks[vid - offset] if vid >= offset and (vid - offset) < len(rescue_blocks) else {}
        primary = min(pages)
        is_in_active = 1 if set(pages).issubset(active_set) else 0
        qk_mean = sum(qk_per_page.get(p, 0.0) for p in pages) / max(1, len(pages))
        rows.append({
            "task": task, "sample_id": sample_id, "layer_id": layer_id, "page_id": primary,
            "candidate_id": int(vid), "candidate_source": str(blk.get("strategy_source", rb.get("rescue_strategy_source", "fixed"))),
            "fixed_page_ids": pages, "token_cost": int(blk.get("expanded_page_tokens", _BLOCK_SIZE)),
            "is_in_active": is_in_active,
            "is_candidate": 0 if is_in_active else 1,
            "feature_drm_delta": round(zd.get(vid, 0.0), 5),
            "feature_g_diffusion": round(zg.get(vid, 0.0), 5),
            "feature_r_active": round(zr.get(vid, 0.0), 5),
            "feature_query_similarity": round(float(rb.get("rescue_query_score", 0.0)), 5),
            "feature_query_covered_count": int(len(rb.get("rescue_query_covered_idx", []) or [])),
            "feature_coverage_gain_proxy": round(float(len(rb.get("rescue_query_covered_idx", []) or [])) / max(1, vocab), 5),
            "feature_answer_overlap": round(float(rb.get("rescue_answer_score", 0.0)), 5),
            "feature_page_position": round(float(primary) / max(1, num_pages), 5),
            "feature_page_token_len": int(blk.get("expanded_page_tokens", _BLOCK_SIZE)),
            "feature_attention_score": round(qk_mean, 6),
            "answer_score_oracle": round(float(rb.get("rescue_answer_score", 0.0)), 5),
            "K_active": int(len(sel_active)),
            "m_default": int(m_default),
            "num_pages": int(num_pages),
            "context_length": int(ctx.get("context_length", 0) or 0),
        })
    import json as _json
    import os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(str(path))), exist_ok=True)
    with open(str(path), "a", encoding="utf-8") as f:
        for row in rows:
            f.write(_json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_dump_ac_utility_features(
    *, cfg: Dict[str, Any], combined: List[Dict[str, Any]], candidate_ids: List[int], offset: int,
    sel_active: List[int], active_pages: Set[int], virtual_scores: torch.Tensor,
    combined_delta: torch.Tensor, g_tensor: torch.Tensor, r_tensor: torch.Tensor,
    rescue_blocks: List[Dict[str, Any]], combined_reps: Optional[torch.Tensor], active_ids: List[int],
) -> None:
    path = cfg.get("ac_utility_dump_path")
    if not path:
        return
    ctx = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    if str(cfg.get("phase", "")) == "decode":
        return
    task = str(ctx.get("task", ""))
    sample_id = str(ctx.get("sample_id", ""))
    ac_map = cfg.get("ac_page_utility_map") or {}
    fup_map = cfg.get("page_utility_map") or {}
    w1 = float(cfg.get("ac_w_drm", 1.0))
    w2 = float(cfg.get("ac_w_fup", 0.5))
    w3 = float(cfg.get("ac_w_novelty", 0.35))
    w4 = float(cfg.get("ac_w_gap", 0.35))
    w5 = float(cfg.get("ac_w_redundancy", 0.30))
    w6 = float(cfg.get("ac_w_noise", 0.25))
    device = combined_delta.device
    zd = _zscore_over(combined_delta, candidate_ids, device)
    zg = _zscore_over(g_tensor, candidate_ids, device)
    zr = _zscore_over(r_tensor, candidate_ids, device)
    qrel = {cid: float(rescue_blocks[cid - offset].get("rescue_query_score", 0.0)) for cid in candidate_ids if cid >= offset}
    tail_sc = _ac_active_tail_score(virtual_scores, sel_active)
    rows = []
    for cid in candidate_ids:
        blk = combined[cid]
        pages = [int(p) for p in blk.get("fixed_page_ids", [])]
        if not pages:
            continue
        primary = min(pages)
        rb = rescue_blocks[cid - offset] if cid >= offset else {}
        src = str(blk.get("strategy_source", rb.get("rescue_strategy_source", "fixed")))
        drm = zd.get(cid, 0.0) + float(cfg.get("alpha_diff", 0.5)) * zg.get(cid, 0.0) - float(cfg.get("lambda_red", 0.25)) * zr.get(cid, 0.0)
        fup_sc = float(ac_map.get(primary, fup_map.get(primary, 0.0)))
        red_a = _maxcos(combined_reps, cid, list(active_ids))
        novelty = max(0.0, 1.0 - red_a)
        gap = drm - tail_sc
        noise = max(0.0, drm) * (1.0 - min(1.0, qrel.get(cid, 0.0)))
        delta_u = w1 * drm + w2 * fup_sc + w3 * novelty + w4 * gap - w5 * red_a - w6 * noise
        rows.append({
            "task": task, "sample_id": sample_id, "page_id": primary,
            "candidate_source": src, "token_cost": int(blk.get("expanded_page_tokens", _BLOCK_SIZE)),
            "drm_score": round(drm, 5), "fup_score": round(fup_sc, 5),
            "query_similarity": round(qrel.get(cid, 0.0), 5),
            "active_redundancy": round(red_a, 5), "rescue_redundancy": 0.0,
            "novelty_gain": round(novelty, 5), "active_gap_score": round(gap, 5),
            "noise_risk": round(noise, 5), "final_delta_utility": round(delta_u, 5),
            "selected": 0,
        })
    import json as _json
    import os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(str(path))), exist_ok=True)
    with open(str(path), "a", encoding="utf-8") as f:
        for row in rows:
            f.write(_json.dumps(row, ensure_ascii=False) + "\n")


def _apply_guarded_swap(
    *, cfg: Dict[str, Any], combined: List[Dict[str, Any]], candidate_ids: List[int], sel_active: List[int],
    rescued_combined: List[int], virtual_scores: torch.Tensor, combined_delta: torch.Tensor,
    g_tensor: torch.Tensor, r_tensor: torch.Tensor, rescue_blocks: List[Dict[str, Any]],
    offset: int, device: torch.device,
) -> Tuple[List[int], Dict[str, Any]]:
    """Swap low-utility active TAIL pages for higher-utility candidates (budget-neutral).
    Active CORE is protected. Runs on top of the Add rescue set."""
    core_ratio = float(cfg.get("active_core_ratio", 0.8))
    max_swaps = int(cfg.get("max_swaps", 2))
    tau = float(cfg.get("swap_tau", 0.05))
    util_src = str(cfg.get("swap_utility_source", "drm") or "drm")
    if not sel_active or max_swaps <= 0:
        return rescued_combined, {"swap_count": 0, "swapped_out_pages": [], "swapped_in_pages": [], "swap_utility_source": util_src}

    page_utility_map = cfg.get("page_utility_map") or {}
    if page_utility_map and not isinstance(next(iter(page_utility_map)), int):
        page_utility_map = {int(k): float(v) for k, v in page_utility_map.items()}

    alpha = float(cfg.get("alpha_diff", 0.5))
    lam_red = float(cfg.get("lambda_red", 0.25))
    all_ids = list(range(len(combined)))
    zd = _zscore_over(combined_delta, all_ids, device)
    zg = _zscore_over(g_tensor, all_ids, device)
    zr = _zscore_over(r_tensor, all_ids, device)

    def util(vid: int) -> float:
        if util_src == "injected" and page_utility_map:
            pages = combined[vid].get("fixed_page_ids", [])
            vals = [page_utility_map[int(p)] for p in pages if int(p) in page_utility_map]
            return max(vals) if vals else 0.0
        # drm functional utility (delta_o + diffusion gain - redundancy); differs from
        # the qk active-selection score, so a candidate can outrank an active tail page.
        return zd.get(vid, 0.0) + alpha * zg.get(vid, 0.0) - lam_red * zr.get(vid, 0.0)

    # core/tail by qk active-selection score (the score SlimInfer used to pick active)
    active_sorted = sorted(sel_active, key=lambda v: float(virtual_scores[v].item()) if v < virtual_scores.numel() else 0.0)
    k = len(active_sorted)
    n_core = int(math.ceil(core_ratio * k))
    tail = active_sorted[: max(0, k - n_core)]  # lowest-score active pages (not core)

    rescued = list(rescued_combined)
    available = [c for c in candidate_ids if c not in set(rescued)]
    available.sort(key=util, reverse=True)
    swapped_out_pages: List[int] = []
    swapped_in_pages: List[int] = []
    margins: List[float] = []
    tail_pool = list(tail)
    for _ in range(max_swaps):
        if not tail_pool or not available:
            break
        a = min(tail_pool, key=util)  # weakest tail page
        r = available[0]
        if util(r) > util(a) + tau:
            rescued.append(r)
            swapped_in_pages.extend(int(p) for p in combined[r].get("fixed_page_ids", []))
            swapped_out_pages.extend(int(p) for p in combined[a].get("fixed_page_ids", []))
            margins.append(util(r) - util(a))
            tail_pool.remove(a)
            available.pop(0)
        else:
            break
    return rescued, {
        "swap_count": int(len(margins)),
        "swapped_out_pages": swapped_out_pages,
        "swapped_in_pages": swapped_in_pages,
        "active_core_size": int(n_core),
        "active_tail_size": int(len(tail)),
        "swap_utility_margin": round(float(sum(margins) / len(margins)), 5) if margins else 0.0,
        "swap_utility_source": util_src,
    }


def _scatter_virtual_scores_to_pages(
    virtual_scores: torch.Tensor,
    original_block_ids: List[int],
    device: torch.device,
) -> torch.Tensor:
    length = (max(original_block_ids) + 1) if original_block_ids else 0
    out = torch.zeros(length, dtype=torch.float32, device=device)
    limit = min(int(virtual_scores.numel()), len(original_block_ids))
    for vid in range(limit):
        page = int(original_block_ids[vid])
        if 0 <= page < out.numel():
            out[page] = virtual_scores[vid]
    return out


def _scatter_virtual_reps_to_pages(
    virtual_reps: Optional[torch.Tensor],
    original_block_ids: List[int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if virtual_reps is None or virtual_reps.numel() == 0 or not original_block_ids:
        return None
    length = max(original_block_ids) + 1
    dim = int(virtual_reps.shape[-1])
    out = torch.zeros(length, dim, dtype=torch.float32, device=device)
    limit = min(int(virtual_reps.shape[0]), len(original_block_ids))
    for vid in range(limit):
        page = int(original_block_ids[vid])
        if 0 <= page < out.shape[0]:
            out[page] = virtual_reps[vid]
    return out


def _structured_microblock_rescue_in_fixed_virtual_path(
    cfg: Dict[str, Any],
    *,
    selected: Set[int],
    inactive: List[int],
    original_block_ids: List[int],
    virtual_delta: torch.Tensor,
    virtual_reps: Optional[torch.Tensor],
    qk_block_scores: torch.Tensor,
    target_m: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any]]:
    """Microblock-first structured rescue for fixed virtual-block add path (layer-26 real hook)."""
    diagnostics: Dict[str, List[str]] = {"warnings": []}
    score_cfg = dict(cfg)
    score_cfg["rescue_score"] = str(cfg.get("rescue_score", "") or "")
    active_block_ids = sorted(int(original_block_ids[vid]) for vid in sorted(selected) if 0 <= vid < len(original_block_ids))
    inactive_block_ids = [int(original_block_ids[vid]) for vid in inactive if 0 <= vid < len(original_block_ids)]
    if not inactive_block_ids or target_m <= 0:
        marker = {
            "sample_id": (cfg.get("audit_context") or {}).get("sample_id", ""),
            "task": (cfg.get("audit_context") or {}).get("task", ""),
            "layer_id": cfg.get("layer_idx", ""),
            "method": score_cfg["rescue_score"],
            "entered_structured_selector": True,
            "entered_real_fixed_block_path": True,
            "entered_cache_construction_path": False,
            "used_microblock_first_selector": False,
            "used_candidate_score_dump": False,
            "used_final_selection_dump": False,
            "used_evidence_density_selector": False,
        }
        marker_written = append_path_marker(score_cfg, marker)
        enforce_strict_structured_path_check(
            score_cfg, path_marker=marker, mb_meta={}, cand_dump_written=False,
            final_dump_written=False, marker_dump_written=marker_written,
        )
        return [], {"microblock_select": True, "selected_microblock_count": 0}

    base_global = _scatter_virtual_scores_to_pages(virtual_delta, original_block_ids, device)
    qk_global = _scatter_local_scores(qk_block_scores, original_block_ids, device)
    reps_global = _scatter_virtual_reps_to_pages(virtual_reps, original_block_ids, device)
    rescued_blocks, mb_meta, candidate_rows = select_structured_rescue_via_microblocks(
        cfg=score_cfg,
        base_global=base_global,
        qk_global=qk_global,
        reps=reps_global,
        active_ids=active_block_ids,
        inactive_ids=inactive_block_ids,
        target_m=target_m,
        diagnostics=diagnostics,
    )
    page_to_vid = {int(original_block_ids[i]): i for i in range(len(original_block_ids))}
    rescued_virtual = [page_to_vid[int(b)] for b in rescued_blocks if int(b) in page_to_vid]
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    method = str(score_cfg.get("rescue_score", "") or "")
    for row in candidate_rows:
        row.update({
            "sample_id": context.get("sample_id", ""),
            "task": context.get("task", ""),
            "layer_id": cfg.get("layer_idx", ""),
            "method": method,
            "block_id": row.get("block_id", row.get("page_id", "")),
        })
    final_row = build_structured_final_selection_row(
        score_cfg,
        active_ids=active_block_ids,
        rescued_ids=rescued_blocks,
        mb_meta=mb_meta,
    )
    cand_written, final_written = append_selection_dumps(score_cfg, candidate_rows, final_row)
    marker = {
        "sample_id": context.get("sample_id", ""),
        "task": context.get("task", ""),
        "layer_id": cfg.get("layer_idx", ""),
        "method": method,
        "entered_structured_selector": True,
        "entered_real_fixed_block_path": True,
        "entered_cache_construction_path": True,
        "used_microblock_first_selector": True,
        "used_candidate_score_dump": bool(cand_written),
        "used_final_selection_dump": bool(final_written),
        "used_evidence_density_selector": bool(
            mb_meta.get("used_evidence_density_selector", False)
        ),
    }
    marker_written = append_path_marker(score_cfg, marker)
    mb_meta.update({
        "rescue_selector": "structured_microblock_fixed_virtual",
        "structured_path_hook": "fixed_virtual_add",
        "path_marker": marker,
        "rescue_policy_warnings": diagnostics.get("warnings", []),
    })
    enforce_strict_structured_path_check(
        score_cfg, path_marker=marker, mb_meta=mb_meta, cand_dump_written=cand_written,
        final_dump_written=final_written, marker_dump_written=marker_written,
    )
    return rescued_virtual, mb_meta


def _calculate_pruning_with_virtual_blocks(
    q_for_pruning: torch.Tensor,
    original_block_ids: List[int],
    cfg: Dict[str, Any],
    block_scores: torch.Tensor,
    qk_block_scores: torch.Tensor,
    delta_o_values: Optional[torch.Tensor],
    represent_values: Optional[torch.Tensor],
    num_key_value_groups: int,
    block_slices: List[Tuple[int, int]],
    must_keep_block_ids: Set[int],
    fixed_keep_tokens: Optional[float],
    head_page_mass: Optional[torch.Tensor] = None,
) -> Optional[Set[int]]:
    strategy = str(cfg.get("block_strategy", "fixed") or "fixed")
    device = q_for_pruning.device
    page_reps = _page_representations_global(represent_values, num_key_value_groups, block_slices, original_block_ids, device)
    virtual_blocks, layout_meta = _virtualize_layout(cfg, original_block_ids, block_scores, page_reps)
    if strategy == "fixed":
        layout_meta = {"fallback_used": False, "fallback_reason": "", "fallback_reason_counts": {}, "coverage_ok": True}
    _write_block_layout_record(cfg, virtual_blocks, layout_meta)
    if not virtual_blocks:
        return None

    n_virtual = len(virtual_blocks)
    target_tokens = max(1, int(cfg.get("target_block_tokens", _BLOCK_SIZE) or _BLOCK_SIZE))
    if fixed_keep_tokens is not None and fixed_keep_tokens > 0:
        base_keep_tokens = float(fixed_keep_tokens) * 1024.0
        num_virtual_to_keep = min(n_virtual, max(1, int(math.ceil(base_keep_tokens / target_tokens))))
    else:
        num_virtual_to_keep = min(n_virtual, max(1, int(math.ceil(len(original_block_ids)))))
    rescue_policy = str(cfg.get("rescue_policy", "none") or "none")
    if strategy == "fixed" and rescue_policy in {
        "budget_neutral_drm_swap",
        "gated_budget_neutral_drm_swap",
        "lssr_policy_b_middle_donor_late_receiver",
    }:
        # The dedicated fixed-page swap path needs the concrete SlimInfer A_K,
        # inactive P set, and protected ids assembled by the caller.
        return None
    budget_mode = str(cfg.get("budget_mode", "block_count") or "block_count")
    active_budget_mode = str(cfg.get("active_budget_mode", budget_mode) or budget_mode)
    rescue_budget_mode = str(cfg.get("rescue_budget_mode", "block_count") or "block_count")
    cost_mode = str(cfg.get("selection_cost", "expanded_page_tokens") or "expanded_page_tokens")
    density_mode = str(cfg.get("score_density_mode", "none") or "none")
    allow_budget_overshoot = bool(int(cfg.get("allow_budget_overshoot", 1) or 0))
    soft_budget_tolerance = float(cfg.get("soft_budget_tolerance", 0.03) or 0.03)
    hard_budget_tolerance = float(cfg.get("hard_budget_tolerance", 0.05) or 0.05)
    if rescue_policy == "sliminfer_larger" and budget_mode != "token_budget":
        ratio = float(cfg.get("rescue_ratio", 0.05))
        extra = int(round(num_virtual_to_keep * ratio))
        if ratio > 0 and num_virtual_to_keep > 0:
            extra = max(1, extra)
        num_virtual_to_keep = min(n_virtual, num_virtual_to_keep + extra)

    virtual_scores = _aggregate_virtual_scores(virtual_blocks, original_block_ids, block_scores, device)
    virtual_delta = _aggregate_virtual_scores(virtual_blocks, original_block_ids, delta_o_values, device) if delta_o_values is not None else virtual_scores
    virtual_reps = _aggregate_virtual_reps(virtual_blocks, page_reps, device)
    if int(cfg.get("boundary_weight_active_score", 0) or 0):
        boundary = _boundary_confidence_tensor(virtual_blocks, device)
        virtual_scores = virtual_scores + float(cfg.get("gamma_active_boundary", 0.1)) * normalize_scores(boundary)
    must_keep_virtual = _virtual_must_keep_ids(virtual_blocks, must_keep_block_ids)
    min_reachable_tokens = _virtual_selected_cost(virtual_blocks, must_keep_virtual, cost_mode)
    max_reachable_tokens = _virtual_selected_cost(virtual_blocks, range(n_virtual), cost_mode)
    target_active_tokens = int(cfg.get("target_active_tokens") or cfg.get("target_token_budget") or 0)
    if fixed_keep_tokens is not None and fixed_keep_tokens > 0 and target_active_tokens <= 0:
        target_active_tokens = int(float(fixed_keep_tokens) * 1024.0)
    if rescue_policy == "sliminfer_larger" and budget_mode == "token_budget":
        target_active_tokens = int(cfg.get("target_token_budget") or target_active_tokens)
    original_target_active_tokens = int(target_active_tokens)
    target_active_tokens = _calibrated_budget(
        cfg,
        "active",
        target_active_tokens,
        min_reachable=min_reachable_tokens,
        max_reachable=max_reachable_tokens,
    )
    cfg["_controller_target_active_tokens"] = int(target_active_tokens)

    if active_budget_mode == "token_budget" or budget_mode == "token_budget":
        selected = _select_virtual_token_budget(
            virtual_scores,
            virtual_blocks,
            target_active_tokens,
            cost_mode=cost_mode,
            density_mode=density_mode,
            must_keep=must_keep_virtual,
            allow_overshoot=allow_budget_overshoot,
            soft_tolerance=soft_budget_tolerance,
            hard_tolerance=hard_budget_tolerance,
        )
        if not allow_budget_overshoot:
            selected = _trim_to_token_budget(selected, virtual_scores, virtual_blocks, target_active_tokens, must_keep_virtual, cost_mode)
    else:
        selected = set(_top_virtual_ids(virtual_scores, num_virtual_to_keep)).union(must_keep_virtual)
    if budget_mode != "token_budget" and len(selected) > num_virtual_to_keep:
        removable = [vid for vid in _top_virtual_ids(-virtual_scores, len(selected)) if vid in selected and vid not in must_keep_virtual]
        for vid in removable:
            if len(selected) <= num_virtual_to_keep:
                break
            selected.discard(int(vid))
    original_virtual_selected_for_dc = set(selected)
    if cfg.get("variant_method") == "dc_sliminfer":
        selected, dc_active_meta = _select_dc_sliminfer_virtual_ids(
            cfg=cfg,
            virtual_blocks=virtual_blocks,
            virtual_scores=virtual_scores,
            original_virtual_selected_ids=original_virtual_selected_for_dc,
            must_keep_virtual_ids=must_keep_virtual,
            num_virtual_to_keep=num_virtual_to_keep,
        )
        layout_meta = {**layout_meta, "dc_active_selection_meta": dc_active_meta}
    elif cfg.get("variant_method") == "adaptive_dc_sliminfer_v3":
        _adv3_gate_cfg = cfg.get("adaptive_dc_v3_gate") or {}
        gate_apply_layer = int(_adv3_gate_cfg.get("gate_apply_layer", cfg.get("gate_apply_layer", 0)) or 0)
        current_layer = int(cfg.get("layer_idx", 0) or 0)
        is_gate_layer = (gate_apply_layer <= 0) or (current_layer >= gate_apply_layer)
        cached_gate = cfg.get("_adaptive_dc_v3_gate_cached")
        has_cached = isinstance(cached_gate, dict) and "triggered" in cached_gate
        block_texts = cfg.get("block_texts") or (cfg.get("audit_context") or {}).get("block_texts") or []
        gate_num_blocks = max(len(block_texts), len(virtual_blocks), int(cfg.get("_virtual_layout_num_blocks", 0) or 0))

        def _adv3_virtual_gate_and_select(apply_dc: bool) -> Dict[str, Any]:
            dc_candidate, dc_active_meta = _adaptive_dc_v3_run_virtual_dc_selector(
                cfg=cfg,
                virtual_blocks=virtual_blocks,
                virtual_scores=virtual_scores,
                original_virtual_selected_ids=original_virtual_selected_for_dc,
                must_keep_virtual_ids=must_keep_virtual,
                num_virtual_to_keep=num_virtual_to_keep,
            )
            gate_result = _adaptive_dc_v3_compute_gate(
                cfg=cfg,
                original_selected_block_ids=set(int(x) for x in original_virtual_selected_for_dc),
                dc_selected_block_ids=set(int(x) for x in dc_candidate),
                num_blocks=gate_num_blocks,
            )
            cfg["_adaptive_dc_v3_gate_last"] = gate_result
            _adaptive_dc_v3_flush_deferred_selection_audit(cfg)
            nonlocal selected
            if apply_dc and not gate_result.get("bug_stop") and gate_result.get("triggered"):
                selected = dc_candidate
            layout_meta["dc_active_selection_meta"] = dc_active_meta
            return gate_result

        if has_cached:
            gate_result = cached_gate
            cfg["_adaptive_dc_v3_gate_last"] = gate_result
            if gate_result.get("triggered"):
                dc_candidate, dc_active_meta = _adaptive_dc_v3_run_virtual_dc_selector(
                    cfg=cfg,
                    virtual_blocks=virtual_blocks,
                    virtual_scores=virtual_scores,
                    original_virtual_selected_ids=original_virtual_selected_for_dc,
                    must_keep_virtual_ids=must_keep_virtual,
                    num_virtual_to_keep=num_virtual_to_keep,
                )
                _adaptive_dc_v3_flush_deferred_selection_audit(cfg)
                selected = dc_candidate
                layout_meta["dc_active_selection_meta"] = dc_active_meta
            layout_meta["adaptive_dc_v3_gate"] = gate_result
        elif is_gate_layer:
            gate_result = _adv3_virtual_gate_and_select(apply_dc=True)
            cfg["_adaptive_dc_v3_gate_cached"] = gate_result
            layout_meta["adaptive_dc_v3_gate"] = gate_result
        else:
            gate_result = {"triggered": False, "reason": f"pre_gate_layer:{current_layer}<{gate_apply_layer}",
                           "evidence_dispersion_score": None, "evidence_dispersion_threshold": None,
                           "entity_coverage_gap": None, "pruned_query_overlap_max": None,
                           "selected_query_overlap_mean": None, "selected_jaccard_vs_original": None,
                           "bug_stop": False}
            layout_meta["adaptive_dc_v3_gate"] = gate_result
            cfg["_adaptive_dc_v3_gate_last"] = gate_result

    if cfg.get("head_profile_dump_path") and head_page_mass is not None:
        profile_rescue_blocks, _ = _candidate_blocks_from_rescue_layout(cfg, original_block_ids)
        _maybe_dump_head_profile_features(
            cfg=cfg,
            original_block_ids=original_block_ids,
            head_page_mass=head_page_mass,
            active_pages=_expand_virtual_to_fixed_pages(virtual_blocks, selected),
            rescue_blocks=profile_rescue_blocks,
        )

    rescued: List[int] = []
    score_meta: Dict[str, Any] = {}
    final_virtual = set(selected)
    final_budget_type = "K+M" if rescue_policy == "sliminfer_larger" else "K"
    if rescue_policy == "sliminfer_larger" and budget_mode == "token_budget":
        target = int(cfg.get("target_token_budget") or cfg.get("target_final_tokens") or target_active_tokens)
        target = _calibrated_budget(
            cfg,
            "final",
            target,
            min_reachable=min_reachable_tokens,
            max_reachable=max_reachable_tokens,
        )
        cfg["_controller_target_final_tokens"] = int(target)
        selected = _select_virtual_token_budget(
            virtual_scores,
            virtual_blocks,
            target,
            cost_mode=cost_mode,
            density_mode=density_mode,
            must_keep=must_keep_virtual,
            allow_overshoot=allow_budget_overshoot,
            soft_tolerance=soft_budget_tolerance,
            hard_tolerance=hard_budget_tolerance,
        )
        final_virtual = set(selected)
        final_budget_type = "K+M"
    if rescue_policy == "add":
        score_name = str(cfg.get("rescue_score", "drm") or "drm")
        target_final_tokens = int(cfg.get("target_final_tokens") or 0)
        if target_final_tokens <= 0 and cfg.get("target_token_budget"):
            target_final_tokens = int(cfg.get("target_token_budget") or 0)
        if target_final_tokens <= 0 and fixed_keep_tokens is not None and fixed_keep_tokens > 0:
            target_final_tokens = int(float(fixed_keep_tokens) * 1024.0 * (1.0 + float(cfg.get("rescue_ratio", 0.05))))
        original_target_final_tokens = int(target_final_tokens)
        target_final_tokens = _calibrated_budget(
            cfg,
            "final",
            target_final_tokens,
            min_reachable=min_reachable_tokens,
            max_reachable=max_reachable_tokens,
        )
        cfg["_controller_target_final_tokens"] = int(target_final_tokens)

        if strategy == "fixed_active_rescue":
            # Structural rescue path (directions 1-5): fixed active set stays fixed;
            # only the selector / scoring / budget / gating vary. Candidate pool is a
            # precomputed layout; selection is matched to fixed DRM-Add K+M token budget
            # via incremental fixed-page cost.
            rescue_blocks, rescue_meta = _candidate_blocks_from_rescue_layout(cfg, original_block_ids)
            active_pages = _expand_virtual_to_fixed_pages(virtual_blocks, selected)
            k_pages = len(active_pages)
            rescue_ratio = float(cfg.get("rescue_ratio", 0.05))
            m_default = int(round(num_virtual_to_keep * rescue_ratio))
            if rescue_ratio > 0 and num_virtual_to_keep > 0:
                m_default = max(1, m_default)
            vocab = max(1, int(rescue_meta.get("query_vocab_size", 0) or 0))

            # ---- active query coverage (for gating + coverage selectors) ----
            active_query_covered: Set[int] = set()
            for blk in rescue_blocks:
                pages = {int(p) for p in blk.get("fixed_page_ids", [])}
                if pages and pages.issubset(active_pages):
                    active_query_covered.update(int(x) for x in blk.get("rescue_query_covered_idx", []))
            active_cov_frac = len(active_query_covered) / vocab if vocab else 1.0

            # ---- gate: legacy rescue_gate_mode or AC-Gated ac_gate_type ----
            gate_mode = str(cfg.get("rescue_gate_mode", "none") or "none")
            ac_gate_type = str(cfg.get("ac_gate_type", "no_gate") or "no_gate")
            sel_active = sorted(selected)
            act_scores = virtual_scores[torch.tensor(sel_active, dtype=torch.long, device=device)] if sel_active else torch.zeros(1, device=device)
            a_std = float(act_scores.std(unbiased=False).item()) if act_scores.numel() > 1 else 0.0
            a_mean = float(act_scores.abs().mean().item()) + 1e-6
            conc = a_std / a_mean
            flat_norm = 1.0 / (1.0 + conc)
            need_score_legacy = 0.6 * (1.0 - active_cov_frac) + 0.4 * flat_norm
            gate_threshold = float(cfg.get("rescue_gate_threshold", 0.4))
            ctx_task = str((cfg.get("audit_context") or {}).get("task", ""))
            # precompute candidate top utility for margin gate (DRM peak on pool)
            offset_pre = len(virtual_blocks)
            cand_pre: List[int] = []
            for i, blk in enumerate(rescue_blocks):
                pages = {int(p) for p in blk.get("fixed_page_ids", [])}
                if pages and pages.issubset(active_pages):
                    continue
                cand_pre.append(offset_pre + i)
            tail_sc = _ac_active_tail_score(virtual_scores, sel_active)
            best_cand_sc = tail_sc
            if cand_pre:
                zd_pre = _zscore_over(
                    _aggregate_virtual_scores(rescue_blocks, original_block_ids, delta_o_values, device)
                    if delta_o_values is not None
                    else _aggregate_virtual_scores(rescue_blocks, original_block_ids, block_scores, device),
                    cand_pre, device,
                )
                alpha_p = float(cfg.get("alpha_diff", 0.5))
                best_cand_sc = max((zd_pre.get(c, 0.0) for c in cand_pre), default=tail_sc)
            ac_map_pre = cfg.get("ac_page_utility_map") or {}
            if ac_map_pre and cand_pre:
                for i, blk in enumerate(rescue_blocks):
                    for p in blk.get("fixed_page_ids", []):
                        best_cand_sc = max(best_cand_sc, float(ac_map_pre.get(int(p), best_cand_sc)))
            fup_map_pre = cfg.get("page_utility_map") or {}
            fup_peak = 0.0
            if fup_map_pre:
                fup_peak = max((float(fup_map_pre.get(int(p), 0.0)) for p in fup_map_pre), default=0.0)
            top_margin = float(best_cand_sc - tail_sc)
            need_score = _ac_compute_need_score(
                cfg=cfg, active_cov_frac=active_cov_frac, conc=conc, flat_norm=flat_norm,
                fup_peak=fup_peak, top_margin=top_margin, task=ctx_task,
            )
            if ac_gate_type != "no_gate":
                m_pages, gate_threshold, ac_gate_type = _ac_gate_m_pages(
                    cfg=cfg, ac_gate_type=ac_gate_type, m_default=m_default,
                    need_score=need_score, top_margin=top_margin, task=ctx_task,
                )
                gate_mode = ac_gate_type
            elif gate_mode == "binary":
                m_pages = m_default if need_score_legacy >= gate_threshold else 0
                need_score = need_score_legacy
            elif gate_mode == "dynamic_M":
                if need_score_legacy < 0.25:
                    m_pages = 0
                elif need_score_legacy < gate_threshold:
                    m_pages = max(1, int(math.ceil(m_default / 2)))
                elif need_score_legacy < 0.75:
                    m_pages = m_default
                else:
                    m_pages = m_default * 2
                need_score = need_score_legacy
            else:
                m_pages = m_default
                need_score = need_score_legacy
            cfg["_ac_need_rescue"] = int(m_pages > 0)
            cfg["_ac_gate_threshold"] = float(gate_threshold)

            target_final = int((k_pages + m_pages) * _BLOCK_SIZE)
            cfg["target_final_tokens"] = int(target_final)
            cfg["_controller_target_final_tokens"] = int(target_final)

            offset = len(virtual_blocks)
            combined = list(virtual_blocks) + [
                {**b, "virtual_id": offset + i} for i, b in enumerate(rescue_blocks)
            ]
            candidate_delta = (
                _aggregate_virtual_scores(rescue_blocks, original_block_ids, delta_o_values, device)
                if delta_o_values is not None
                else _aggregate_virtual_scores(rescue_blocks, original_block_ids, block_scores, device)
            )
            candidate_reps = _aggregate_virtual_reps(rescue_blocks, page_reps, device)
            combined_delta = torch.cat([virtual_delta, candidate_delta], dim=0) if candidate_delta.numel() else virtual_delta
            if virtual_reps is not None and candidate_reps is not None and candidate_reps.numel():
                combined_reps = torch.cat([virtual_reps, candidate_reps], dim=0)
            else:
                combined_reps = virtual_reps
            candidate_ids: List[int] = []
            for i, blk in enumerate(rescue_blocks):
                pages = {int(p) for p in blk.get("fixed_page_ids", [])}
                if pages and pages.issubset(active_pages):
                    continue
                candidate_ids.append(offset + i)

            # Always compute DRM components (cheap) so feature dump / fup / swap have them.
            if candidate_ids:
                _, drm_meta = drm_s0_score(
                    combined_delta, combined_reps, sel_active, candidate_ids,
                    alpha_diff=float(cfg.get("alpha_diff", 0.5)),
                    lambda_red=float(cfg.get("lambda_red", 0.25)),
                    diagnostics={"warnings": []},
                )
            else:
                drm_meta = {"G_diffusion": torch.zeros(len(combined), device=device), "R_active": torch.zeros(len(combined), device=device)}

            # ---- FUP / Guarded-Swap hook: feature dump (side effect, once per sample) ----
            if cfg.get("feature_dump_path"):
                _maybe_dump_fup_features(
                    cfg=cfg, combined=combined, candidate_ids=candidate_ids, offset=offset,
                    sel_active=sel_active, active_pages=active_pages, virtual_scores=virtual_scores,
                    combined_delta=combined_delta, g_tensor=drm_meta["G_diffusion"], r_tensor=drm_meta["R_active"],
                    rescue_blocks=rescue_blocks, qk_block_scores=qk_block_scores, original_block_ids=original_block_ids,
                    vocab=vocab, m_default=m_default,
                )
            if cfg.get("ac_utility_dump_path"):
                _maybe_dump_ac_utility_features(
                    cfg=cfg, combined=combined, candidate_ids=candidate_ids, offset=offset,
                    sel_active=sel_active, active_pages=active_pages, virtual_scores=virtual_scores,
                    combined_delta=combined_delta, g_tensor=drm_meta["G_diffusion"], r_tensor=drm_meta["R_active"],
                    rescue_blocks=rescue_blocks, combined_reps=combined_reps, active_ids=sel_active,
                )

            forced_rescue = cfg.get("forced_rescue_pages")
            guarded_swap = bool(int(cfg.get("guarded_swap", 0) or 0))
            if forced_rescue is not None:
                # Driver-controlled Add-only rescue set (labels / FUP apply / DRM apply).
                forced_set = {int(p) for p in forced_rescue}
                rescue_pages_forced = forced_set - active_pages
                rescued_combined = [cid for cid in candidate_ids
                                    if {int(p) for p in combined[cid].get("fixed_page_ids", [])} & rescue_pages_forced]
                final_fixed = active_pages | rescue_pages_forced
                score_meta = {
                    "rescue_candidate_count": int(len(candidate_ids)),
                    "rescue_selector": "forced",
                    "rescue_m_pages": int(len(rescue_pages_forced)),
                    "rescue_m_default": int(m_default),
                    "forced_rescue": True,
                    "active_fixed_page_ids": sorted(int(p) for p in active_pages),
                    "rescued_fixed_page_ids": sorted(int(p) for p in rescue_pages_forced),
                }
                if not final_fixed:
                    return None
                _write_generic_pruning_audit(
                    cfg=cfg, original_block_ids=original_block_ids, final_ids=final_fixed,
                    block_scores=qk_block_scores, num_blocks_to_keep=min(num_virtual_to_keep, len(original_block_ids)),
                    final_budget_type="K+M", layout_meta={**layout_meta, **{"rescue_layout_meta": rescue_meta}},
                    virtual_blocks=combined, active_virtual_ids=selected,
                    final_virtual_ids=set(selected).union(int(x) for x in rescued_combined),
                    rescued_virtual_ids=[int(x) for x in rescued_combined], score_meta=score_meta,
                )
                return final_fixed

            if m_pages <= 0 or not candidate_ids:
                rescued_combined = []
                sel_extra = {}
            else:
                rescued_combined, sel_extra = _structural_rescue_select(
                    cfg=cfg, combined=combined, candidate_ids=candidate_ids, active_ids=sel_active,
                    combined_delta=combined_delta, g_tensor=drm_meta["G_diffusion"], r_tensor=drm_meta["R_active"],
                    combined_reps=combined_reps, offset=offset, rescue_blocks=rescue_blocks,
                    target_final=target_final, m_pages=m_pages, active_query_covered=active_query_covered,
                    query_vocab_size=vocab, cost_mode=cost_mode, device=device,
                    original_block_ids=original_block_ids, head_page_mass=head_page_mass,
                )

            # ---- Guarded Swap (direction 2): swap low-utility active tail for higher-utility
            # candidates, budget-neutral, on top of the Add rescue. Active core is protected. ----
            swap_meta: Dict[str, Any] = {}
            if guarded_swap and candidate_ids:
                rescued_combined, swap_meta = _apply_guarded_swap(
                    cfg=cfg, combined=combined, candidate_ids=candidate_ids, sel_active=sel_active,
                    rescued_combined=rescued_combined, virtual_scores=virtual_scores,
                    combined_delta=combined_delta, g_tensor=drm_meta["G_diffusion"], r_tensor=drm_meta["R_active"],
                    rescue_blocks=rescue_blocks, offset=offset, device=device,
                )
            score_meta = {
                "D_base_stats": _score_stats_for_block_ids(combined_delta, candidate_ids),
                "G_diffusion_stats": _score_stats_for_block_ids(drm_meta["G_diffusion"], candidate_ids),
                "R_active_stats": _score_stats_for_block_ids(drm_meta["R_active"], candidate_ids),
                "rescue_candidate_count": int(len(candidate_ids)),
                "rescue_candidate_strategy": str(cfg.get("rescue_candidate_strategy", "")),
                "rescue_selector": str(cfg.get("rescue_selector", "drm_mmr")),
                "rescue_gate_mode": gate_mode,
                "ac_gate_type": str(cfg.get("ac_gate_type", ac_gate_type if ac_gate_type != "no_gate" else "")),
                "need_rescue": int(m_pages > 0),
                "rescue_need_score": round(float(need_score), 4),
                "gate_threshold": round(float(cfg.get("_ac_gate_threshold", gate_threshold)), 4),
                "top_rescue_margin": round(float(top_margin), 4),
                "rescue_active_cov_frac": round(float(active_cov_frac), 4),
                "rescue_m_pages": int(m_pages),
                "rescue_m_default": int(m_default),
                "num_rescue_selected": int(len(rescued_combined)),
                "block_budget_mode": str(cfg.get("block_budget_mode", "none")),
                "verifier_mode": str(cfg.get("verifier_mode", "none")),
                **sel_extra,
                **swap_meta,
            }
            rescue_pages = _expand_virtual_to_fixed_pages(combined, rescued_combined)
            dropped_active = {int(p) for p in swap_meta.get("swapped_out_pages", [])}
            final_fixed = (active_pages - dropped_active).union(rescue_pages)
            score_meta["active_fixed_page_ids"] = sorted(int(p) for p in active_pages)
            score_meta["rescued_fixed_page_ids"] = sorted(int(p) for p in rescue_pages)
            if not final_fixed:
                return None
            _write_generic_pruning_audit(
                cfg=cfg,
                original_block_ids=original_block_ids,
                final_ids=final_fixed,
                block_scores=qk_block_scores,
                num_blocks_to_keep=min(num_virtual_to_keep, len(original_block_ids)),
                final_budget_type="K+M",
                layout_meta={**layout_meta, **{"rescue_layout_meta": rescue_meta}},
                virtual_blocks=combined,
                active_virtual_ids=selected,
                final_virtual_ids=set(selected).union(int(x) for x in rescued_combined),
                rescued_virtual_ids=[int(x) for x in rescued_combined],
                score_meta=score_meta,
            )
            return final_fixed

        if strategy.startswith("hybrid"):
            rescue_blocks, rescue_meta = _candidate_blocks_for_hybrid(cfg, original_block_ids, block_scores)
            candidate_scores = _aggregate_virtual_scores(rescue_blocks, original_block_ids, block_scores, device)
            candidate_delta = _aggregate_virtual_scores(rescue_blocks, original_block_ids, delta_o_values, device) if delta_o_values is not None else candidate_scores
            candidate_reps = _aggregate_virtual_reps(rescue_blocks, page_reps, device)
            active_pages = _expand_virtual_to_fixed_pages(virtual_blocks, selected)
            inactive = []
            threshold = float(cfg.get("hybrid_overlap_threshold", 0.5))
            for block in rescue_blocks:
                pages = {int(x) for x in block.get("fixed_page_ids", [])}
                overlap = len(pages.intersection(active_pages)) / max(1, len(pages))
                if int(cfg.get("hybrid_exclude_active_overlap", 1) or 1) and overlap > threshold:
                    continue
                inactive.append(int(block["virtual_id"]))
            if score_name in {"drm", "boundary_drm"}:
                base = normalize_scores(candidate_delta)
                if candidate_reps is not None and candidate_reps.numel() > 0:
                    diff = normalize_scores(torch.norm(candidate_reps.float(), dim=-1))
                    if selected and virtual_reps is not None and virtual_reps.numel() > 0:
                        a_ids = torch.tensor([x for x in selected if 0 <= x < virtual_reps.shape[0]], dtype=torch.long, device=device)
                        if a_ids.numel() > 0:
                            cand_norm = torch.nn.functional.normalize(candidate_reps.float(), dim=-1, eps=1e-6)
                            active_norm = torch.nn.functional.normalize(virtual_reps[a_ids].float(), dim=-1, eps=1e-6)
                            red = torch.matmul(cand_norm, active_norm.T).max(dim=-1).values
                        else:
                            red = torch.zeros_like(base)
                    else:
                        red = torch.zeros_like(base)
                else:
                    diff = torch.zeros_like(base)
                    red = torch.zeros_like(base)
                s0 = base + float(cfg.get("alpha_diff", 0.5)) * normalize_scores(diff) - float(cfg.get("lambda_red", 0.25)) * normalize_scores(red)
                if score_name == "boundary_drm":
                    s0 = s0 + float(cfg.get("gamma_boundary", 0.25)) * normalize_scores(_boundary_confidence_tensor(rescue_blocks, device))
                score_meta = {
                    "D_base_stats": _score_stats_for_block_ids(candidate_delta, inactive),
                    "G_diffusion_stats": _score_stats_for_block_ids(diff, inactive),
                    "R_active_stats": _score_stats_for_block_ids(red, inactive),
                    "hybrid_rescue_candidate_count": int(len(inactive)),
                }
            else:
                # Random-Add must sample residual blocks with cfg["seed"]. Using
                # candidate_scores (original SlimInfer relevance) here made seed0
                # and seed1 produce identical selections.
                if score_name == "random":
                    _rng = torch.Generator(device="cpu")
                    _rng.manual_seed(int(cfg.get("seed", 0) or 0))
                    s0 = torch.rand(
                        int(candidate_scores.numel()),
                        generator=_rng,
                        dtype=torch.float32,
                    ).to(device=candidate_scores.device)
                else:
                    s0 = candidate_delta if score_name == "delta_o" else candidate_scores
            if rescue_budget_mode == "token_budget":
                rescued = _greedy_mmr_token_budget(
                    inactive_ids=inactive,
                    s0_scores=s0,
                    virtual_blocks=rescue_blocks,
                    block_reps=candidate_reps,
                    active_ids=[],
                    initial_selected_ids=set(),
                    target_tokens=max(1, target_final_tokens - _virtual_selected_cost(virtual_blocks, selected, cost_mode)),
                    cost_mode=cost_mode,
                    density_mode=density_mode,
                    eta_rescue=float(cfg.get("eta_rescue", 0.25)),
                    eta_active=float(cfg.get("eta_active", 0.25)),
                    allow_overshoot=allow_budget_overshoot,
                    hard_tolerance=hard_budget_tolerance,
                )
            else:
                target_m = min(len(inactive), max(1, int(round(len(selected) * float(cfg.get("rescue_ratio", 0.05)))))) if inactive else 0
                rescued = _top_virtual_ids(s0, target_m, exclude=set())
            rescue_pages = _expand_virtual_to_fixed_pages(rescue_blocks, rescued)
            final_fixed = active_pages.union(rescue_pages)
            if not final_fixed:
                return None
            score_meta["active_fixed_page_ids"] = sorted(int(p) for p in active_pages)
            score_meta["rescued_fixed_page_ids"] = sorted(int(p) for p in rescue_pages)
            hybrid_virtual_for_audit = virtual_blocks + [
                {**block, "virtual_id": int(block.get("virtual_id", 0)) + len(virtual_blocks), "source_type": f"rescue_{block.get('source_type', '')}"}
                for block in rescue_blocks
            ]
            shifted_rescued = [int(x) + len(virtual_blocks) for x in rescued]
            _write_generic_pruning_audit(
                cfg=cfg,
                original_block_ids=original_block_ids,
                final_ids=final_fixed,
                block_scores=qk_block_scores,
                num_blocks_to_keep=min(num_virtual_to_keep, len(original_block_ids)),
                final_budget_type="K+M",
                layout_meta={**layout_meta, **{"rescue_layout_meta": rescue_meta}},
                virtual_blocks=hybrid_virtual_for_audit,
                active_virtual_ids=selected,
                final_virtual_ids=set(selected).union(shifted_rescued),
                rescued_virtual_ids=shifted_rescued,
                score_meta=score_meta,
            )
            return final_fixed

        inactive = [vid for vid in range(n_virtual) if vid not in selected]
        target_m = int(round(len(selected) * float(cfg.get("rescue_ratio", 0.05))))
        if float(cfg.get("rescue_ratio", 0.05)) > 0.0 and selected and inactive:
            target_m = max(1, target_m)
        target_m = min(target_m, len(inactive))
        if inactive:
            if (
                str(cfg.get("rescue_granularity", "page") or "page") == "microblock"
                and score_name in _STRUCTURED_RESCUE_SCORES
            ):
                rescued, structured_meta = _structured_microblock_rescue_in_fixed_virtual_path(
                    cfg=cfg,
                    selected=selected,
                    inactive=inactive,
                    original_block_ids=original_block_ids,
                    virtual_delta=virtual_delta,
                    virtual_reps=virtual_reps,
                    qk_block_scores=qk_block_scores,
                    target_m=target_m,
                    device=device,
                )
                score_meta.update(structured_meta)
            elif score_name in {"drm", "boundary_drm"}:
                s0, drm_meta = drm_s0_score(
                    virtual_delta,
                    virtual_reps,
                    sorted(selected),
                    inactive,
                    alpha_diff=float(cfg.get("alpha_diff", 0.5)),
                    lambda_red=float(cfg.get("lambda_red", 0.25)),
                    diagnostics={"warnings": []},
                )
                if score_name == "boundary_drm":
                    s0 = s0 + float(cfg.get("gamma_boundary", 0.25)) * normalize_scores(_boundary_confidence_tensor(virtual_blocks, device))
                score_meta = {
                    "D_base_stats": _score_stats_for_block_ids(virtual_delta, inactive),
                    "G_diffusion_stats": _score_stats_for_block_ids(drm_meta["G_diffusion"], inactive),
                    "R_active_stats": _score_stats_for_block_ids(drm_meta["R_active"], inactive),
                }
                if rescue_budget_mode == "token_budget":
                    rescued = _greedy_mmr_token_budget(
                        inactive_ids=inactive,
                        s0_scores=s0,
                        virtual_blocks=virtual_blocks,
                        block_reps=virtual_reps,
                        active_ids=sorted(selected),
                        initial_selected_ids=set(selected),
                        target_tokens=target_final_tokens,
                        cost_mode=cost_mode,
                        density_mode=density_mode,
                        eta_rescue=float(cfg.get("eta_rescue", 0.25)),
                        eta_active=float(cfg.get("eta_active", 0.25)),
                        allow_overshoot=allow_budget_overshoot,
                        hard_tolerance=hard_budget_tolerance,
                    )
                elif target_m > 0:
                    rescued, mmr_meta = mmr_select_rescue_ids(
                        inactive_ids=inactive,
                        s0_scores=s0,
                        block_reps=virtual_reps,
                        active_ids=sorted(selected),
                        m=target_m,
                        eta_rescue=float(cfg.get("eta_rescue", 0.25)),
                        eta_active=float(cfg.get("eta_active", 0.25)),
                        seed=int(cfg.get("seed", 0) or 0),
                        diagnostics={"warnings": []},
                    )
                    score_meta["MMR_diversity_stats"] = mmr_meta.get("MMR_diversity_stats", {})
            elif target_m > 0:
                # Random-Add residual selection must depend on cfg["seed"]. The fixed
                # virtual-block path previously reused original SlimInfer scores for
                # every non-delta_o score name, so seed0/seed1 were identical.
                if score_name == "random":
                    _rng = torch.Generator(device="cpu")
                    _rng.manual_seed(int(cfg.get("seed", 0) or 0))
                    rescue_base = torch.rand(
                        int(n_virtual),
                        generator=_rng,
                        dtype=torch.float32,
                    ).to(device=device)
                else:
                    rescue_base = virtual_delta if score_name == "delta_o" else virtual_scores
                if rescue_budget_mode == "token_budget":
                    rescued = list(_select_virtual_token_budget(
                        rescue_base,
                        virtual_blocks,
                        target_final_tokens,
                        cost_mode=cost_mode,
                        density_mode=density_mode,
                        exclude=selected,
                        must_keep=selected,
                        allow_overshoot=allow_budget_overshoot,
                        soft_tolerance=soft_budget_tolerance,
                        hard_tolerance=hard_budget_tolerance,
                    ).difference(selected))
                else:
                    rescued = _top_virtual_ids(rescue_base, target_m, exclude=selected)
        final_virtual = set(selected).union(int(x) for x in rescued)
        final_budget_type = "K+M"

    active_fixed = _expand_virtual_to_fixed_pages(virtual_blocks, selected)
    rescued_fixed = _expand_virtual_to_fixed_pages(virtual_blocks, rescued)
    if cfg.get("variant_method") == "qdc_sliminfer" and _qdc_is_gate_layer(cfg):
        qdc_active_fixed = _qdc_maybe_apply_page_selection(cfg, set(active_fixed))
        final_fixed = set(qdc_active_fixed) | set(rescued_fixed)
    elif cfg.get("variant_method") == "mcu_sliminfer" and _mcu_is_gate_layer(cfg):
        mcu_active_fixed = _mcu_maybe_apply_page_selection(cfg, set(active_fixed))
        final_fixed = set(mcu_active_fixed) | set(rescued_fixed)
    else:
        final_fixed = _expand_virtual_to_fixed_pages(virtual_blocks, final_virtual)
    if not final_fixed:
        return None
    score_meta["active_fixed_page_ids"] = sorted(int(p) for p in active_fixed)
    score_meta["rescued_fixed_page_ids"] = sorted(int(p) for p in rescued_fixed)
    _write_generic_pruning_audit(
        cfg=cfg,
        original_block_ids=original_block_ids,
        final_ids=final_fixed,
        block_scores=qk_block_scores,
        num_blocks_to_keep=min(num_virtual_to_keep, len(original_block_ids)),
        final_budget_type=final_budget_type,
        layout_meta=layout_meta,
        virtual_blocks=virtual_blocks,
        active_virtual_ids=selected,
        final_virtual_ids=final_virtual,
        rescued_virtual_ids=rescued,
        score_meta=score_meta,
    )
    return final_fixed

def _maybe_record_sp_trajectory_layer(
    *,
    cfg: Dict[str, Any],
    q_for_pruning: torch.Tensor,
    represent_keys: torch.Tensor,
    original_block_ids: List[int],
    block_scores: torch.Tensor,
    qk_block_scores: torch.Tensor,
    block_to_kernel_map: Dict[int, Tuple[int, int]],
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    final_kept_ids: Set[int],
) -> None:
    if not cfg.get("sp_trajectory_dump_path"):
        return
    try:
        from bridgekv.sp_trajectory import record_trajectory_prune_layer

        record_trajectory_prune_layer(
            cfg=cfg,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            qk_block_scores=qk_block_scores,
            q_for_pruning=q_for_pruning,
            represent_keys=represent_keys,
            block_to_kernel_map=block_to_kernel_map,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            final_kept_ids=final_kept_ids,
        )
    except Exception as exc:
        print(f"[sp_trajectory] record failed layer={cfg.get('layer_idx')}: {exc}")


def _calculate_pruning_with_rep_keys(
    q_for_pruning: torch.Tensor,
    represent_keys: torch.Tensor,
    original_block_ids: List[int],
    cfg: Dict[str, Any],
    num_key_value_groups: int,
    block_to_kernel_map: Dict[int, Tuple[int, int]],
    represent_values: Optional[torch.Tensor] = None,
) -> Set[int]:
    """
    use represent_keys to calculate the pruning decision.
    Args:
        q_for_pruning (torch.Tensor): the query tensor for pruning, shape [bsz, num_heads, 1, head_dim].
        represent_keys (torch.Tensor): the cached representational keys, shape [bsz, num_kv_heads, num_blocks, head_dim].
        original_block_ids (List[int]): the original block IDs corresponding to the blocks in represent_keys (sorted).
        cfg (Dict[str, Any]): the pruning configuration for the current layer.
        num_key_value_groups (int): number of groups for GQA/MQA.
        
    Returns:
        Set[int]: the set of block IDs that should be kept.
    """    
    score_start = time.perf_counter()
    if represent_keys.shape[2] == 0:
        if cfg.get("variant_method") in ("dc_sliminfer", "adaptive_dc_sliminfer_v3"):
            _set_dc_activation_state(
                cfg,
                _dc_activation_record(
                    cfg=cfg,
                    original_block_ids=original_block_ids,
                    original_selected_ids=set(original_block_ids),
                    dc_selected_ids=set(original_block_ids),
                    num_blocks_to_keep=len(original_block_ids),
                    entered_dc_selector=False,
                    early_return_reason="represent_keys_empty",
                    keep_all=True,
                ),
            )
        return set(original_block_ids)

    head_dim = q_for_pruning.shape[-1]
    k_kernels_repeated = repeat_kv(represent_keys, num_key_value_groups)
    kernel_scores = torch.einsum('bhid,bhkd->bhik', q_for_pruning, k_kernels_repeated) / math.sqrt(head_dim)
    kernel_scores_flat = kernel_scores.mean(dim=[0, 1, 2])
    block_scores = torch.full((len(original_block_ids),), -torch.inf, device=q_for_pruning.device, dtype=kernel_scores_flat.dtype)
    qk_block_scores = block_scores
    block_slices: List[Tuple[int, int]] = []

    # Use the most relevant semantic kernel within each block to represent the original SlimInfer score.
    current_kernel_offset = 0
    for i, block_id in enumerate(original_block_ids):
        if block_id in block_to_kernel_map:
            _, num_kernels = block_to_kernel_map[block_id]
            block_slices.append((current_kernel_offset, num_kernels))
            scores_for_this_block = kernel_scores_flat[current_kernel_offset : current_kernel_offset + num_kernels]
            max_score = torch.max(scores_for_this_block)
            block_scores[i] = max_score
            current_kernel_offset += num_kernels
        else:
            block_slices.append((current_kernel_offset, 0))
    qk_block_scores = block_scores.clone()
    head_page_mass = _head_page_mass_from_kernel_scores(kernel_scores, block_slices)

    variant_method = cfg.get("variant_method")
    if variant_method == "ud_sliminfer":
        block_scores, ud_meta = compute_unit_density_block_scores(
            block_scores=block_scores,
            block_slices=block_slices,
            kernel_scores_flat=kernel_scores_flat,
            w_max=float(cfg.get("ud_w_max", 0.50)),
            w_density=float(cfg.get("ud_w_density", 0.35)),
            w_spike=float(cfg.get("ud_w_spike", 0.15)),
        )
        qk_block_scores = block_scores.clone()
        cfg["_ud_last"] = ud_meta

    score_kind = cfg.get("score_kind", cfg.get("scoring", "token_unit_qk"))
    if score_kind == "token_unit":
        score_kind = "token_unit_qk"
    if variant_method is None:
        variant_method = cfg.get("variant_method")

    if variant_method == "multi_window_sliminfer":
        windows = cfg.get("mw_windows", [1, 4, 8])
        mw_scores, mw_meta = compute_multi_window_block_scores(
            q_for_pruning=q_for_pruning,
            represent_keys=represent_keys,
            original_block_ids=original_block_ids,
            block_to_kernel_map=block_to_kernel_map,
            num_key_value_groups=num_key_value_groups,
            windows=windows,
        )
        block_scores = mw_scores
        qk_block_scores = mw_scores.clone()
        cfg["_mw_last"] = mw_meta
    delta_o_values: Optional[torch.Tensor] = None
    delta_h_values: Optional[torch.Tensor] = None
    delta_s_values: Optional[torch.Tensor] = None
    value_norm_values: Optional[torch.Tensor] = None
    block_modes: Optional[Dict[str, int]] = None
    bridge_result_for_stats: Optional[Dict[str, Any]] = None
    bridge_state_for_stats: Optional[Dict[str, Any]] = None

    functional_score_kinds = {
        "delta_o",
        "delta_oh",
        "delta_ov",
        "delta_soh",
        "delta_qk_o_hybrid",
        "entropy_guarded_qk",
        "low_cost_delta_o",
        "functional_damage",
        "bridgekv",
    }
    rescue_policy = str(cfg.get("rescue_policy", "none") or "none")
    rescue_score = str(cfg.get("rescue_score", "delta_o") or "delta_o")
    need_functional = (
        score_kind in functional_score_kinds
        or variant_method == "kv_separated"
        or cfg.get("kv_mode", "none") != "none"
        or rescue_policy not in {"", "none"}
        or rescue_score in {
            "delta_o", "drm", "mmr", "diffusion_gap", "redundancy", "obd", "abc", "random",
            "query_drm", "stochastic_drm", "query_stochastic_drm", "boundary_drm",
            *_STRUCTURED_RESCUE_SCORES,
        }
    )
    if need_functional:
        functional_scores = _calculate_functional_damage_scores(
            q_for_pruning=q_for_pruning,
            kernel_scores=kernel_scores,
            represent_values=represent_values,
            num_key_value_groups=num_key_value_groups,
            block_slices=block_slices,
        )
        if functional_scores is not None:
            delta_o_values, delta_h_values, delta_s_values = functional_scores
            value_norm_values = _calculate_block_value_norms(
                represent_values=represent_values,
                num_key_value_groups=num_key_value_groups,
                block_slices=block_slices,
                device=q_for_pruning.device,
            )
            if variant_method == "kv_separated" or cfg.get("kv_mode", "none") in {"guardrail", "weighted", "true_separated_if_supported"}:
                dk = delta_h_values + delta_s_values
                dv = delta_o_values
                if cfg.get("kv_mode", "guardrail") == "weighted":
                    block_scores = cfg.get("lambda_k", 0.5) * _normalize_scores(dk) + cfg.get("lambda_v", 0.5) * _normalize_scores(dv)
                    block_modes = {"full_gpu": int(len(original_block_ids)), "cpu_offload": 0, "k_gpu_v_cpu": 0, "k_cpu_v_gpu": 0}
                else:
                    risk_ratio = float(cfg.get("kv_topk_risk_ratio", 0.10))
                    risk_k = _topk_mask(dk, max(1, math.ceil(len(original_block_ids) * risk_ratio)))
                    risk_v = _topk_mask(dv, max(1, math.ceil(len(original_block_ids) * risk_ratio)))
                    high_risk = risk_k | risk_v
                    block_scores = _normalize_scores(dk + dv) + high_risk.to(dk.dtype)
                    block_modes = {
                        "full_gpu": int(high_risk.sum().item()),
                        "cpu_offload": int((~high_risk).sum().item()),
                        "k_gpu_v_cpu": int((risk_k & ~risk_v).sum().item()),
                        "k_cpu_v_gpu": int((risk_v & ~risk_k).sum().item()),
                    }
            else:
                alpha_d = float(cfg.get("fd_alpha_o", 1.0))
                beta_d = float(cfg.get("fd_beta_h", 0.0))
                gamma_d = float(cfg.get("fd_gamma_s", 0.0))
                normalize_mode = cfg.get("fd_normalize", "per_layer")
                do = _maybe_normalize_scores(delta_o_values, normalize_mode)
                dh = _maybe_normalize_scores(delta_h_values, normalize_mode)
                ds = _maybe_normalize_scores(delta_s_values, normalize_mode)
                qk_norm = _maybe_normalize_scores(qk_block_scores, normalize_mode)
                fairness_base_scoring = str(cfg.get("fairness_base_scoring", "") or "")

                if fairness_base_scoring == "token_unit_qk":
                    block_scores = qk_block_scores
                elif fairness_base_scoring == "delta_o":
                    block_scores = alpha_d * do
                elif score_kind == "delta_qk_o_hybrid":
                    qk_w = float(cfg.get("hybrid_qk_weight", 0.5))
                    o_w = float(cfg.get("hybrid_o_weight", 0.5))
                    block_scores = qk_w * qk_norm + o_w * do
                elif score_kind == "entropy_guarded_qk":
                    risk_ratio = float(cfg.get("risk_ratio", cfg.get("kv_topk_risk_ratio", 0.10)))
                    risk_h = _topk_mask(delta_h_values, max(1, math.ceil(len(original_block_ids) * risk_ratio)))
                    block_scores = qk_norm + risk_h.to(qk_norm.dtype) * 2.0
                elif score_kind == "delta_ov":
                    beta_v = float(cfg.get("fd_beta_v", 0.0))
                    vn = _maybe_normalize_scores(value_norm_values, normalize_mode) if value_norm_values is not None else torch.zeros_like(do)
                    block_scores = alpha_d * do + beta_v * vn
                elif score_kind == "delta_o" or score_kind == "low_cost_delta_o":
                    block_scores = alpha_d * do
                elif score_kind == "bridgekv":
                    bridge_state = get_bridgekv_state(cfg.get("bridgekv_state_id"))
                    if bridge_state is None:
                        block_scores = alpha_d * do
                    else:
                        bridge_result = compute_bridge_utilities(
                            state=bridge_state,
                            qk_scores=qk_block_scores,
                            delta_o_values=delta_o_values,
                            block_ids=original_block_ids,
                            args=type("Args", (), {  # lightweight proxy
                                "local_score": cfg.get("local_score", "delta_o"),
                                "bridge_norm": cfg.get("bridge_norm", "per_layer"),
                                "bridge_alpha": float(cfg.get("bridge_alpha", 0.5)),
                                "bridge_beta": float(cfg.get("bridge_beta", 0.4)),
                                "bridge_gamma": float(cfg.get("bridge_gamma", 0.0)),
                                "bridge_eta": float(cfg.get("bridge_eta", 0.1)),
                                "bridge_score": cfg.get("bridge_score", "ppr"),
                                "bridge_ppr_alpha": float(cfg.get("bridge_ppr_alpha", 0.85)),
                                "bridge_pseudo_evidence_ratio": float(cfg.get("bridge_pseudo_evidence_ratio", 0.10)),
                                "bridge_anchor_mode": cfg.get("bridge_anchor_mode", "union_weighted"),
                                "bridge_anchor_ratio": float(cfg.get("bridge_anchor_ratio", cfg.get("bridge_pseudo_evidence_ratio", 0.10))),
                                "bridge_hub_penalty": bool(cfg.get("bridge_hub_penalty", False)),
                                "bridge_hub_lambda": float(cfg.get("bridge_hub_lambda", 0.0) or 0.0),
                                "bridge_idf_node": bool(cfg.get("bridge_idf_node", False)),
                                "num_evidence_anchors": int(cfg.get("num_evidence_anchors", 16)),
                                "seed": int(cfg.get("seed", 0) or 0),
                            }),
                        )
                        block_scores = bridge_result["final_utility"]
                        bridge_result_for_stats = bridge_result
                        bridge_state_for_stats = bridge_state
                        graph_summary = bridge_state.setdefault("graph_summary", {})
                        graph_summary["bridge_overhead_ms"] = float(graph_summary.get("graph_build_time_ms", 0.0)) + float(bridge_result.get("bridge_ppr_time_ms", 0.0))
                        bridge_state["bridge_score_summary"] = {
                            **_tensor_summary("bridge_score", bridge_result["bridge_scores"]),
                            **_tensor_summary("local_score", bridge_result["local_scores"]),
                            **_tensor_summary("structural_score", bridge_result["structural_scores"]),
                            **_tensor_summary("final_utility", bridge_result["final_utility"]),
                            "bridge_ppr_time_ms": float(bridge_result.get("bridge_ppr_time_ms", 0.0)),
                            "ppr_time_ms": float(bridge_result.get("bridge_ppr_time_ms", 0.0)),
                            "bridge_overhead_ms": float(graph_summary.get("bridge_overhead_ms", 0.0)),
                            "bridge_score_type": bridge_result.get("bridge_score_type", cfg.get("bridge_score", "ppr")),
                            "path_coverage_time_ms": float(bridge_result.get("bridge_ppr_time_ms", 0.0)) if bridge_result.get("bridge_score_type", cfg.get("bridge_score", "")) == "path_coverage" else "",
                            "path_coverage_blocks_count": int((bridge_result["bridge_scores"].detach().float() > 0).sum().item()) if bridge_result.get("bridge_score_type", cfg.get("bridge_score", "")) == "path_coverage" else "",
                            "top_bridge_blocks": _top_block_ids_for_summary(bridge_result["bridge_scores"], original_block_ids),
                            "top_local_blocks": _top_block_ids_for_summary(bridge_result["local_scores"], original_block_ids),
                            "top_structural_blocks": _top_block_ids_for_summary(bridge_result["structural_scores"], original_block_ids),
                        }
                else:
                    block_scores = alpha_d * do + beta_d * dh + gamma_d * ds

                clip_value = float(cfg.get("fd_score_clip", 0.0) or 0.0)
                if clip_value > 0:
                    block_scores = torch.clamp(block_scores, min=-clip_value, max=clip_value)

    roleflow_policy = str(cfg.get("roleflow_policy", "") or "")
    variant_method = cfg.get("variant_method")
    silm_block_scores_snapshot: Optional[torch.Tensor] = None
    if not roleflow_policy and str(variant_method or "").startswith("roleflow_"):
        roleflow_policy = str(variant_method)
        cfg["roleflow_policy"] = roleflow_policy
    if roleflow_policy and roleflow_policy not in {"", "none", "sliminfer_original"}:
        silm_block_scores_snapshot = block_scores.detach().clone()
        block_scores, roleflow_adjust_meta = apply_roleflow_score_adjustment(
            block_scores=block_scores,
            original_block_ids=original_block_ids,
            cfg=cfg,
        )
        cfg["_roleflow_adjust_last"] = roleflow_adjust_meta

    layer_idx = int(cfg.get("layer_idx", -1))
    abs_prune_layers = [int(x) for x in cfg.get("abs_prune_layers", [9, 18, 26])]
    if variant_method in {"adaptive_budget_sliminfer", "reliability_gated_sliminfer"} and layer_idx in abs_prune_layers:
        if layer_idx == abs_prune_layers[0]:
            k_probe = max(1, min(len(original_block_ids), max(1, len(original_block_ids) // 4)))
            conf, conf_meta = compute_pruning_confidence(
                block_scores,
                k=k_probe,
                prev_active_local_indices=None,
            )
            cfg["_abs_difficulty"] = 1.0 - conf
            cfg["_abs_confidence"] = conf
            cfg["_abs_conf_meta"] = conf_meta
            if variant_method == "reliability_gated_sliminfer":
                cfg["_rgs_policy"] = reliability_policy_from_confidence(
                    conf,
                    low=float(cfg.get("rgs_confidence_low", 0.35)),
                    high=float(cfg.get("rgs_confidence_high", 0.65)),
                )
                cfg["_rgs_confidence"] = conf
        if variant_method == "adaptive_budget_sliminfer":
            alt_k = adaptive_retain_tokens_k(
                layer_idx,
                difficulty=float(cfg.get("_abs_difficulty", 0.5)),
                easy_schedule=tuple(cfg.get("abs_easy_schedule", (12, 6, 4))),
                hard_schedule=tuple(cfg.get("abs_hard_schedule", (12, 8, 6))),
                prune_layers=abs_prune_layers,
            )
            if alt_k is not None:
                cfg["fixed_keep_tokens"] = alt_k
                cfg["_abs_applied_keep_k"] = alt_k

    num_blocks_k = len(original_block_ids)
    pruning_method = cfg.get("pruning_method", "ratio")
    num_blocks_to_keep = num_blocks_k
    fixed_keep_tokens_for_virtual: Optional[float] = None
    top_k_indices = torch.arange(num_blocks_k, device=q_for_pruning.device)

    def _current_must_keep_block_ids() -> Set[int]:
        ids: Set[int] = set()
        if cfg.get("always_keep_sink_recent_token", True) and original_block_ids:
            ids.add(original_block_ids[0])  # sink tokens
            if cfg.get("always_keep_recent_block", True):
                ids.add(original_block_ids[-1])  # recent prompt block
        return ids

    if pruning_method == "fixed_tokens":
        fixed_keep_tokens = cfg.get("fixed_keep_tokens")
        if fixed_keep_tokens is None or fixed_keep_tokens <= 0:
            print(f"Pruning method is 'fixed_tokens' but 'fixed_keep_tokens' is not set or invalid. Keeping all blocks.")
        else:
            # calculate number of blocks to keep
            fixed_keep_tokens_for_virtual = float(fixed_keep_tokens)
            fixed_keep_tokens *= 1024
            num_blocks_to_keep = math.ceil(fixed_keep_tokens / _BLOCK_SIZE)
            num_blocks_to_keep = min(num_blocks_to_keep, num_blocks_k)
            if rescue_policy == "sliminfer_larger":
                rescue_ratio = float(cfg.get("rescue_ratio", 0.05))
                extra_m = int(round(num_blocks_to_keep * rescue_ratio))
                if rescue_ratio > 0.0 and num_blocks_to_keep > 0:
                    extra_m = max(1, extra_m)
                num_blocks_to_keep = min(num_blocks_k, num_blocks_to_keep + extra_m)
            if num_blocks_to_keep >= num_blocks_k:
                all_kept_ids = set(original_block_ids)
                if variant_method in ("dc_sliminfer", "adaptive_dc_sliminfer_v3"):
                    _select_dc_sliminfer_ids(
                        cfg=cfg,
                        original_block_ids=original_block_ids,
                        block_scores=block_scores,
                        top_k_indices=torch.arange(num_blocks_k, device=q_for_pruning.device),
                        must_keep_block_ids=_current_must_keep_block_ids(),
                        num_blocks_to_keep=num_blocks_k,
                    )
                _write_uasr_score_dump(
                    cfg=cfg,
                    original_block_ids=original_block_ids,
                    final_kept_ids=all_kept_ids,
                    block_scores=block_scores,
                    qk_block_scores=qk_block_scores,
                    delta_o_values=delta_o_values,
                    num_blocks_to_keep=num_blocks_to_keep,
                    must_keep_block_ids=_current_must_keep_block_ids(),
                )
                mk = _current_must_keep_block_ids()
                _maybe_record_sp_trajectory_layer(
                    cfg=cfg,
                    q_for_pruning=q_for_pruning,
                    represent_keys=represent_keys,
                    original_block_ids=original_block_ids,
                    block_scores=block_scores,
                    qk_block_scores=qk_block_scores,
                    block_to_kernel_map=block_to_kernel_map,
                    top_k_indices=torch.arange(num_blocks_k, device=q_for_pruning.device),
                    must_keep_block_ids=mk,
                    num_blocks_to_keep=num_blocks_to_keep,
                    final_kept_ids=all_kept_ids,
                )
                _update_score_runtime_stats(cfg, score_start)
                return all_kept_ids
            _, top_k_indices = torch.topk(block_scores, k=int(num_blocks_to_keep))
    elif pruning_method == "threshold":
        pruning_threshold = cfg.get("pruning_threshold")
        min_keep_tokens = cfg.get("min_keep_tokens", 1024)
        min_keep_blocks = math.ceil(min_keep_tokens / _BLOCK_SIZE)
        min_keep_ratio = cfg.get("min_keep_ratio", 0.0)
        if min_keep_ratio > 0.0:
            min_keep_blocks = max(min_keep_blocks, math.ceil(num_blocks_k * min_keep_ratio))

        valid_scores_mask = ~torch.isinf(block_scores)
        if not valid_scores_mask.any():
            raise KeyError("repre key not set")
        contributions = torch.softmax(block_scores[valid_scores_mask], dim=-1)
        sorted_contributions, sorted_indices_local = torch.sort(contributions, descending=True)
        
        cumulative_contributions = torch.cumsum(sorted_contributions, dim=-1)
        k_dynamic = (cumulative_contributions >= pruning_threshold).to(torch.int32).argmax().item() + 1
        num_blocks_to_keep = max(int(min_keep_blocks), k_dynamic)
        num_blocks_to_keep = min(valid_scores_mask.sum().item(), num_blocks_to_keep)
        
        original_indices_tensor = torch.arange(len(original_block_ids), device=q_for_pruning.device)
        valid_original_indices = original_indices_tensor[valid_scores_mask]
        top_k_indices = valid_original_indices[sorted_indices_local[:num_blocks_to_keep]]
    elif pruning_method == "ratio":
        pruning_ratio = cfg.get("pruning_ratio", 0.0)
        num_blocks_to_keep = math.ceil(num_blocks_k * (1.0 - pruning_ratio))
        if rescue_policy == "sliminfer_larger":
            rescue_ratio = float(cfg.get("rescue_ratio", 0.05))
            extra_m = int(round(num_blocks_to_keep * rescue_ratio))
            if rescue_ratio > 0.0 and num_blocks_to_keep > 0:
                extra_m = max(1, extra_m)
                num_blocks_to_keep = min(num_blocks_k, num_blocks_to_keep + extra_m)
        if num_blocks_to_keep >= num_blocks_k:
            all_kept_ids = set(original_block_ids)
            if variant_method in ("dc_sliminfer", "adaptive_dc_sliminfer_v3"):
                _select_dc_sliminfer_ids(
                    cfg=cfg,
                    original_block_ids=original_block_ids,
                    block_scores=block_scores,
                    top_k_indices=torch.arange(num_blocks_k, device=q_for_pruning.device),
                    must_keep_block_ids=_current_must_keep_block_ids(),
                    num_blocks_to_keep=num_blocks_k,
                )
            _write_uasr_score_dump(
                cfg=cfg,
                original_block_ids=original_block_ids,
                final_kept_ids=all_kept_ids,
                block_scores=block_scores,
                qk_block_scores=qk_block_scores,
                delta_o_values=delta_o_values,
                num_blocks_to_keep=num_blocks_to_keep,
                must_keep_block_ids=_current_must_keep_block_ids(),
            )
            _update_score_runtime_stats(cfg, score_start)
            return all_kept_ids
        _, top_k_indices = torch.topk(block_scores, k=int(num_blocks_to_keep))
    else:
        top_k_indices = torch.arange(num_blocks_k, device=q_for_pruning.device)

    must_keep_block_ids = _current_must_keep_block_ids()

    roleflow_policy_early = str(cfg.get("roleflow_policy", "") or "")
    variant_method_early = str(cfg.get("variant_method", "") or "")
    if not roleflow_policy_early and variant_method_early.startswith("roleflow_"):
        roleflow_policy_early = variant_method_early
    skip_virtual_for_roleflow = roleflow_policy_early not in {"", "none", "sliminfer_original"}
    # When selection-audit is enabled, force fixed-page selection for all methods
    # (incl. sliminfer_original) so active-set dumps are comparable to CC/ACC.
    skip_virtual_for_audit = bool(cfg.get("cc_selection_audit_path"))
    explicit_fixed_page = str(cfg.get("selection_path", "original_runtime")) == "canonical_fixed_page"
    skip_virtual_for_custom_policy = skip_virtual_for_roleflow or skip_virtual_for_audit or explicit_fixed_page or variant_method_early in {
        "sp_sliminfer",
        "sp_sliminfer_v2",
        "rank_stability_sliminfer",
        "cc_sliminfer",
        "cc_gated_v0",
        "cc_spike_hybrid_v0",
        "cc_coverage_v0",
        "cc_diversify_v0",
        "cc_router_v0",
        "acc_sliminfer",
        "acc_sliminfer_v1",
        "drm_swap_sliminfer",
        "drm_acc_swap_sliminfer",
        "drm_acc_swap_sliminfer_v1",
        "drm_acc_add_sliminfer",
        "ud_sliminfer",
        "divcov_sliminfer",
    } or variant_method_early in ACC_DRM_METHODS

    virtual_kept_ids = None
    if not skip_virtual_for_custom_policy:
        virtual_kept_ids = _calculate_pruning_with_virtual_blocks(
            q_for_pruning=q_for_pruning,
            original_block_ids=original_block_ids,
            cfg=cfg,
            block_scores=block_scores,
            qk_block_scores=qk_block_scores,
            delta_o_values=delta_o_values,
            represent_values=represent_values,
            num_key_value_groups=num_key_value_groups,
            block_slices=block_slices,
            must_keep_block_ids=must_keep_block_ids,
            fixed_keep_tokens=fixed_keep_tokens_for_virtual,
            head_page_mass=head_page_mass,
        )
    if virtual_kept_ids is not None:
        if variant_method in ("dc_sliminfer", "adaptive_dc_sliminfer_v3"):
            existing_dc_record = cfg.get("_dc_activation_last")
            if not (
                isinstance(existing_dc_record, dict)
                and bool(existing_dc_record.get("entered_dc_selector_before_virtual_return"))
            ):
                original_selected_for_activation = _dc_original_topk_ids(
                    original_block_ids=original_block_ids,
                    block_scores=block_scores,
                    top_k_indices=top_k_indices,
                    must_keep_block_ids=must_keep_block_ids,
                    num_blocks_to_keep=num_blocks_to_keep,
                )
                _set_dc_activation_state(
                    cfg,
                    _dc_activation_record(
                        cfg=cfg,
                        original_block_ids=original_block_ids,
                        original_selected_ids=original_selected_for_activation,
                        dc_selected_ids=virtual_kept_ids,
                        num_blocks_to_keep=num_blocks_to_keep,
                        entered_dc_selector=False,
                        entered_virtual_block_path=True,
                        virtual_return_reason="virtual_block_path_returned_before_dc_selector",
                    ),
                )
        _write_uasr_score_dump(
            cfg=cfg,
            original_block_ids=original_block_ids,
            final_kept_ids=virtual_kept_ids,
            block_scores=block_scores,
            qk_block_scores=qk_block_scores,
            delta_o_values=delta_o_values,
            num_blocks_to_keep=num_blocks_to_keep,
            must_keep_block_ids=must_keep_block_ids,
        )
        _maybe_record_sp_trajectory_layer(
            cfg=cfg,
            q_for_pruning=q_for_pruning,
            represent_keys=represent_keys,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            qk_block_scores=qk_block_scores,
            block_to_kernel_map=block_to_kernel_map,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            final_kept_ids=virtual_kept_ids,
        )
        if _selection_trace_enabled(cfg):
            _write_unified_selection_audit(
                cfg,
                original_block_ids=original_block_ids,
                block_scores=block_scores,
                top_k_indices=top_k_indices,
                must_keep_block_ids=must_keep_block_ids,
                num_blocks_to_keep=num_blocks_to_keep,
                final_kept_ids=virtual_kept_ids,
            )
        elif _selection_digest_enabled(cfg):
            _write_lightweight_selection_digest(
                cfg,
                original_block_ids=original_block_ids,
                num_blocks_to_keep=num_blocks_to_keep,
                final_kept_ids=virtual_kept_ids,
            )
        _update_score_runtime_stats(cfg, score_start)
        return virtual_kept_ids
    
    if len(must_keep_block_ids) >= num_blocks_to_keep:
        if variant_method in ("dc_sliminfer", "adaptive_dc_sliminfer_v3"):
            _set_dc_activation_state(
                cfg,
                _dc_activation_record(
                    cfg=cfg,
                    original_block_ids=original_block_ids,
                    original_selected_ids=must_keep_block_ids,
                    dc_selected_ids=must_keep_block_ids,
                    num_blocks_to_keep=num_blocks_to_keep,
                    entered_dc_selector=False,
                    must_keep_return_reason="must_keep_count_reaches_budget_before_dc_selector",
                ),
            )
        _write_uasr_score_dump(
            cfg=cfg,
            original_block_ids=original_block_ids,
            final_kept_ids=must_keep_block_ids,
            block_scores=block_scores,
            qk_block_scores=qk_block_scores,
            delta_o_values=delta_o_values,
            num_blocks_to_keep=num_blocks_to_keep,
            must_keep_block_ids=must_keep_block_ids,
        )
        _maybe_record_sp_trajectory_layer(
            cfg=cfg,
            q_for_pruning=q_for_pruning,
            represent_keys=represent_keys,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            qk_block_scores=qk_block_scores,
            block_to_kernel_map=block_to_kernel_map,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            final_kept_ids=must_keep_block_ids,
        )
        if _selection_trace_enabled(cfg):
            _write_unified_selection_audit(
                cfg,
                original_block_ids=original_block_ids,
                block_scores=block_scores,
                top_k_indices=top_k_indices,
                must_keep_block_ids=must_keep_block_ids,
                num_blocks_to_keep=num_blocks_to_keep,
                final_kept_ids=must_keep_block_ids,
            )
        elif _selection_digest_enabled(cfg):
            _write_lightweight_selection_digest(
                cfg,
                original_block_ids=original_block_ids,
                num_blocks_to_keep=num_blocks_to_keep,
                final_kept_ids=must_keep_block_ids,
            )
        _update_score_runtime_stats(cfg, score_start)
        return must_keep_block_ids

    policy_details: Dict[str, Any] = {}
    if score_kind == "bridgekv" and bridge_result_for_stats is not None:
        final_kept_ids, policy_details = _select_bridgekv_policy_ids(
            cfg=cfg,
            original_block_ids=original_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            must_keep_block_ids=must_keep_block_ids,
            bridge_result=bridge_result_for_stats,
            qk_scores=qk_block_scores,
            delta_o_values=delta_o_values,
        )
    elif variant_method == "dc_sliminfer":
        final_kept_ids, policy_details = _select_dc_sliminfer_ids(
            cfg=cfg,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
        )
    elif variant_method == "adaptive_dc_sliminfer_v3":
        top_k_block_ids = {original_block_ids[i] for i in top_k_indices.tolist()}
        original_kept = top_k_block_ids.union(must_keep_block_ids)
        if len(original_kept) > num_blocks_to_keep:
            candidates_for_removal = list(top_k_block_ids - must_keep_block_ids)
            id_to_score_map = {bid: sc for bid, sc in zip(original_block_ids, block_scores.tolist())}
            candidates_for_removal.sort(key=lambda bid: id_to_score_map.get(bid, float("-inf")))
            for i in range(min(len(candidates_for_removal), len(original_kept) - num_blocks_to_keep)):
                original_kept.discard(candidates_for_removal[i])
        _adv3_gate_cfg_nv = cfg.get("adaptive_dc_v3_gate") or {}
        gate_apply_layer = int(_adv3_gate_cfg_nv.get("gate_apply_layer", cfg.get("gate_apply_layer", 0)) or 0)
        current_layer = int(cfg.get("layer_idx", 0) or 0)
        is_gate_layer = (gate_apply_layer <= 0) or (current_layer >= gate_apply_layer)
        cached_gate = cfg.get("_adaptive_dc_v3_gate_cached")
        has_cached = isinstance(cached_gate, dict) and "triggered" in cached_gate
        if has_cached:
            gate_result = cached_gate
            cfg["_adaptive_dc_v3_gate_last"] = gate_result
            if gate_result.get("triggered"):
                dc_kept, dc_policy = _adaptive_dc_v3_run_block_dc_selector(
                    cfg=cfg,
                    original_block_ids=original_block_ids,
                    block_scores=block_scores,
                    top_k_indices=top_k_indices,
                    must_keep_block_ids=must_keep_block_ids,
                    num_blocks_to_keep=num_blocks_to_keep,
                )
                _adaptive_dc_v3_flush_deferred_selection_audit(cfg)
                final_kept_ids = dc_kept
                policy_details.update(dc_policy)
            else:
                final_kept_ids = original_kept
        elif is_gate_layer:
            dc_kept, dc_policy = _adaptive_dc_v3_run_block_dc_selector(
                cfg=cfg,
                original_block_ids=original_block_ids,
                block_scores=block_scores,
                top_k_indices=top_k_indices,
                must_keep_block_ids=must_keep_block_ids,
                num_blocks_to_keep=num_blocks_to_keep,
            )
            gate_result = _adaptive_dc_v3_compute_gate(
                cfg=cfg,
                original_selected_block_ids=original_kept,
                dc_selected_block_ids=dc_kept,
                num_blocks=len(original_block_ids),
            )
            cfg["_adaptive_dc_v3_gate_cached"] = gate_result
            cfg["_adaptive_dc_v3_gate_last"] = gate_result
            _adaptive_dc_v3_flush_deferred_selection_audit(cfg)
            if gate_result.get("bug_stop"):
                final_kept_ids = original_kept
            elif gate_result.get("triggered"):
                final_kept_ids = dc_kept
                policy_details.update(dc_policy)
            else:
                final_kept_ids = original_kept
        else:
            gate_result = {"triggered": False, "reason": f"pre_gate_layer:{current_layer}<{gate_apply_layer}",
                           "evidence_dispersion_score": None, "evidence_dispersion_threshold": None,
                           "entity_coverage_gap": None, "pruned_query_overlap_max": None,
                           "selected_query_overlap_mean": None, "selected_jaccard_vs_original": None,
                           "bug_stop": False}
            final_kept_ids = original_kept
        policy_details["adaptive_dc_v3_gate"] = gate_result
        cfg["_adaptive_dc_v3_gate_last"] = gate_result
    elif variant_method == "qdc_sliminfer":
        top_k_block_ids = {original_block_ids[i] for i in top_k_indices.tolist()}
        original_kept = top_k_block_ids.union(must_keep_block_ids)
        if len(original_kept) > num_blocks_to_keep:
            candidates_for_removal = list(top_k_block_ids - must_keep_block_ids)
            id_to_score_map = {block_id: score for block_id, score in zip(original_block_ids, block_scores.tolist())}
            candidates_for_removal.sort(key=lambda block_id: id_to_score_map.get(block_id, -torch.inf))
            num_to_remove = len(original_kept) - num_blocks_to_keep
            for i in range(num_to_remove):
                original_kept.discard(candidates_for_removal[i])
        if _qdc_is_gate_layer(cfg):
            final_kept_ids = _qdc_maybe_apply_page_selection(cfg, original_kept)
        else:
            final_kept_ids = original_kept
        policy_details["qdc_sliminfer"] = cfg.get("_qdc_last", {})
    elif variant_method == "mcu_sliminfer":
        top_k_block_ids = {original_block_ids[i] for i in top_k_indices.tolist()}
        original_kept = top_k_block_ids.union(must_keep_block_ids)
        if len(original_kept) > num_blocks_to_keep:
            candidates_for_removal = list(top_k_block_ids - must_keep_block_ids)
            id_to_score_map = {block_id: score for block_id, score in zip(original_block_ids, block_scores.tolist())}
            candidates_for_removal.sort(key=lambda block_id: id_to_score_map.get(block_id, -torch.inf))
            num_to_remove = len(original_kept) - num_blocks_to_keep
            for i in range(num_to_remove):
                original_kept.discard(candidates_for_removal[i])
        if _mcu_is_gate_layer(cfg):
            final_kept_ids = _mcu_maybe_apply_page_selection(cfg, original_kept)
        else:
            final_kept_ids = original_kept
        policy_details["mcu_sliminfer"] = cfg.get("_mcu_last", {})
    elif variant_method == "confidence_gated_sliminfer":
        prev_raw = cfg.get("_cgs_prev_active_block_ids")
        prev_active: Optional[Set[int]] = None
        if isinstance(prev_raw, (list, set, tuple)):
            prev_active = {int(x) for x in prev_raw}
        final_kept_ids, cgs_details = confidence_gated_active_ids(
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            prev_active_block_ids=prev_active,
            confidence_threshold=float(cfg.get("cgs_confidence_threshold", 0.35)),
        )
        cfg["_cgs_confidence"] = float(cgs_details.get("confidence", 1.0))
        cfg["_cgs_last"] = cgs_details
        cfg["_cgs_prev_active_block_ids"] = sorted(int(x) for x in final_kept_ids)
        policy_details["confidence_gated_sliminfer"] = cgs_details
    elif variant_method == "margin_stable_sliminfer":
        prev_raw = cfg.get("_mss_prev_active_block_ids")
        prev_active = {int(x) for x in prev_raw} if isinstance(prev_raw, (list, set, tuple)) else None
        final_kept_ids, mss_details = margin_stable_active_ids(
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            prev_active_block_ids=prev_active,
            swap_margin_ratio=float(cfg.get("mss_swap_margin_ratio", 0.08)),
        )
        cfg["_mss_prev_active_block_ids"] = sorted(int(x) for x in final_kept_ids)
        policy_details["margin_stable_sliminfer"] = mss_details
    elif variant_method == "rank_stability_sliminfer":
        abs_prune_layers = [int(x) for x in cfg.get("abs_prune_layers", [9, 18, 26])]
        final_kept_ids, rss_details = rank_stability_sliminfer_active_ids(
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            cfg=cfg,
            layer_idx=layer_idx,
            prune_layers=abs_prune_layers,
            weight=float(cfg.get("rss_weight", 0.12)),
        )
        cfg["_rss_last"] = rss_details
        policy_details["rank_stability_sliminfer"] = rss_details
    elif variant_method == "sp_sliminfer":
        abs_prune_layers = [int(x) for x in cfg.get("abs_prune_layers", [9, 18, 26])]
        sp_weights = cfg.get("sp_weights")
        final_kept_ids, sp_details = sp_sliminfer_active_ids(
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            cfg=cfg,
            layer_idx=layer_idx,
            prune_layers=abs_prune_layers,
            weights=sp_weights if isinstance(sp_weights, dict) else None,
        )
        cfg["_sp_last"] = sp_details
        policy_details["sp_sliminfer"] = sp_details
    elif variant_method == "sp_sliminfer_v2":
        try:
            capture_sp_v2_runtime_layer_state(
                cfg=cfg,
                original_block_ids=original_block_ids,
                qk_block_scores=qk_block_scores,
                q_for_pruning=q_for_pruning,
                represent_keys=represent_keys,
                block_to_kernel_map=block_to_kernel_map,
            )
            final_kept_ids, sp_v2_details = sp_v2_active_ids_runtime(
                original_block_ids=original_block_ids,
                block_scores=block_scores,
                top_k_indices=top_k_indices,
                must_keep_block_ids=must_keep_block_ids,
                num_blocks_to_keep=num_blocks_to_keep,
                cfg=cfg,
                layer_idx=int(cfg.get("layer_idx", 0) or 0),
            )
            cfg["_sp_v2_last"] = sp_v2_details
            policy_details["sp_sliminfer_v2"] = sp_v2_details
        except Exception as exc:
            print(f"[sp_sliminfer_v2] fallback to original top-k layer={cfg.get('layer_idx')}: {exc}")
            top_k_block_ids = {original_block_ids[i] for i in top_k_indices.tolist()}
            final_kept_ids = top_k_block_ids.union(must_keep_block_ids)
            if len(final_kept_ids) > num_blocks_to_keep:
                candidates_for_removal = list(top_k_block_ids - must_keep_block_ids)
                id_to_score_map = {block_id: score for block_id, score in zip(original_block_ids, block_scores.tolist())}
                candidates_for_removal.sort(key=lambda block_id: id_to_score_map.get(block_id, -torch.inf))
                num_to_remove = len(final_kept_ids) - num_blocks_to_keep
                for i in range(num_to_remove):
                    final_kept_ids.discard(candidates_for_removal[i])
    elif variant_method == "cc_sliminfer":
        cc_selection_started = time.perf_counter()
        final_kept_ids, cc_details = cluster_continuity_active_ids(
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            alpha=float(cfg.get("cc_alpha", 0.25)),
            min_cluster_len=int(cfg.get("cc_min_cluster_len", 2)),
            max_neighbors=int(cfg.get("cc_max_neighbors", 2)),
            replacement_cap=float(cfg.get("cc_replacement_cap", 0.05)),
        )
        cc_details["selection_time_ms"] = (
            time.perf_counter() - cc_selection_started
        ) * 1000.0
        cfg["_cc_last"] = cc_details
        policy_details["cc_sliminfer"] = cc_details
        policy_details["cc_sliminfer_v1"] = cc_details
        _write_cc_selection_audit(cfg, cc_details)
    elif str(variant_method) in CC_TOPOLOGY_METHODS:
        gate_cfg = cfg.get("_cc_gate_cfg")
        if not isinstance(gate_cfg, dict):
            gate_cfg = load_cc_gate_config(cfg.get("cc_gate_config_path"))
            cfg["_cc_gate_cfg"] = gate_cfg
        query_len_tokens = None
        audit_ctx = cfg.get("audit_context") if isinstance(cfg.get("audit_context"), dict) else {}
        if audit_ctx.get("query_text") is not None:
            query_len_tokens = len(str(audit_ctx.get("query_text") or "").split())
        final_kept_ids, cc_details = dispatch_cc_topology_policy(
            str(variant_method),
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            alpha=float(cfg.get("cc_alpha", 0.25)),
            min_cluster_len=int(cfg.get("cc_min_cluster_len", 2)),
            max_neighbors=int(cfg.get("cc_max_neighbors", 2)),
            replacement_cap=float(cfg.get("cc_replacement_cap", 0.05)),
            gate_cfg=gate_cfg,
            query_len_tokens=query_len_tokens,
        )
        cfg["_cc_last"] = cc_details
        policy_details[str(variant_method)] = cc_details
        policy_details["cc_topology"] = cc_details
        _write_cc_selection_audit(cfg, cc_details)
    elif str(variant_method) in ACC_DRM_METHODS:
        acc_cfg = cfg.get("_acc_cfg")
        if not isinstance(acc_cfg, dict):
            acc_cfg = {}
            for key in (
                "tau_nei", "tau_spike", "tau_spike_nei_max", "tau_cont", "tau_str", "tau_second", "tau_gap", "tau_margin",
                "diversify_top_g", "cluster_share_cap", "n_buckets", "rescue_ratio",
                "alpha_diff", "lambda_red", "eta_rescue", "eta_active",
            ):
                if key in cfg:
                    try:
                        acc_cfg[key] = float(cfg[key])
                    except (TypeError, ValueError):
                        pass
            cfg["_acc_cfg"] = acc_cfg
        block_reps = _block_value_representations_global(
            represent_values,
            num_key_value_groups,
            block_slices,
            original_block_ids,
            block_scores.device,
        )
        final_kept_ids, acc_details = dispatch_acc_drm_policy(
            str(variant_method),
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            block_reps=block_reps,
            alpha=float(cfg.get("cc_alpha", 0.25)),
            min_cluster_len=int(cfg.get("cc_min_cluster_len", 2)),
            max_neighbors=int(cfg.get("cc_max_neighbors", 2)),
            replacement_cap=float(cfg.get("cc_replacement_cap", cfg.get("rescue_ratio", 0.05))),
            acc_cfg=acc_cfg,
        )
        cfg["_acc_last"] = acc_details
        policy_details[str(variant_method)] = acc_details
        policy_details["acc_drm"] = acc_details
        _write_cc_selection_audit(cfg, acc_details)
    elif variant_method == "ud_sliminfer":
        top_k_block_ids = {original_block_ids[i] for i in top_k_indices.tolist()}
        final_kept_ids = top_k_block_ids.union(must_keep_block_ids)
        if len(final_kept_ids) > num_blocks_to_keep:
            candidates_for_removal = list(top_k_block_ids - must_keep_block_ids)
            id_to_score_map = {block_id: score for block_id, score in zip(original_block_ids, block_scores.tolist())}
            candidates_for_removal.sort(key=lambda block_id: id_to_score_map.get(block_id, -torch.inf))
            num_to_remove = len(final_kept_ids) - num_blocks_to_keep
            for i in range(num_to_remove):
                final_kept_ids.discard(candidates_for_removal[i])
        silm_top = {int(original_block_ids[i]) for i in top_k_indices.detach().cpu().tolist()}
        inter = len(final_kept_ids & silm_top)
        union = len(final_kept_ids | silm_top)
        ud_details = dict(cfg.get("_ud_last") or {})
        ud_details.update({
            "mode": "ud_sliminfer",
            "jaccard_vs_original_max_pool_topk": round(inter / max(1, union), 6),
            "num_selected": len(final_kept_ids),
            "sink_kept": int(original_block_ids[0] in final_kept_ids) if original_block_ids else 0,
        })
        cfg["_ud_last"] = ud_details
        policy_details["ud_sliminfer"] = ud_details
    elif variant_method == "divcov_sliminfer":
        final_kept_ids, dc_details = divcov_active_ids(
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            represent_keys=represent_keys,
            block_to_kernel_map=block_to_kernel_map,
            mmr_lambda=float(cfg.get("divcov_mmr_lambda", 0.15)),
        )
        cfg["_divcov_last"] = dc_details
        policy_details["divcov_sliminfer"] = dc_details
    elif variant_method == "reliability_gated_sliminfer":
        prev_raw = cfg.get("_rgs_prev_active_block_ids")
        prev_active = {int(x) for x in prev_raw} if isinstance(prev_raw, (list, set, tuple)) else None
        policy = str(cfg.get("_rgs_policy", "original"))
        final_kept_ids, rgs_details = reliability_gated_active_ids(
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            prev_active_block_ids=prev_active,
            policy=policy,
            confidence_threshold=float(cfg.get("cgs_confidence_threshold", 0.35)),
        )
        cfg["_rgs_prev_active_block_ids"] = sorted(int(x) for x in final_kept_ids)
        if policy == "conservative":
            cfg["_cgs_confidence"] = float(cfg.get("_rgs_confidence", 0.0))
        policy_details["reliability_gated_sliminfer"] = {**rgs_details, "policy": policy}
    elif roleflow_policy and roleflow_policy not in {"", "none", "sliminfer_original"}:
        final_kept_ids, roleflow_policy_details = select_roleflow_kv_ids(
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            cfg=cfg,
            silm_block_scores=silm_block_scores_snapshot,
        )
        policy_details["roleflow_kv"] = roleflow_policy_details
        cfg["_roleflow_selection_last"] = roleflow_policy_details
    else:
        # get topk block IDs
        top_k_block_ids = {original_block_ids[i] for i in top_k_indices.tolist()}
        final_kept_ids = top_k_block_ids.union(must_keep_block_ids)

        if len(final_kept_ids) > num_blocks_to_keep:
            candidates_for_removal = list(top_k_block_ids - must_keep_block_ids)
            id_to_score_map = {block_id: score for block_id, score in zip(original_block_ids, block_scores.tolist())}
            candidates_for_removal.sort(key=lambda block_id: id_to_score_map.get(block_id, -torch.inf))
            
            num_to_remove = len(final_kept_ids) - num_blocks_to_keep
            for i in range(num_to_remove):
                final_kept_ids.remove(candidates_for_removal[i])

    rescue_policy_details = _apply_optional_rescue_policy(
        cfg=cfg,
        original_block_ids=original_block_ids,
        final_kept_ids=final_kept_ids,
        must_keep_block_ids=must_keep_block_ids,
        block_scores=block_scores,
        qk_scores=qk_block_scores,
        delta_o_values=delta_o_values,
        kernel_scores=kernel_scores,
        represent_values=represent_values,
        num_key_value_groups=num_key_value_groups,
        block_slices=block_slices,
    )
    if rescue_policy_details:
        final_kept_ids = rescue_policy_details.pop("final_kept_ids")
        policy_details.update(rescue_policy_details)

    if bridge_state_for_stats is not None and bridge_result_for_stats is not None and num_blocks_to_keep > 0:
        target_k = min(int(num_blocks_to_keep), len(original_block_ids))
        selected_set = {int(block_id) for block_id in final_kept_ids}
        bridge_top = _top_block_id_set(bridge_result_for_stats["bridge_scores"], original_block_ids, target_k).union(must_keep_block_ids)
        local_top = _top_block_id_set(bridge_result_for_stats["local_scores"], original_block_ids, target_k).union(must_keep_block_ids)
        structural_top = _top_block_id_set(bridge_result_for_stats["structural_scores"], original_block_ids, target_k).union(must_keep_block_ids)
        qk_top = _top_block_id_set(qk_block_scores, original_block_ids, target_k).union(must_keep_block_ids)
        delta_o_top = _top_block_id_set(delta_o_values, original_block_ids, target_k).union(must_keep_block_ids) if delta_o_values is not None else set()
        bridge_state_for_stats["selection_summary"] = {
            **policy_details,
            "selected_by_bridge_count": int(len(selected_set.intersection(bridge_top))),
            "selected_by_local_count": int(len(selected_set.intersection(local_top))),
            "selected_by_structural_count": int(len(selected_set.intersection(structural_top))),
            "jaccard_vs_qk_reuse": _jaccard(selected_set, qk_top),
            "jaccard_vs_delta_o_reuse": _jaccard(selected_set, delta_o_top) if delta_o_top else "",
            "selected_block_ids_sampled": sorted(selected_set)[:64],
            "num_selected": int(len(selected_set)),
            "num_keep_target": int(num_blocks_to_keep),
            "first_block_selected": bool(original_block_ids and original_block_ids[0] in selected_set),
        }

    runtime_stats = cfg.get("runtime_stats")
    if isinstance(runtime_stats, dict) and num_blocks_to_keep > 0 and qk_block_scores.numel() > 0:
        qk_k = min(int(num_blocks_to_keep), qk_block_scores.numel())
        _, qk_top_indices = torch.topk(qk_block_scores, k=qk_k)
        qk_ids = {original_block_ids[i] for i in qk_top_indices.tolist()}.union(must_keep_block_ids)
        inter = len(final_kept_ids.intersection(qk_ids))
        union = len(final_kept_ids.union(qk_ids))
        if union > 0:
            runtime_stats["jaccard_vs_original_sum"] = float(runtime_stats.get("jaccard_vs_original_sum", 0.0)) + float(inter / union)
            runtime_stats["jaccard_vs_original_count"] = int(runtime_stats.get("jaccard_vs_original_count", 0)) + 1

    _maybe_record_sp_trajectory_layer(
        cfg=cfg,
        q_for_pruning=q_for_pruning,
        represent_keys=represent_keys,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        qk_block_scores=qk_block_scores,
        block_to_kernel_map=block_to_kernel_map,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        final_kept_ids=final_kept_ids,
    )

    _write_uasr_score_dump(
        cfg=cfg,
        original_block_ids=original_block_ids,
        final_kept_ids=final_kept_ids,
        block_scores=block_scores,
        qk_block_scores=qk_block_scores,
        delta_o_values=delta_o_values,
        num_blocks_to_keep=num_blocks_to_keep,
        must_keep_block_ids=must_keep_block_ids,
    )

    _write_pruning_stats(
        cfg=cfg,
        original_block_ids=original_block_ids,
        final_kept_ids=final_kept_ids,
        block_scores=block_scores,
        delta_o_values=delta_o_values,
        delta_h_values=delta_h_values,
        delta_s_values=delta_s_values,
        block_modes=block_modes,
        num_blocks_to_keep=num_blocks_to_keep,
    )

    if str(cfg.get("rescue_policy", "none") or "none") in {"none", "sliminfer_larger"}:
        _write_generic_pruning_audit(
            cfg=cfg,
            original_block_ids=original_block_ids,
            final_ids=final_kept_ids,
            block_scores=qk_block_scores,
            num_blocks_to_keep=num_blocks_to_keep,
            final_budget_type="K+M" if str(cfg.get("rescue_policy", "none") or "none") == "sliminfer_larger" else "K",
        )

    if _selection_trace_enabled(cfg):
        _write_unified_selection_audit(
            cfg,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            final_kept_ids=final_kept_ids,
        )
    elif _selection_digest_enabled(cfg):
        _write_lightweight_selection_digest(
            cfg,
            original_block_ids=original_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            final_kept_ids=final_kept_ids,
        )

    _update_score_runtime_stats(cfg, score_start)
    return final_kept_ids

def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    finite_scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    min_val = finite_scores.min()
    max_val = finite_scores.max()
    denom = (max_val - min_val).clamp_min(1e-6)
    return (finite_scores - min_val) / denom

def _tensor_summary(prefix: str, scores: torch.Tensor) -> Dict[str, float]:
    if scores is None or scores.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    finite = torch.where(torch.isfinite(scores.detach().float()), scores.detach().float(), torch.zeros_like(scores.detach().float()))
    return {
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_std": float(finite.std(unbiased=False).item()),
        f"{prefix}_min": float(finite.min().item()),
        f"{prefix}_max": float(finite.max().item()),
    }

def _top_block_ids_for_summary(scores: torch.Tensor, original_block_ids: List[int], k: int = 10) -> List[int]:
    if scores is None or scores.numel() == 0 or not original_block_ids:
        return []
    top_k = min(int(k), int(scores.numel()), len(original_block_ids))
    _, indices = torch.topk(scores.detach().float(), k=top_k)
    return [int(original_block_ids[int(i)]) for i in indices.detach().cpu().tolist()]

def _top_block_id_set(scores: torch.Tensor, original_block_ids: List[int], k: int) -> Set[int]:
    if scores is None or scores.numel() == 0 or not original_block_ids or k <= 0:
        return set()
    top_k = min(int(k), int(scores.numel()), len(original_block_ids))
    _, indices = torch.topk(scores.detach().float(), k=top_k)
    return {int(original_block_ids[int(i)]) for i in indices.detach().cpu().tolist()}

def _jaccard(left: Set[int], right: Set[int]) -> float:
    union = left.union(right)
    if not union:
        return 0.0
    return float(len(left.intersection(right)) / len(union))

def _scatter_local_scores(scores: Optional[torch.Tensor], original_block_ids: List[int], device: torch.device) -> torch.Tensor:
    length = (max(original_block_ids) + 1) if original_block_ids else 0
    out = torch.zeros(length, dtype=torch.float32, device=device)
    if scores is None or len(original_block_ids) == 0:
        return out
    local = torch.where(torch.isfinite(scores.detach().float()), scores.detach().float(), torch.zeros_like(scores.detach().float())).to(device)
    limit = min(local.numel(), len(original_block_ids))
    for local_idx in range(limit):
        block_id = int(original_block_ids[local_idx])
        if 0 <= block_id < out.numel():
            out[block_id] = local[local_idx]
    return out

def _block_value_representations_global(
    represent_values: Optional[torch.Tensor],
    num_key_value_groups: int,
    block_slices: List[Tuple[int, int]],
    original_block_ids: List[int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if represent_values is None or represent_values.shape[2] == 0 or not original_block_ids:
        return None
    values = repeat_kv(represent_values, num_key_value_groups).float()
    dim = int(values.shape[-1])
    reps = torch.zeros(max(original_block_ids) + 1, dim, dtype=torch.float32, device=device)
    for local_idx, (start, num_kernels) in enumerate(block_slices):
        if local_idx >= len(original_block_ids) or num_kernels <= 0:
            continue
        block_id = int(original_block_ids[local_idx])
        end = start + num_kernels
        reps[block_id] = values[:, :, start:end, :].mean(dim=(0, 1, 2))
    return reps


def _drm_trace_dir(cfg: Dict[str, Any]) -> Optional[str]:
    path = str(cfg.get("drm_trace_dir", "") or "")
    if not path or str(cfg.get("phase", "")) == "decode":
        return None
    return path


def _drm_json_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return float(result)


def _drm_tensor_value(scores: Optional[torch.Tensor], block_id: int) -> Optional[float]:
    if scores is None or scores.numel() == 0:
        return None
    idx = int(block_id)
    if idx < 0 or idx >= int(scores.numel()):
        return None
    return _drm_json_float(scores[idx].detach().float().cpu())


def _drm_subset_z_map(scores: Optional[torch.Tensor], block_ids: List[int]) -> Dict[int, float]:
    if scores is None or scores.numel() == 0:
        return {}
    valid = [int(x) for x in block_ids if 0 <= int(x) < int(scores.numel())]
    if not valid:
        return {}
    idx = torch.tensor(valid, dtype=torch.long, device=scores.device)
    vals = scores[idx].detach().float()
    finite = torch.isfinite(vals)
    if not bool(finite.any().item()):
        return {int(block_id): 0.0 for block_id in valid}
    mean = vals[finite].mean()
    std = vals[finite].std(unbiased=False)
    if not torch.isfinite(std) or float(std.abs()) <= 1e-12:
        z = torch.zeros_like(vals)
    else:
        z = torch.where(finite, (vals - mean) / std, torch.zeros_like(vals))
    return {
        int(block_id): float(z[pos].detach().cpu())
        for pos, block_id in enumerate(valid)
    }


def _drm_rep_vector(
    reps: Optional[torch.Tensor],
    block_id: int,
    *,
    decimals: int = 6,
) -> Optional[List[float]]:
    if reps is None or reps.numel() == 0:
        return None
    idx = int(block_id)
    if idx < 0 or idx >= int(reps.shape[0]):
        return None
    vec = reps[idx].detach().float().cpu().tolist()
    return [round(float(x), int(decimals)) for x in vec]


def _drm_json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return _drm_json_float(tensor.item())
        return _drm_json_safe(tensor.tolist())
    if isinstance(value, dict):
        return {str(_drm_json_safe(k)): _drm_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_drm_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_drm_json_safe(v) for v in sorted(value, key=lambda x: str(x))]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _drm_json_safe(item())
        except Exception:
            pass
    return str(value)


def _drm_write_jsonl(path: str, row: Dict[str, Any]) -> None:
    import os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(str(path))) or ".", exist_ok=True)
    with open(str(path), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_drm_json_safe(row), ensure_ascii=False) + "\n")


def _write_drm_runtime_trace(
    *,
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    active_ids: List[int],
    inactive_ids: List[int],
    final_ids: List[int],
    proposed_rescue_ids: List[int],
    applied_rescue_ids: List[int],
    dropped_ids: List[int],
    protected_ids: Set[int],
    rescue_scores: torch.Tensor,
    reps: Optional[torch.Tensor],
    score_meta: Dict[str, Any],
    target_m: int,
    final_budget_type: str,
    rejected_reason: str = "",
    mmr_diag: Optional[Dict[str, Any]] = None,
    active_scores: Optional[torch.Tensor] = None,
) -> None:
    trace_dir = _drm_trace_dir(cfg)
    if not trace_dir:
        return
    mmr_diag = mmr_diag or {}
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    layer_idx = int(cfg.get("layer_idx", -1))
    method = str(context.get("method", cfg.get("variant_method", "")) or "")
    variant_method = str(cfg.get("variant_method", "") or "")
    rescue_policy = str(cfg.get("rescue_policy", "none") or "none")
    rescue_score = str(cfg.get("rescue_score", "delta_o") or "delta_o")
    tensors = score_meta.get("_drm_trace_tensors", {}) if isinstance(score_meta.get("_drm_trace_tensors"), dict) else {}
    d_base = tensors.get("D_base")
    c_active = tensors.get("C_active")
    g_diff = tensors.get("G_diffusion")
    r_active = tensors.get("R_active")
    qk_scores = tensors.get("QK")
    z_d = _drm_subset_z_map(d_base, inactive_ids)
    z_g = _drm_subset_z_map(g_diff, inactive_ids)
    z_r = _drm_subset_z_map(r_active, inactive_ids)
    active_set = set(int(x) for x in active_ids)
    inactive_set = set(int(x) for x in inactive_ids)
    proposed_set = set(int(x) for x in proposed_rescue_ids)
    applied_set = set(int(x) for x in applied_rescue_ids)
    dropped_set = set(int(x) for x in dropped_ids)
    final_set = set(int(x) for x in final_ids)
    protected_set = set(int(x) for x in protected_ids)
    finite_candidates = [
        int(block_id)
        for block_id in inactive_ids
        if _drm_tensor_value(rescue_scores, int(block_id)) is not None
    ]
    feature_missing: List[str] = []
    if rescue_score == "drm":
        if d_base is None or d_base.numel() == 0:
            feature_missing.append("D_base")
        if g_diff is None or g_diff.numel() == 0:
            feature_missing.append("G")
        if r_active is None or r_active.numel() == 0:
            feature_missing.append("R_active")
        if reps is None or reps.numel() == 0:
            feature_missing.append("block_representation")
        if not mmr_diag.get("MMR_selection_trace") and int(target_m) > 0:
            feature_missing.append("MMR_selection_trace")
    if not original_block_ids:
        feature_missing.append("original_block_ids")
    sink_id = int(original_block_ids[0]) if original_block_ids else None
    recent_id = int(original_block_ids[-1]) if original_block_ids else None
    forced_sink_ids = [sink_id] if sink_id is not None else []
    forced_recent_ids = [recent_id] if recent_id is not None and recent_id != sink_id else []
    base = {
        "task": context.get("task", ""),
        "sample_id": context.get("sample_id", ""),
        "sample_index": context.get("sample_index", ""),
        "method": method,
        "variant_method": variant_method,
        "layer_idx": layer_idx,
        "phase": cfg.get("phase", ""),
        "rescue_policy": rescue_policy,
        "rescue_score": rescue_score,
        "rescue_ratio": float(cfg.get("rescue_ratio", 0.05) or 0.0),
        "alpha_diff": float(cfg.get("alpha_diff", 0.5) or 0.5),
        "lambda_red": float(cfg.get("lambda_red", 0.25) or 0.25),
        "eta_rescue": float(cfg.get("eta_rescue", 0.25) or 0.25),
        "eta_active": float(cfg.get("eta_active", 0.25) or 0.25),
        "seed": int(cfg.get("seed", 0) or 0),
    }
    trace_paths = {
        "candidate": f"{trace_dir}/drm_candidate_trace.jsonl",
        "mmr": f"{trace_dir}/drm_mmr_selection_trace.jsonl",
        "selection": f"{trace_dir}/drm_selection_trace.jsonl",
        "reps": f"{trace_dir}/drm_block_representation_trace.jsonl",
        "mapping": f"{trace_dir}/drm_virtual_physical_mapping.jsonl",
    }
    plain_top = set(int(x) for x in (mmr_diag.get("plain_top_m_ids") or []))
    uses_mmr = bool(mmr_diag.get("MMR_selection_trace"))
    for block_id in inactive_ids:
        block_id = int(block_id)
        if block_id in applied_set:
            candidate_rejected_reason = ""
        elif block_id in proposed_set:
            candidate_rejected_reason = str(rejected_reason or "selected_but_not_applied")
        elif block_id in set(finite_candidates):
            candidate_rejected_reason = "not_selected_by_mmr" if uses_mmr else "not_selected_by_score"
        else:
            candidate_rejected_reason = "not_eligible"
        row = {
            **base,
            "block_id": block_id,
            "from_pruned_set": block_id in inactive_set,
            "candidate_available": True,
            "candidate_eligible": block_id in set(finite_candidates),
            "candidate_available_count": int(len(inactive_ids)),
            "candidate_eligible_count": int(len(finite_candidates)),
            "target_m": int(target_m),
            "D_base": _drm_tensor_value(d_base, block_id),
            "G": _drm_tensor_value(g_diff, block_id),
            "R_active": _drm_tensor_value(r_active, block_id),
            "C_active": _drm_tensor_value(c_active, block_id),
            "z_D_base": _drm_json_float(z_d.get(block_id)),
            "z_G": _drm_json_float(z_g.get(block_id)),
            "z_R_active": _drm_json_float(z_r.get(block_id)),
            "drm_score_s0": _drm_tensor_value(rescue_scores, block_id),
            "qk_score": _drm_tensor_value(qk_scores, block_id),
            "plain_top_m": block_id in plain_top,
            "mmr_selected": block_id in proposed_set,
            "add_or_swap_applied": block_id in applied_set,
            "rejected_reason": candidate_rejected_reason,
            "feature_complete": not feature_missing,
            "missing_features": feature_missing,
            "block_representation_dim": int(reps.shape[-1]) if reps is not None and reps.numel() else 0,
        }
        _drm_write_jsonl(trace_paths["candidate"], row)
    for row in mmr_diag.get("MMR_selection_trace") or []:
        if not isinstance(row, dict):
            continue
        _drm_write_jsonl(trace_paths["mmr"], {**base, **row})
    rep_decimals = int(cfg.get("drm_trace_rep_decimals", 6) or 6)
    for block_id in original_block_ids:
        block_id = int(block_id)
        _drm_write_jsonl(trace_paths["reps"], {
            **base,
            "block_id": block_id,
            "is_active_before": block_id in active_set,
            "is_pruned_candidate": block_id in inactive_set,
            "is_final": block_id in final_set,
            "representation_dim": int(reps.shape[-1]) if reps is not None and reps.numel() else 0,
            "block_representation": _drm_rep_vector(reps, block_id, decimals=rep_decimals),
        })
    _drm_write_jsonl(trace_paths["mapping"], {
        **base,
        "virtual_block_path_used": False,
        "physical_block_mapping_valid": True,
        "mapping_type": "identity_fixed_block",
        "virtual_to_physical": [
            {"virtual_block_id": int(block_id), "physical_block_ids": [int(block_id)]}
            for block_id in original_block_ids
        ],
    })
    summary = {
        **base,
        "base_active_count": int(len(active_ids)),
        "candidate_available_count": int(len(inactive_ids)),
        "candidate_eligible_count": int(len(finite_candidates)),
        "drm_mmr_selected_count": int(len(proposed_rescue_ids)),
        "add_or_swap_applied_count": int(len(applied_rescue_ids)),
        "swap_removed_count": int(len(dropped_ids)),
        "target_m": int(target_m),
        "final_budget_type": final_budget_type,
        "final_count": int(len(final_ids)),
        "active_block_ids": [int(x) for x in active_ids],
        "pruned_candidate_block_ids": [int(x) for x in inactive_ids],
        "proposed_rescue_block_ids": [int(x) for x in proposed_rescue_ids],
        "applied_rescue_block_ids": [int(x) for x in applied_rescue_ids],
        "dropped_block_ids": [int(x) for x in dropped_ids],
        "final_block_ids": [int(x) for x in final_ids],
        "forced_sink_block_ids": forced_sink_ids,
        "forced_recent_block_ids": forced_recent_ids,
        "protected_block_ids": sorted(int(x) for x in protected_set),
        "rejected_reason": str(rejected_reason or ""),
        "feature_complete": not feature_missing,
        "missing_features": feature_missing,
        "mmr_vs_plain_top_m_jaccard": _drm_json_float(mmr_diag.get("mmr_vs_plain_top_m_jaccard")),
        "plain_top_m_ids": [int(x) for x in (mmr_diag.get("plain_top_m_ids") or [])],
        "virtual_block_path_used": False,
        "physical_block_mapping_valid": True,
    }
    _drm_write_jsonl(trace_paths["selection"], summary)

def _obd_scores_global(
    kernel_scores: torch.Tensor,
    represent_values: Optional[torch.Tensor],
    num_key_value_groups: int,
    block_slices: List[Tuple[int, int]],
    original_block_ids: List[int],
    device: torch.device,
    diagnostics: Dict[str, List[str]],
) -> torch.Tensor:
    if represent_values is None or represent_values.shape[2] == 0 or not original_block_ids:
        diagnostics.setdefault("warnings", []).append("OBD rescue score unavailable in cache path; missing represent_values.")
        return torch.zeros((max(original_block_ids) + 1) if original_block_ids else 0, dtype=torch.float32, device=device)
    probs = torch.softmax(kernel_scores.float(), dim=-1)
    values = repeat_kv(represent_values, num_key_value_groups).float()
    token_block_ids: List[int] = []
    for local_idx, (_, num_kernels) in enumerate(block_slices):
        token_block_ids.extend([local_idx] * max(0, int(num_kernels)))
    if not token_block_ids:
        return torch.zeros((max(original_block_ids) + 1) if original_block_ids else 0, dtype=torch.float32, device=device)
    local_obd = block_obd_saliency(
        attn_probs=probs,
        values=values,
        block_ids=torch.tensor(token_block_ids, dtype=torch.long, device=device),
        num_blocks=len(original_block_ids),
        diagnostics=diagnostics,
    )
    return _scatter_local_scores(local_obd, original_block_ids, device)

def _query_drm_score_bundle(
    cfg: Dict[str, Any],
    base_global: torch.Tensor,
    reps: Optional[torch.Tensor],
    active_ids: List[int],
    inactive_ids: List[int],
    diagnostics: Dict[str, List[str]],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    block_texts = cfg.get("block_texts") or cfg.get("audit_context", {}).get("block_texts")
    query_text = cfg.get("query_text") or cfg.get("audit_context", {}).get("query_text")
    s0, qmeta = query_drm_s0_score(
        base_global,
        reps,
        active_ids,
        inactive_ids,
        block_texts=block_texts,
        query_text=query_text,
        alpha_diff=float(cfg.get("alpha_diff", 0.5)),
        lambda_red=float(cfg.get("lambda_red", 0.25)),
        beta_query=float(cfg.get("beta_query", 0.5)),
        diagnostics=diagnostics,
    )
    meta: Dict[str, Any] = {
        "mmr_selection": True,
        "C_active_stats": _score_stats_for_block_ids(qmeta["C_active"], inactive_ids),
        "G_diffusion_stats": _score_stats_for_block_ids(qmeta["G_diffusion"], inactive_ids),
        "R_active_stats": _score_stats_for_block_ids(qmeta["R_active"], inactive_ids),
        "E_query_stats": qmeta.get("query_overlap_stats", {}),
        "S_query_drm_stats": qmeta.get("S_query_drm_stats", {}),
        "D_base_stats": qmeta.get("D_base_stats", {}),
        "query_fallback_used": bool(qmeta.get("query_fallback_used", False)),
        "query_fallback_reason": str(qmeta.get("query_fallback_reason", "")),
        "drm_used_base": bool(float(qmeta.get("used_base", torch.tensor([0.0]))[0].item()) > 0.5),
    }
    return s0, meta


def _active_redundancy_scores(
    reps: Optional[torch.Tensor],
    active_ids: List[int],
    num_blocks: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if reps is None or reps.numel() == 0 or len(active_ids) < 2:
        return None
    valid_ids = [int(block_id) for block_id in active_ids if 0 <= int(block_id) < reps.shape[0]]
    if len(valid_ids) < 2:
        return None
    idx = torch.tensor(valid_ids, dtype=torch.long, device=device)
    active_reps = torch.nn.functional.normalize(reps[idx].float(), dim=-1, eps=1e-6)
    similarity = active_reps @ active_reps.transpose(0, 1)
    similarity.fill_diagonal_(float("-inf"))
    redundancy = similarity.max(dim=1).values
    redundancy = torch.nan_to_num(redundancy, nan=0.0, posinf=0.0, neginf=0.0)
    result = torch.zeros(max(0, int(num_blocks)), dtype=torch.float32, device=device)
    result[idx] = redundancy
    return result


def _normalized_subset_scores(
    scores: torch.Tensor,
    block_ids: List[int],
) -> torch.Tensor:
    result = torch.zeros_like(scores, dtype=torch.float32)
    valid_ids = [int(block_id) for block_id in block_ids if 0 <= int(block_id) < scores.numel()]
    if not valid_ids:
        return result
    idx = torch.tensor(valid_ids, dtype=torch.long, device=scores.device)
    result[idx] = normalize_scores(scores[idx].float(), "minmax")
    return result


def _mean_score_for_ids(scores: torch.Tensor, block_ids: List[int]) -> float:
    valid_ids = [int(block_id) for block_id in block_ids if 0 <= int(block_id) < scores.numel()]
    if not valid_ids:
        return 0.0
    idx = torch.tensor(valid_ids, dtype=torch.long, device=scores.device)
    return float(scores[idx].float().mean().detach().cpu())


def _max_score_for_ids(scores: torch.Tensor, block_ids: List[int]) -> Optional[float]:
    valid_ids = [int(block_id) for block_id in block_ids if 0 <= int(block_id) < scores.numel()]
    if not valid_ids:
        return None
    idx = torch.tensor(valid_ids, dtype=torch.long, device=scores.device)
    return float(scores[idx].float().max().detach().cpu())


def _score_values_for_ids(scores: Optional[torch.Tensor], block_ids: Iterable[int]) -> List[float]:
    if scores is None or scores.numel() == 0:
        return []
    values: List[float] = []
    for block_id in block_ids:
        idx = int(block_id)
        if 0 <= idx < scores.numel():
            value = float(scores[idx].detach().float().cpu())
            if math.isfinite(value):
                values.append(value)
    return values


def _mean_or_none(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _max_or_none(values: List[float]) -> Optional[float]:
    return float(max(values)) if values else None


def _iqr_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(float(v) for v in values)

    def percentile(p: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        pos = (len(sorted_values) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return sorted_values[lo]
        return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)

    return float(percentile(0.75) - percentile(0.25))


def _std_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.std(unbiased=False).item())


def _block_text_previews(
    cfg: Dict[str, Any],
    block_ids: Iterable[int],
    max_chars: int = 160,
) -> List[Dict[str, Any]]:
    block_texts = cfg.get("block_texts") or cfg.get("audit_context", {}).get("block_texts")
    if not isinstance(block_texts, list):
        return []
    rows: List[Dict[str, Any]] = []
    for block_id in block_ids:
        idx = int(block_id)
        text = str(block_texts[idx]) if 0 <= idx < len(block_texts) else ""
        rows.append({
            "block_id": idx,
            "text_preview": text.replace("\n", " ")[:max_chars],
        })
    return rows


def _rescue_redundancy_against_active(
    reps: Optional[torch.Tensor],
    active_ids: List[int],
    target_ids: List[int],
    num_blocks: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if reps is None or reps.numel() == 0 or not active_ids or not target_ids:
        return None
    valid_active = [int(block_id) for block_id in active_ids if 0 <= int(block_id) < reps.shape[0]]
    valid_targets = [int(block_id) for block_id in target_ids if 0 <= int(block_id) < reps.shape[0]]
    if not valid_active or not valid_targets:
        return None
    active_idx = torch.tensor(valid_active, dtype=torch.long, device=device)
    target_idx = torch.tensor(valid_targets, dtype=torch.long, device=device)
    active_reps = torch.nn.functional.normalize(reps[active_idx].float(), dim=-1, eps=1e-6)
    target_reps = torch.nn.functional.normalize(reps[target_idx].float(), dim=-1, eps=1e-6)
    similarity = target_reps @ active_reps.transpose(0, 1)
    redundancy = torch.nan_to_num(similarity.max(dim=1).values, nan=0.0, posinf=0.0, neginf=0.0)
    result = torch.zeros(max(0, int(num_blocks)), dtype=torch.float32, device=device)
    result[target_idx] = redundancy
    return result


def _residual_aware_proxy_fields(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    active_ids: List[int],
    inactive_ids: List[int],
    rescue_ids: List[int],
    dropped_ids: List[int],
    protected_ids: Set[int],
    query_critical_ids: Set[int],
    rescue_scores: torch.Tensor,
    active_global: torch.Tensor,
    reps: Optional[torch.Tensor],
    redundancy_scores: Optional[torch.Tensor],
    represent_values: Optional[torch.Tensor],
    num_key_value_groups: int,
    block_slices: List[Tuple[int, int]],
) -> Dict[str, Any]:
    device = active_global.device
    unavailable: List[str] = []
    value_norms_local = _calculate_block_value_norms(
        represent_values,
        num_key_value_groups,
        block_slices,
        device,
    )
    value_norms_global: Optional[torch.Tensor] = None
    if value_norms_local is None:
        unavailable.append("value_norm_proxy")
    else:
        value_norms_global = _scatter_local_scores(value_norms_local, original_block_ids, device)

    candidate_scores = _score_values_for_ids(rescue_scores, inactive_ids)
    sorted_candidate_scores = sorted(candidate_scores, reverse=True)
    rescue_score_top1 = sorted_candidate_scores[0] if sorted_candidate_scores else None
    rescue_score_top2 = sorted_candidate_scores[1] if len(sorted_candidate_scores) > 1 else None
    rescue_score_margin = (
        float(rescue_score_top1 - rescue_score_top2)
        if rescue_score_top1 is not None and rescue_score_top2 is not None
        else None
    )

    normalized_removed_score = _normalized_subset_scores(active_global, dropped_ids)
    normalized_rescue_score = _normalized_subset_scores(rescue_scores, rescue_ids)
    normalized_value_norm = (
        _normalized_subset_scores(value_norms_global, list(set(dropped_ids) | set(rescue_ids)))
        if value_norms_global is not None
        else None
    )
    rescue_redundancy = _rescue_redundancy_against_active(
        reps,
        active_ids,
        rescue_ids,
        (max(original_block_ids) + 1) if original_block_ids else 0,
        device,
    )

    delete_risk_values: List[float] = []
    rescue_gain_values: List[float] = []
    if normalized_value_norm is None:
        unavailable.append("delete_risk_proxy")
        unavailable.append("rescue_gain_proxy")
    else:
        for block_id in dropped_ids:
            idx = int(block_id)
            if 0 <= idx < normalized_removed_score.numel() and 0 <= idx < normalized_value_norm.numel():
                penalty = 2.0 if idx in protected_ids or idx in query_critical_ids else 1.0
                delete_risk_values.append(
                    float(
                        normalized_removed_score[idx].detach().float().cpu()
                        * normalized_value_norm[idx].detach().float().cpu()
                        * penalty
                    )
                )
        for block_id in rescue_ids:
            idx = int(block_id)
            if 0 <= idx < normalized_rescue_score.numel() and 0 <= idx < normalized_value_norm.numel():
                redundancy_penalty = (
                    float(rescue_redundancy[idx].detach().float().cpu())
                    if rescue_redundancy is not None and 0 <= idx < rescue_redundancy.numel()
                    else 0.0
                )
                rescue_gain_values.append(
                    float(
                        normalized_rescue_score[idx].detach().float().cpu()
                        * normalized_value_norm[idx].detach().float().cpu()
                        - redundancy_penalty
                    )
                )
    residual_benefit_proxy = (
        float(_mean_or_none(rescue_gain_values) - _mean_or_none(delete_risk_values))
        if rescue_gain_values and delete_risk_values
        else None
    )
    if residual_benefit_proxy is None:
        unavailable.append("residual_benefit_proxy")

    removed_value_norms = (
        _score_values_for_ids(value_norms_global, dropped_ids)
        if value_norms_global is not None
        else []
    )
    rescued_value_norms = (
        _score_values_for_ids(value_norms_global, rescue_ids)
        if value_norms_global is not None
        else []
    )
    removed_qk_or_delta = _score_values_for_ids(active_global, dropped_ids)
    rescued_qk_or_delta = _score_values_for_ids(active_global, rescue_ids)

    return {
        "removed_value_norm_mean": _mean_or_none(removed_value_norms),
        "removed_value_norm_max": _max_or_none(removed_value_norms),
        "rescued_value_norm_mean": _mean_or_none(rescued_value_norms),
        "rescued_value_norm_max": _max_or_none(rescued_value_norms),
        "removed_qk_or_delta_score_mean": _mean_or_none(removed_qk_or_delta),
        "removed_qk_or_delta_score_max": _max_or_none(removed_qk_or_delta),
        "rescued_qk_or_delta_score_mean": _mean_or_none(rescued_qk_or_delta),
        "rescued_qk_or_delta_score_max": _max_or_none(rescued_qk_or_delta),
        "delete_risk_proxy_mean": _mean_or_none(delete_risk_values),
        "delete_risk_proxy_max": _max_or_none(delete_risk_values),
        "rescue_gain_proxy_mean": _mean_or_none(rescue_gain_values),
        "rescue_gain_proxy_max": _max_or_none(rescue_gain_values),
        "residual_benefit_proxy": residual_benefit_proxy,
        "rescue_score_top1": rescue_score_top1,
        "rescue_score_top2": rescue_score_top2,
        "rescue_score_margin": rescue_score_margin,
        "candidate_score_std": _std_or_none(candidate_scores),
        "candidate_score_iqr": _iqr_or_none(candidate_scores),
        "candidate_rescued_block_ids": [int(block_id) for block_id in rescue_ids],
        "candidate_removed_block_ids": [int(block_id) for block_id in dropped_ids],
        "removed_block_text_preview": _block_text_previews(cfg, dropped_ids),
        "rescued_block_text_preview": _block_text_previews(cfg, rescue_ids),
        "residual_proxy_unavailable_fields": sorted(set(unavailable)),
        "residual_proxy_version": "value_norm_v0_audit_only",
    }


def _score_rows_for_ids(
    scores: Optional[torch.Tensor],
    block_ids: Iterable[int],
    score_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for block_id in block_ids:
        idx = int(block_id)
        value: Optional[float] = None
        if scores is not None and 0 <= idx < scores.numel():
            raw = float(scores[idx].detach().float().cpu())
            value = raw if math.isfinite(raw) else None
        rows.append({"block_id": idx, score_name: value})
    return rows


_GATE_QUERY_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and",
    "or", "in", "on", "at", "for", "from", "by", "with", "as", "that", "this",
    "which", "what", "who", "whom", "whose", "when", "where", "why", "how",
    "did", "does", "do", "has", "have", "had", "its", "their", "his", "her",
}
_GATE_DATE_NUMBER_PATTERN = re.compile(
    r"\b(?:\d{1,4}(?:[-/:]\d{1,4})*|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    flags=re.IGNORECASE,
)
_GATE_CAPITALIZED_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z'’-]+(?:\s+(?:(?:of|the|and)\s+)?[A-Z][A-Za-z'’-]+){0,4}\b"
)
_GATE_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
_GATE_QUOTED_PATTERN = re.compile(r"[\"“]([^\"”]{2,80})[\"”]")
_GATE_HYPHENATED_PATTERN = re.compile(r"\b[A-Za-z]+(?:[-–—][A-Za-z]+)+\b")
_GATE_STRUCTURAL_LABEL_PATTERN = re.compile(
    r"\b(?:passage|document|article|chunk)\s*\d+\s*:?",
    flags=re.IGNORECASE,
)
_GATE_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_GATE_ENTITY_NOISE = {
    "answer", "answers", "article", "assistant", "biography", "chunk",
    "context", "document", "passage", "question", "user",
}


def _gate_normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _gate_evidence_units(query_text: str) -> Dict[str, List[str]]:
    query = str(query_text or "")
    query_terms = sorted({
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’_-]*", query)
        if len(token) >= 3 and token.lower() not in _GATE_QUERY_STOPWORDS
    })
    entities = {
        item.strip().lower()
        for item in _GATE_CAPITALIZED_PATTERN.findall(query)
        if item.strip().lower() not in {"what", "who", "when", "where", "which", "how"}
    }
    entities.update(item.strip().lower() for item in _GATE_ACRONYM_PATTERN.findall(query))
    entities.update(item.strip().lower() for item in _GATE_QUOTED_PATTERN.findall(query))
    entities.update(item.strip().lower() for item in _GATE_HYPHENATED_PATTERN.findall(query))
    date_numbers = {
        item.strip().lower() for item in _GATE_DATE_NUMBER_PATTERN.findall(query)
    }
    return {
        "query_terms": query_terms,
        "entity_units": sorted(item for item in entities if item),
        "date_number_units": sorted(item for item in date_numbers if item),
    }


def _gate_entity_like_units(text: str) -> List[str]:
    """Cheap entity/title/date proxy over arbitrary context text."""
    cleaned = _GATE_STRUCTURAL_LABEL_PATTERN.sub(" ", str(text or ""))
    units: Set[str] = {
        item.strip().lower()
        for item in _GATE_CAPITALIZED_PATTERN.findall(cleaned)
        if item.strip()
    }
    units.update(
        item.strip().lower() for item in _GATE_ACRONYM_PATTERN.findall(cleaned)
    )
    units.update(
        item.strip().lower() for item in _GATE_QUOTED_PATTERN.findall(cleaned)
    )
    units.update(
        item.strip().lower() for item in _GATE_HYPHENATED_PATTERN.findall(cleaned)
    )
    units.update(
        item.strip().lower() for item in _GATE_DATE_NUMBER_PATTERN.findall(cleaned)
    )
    normalized: Set[str] = set()
    for unit in units:
        value = _gate_normalize_text(unit).strip(" \t\r\n.,;:!?()[]{}")
        if not value or len(value) > 120:
            continue
        words = re.findall(r"[a-z0-9][a-z0-9'’_-]*", value)
        if not words:
            continue
        if all(
            word in _GATE_QUERY_STOPWORDS or word in _GATE_ENTITY_NOISE
            for word in words
        ):
            continue
        normalized.add(value)
    return sorted(normalized)


def _gate_date_number_units(text: str) -> List[str]:
    cleaned = _GATE_STRUCTURAL_LABEL_PATTERN.sub(" ", str(text or ""))
    return sorted({
        item.strip().lower()
        for item in _GATE_DATE_NUMBER_PATTERN.findall(cleaned)
        if item.strip()
    })


def _gate_content_tokens(text: str) -> Set[str]:
    cleaned = _GATE_STRUCTURAL_LABEL_PATTERN.sub(" ", str(text or "").lower())
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'’_-]*", cleaned)
        if (
            (len(token) >= 3 or token.isdigit())
            and token not in _GATE_QUERY_STOPWORDS
            and token not in _GATE_ENTITY_NOISE
        )
    }


def _gate_token_jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left.intersection(right)) / len(left.union(right)))


def _gate_sentence_token_sets(block_texts: Iterable[str]) -> List[Set[str]]:
    sentences: List[Set[str]] = []
    for block_text in block_texts:
        for span in _GATE_SENTENCE_SPLIT_PATTERN.split(str(block_text or "")):
            tokens = _gate_content_tokens(span)
            if len(tokens) >= 2:
                sentences.append(tokens)
    return sentences


def _gate_answer_type_proxy(query_text: str) -> str:
    normalized = _gate_normalize_text(query_text)
    if re.search(r"\bhow\s+many\b|\bhow\s+much\b", normalized):
        return "how_many"
    if re.search(r"\bwhen\b|\bwhat\s+(?:year|date|month|day)\b", normalized):
        return "when"
    if re.search(r"\bwho\b|\bwhom\b|\bwhose\b", normalized):
        return "who"
    if re.search(r"\bwhere\b|\bwhich\s+(?:place|city|country|state)\b", normalized):
        return "where"
    if re.search(r"\bwhich\b", normalized):
        return "which"
    if re.search(r"\bwhat\b", normalized):
        return "what"
    return "other"


def _gate_clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _gate_percentile(values: List[float], quantile: float) -> Optional[float]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = _gate_clip01(quantile) * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return float(finite[lower] * (1.0 - fraction) + finite[upper] * fraction)


def _gate_rank_percentile(value: float, pool: List[float]) -> Optional[float]:
    finite = [float(item) for item in pool if math.isfinite(float(item))]
    if not finite or not math.isfinite(float(value)):
        return None
    if len(finite) == 1:
        return 0.5
    less = sum(item < float(value) for item in finite)
    equal = sum(item == float(value) for item in finite)
    return _gate_clip01((less + 0.5 * max(0, equal - 1)) / (len(finite) - 1))


def _gate_confident_entity_units(
    block_texts: Iterable[str],
    configured_titles: Iterable[str],
) -> List[str]:
    texts = [str(text or "") for text in block_texts]
    counts: Counter = Counter()
    strong: Set[str] = {
        _gate_normalize_text(title) for title in configured_titles if title
    }
    for text in texts:
        units = set(_gate_entity_like_units(text))
        counts.update(units)
        cleaned = _GATE_STRUCTURAL_LABEL_PATTERN.sub(" ", text)
        strong.update(
            _gate_normalize_text(item)
            for item in _GATE_QUOTED_PATTERN.findall(cleaned)
            if item.strip()
        )
        strong.update(
            _gate_normalize_text(item)
            for item in _GATE_HYPHENATED_PATTERN.findall(cleaned)
            if item.strip()
        )
        strong.update(
            _gate_normalize_text(item)
            for item in _GATE_ACRONYM_PATTERN.findall(cleaned)
            if item.strip()
        )
        strong.update(_gate_date_number_units(cleaned))
        strong.update(
            unit for unit in units if len(unit.split()) >= 2
        )
    return sorted(
        unit for unit, count in counts.items() if count >= 2 or unit in strong
    )


def _gate_novelty_scores(
    block_texts: Iterable[str],
    reference_block_texts: Iterable[str],
) -> List[float]:
    references = [
        tokens
        for tokens in (
            _gate_content_tokens(text) for text in reference_block_texts
        )
        if tokens
    ]
    scores: List[float] = []
    for text in block_texts:
        tokens = _gate_content_tokens(text)
        if not tokens:
            continue
        overlap = max(
            (_gate_token_jaccard(tokens, reference) for reference in references),
            default=0.0,
        )
        scores.append(float(1.0 - overlap))
    return scores


def _gate_answer_type_support(
    block_texts: Iterable[str],
    answer_type: str,
    configured_titles: Iterable[str],
) -> Optional[float]:
    texts = [str(text or "") for text in block_texts]
    if not texts:
        return None
    titles = [_gate_normalize_text(title) for title in configured_titles if title]
    supported = 0
    for text in texts:
        normalized = _gate_normalize_text(text)
        entities = _gate_confident_entity_units([text], titles)
        dates = _gate_date_number_units(text)
        title_hit = any(title in normalized for title in titles)
        if answer_type in {"when", "how_many"}:
            has_support = bool(dates)
        elif answer_type == "who":
            has_support = bool(entities)
        elif answer_type == "where":
            has_support = bool(
                title_hit or any(len(entity.split()) >= 2 for entity in entities)
            )
        elif answer_type in {"what", "which"}:
            has_support = bool(title_hit or entities)
        else:
            has_support = bool(title_hit or entities or dates)
        supported += int(has_support)
    return float(supported / len(texts))


def _gate_overlap(units: List[str], text: str) -> Optional[float]:
    if not units:
        return None
    normalized = _gate_normalize_text(text)
    return float(sum(unit in normalized for unit in units) / len(units))


def _gate_context_block_limit(block_texts: List[str], query_text: str) -> int:
    """Best-effort start block of the prompt-tail question, without tokenizing again."""
    normalized_query = _gate_normalize_text(query_text)
    if not block_texts or not normalized_query:
        return len(block_texts)
    # Checking adjacent decoded blocks handles questions split at a 64-token boundary.
    for block_id in range(len(block_texts) - 1, -1, -1):
        joined = _gate_normalize_text(
            str(block_texts[block_id])
            + " "
            + (str(block_texts[block_id + 1]) if block_id + 1 < len(block_texts) else "")
        )
        if normalized_query in joined:
            return block_id
    prefix_tokens = normalized_query.split()[: min(5, len(normalized_query.split()))]
    prefix = " ".join(prefix_tokens)
    if prefix:
        for block_id in range(len(block_texts) - 1, -1, -1):
            if prefix in _gate_normalize_text(block_texts[block_id]):
                return block_id
    return len(block_texts)


def _gate_join_block_text(
    block_texts: List[str],
    block_ids: Iterable[int],
    context_limit: Optional[int] = None,
) -> str:
    selected: List[str] = []
    for block_id in block_ids:
        idx = int(block_id)
        if idx < 0 or idx >= len(block_texts):
            continue
        if context_limit is not None and idx >= int(context_limit):
            continue
        selected.append(str(block_texts[idx]))
    return "\n".join(selected)


def _compute_gated_swap_features(
    *,
    cfg: Dict[str, Any],
    active_ids: List[int],
    inactive_ids: List[int],
    rescue_ids: List[int],
    dropped_ids: List[int],
    safe_drop_ids: List[int],
    protected_ids: Set[int],
    protected_reasons: Dict[int, List[str]],
    rescue_scores: torch.Tensor,
    active_base_scores: torch.Tensor,
    redundancy_scores: Optional[torch.Tensor],
    drop_utility: torch.Tensor,
) -> Dict[str, Any]:
    block_texts = list(
        cfg.get("block_texts")
        or cfg.get("audit_context", {}).get("block_texts")
        or []
    )
    query_text = str(
        cfg.get("query_text")
        or cfg.get("audit_context", {}).get("query_text")
        or ""
    )
    units = _gate_evidence_units(query_text)
    context_limit = _gate_context_block_limit(block_texts, query_text)
    rescued_text = _gate_join_block_text(block_texts, rescue_ids, context_limit)
    removed_text = _gate_join_block_text(block_texts, dropped_ids, context_limit)
    active_context_text = _gate_join_block_text(block_texts, active_ids, context_limit)
    rescued_block_texts = [
        str(block_texts[int(block_id)])
        for block_id in rescue_ids
        if 0 <= int(block_id) < min(len(block_texts), context_limit)
    ]
    removed_block_texts = [
        str(block_texts[int(block_id)])
        for block_id in dropped_ids
        if 0 <= int(block_id) < min(len(block_texts), context_limit)
    ]
    active_context_block_texts = [
        str(block_texts[int(block_id)])
        for block_id in active_ids
        if 0 <= int(block_id) < min(len(block_texts), context_limit)
    ]
    retained_active_block_texts = [
        str(block_texts[int(block_id)])
        for block_id in active_ids
        if (
            int(block_id) not in set(dropped_ids)
            and 0 <= int(block_id) < min(len(block_texts), context_limit)
        )
    ]

    query_overlap_rescued = _gate_overlap(units["query_terms"], rescued_text)
    query_overlap_removed = _gate_overlap(units["query_terms"], removed_text)
    entity_overlap_rescued = _gate_overlap(units["entity_units"], rescued_text)
    entity_overlap_removed = _gate_overlap(units["entity_units"], removed_text)
    date_number_overlap_rescued = _gate_overlap(
        units["date_number_units"], rescued_text
    )
    date_number_overlap_removed = _gate_overlap(
        units["date_number_units"], removed_text
    )

    evidence_units = sorted(set(
        units["query_terms"] + units["entity_units"] + units["date_number_units"]
    ))
    active_normalized = _gate_normalize_text(active_context_text)
    rescued_normalized = _gate_normalize_text(rescued_text)
    rescue_coverage_gap: Optional[float]
    if not evidence_units:
        rescue_coverage_gap = None
    else:
        rescue_coverage_gap = float(
            sum(
                unit in rescued_normalized and unit not in active_normalized
                for unit in evidence_units
            )
            / len(evidence_units)
        )

    gate_feature_version = str(
        cfg.get("gate_feature_version", "selector_native_v2")
        or "selector_native_v2"
    )
    question_entity_units = _gate_entity_like_units(query_text)
    rescued_entity_units = _gate_entity_like_units(rescued_text)
    active_entity_units = _gate_entity_like_units(active_context_text)
    removed_entity_units = _gate_entity_like_units(removed_text)
    question_entity_set = set(question_entity_units)
    rescued_entity_set = set(rescued_entity_units)
    active_entity_set = set(active_entity_units)
    rescued_only_entity_units = sorted(
        rescued_entity_set.difference(active_entity_set)
    )
    context_entity_novelty = (
        float(len(rescued_only_entity_units) / len(rescued_entity_set))
        if rescued_entity_set
        else None
    )

    configured_titles = sorted({
        _gate_normalize_text(title)
        for title in (
            cfg.get("source_title_units")
            or cfg.get("audit_context", {}).get("source_title_units")
            or []
        )
        if _gate_normalize_text(title)
    })
    title_or_heading_available = bool(
        cfg.get(
            "title_or_heading_available",
            cfg.get("audit_context", {}).get(
                "title_or_heading_available", bool(configured_titles)
            ),
        )
    )

    def titles_in_text(text: str) -> List[str]:
        normalized = _gate_normalize_text(text)
        return sorted(title for title in configured_titles if title in normalized)

    rescued_title_like_units = (
        titles_in_text(rescued_text) if title_or_heading_available else []
    )
    active_title_like_units = (
        titles_in_text(active_context_text) if title_or_heading_available else []
    )
    removed_title_like_units = (
        titles_in_text(removed_text) if title_or_heading_available else []
    )
    rescued_only_title_units = sorted(
        set(rescued_title_like_units).difference(active_title_like_units)
    )
    title_or_heading_novelty = (
        float(
            len(rescued_only_title_units)
            / max(1, len(rescued_title_like_units))
        )
        if title_or_heading_available
        else None
    )

    sentence_novelty_threshold = float(
        cfg.get("gate_sentence_jaccard_novel_max", 0.35)
    )
    rescued_sentences = _gate_sentence_token_sets(rescued_block_texts)
    active_sentences = _gate_sentence_token_sets(active_context_block_texts)
    rescued_sentence_max_overlap = [
        max(
            (_gate_token_jaccard(sentence, active) for active in active_sentences),
            default=0.0,
        )
        for sentence in rescued_sentences
    ]
    rescued_novel_sentence_ratio = (
        float(
            sum(
                overlap < sentence_novelty_threshold
                for overlap in rescued_sentence_max_overlap
            )
            / len(rescued_sentence_max_overlap)
        )
        if rescued_sentence_max_overlap
        else None
    )

    query_signal_units = sorted(set(
        units["query_terms"]
        + units["entity_units"]
        + units["date_number_units"]
        + question_entity_units
    ))
    bridge_entities: Set[str] = set()
    for block_text in rescued_block_texts:
        normalized_block = _gate_normalize_text(block_text)
        has_query_signal = any(
            unit in normalized_block for unit in query_signal_units
        )
        if not has_query_signal:
            continue
        bridge_entities.update(
            set(_gate_entity_like_units(block_text))
            .difference(question_entity_set)
            .difference(active_entity_set)
        )
    bridge_entity_count = len(bridge_entities)
    bridge_entity_novelty_ratio = (
        float(bridge_entity_count / len(rescued_entity_set))
        if rescued_entity_set
        else None
    )

    active_block_token_sets = [
        tokens
        for tokens in (
            _gate_content_tokens(text) for text in active_context_block_texts
        )
        if tokens
    ]
    rescued_active_semantic_novelty = []
    for block_text in rescued_block_texts:
        rescued_tokens = _gate_content_tokens(block_text)
        if not rescued_tokens:
            continue
        max_overlap = max(
            (
                _gate_token_jaccard(rescued_tokens, active_tokens)
                for active_tokens in active_block_token_sets
            ),
            default=0.0,
        )
        rescued_active_semantic_novelty.append(float(1.0 - max_overlap))
    rescued_active_semantic_novelty_mean = (
        float(
            sum(rescued_active_semantic_novelty)
            / len(rescued_active_semantic_novelty)
        )
        if rescued_active_semantic_novelty
        else None
    )
    rescued_active_semantic_novelty_max = (
        max(rescued_active_semantic_novelty)
        if rescued_active_semantic_novelty
        else None
    )
    rescued_active_semantic_novelty_min = (
        min(rescued_active_semantic_novelty)
        if rescued_active_semantic_novelty
        else None
    )

    selected_scores = [
        float(rescue_scores[int(block_id)].detach().float().cpu())
        for block_id in rescue_ids
        if 0 <= int(block_id) < rescue_scores.numel()
    ]
    unselected_ids = [
        int(block_id) for block_id in inactive_ids if int(block_id) not in set(rescue_ids)
    ]
    unselected_scores = [
        float(rescue_scores[int(block_id)].detach().float().cpu())
        for block_id in unselected_ids
        if 0 <= int(block_id) < rescue_scores.numel()
    ]
    sorted_candidate_scores = sorted(
        [
            float(rescue_scores[int(block_id)].detach().float().cpu())
            for block_id in inactive_ids
            if 0 <= int(block_id) < rescue_scores.numel()
        ],
        reverse=True,
    )
    rescue_boundary_selected = min(selected_scores) if selected_scores else None
    rescue_boundary_unselected = max(unselected_scores) if unselected_scores else None
    rescue_boundary_margin = (
        float(rescue_boundary_selected - rescue_boundary_unselected)
        if rescue_boundary_selected is not None
        and rescue_boundary_unselected is not None
        else None
    )
    candidate_pool_exhausted = bool(rescue_ids and not unselected_ids)

    removed_drop_scores = [
        float(drop_utility[int(block_id)].detach().float().cpu())
        for block_id in dropped_ids
        if 0 <= int(block_id) < drop_utility.numel()
    ]
    removed_base_scores = [
        float(active_base_scores[int(block_id)].detach().float().cpu())
        for block_id in dropped_ids
        if 0 <= int(block_id) < active_base_scores.numel()
    ]
    removed_redundancy = [
        float(redundancy_scores[int(block_id)].detach().float().cpu())
        for block_id in dropped_ids
        if redundancy_scores is not None
        and 0 <= int(block_id) < redundancy_scores.numel()
    ]
    active_redundancy = [
        float(redundancy_scores[int(block_id)].detach().float().cpu())
        for block_id in active_ids
        if redundancy_scores is not None
        and 0 <= int(block_id) < redundancy_scores.numel()
    ]
    proximity_risk = (
        float(
            sum(
                any(abs(int(block_id) - int(protected_id)) <= 1 for protected_id in protected_ids)
                for block_id in dropped_ids
            )
            / len(dropped_ids)
        )
        if dropped_ids
        else None
    )
    overlap_parts = [
        value
        for value in [
            query_overlap_removed,
            entity_overlap_removed,
            date_number_overlap_removed,
        ]
        if value is not None
    ]
    base_importance = (
        float(sum(removed_base_scores) / len(removed_base_scores))
        if removed_base_scores
        else None
    )
    deletion_risk = None
    if dropped_ids and overlap_parts and base_importance is not None and proximity_risk is not None:
        deletion_risk = float(
            0.25 * (query_overlap_removed or 0.0)
            + 0.20 * (entity_overlap_removed or 0.0)
            + 0.15 * (date_number_overlap_removed or 0.0)
            + 0.25 * max(0.0, min(1.0, base_importance))
            + 0.15 * proximity_risk
        )

    removed_date_number_units = _gate_date_number_units(removed_text)
    answer_type_proxy = _gate_answer_type_proxy(query_text)
    high_information_removed_entities = [
        entity
        for entity in removed_entity_units
        if (
            entity not in set(removed_date_number_units)
            and (
                len(entity.split()) >= 2
                or len(entity.replace(" ", "")) >= 7
            )
        )
    ]
    entity_information_risk = (
        min(
            1.0,
            len(high_information_removed_entities)
            / max(1.0, 2.0 * len(removed_block_texts)),
        )
        if removed_block_texts
        else None
    )
    multiple_date_number_risk = (
        min(1.0, len(removed_date_number_units) / 2.0)
        if removed_block_texts
        else None
    )
    answer_type_match_risk = 0.0
    if answer_type_proxy in {"when", "how_many"}:
        answer_type_match_risk = float(bool(removed_date_number_units))
    elif answer_type_proxy == "who":
        answer_type_match_risk = float(bool(high_information_removed_entities))
    elif answer_type_proxy == "where":
        answer_type_match_risk = float(
            bool(high_information_removed_entities or removed_title_like_units)
        )
    title_removal_risk = (
        float(bool(removed_title_like_units))
        if title_or_heading_available
        else None
    )
    deletion_components = [
        (0.45, deletion_risk),
        (0.15, entity_information_risk),
        (0.10, multiple_date_number_risk),
        (0.15, answer_type_match_risk),
        (0.15, title_removal_risk),
    ]
    available_deletion_components = [
        (weight, value)
        for weight, value in deletion_components
        if value is not None
    ]
    deletion_evidence_risk_v3 = (
        float(
            sum(weight * float(value) for weight, value in available_deletion_components)
            / sum(weight for weight, _ in available_deletion_components)
        )
        if available_deletion_components
        else None
    )

    coverage_gap_v2 = rescue_coverage_gap
    coverage_v3_components = [
        context_entity_novelty,
        rescued_novel_sentence_ratio,
        bridge_entity_novelty_ratio,
        rescued_active_semantic_novelty_mean,
    ]
    coverage_gap_v3 = (
        max(value for value in coverage_v3_components if value is not None)
        if any(value is not None for value in coverage_v3_components)
        else None
    )

    # selector_native_v3_1: compare rescued evidence against what would be
    # removed and calibrate absolute lexical novelty within each sample.
    retained_active_text = "\n".join(retained_active_block_texts)
    retained_active_entity_set = set(
        _gate_entity_like_units(retained_active_text)
    )
    rescued_confident_entity_units = _gate_confident_entity_units(
        rescued_block_texts, configured_titles
    )
    removed_confident_entity_units = _gate_confident_entity_units(
        removed_block_texts, configured_titles
    )
    active_confident_entity_units = _gate_confident_entity_units(
        active_context_block_texts, configured_titles
    )
    retained_active_confident_entity_units = _gate_confident_entity_units(
        retained_active_block_texts, configured_titles
    )
    rescued_confident_set = set(rescued_confident_entity_units)
    removed_confident_set = set(removed_confident_entity_units)
    rescued_confident_only = sorted(
        rescued_confident_set.difference(active_entity_set)
    )
    removed_confident_only = sorted(
        removed_confident_set.difference(retained_active_entity_set)
    )
    rescued_vs_active_entity_gain = (
        float(len(rescued_confident_only) / len(rescued_confident_set))
        if rescued_confident_set
        else None
    )
    removed_vs_retained_entity_novelty = (
        float(len(removed_confident_only) / len(removed_confident_set))
        if removed_confident_set
        else None
    )
    rescued_vs_removed_entity_gain = (
        _gate_clip01(
            rescued_vs_active_entity_gain
            - removed_vs_retained_entity_novelty
        )
        if rescued_vs_active_entity_gain is not None
        and removed_vs_retained_entity_novelty is not None
        else None
    )

    rescued_sentence_novelty_scores = [
        float(1.0 - overlap) for overlap in rescued_sentence_max_overlap
    ]
    removed_sentences = _gate_sentence_token_sets(removed_block_texts)
    retained_active_sentences = _gate_sentence_token_sets(
        retained_active_block_texts
    )
    removed_sentence_novelty_scores = [
        float(
            1.0
            - max(
                (
                    _gate_token_jaccard(sentence, active)
                    for active in retained_active_sentences
                ),
                default=0.0,
            )
        )
        for sentence in removed_sentences
    ]
    rescued_sentence_novelty_mean = (
        float(
            sum(rescued_sentence_novelty_scores)
            / len(rescued_sentence_novelty_scores)
        )
        if rescued_sentence_novelty_scores
        else None
    )
    removed_sentence_novelty_mean = (
        float(
            sum(removed_sentence_novelty_scores)
            / len(removed_sentence_novelty_scores)
        )
        if removed_sentence_novelty_scores
        else None
    )
    sentence_novelty_gain = (
        _gate_clip01(
            rescued_sentence_novelty_mean - removed_sentence_novelty_mean
        )
        if rescued_sentence_novelty_mean is not None
        and removed_sentence_novelty_mean is not None
        else None
    )

    candidate_semantic_novelty: Dict[int, float] = {}
    for block_id in inactive_ids:
        idx = int(block_id)
        if idx < 0 or idx >= min(len(block_texts), context_limit):
            continue
        tokens = _gate_content_tokens(block_texts[idx])
        if not tokens:
            continue
        max_overlap = max(
            (
                _gate_token_jaccard(tokens, active_tokens)
                for active_tokens in active_block_token_sets
            ),
            default=0.0,
        )
        candidate_semantic_novelty[idx] = float(1.0 - max_overlap)
    candidate_semantic_pool = list(candidate_semantic_novelty.values())
    selected_semantic_percentiles = [
        percentile
        for block_id in rescue_ids
        if int(block_id) in candidate_semantic_novelty
        for percentile in [
            _gate_rank_percentile(
                candidate_semantic_novelty[int(block_id)],
                candidate_semantic_pool,
            )
        ]
        if percentile is not None
    ]
    semantic_novelty_rank_mean = (
        float(
            sum(selected_semantic_percentiles)
            / len(selected_semantic_percentiles)
        )
        if selected_semantic_percentiles
        else None
    )
    semantic_novelty_rank_p50 = _gate_percentile(
        selected_semantic_percentiles, 0.50
    )
    semantic_novelty_rank_min = (
        min(selected_semantic_percentiles)
        if selected_semantic_percentiles
        else None
    )
    semantic_novelty_rank = semantic_novelty_rank_mean

    query_anchor_tokens = _gate_content_tokens(query_text)
    title_anchor_tokens: Set[str] = set()
    for title in configured_titles:
        title_anchor_tokens.update(_gate_content_tokens(title))

    def rare_active_tokens(reference_texts: List[str]) -> Set[str]:
        document_frequency: Counter = Counter()
        for text in reference_texts:
            document_frequency.update(_gate_content_tokens(text))
        limit = max(1, int(math.ceil(0.20 * max(1, len(reference_texts)))))
        return {
            token for token, count in document_frequency.items()
            if count <= limit
        }

    active_rare_tokens = rare_active_tokens(active_context_block_texts)
    retained_active_rare_tokens = rare_active_tokens(
        retained_active_block_texts
    )

    def bridge_units_for_blocks(
        texts: List[str],
        reference_entity_set: Set[str],
        reference_rare_tokens: Set[str],
    ) -> Set[str]:
        bridge_units: Set[str] = set()
        for text in texts:
            block_tokens = _gate_content_tokens(text)
            has_anchor = bool(
                block_tokens.intersection(query_anchor_tokens)
                or block_tokens.intersection(title_anchor_tokens)
                or block_tokens.intersection(reference_rare_tokens)
            )
            if not has_anchor:
                continue
            bridge_units.update(
                set(_gate_confident_entity_units([text], configured_titles))
                .difference(question_entity_set)
                .difference(reference_entity_set)
            )
        return bridge_units

    rescued_bridge_entity_units_v3_1 = bridge_units_for_blocks(
        rescued_block_texts,
        active_entity_set,
        active_rare_tokens,
    )
    removed_bridge_entity_units_v3_1 = bridge_units_for_blocks(
        removed_block_texts,
        retained_active_entity_set,
        retained_active_rare_tokens,
    )
    rescued_bridge_ratio_v3_1 = (
        float(
            len(rescued_bridge_entity_units_v3_1)
            / len(rescued_confident_set)
        )
        if rescued_confident_set
        else None
    )
    removed_bridge_ratio_v3_1 = (
        float(
            len(removed_bridge_entity_units_v3_1)
            / len(removed_confident_set)
        )
        if removed_confident_set
        else None
    )
    bridge_entity_gain = (
        _gate_clip01(
            rescued_bridge_ratio_v3_1 - removed_bridge_ratio_v3_1
        )
        if rescued_bridge_ratio_v3_1 is not None
        and removed_bridge_ratio_v3_1 is not None
        else None
    )

    rescued_answer_type_support = _gate_answer_type_support(
        rescued_block_texts, answer_type_proxy, configured_titles
    )
    removed_answer_type_support = _gate_answer_type_support(
        removed_block_texts, answer_type_proxy, configured_titles
    )
    answer_type_support_gain = (
        _gate_clip01(
            rescued_answer_type_support - removed_answer_type_support
        )
        if rescued_answer_type_support is not None
        and removed_answer_type_support is not None
        else None
    )

    coverage_component_weights = {
        "rescued_vs_removed_entity_gain": (
            0.25, rescued_vs_removed_entity_gain
        ),
        "sentence_novelty_gain": (0.20, sentence_novelty_gain),
        "semantic_novelty_rank": (0.20, semantic_novelty_rank),
        "bridge_entity_gain": (0.20, bridge_entity_gain),
        "answer_type_support_gain": (0.15, answer_type_support_gain),
    }
    coverage_components_available = [
        name
        for name, (_, value) in coverage_component_weights.items()
        if value is not None
    ]
    coverage_components_missing = [
        name
        for name, (_, value) in coverage_component_weights.items()
        if value is None
    ]
    coverage_gap_v3_1 = (
        float(
            sum(
                weight * float(value)
                for weight, value in (
                    coverage_component_weights[name]
                    for name in coverage_components_available
                )
            )
            / sum(
                coverage_component_weights[name][0]
                for name in coverage_components_available
            )
        )
        if len(coverage_components_available) >= 3
        else None
    )

    candidate_score_values = [
        float(rescue_scores[int(block_id)].detach().float().cpu())
        for block_id in inactive_ids
        if (
            0 <= int(block_id) < rescue_scores.numel()
            and math.isfinite(
                float(rescue_scores[int(block_id)].detach().float().cpu())
            )
        )
    ]
    candidate_pool_size = len(inactive_ids)
    candidate_pool_effective_size = len(candidate_score_values)
    candidate_score_mean = (
        sum(candidate_score_values) / len(candidate_score_values)
        if candidate_score_values
        else None
    )
    candidate_score_std = (
        math.sqrt(
            sum(
                (value - float(candidate_score_mean)) ** 2
                for value in candidate_score_values
            )
            / len(candidate_score_values)
        )
        if candidate_score_values and candidate_score_mean is not None
        else None
    )
    score_p25 = _gate_percentile(candidate_score_values, 0.25)
    score_p75 = _gate_percentile(candidate_score_values, 0.75)
    candidate_score_iqr = (
        float(score_p75 - score_p25)
        if score_p25 is not None and score_p75 is not None
        else None
    )
    candidate_score_entropy_proxy: Optional[float] = None
    if len(candidate_score_values) == 1:
        candidate_score_entropy_proxy = 0.0
    elif len(candidate_score_values) > 1:
        score_max = max(candidate_score_values)
        exponentials = [
            math.exp(max(-80.0, min(80.0, value - score_max)))
            for value in candidate_score_values
        ]
        total_exp = sum(exponentials)
        probabilities = [value / total_exp for value in exponentials]
        entropy = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in probabilities
        )
        candidate_score_entropy_proxy = float(
            entropy / math.log(len(probabilities))
        )
    candidate_score_flat = bool(
        candidate_pool_effective_size <= 1
        or (
            candidate_score_std is not None
            and candidate_score_std <= 1e-4
        )
    )

    selected_score_percentiles = [
        percentile
        for score in selected_scores
        for percentile in [
            _gate_rank_percentile(score, candidate_score_values)
        ]
        if percentile is not None
    ]
    unselected_score_percentiles = [
        percentile
        for score in unselected_scores
        for percentile in [
            _gate_rank_percentile(score, candidate_score_values)
        ]
        if percentile is not None
    ]
    rescue_selected_min_percentile = (
        min(selected_score_percentiles)
        if selected_score_percentiles
        else None
    )
    rescue_selected_mean_percentile = (
        float(
            sum(selected_score_percentiles)
            / len(selected_score_percentiles)
        )
        if selected_score_percentiles
        else None
    )
    rescue_unselected_max_percentile = (
        max(unselected_score_percentiles)
        if unselected_score_percentiles
        else None
    )
    rescue_boundary_margin_rank = (
        float(
            rescue_selected_min_percentile
            - rescue_unselected_max_percentile
        )
        if rescue_selected_min_percentile is not None
        and rescue_unselected_max_percentile is not None
        else None
    )

    deletion_query_overlap_risk = (
        _gate_clip01(query_overlap_removed)
        if query_overlap_removed is not None
        else None
    )
    deletion_entity_risk = (
        _gate_clip01(
            len(removed_confident_entity_units)
            / max(1.0, 2.0 * len(removed_block_texts))
        )
        if removed_block_texts
        else None
    )
    deletion_date_number_risk = (
        _gate_clip01(
            len(removed_date_number_units)
            / max(1.0, 2.0 * len(removed_block_texts))
        )
        if removed_block_texts
        else None
    )
    deletion_title_risk = (
        float(bool(removed_title_like_units))
        if title_or_heading_available
        else None
    )
    deletion_answer_type_risk = removed_answer_type_support
    deletion_base_importance_risk = (
        _gate_clip01(base_importance)
        if base_importance is not None
        else None
    )
    deletion_protected_proximity_risk = proximity_risk
    deletion_composite_values = [
        (0.5, deletion_query_overlap_risk),
        (0.3, deletion_entity_risk),
        (0.2, deletion_base_importance_risk),
    ]
    deletion_composite_available = [
        (weight, value)
        for weight, value in deletion_composite_values
        if value is not None
    ]
    deletion_composite = (
        sum(
            weight * float(value)
            for weight, value in deletion_composite_available
        )
        / sum(weight for weight, _ in deletion_composite_available)
        if deletion_composite_available
        else None
    )
    deletion_v3_1_candidates = [
        value
        for value in [
            deletion_answer_type_risk,
            deletion_composite,
            (
                deletion_date_number_risk
                if answer_type_proxy in {"when", "how_many"}
                else 0.0
            ),
            deletion_title_risk,
        ]
        if value is not None
    ]
    deletion_risk_v3_raw = deletion_evidence_risk_v3
    deletion_risk_v3_1 = (
        _gate_clip01(max(deletion_v3_1_candidates))
        if deletion_v3_1_candidates
        else None
    )

    removed_block_risk_sources: List[Dict[str, Any]] = []
    for block_id in dropped_ids:
        idx = int(block_id)
        text = (
            str(block_texts[idx])
            if 0 <= idx < min(len(block_texts), context_limit)
            else ""
        )
        block_entities = _gate_confident_entity_units(
            [text], configured_titles
        )
        block_dates = _gate_date_number_units(text)
        block_titles = titles_in_text(text) if title_or_heading_available else []
        block_query_overlap = _gate_overlap(units["query_terms"], text)
        block_answer_support = _gate_answer_type_support(
            [text], answer_type_proxy, configured_titles
        )
        block_base_importance = (
            _gate_clip01(
                float(active_base_scores[idx].detach().float().cpu())
            )
            if 0 <= idx < active_base_scores.numel()
            else None
        )
        block_protected_proximity = bool(
            any(abs(idx - int(protected_id)) <= 1 for protected_id in protected_ids)
        )
        risk_sources = []
        if block_query_overlap:
            risk_sources.append("query_overlap")
        if block_entities:
            risk_sources.append("entity")
        if block_dates:
            risk_sources.append("date_number")
        if block_titles:
            risk_sources.append("title")
        if block_answer_support:
            risk_sources.append("answer_type_support")
        if block_base_importance:
            risk_sources.append("base_importance")
        if block_protected_proximity:
            risk_sources.append("protected_proximity")
        removed_block_risk_sources.append({
            "block_id": idx,
            "risk_sources": risk_sources,
            "query_overlap": block_query_overlap,
            "entity_units": block_entities,
            "date_number_units": block_dates,
            "title_units": block_titles,
            "answer_type_support": block_answer_support,
            "base_importance": block_base_importance,
            "protected_proximity": block_protected_proximity,
        })

    if gate_feature_version == "selector_native_v3_1":
        required = {
            "coverage_gap_v3_1": coverage_gap_v3_1,
            "deletion_risk_v3_1": deletion_risk_v3_1,
        }
        if not candidate_pool_exhausted:
            required["rescue_boundary_margin_rank"] = (
                rescue_boundary_margin_rank
            )
    elif gate_feature_version == "selector_native_v3":
        required = {
            "coverage_gap_v3": coverage_gap_v3,
            "deletion_evidence_risk_v3": deletion_evidence_risk_v3,
        }
        if not candidate_pool_exhausted:
            required["rescue_boundary_margin"] = rescue_boundary_margin
    else:
        required = {
            "rescue_boundary_margin": rescue_boundary_margin,
            "rescue_coverage_gap": rescue_coverage_gap,
            "deletion_risk": deletion_risk,
            "query_overlap_rescued": query_overlap_rescued,
            "query_overlap_removed": query_overlap_removed,
        }
    missing_fields = sorted(key for key, value in required.items() if value is None)

    all_drop_candidate_rows: List[Dict[str, Any]] = []
    for block_id in safe_drop_ids:
        idx = int(block_id)
        all_drop_candidate_rows.append({
            "block_id": idx,
            "drop_utility": (
                float(drop_utility[idx].detach().float().cpu())
                if 0 <= idx < drop_utility.numel()
                else None
            ),
            "base_score": (
                float(active_base_scores[idx].detach().float().cpu())
                if 0 <= idx < active_base_scores.numel()
                else None
            ),
            "redundancy_score": (
                float(redundancy_scores[idx].detach().float().cpu())
                if redundancy_scores is not None and 0 <= idx < redundancy_scores.numel()
                else None
            ),
        })
    removed_rows = [
        row for row in all_drop_candidate_rows if int(row["block_id"]) in set(dropped_ids)
    ]
    protected_rows = [
        {
            "block_id": int(block_id),
            "protected_reasons": protected_reasons.get(int(block_id), ["unknown"]),
        }
        for block_id in sorted(protected_ids)
    ]
    return {
        "pre_gate_active_block_ids": [int(block_id) for block_id in active_ids],
        "all_pruned_candidates": _score_rows_for_ids(
            rescue_scores, inactive_ids, "rescue_score"
        ),
        "selected_rescued_candidates": _score_rows_for_ids(
            rescue_scores, rescue_ids, "rescue_score"
        ),
        "unselected_pruned_candidates": _score_rows_for_ids(
            rescue_scores, unselected_ids, "rescue_score"
        ),
        "all_active_drop_candidates": all_drop_candidate_rows,
        "selected_removed_candidates": removed_rows,
        "protected_blocks_with_reasons": protected_rows,
        "protected_reasons": {
            str(block_id): reasons
            for block_id, reasons in sorted(protected_reasons.items())
        },
        "rescue_score_top1": sorted_candidate_scores[0] if sorted_candidate_scores else None,
        "rescue_score_top2": (
            sorted_candidate_scores[1] if len(sorted_candidate_scores) > 1 else None
        ),
        "rescue_score_mean": (
            float(sum(selected_scores) / len(selected_scores)) if selected_scores else None
        ),
        "rescue_score_min_selected": rescue_boundary_selected,
        "rescue_score_boundary_selected": rescue_boundary_selected,
        "rescue_score_boundary_unselected": rescue_boundary_unselected,
        "rescue_boundary_margin": rescue_boundary_margin,
        "candidate_pool_exhausted": bool(candidate_pool_exhausted),
        "removed_score_min": min(removed_drop_scores) if removed_drop_scores else None,
        "removed_score_max": max(removed_drop_scores) if removed_drop_scores else None,
        "removed_score_mean": (
            float(sum(removed_drop_scores) / len(removed_drop_scores))
            if removed_drop_scores
            else None
        ),
        "removed_drop_utility_mean": (
            float(sum(removed_drop_scores) / len(removed_drop_scores))
            if removed_drop_scores
            else None
        ),
        "removed_drop_utility_max": (
            max(removed_drop_scores) if removed_drop_scores else None
        ),
        "removed_base_score_mean": base_importance,
        "removed_redundancy_mean": (
            float(sum(removed_redundancy) / len(removed_redundancy))
            if removed_redundancy
            else None
        ),
        "rescued_vs_removed_score_gap": (
            float(sum(selected_scores) / len(selected_scores))
            - float(sum(removed_drop_scores) / len(removed_drop_scores))
            if selected_scores and removed_drop_scores
            else None
        ),
        "active_redundancy_mean": (
            float(sum(active_redundancy) / len(active_redundancy))
            if active_redundancy
            else None
        ),
        "active_redundancy_max": max(active_redundancy) if active_redundancy else None,
        "deletion_risk": deletion_risk,
        "rescue_coverage_gap": rescue_coverage_gap,
        "coverage_gap_v2": coverage_gap_v2,
        "coverage_gap_v3": coverage_gap_v3,
        "context_entity_novelty": context_entity_novelty,
        "title_or_heading_novelty": title_or_heading_novelty,
        "title_or_heading_available": bool(title_or_heading_available),
        "rescued_novel_sentence_ratio": rescued_novel_sentence_ratio,
        "sentence_novelty_jaccard_threshold": sentence_novelty_threshold,
        "bridge_entity_novelty_ratio": bridge_entity_novelty_ratio,
        "bridge_entity_count": int(bridge_entity_count),
        "rescued_active_semantic_novelty_mean": (
            rescued_active_semantic_novelty_mean
        ),
        "rescued_active_semantic_novelty_max": (
            rescued_active_semantic_novelty_max
        ),
        "rescued_active_semantic_novelty_min": (
            rescued_active_semantic_novelty_min
        ),
        "deletion_evidence_risk_v3": deletion_evidence_risk_v3,
        "coverage_gap_v3_raw": coverage_gap_v3,
        "coverage_gap_v3_1": coverage_gap_v3_1,
        "rescued_vs_removed_entity_gain": rescued_vs_removed_entity_gain,
        "rescued_vs_active_entity_gain": rescued_vs_active_entity_gain,
        "removed_vs_retained_entity_novelty": (
            removed_vs_retained_entity_novelty
        ),
        "sentence_novelty_gain": sentence_novelty_gain,
        "rescued_sentence_novelty_mean": rescued_sentence_novelty_mean,
        "removed_sentence_novelty_mean": removed_sentence_novelty_mean,
        "semantic_novelty_rank": semantic_novelty_rank,
        "semantic_novelty_rank_mean": semantic_novelty_rank_mean,
        "semantic_novelty_rank_p50": semantic_novelty_rank_p50,
        "semantic_novelty_rank_min": semantic_novelty_rank_min,
        "bridge_entity_gain": bridge_entity_gain,
        "answer_type_support_gain": answer_type_support_gain,
        "rescued_answer_type_support": rescued_answer_type_support,
        "removed_answer_type_support": removed_answer_type_support,
        "deletion_query_overlap_risk": deletion_query_overlap_risk,
        "deletion_entity_risk": deletion_entity_risk,
        "deletion_date_number_risk": deletion_date_number_risk,
        "deletion_title_risk": deletion_title_risk,
        "deletion_answer_type_risk": deletion_answer_type_risk,
        "deletion_base_importance_risk": deletion_base_importance_risk,
        "deletion_protected_proximity_risk": (
            deletion_protected_proximity_risk
        ),
        "deletion_risk_v3_raw": deletion_risk_v3_raw,
        "deletion_risk_v3_1": deletion_risk_v3_1,
        "removed_block_risk_sources": removed_block_risk_sources,
        "rescue_selected_min_percentile": rescue_selected_min_percentile,
        "rescue_selected_mean_percentile": (
            rescue_selected_mean_percentile
        ),
        "rescue_unselected_max_percentile": (
            rescue_unselected_max_percentile
        ),
        "rescue_boundary_margin_rank": rescue_boundary_margin_rank,
        "candidate_pool_size": int(candidate_pool_size),
        "candidate_pool_effective_size": int(
            candidate_pool_effective_size
        ),
        "candidate_score_std": candidate_score_std,
        "candidate_score_iqr": candidate_score_iqr,
        "candidate_score_entropy_proxy": candidate_score_entropy_proxy,
        "candidate_score_flat": bool(candidate_score_flat),
        "candidate_semantic_novelty": [
            {
                "block_id": int(block_id),
                "semantic_novelty": float(value),
                "rank_percentile": _gate_rank_percentile(
                    value, candidate_semantic_pool
                ),
            }
            for block_id, value in sorted(
                candidate_semantic_novelty.items()
            )
        ],
        "coverage_components_available": coverage_components_available,
        "coverage_components_missing": coverage_components_missing,
        "rescued_confident_entity_units": (
            rescued_confident_entity_units
        ),
        "active_confident_entity_units": active_confident_entity_units,
        "retained_active_confident_entity_units": (
            retained_active_confident_entity_units
        ),
        "removed_confident_entity_units": (
            removed_confident_entity_units
        ),
        "rescued_confident_only_entity_units": rescued_confident_only,
        "removed_confident_only_entity_units": removed_confident_only,
        "rescued_bridge_entity_units_v3_1": sorted(
            rescued_bridge_entity_units_v3_1
        ),
        "removed_bridge_entity_units_v3_1": sorted(
            removed_bridge_entity_units_v3_1
        ),
        "question_entity_units": question_entity_units,
        "rescued_entity_units": rescued_entity_units,
        "active_entity_units": active_entity_units,
        "rescued_only_entity_units": rescued_only_entity_units,
        "rescued_title_like_units": rescued_title_like_units,
        "rescued_only_title_units": rescued_only_title_units,
        "removed_entity_units": removed_entity_units,
        "removed_title_like_units": removed_title_like_units,
        "removed_date_number_units": removed_date_number_units,
        "answer_type_proxy": answer_type_proxy,
        "query_overlap_rescued": query_overlap_rescued,
        "query_overlap_removed": query_overlap_removed,
        "entity_overlap_rescued": entity_overlap_rescued,
        "entity_overlap_removed": entity_overlap_removed,
        "date_number_overlap_rescued": date_number_overlap_rescued,
        "date_number_overlap_removed": date_number_overlap_removed,
        "gate_feature_missing": bool(missing_fields),
        "gate_feature_missing_fields": missing_fields,
        "gate_feature_version": gate_feature_version,
        "gate_evidence_units": units,
        "gate_context_block_limit": int(context_limit),
        "removed_protected_proximity_rate": proximity_risk,
    }


def _maybe_stochastic_scores(
    cfg: Dict[str, Any],
    base_scores: torch.Tensor,
    inactive_ids: List[int],
    meta: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    mode = str(cfg.get("stochastic_mode", "gumbel") or "gumbel").lower()
    layer_idx = int(cfg.get("layer_idx", 0) or 0)
    seed = int(cfg.get("seed", 0) or 0)
    if mode == "pool":
        meta["pool_sampling"] = True
        meta["stochastic_noise_stats"] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return base_scores, meta
    noisy, noise_stats = apply_gumbel_noise(
        base_scores,
        inactive_ids,
        tau=float(cfg.get("gumbel_tau", 0.05)),
        seed=seed,
        layer_idx=layer_idx,
    )
    meta["stochastic_noise_stats"] = noise_stats
    meta["pool_sampling"] = False
    return noisy, meta

def _optional_rescue_score_vector(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    block_scores: torch.Tensor,
    qk_scores: torch.Tensor,
    delta_o_values: Optional[torch.Tensor],
    kernel_scores: torch.Tensor,
    represent_values: Optional[torch.Tensor],
    num_key_value_groups: int,
    block_slices: List[Tuple[int, int]],
    active_ids: List[int],
    inactive_ids: List[int],
    diagnostics: Dict[str, List[str]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    device = block_scores.device
    base_global = _scatter_local_scores(delta_o_values if delta_o_values is not None else block_scores, original_block_ids, device)
    qk_global = _scatter_local_scores(qk_scores, original_block_ids, device)
    reps = _block_value_representations_global(represent_values, num_key_value_groups, block_slices, original_block_ids, device)
    if reps is None and represent_values is not None:
        diagnostics.setdefault("warnings", []).append("Block rep fallback: using pooled V representatives.")
    score_name = str(cfg.get("rescue_score", "delta_o") or "delta_o")
    meta: Dict[str, Any] = {"mmr_selection": False}
    if score_name == "sliminfer_next":
        return _scatter_local_scores(block_scores, original_block_ids, device), reps, meta
    if score_name == "random":
        length = (max(original_block_ids) + 1) if original_block_ids else 0
        seed = int(cfg.get("seed", 0) or 0)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        vals = torch.rand(length, generator=generator, dtype=torch.float32)
        return vals.to(device=device), reps, meta
    if score_name == "delta_o":
        return base_global, reps, meta
    if score_name == "obd":
        return _obd_scores_global(kernel_scores, represent_values, num_key_value_groups, block_slices, original_block_ids, device, diagnostics), reps, meta
    if score_name == "diffusion_gap":
        c_active, g_diff, r_active = diffusion_gap_values(reps, active_ids, inactive_ids, base_global.numel(), diagnostics)
        meta.update({
            "C_active_stats": _score_stats_for_block_ids(c_active, inactive_ids),
            "G_diffusion_stats": _score_stats_for_block_ids(g_diff, inactive_ids),
            "R_active_stats": _score_stats_for_block_ids(r_active, inactive_ids),
        })
        return diffusion_gap_score(base_global, reps, active_ids, inactive_ids, diagnostics), reps, meta
    if score_name == "redundancy":
        c_active, g_diff, r_active = diffusion_gap_values(reps, active_ids, inactive_ids, base_global.numel(), diagnostics)
        meta.update({
            "C_active_stats": _score_stats_for_block_ids(c_active, inactive_ids),
            "G_diffusion_stats": _score_stats_for_block_ids(g_diff, inactive_ids),
            "R_active_stats": _score_stats_for_block_ids(r_active, inactive_ids),
        })
        return redundancy_aware_score(base_global, reps, active_ids, inactive_ids, lambda_red=float(cfg.get("lambda_red", 0.25)), diagnostics=diagnostics), reps, meta
    if score_name == "path":
        return path_aware_score(attn_mass=qk_global, active_ids=active_ids, inactive_ids=inactive_ids, num_blocks=base_global.numel(), diagnostics=diagnostics), reps, meta
    if score_name == "mmr":
        s0 = torch.full_like(base_global, float("-inf"))
        valid_inactive = [idx for idx in inactive_ids if 0 <= idx < base_global.numel()]
        if valid_inactive:
            idx = torch.tensor(valid_inactive, dtype=torch.long, device=device)
            s0[idx] = normalize_scores(base_global[idx], "zscore")
        meta["mmr_selection"] = True
        return s0, reps, meta
    if score_name == "drm":
        s0, drm_meta = drm_s0_score(
            base_global,
            reps,
            active_ids,
            inactive_ids,
            alpha_diff=float(cfg.get("alpha_diff", 0.5)),
            lambda_red=float(cfg.get("lambda_red", 0.25)),
            diagnostics=diagnostics,
        )
        inactive_idx = [idx for idx in inactive_ids if 0 <= idx < s0.numel()]
        meta.update({
            "mmr_selection": True,
            "C_active_stats": _score_stats_for_block_ids(drm_meta["C_active"], inactive_ids),
            "G_diffusion_stats": _score_stats_for_block_ids(drm_meta["G_diffusion"], inactive_ids),
            "R_active_stats": _score_stats_for_block_ids(drm_meta["R_active"], inactive_ids),
            "drm_used_base": bool(float(drm_meta.get("used_base", torch.tensor([0.0]))[0].item()) > 0.5),
            "D_base_stats": _score_stats_for_block_ids(base_global, inactive_ids),
            "_drm_trace_tensors": {
                "D_base": base_global.detach(),
                "QK": qk_global.detach(),
                "C_active": drm_meta["C_active"].detach(),
                "G_diffusion": drm_meta["G_diffusion"].detach(),
                "R_active": drm_meta["R_active"].detach(),
            },
        })
        return s0, reps, meta
    if score_name == "query_drm":
        s0, meta = _query_drm_score_bundle(cfg, base_global, reps, active_ids, inactive_ids, diagnostics)
        return s0, reps, meta
    if score_name == "stochastic_drm":
        s0, drm_meta = drm_s0_score(
            base_global, reps, active_ids, inactive_ids,
            alpha_diff=float(cfg.get("alpha_diff", 0.5)),
            lambda_red=float(cfg.get("lambda_red", 0.25)),
            diagnostics=diagnostics,
        )
        meta = {
            "mmr_selection": True,
            "C_active_stats": _score_stats_for_block_ids(drm_meta["C_active"], inactive_ids),
            "G_diffusion_stats": _score_stats_for_block_ids(drm_meta["G_diffusion"], inactive_ids),
            "R_active_stats": _score_stats_for_block_ids(drm_meta["R_active"], inactive_ids),
            "D_base_stats": _score_stats_for_block_ids(base_global, inactive_ids),
            "drm_used_base": bool(float(drm_meta.get("used_base", torch.tensor([0.0]))[0].item()) > 0.5),
        }
        s0, meta = _maybe_stochastic_scores(cfg, s0, inactive_ids, meta)
        return s0, reps, meta
    if score_name == "query_stochastic_drm":
        s0, meta = _query_drm_score_bundle(cfg, base_global, reps, active_ids, inactive_ids, diagnostics)
        s0, meta = _maybe_stochastic_scores(cfg, s0, inactive_ids, meta)
        return s0, reps, meta
    if score_name == "abc":
        obd = _obd_scores_global(kernel_scores, represent_values, num_key_value_groups, block_slices, original_block_ids, device, diagnostics)
        path = path_aware_score(attn_mass=qk_global, active_ids=active_ids, inactive_ids=inactive_ids, num_blocks=base_global.numel(), diagnostics=diagnostics)
        return combined_abc_score(
            obd_scores=obd,
            block_reps=reps,
            active_ids=active_ids,
            inactive_ids=inactive_ids,
            path_scores=path,
            alpha_diff=float(cfg.get("alpha_diff", 0.5)),
            lambda_red=float(cfg.get("lambda_red", 0.25)),
            beta_path=float(cfg.get("beta_path", 0.25)),
            diagnostics=diagnostics,
        ), reps, meta
    if score_name == "rh_pyramid_srm":
        scores, meta = score_rh_pyramid_srm(cfg, base_global, reps, qk_global, active_ids, inactive_ids, diagnostics)
        meta["mmr_selection"] = True
        return scores, reps, meta
    if score_name == "distilled_blockdrm":
        scores, meta = score_distilled_blockdrm(cfg, base_global, reps, qk_global, active_ids, inactive_ids, diagnostics)
        meta["mmr_selection"] = True
        return scores, reps, meta
    if score_name == "bridge_graph_srm":
        scores, meta = score_bridge_graph_srm(cfg, base_global, reps, qk_global, active_ids, inactive_ids, diagnostics)
        meta["mmr_selection"] = True
        return scores, reps, meta
    if score_name == "deferred_uncertainty_srm":
        scores, meta = score_deferred_uncertainty_srm(cfg, base_global, reps, qk_global, active_ids, inactive_ids, diagnostics)
        meta["mmr_selection"] = True
        return scores, reps, meta
    if score_name == "evidence_density_srm":
        scores, meta = score_evidence_density_srm(
            cfg,
            base_global,
            reps,
            qk_global,
            active_ids,
            inactive_ids,
            diagnostics,
        )
        return scores, reps, meta
    diagnostics.setdefault("warnings", []).append(f"Unknown rescue_score={score_name}; falling back to delta_o/base score.")
    return base_global, reps, meta


def _load_lssr_policy_b_plan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = str(cfg.get("lssr_policy_b_plan_path", "") or "")
    if not path:
        raise RuntimeError("lssr_policy_b_middle_donor_late_receiver requires lssr_policy_b_plan_path")
    cached = _LSSR_POLICY_B_PLAN_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    _LSSR_POLICY_B_PLAN_CACHE[path] = plan
    return plan


def _lssr_sample_key(cfg: Dict[str, Any]) -> str:
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    task = str(context.get("task", ""))
    sample_index = context.get("sample_index", "")
    if task == "" or sample_index == "":
        raise RuntimeError("LSSR plan lookup requires task and sample_index in audit_context")
    return f"{task}:{int(sample_index)}"


def _lssr_layer_group(layer_id: int, sample_plan: Dict[str, Any]) -> str:
    definition = sample_plan.get("layer_group_definition", {})
    if isinstance(definition, dict):
        for group_name, layer_ids in definition.items():
            try:
                if int(layer_id) in {int(x) for x in layer_ids}:
                    return str(group_name)
            except Exception:
                continue
    return str(sample_plan.get("actions_by_layer", {}).get(str(layer_id), {}).get("layer_group", "unknown"))


def _lssr_policy_b_gate_from_features(sample_plan: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    middle_capacity = int(sample_plan.get("middle_donor_capacity") or 0)
    late_need = int(sample_plan.get("late_receiver_need") or 0)
    late_qe = int(sample_plan.get("late_query_or_evidence_candidate_count") or 0)
    risk = sample_plan.get("middle_donor_risk_mean")
    gain = sample_plan.get("late_receiver_gain_max")
    risk_limit = float(thresholds.get("middle_donor_risk_low", 2.0e-6))
    gain_min = float(thresholds.get("late_receiver_gain_high", 4.0e-2))
    ambiguity_cap = int(thresholds.get("late_receiver_ambiguity_cap", 6))
    if middle_capacity <= 0:
        reasons.append("middle_donor_capacity_zero")
    if late_need <= 0:
        reasons.append("late_receiver_need_zero")
    try:
        risk_value = float(risk)
    except (TypeError, ValueError):
        risk_value = float("inf")
    if not math.isfinite(risk_value) or risk_value > risk_limit:
        reasons.append("middle_donor_risk_not_low")
    try:
        gain_value = float(gain)
    except (TypeError, ValueError):
        gain_value = float("-inf")
    if not math.isfinite(gain_value) or gain_value < gain_min:
        reasons.append("late_receiver_gain_not_high")
    if late_qe <= 0:
        reasons.append("late_query_or_evidence_proxy_absent")
    if late_qe > ambiguity_cap:
        reasons.append("late_receiver_signal_too_broad")
    if not bool(sample_plan.get("budget_preserved")):
        reasons.append("budget_not_preserved")
    if int(sample_plan.get("protected_violation") or 0) != 0:
        reasons.append("protected_violation")
    if sample_plan.get("fallback_reason") not in (None, "", "None"):
        reasons.append("fallback_reason_present")
    return not reasons, reasons


def _apply_lssr_policy_b_plan(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    active_ids: List[int],
    must_keep_block_ids: Set[int],
    active_global: torch.Tensor,
) -> Dict[str, Any]:
    plan = _load_lssr_policy_b_plan(cfg)
    samples = plan.get("samples", {})
    sample_key = _lssr_sample_key(cfg)
    if sample_key not in samples:
        raise RuntimeError(f"LSSR Policy B plan missing sample {sample_key}")
    sample_plan = samples[sample_key]
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    expected_sample_id = str(sample_plan.get("sample_id", ""))
    actual_sample_id = str(context.get("sample_id", ""))
    if expected_sample_id and actual_sample_id and expected_sample_id != actual_sample_id:
        raise RuntimeError(
            f"LSSR sample_id mismatch for {sample_key}: expected {expected_sample_id}, got {actual_sample_id}"
        )
    thresholds = plan.get("thresholds", {})
    runtime_gate, runtime_reasons = _lssr_policy_b_gate_from_features(sample_plan, thresholds)
    expected_gate = bool(sample_plan.get("gate_triggered"))
    if runtime_gate != expected_gate:
        raise RuntimeError(
            f"LSSR Policy B gate mismatch for {sample_key}: plan={expected_gate}, runtime={runtime_gate}, "
            f"reasons={runtime_reasons}"
        )

    layer_id = int(cfg.get("layer_idx", -1))
    layer_group = _lssr_layer_group(layer_id, sample_plan)
    layer_action = sample_plan.get("actions_by_layer", {}).get(str(layer_id), {})
    remove_ids = [int(x) for x in layer_action.get("remove_block_ids", [])]
    planned_add_ids = [int(x) for x in layer_action.get("add_block_ids", [])]
    add_ids = list(planned_add_ids)
    late_action_layer_ids = sorted(
        int(layer)
        for layer, action in sample_plan.get("actions_by_layer", {}).items()
        if isinstance(action, dict) and str(action.get("layer_group")) == "late"
    )
    is_last_late_action_layer = bool(late_action_layer_ids) and layer_id == late_action_layer_ids[-1]
    state_key = f"{str(cfg.get('lssr_policy_b_plan_path', ''))}:{sample_key}"
    runtime_state = _LSSR_POLICY_B_RUNTIME_STATE.setdefault(state_key, {"remaining_add_budget": 0})
    if not expected_gate:
        remove_ids = []
        add_ids = []
    elif layer_group == "late":
        remaining_budget = int(runtime_state.get("remaining_add_budget", 0) or 0)
        configured_target = int(layer_action.get("add_target_count", len(planned_add_ids)) or 0)
        target_add_count = min(
            remaining_budget,
            configured_target if configured_target > 0 else remaining_budget,
        )
        ranked_candidates = [int(x) for x in layer_action.get("candidate_add_block_ids", planned_add_ids)]
        candidate_order: List[int] = []
        for block_id in planned_add_ids + ranked_candidates:
            if int(block_id) not in candidate_order:
                candidate_order.append(int(block_id))
        active_set_for_choice = set(int(x) for x in active_ids)
        original_set_for_choice = set(int(x) for x in original_block_ids)
        add_ids = []
        if target_add_count > 0:
            for block_id in candidate_order:
                if block_id in active_set_for_choice:
                    continue
                if block_id not in original_set_for_choice:
                    continue
                add_ids.append(block_id)
                if len(add_ids) >= target_add_count:
                    break
        if len(add_ids) != target_add_count and is_last_late_action_layer:
            raise RuntimeError(
                f"LSSR receiver backfill failed for {sample_key} layer {layer_id}: "
                f"needed {target_add_count}, got {len(add_ids)}"
            )
    if layer_group == "early" and remove_ids:
        raise RuntimeError(f"LSSR attempted early-layer deletion for {sample_key} layer {layer_id}")
    if remove_ids and layer_group != "middle":
        raise RuntimeError(f"LSSR donor layer is not middle for {sample_key} layer {layer_id}: {layer_group}")
    if add_ids and layer_group != "late":
        raise RuntimeError(f"LSSR receiver layer is not late for {sample_key} layer {layer_id}: {layer_group}")

    active_set = set(int(x) for x in active_ids)
    original_set = set(int(x) for x in original_block_ids)
    protected_set = set(int(x) for x in must_keep_block_ids)
    missing_remove = sorted(set(remove_ids) - active_set)
    if missing_remove:
        raise RuntimeError(f"LSSR donor blocks not active for {sample_key} layer {layer_id}: {missing_remove[:16]}")
    protected_remove = sorted(set(remove_ids).intersection(protected_set))
    if protected_remove:
        raise RuntimeError(f"LSSR protected donor violation for {sample_key} layer {layer_id}: {protected_remove[:16]}")
    missing_add = sorted(set(add_ids) - original_set)
    if missing_add:
        raise RuntimeError(f"LSSR receiver blocks not in original block set for {sample_key} layer {layer_id}: {missing_add[:16]}")
    duplicate_add = sorted(set(add_ids).intersection(active_set))
    if duplicate_add:
        raise RuntimeError(f"LSSR receiver backfill selected active blocks for {sample_key} layer {layer_id}: {duplicate_add[:16]}")

    new_ids = (active_set - set(remove_ids)) | set(add_ids)
    if len(new_ids) != len(active_set) - len(set(remove_ids)) + len(set(add_ids)):
        raise RuntimeError(f"LSSR duplicate block accounting mismatch for {sample_key} layer {layer_id}")
    if expected_gate and layer_group == "middle":
        runtime_state["remaining_add_budget"] = int(runtime_state.get("remaining_add_budget", 0) or 0) + len(remove_ids)
    if expected_gate and layer_group == "late":
        runtime_state["remaining_add_budget"] = max(
            0,
            int(runtime_state.get("remaining_add_budget", 0) or 0) - len(add_ids),
        )

    gate_reason = (
        "policy_b_middle_donor_late_receiver_pass"
        if expected_gate
        else "policy_b_reject:" + ",".join(runtime_reasons)
    )
    final_budget_type = "K_cross_layer_lssr_policy_b" if expected_gate else "K_original_gate_reject"
    extra_fields = {
        "lssr_policy": "policy_b_conservative_middle_donor",
        "gate_triggered": bool(expected_gate),
        "gate_reason": gate_reason,
        "gate_reject_reasons": runtime_reasons,
        "donor_layer_group": "middle" if remove_ids else "",
        "receiver_layer_group": "late" if add_ids else "",
        "layer_group": layer_group,
        "middle_donor_block_ids": sample_plan.get("middle_donor_block_ids", []),
        "late_receiver_block_ids": add_ids if add_ids else sample_plan.get("late_receiver_block_ids", []),
        "planned_late_receiver_block_ids": planned_add_ids,
        "middle_donor_risk_mean": sample_plan.get("middle_donor_risk_mean"),
        "late_receiver_gain_max": sample_plan.get("late_receiver_gain_max"),
        "late_query_or_evidence_candidate_count": sample_plan.get("late_query_or_evidence_candidate_count"),
        "budget_preserved": bool(sample_plan.get("budget_preserved")),
        "sample_budget_delta_blocks": int(sample_plan.get("sample_budget_delta_blocks", 0) or 0),
        "lssr_remaining_add_budget_after_layer": int(runtime_state.get("remaining_add_budget", 0) or 0),
        "selection_behavior_unchanged": bool(sample_plan.get("selection_behavior_unchanged", True)),
        "removed_block_ids": remove_ids,
        "rescued_block_ids": add_ids,
        "removed_count": int(len(remove_ids)),
        "rescue_count": int(len(add_ids)),
        "swap_count": int(max(len(remove_ids), len(add_ids))),
        "fallback_used": False,
        "fallback_reason": "",
        "protected_violation": False,
        "protected_block_violation": False,
    }
    _write_rescue_audit_record(
        cfg=cfg,
        original_block_ids=original_block_ids,
        active_before_ids=active_ids,
        final_ids=sorted(new_ids),
        rescued_ids=add_ids,
        dropped_ids=remove_ids,
        protected_ids=sorted(protected_set),
        rescue_scores=active_global,
        target_m=max(len(remove_ids), len(add_ids)),
        duplicate_block_ids=False,
        protected_violation=False,
        warnings=[],
        final_budget_type=final_budget_type,
        score_meta={"jaccard_vs_sliminfer": _jaccard(new_ids, active_set)},
        extra_fields=extra_fields,
    )
    return {
        "final_kept_ids": new_ids,
        "rescue_policy": "lssr_policy_b_middle_donor_late_receiver",
        "rescue_score": str(cfg.get("rescue_score", "delta_o")),
        "final_budget_type": final_budget_type,
        "selected_by_rescue_count": int(len(add_ids)),
        "bridge_quota_blocks_count": int(len(add_ids)),
        "replaced_blocks_count": int(len(remove_ids)),
        "replaced_block_ids_sampled": sorted(add_ids)[:32],
        "replaced_out_block_ids_sampled": sorted(remove_ids)[:32],
        "rescue_policy_warnings": [],
        "jaccard_vs_base": _jaccard(new_ids, active_set),
    }


def _apply_optional_rescue_policy(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    final_kept_ids: Set[int],
    must_keep_block_ids: Set[int],
    block_scores: torch.Tensor,
    qk_scores: torch.Tensor,
    delta_o_values: Optional[torch.Tensor],
    kernel_scores: torch.Tensor,
    represent_values: Optional[torch.Tensor],
    num_key_value_groups: int,
    block_slices: List[Tuple[int, int]],
) -> Dict[str, Any]:
    policy = str(cfg.get("rescue_policy", "none") or "none")
    if policy in {"none", "sliminfer_larger"}:
        return {}
    active_ids = sorted(int(x) for x in final_kept_ids)
    inactive_ids = [int(x) for x in original_block_ids if int(x) not in final_kept_ids]
    diagnostics: Dict[str, List[str]] = {"warnings": []}
    score_cfg = dict(cfg)
    if policy in {"debug_recent_only", "debug_inverted"}:
        active_global = _scatter_local_scores(block_scores, original_block_ids, block_scores.device)
        if policy == "debug_recent_only":
            new_ids = set(int(x) for x in must_keep_block_ids)
            final_budget_type = "debug_recent_only"
        else:
            target_k = len(active_ids)
            ordered_low = _ordered_ids_by_score(block_scores, original_block_ids, descending=False)
            new_ids = set(int(x) for x in must_keep_block_ids)
            for block_id in ordered_low:
                new_ids.add(int(block_id))
                if len(new_ids) >= target_k:
                    break
            final_budget_type = "debug_inverted_K"
        rescued_ids = sorted(new_ids - set(active_ids))
        dropped_ids = sorted(set(active_ids) - new_ids)
        protected_ids = set(int(x) for x in must_keep_block_ids)
        duplicate_block_ids = False
        protected_violation = bool(set(dropped_ids).intersection(protected_ids))
        _write_rescue_audit_record(
            cfg=cfg,
            original_block_ids=original_block_ids,
            active_before_ids=active_ids,
            final_ids=sorted(new_ids),
            rescued_ids=rescued_ids,
            dropped_ids=dropped_ids,
            protected_ids=sorted(protected_ids),
            rescue_scores=active_global,
            target_m=len(rescued_ids),
            duplicate_block_ids=duplicate_block_ids,
            protected_violation=protected_violation,
            warnings=diagnostics.get("warnings", []),
            final_budget_type=final_budget_type,
        )
        return {
            "final_kept_ids": new_ids,
            "rescue_policy": policy,
            "rescue_score": str(score_cfg.get("rescue_score", "delta_o")),
            "final_budget_type": final_budget_type,
            "selected_by_rescue_count": int(len(rescued_ids)),
            "bridge_quota_blocks_count": int(len(rescued_ids)),
            "replaced_blocks_count": int(len(rescued_ids)),
            "replaced_block_ids_sampled": sorted(rescued_ids)[:32],
            "replaced_out_block_ids_sampled": sorted(dropped_ids)[:32],
            "rescue_policy_warnings": diagnostics.get("warnings", []),
            "jaccard_vs_base": _jaccard(new_ids, set(active_ids)),
        }
    rescue_scores, reps, score_meta = _optional_rescue_score_vector(
        cfg=score_cfg,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        qk_scores=qk_scores,
        delta_o_values=delta_o_values,
        kernel_scores=kernel_scores,
        represent_values=represent_values,
        num_key_value_groups=num_key_value_groups,
        block_slices=block_slices,
        active_ids=active_ids,
        inactive_ids=inactive_ids,
        diagnostics=diagnostics,
    )
    active_global = _scatter_local_scores(block_scores, original_block_ids, block_scores.device)
    if policy == "lssr_policy_b_middle_donor_late_receiver":
        return _apply_lssr_policy_b_plan(
            cfg=cfg,
            original_block_ids=original_block_ids,
            active_ids=active_ids,
            must_keep_block_ids=must_keep_block_ids,
            active_global=active_global,
        )
    target_m = int(round(len(active_ids) * float(cfg.get("rescue_ratio", 0.05))))
    if policy in {"budget_neutral_drm_swap", "gated_budget_neutral_drm_swap"}:
        is_gated = policy == "gated_budget_neutral_drm_swap"
        if float(cfg.get("rescue_ratio", 0.05)) > 0.0 and active_ids and inactive_ids:
            target_m = max(1, target_m)
        target_m = min(target_m, len(inactive_ids))
        forced_ids = {
            int(block_id)
            for block_id in cfg.get("forced_keep_block_ids", [])
            if int(block_id) in set(active_ids)
        }
        query_scores, query_meta = query_evidence_score(
            block_texts=cfg.get("block_texts") or cfg.get("audit_context", {}).get("block_texts"),
            query_text=cfg.get("query_text") or cfg.get("audit_context", {}).get("query_text"),
            inactive_ids=active_ids,
            block_reps=reps,
            num_blocks=(max(original_block_ids) + 1) if original_block_ids else 0,
            diagnostics=diagnostics,
        )
        query_ratio = max(0.0, float(cfg.get("query_top_ratio", 0.2) or 0.0))
        query_critical_ids: Set[int] = set()
        positive_query_ids = [
            int(block_id)
            for block_id in active_ids
            if 0 <= int(block_id) < query_scores.numel()
            and float(query_scores[int(block_id)].detach().cpu()) > 0.0
        ]
        if positive_query_ids and query_ratio > 0.0:
            query_keep = min(
                len(positive_query_ids),
                max(1, int(math.ceil(len(active_ids) * query_ratio))),
            )
            query_critical_ids = set(
                sorted(
                    positive_query_ids,
                    key=lambda block_id: float(query_scores[block_id].detach().cpu()),
                    reverse=True,
                )[:query_keep]
            )
        protected_ids = (
            set(int(block_id) for block_id in must_keep_block_ids)
            | forced_ids
            | query_critical_ids
        )
        protected_reasons: Dict[int, List[str]] = {}
        for block_id in must_keep_block_ids:
            protected_reasons.setdefault(int(block_id), []).append(
                "must_keep_sink_or_recent"
            )
        for block_id in forced_ids:
            protected_reasons.setdefault(int(block_id), []).append("forced_keep")
        for block_id in query_critical_ids:
            protected_reasons.setdefault(int(block_id), []).append("query_critical")
        redundancy_scores = _active_redundancy_scores(
            reps,
            active_ids,
            (max(original_block_ids) + 1) if original_block_ids else 0,
            block_scores.device,
        )
        fallback_reason = ""
        if not inactive_ids or target_m <= 0:
            fallback_reason = "no_pruned_rescue_candidates"
        elif redundancy_scores is None:
            fallback_reason = "active_redundancy_unavailable"
        safe_drop_ids = [
            int(block_id) for block_id in active_ids if int(block_id) not in protected_ids
        ]
        if not fallback_reason and len(safe_drop_ids) < target_m:
            fallback_reason = (
                f"insufficient_safe_drop_blocks:{len(safe_drop_ids)}<{target_m}"
            )
        rescue_ids: List[int] = []
        dropped_ids: List[int] = []
        mmr_diag: Dict[str, Any] = {}
        drop_utility = torch.zeros_like(active_global, dtype=torch.float32)
        if not fallback_reason:
            if bool(score_meta.get("mmr_selection", False)):
                rescue_ids, mmr_diag = mmr_select_rescue_ids(
                    inactive_ids=inactive_ids,
                    s0_scores=rescue_scores,
                    block_reps=reps,
                    active_ids=active_ids,
                    m=target_m,
                    eta_rescue=float(cfg.get("eta_rescue", 0.25)),
                    eta_active=float(cfg.get("eta_active", 0.25)),
                    seed=int(cfg.get("seed", 0) or 0),
                    diagnostics=diagnostics,
                )
                score_meta["MMR_diversity_stats"] = mmr_diag.get("MMR_diversity_stats", {})
                score_meta["mmr_vs_plain_top_m_jaccard"] = mmr_diag.get("mmr_vs_plain_top_m_jaccard", "")
            else:
                rescue_ids = sorted(
                    inactive_ids,
                    key=lambda block_id: float(rescue_scores[int(block_id)].detach().cpu()),
                    reverse=True,
                )[:target_m]
            value_norm = _normalized_subset_scores(active_global, active_ids)
            redundancy_norm = _normalized_subset_scores(redundancy_scores, active_ids)
            drop_utility = value_norm - float(cfg.get("lambda_red", 0.25)) * redundancy_norm
            dropped_ids = sorted(
                safe_drop_ids,
                key=lambda block_id: float(drop_utility[int(block_id)].detach().cpu()),
            )[:target_m]
            if len(rescue_ids) != target_m or len(dropped_ids) != target_m:
                fallback_reason = "incomplete_rescue_drop_pairing"
        if fallback_reason:
            diagnostics.setdefault("warnings", []).append(fallback_reason)
            rescue_ids = []
            dropped_ids = []
            new_ids = set(active_ids)
        else:
            new_ids = (set(active_ids) - set(dropped_ids)) | set(rescue_ids)
            if len(new_ids) != len(active_ids):
                fallback_reason = "budget_neutral_swap_size_mismatch"
                diagnostics.setdefault("warnings", []).append(fallback_reason)
                rescue_ids = []
                dropped_ids = []
                new_ids = set(active_ids)
        protected_violation = bool(set(dropped_ids).intersection(protected_ids))
        if protected_violation:
            fallback_reason = "protected_block_drop_attempt"
            diagnostics.setdefault("warnings", []).append(fallback_reason)
            rescue_ids = []
            dropped_ids = []
            new_ids = set(active_ids)
        active_redundancy_mean = (
            _mean_score_for_ids(redundancy_scores, active_ids)
            if redundancy_scores is not None
            else 0.0
        )
        candidate_rescue_ids = list(rescue_ids)
        candidate_dropped_ids = list(dropped_ids)
        candidate_new_ids = set(new_ids)
        gate_fields: Dict[str, Any] = {}
        selected_policy_before_gate = (
            "drm_swap"
            if candidate_rescue_ids
            and candidate_dropped_ids
            and not fallback_reason
            else "original"
        )
        selected_policy_after_gate = selected_policy_before_gate
        would_gate_trigger = selected_policy_before_gate == "drm_swap"
        gate_reason = (
            "candidate_swap_available"
            if would_gate_trigger
            else (fallback_reason or "no_candidate_swap")
        )
        gate_score: Optional[float] = None
        gate_fallback_to_original = False
        residual_proxy_fields = _residual_aware_proxy_fields(
            cfg=cfg,
            original_block_ids=original_block_ids,
            active_ids=active_ids,
            inactive_ids=inactive_ids,
            rescue_ids=candidate_rescue_ids,
            dropped_ids=candidate_dropped_ids,
            protected_ids=protected_ids,
            query_critical_ids=query_critical_ids,
            rescue_scores=rescue_scores,
            active_global=active_global,
            reps=reps,
            redundancy_scores=redundancy_scores,
            represent_values=represent_values,
            num_key_value_groups=num_key_value_groups,
            block_slices=block_slices,
        )
        if is_gated:
            gate_fields = _compute_gated_swap_features(
                cfg=cfg,
                active_ids=active_ids,
                inactive_ids=inactive_ids,
                rescue_ids=candidate_rescue_ids,
                dropped_ids=candidate_dropped_ids,
                safe_drop_ids=safe_drop_ids,
                protected_ids=protected_ids,
                protected_reasons=protected_reasons,
                rescue_scores=rescue_scores,
                active_base_scores=_normalized_subset_scores(
                    active_global, active_ids
                ),
                redundancy_scores=redundancy_scores,
                drop_utility=drop_utility,
            )
            gate_mode = str(cfg.get("gate_mode", "shadow") or "shadow")
            gate_default_policy = str(
                cfg.get("gate_default_policy", "original") or "original"
            )
            gate_policy = str(
                cfg.get("gate_policy", "conservative_v1") or "conservative_v1"
            )
            gate_feature_version = str(
                gate_fields.get(
                    "gate_feature_version",
                    cfg.get("gate_feature_version", "selector_native_v2"),
                )
            )
            is_v3_1 = gate_feature_version == "selector_native_v3_1"
            if is_v3_1:
                margin = gate_fields.get("rescue_boundary_margin_rank")
                coverage = gate_fields.get("coverage_gap_v3_1")
                deletion_risk = gate_fields.get("deletion_risk_v3_1")
            elif gate_feature_version == "selector_native_v3":
                margin = gate_fields.get("rescue_boundary_margin")
                coverage = gate_fields.get("coverage_gap_v3")
                deletion_risk = gate_fields.get("deletion_evidence_risk_v3")
            else:
                margin = gate_fields.get("rescue_boundary_margin")
                coverage = gate_fields.get("rescue_coverage_gap")
                deletion_risk = gate_fields.get("deletion_risk")
            if (
                margin is not None
                and coverage is not None
                and deletion_risk is not None
            ):
                gate_score = float(margin + coverage - deletion_risk)
            if gate_policy == "gate_a_removal_risk":
                delete_proxy = residual_proxy_fields.get("delete_risk_proxy_max")
                delete_proxy_float: Optional[float]
                try:
                    delete_proxy_float = (
                        float(delete_proxy) if delete_proxy is not None else None
                    )
                except (TypeError, ValueError):
                    delete_proxy_float = None
                if delete_proxy_float is not None and math.isfinite(
                    delete_proxy_float
                ):
                    gate_score = 1.0 - delete_proxy_float
                else:
                    delete_proxy_float = None
                if int(cfg.get("layer_idx", -1)) != int(
                    cfg.get("gate_apply_layer", cfg.get("layer_idx", -1))
                ):
                    would_gate_trigger = False
                    gate_reason = "non_final_gate_layer_audit_only"
                elif fallback_reason:
                    would_gate_trigger = False
                    gate_reason = f"selector_fallback:{fallback_reason}"
                elif not inactive_ids:
                    would_gate_trigger = False
                    gate_reason = "no_pruned_candidates"
                elif not candidate_rescue_ids or not candidate_dropped_ids:
                    would_gate_trigger = False
                    gate_reason = "swap_count_zero"
                elif delete_proxy_float is None:
                    would_gate_trigger = False
                    gate_reason = "missing_delete_risk_proxy_max"
                    gate_fields["gate_feature_missing"] = True
                    missing = list(
                        gate_fields.get("gate_feature_missing_fields", [])
                    )
                    if "delete_risk_proxy_max" not in missing:
                        missing.append("delete_risk_proxy_max")
                    gate_fields["gate_feature_missing_fields"] = missing
                elif delete_proxy_float >= float(
                    cfg.get("gate_deletion_risk_max", 0.20)
                ):
                    would_gate_trigger = False
                    gate_reason = "delete_risk_at_or_above_threshold"
                else:
                    would_gate_trigger = True
                    gate_reason = "delete_risk_below_threshold"
            else:
                if int(cfg.get("layer_idx", -1)) != int(
                    cfg.get("gate_apply_layer", cfg.get("layer_idx", -1))
                ):
                    would_gate_trigger = False
                    gate_reason = "non_final_gate_layer_audit_only"
                elif fallback_reason:
                    would_gate_trigger = False
                    gate_reason = f"selector_fallback:{fallback_reason}"
                elif not inactive_ids:
                    would_gate_trigger = False
                    gate_reason = "no_pruned_candidates"
                elif not candidate_rescue_ids or not candidate_dropped_ids:
                    would_gate_trigger = False
                    gate_reason = "swap_count_zero"
                elif bool(gate_fields.get("candidate_pool_exhausted")):
                    would_gate_trigger = False
                    gate_reason = "candidate_pool_exhausted"
                elif bool(gate_fields.get("gate_feature_missing")):
                    would_gate_trigger = False
                    gate_reason = (
                        "gate_feature_missing:"
                        + ",".join(gate_fields.get("gate_feature_missing_fields", []))
                    )
                elif is_v3_1 and bool(gate_fields.get("candidate_score_flat")):
                    would_gate_trigger = False
                    gate_reason = "candidate_score_flat"
                elif float(deletion_risk) >= (
                    0.45
                    if is_v3_1
                    else float(cfg.get("gate_deletion_risk_max", 0.20))
                ):
                    would_gate_trigger = False
                    gate_reason = (
                        "deletion_risk_v3_1_at_or_above_max"
                        if is_v3_1
                        else (
                            "deletion_evidence_risk_v3_at_or_above_max"
                            if gate_feature_version == "selector_native_v3"
                            else "deletion_risk_at_or_above_max"
                        )
                    )
                elif float(margin) < (
                    0.10
                    if is_v3_1
                    else float(cfg.get("gate_rescue_margin_min", 0.25))
                ):
                    would_gate_trigger = False
                    gate_reason = (
                        "rescue_boundary_margin_rank_below_min"
                        if is_v3_1
                        else "rescue_boundary_margin_below_min"
                    )
                elif float(coverage) <= (
                    0.20
                    if is_v3_1
                    else float(cfg.get("gate_coverage_gap_min", 0.0))
                ):
                    would_gate_trigger = False
                    gate_reason = (
                        "coverage_gap_v3_1_at_or_below_min"
                        if is_v3_1
                        else (
                            "coverage_gap_v3_at_or_below_min"
                            if gate_feature_version == "selector_native_v3"
                            else "rescue_coverage_gap_at_or_below_min"
                        )
                    )
                else:
                    would_gate_trigger = True
                    gate_reason = "conservative_v1_pass"

            selected_policy_after_gate = (
                "drm_swap"
                if gate_mode == "apply" and would_gate_trigger
                else gate_default_policy
            )
            if selected_policy_after_gate not in {"original", "drm_swap"}:
                selected_policy_after_gate = "original"
                gate_reason = "invalid_selected_policy_fail_closed"
                would_gate_trigger = False
            if gate_mode == "shadow":
                selected_policy_after_gate = "original"
            gate_fallback_to_original = selected_policy_after_gate == "original"
            if selected_policy_after_gate == "drm_swap":
                new_ids = set(candidate_new_ids)
                rescue_ids = list(candidate_rescue_ids)
                dropped_ids = list(candidate_dropped_ids)
            else:
                new_ids = set(active_ids)
                rescue_ids = []
                dropped_ids = []
            gate_fields.update({
                "gate_mode": gate_mode,
                "gate_policy": gate_policy,
                "gate_triggered": bool(
                    gate_mode == "apply"
                    and would_gate_trigger
                    and selected_policy_after_gate == "drm_swap"
                ),
                "would_gate_trigger": bool(would_gate_trigger),
                "would_gate_trigger_v3_1": bool(
                    would_gate_trigger if is_v3_1 else False
                ),
                "gate_score": gate_score,
                "gate_reason": gate_reason,
                "gate_reason_v3_1": (
                    gate_reason if is_v3_1 else "not_applicable"
                ),
                "gate_v3_1_thresholds": {
                    "deletion_risk_max": 0.45,
                    "rescue_boundary_margin_rank_min": 0.10,
                    "coverage_gap_min": 0.20,
                },
                "gate_default_policy": gate_default_policy,
                "gate_fallback_to_original": bool(gate_fallback_to_original),
                "selected_policy": selected_policy_after_gate,
                "selected_policy_before_gate": selected_policy_before_gate,
                "selected_policy_after_gate": selected_policy_after_gate,
                "candidate_swap_count": int(len(candidate_dropped_ids)),
                "candidate_rescued_block_ids": [
                    int(block_id) for block_id in candidate_rescue_ids
                ],
                "candidate_removed_block_ids": [
                    int(block_id) for block_id in candidate_dropped_ids
                ],
            })
        _write_rescue_audit_record(
            cfg=cfg,
            original_block_ids=original_block_ids,
            active_before_ids=active_ids,
            final_ids=sorted(new_ids),
            rescued_ids=rescue_ids,
            dropped_ids=dropped_ids,
            protected_ids=sorted(protected_ids),
            rescue_scores=rescue_scores,
            target_m=target_m,
            duplicate_block_ids=len(new_ids) != len(set(new_ids)),
            protected_violation=False,
            warnings=diagnostics.get("warnings", []),
            final_budget_type="K",
            score_meta={
                **score_meta,
                "query_fallback_used": bool(query_meta.get("fallback_used", False)),
                "query_fallback_reason": str(query_meta.get("fallback_reason", "")),
            },
            extra_fields={
                "swap_count": int(len(dropped_ids)),
                "rescue_count": int(len(rescue_ids)),
                "removed_count": int(len(dropped_ids)),
                "budget_equal_to_sliminfer": int(len(new_ids)) == int(len(active_ids)),
                "removed_block_ids": [int(block_id) for block_id in dropped_ids],
                "rescued_block_ids": [int(block_id) for block_id in rescue_ids],
                "rescue_score_mean": _mean_score_for_ids(rescue_scores, rescue_ids),
                "removed_score_mean": _mean_score_for_ids(drop_utility, dropped_ids),
                "active_redundancy_mean": float(active_redundancy_mean),
                "fallback_used": bool(fallback_reason),
                "fallback_reason": fallback_reason,
                "query_critical_block_ids": sorted(query_critical_ids),
                "forced_keep_block_ids": sorted(forced_ids),
                "protected_block_ids": sorted(protected_ids),
                "protected_reasons": {
                    str(block_id): reasons
                    for block_id, reasons in sorted(protected_reasons.items())
                },
                "gate_reason": gate_reason,
                "gate_triggered": False,
                **residual_proxy_fields,
                **gate_fields,
            },
        )
        _write_drm_runtime_trace(
            cfg=cfg,
            original_block_ids=original_block_ids,
            active_ids=active_ids,
            inactive_ids=inactive_ids,
            final_ids=sorted(new_ids),
            proposed_rescue_ids=candidate_rescue_ids,
            applied_rescue_ids=rescue_ids,
            dropped_ids=dropped_ids,
            protected_ids=protected_ids,
            rescue_scores=rescue_scores,
            reps=reps,
            score_meta=score_meta,
            target_m=target_m,
            final_budget_type="K",
            rejected_reason=gate_reason if not rescue_ids else fallback_reason,
            mmr_diag=mmr_diag,
            active_scores=active_global,
        )
        return {
            "final_kept_ids": new_ids,
            "rescue_policy": policy,
            "rescue_score": str(score_cfg.get("rescue_score", "drm")),
            "final_budget_type": "K",
            "selected_by_rescue_count": int(len(rescue_ids)),
            "bridge_quota_blocks_count": int(len(rescue_ids)),
            "replaced_blocks_count": int(len(dropped_ids)),
            "replaced_block_ids_sampled": sorted(rescue_ids)[:32],
            "replaced_out_block_ids_sampled": sorted(dropped_ids)[:32],
            "rescue_policy_warnings": diagnostics.get("warnings", []),
            "jaccard_vs_base": _jaccard(new_ids, set(active_ids)),
        }
    if (
        str(cfg.get("rescue_granularity", "page") or "page") == "microblock"
        and str(cfg.get("rescue_score", "")) in _STRUCTURED_RESCUE_SCORES
        and score_meta.get("microblock_select")
        and policy == "add"
        and inactive_ids
    ):
        if float(cfg.get("rescue_ratio", 0.05)) > 0.0 and active_ids:
            target_m = max(1, target_m)
        target_m = min(target_m, len(inactive_ids))
        base_global = _scatter_local_scores(
            delta_o_values if delta_o_values is not None else block_scores,
            original_block_ids,
            block_scores.device,
        )
        rescued_ids, mb_meta, candidate_rows = select_structured_rescue_via_microblocks(
            cfg=score_cfg,
            base_global=base_global,
            qk_global=_scatter_local_scores(qk_scores, original_block_ids, block_scores.device),
            reps=reps,
            active_ids=active_ids,
            inactive_ids=inactive_ids,
            target_m=target_m,
            diagnostics=diagnostics,
        )
        score_meta.update(mb_meta)
        context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
        method = str(cfg.get("rescue_score", ""))
        for row in candidate_rows:
            row.update({
                "task": context.get("task", ""),
                "sample_id": context.get("sample_id", ""),
                "layer_id": cfg.get("layer_idx", ""),
                "method": method,
                "block_id": row.get("block_id", row.get("page_id", "")),
            })
        final_row = build_structured_final_selection_row(
            score_cfg,
            active_ids=active_ids,
            rescued_ids=rescued_ids,
            mb_meta=mb_meta,
        )
        cand_written, final_written = append_selection_dumps(score_cfg, candidate_rows, final_row)
        marker = {
            "sample_id": context.get("sample_id", ""),
            "task": context.get("task", ""),
            "layer_id": cfg.get("layer_idx", ""),
            "method": method,
            "entered_structured_selector": True,
            "entered_real_fixed_block_path": False,
            "entered_cache_construction_path": True,
            "used_microblock_first_selector": True,
            "used_candidate_score_dump": bool(cand_written),
            "used_final_selection_dump": bool(final_written),
            "used_evidence_density_selector": bool(
                mb_meta.get("used_evidence_density_selector", False)
            ),
        }
        marker_written = append_path_marker(score_cfg, marker)
        enforce_strict_structured_path_check(
            score_cfg, path_marker=marker, mb_meta=mb_meta, cand_dump_written=cand_written,
            final_dump_written=final_written, marker_dump_written=marker_written,
        )
        new_ids = set(active_ids) | set(rescued_ids)
        protected_ids = set(int(x) for x in must_keep_block_ids)
        _write_rescue_audit_record(
            cfg=cfg,
            original_block_ids=original_block_ids,
            active_before_ids=active_ids,
            final_ids=sorted(new_ids),
            rescued_ids=rescued_ids,
            dropped_ids=[],
            protected_ids=sorted(protected_ids),
            rescue_scores=rescue_scores,
            target_m=target_m,
            duplicate_block_ids=len(new_ids) != len(active_ids) + len(rescued_ids),
            protected_violation=False,
            warnings=diagnostics.get("warnings", []),
            final_budget_type="K+M",
            score_meta=score_meta,
        )
        return {
            "final_kept_ids": new_ids,
            "rescue_policy": policy,
            "rescue_score": str(score_cfg.get("rescue_score", "delta_o")),
            "final_budget_type": "K+M",
            "selected_by_rescue_count": int(len(rescued_ids)),
            "bridge_quota_blocks_count": int(len(rescued_ids)),
            "replaced_blocks_count": int(len(rescued_ids)),
            "replaced_block_ids_sampled": sorted(rescued_ids)[:32],
            "replaced_out_block_ids_sampled": [],
            "rescue_policy_warnings": diagnostics.get("warnings", []),
            "jaccard_vs_base": _jaccard(new_ids, set(active_ids)),
        }
    if float(cfg.get("rescue_ratio", 0.05)) > 0.0 and active_ids and inactive_ids:
        target_m = max(1, target_m)
    target_m = min(target_m, len(inactive_ids))
    if score_meta.get("pool_sampling") and target_m > 0 and inactive_ids:
        rescue_ids, mmr_diag = pool_mmr_select_rescue_ids(
            inactive_ids=inactive_ids,
            base_scores=rescue_scores,
            block_reps=reps,
            active_ids=active_ids,
            m=target_m,
            candidate_pool_ratio=float(cfg.get("candidate_pool_ratio", 0.3)),
            temperature=float(cfg.get("sampling_temperature", 1.0)),
            eta_rescue=float(cfg.get("eta_rescue", 0.25)),
            eta_active=float(cfg.get("eta_active", 0.25)),
            seed=int(cfg.get("seed", 0) or 0),
            layer_idx=int(cfg.get("layer_idx", 0) or 0),
            diagnostics=diagnostics,
        )
        new_ids = set(active_ids) | set(rescue_ids)
        protected_ids = set(int(x) for x in must_keep_block_ids)
        duplicate_block_ids = len(new_ids) != len(active_ids) + len(rescue_ids)
        protected_violation = False
        score_meta.setdefault("MMR_diversity_stats", mmr_diag.get("MMR_diversity_stats", {}))
        _write_rescue_audit_record(
            cfg=cfg,
            original_block_ids=original_block_ids,
            active_before_ids=active_ids,
            final_ids=sorted(new_ids),
            rescued_ids=rescue_ids,
            dropped_ids=[],
            protected_ids=sorted(protected_ids),
            rescue_scores=rescue_scores,
            target_m=target_m,
            duplicate_block_ids=duplicate_block_ids,
            protected_violation=protected_violation,
            warnings=diagnostics.get("warnings", []),
            final_budget_type="K+M",
            score_meta=score_meta,
            policy_diagnostics={"mmr_diversity_stats": [str(mmr_diag.get("MMR_diversity_stats", {}))]},
        )
        return {
            "final_kept_ids": new_ids,
            "rescue_policy": policy,
            "rescue_score": str(score_cfg.get("rescue_score", "delta_o")),
            "final_budget_type": "K+M",
            "selected_by_rescue_count": int(len(rescue_ids)),
            "bridge_quota_blocks_count": int(len(rescue_ids)),
            "replaced_blocks_count": int(len(rescue_ids)),
            "replaced_block_ids_sampled": sorted(rescue_ids)[:32],
            "replaced_out_block_ids_sampled": [],
            "rescue_policy_warnings": diagnostics.get("warnings", []),
            "jaccard_vs_base": _jaccard(new_ids, set(active_ids)),
        }
    if policy == "add" and bool(score_meta.get("mmr_selection", False)):
        if target_m > 0 and inactive_ids:
            rescue_ids, mmr_diag = mmr_select_rescue_ids(
                inactive_ids=inactive_ids,
                s0_scores=rescue_scores,
                block_reps=reps,
                active_ids=active_ids,
                m=target_m,
                eta_rescue=float(cfg.get("eta_rescue", 0.25)),
                eta_active=float(cfg.get("eta_active", 0.25)),
                seed=int(cfg.get("seed", 0) or 0),
                diagnostics=diagnostics,
            )
        else:
            rescue_ids, mmr_diag = [], {}
        new_ids = set(active_ids) | set(rescue_ids)
        protected_ids = set(int(x) for x in must_keep_block_ids)
        duplicate_block_ids = len(new_ids) != len(active_ids) + len(rescue_ids)
        protected_violation = False
        score_meta["MMR_diversity_stats"] = mmr_diag.get("MMR_diversity_stats", {})
        score_meta["mmr_vs_plain_top_m_jaccard"] = mmr_diag.get("mmr_vs_plain_top_m_jaccard", "")
        _write_rescue_audit_record(
            cfg=cfg,
            original_block_ids=original_block_ids,
            active_before_ids=active_ids,
            final_ids=sorted(new_ids),
            rescued_ids=rescue_ids,
            dropped_ids=[],
            protected_ids=sorted(protected_ids),
            rescue_scores=rescue_scores,
            target_m=target_m,
            duplicate_block_ids=duplicate_block_ids,
            protected_violation=protected_violation,
            warnings=diagnostics.get("warnings", []),
            final_budget_type="K+M",
            score_meta=score_meta,
            policy_diagnostics={
                "mmr_diversity_stats": [str(mmr_diag.get("MMR_diversity_stats", {}))]
            },
        )
        _write_drm_runtime_trace(
            cfg=cfg,
            original_block_ids=original_block_ids,
            active_ids=active_ids,
            inactive_ids=inactive_ids,
            final_ids=sorted(new_ids),
            proposed_rescue_ids=rescue_ids,
            applied_rescue_ids=rescue_ids,
            dropped_ids=[],
            protected_ids=protected_ids,
            rescue_scores=rescue_scores,
            reps=reps,
            score_meta=score_meta,
            target_m=target_m,
            final_budget_type="K+M",
            rejected_reason="",
            mmr_diag=mmr_diag,
            active_scores=active_global,
        )
        return {
            "final_kept_ids": new_ids,
            "rescue_policy": policy,
            "rescue_score": str(score_cfg.get("rescue_score", "delta_o")),
            "final_budget_type": "K+M",
            "selected_by_rescue_count": int(len(rescue_ids)),
            "bridge_quota_blocks_count": int(len(rescue_ids)),
            "replaced_blocks_count": 0,
            "replaced_block_ids_sampled": sorted(rescue_ids)[:32],
            "replaced_out_block_ids_sampled": [],
            "rescue_policy_warnings": diagnostics.get("warnings", []),
            "jaccard_vs_base": _jaccard(new_ids, set(active_ids)),
        }
    obd_global = None
    if str(cfg.get("swap_drop_score", "sliminfer")) == "obd":
        obd_global = _obd_scores_global(kernel_scores, represent_values, num_key_value_groups, block_slices, original_block_ids, block_scores.device, diagnostics)
    result = apply_rescue_policy(
        active_ids=active_ids,
        inactive_ids=inactive_ids,
        rescue_scores=rescue_scores,
        active_scores=active_global,
        obd_scores=obd_global,
        block_reps=reps,
        policy=policy,
        rescue_ratio=float(cfg.get("rescue_ratio", 0.05)),
        swap_drop_score=str(cfg.get("swap_drop_score", "sliminfer")),
        preserve_ids=must_keep_block_ids,
        recent_ids=[],
        sink_ids=[],
        forced_keep_ids=[],
        tau=float(cfg.get("swap_tau", 0.0)),
        mmr_selection=bool(score_meta.get("mmr_selection", False)),
        eta_rescue=float(cfg.get("eta_rescue", 0.25)),
        eta_active=float(cfg.get("eta_active", 0.25)),
        seed=int(cfg.get("seed", 0) or 0),
        diagnostics=diagnostics,
    )
    new_ids = set(int(x) for x in result.final_ids.detach().cpu().tolist())
    rescued_ids = [int(x) for x in result.rescued_ids.detach().cpu().tolist()]
    dropped_ids = [int(x) for x in result.dropped_ids.detach().cpu().tolist()]
    protected_ids = set(int(x) for x in must_keep_block_ids)
    target_m = int(round(len(active_ids) * float(cfg.get("rescue_ratio", 0.05))))
    if float(cfg.get("rescue_ratio", 0.05)) > 0.0 and active_ids and inactive_ids:
        target_m = max(1, target_m)
    target_m = min(target_m, len(inactive_ids))
    duplicate_block_ids = len(new_ids) != int(result.final_ids.numel())
    protected_violation = bool(set(dropped_ids).intersection(protected_ids))
    _write_rescue_audit_record(
        cfg=cfg,
        original_block_ids=original_block_ids,
        active_before_ids=active_ids,
        final_ids=sorted(new_ids),
        rescued_ids=rescued_ids,
        dropped_ids=dropped_ids,
        protected_ids=sorted(protected_ids),
        rescue_scores=rescue_scores,
        target_m=target_m,
        duplicate_block_ids=duplicate_block_ids,
        protected_violation=protected_violation,
        warnings=diagnostics.get("warnings", []),
        final_budget_type=result.final_budget_type,
        score_meta=score_meta,
        policy_diagnostics=result.diagnostics,
    )
    _write_drm_runtime_trace(
        cfg=cfg,
        original_block_ids=original_block_ids,
        active_ids=active_ids,
        inactive_ids=inactive_ids,
        final_ids=sorted(new_ids),
        proposed_rescue_ids=rescued_ids,
        applied_rescue_ids=rescued_ids,
        dropped_ids=dropped_ids,
        protected_ids=protected_ids,
        rescue_scores=rescue_scores,
        reps=reps,
        score_meta=score_meta,
        target_m=target_m,
        final_budget_type=result.final_budget_type,
        rejected_reason="; ".join(str(x) for x in diagnostics.get("warnings", [])),
        mmr_diag={},
        active_scores=active_global,
    )
    return {
        "final_kept_ids": new_ids,
        "rescue_policy": policy,
        "rescue_score": str(score_cfg.get("rescue_score", "delta_o")),
        "final_budget_type": result.final_budget_type,
        "selected_by_rescue_count": int(len(rescued_ids)),
        "bridge_quota_blocks_count": int(len(rescued_ids)),
        "replaced_blocks_count": int(len(rescued_ids)),
        "replaced_block_ids_sampled": sorted(rescued_ids)[:32],
        "replaced_out_block_ids_sampled": sorted(dropped_ids)[:32],
        "rescue_policy_warnings": diagnostics.get("warnings", []),
        "jaccard_vs_base": _jaccard(new_ids, set(active_ids)),
    }

def _score_stats_for_block_ids(scores: torch.Tensor, block_ids: List[int]) -> Dict[str, float]:
    if scores is None or scores.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    vals = [float(scores[int(bid)]) for bid in block_ids if 0 <= int(bid) < scores.numel()]
    if not vals:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return _score_stats(torch.tensor(vals, dtype=torch.float32))

def _score_stats(scores: torch.Tensor) -> Dict[str, float]:
    if scores is None or scores.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    finite = torch.where(torch.isfinite(scores.detach().float()), scores.detach().float(), torch.zeros_like(scores.detach().float()))
    return {
        "min": float(finite.min().detach().cpu()),
        "max": float(finite.max().detach().cpu()),
        "mean": float(finite.mean().detach().cpu()),
        "std": float(finite.std(unbiased=False).detach().cpu()),
    }

def _deterministic_seeded_random_value(idx: int, seed: int) -> float:
    # SplitMix-style integer mixing; unlike adding a seed offset, this changes
    # the ordering across seeds even for small contiguous block-id ranges.
    x = (int(idx) + 0x9E3779B9 + (int(seed) + 1) * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return float(x) / 4294967296.0

def _write_rescue_audit_record(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    active_before_ids: List[int],
    final_ids: List[int],
    rescued_ids: List[int],
    dropped_ids: List[int],
    protected_ids: List[int],
    rescue_scores: torch.Tensor,
    target_m: int,
    duplicate_block_ids: bool,
    protected_violation: bool,
    warnings: List[str],
    final_budget_type: str,
    score_meta: Optional[Dict[str, Any]] = None,
    policy_diagnostics: Optional[Dict[str, List[str]]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    audit_path = cfg.get("rescue_audit_path")
    if not audit_path:
        return
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    score_meta = score_meta or {}
    policy_diagnostics = policy_diagnostics or {}
    extra_fields = extra_fields or {}
    context_len = int(cfg.get("original_prompt_len", context.get("context_length", 0)) or 0)
    selected_abs_indices = from_blocks_to_abs_indices_list([int(x) for x in final_ids], context_len) if context_len > 0 else []
    selected_token_count = int(len(selected_abs_indices))
    selected_token_ratio = float(selected_token_count / max(1, context_len))
    block_lengths = [
        max(0, min(context_len, (int(block_id) + 1) * _BLOCK_SIZE) - int(block_id) * _BLOCK_SIZE)
        for block_id in original_block_ids
    ]
    len_stats = _block_length_stats(block_lengths)
    mmr_stats = score_meta.get("MMR_diversity_stats", {})
    for item in policy_diagnostics.get("mmr_diversity_stats", []):
        if isinstance(item, str) and item.startswith("{"):
            try:
                import ast
                mmr_stats = ast.literal_eval(item)
            except Exception:
                pass
    record = {
        "task": context.get("task", ""),
        "sample_id": context.get("sample_id", ""),
        "sample_index": context.get("sample_index", ""),
        "method": context.get("method", cfg.get("variant_method", "")),
        "block_strategy": cfg.get("block_strategy", "fixed"),
        "context_length": context.get("context_length", ""),
        "layer_id": cfg.get("layer_idx", ""),
        "phase": cfg.get("phase", ""),
        "rescue_policy": cfg.get("rescue_policy", "none"),
        "rescue_score": cfg.get("rescue_score", "delta_o"),
        "seed": cfg.get("seed", 0),
        "K": int(len(active_before_ids)),
        "M": int(target_m),
        "active_before_count": int(len(active_before_ids)),
        "final_block_count_policy": int(len(final_ids)),
        "final_block_count_used_by_attention": int(len(final_ids)),
        "final_block_count": int(len(final_ids)),
        "selected_token_count": selected_token_count,
        "selected_token_ratio": selected_token_ratio,
        "avg_block_tokens": float(len_stats["mean"]),
        "block_count_total": int(len(original_block_ids)),
        "block_len_mean": float(len_stats["mean"]),
        "block_len_std": float(len_stats["std"]),
        "block_len_min": float(len_stats["min"]),
        "block_len_max": float(len_stats["max"]),
        "rescued_count": int(len(rescued_ids)),
        "swapped_count": int(len(dropped_ids)),
        "protected_blocks": int(len(protected_ids)),
        "protected_block_ids_sampled": protected_ids[:16],
        "duplicate_block_ids": bool(duplicate_block_ids),
        "protected_block_violation": bool(protected_violation),
        "fallback_used": bool(warnings),
        "fallback_reason": "; ".join(str(x) for x in warnings) if warnings else "",
        "score_stats": _score_stats(rescue_scores),
        "D_base_stats": score_meta.get("D_base_stats", {}),
        "C_active_stats": score_meta.get("C_active_stats", {}),
        "G_diffusion_stats": score_meta.get("G_diffusion_stats", {}),
        "R_active_stats": score_meta.get("R_active_stats", {}),
        "E_query_stats": score_meta.get("E_query_stats", {}),
        "S_query_drm_stats": score_meta.get("S_query_drm_stats", {}),
        "stochastic_noise_stats": score_meta.get("stochastic_noise_stats", {}),
        "query_overlap_stats": score_meta.get("query_overlap_stats", score_meta.get("E_query_stats", {})),
        "query_fallback_used": bool(score_meta.get("query_fallback_used", False)),
        "query_fallback_reason": str(score_meta.get("query_fallback_reason", "")),
        "MMR_diversity_stats": mmr_stats or score_meta.get("MMR_diversity_stats", {}),
        "jaccard_vs_sliminfer": score_meta.get("jaccard_vs_sliminfer", ""),
        "drm_trace_dir": str(cfg.get("drm_trace_dir", "") or ""),
        "drm_trace_enabled": bool(_drm_trace_dir(cfg)),
        "final_budget_type": final_budget_type,
        "num_original_blocks": int(len(original_block_ids)),
        "final_block_ids_first100": final_ids[:100],
        "rescued_block_ids_first50": rescued_ids[:50],
        "final_ids_sampled": final_ids[:32],
        "rescued_ids_sampled": rescued_ids[:32],
        "dropped_ids_sampled": dropped_ids[:32],
    }
    if str(cfg.get("rescue_score", "")) in _STRUCTURED_RESCUE_SCORES:
        enrich_structured_audit_fields(
            record, cfg, score_meta or {},
            active_before_ids, rescued_ids, final_ids,
        )
    record.update(extra_fields)
    try:
        with open(str(audit_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _ordered_ids_by_score(scores: torch.Tensor, original_block_ids: List[int], descending: bool = True) -> List[int]:
    if scores is None or scores.numel() == 0:
        return []
    order = torch.argsort(scores.detach().float(), descending=descending).detach().cpu().tolist()
    return [int(original_block_ids[int(idx)]) for idx in order if int(idx) < len(original_block_ids)]

def _rank_map(scores: torch.Tensor, original_block_ids: List[int], descending: bool = True) -> Dict[int, int]:
    return {block_id: rank + 1 for rank, block_id in enumerate(_ordered_ids_by_score(scores, original_block_ids, descending=descending))}

def _select_top_ids(scores: torch.Tensor, original_block_ids: List[int], k: int, exclude: Optional[Set[int]] = None) -> List[int]:
    exclude = exclude or set()
    out = []
    for block_id in _ordered_ids_by_score(scores, original_block_ids, descending=True):
        if block_id in exclude:
            continue
        out.append(block_id)
        if len(out) >= k:
            break
    return out

def _bridge_base_scores(
    cfg: Dict[str, Any],
    qk_scores: torch.Tensor,
    delta_o_values: Optional[torch.Tensor],
    bridge_result: Dict[str, Any],
) -> torch.Tensor:
    base = cfg.get("base_score", cfg.get("local_score", "delta_o"))
    if base == "qk":
        return _normalize_scores(qk_scores)
    if delta_o_values is not None:
        return _normalize_scores(delta_o_values)
    return bridge_result.get("local_scores", _normalize_scores(qk_scores))

def _rescue_source_scores(
    cfg: Dict[str, Any],
    base_scores: torch.Tensor,
    qk_scores: torch.Tensor,
    bridge_scores: torch.Tensor,
    bridge_result: Dict[str, Any],
) -> torch.Tensor:
    source = str(cfg.get("rescue_source", "bridge") or "bridge")
    if source in {"bridge", "ppr", "path_coverage"}:
        return bridge_scores
    if source == "delta_o_boundary":
        return _normalize_scores(base_scores)
    if source == "qk_disagreement":
        qk_norm = _normalize_scores(qk_scores)
        base_norm = _normalize_scores(base_scores)
        return _normalize_scores(qk_norm * (1.0 - base_norm))
    if source == "semantic_diversity":
        lexical = bridge_result.get("lexical_scores", torch.zeros_like(base_scores))
        idf_node = bridge_result.get("idf_node_scores", torch.ones_like(base_scores))
        structural = bridge_result.get("structural_scores", torch.zeros_like(base_scores))
        base_norm = _normalize_scores(base_scores)
        score = _normalize_scores(lexical) * (0.5 + 0.5 * _normalize_scores(idf_node))
        score = score * (1.0 - base_norm) + 0.25 * _normalize_scores(structural)
        return _normalize_scores(score)
    if source == "random_boundary":
        n = int(base_scores.numel())
        seed = int(cfg.get("seed", 0) or 0)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        vals = torch.rand(n, generator=generator, dtype=torch.float32)
        return _normalize_scores(vals.to(device=base_scores.device))
    return bridge_scores

def _select_bridgekv_policy_ids(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    num_blocks_to_keep: int,
    must_keep_block_ids: Set[int],
    bridge_result: Dict[str, Any],
    qk_scores: torch.Tensor,
    delta_o_values: Optional[torch.Tensor],
) -> Tuple[Set[int], Dict[str, Any]]:
    policy = cfg.get("selection_policy", "linear_fusion")
    k = min(max(int(num_blocks_to_keep), len(must_keep_block_ids)), len(original_block_ids))
    base_scores = _bridge_base_scores(cfg, qk_scores, delta_o_values, bridge_result)
    bridge_scores = bridge_result["bridge_scores"]
    rescue_scores = _rescue_source_scores(cfg, base_scores, qk_scores, bridge_scores, bridge_result)
    structural_scores = bridge_result.get("structural_scores", torch.zeros_like(base_scores))
    final_scores = bridge_result.get("final_utility", base_scores)
    base_top_k = set(_select_top_ids(base_scores, original_block_ids, k)).union(must_keep_block_ids)
    base_top_k = set(_select_top_ids(base_scores, original_block_ids, k, exclude=set())) | must_keep_block_ids
    selected: Set[int] = set(base_top_k)
    bridge_added: Set[int] = set()
    replaced: Set[int] = set()
    selection_metric = final_scores if policy == "linear_fusion" else rescue_scores

    if policy == "linear_fusion":
        selected = set(_select_top_ids(final_scores, original_block_ids, k)).union(must_keep_block_ids)
    elif policy == "core_bridge_quota":
        quota = min(max(0, math.ceil(float(cfg.get("quota_ratio", 0.05)) * k)), max(0, k - len(must_keep_block_ids)))
        core_budget = max(len(must_keep_block_ids), k - quota)
        selected = set(_select_top_ids(base_scores, original_block_ids, core_budget)).union(must_keep_block_ids)
        for block_id in _select_top_ids(rescue_scores, original_block_ids, k, exclude=selected):
            if len(selected) >= k:
                break
            selected.add(block_id)
            bridge_added.add(block_id)
    elif policy == "boundary_rerank":
        selected = set(base_top_k)
        replace_budget = min(max(1, math.ceil(float(cfg.get("replace_ratio", 0.05)) * k)), max(0, k - len(must_keep_block_ids)))
        boundary_count = max(replace_budget, math.ceil(float(cfg.get("boundary_ratio", 0.10)) * k))
        core_count = min(k, max(len(must_keep_block_ids), math.ceil(float(cfg.get("core_ratio", 0.90)) * k)))
        core_ids = set(_select_top_ids(base_scores, original_block_ids, core_count)).union(must_keep_block_ids)
        removable = [block_id for block_id in _ordered_ids_by_score(base_scores, original_block_ids, descending=False) if block_id in selected and block_id not in core_ids and block_id not in must_keep_block_ids][:boundary_count]
        unselected_boundary = _select_top_ids(base_scores, original_block_ids, boundary_count, exclude=selected)
        utility = _normalize_scores(base_scores) + float(cfg.get("bridge_beta", 0.1)) * _normalize_scores(rescue_scores)
        incoming = [block_id for block_id in _ordered_ids_by_score(utility, original_block_ids, descending=True) if block_id in set(unselected_boundary)]
        for out_id, in_id in zip(removable[:replace_budget], incoming[:replace_budget]):
            if in_id in selected:
                continue
            selected.remove(out_id)
            selected.add(in_id)
            replaced.add(out_id)
            bridge_added.add(in_id)
    elif policy == "agreement_gated_rescue":
        selected = set(base_top_k)
        rescue_budget = min(max(1, math.ceil(float(cfg.get("rescue_ratio", 0.02)) * k)), max(0, k - len(must_keep_block_ids)))
        p = float(cfg.get("agreement_top_p", 0.10))
        p_k = max(1, math.ceil(len(original_block_ids) * p))
        high_bridge = set(_select_top_ids(rescue_scores, original_block_ids, p_k))
        high_qk = set(_select_top_ids(qk_scores, original_block_ids, p_k))
        high_do = set(_select_top_ids(delta_o_values, original_block_ids, p_k)) if delta_o_values is not None else set()
        high_struct = set(_select_top_ids(structural_scores, original_block_ids, p_k))
        agreement = high_qk | high_do | high_struct
        rescue_pool = [block_id for block_id in _ordered_ids_by_score(rescue_scores, original_block_ids, descending=True) if block_id in high_bridge and block_id in agreement and block_id not in selected]
        removable = [block_id for block_id in _ordered_ids_by_score(base_scores, original_block_ids, descending=False) if block_id in selected and block_id not in must_keep_block_ids]
        for out_id, in_id in zip(removable[:rescue_budget], rescue_pool[:rescue_budget]):
            selected.remove(out_id)
            selected.add(in_id)
            replaced.add(out_id)
            bridge_added.add(in_id)
    elif policy == "evidence_first":
        evidence_budget = min(k, max(len(must_keep_block_ids), math.ceil(float(cfg.get("evidence_ratio", 0.90)) * k)))
        selected = set(_select_top_ids(base_scores, original_block_ids, evidence_budget)).union(must_keep_block_ids)
        for block_id in _select_top_ids(rescue_scores, original_block_ids, k, exclude=selected):
            if len(selected) >= k:
                break
            selected.add(block_id)
            bridge_added.add(block_id)

    if len(selected) < k:
        for block_id in _select_top_ids(base_scores, original_block_ids, k, exclude=selected):
            selected.add(block_id)
            if len(selected) >= k:
                break
    if len(selected) > k:
        removable = [block_id for block_id in _ordered_ids_by_score(base_scores, original_block_ids, descending=False) if block_id in selected and block_id not in must_keep_block_ids]
        for block_id in removable:
            if len(selected) <= k:
                break
            selected.remove(block_id)
            replaced.add(block_id)

    changed = selected.difference(base_top_k)
    base_ranks = _rank_map(base_scores, original_block_ids, descending=True)
    bridge_ranks = _rank_map(rescue_scores, original_block_ids, descending=True)
    replacement_source = str(cfg.get("rescue_source", cfg.get("bridge_score", "bridge")) or "bridge")
    if replacement_source in {"bridge", "ppr", "path_coverage"}:
        replacement_source = str(cfg.get("bridge_score", "bridge"))
    control_type = str(cfg.get("control_type", "none"))
    if control_type and control_type != "none":
        replacement_source = control_type
    elif policy == "evidence_first":
        replacement_source = "evidence_first"
    elif policy == "boundary_rerank":
        replacement_source = "boundary"
    details = {
        "selection_policy": policy,
        "base_score": cfg.get("base_score", cfg.get("local_score", "delta_o")),
        "bridge_score": cfg.get("bridge_score", "ppr"),
        "control_type": control_type,
        "replacement_source": replacement_source,
        "rescue_source": replacement_source,
        "bridge_quota_blocks_count": int(len(bridge_added)),
        "selected_by_rescue_count": int(len(bridge_added)),
        "replaced_blocks_count": int(len(changed)),
        "replaced_block_ids_sampled": sorted(int(x) for x in changed)[:32],
        "replaced_out_block_ids_sampled": sorted(int(x) for x in replaced)[:32],
        "replaced_block_base_ranks_sampled": [int(base_ranks.get(int(x), -1)) for x in sorted(int(y) for y in changed)[:32]],
        "replaced_block_bridge_ranks_sampled": [int(bridge_ranks.get(int(x), -1)) for x in sorted(int(y) for y in changed)[:32]],
        "jaccard_vs_base": _jaccard(selected, base_top_k),
    }
    return selected, details

def _maybe_normalize_scores(scores: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    return _normalize_scores(scores)

def _update_score_runtime_stats(cfg: Dict[str, Any], score_start: float) -> None:
    runtime_stats = cfg.get("runtime_stats")
    if not isinstance(runtime_stats, dict):
        return
    elapsed_ms = (time.perf_counter() - score_start) * 1000.0
    runtime_stats["score_overhead_ms"] = float(runtime_stats.get("score_overhead_ms", 0.0)) + float(elapsed_ms)
    runtime_stats["score_calls"] = int(runtime_stats.get("score_calls", 0)) + 1
    phase = cfg.get("phase", "unknown")
    if phase == "prefill":
        runtime_stats["prefill_score_calls"] = int(runtime_stats.get("prefill_score_calls", 0)) + 1
    elif phase == "decode":
        runtime_stats["decode_score_calls"] = int(runtime_stats.get("decode_score_calls", 0)) + 1

def _uasr_json_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    if not math.isfinite(result):
        return None
    return result

def _uasr_tensor_to_float_list(scores: Optional[torch.Tensor]) -> List[Optional[float]]:
    if scores is None:
        return []
    values = scores.detach().float().cpu().tolist()
    return [_uasr_json_float(value) for value in values]

def _uasr_percentile(values: List[float], q: float) -> Optional[float]:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return None
    if len(finite) == 1:
        return float(finite[0])
    pos = (len(finite) - 1) * max(0.0, min(1.0, float(q)))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(finite[lo])
    frac = pos - lo
    return float(finite[lo] * (1.0 - frac) + finite[hi] * frac)

def _uasr_iqr(values: List[float]) -> Optional[float]:
    p75 = _uasr_percentile(values, 0.75)
    p25 = _uasr_percentile(values, 0.25)
    if p75 is None or p25 is None:
        return None
    return float(p75 - p25)

def _uasr_std(values: List[float]) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    mean = sum(finite) / len(finite)
    return float(math.sqrt(sum((v - mean) ** 2 for v in finite) / len(finite)))

def _uasr_entropy(values: List[float]) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    shift = max(finite)
    exps = [math.exp(max(-80.0, min(80.0, v - shift))) for v in finite]
    total = sum(exps)
    if total <= 0:
        return None
    probs = [v / total for v in exps]
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))

_UASR_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "with",
}

def _uasr_query_terms(query_text: str) -> Set[str]:
    terms = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-']*", str(query_text).lower()):
        if len(token) < 3 or token in _UASR_STOPWORDS:
            continue
        terms.add(token)
    return terms

def _uasr_query_overlap_proxy(block_text: str, query_terms: Set[str]) -> Optional[float]:
    if not query_terms:
        return None
    text_terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-']*", str(block_text).lower()))
    if not text_terms:
        return 0.0
    return float(len(query_terms.intersection(text_terms)) / max(1, len(query_terms)))

def _write_uasr_score_dump(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    final_kept_ids: Set[int],
    block_scores: torch.Tensor,
    qk_block_scores: Optional[torch.Tensor],
    delta_o_values: Optional[torch.Tensor],
    num_blocks_to_keep: int,
    must_keep_block_ids: Set[int],
) -> None:
    dump_path = cfg.get("uasr_score_dump_path")
    if not dump_path:
        return
    if str(cfg.get("phase", "")) == "decode":
        return

    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    prompt_len = int(cfg.get("original_prompt_len", context.get("context_length", 0)) or 0)
    prompt_block_count = int(math.ceil(prompt_len / _BLOCK_SIZE)) if prompt_len > 0 else int(len(original_block_ids))
    selected = sorted(int(x) for x in final_kept_ids)
    selected_set = set(selected)
    pruned = [int(block_id) for block_id in original_block_ids if int(block_id) not in selected_set]
    score_values = _uasr_tensor_to_float_list(block_scores)
    qk_values = _uasr_tensor_to_float_list(qk_block_scores)
    delta_o_score_values = _uasr_tensor_to_float_list(delta_o_values)
    id_to_score = {
        int(block_id): score_values[index]
        for index, block_id in enumerate(original_block_ids)
        if index < len(score_values)
    }

    def score_for(block_id: int) -> Optional[float]:
        return id_to_score.get(int(block_id))

    finite_pairs = [
        (int(block_id), float(score))
        for block_id, score in id_to_score.items()
        if score is not None and math.isfinite(float(score))
    ]
    sorted_pairs = sorted(finite_pairs, key=lambda item: item[1], reverse=True)
    sorted_ids = [int(block_id) for block_id, _ in sorted_pairs]
    sorted_scores = [float(score) for _, score in sorted_pairs]
    rank_by_block = {int(block_id): int(rank) for rank, block_id in enumerate(sorted_ids, 1)}

    selected_scored = [
        (int(block_id), score_for(int(block_id)))
        for block_id in selected
        if score_for(int(block_id)) is not None
    ]
    pruned_scored = [
        (int(block_id), score_for(int(block_id)))
        for block_id in pruned
        if score_for(int(block_id)) is not None
    ]
    selected_tail_width = max(1, int(cfg.get("uasr_tail_width", 8) or 8))
    selected_tail_pairs = sorted(
        [(bid, float(score)) for bid, score in selected_scored if score is not None],
        key=lambda item: item[1],
    )[:selected_tail_width]
    pruned_head_pairs = sorted(
        [(bid, float(score)) for bid, score in pruned_scored if score is not None],
        key=lambda item: item[1],
        reverse=True,
    )[:selected_tail_width]

    selected_scores_finite = [float(score) for _, score in selected_scored if score is not None and math.isfinite(float(score))]
    pruned_scores_finite = [float(score) for _, score in pruned_scored if score is not None and math.isfinite(float(score))]
    selected_min = min(selected_scores_finite) if selected_scores_finite else None
    pruned_max = max(pruned_scores_finite) if pruned_scores_finite else None
    cutoff_margin = (selected_min - pruned_max) if selected_min is not None and pruned_max is not None else None
    denom = abs(selected_min or 0.0) + abs(pruned_max or 0.0) + 1e-6
    cutoff_margin_ratio = (cutoff_margin / denom) if cutoff_margin is not None else None

    def within_eps_count(eps: float) -> int:
        if selected_min is None or pruned_max is None:
            return 0
        lower = min(selected_min, pruned_max) - eps
        upper = max(selected_min, pruned_max) + eps
        return sum(1 for _, score in finite_pairs if lower <= score <= upper)

    near_scores: List[float] = []
    if selected_min is not None and pruned_max is not None:
        lower = min(selected_min, pruned_max) - 0.02
        upper = max(selected_min, pruned_max) + 0.02
        near_scores = [score for _, score in finite_pairs if lower <= score <= upper]
    if not near_scores:
        near_scores = [score for _, score in selected_tail_pairs + pruned_head_pairs]

    block_texts = cfg.get("block_texts", context.get("block_texts", []))
    if not isinstance(block_texts, list):
        block_texts = []
    query_text = str(cfg.get("query_text", context.get("query_text", "")) or "")
    query_terms = _uasr_query_terms(query_text)

    def block_preview(block_id: int) -> str:
        if 0 <= int(block_id) < len(block_texts):
            return str(block_texts[int(block_id)])[:240]
        return ""

    def block_span(block_id: int) -> List[int]:
        start = int(block_id) * _BLOCK_SIZE
        end = min(prompt_len, (int(block_id) + 1) * _BLOCK_SIZE) if prompt_len > 0 else (int(block_id) + 1) * _BLOCK_SIZE
        return [int(start), int(end)]

    near_cutoff_metadata: List[Dict[str, Any]] = []
    seen_meta: Set[int] = set()
    for block_id, score in selected_tail_pairs + pruned_head_pairs:
        if int(block_id) in seen_meta:
            continue
        seen_meta.add(int(block_id))
        preview = block_preview(int(block_id))
        overlap = _uasr_query_overlap_proxy(preview, query_terms) if preview else None
        near_cutoff_metadata.append({
            "block_id": int(block_id),
            "is_selected": bool(int(block_id) in selected_set),
            "score": _uasr_json_float(score),
            "rank": int(rank_by_block.get(int(block_id), -1)),
            "query_overlap_proxy": overlap,
            "protected_flag": bool(int(block_id) in must_keep_block_ids),
            "query_critical_flag": bool(overlap is not None and overlap > 0.0),
            "query_critical_flag_source": "query_overlap_proxy" if overlap is not None else "unavailable",
            "block_text_preview": preview,
            "token_start_end": block_span(int(block_id)),
        })

    fixed_abs = from_blocks_to_abs_indices_list(selected, prompt_len) if prompt_len > 0 else []
    record = {
        "sample_id": context.get("sample_id", ""),
        "sample_index": context.get("sample_index", ""),
        "task": context.get("task", ""),
        "method": context.get("method", cfg.get("variant_method", "")),
        "phase": cfg.get("phase", "unknown"),
        "layer_id": int(cfg.get("layer_idx", -1)),
        "scoring": cfg.get("score_kind", cfg.get("scoring", "")),
        "fairness_base_scoring": cfg.get("fairness_base_scoring", ""),
        "rescue_policy": cfg.get("rescue_policy", "none"),
        "prompt_block_count": prompt_block_count,
        "prompt_token_count": prompt_len,
        "budget_k": int(num_blocks_to_keep),
        "used_blocks": selected,
        "used_tokens": [int(x) for x in fixed_abs],
        "score_dump_available": True,
        "score_dump_unavailable_reason": "",
        "all_candidate_block_ids": [int(x) for x in original_block_ids],
        "all_candidate_scores": score_values,
        "all_candidate_qk_scores": qk_values,
        "all_candidate_delta_o_scores": delta_o_score_values,
        "sorted_block_ids": sorted_ids,
        "sorted_block_scores": sorted_scores,
        "selected_block_ids": selected,
        "selected_block_scores": [_uasr_json_float(score_for(int(block_id))) for block_id in selected],
        "pruned_block_ids": pruned,
        "pruned_block_scores": [_uasr_json_float(score_for(int(block_id))) for block_id in pruned],
        "selected_tail_block_ids": [int(block_id) for block_id, _ in selected_tail_pairs],
        "selected_tail_scores": [_uasr_json_float(score) for _, score in selected_tail_pairs],
        "pruned_head_block_ids": [int(block_id) for block_id, _ in pruned_head_pairs],
        "pruned_head_scores": [_uasr_json_float(score) for _, score in pruned_head_pairs],
        "cutoff_score_selected_min": _uasr_json_float(selected_min),
        "cutoff_score_pruned_max": _uasr_json_float(pruned_max),
        "cutoff_margin": _uasr_json_float(cutoff_margin),
        "cutoff_margin_ratio": _uasr_json_float(cutoff_margin_ratio),
        "num_blocks_within_margin_eps": int(within_eps_count(0.05)),
        "num_blocks_within_eps_0_01": int(within_eps_count(0.01)),
        "num_blocks_within_eps_0_02": int(within_eps_count(0.02)),
        "score_entropy_near_cutoff": _uasr_json_float(_uasr_entropy(near_scores)),
        "score_std_near_cutoff": _uasr_json_float(_uasr_std(near_scores)),
        "score_iqr_near_cutoff": _uasr_json_float(_uasr_iqr(near_scores)),
        "near_cutoff_block_metadata": near_cutoff_metadata,
        "block_text_available": bool(block_texts),
        "query_text_available": bool(query_text),
    }
    with open(str(dump_path), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(max(1, int(k)), scores.numel())
    mask = torch.zeros_like(scores, dtype=torch.bool)
    _, indices = torch.topk(scores, k=k)
    mask[indices] = True
    return mask

def _calculate_block_value_norms(
    represent_values: Optional[torch.Tensor],
    num_key_value_groups: int,
    block_slices: List[Tuple[int, int]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if represent_values is None or represent_values.shape[2] == 0:
        return None
    values = repeat_kv(represent_values, num_key_value_groups).float()
    norms = []
    for start, num_kernels in block_slices:
        if num_kernels <= 0:
            norms.append(torch.zeros((), device=device, dtype=torch.float32))
            continue
        end = start + num_kernels
        norms.append(values[:, :, start:end, :].norm(dim=-1).mean())
    return torch.stack(norms)

def _calculate_functional_damage_scores(
    q_for_pruning: torch.Tensor,
    kernel_scores: torch.Tensor,
    represent_values: Optional[torch.Tensor],
    num_key_value_groups: int,
    block_slices: List[Tuple[int, int]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if represent_values is None or represent_values.shape[2] == 0:
        return None

    eps = 1e-6
    value_repeated = repeat_kv(represent_values, num_key_value_groups)
    probs = torch.softmax(kernel_scores.float(), dim=-1)
    values = value_repeated.float()
    o_full = torch.einsum("bhqk,bhkd->bhqd", probs, values)

    p_log_p = probs * torch.log(probs.clamp_min(eps))
    total_p_log_p = p_log_p.sum(dim=-1, keepdim=True)
    h_full = -total_p_log_p
    p_sq = probs.pow(2)
    total_p_sq = p_sq.sum(dim=-1, keepdim=True)

    delta_o = []
    delta_h = []
    delta_s = []
    for start, num_kernels in block_slices:
        if num_kernels <= 0:
            zero = torch.zeros((), device=q_for_pruning.device, dtype=torch.float32)
            delta_o.append(zero)
            delta_h.append(zero)
            delta_s.append(zero)
            continue

        end = start + num_kernels
        p_b = probs[..., start:end]
        v_b = values[:, :, start:end, :]
        alpha_b = p_b.sum(dim=-1, keepdim=True).clamp(max=1.0 - eps)
        c_b = torch.einsum("bhqk,bhkd->bhqd", p_b, v_b)
        keep_mass = (1.0 - alpha_b).clamp_min(eps)
        o_minus_b = (o_full - c_b) / keep_mass
        delta_o.append((o_full - o_minus_b).pow(2).mean())

        block_p_log_p = p_log_p[..., start:end].sum(dim=-1, keepdim=True)
        keep_p_log_p = total_p_log_p - block_p_log_p
        h_minus_b = -(keep_p_log_p / keep_mass) + torch.log(keep_mass)
        delta_h.append((h_full - h_minus_b).abs().mean())

        block_p_sq = p_sq[..., start:end].sum(dim=-1, keepdim=True)
        keep_p_sq = total_p_sq - block_p_sq
        prob_l2 = block_p_sq + keep_p_sq * (alpha_b / keep_mass).pow(2)
        delta_s.append(prob_l2.mean())

    return torch.stack(delta_o), torch.stack(delta_h), torch.stack(delta_s)

def _write_pruning_stats(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    final_kept_ids: Set[int],
    block_scores: torch.Tensor,
    delta_o_values: Optional[torch.Tensor],
    delta_h_values: Optional[torch.Tensor],
    delta_s_values: Optional[torch.Tensor],
    block_modes: Optional[Dict[str, int]],
    num_blocks_to_keep: int,
) -> None:
    stats_path = cfg.get("layer_stats_path")
    block_scores_path = cfg.get("block_scores_path")
    if not stats_path and not block_scores_path:
        return
    if cfg.get("stats_mode") == "aggregate" and cfg.get("phase") == "decode":
        return

    selected = sorted(int(x) for x in final_kept_ids)
    record: Dict[str, Any] = {
        "layer_idx": int(cfg.get("layer_idx", -1)),
        "phase": cfg.get("phase", "unknown"),
        "method": cfg.get("variant_method", "sliminfer_original"),
        "scoring": cfg.get("score_kind", cfg.get("scoring", "token_unit")),
        "num_blocks": len(original_block_ids),
        "num_keep_target": int(num_blocks_to_keep),
        "num_selected": len(selected),
        "selected_block_ids": selected,
    }
    if delta_o_values is not None:
        record["mean_delta_o"] = float(delta_o_values.mean().detach().cpu())
        record["mean_delta_h"] = float(delta_h_values.mean().detach().cpu())
        record["mean_delta_s"] = float(delta_s_values.mean().detach().cpu())
    if block_modes is not None:
        record["block_modes"] = block_modes

    if stats_path:
        with open(stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    if block_scores_path:
        if cfg.get("block_scores_mode") == "aggregate":
            score_record = {
                "layer_idx": record["layer_idx"],
                "phase": record["phase"],
                "num_blocks": len(original_block_ids),
                "num_selected": len(selected),
                "score_mean": float(block_scores.detach().float().mean().cpu()),
                "score_max": float(block_scores.detach().float().max().cpu()),
                "score_min": float(block_scores.detach().float().min().cpu()),
                "selected_block_ids": selected,
            }
            if delta_o_values is not None:
                score_record["delta_o_mean"] = float(delta_o_values.mean().detach().cpu())
                score_record["delta_h_mean"] = float(delta_h_values.mean().detach().cpu())
                score_record["delta_s_mean"] = float(delta_s_values.mean().detach().cpu())
            with open(block_scores_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(score_record) + "\n")
            return

        score_record = {
            "layer_idx": record["layer_idx"],
            "phase": record["phase"],
            "block_ids": [int(x) for x in original_block_ids],
            "scores": [float(x) for x in block_scores.detach().float().cpu().tolist()],
            "selected_block_ids": selected,
        }
        if delta_o_values is not None:
            score_record["delta_o"] = [float(x) for x in delta_o_values.detach().float().cpu().tolist()]
            score_record["delta_h"] = [float(x) for x in delta_h_values.detach().float().cpu().tolist()]
            score_record["delta_s"] = [float(x) for x in delta_s_values.detach().float().cpu().tolist()]
        with open(block_scores_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(score_record) + "\n")

def _dc_json_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result

def _dc_score_values(scores: torch.Tensor, original_block_ids: List[int]) -> Dict[int, float]:
    values = scores.detach().float().cpu().tolist()
    return {
        int(block_id): float(values[index])
        for index, block_id in enumerate(original_block_ids)
        if index < len(values) and math.isfinite(float(values[index]))
    }

def _dc_normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    values = scores.detach().float()
    finite = torch.isfinite(values)
    if not bool(finite.any().item()):
        return torch.zeros_like(values)
    finite_values = values[finite]
    min_val = finite_values.min()
    max_val = finite_values.max()
    denom = (max_val - min_val).clamp_min(1e-6)
    normalized = (values - min_val) / denom
    return torch.where(finite, normalized, torch.zeros_like(values))

def _dc_original_topk_ids(
    original_block_ids: List[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
) -> Set[int]:
    top_k_block_ids = {int(original_block_ids[int(i)]) for i in top_k_indices.detach().cpu().tolist()}
    selected = set(top_k_block_ids).union(int(x) for x in must_keep_block_ids)
    if len(selected) > int(num_blocks_to_keep):
        score_map = _dc_score_values(block_scores, original_block_ids)
        removable = list(top_k_block_ids.difference(must_keep_block_ids))
        removable.sort(key=lambda block_id: score_map.get(int(block_id), float("-inf")))
        for block_id in removable[: max(0, len(selected) - int(num_blocks_to_keep))]:
            selected.discard(int(block_id))
    return {int(x) for x in selected}

def _dc_block_text_map(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
) -> Tuple[Dict[int, str], Optional[str]]:
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    block_texts = cfg.get("block_texts", context.get("block_texts"))
    if not isinstance(block_texts, list) or not block_texts:
        return {}, "block_text_unavailable"
    result: Dict[int, str] = {}
    missing = 0
    for block_id in original_block_ids:
        idx = int(block_id)
        if idx < 0 or idx >= len(block_texts):
            missing += 1
            continue
        text = str(block_texts[idx] or "")
        if not text.strip():
            missing += 1
            continue
        result[idx] = text
    if missing:
        return result, "block_text_partial_unavailable"
    return result, None

def _dc_query_text(cfg: Dict[str, Any]) -> str:
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    return str(cfg.get("query_text", context.get("query_text", "")) or "")

def _dc_coverage_key(cfg: Dict[str, Any], block_id: int, num_blocks: int) -> str:
    layout = cfg.get("block_layout") if isinstance(cfg.get("block_layout"), dict) else {}
    source_types = layout.get("source_types") if isinstance(layout.get("source_types"), list) else []
    if 0 <= int(block_id) < len(source_types):
        source = str(source_types[int(block_id)] or "").strip()
        if source:
            return f"source:{source}:pos:{int(block_id) // max(1, math.ceil(max(1, num_blocks) / 8))}"
    bin_count = 8
    denom = max(1, int(num_blocks))
    position_bin = min(bin_count - 1, int((int(block_id) / denom) * bin_count))
    return f"pos:{position_bin}"

def _dc_token_jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left.intersection(right)) / max(1, len(left.union(right))))

def _dc_selected_redundancy(selected: Set[int], token_map: Dict[int, Set[str]]) -> float:
    selected_ids = sorted(int(x) for x in selected)
    if len(selected_ids) <= 1:
        return 0.0
    maxima: List[float] = []
    for block_id in selected_ids:
        tokens = token_map.get(block_id, set())
        best = 0.0
        for other_id in selected_ids:
            if other_id == block_id:
                continue
            best = max(best, _dc_token_jaccard(tokens, token_map.get(other_id, set())))
        maxima.append(best)
    return float(sum(maxima) / max(1, len(maxima)))

def _dc_mean_query_overlap(selected: Set[int], query_overlaps: Dict[int, float]) -> float:
    if not selected:
        return 0.0
    return float(sum(float(query_overlaps.get(int(block_id), 0.0)) for block_id in selected) / max(1, len(selected)))

def _dc_coverage_bins(selected: Set[int], coverage_keys: Dict[int, str]) -> List[str]:
    return sorted({str(coverage_keys.get(int(block_id), "")) for block_id in selected if coverage_keys.get(int(block_id), "")})

def _dc_virtual_text_map(
    cfg: Dict[str, Any],
    virtual_blocks: List[Dict[str, Any]],
) -> Tuple[Dict[int, str], bool]:
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    block_texts = cfg.get("block_texts", context.get("block_texts"))
    if not isinstance(block_texts, list):
        block_texts = []
    result: Dict[int, str] = {}
    for vid, block in enumerate(virtual_blocks):
        text = str(block.get("text_preview", "") or "").strip()
        if not text:
            pieces = []
            for page_id in block.get("fixed_page_ids", []):
                page_id = int(page_id)
                if 0 <= page_id < len(block_texts):
                    piece = str(block_texts[page_id] or "").strip()
                    if piece:
                        pieces.append(piece)
            text = "\n".join(pieces).strip()
        if text:
            result[int(vid)] = text
    return result, bool(result)

def _dc_virtual_coverage_key(block: Dict[str, Any], virtual_id: int, num_blocks: int) -> str:
    source = str(block.get("source_type", "") or "").strip()
    token_start = int(block.get("token_start", virtual_id * _BLOCK_SIZE) or 0)
    max_token = max(
        1,
        max(
            int(block.get("token_end", 0) or 0),
            int(num_blocks) * _BLOCK_SIZE,
        ),
    )
    position_bin = min(7, max(0, int((float(token_start) / float(max_token)) * 8.0)))
    if source:
        return f"source:{source}:pos:{position_bin}"
    return f"pos:{position_bin}"

def _adaptive_dc_v3_flush_deferred_selection_audit(cfg: Dict[str, Any]) -> None:
    pending = cfg.pop("_adaptive_dc_v3_pending_selection_audit", None)
    if not isinstance(pending, dict):
        return
    _write_dc_selection_audit(cfg=cfg, **pending)


def _adaptive_dc_v3_run_virtual_dc_selector(
    cfg: Dict[str, Any],
    virtual_blocks: List[Dict[str, Any]],
    virtual_scores: torch.Tensor,
    original_virtual_selected_ids: Set[int],
    must_keep_virtual_ids: Set[int],
    num_virtual_to_keep: int,
) -> Tuple[Set[int], Dict[str, Any]]:
    defer = cfg.get("variant_method") == "adaptive_dc_sliminfer_v3"
    if defer:
        cfg["_adaptive_dc_v3_defer_selection_audit"] = True
    try:
        return _select_dc_sliminfer_virtual_ids(
            cfg=cfg,
            virtual_blocks=virtual_blocks,
            virtual_scores=virtual_scores,
            original_virtual_selected_ids=original_virtual_selected_ids,
            must_keep_virtual_ids=must_keep_virtual_ids,
            num_virtual_to_keep=num_virtual_to_keep,
        )
    finally:
        if defer:
            cfg["_adaptive_dc_v3_defer_selection_audit"] = False


def _adaptive_dc_v3_run_block_dc_selector(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
) -> Tuple[Set[int], Dict[str, Any]]:
    defer = cfg.get("variant_method") == "adaptive_dc_sliminfer_v3"
    if defer:
        cfg["_adaptive_dc_v3_defer_selection_audit"] = True
    try:
        return _select_dc_sliminfer_ids(
            cfg=cfg,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
        )
    finally:
        if defer:
            cfg["_adaptive_dc_v3_defer_selection_audit"] = False


def _selection_trace_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(
        cfg.get("cc_selection_audit_path")
        or callable(cfg.get("_active_block_trace_emit"))
    )


def _selection_digest_enabled(cfg: Dict[str, Any]) -> bool:
    return callable(cfg.get("_selection_digest_emit"))


def _write_cc_selection_audit(cfg: Dict[str, Any], cc_details: Dict[str, Any]) -> None:
    legacy_audit = bool(cfg.get("cc_selection_audit_path"))
    active_trace = callable(cfg.get("_active_block_trace_emit"))
    if not legacy_audit and not active_trace:
        return
    if str(cfg.get("phase", "")) == "decode" and not active_trace:
        return
    # Defer to unified layer dump (includes silm baseline + selected ids).
    cfg["_acc_pending_details"] = dict(cc_details)


def _write_lightweight_selection_digest(
    cfg: Dict[str, Any],
    *,
    original_block_ids: List[int],
    num_blocks_to_keep: int,
    final_kept_ids: Set[int],
) -> None:
    """Hash final IDs without recomputing any selector tensors."""
    selection_digest_emit = cfg.get("_selection_digest_emit")
    if not callable(selection_digest_emit):
        return
    final = sorted(int(x) for x in final_kept_ids)
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    sink_id = int(original_block_ids[0]) if original_block_ids else None
    phase = str(cfg.get("phase", "") or "")
    runtime_stats = cfg.get("runtime_stats") if isinstance(cfg.get("runtime_stats"), dict) else {}
    prune_layer_count = max(1, len(cfg.get("abs_prune_layers", []) or []))
    decode_step = (
        int(runtime_stats.get("decode_score_calls", 0) or 0) // prune_layer_count
        if phase == "decode"
        else 0
    )
    event_index = int(cfg.get("_active_block_trace_event_index", 0) or 0)
    cfg["_active_block_trace_event_index"] = event_index + 1
    selected_text = ",".join(str(value) for value in final)
    selection_digest_emit({
        "protocol_id": cfg.get("protocol_id"),
        "implementation_revision": cfg.get("implementation_revision"),
        "model_id": cfg.get("model_id"),
        "task": context.get("task"),
        "sample_id": context.get("sample_id"),
        "sample_index": context.get("sample_index"),
        "decode_step": decode_step,
        "selection_event_index": event_index,
        "layer_idx": cfg.get("layer_idx"),
        "abs_layer": cfg.get("abs_layer"),
        "phase": phase,
        "method": context.get("method", cfg.get("variant_method", "")),
        "selected_count": len(final),
        "num_keep_target": int(num_blocks_to_keep),
        "budget_parity": int(len(final) == int(num_blocks_to_keep)),
        "sink_violation": int(sink_id is not None and sink_id not in final_kept_ids),
        "duplicate_violation": int(len(final) != len(final_kept_ids)),
        "selected_ids_sha256": hashlib.sha256(selected_text.encode("ascii")).hexdigest(),
        # For the Original canary, the final active set is the Original baseline.
        # Any disagreement with the independently recomputed full trace fails parity.
        "base_original_ids_sha256": hashlib.sha256(selected_text.encode("ascii")).hexdigest(),
    })


def _write_unified_selection_audit(
    cfg: Dict[str, Any],
    *,
    original_block_ids: List[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    final_kept_ids: Set[int],
) -> None:
    """Dump comparable per-layer selections for any method when audit path is set."""
    legacy_audit_path = cfg.get("cc_selection_audit_path")
    active_trace_emit = cfg.get("_active_block_trace_emit")
    active_trace = callable(active_trace_emit)
    selection_digest_emit = cfg.get("_selection_digest_emit")
    selection_digest = callable(selection_digest_emit)
    if not legacy_audit_path and not active_trace and not selection_digest:
        return
    if str(cfg.get("phase", "")) == "decode" and not active_trace and not selection_digest:
        return
    from bridgekv.structural_sliminfer import _must_keep, _silm_baseline

    must_keep = set(must_keep_block_ids) | _must_keep(original_block_ids)
    silm = _silm_baseline(
        original_block_ids, block_scores, top_k_indices, must_keep, int(num_blocks_to_keep)
    )
    final = {int(x) for x in final_kept_ids}
    pending = cfg.pop("_acc_pending_details", None)
    if not isinstance(pending, dict):
        pending = {}
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    sink_id = int(original_block_ids[0]) if original_block_ids else None
    k = int(num_blocks_to_keep)
    phase = str(cfg.get("phase", "") or "")
    runtime_stats = cfg.get("runtime_stats") if isinstance(cfg.get("runtime_stats"), dict) else {}
    prune_layer_count = max(1, len(cfg.get("abs_prune_layers", []) or []))
    decode_step = (
        int(runtime_stats.get("decode_score_calls", 0) or 0) // prune_layer_count
        if phase == "decode"
        else 0
    )
    event_index = int(cfg.get("_active_block_trace_event_index", 0) or 0)
    cfg["_active_block_trace_event_index"] = event_index + 1
    payload = {
        "protocol_id": cfg.get("protocol_id"),
        "implementation_revision": cfg.get("implementation_revision"),
        "model_id": cfg.get("model_id"),
        "task": context.get("task"),
        "sample_id": context.get("sample_id"),
        "sample_index": context.get("sample_index"),
        "decode_step": decode_step,
        "selection_event_index": event_index,
        "layer_idx": cfg.get("layer_idx"),
        "abs_layer": cfg.get("abs_layer"),
        "phase": phase,
        "method": context.get("method", cfg.get("variant_method", "")),
        "variant_method": cfg.get("variant_method"),
        "selected_block_ids": sorted(final),
        "silm_baseline_ids": sorted(int(x) for x in silm),
        "base_original_active_ids": sorted(int(x) for x in silm),
        "cc_active_ids": sorted(final),
        "active_before_ids": sorted(int(x) for x in silm),
        "candidate_ids": sorted(int(x) for x in original_block_ids),
        "active_after_ids": sorted(final),
        "num_selected": len(final),
        "selected_count": len(final),
        "num_keep_target": k,
        "budget_parity": int(len(final) == k),
        "over_budget": int(max(0, len(final) - k)),
        "sink_kept": int(sink_id in final) if sink_id is not None else 0,
        "sink_violation": int(sink_id is not None and sink_id not in final),
        "churn_vs_silm": round(1.0 - len(final & silm) / max(1, k), 6),
        "jaccard_vs_silm": round(len(final & silm) / max(1, len(final | silm)), 6),
        "added_vs_silm": sorted(final - silm),
        "removed_vs_silm": sorted(silm - final),
        "added_ids": sorted(final - silm),
        "removed_ids": sorted(silm - final),
        "sink_ids": [sink_id] if sink_id is not None else [],
        "selection_time_ms": pending.get("selection_time_ms"),
        "selection_time_source": (
            "cc_selector_runtime" if pending.get("selection_time_ms") is not None
            else "selector_not_entered_or_unavailable"
        ),
        **{k: v for k, v in pending.items() if k not in {"selected_block_ids", "silm_baseline_ids"}},
    }
    if selection_digest:
        selected_text = ",".join(str(value) for value in sorted(final))
        baseline_text = ",".join(str(value) for value in sorted(int(x) for x in silm))
        selection_digest_emit({
            "protocol_id": payload["protocol_id"],
            "implementation_revision": payload["implementation_revision"],
            "model_id": payload["model_id"],
            "task": payload["task"],
            "sample_id": payload["sample_id"],
            "sample_index": payload["sample_index"],
            "decode_step": payload["decode_step"],
            "selection_event_index": payload["selection_event_index"],
            "layer_idx": payload["layer_idx"],
            "abs_layer": payload["abs_layer"],
            "phase": payload["phase"],
            "method": payload["method"],
            "selected_count": payload["selected_count"],
            "num_keep_target": payload["num_keep_target"],
            "budget_parity": payload["budget_parity"],
            "sink_violation": payload["sink_violation"],
            "duplicate_violation": int(len(final) != len(set(final))),
            "selected_ids_sha256": hashlib.sha256(selected_text.encode("ascii")).hexdigest(),
            "base_original_ids_sha256": hashlib.sha256(baseline_text.encode("ascii")).hexdigest(),
        })
    if active_trace:
        trace_level = str(cfg.get("active_block_trace_level", "summary") or "summary")
        trace_payload = dict(payload)
        trace_payload["trace_level"] = trace_level
        trace_payload["trace_source"] = "runtime"
        if trace_level == "summary":
            for field in (
                "selected_block_ids", "silm_baseline_ids", "base_original_active_ids",
                "cc_active_ids", "active_before_ids", "candidate_ids", "active_after_ids",
                "added_vs_silm", "removed_vs_silm", "added_ids", "removed_ids", "sink_ids",
            ):
                trace_payload.pop(field, None)
        active_trace_emit(trace_payload)
    if legacy_audit_path:
        import os as _os
        _os.makedirs(_os.path.dirname(_os.path.abspath(str(legacy_audit_path))) or ".", exist_ok=True)
        with open(str(legacy_audit_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_dc_selection_audit(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    original_selected_ids: Set[int],
    dc_selected_ids: Set[int],
    block_scores: torch.Tensor,
    num_blocks_to_keep: int,
    fallback_used: bool,
    fallback_reason: str,
    token_map: Optional[Dict[int, Set[str]]] = None,
    query_overlaps: Optional[Dict[int, float]] = None,
    coverage_keys: Optional[Dict[int, str]] = None,
    adjusted_scores: Optional[Dict[int, float]] = None,
) -> None:
    if cfg.get("_adaptive_dc_v3_defer_selection_audit"):
        cfg["_adaptive_dc_v3_pending_selection_audit"] = {
            "original_block_ids": original_block_ids,
            "original_selected_ids": original_selected_ids,
            "dc_selected_ids": dc_selected_ids,
            "block_scores": block_scores,
            "num_blocks_to_keep": num_blocks_to_keep,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "token_map": token_map,
            "query_overlaps": query_overlaps,
            "coverage_keys": coverage_keys,
            "adjusted_scores": adjusted_scores,
        }
        return
    audit_path = cfg.get("dc_selection_audit_path")
    if not audit_path:
        return
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    token_map = token_map or {}
    query_overlaps = query_overlaps or {}
    coverage_keys = coverage_keys or {}
    score_map = _dc_score_values(block_scores, original_block_ids)
    original_sorted = sorted(int(x) for x in original_selected_ids)
    dc_sorted = sorted(int(x) for x in dc_selected_ids)
    removed = sorted(set(original_sorted).difference(dc_sorted))
    added = sorted(set(dc_sorted).difference(original_sorted))
    original_bins = _dc_coverage_bins(set(original_sorted), coverage_keys)
    dc_bins = _dc_coverage_bins(set(dc_sorted), coverage_keys)
    record = {
        "task": context.get("task", ""),
        "sample_id": context.get("sample_id", ""),
        "sample_index": context.get("sample_index", ""),
        "method": cfg.get("variant_method", ""),
        "layer_id": int(cfg.get("layer_idx", -1)),
        "phase": cfg.get("phase", ""),
        "num_blocks": int(len(original_block_ids)),
        "num_keep_target": int(num_blocks_to_keep),
        "original_selected_block_ids": original_sorted,
        "dc_selected_block_ids": dc_sorted,
        "removed_vs_original_block_ids": removed,
        "added_vs_original_block_ids": added,
        "selection_changed": bool(set(original_sorted) != set(dc_sorted)),
        "selection_changed_count": int(len(set(original_sorted).symmetric_difference(dc_sorted))),
        "selected_redundancy_original": _dc_selected_redundancy(set(original_sorted), token_map),
        "selected_redundancy_dc": _dc_selected_redundancy(set(dc_sorted), token_map),
        "query_overlap_original": _dc_mean_query_overlap(set(original_sorted), query_overlaps),
        "query_overlap_dc": _dc_mean_query_overlap(set(dc_sorted), query_overlaps),
        "coverage_bins_original": original_bins,
        "coverage_bins_dc": dc_bins,
        "coverage_bin_count_original": int(len(original_bins)),
        "coverage_bin_count_dc": int(len(dc_bins)),
        "budget_preserved": bool(len(dc_sorted) == len(original_sorted)),
        "budget_violation": bool(len(dc_sorted) != len(original_sorted)),
        "fallback_used": bool(fallback_used),
        "fallback_reason": str(fallback_reason or ""),
        "dc_alpha": float(cfg.get("dc_alpha", 0.10)),
        "dc_beta": float(cfg.get("dc_beta", 0.05)),
        "dc_lambda": float(cfg.get("dc_lambda", 0.15)),
        "dc_gamma": float(cfg.get("dc_gamma", 0.05)),
        "selection_behavior": "dc_sliminfer" if not fallback_used else "original_fallback",
        "score_sample_original_selected": [
            {"block_id": int(block_id), "sliminfer_score": _dc_json_float(score_map.get(int(block_id)))}
            for block_id in original_sorted[:32]
        ],
        "score_sample_dc_selected": [
            {
                "block_id": int(block_id),
                "sliminfer_score": _dc_json_float(score_map.get(int(block_id))),
                "dc_adjusted_score": _dc_json_float((adjusted_scores or {}).get(int(block_id))),
                "query_overlap": _dc_json_float(query_overlaps.get(int(block_id), 0.0)),
                "coverage_bin": coverage_keys.get(int(block_id), ""),
            }
            for block_id in dc_sorted[:32]
        ],
    }
    if cfg.get("variant_method") == "adaptive_dc_sliminfer_v3":
        gate_last = cfg.get("_adaptive_dc_v3_gate_last")
        if isinstance(gate_last, dict):
            record["adaptive_dc_v3_gate"] = gate_last
            record["adaptive_gate_triggered"] = bool(gate_last.get("triggered"))
            record["adaptive_gate_reason"] = str(gate_last.get("reason", ""))
            record["evidence_dispersion_score"] = gate_last.get("evidence_dispersion_score")
            record["evidence_dispersion_threshold"] = gate_last.get("evidence_dispersion_threshold")
            record["entity_coverage_gap"] = gate_last.get("entity_coverage_gap")
            record["pruned_query_overlap_max"] = gate_last.get("pruned_query_overlap_max")
            record["selected_query_overlap_mean"] = gate_last.get("selected_query_overlap_mean")
            record["selected_jaccard_vs_original"] = gate_last.get("selected_jaccard_vs_original")
            record["adaptive_gate_feature_source"] = gate_last.get("feature_source")
            record["chosen_selector"] = "dc" if gate_last.get("triggered") else "original"
    try:
        path = str(audit_path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _dc_activation_record(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    original_selected_ids: Set[int],
    dc_selected_ids: Set[int],
    num_blocks_to_keep: int,
    *,
    entered_dc_selector: bool,
    early_return_reason: str = "",
    virtual_return_reason: str = "",
    must_keep_return_reason: str = "",
    fallback_reason: str = "",
    token_map: Optional[Dict[int, Set[str]]] = None,
    query_overlaps: Optional[Dict[int, float]] = None,
    coverage_keys: Optional[Dict[int, str]] = None,
    keep_all: Optional[bool] = None,
    entered_virtual_block_path: bool = False,
    entered_dc_selector_before_virtual_return: bool = False,
    virtual_block_path_bypassed_by_dc: bool = False,
    candidate_block_ids_available: Optional[bool] = None,
    candidate_scores_available: Optional[bool] = None,
    candidate_text_available: Optional[bool] = None,
    query_text_available: Optional[bool] = None,
    original_virtual_selected_ids: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    token_map = token_map or {}
    query_overlaps = query_overlaps or {}
    coverage_keys = coverage_keys or {}
    original_sorted = sorted(int(x) for x in original_selected_ids)
    dc_sorted = sorted(int(x) for x in dc_selected_ids)
    original_virtual_sorted = sorted(int(x) for x in (original_virtual_selected_ids or original_selected_ids))
    num_blocks = int(len(original_block_ids))
    keep_target = int(num_blocks_to_keep)
    if keep_all is None:
        keep_all = bool(keep_target >= num_blocks)
    selection_pressure = bool(keep_target < num_blocks)
    budget_preserved = bool(len(dc_sorted) == len(original_sorted))
    return {
        "task": context.get("task", ""),
        "sample_index": context.get("sample_index", ""),
        "sample_id": context.get("sample_id", ""),
        "layer_id": int(cfg.get("layer_idx", -1)),
        "method": cfg.get("variant_method", ""),
        "phase": cfg.get("phase", ""),
        "dc_flag_received": bool(cfg.get("variant_method") in ("dc_sliminfer", "adaptive_dc_sliminfer_v3")),
        "entered_pruning_layer": True,
        "entered_virtual_block_path": bool(entered_virtual_block_path),
        "entered_dc_selector": bool(entered_dc_selector),
        "entered_dc_selector_before_virtual_return": bool(entered_dc_selector_before_virtual_return),
        "virtual_block_path_bypassed_by_dc": bool(virtual_block_path_bypassed_by_dc),
        "audit_row_written": True,
        "num_blocks": num_blocks,
        "num_keep_target": keep_target,
        "num_pruned_candidates": max(0, num_blocks - keep_target),
        "keep_all": bool(keep_all),
        "early_return_reason": str(early_return_reason or ""),
        "virtual_return_reason": str(virtual_return_reason or ""),
        "must_keep_return_reason": str(must_keep_return_reason or ""),
        "selection_pressure": selection_pressure,
        "candidate_block_ids_available": bool(num_blocks > 0) if candidate_block_ids_available is None else bool(candidate_block_ids_available),
        "candidate_scores_available": True if candidate_scores_available is None else bool(candidate_scores_available),
        "candidate_text_available": bool(token_map) if candidate_text_available is None else bool(candidate_text_available),
        "query_text_available": bool(_dc_query_text(cfg).strip()) if query_text_available is None else bool(query_text_available),
        "original_virtual_selected_block_ids": original_virtual_sorted,
        "original_selected_block_ids": original_sorted,
        "dc_selected_block_ids": dc_sorted,
        "selection_changed": bool(set(original_sorted) != set(dc_sorted)),
        "selected_jaccard": _jaccard(set(original_sorted), set(dc_sorted)),
        "original_redundancy": _dc_selected_redundancy(set(original_sorted), token_map),
        "dc_redundancy": _dc_selected_redundancy(set(dc_sorted), token_map),
        "original_query_overlap": _dc_mean_query_overlap(set(original_sorted), query_overlaps),
        "dc_query_overlap": _dc_mean_query_overlap(set(dc_sorted), query_overlaps),
        "original_coverage_bins": _dc_coverage_bins(set(original_sorted), coverage_keys),
        "dc_coverage_bins": _dc_coverage_bins(set(dc_sorted), coverage_keys),
        "budget_preserved": budget_preserved,
        "fallback_reason": str(fallback_reason or ""),
    }

def _set_dc_activation_state(
    cfg: Dict[str, Any],
    record: Dict[str, Any],
) -> None:
    if cfg.get("dc_activation_audit_path"):
        cfg["_dc_activation_last"] = record

def _write_dc_activation_audit_record(cfg: Dict[str, Any]) -> None:
    audit_path = cfg.get("dc_activation_audit_path")
    if not audit_path:
        return
    record = cfg.get("_dc_activation_last")
    if not isinstance(record, dict):
        context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
        record = {
            "task": context.get("task", ""),
            "sample_index": context.get("sample_index", ""),
            "sample_id": context.get("sample_id", ""),
            "layer_id": int(cfg.get("layer_idx", -1)),
            "method": cfg.get("variant_method", ""),
            "phase": cfg.get("phase", ""),
            "dc_flag_received": bool(cfg.get("variant_method") in ("dc_sliminfer", "adaptive_dc_sliminfer_v3")),
            "entered_pruning_layer": True,
            "entered_virtual_block_path": False,
            "entered_dc_selector": False,
            "entered_dc_selector_before_virtual_return": False,
            "virtual_block_path_bypassed_by_dc": False,
            "audit_row_written": True,
            "num_blocks": 0,
            "num_keep_target": 0,
            "num_pruned_candidates": 0,
            "keep_all": False,
            "early_return_reason": "missing_cache_activation_state",
            "virtual_return_reason": "",
            "must_keep_return_reason": "",
            "selection_pressure": False,
            "candidate_block_ids_available": False,
            "candidate_scores_available": False,
            "candidate_text_available": False,
            "query_text_available": bool(_dc_query_text(cfg).strip()),
            "original_virtual_selected_block_ids": [],
            "original_selected_block_ids": [],
            "dc_selected_block_ids": [],
            "selection_changed": False,
            "selected_jaccard": 1.0,
            "original_redundancy": 0.0,
            "dc_redundancy": 0.0,
            "original_query_overlap": 0.0,
            "dc_query_overlap": 0.0,
            "original_coverage_bins": [],
            "dc_coverage_bins": [],
            "budget_preserved": False,
            "fallback_reason": "",
        }
    try:
        with open(str(audit_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

_ADAPTIVE_DC_V3_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me",
    "more", "most", "my", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "out", "over", "own", "same", "she", "so",
    "some", "such", "than", "that", "the", "their", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your",
}


def _adaptive_dc_v3_text_tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in re.finditer(r"[A-Za-z][A-Za-z0-9_\-']*", str(text or ""))]


def _adaptive_dc_v3_keyword_set(text: str) -> Set[str]:
    words = {
        t.strip("'").lower()
        for t in _adaptive_dc_v3_text_tokens(text)
        if len(t.strip("'")) >= 3 and t.lower() not in _ADAPTIVE_DC_V3_STOPWORDS
    }
    return {w for w in words if w}


def _adaptive_dc_v3_extract_query_keywords(query: str) -> List[str]:
    keywords = sorted(_adaptive_dc_v3_keyword_set(query))
    for match in re.finditer(r"(?:[A-Z][A-Za-z0-9'.-]+(?:\s+|$)){1,5}", query):
        phrase = " ".join(match.group(0).split()).strip()
        if phrase and phrase.lower() not in _ADAPTIVE_DC_V3_STOPWORDS:
            keywords = sorted(set(keywords) | {
                t for t in _adaptive_dc_v3_text_tokens(phrase)
                if t not in _ADAPTIVE_DC_V3_STOPWORDS
            })
    return keywords


def _adaptive_dc_v3_lexical_overlap(text: str, query_keywords: List[str]) -> float:
    if not query_keywords:
        return 0.0
    tokens = _adaptive_dc_v3_keyword_set(text)
    if not tokens:
        return 0.0
    hits = sum(1 for kw in query_keywords if kw.lower() in tokens)
    return hits / len(set(query_keywords))


def _adaptive_dc_v3_coverage_ratio(texts: List[str], query_keywords: List[str]) -> float:
    if not query_keywords:
        return 0.0
    joined: Set[str] = set()
    for text in texts:
        joined.update(_adaptive_dc_v3_keyword_set(text))
    hits = sum(1 for kw in query_keywords if kw.lower() in joined)
    return hits / len(set(query_keywords))


def _adaptive_dc_v3_doc_proxy(block_id: int, block_text: str, num_blocks: int) -> str:
    matches = re.findall(r"\bPassage\s+(\d+)\s*:", block_text, flags=re.IGNORECASE)
    if matches:
        return f"passage:{matches[-1]}"
    bins = 8
    width = max(1, math.ceil(max(1, num_blocks) / bins))
    return f"posbin:{min(bins - 1, int(block_id) // width)}"


def _adaptive_dc_v3_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            value -= p * math.log(p)
    return value


def _adaptive_dc_v3_minmax_norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return (value - lo) / (hi - lo)


def _adaptive_dc_v3_compute_gate(
    cfg: Dict[str, Any],
    original_selected_block_ids: Set[int],
    dc_selected_block_ids: Set[int],
    num_blocks: int,
) -> Dict[str, Any]:
    gate_cfg = cfg.get("adaptive_dc_v3_gate") or {}
    threshold = gate_cfg.get("evidence_dispersion_threshold")
    if threshold is None:
        return {
            "triggered": False,
            "reason": "frozen_threshold_missing",
            "evidence_dispersion_score": None,
            "evidence_dispersion_threshold": None,
            "entity_coverage_gap": None,
            "pruned_query_overlap_max": None,
            "selected_query_overlap_mean": None,
            "selected_jaccard_vs_original": None,
            "bug_stop": True,
        }
    threshold = float(threshold)

    frozen_features = gate_cfg.get("frozen_sample_features") or {}
    ctx = cfg.get("audit_context") or {}
    task = str(ctx.get("task", ""))
    sample_index = ctx.get("sample_index")
    sample_key = f"{task}:{sample_index}" if task and sample_index is not None else None

    if sample_key and sample_key in frozen_features:
        sf = frozen_features[sample_key]
        eds = float(sf.get("evidence_dispersion_score", 0))
        ecg = float(sf.get("entity_coverage_gap", -1))
        pqo = float(sf.get("pruned_query_overlap_max", 0))
        sqm = float(sf.get("selected_query_overlap_mean", 1))
        jac = float(sf.get("selected_jaccard_vs_original", 0))

        conditions = {
            "evidence_dispersion_score_ge_threshold": eds >= threshold,
            "entity_coverage_gap_ge_zero": ecg >= 0.0,
            "pruned_query_overlap_max_gt_selected_mean": pqo > sqm,
            "selected_jaccard_vs_original_ge_030": jac >= 0.30,
        }
        triggered = all(conditions.values())
        reasons = [f"{k}=True" for k in conditions] if triggered else [f"{k}={v}" for k, v in conditions.items() if not v]
        return {
            "triggered": triggered,
            "reason": "; ".join(reasons),
            "evidence_dispersion_score": eds,
            "evidence_dispersion_threshold": threshold,
            "entity_coverage_gap": ecg,
            "pruned_query_overlap_max": pqo,
            "selected_query_overlap_mean": sqm,
            "selected_jaccard_vs_original": jac,
            "conditions": conditions,
            "feature_source": "frozen_v3_audit",
            "sample_key": sample_key,
            "bug_stop": False,
        }

    block_texts = cfg.get("block_texts") or (cfg.get("audit_context") or {}).get("block_texts") or []
    query_text = cfg.get("query_text") or (cfg.get("audit_context") or {}).get("query_text") or ""
    norm_min = gate_cfg.get("norm_min") or {}
    norm_max = gate_cfg.get("norm_max") or {}
    top_pruned_m = int(gate_cfg.get("top_pruned_m", 16) or 16)

    dc_set = set(int(x) for x in dc_selected_block_ids)
    orig_set = set(int(x) for x in original_selected_block_ids)
    union = dc_set | orig_set
    jaccard = float(len(dc_set & orig_set) / len(union)) if union else 1.0

    if not block_texts:
        return {
            "triggered": False,
            "reason": "runtime_features_unavailable_no_block_texts",
            "evidence_dispersion_score": None,
            "evidence_dispersion_threshold": threshold,
            "entity_coverage_gap": None,
            "pruned_query_overlap_max": None,
            "selected_query_overlap_mean": None,
            "selected_jaccard_vs_original": jaccard,
            "conditions": {},
            "feature_source": "unavailable",
            "sample_key": sample_key,
            "bug_stop": False,
        }

    query_keywords = _adaptive_dc_v3_extract_query_keywords(str(query_text))
    total_blocks = min(int(num_blocks), len(block_texts))
    all_block_ids = list(range(total_blocks))
    selected_set = {int(bid) for bid in orig_set if 0 <= int(bid) < len(block_texts)}
    if not selected_set:
        selected_set = {int(bid) for bid in all_block_ids[: min(len(all_block_ids), 64)]}
    pruned_ids = [bid for bid in all_block_ids if bid not in selected_set]

    selected_texts = [str(block_texts[bid]) for bid in sorted(selected_set)]
    pruned_overlap_pairs = [
        (bid, _adaptive_dc_v3_lexical_overlap(str(block_texts[bid]), query_keywords))
        for bid in pruned_ids
    ]
    top_pruned = sorted(pruned_overlap_pairs, key=lambda item: (-item[1], item[0]))[:top_pruned_m]
    top_pruned_texts = [str(block_texts[bid]) for bid, _ in top_pruned]
    selected_overlaps = [_adaptive_dc_v3_lexical_overlap(text, query_keywords) for text in selected_texts]

    doc_counts: Dict[str, int] = {}
    for bid in selected_set:
        doc_key = _adaptive_dc_v3_doc_proxy(bid, str(block_texts[bid]), max(1, total_blocks))
        doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
    doc_entropy = _adaptive_dc_v3_entropy(doc_counts)

    selected_query_overlap_mean = (sum(selected_overlaps) / len(selected_overlaps)) if selected_overlaps else 0.0
    pruned_query_overlap_max = max((overlap for _, overlap in top_pruned), default=0.0)
    selected_entity_coverage = _adaptive_dc_v3_coverage_ratio(selected_texts, query_keywords)
    pruned_entity_coverage = _adaptive_dc_v3_coverage_ratio(top_pruned_texts, query_keywords)
    entity_coverage_gap = pruned_entity_coverage - selected_entity_coverage

    doc_entropy_norm = _adaptive_dc_v3_minmax_norm(
        doc_entropy,
        float(norm_min.get("original_doc_entropy", doc_entropy)),
        float(norm_max.get("original_doc_entropy", doc_entropy)),
    )
    ecg_norm = _adaptive_dc_v3_minmax_norm(
        entity_coverage_gap,
        float(norm_min.get("entity_coverage_gap", entity_coverage_gap)),
        float(norm_max.get("entity_coverage_gap", entity_coverage_gap)),
    )
    pqo_norm = _adaptive_dc_v3_minmax_norm(
        pruned_query_overlap_max,
        float(norm_min.get("pruned_query_overlap_max", pruned_query_overlap_max)),
        float(norm_max.get("pruned_query_overlap_max", pruned_query_overlap_max)),
    )
    evidence_dispersion_score = (doc_entropy_norm + ecg_norm + pqo_norm) / 3.0

    conditions = {
        "evidence_dispersion_score_ge_threshold": evidence_dispersion_score >= threshold,
        "entity_coverage_gap_ge_zero": entity_coverage_gap >= 0.0,
        "pruned_query_overlap_max_gt_selected_mean": pruned_query_overlap_max > selected_query_overlap_mean,
        "selected_jaccard_vs_original_ge_030": jaccard >= 0.30,
    }
    triggered = all(conditions.values())
    reasons = [f"{k}=True" for k in conditions] if triggered else [f"{k}={v}" for k, v in conditions.items() if not v]

    return {
        "triggered": triggered,
        "reason": "; ".join(reasons),
        "evidence_dispersion_score": float(evidence_dispersion_score),
        "evidence_dispersion_threshold": threshold,
        "entity_coverage_gap": float(entity_coverage_gap),
        "pruned_query_overlap_max": float(pruned_query_overlap_max),
        "selected_query_overlap_mean": float(selected_query_overlap_mean),
        "selected_jaccard_vs_original": float(jaccard),
        "doc_entropy": float(doc_entropy),
        "doc_entropy_norm": float(doc_entropy_norm),
        "ecg_norm": float(ecg_norm),
        "pqo_norm": float(pqo_norm),
        "conditions": conditions,
        "feature_source": "runtime_frozen_norm",
        "sample_key": sample_key,
        "bug_stop": False,
    }


def _select_dc_sliminfer_virtual_ids(
    cfg: Dict[str, Any],
    virtual_blocks: List[Dict[str, Any]],
    virtual_scores: torch.Tensor,
    original_virtual_selected_ids: Set[int],
    must_keep_virtual_ids: Set[int],
    num_virtual_to_keep: int,
) -> Tuple[Set[int], Dict[str, Any]]:
    virtual_ids = [int(i) for i in range(len(virtual_blocks))]
    original_selected = {int(x) for x in original_virtual_selected_ids if 0 <= int(x) < len(virtual_blocks)}
    keep_target = min(max(1, int(num_virtual_to_keep)), len(virtual_blocks))
    target_count = len(original_selected) if original_selected else keep_target
    text_map_raw, text_available = _dc_virtual_text_map(cfg, virtual_blocks)
    token_map = {int(vid): _gate_content_tokens(text_map_raw.get(int(vid), "")) for vid in virtual_ids}
    query_tokens = _gate_content_tokens(_dc_query_text(cfg))
    query_overlaps: Dict[int, float] = {}
    for vid in virtual_ids:
        tokens = token_map.get(int(vid), set())
        query_overlaps[int(vid)] = (
            float(len(tokens.intersection(query_tokens)) / max(1, len(query_tokens)))
            if query_tokens else 0.0
        )
    coverage_keys = {
        int(vid): _dc_virtual_coverage_key(block, int(vid), max(1, len(virtual_blocks)))
        for vid, block in enumerate(virtual_blocks)
    }
    score_len_ok = bool(virtual_scores is not None and int(virtual_scores.numel()) >= len(virtual_blocks))
    finite_scores = torch.isfinite(virtual_scores.detach().float()) if score_len_ok else torch.zeros(0, dtype=torch.bool)
    scores_available = bool(score_len_ok and finite_scores[: len(virtual_blocks)].any().item())
    if not virtual_ids or not scores_available:
        fallback_reason = "candidate_or_sliminfer_score_unavailable"
        _set_dc_activation_state(
            cfg,
            _dc_activation_record(
                cfg=cfg,
                original_block_ids=virtual_ids,
                original_selected_ids=original_selected,
                original_virtual_selected_ids=original_selected,
                dc_selected_ids=original_selected,
                num_blocks_to_keep=num_virtual_to_keep,
                entered_dc_selector=True,
                entered_virtual_block_path=True,
                entered_dc_selector_before_virtual_return=True,
                virtual_block_path_bypassed_by_dc=False,
                fallback_reason=fallback_reason,
                token_map=token_map,
                query_overlaps=query_overlaps,
                coverage_keys=coverage_keys,
                candidate_block_ids_available=bool(virtual_ids),
                candidate_scores_available=scores_available,
                candidate_text_available=text_available,
                query_text_available=bool(query_tokens),
            ),
        )
        _write_dc_selection_audit(
            cfg=cfg,
            original_block_ids=virtual_ids,
            original_selected_ids=original_selected,
            dc_selected_ids=original_selected,
            block_scores=virtual_scores.detach().float() if score_len_ok else torch.zeros(len(virtual_ids), device=virtual_scores.device if virtual_scores is not None else "cpu"),
            num_blocks_to_keep=num_virtual_to_keep,
            fallback_used=True,
            fallback_reason=fallback_reason,
            token_map=token_map,
            query_overlaps=query_overlaps,
            coverage_keys=coverage_keys,
        )
        return original_selected, {
            "selection_policy": "dc_sliminfer_virtual",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "jaccard_vs_base": 1.0,
        }

    alpha = float(cfg.get("dc_alpha", 0.10))
    beta = float(cfg.get("dc_beta", 0.05))
    lam = float(cfg.get("dc_lambda", 0.15))
    gamma = float(cfg.get("dc_gamma", 0.05))
    scores = virtual_scores.detach().float()[: len(virtual_blocks)]
    base_norm = _dc_normalize_scores(scores)
    base_values = base_norm.detach().float().cpu().tolist()
    base_raw = scores.detach().float().cpu().tolist()
    selected: Set[int] = {int(x) for x in must_keep_virtual_ids if 0 <= int(x) < len(virtual_blocks)}
    if len(selected) > target_count:
        score_map = {int(vid): float(base_raw[int(vid)]) for vid in virtual_ids if int(vid) < len(base_raw)}
        removable = sorted(selected, key=lambda vid: score_map.get(int(vid), float("-inf")))
        for vid in removable[: max(0, len(selected) - target_count)]:
            selected.discard(int(vid))

    adjusted_scores: Dict[int, float] = {}
    high_score_threshold = 0.75
    while len(selected) < target_count:
        selected_coverage = {coverage_keys.get(int(vid), "") for vid in selected}
        high_selected = {
            int(vid)
            for vid in selected
            if 0 <= int(vid) < len(base_values) and float(base_values[int(vid)]) >= high_score_threshold
        }
        best_vid: Optional[int] = None
        best_tuple: Optional[Tuple[float, float, float, int]] = None
        for vid in virtual_ids:
            if vid in selected:
                continue
            base_score = float(base_values[vid])
            coverage_bonus = 0.0 if coverage_keys.get(vid, "") in selected_coverage else 1.0
            redundancy = 0.0
            if selected:
                redundancy = max(
                    _dc_token_jaccard(token_map.get(vid, set()), token_map.get(int(other_id), set()))
                    for other_id in selected
                )
            neighbor_bonus = 1.0 if any(abs(vid - other_id) == 1 for other_id in high_selected) else 0.0
            adjusted = (
                base_score
                + alpha * float(query_overlaps.get(vid, 0.0))
                + beta * coverage_bonus
                - lam * redundancy
                + gamma * neighbor_bonus
            )
            adjusted_scores[int(vid)] = float(adjusted)
            raw_score = float(base_raw[vid]) if vid < len(base_raw) and math.isfinite(float(base_raw[vid])) else float("-inf")
            candidate_tuple = (float(adjusted), raw_score, -float(vid), -int(vid))
            if best_tuple is None or candidate_tuple > best_tuple:
                best_tuple = candidate_tuple
                best_vid = int(vid)
        if best_vid is None:
            break
        selected.add(int(best_vid))

    fallback_used = False
    fallback_reason = ""
    if len(selected) != len(original_selected):
        selected = set(original_selected)
        fallback_used = True
        fallback_reason = "budget_mismatch_fallback_original_virtual"

    _set_dc_activation_state(
        cfg,
        _dc_activation_record(
            cfg=cfg,
            original_block_ids=virtual_ids,
            original_selected_ids=original_selected,
            original_virtual_selected_ids=original_selected,
            dc_selected_ids=selected,
            num_blocks_to_keep=num_virtual_to_keep,
            entered_dc_selector=True,
            entered_virtual_block_path=True,
            entered_dc_selector_before_virtual_return=True,
            virtual_block_path_bypassed_by_dc=not fallback_used,
            fallback_reason=fallback_reason,
            token_map=token_map,
            query_overlaps=query_overlaps,
            coverage_keys=coverage_keys,
            candidate_block_ids_available=True,
            candidate_scores_available=True,
            candidate_text_available=text_available,
            query_text_available=bool(query_tokens),
        ),
    )
    _write_dc_selection_audit(
        cfg=cfg,
        original_block_ids=virtual_ids,
        original_selected_ids=original_selected,
        dc_selected_ids=selected,
        block_scores=scores,
        num_blocks_to_keep=num_virtual_to_keep,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        token_map=token_map,
        query_overlaps=query_overlaps,
        coverage_keys=coverage_keys,
        adjusted_scores=adjusted_scores,
    )
    return selected, {
        "selection_policy": "dc_sliminfer_virtual",
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "selected_by_dc_count": int(len(set(selected).difference(original_selected))),
        "replaced_by_dc_count": int(len(set(original_selected).difference(selected))),
        "jaccard_vs_base": _jaccard(set(selected), original_selected),
    }

def _select_dc_sliminfer_ids(
    cfg: Dict[str, Any],
    original_block_ids: List[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
) -> Tuple[Set[int], Dict[str, Any]]:
    original_selected = _dc_original_topk_ids(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
    )
    text_map, text_error = _dc_block_text_map(cfg, original_block_ids)
    text_available = text_error is None

    finite_scores = torch.isfinite(block_scores.detach().float())
    if not original_block_ids or not bool(finite_scores.any().item()):
        fallback_reason = "candidate_or_sliminfer_score_unavailable"
        _set_dc_activation_state(
            cfg,
            _dc_activation_record(
                cfg=cfg,
                original_block_ids=original_block_ids,
                original_selected_ids=original_selected,
                dc_selected_ids=original_selected,
                num_blocks_to_keep=num_blocks_to_keep,
                entered_dc_selector=True,
                fallback_reason=fallback_reason,
                candidate_block_ids_available=bool(original_block_ids),
                candidate_scores_available=bool(finite_scores.any().item()),
                candidate_text_available=text_available,
            ),
        )
        _write_dc_selection_audit(
            cfg=cfg,
            original_block_ids=original_block_ids,
            original_selected_ids=original_selected,
            dc_selected_ids=original_selected,
            block_scores=block_scores,
            num_blocks_to_keep=num_blocks_to_keep,
            fallback_used=True,
            fallback_reason=fallback_reason,
        )
        return original_selected, {
            "selection_policy": "dc_sliminfer",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "jaccard_vs_base": 1.0,
        }

    alpha = float(cfg.get("dc_alpha", 0.10))
    beta = float(cfg.get("dc_beta", 0.05))
    lam = float(cfg.get("dc_lambda", 0.15))
    gamma = float(cfg.get("dc_gamma", 0.05))
    keep_target = min(max(1, int(num_blocks_to_keep)), len(original_block_ids))
    if len(must_keep_block_ids) >= keep_target:
        selected = {int(x) for x in must_keep_block_ids}
        _set_dc_activation_state(
            cfg,
            _dc_activation_record(
                cfg=cfg,
                original_block_ids=original_block_ids,
                original_selected_ids=original_selected,
                dc_selected_ids=selected,
                num_blocks_to_keep=num_blocks_to_keep,
                entered_dc_selector=True,
                must_keep_return_reason="must_keep_count_reaches_budget",
                candidate_text_available=text_available,
            ),
        )
        _write_dc_selection_audit(
            cfg=cfg,
            original_block_ids=original_block_ids,
            original_selected_ids=original_selected,
            dc_selected_ids=selected,
            block_scores=block_scores,
            num_blocks_to_keep=num_blocks_to_keep,
            fallback_used=False,
            fallback_reason="",
        )
        return selected, {"selection_policy": "dc_sliminfer", "fallback_used": False, "fallback_reason": ""}

    base_norm = _dc_normalize_scores(block_scores)
    base_values = base_norm.detach().float().cpu().tolist()
    base_raw = block_scores.detach().float().cpu().tolist()
    block_index = {int(block_id): index for index, block_id in enumerate(original_block_ids)}
    token_map = {int(block_id): _gate_content_tokens(text_map.get(int(block_id), "")) for block_id in original_block_ids}
    query_tokens = _gate_content_tokens(_dc_query_text(cfg))
    query_overlaps: Dict[int, float] = {}
    for block_id in original_block_ids:
        tokens = token_map.get(int(block_id), set())
        query_overlaps[int(block_id)] = (
            float(len(tokens.intersection(query_tokens)) / max(1, len(query_tokens)))
            if query_tokens else 0.0
        )
    coverage_keys = {
        int(block_id): _dc_coverage_key(cfg, int(block_id), max(1, len(original_block_ids)))
        for block_id in original_block_ids
    }

    selected: Set[int] = {int(x) for x in must_keep_block_ids}
    adjusted_scores: Dict[int, float] = {}
    high_score_threshold = 0.75
    while len(selected) < keep_target:
        selected_coverage = {coverage_keys.get(int(block_id), "") for block_id in selected}
        high_selected = {
            int(block_id)
            for block_id in selected
            if int(block_id) in block_index and float(base_values[block_index[int(block_id)]]) >= high_score_threshold
        }
        best_block: Optional[int] = None
        best_tuple: Optional[Tuple[float, float, float, int]] = None
        for block_id in original_block_ids:
            block_id = int(block_id)
            if block_id in selected:
                continue
            idx = block_index[block_id]
            base_score = float(base_values[idx])
            coverage_bonus = 0.0 if coverage_keys.get(block_id, "") in selected_coverage else 1.0
            redundancy = 0.0
            if selected:
                redundancy = max(
                    _dc_token_jaccard(token_map.get(block_id, set()), token_map.get(int(other_id), set()))
                    for other_id in selected
                )
            neighbor_bonus = 1.0 if any(abs(block_id - other_id) == 1 for other_id in high_selected) else 0.0
            adjusted = (
                base_score
                + alpha * float(query_overlaps.get(block_id, 0.0))
                + beta * coverage_bonus
                - lam * redundancy
                + gamma * neighbor_bonus
            )
            adjusted_scores[block_id] = float(adjusted)
            raw_score = float(base_raw[idx]) if idx < len(base_raw) and math.isfinite(float(base_raw[idx])) else float("-inf")
            candidate_tuple = (float(adjusted), raw_score, -float(block_id), -idx)
            if best_tuple is None or candidate_tuple > best_tuple:
                best_tuple = candidate_tuple
                best_block = block_id
        if best_block is None:
            break
        selected.add(int(best_block))

    fallback_used = False
    fallback_reason = ""
    if len(selected) != len(original_selected):
        selected = set(original_selected)
        fallback_used = True
        fallback_reason = "budget_mismatch_fallback_original"

    _set_dc_activation_state(
        cfg,
        _dc_activation_record(
            cfg=cfg,
            original_block_ids=original_block_ids,
            original_selected_ids=original_selected,
            dc_selected_ids=selected,
            num_blocks_to_keep=num_blocks_to_keep,
            entered_dc_selector=True,
            fallback_reason=fallback_reason,
            token_map=token_map,
            query_overlaps=query_overlaps,
            coverage_keys=coverage_keys,
            candidate_text_available=text_available,
        ),
    )
    _write_dc_selection_audit(
        cfg=cfg,
        original_block_ids=original_block_ids,
        original_selected_ids=original_selected,
        dc_selected_ids=selected,
        block_scores=block_scores,
        num_blocks_to_keep=num_blocks_to_keep,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        token_map=token_map,
        query_overlaps=query_overlaps,
        coverage_keys=coverage_keys,
        adjusted_scores=adjusted_scores,
    )
    return selected, {
        "selection_policy": "dc_sliminfer",
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "selected_by_dc_count": int(len(set(selected).difference(original_selected))),
        "replaced_by_dc_count": int(len(set(original_selected).difference(selected))),
        "jaccard_vs_base": _jaccard(set(selected), original_selected),
    }

def _create_sliminfer_attention_mask(
    query_indices_abs: List[int],
    kv_indices_abs: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """        
    create causal attention mask in a vectorized way.
    Args:
        query_indices_abs (List[int]): list of absolute indices for query tokens
        kv_indices_abs (List[int]): list of absolute indices for key/value tokens
        device (torch.device): device for the returned tensor
        dtype (torch.dtype): data type for the returned tensor
        
    Returns:
        torch.Tensor: attention mask of shape [1, 1, num_q_tokens, num_kv_tokens]
    """
    num_q = len(query_indices_abs)
    num_kv = len(kv_indices_abs)
    
    if num_q == 0 or num_kv == 0:
        return torch.empty((1, 1, num_q, num_kv), device=device, dtype=dtype)
    
    q_pos = torch.tensor(query_indices_abs, device=device).view(-1, 1)  # [num_q, 1]
    kv_pos = torch.tensor(kv_indices_abs, device=device).view(1, -1)    # [1, num_kv]
    
    # vectorized  comparison: generate a [num_q, num_kv] boolean matrix
    mask = kv_pos <= q_pos
    float_mask = torch.where(mask, 
                           torch.zeros((), device=device, dtype=dtype),
                           torch.full((), -torch.inf, device=device, dtype=dtype))
    
    return float_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_q, num_kv]


_QDC_RELATION_WORDS = {
    "where", "when", "who", "whom", "whose", "which", "what", "how",
    "born", "died", "directed", "founded", "located", "nationality", "capital",
    "headquartered", "married", "parent", "child", "author", "written",
    "released", "published", "formed", "member", "leader", "president",
    "governor", "mayor", "country", "city", "state", "province", "county",
}
_QDC_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "that", "this", "these", "those",
    "there", "their", "they", "them", "then", "than", "such", "into", "through",
    "during", "before", "after", "above", "below", "between", "under", "again",
    "further", "once", "here", "why", "all", "each", "few", "more", "most",
    "other", "some", "any", "both", "same", "own", "only", "also", "very", "just",
    "about", "many", "much", "first", "last", "new", "old", "one", "two", "three",
}


def _qdc_is_gate_layer(cfg: Dict[str, Any]) -> bool:
    gate_apply_layer = int(cfg.get("qdc_gate_apply_layer", cfg.get("gate_apply_layer", 26)) or 26)
    current_layer = int(cfg.get("layer_idx", 0) or 0)
    return gate_apply_layer <= 0 or current_layer >= gate_apply_layer


def _qdc_extract_query_components(query: str) -> List[str]:
    components: List[str] = []
    seen: Set[str] = set()

    def add(kind: str, value: str) -> None:
        text = " ".join(str(value).strip().lower().split())
        if not text or len(text) > 120:
            return
        key = f"{kind}:{text}"
        if key not in seen:
            seen.add(key)
            components.append(key)

    for entity in _gate_entity_like_units(query):
        add("entity", entity)
    for match in re.findall(r"\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\b", query):
        add("number", match.replace(",", ""))
    for match in re.findall(r"\b(?:19|20)\d{2}\b", query):
        add("date", match)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", str(query or "").lower()):
        if len(token) >= 4 and token not in _QDC_STOPWORDS and not token.isdigit():
            add("noun", token)
    for word in _QDC_RELATION_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", str(query or "").lower()):
            add("constraint", word)
    for match in re.findall(
        r"\b(?:where|when|who|which|what|how)\s+(?:was|is|are|were|did|does|do)\b",
        str(query or "").lower(),
    ):
        add("wh", match.strip())
    return components


def _qdc_component_in_text(component: str, text: str) -> bool:
    kind, _, value = component.partition(":")
    haystack = str(text or "").lower()
    if not value:
        return False
    if kind in ("entity", "noun", "constraint", "wh"):
        if value in haystack:
            return True
        tokens = _gate_content_tokens(value)
        return bool(tokens and tokens.issubset(_gate_content_tokens(haystack)))
    if kind in ("number", "date"):
        return value in haystack.replace(",", "")
    return value in haystack


def _qdc_component_match_score(component: str, text: str) -> float:
    if not _qdc_component_in_text(component, text):
        return 0.0
    kind, _, value = component.partition(":")
    if kind == "entity":
        return 1.0 + min(0.5, len(value.split()) * 0.1)
    if kind in ("number", "date"):
        return 0.9
    if kind == "constraint":
        return 0.6
    return 0.5


def _qdc_covered_components(block_ids: Sequence[int], block_texts: Sequence[str], components: Sequence[str]) -> Set[str]:
    covered: Set[str] = set()
    for block_id in block_ids:
        if not (0 <= int(block_id) < len(block_texts)):
            continue
        text = block_texts[int(block_id)]
        for component in components:
            if _qdc_component_in_text(component, text):
                covered.add(component)
    return covered


def _qdc_block_token_lengths(cfg: Dict[str, Any], num_blocks: int) -> List[int]:
    cached = cfg.get("_qdc_block_token_lengths")
    if isinstance(cached, list) and len(cached) >= num_blocks:
        return [int(x) for x in cached[:num_blocks]]
    block_texts = cfg.get("block_texts") or []
    block_size = int(cfg.get("block_size", _BLOCK_SIZE) or _BLOCK_SIZE)
    lengths = [block_size] * num_blocks
    if block_texts:
        avg_chars = sum(len(str(text)) for text in block_texts[:-1]) / max(1, len(block_texts) - 1)
        last_len = len(str(block_texts[-1]))
        if avg_chars > 0 and last_len < avg_chars * 0.95:
            lengths[-1] = max(1, min(block_size, int(round(block_size * last_len / avg_chars))))
    cfg["_qdc_block_token_lengths"] = lengths
    return lengths


def _qdc_token_count(selected: Set[int], token_lengths: Sequence[int]) -> int:
    return sum(int(token_lengths[block_id]) for block_id in selected if 0 <= int(block_id) < len(token_lengths))


def _qdc_block_token_map(block_texts: Sequence[str]) -> Dict[int, Set[str]]:
    return {idx: _gate_content_tokens(text) for idx, text in enumerate(block_texts)}


def _qdc_block_query_overlaps(block_texts: Sequence[str], query: str) -> Dict[int, float]:
    query_tokens = _gate_content_tokens(query)
    if not query_tokens:
        return {idx: 0.0 for idx in range(len(block_texts))}
    overlaps: Dict[int, float] = {}
    for idx, text in enumerate(block_texts):
        tokens = _gate_content_tokens(text)
        overlaps[idx] = len(tokens.intersection(query_tokens)) / len(query_tokens) if tokens else 0.0
    return overlaps


def _qdc_marginal_component_contribution(
    block_id: int, selected: Set[int], block_texts: Sequence[str], components: Sequence[str],
) -> int:
    if block_id not in selected:
        return 0
    before = _qdc_covered_components(selected, block_texts, components)
    after = _qdc_covered_components(selected - {block_id}, block_texts, components)
    return len(before - after)


def _qdc_removal_score(
    block_id: int,
    selected: Set[int],
    token_map: Dict[int, Set[str]],
    query_overlaps: Dict[int, float],
    block_texts: Sequence[str],
    components: Sequence[str],
) -> float:
    contrib = _qdc_marginal_component_contribution(block_id, selected, block_texts, components)
    redundancies = [
        _dc_token_jaccard(token_map.get(block_id, set()), token_map.get(other, set()))
        for other in selected
        if other != block_id
    ]
    redundancy = max(redundancies) if redundancies else 0.0
    query_overlap = float(query_overlaps.get(block_id, 0.0))
    return redundancy - 0.35 * contrib - 0.25 * query_overlap


def _qdc_selected_jaccard(left: Set[int], right: Set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _qdc_apply_coverage_repair(
    cfg: Dict[str, Any],
    original_selected: Set[int],
    block_texts: Sequence[str],
    token_lengths: Sequence[int],
    query_components: Sequence[str],
    query_text: str,
    max_swaps: int = 8,
) -> Dict[str, Any]:
    num_blocks = len(block_texts)
    all_ids = set(range(num_blocks))
    selected = {int(x) for x in original_selected if 0 <= int(x) < num_blocks}
    if not selected:
        return {
            "selected": set(),
            "removed_block_ids": [],
            "added_block_ids": [],
            "fallback_used": True,
            "qdc_triggered": False,
            "qdc_trigger_reason": "empty_original_selection",
        }

    token_map = _qdc_block_token_map(block_texts)
    query_overlaps = _qdc_block_query_overlaps(block_texts, query_text)
    pruned = all_ids - selected
    target_keep = len(selected)
    original_token_count = _qdc_token_count(selected, token_lengths)
    original_cov = _qdc_covered_components(selected, block_texts, query_components)
    missing = [comp for comp in query_components if comp not in original_cov]

    if not query_components:
        return {
            "selected": set(selected),
            "removed_block_ids": [],
            "added_block_ids": [],
            "fallback_used": True,
            "qdc_triggered": False,
            "qdc_trigger_reason": "no_query_components",
            "original_component_coverage": 1.0,
            "qdc_component_coverage": 1.0,
            "coverage_delta": 0.0,
        }
    if not missing:
        return {
            "selected": set(selected),
            "removed_block_ids": [],
            "added_block_ids": [],
            "fallback_used": False,
            "qdc_triggered": False,
            "qdc_trigger_reason": "full_component_coverage",
            "original_component_coverage": 1.0,
            "qdc_component_coverage": 1.0,
            "coverage_delta": 0.0,
        }

    removed: List[int] = []
    added: List[int] = []
    swaps = 0
    fallback_reason = ""

    while missing and swaps < max_swaps:
        best_pair: Optional[Tuple[int, int, str, float]] = None
        for comp in missing:
            priority = 1.0 if comp.startswith("entity:") else 0.8 if comp.startswith(("number:", "date:")) else 0.5
            candidate_scores: List[Tuple[float, int]] = []
            for block_id in pruned:
                score = _qdc_component_match_score(comp, block_texts[block_id])
                if score > 0:
                    candidate_scores.append((score + priority, block_id))
            if not candidate_scores:
                continue
            candidate_scores.sort(reverse=True)
            cand_id = candidate_scores[0][1]
            removal_candidates = sorted(
                (
                    _qdc_removal_score(block_id, selected, token_map, query_overlaps, block_texts, query_components),
                    block_id,
                )
                for block_id in selected
            )
            victim_id = removal_candidates[-1][1]
            if token_lengths[victim_id] != token_lengths[cand_id]:
                continue
            new_selected = (selected - {victim_id}) | {cand_id}
            if len(new_selected) != target_keep:
                continue
            new_cov = _qdc_covered_components(new_selected, block_texts, query_components)
            gain = len(new_cov) - len(_qdc_covered_components(selected, block_texts, query_components))
            if comp not in new_cov or gain <= 0:
                continue
            pair_score = gain + candidate_scores[0][0]
            if best_pair is None or pair_score > best_pair[3]:
                best_pair = (victim_id, cand_id, comp, pair_score)

        if best_pair is None:
            fallback_reason = "no_gainful_swap_for_missing_components"
            break

        victim_id, cand_id, _, _ = best_pair
        selected = (selected - {victim_id}) | {cand_id}
        pruned = all_ids - selected
        removed.append(victim_id)
        added.append(cand_id)
        swaps += 1
        missing = [comp for comp in query_components if comp not in _qdc_covered_components(selected, block_texts, query_components)]

    qdc_cov = _qdc_covered_components(selected, block_texts, query_components)
    coverage_gain = len(qdc_cov) - len(original_cov)
    num_components = max(1, len(query_components))
    original_coverage = len(original_cov) / num_components
    qdc_coverage = len(qdc_cov) / num_components
    changed = bool(removed or added)

    if not changed:
        return {
            "selected": set(original_selected),
            "removed_block_ids": [],
            "added_block_ids": [],
            "fallback_used": True,
            "qdc_triggered": False,
            "qdc_trigger_reason": fallback_reason or "no_budget_neutral_repair_found",
            "original_component_coverage": original_coverage,
            "qdc_component_coverage": original_coverage,
            "coverage_delta": 0.0,
            "missing_components_before": [c for c in query_components if c not in original_cov],
            "missing_components_after": [c for c in query_components if c not in original_cov],
        }

    qdc_token_count = _qdc_token_count(selected, token_lengths)
    budget_preserved = len(selected) == target_keep and qdc_token_count == original_token_count
    if not budget_preserved:
        return {
            "selected": set(original_selected),
            "removed_block_ids": [],
            "added_block_ids": [],
            "fallback_used": True,
            "qdc_triggered": False,
            "qdc_trigger_reason": "budget_not_preserved_reverted_to_original",
            "original_component_coverage": original_coverage,
            "qdc_component_coverage": original_coverage,
            "coverage_delta": 0.0,
            "missing_components_before": [c for c in query_components if c not in original_cov],
            "missing_components_after": [c for c in query_components if c not in original_cov],
        }

    original_redundancy = _dc_selected_redundancy(original_selected, token_map)
    qdc_redundancy = _dc_selected_redundancy(selected, token_map)
    original_query_overlap = float(sum(query_overlaps.get(bid, 0.0) for bid in original_selected) / max(1, len(original_selected)))
    qdc_query_overlap = float(sum(query_overlaps.get(bid, 0.0) for bid in selected) / max(1, len(selected)))

    return {
        "selected": set(selected),
        "removed_block_ids": removed,
        "added_block_ids": added,
        "fallback_used": False,
        "qdc_triggered": coverage_gain > 0,
        "qdc_trigger_reason": f"coverage_repair_gain={coverage_gain}; swaps={swaps}",
        "original_component_coverage": original_coverage,
        "qdc_component_coverage": qdc_coverage,
        "coverage_delta": qdc_coverage - original_coverage,
        "missing_components_before": [c for c in query_components if c not in original_cov],
        "missing_components_after": [c for c in query_components if c not in qdc_cov],
        "selected_jaccard_vs_original": _qdc_selected_jaccard(original_selected, selected),
        "original_redundancy": original_redundancy,
        "qdc_redundancy": qdc_redundancy,
        "redundancy_delta": qdc_redundancy - original_redundancy,
        "original_query_overlap": original_query_overlap,
        "qdc_query_overlap": qdc_query_overlap,
        "query_overlap_delta": qdc_query_overlap - original_query_overlap,
        "budget_preserved": True,
        "selected_token_count_original": original_token_count,
        "selected_token_count_qdc": qdc_token_count,
        "over_budget": False,
        "query_components": list(query_components),
    }


def _qdc_sliminfer_apply(cfg: Dict[str, Any], original_selected: Set[int]) -> Dict[str, Any]:
    block_texts = cfg.get("block_texts") or (cfg.get("audit_context") or {}).get("block_texts") or []
    query_text = str(cfg.get("query_text") or (cfg.get("audit_context") or {}).get("query_text") or "")
    if not block_texts:
        return {
            "selected": set(original_selected),
            "fallback_used": True,
            "qdc_triggered": False,
            "qdc_trigger_reason": "block_texts_unavailable",
        }
    query_components = _qdc_extract_query_components(query_text)
    token_lengths = _qdc_block_token_lengths(cfg, len(block_texts))
    return _qdc_apply_coverage_repair(
        cfg=cfg,
        original_selected=set(original_selected),
        block_texts=block_texts,
        token_lengths=token_lengths,
        query_components=query_components,
        query_text=query_text,
    )


def _write_qdc_selection_audit(
    cfg: Dict[str, Any],
    original_selected: Set[int],
    qdc_selected: Set[int],
    qdc_result: Dict[str, Any],
) -> None:
    audit_path = cfg.get("qdc_selection_audit_path")
    if not audit_path:
        return
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    record = {
        "task": context.get("task", ""),
        "sample_id": context.get("sample_id", ""),
        "sample_index": context.get("sample_index", ""),
        "method": cfg.get("variant_method", "qdc_sliminfer"),
        "layer_id": int(cfg.get("layer_idx", -1)),
        "phase": cfg.get("phase", ""),
        "original_selected_block_ids": sorted(int(x) for x in original_selected),
        "qdc_selected_block_ids": sorted(int(x) for x in qdc_selected),
        "removed_block_ids": qdc_result.get("removed_block_ids", []),
        "added_block_ids": qdc_result.get("added_block_ids", []),
        "qdc_triggered": bool(qdc_result.get("qdc_triggered")),
        "qdc_trigger_reason": str(qdc_result.get("qdc_trigger_reason", "")),
        "fallback_used": bool(qdc_result.get("fallback_used")),
        "budget_preserved": bool(qdc_result.get("budget_preserved", True)),
        "original_component_coverage": qdc_result.get("original_component_coverage"),
        "qdc_component_coverage": qdc_result.get("qdc_component_coverage"),
        "coverage_delta": qdc_result.get("coverage_delta"),
        "selected_jaccard_vs_original": qdc_result.get("selected_jaccard_vs_original"),
        "original_redundancy": qdc_result.get("original_redundancy"),
        "qdc_redundancy": qdc_result.get("qdc_redundancy"),
        "redundancy_delta": qdc_result.get("redundancy_delta"),
        "original_query_overlap": qdc_result.get("original_query_overlap"),
        "qdc_query_overlap": qdc_result.get("qdc_query_overlap"),
        "query_overlap_delta": qdc_result.get("query_overlap_delta"),
        "selected_token_count_original": qdc_result.get("selected_token_count_original"),
        "selected_token_count_qdc": qdc_result.get("selected_token_count_qdc"),
        "query_components": qdc_result.get("query_components", []),
    }
    try:
        with open(str(audit_path), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _qdc_maybe_apply_page_selection(cfg: Dict[str, Any], original_selected: Set[int]) -> Set[int]:
    if cfg.get("variant_method") != "qdc_sliminfer" or not _qdc_is_gate_layer(cfg):
        return set(original_selected)
    cached = cfg.get("_qdc_cached_result")
    if isinstance(cached, dict) and "selected" in cached:
        return set(int(x) for x in cached["selected"])
    qdc_result = _qdc_sliminfer_apply(cfg, set(original_selected))
    cfg["_qdc_cached_result"] = qdc_result
    cfg["_qdc_last"] = qdc_result
    qdc_selected = set(qdc_result.get("selected", original_selected))
    _write_qdc_selection_audit(cfg, original_selected, qdc_selected, qdc_result)
    return qdc_selected


def _mcu_gate_apply_layer(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("mcu_gate_apply_layer", cfg.get("gate_apply_layer", 26)) or 26)


def _mcu_runtime_kv_layer(cfg: Dict[str, Any]) -> int:
    """Pruning at gate layer L updates KV consumed at attention layer L+1."""
    return _mcu_gate_apply_layer(cfg) + 1


def _mcu_is_gate_layer(cfg: Dict[str, Any]) -> bool:
    gate_apply_layer = _mcu_gate_apply_layer(cfg)
    current_layer = int(cfg.get("layer_idx", 0) or 0)
    return gate_apply_layer <= 0 or current_layer >= gate_apply_layer


def _mcu_frozen_selected_block_ids(cfg: Dict[str, Any]) -> Optional[Set[int]]:
    cached = cfg.get("_mcu_cached_result")
    if not isinstance(cached, dict):
        frozen_all = cfg.get("mcu_frozen_selections") or {}
        ctx = cfg.get("audit_context") or {}
        task = str(ctx.get("task", ""))
        sample_index = ctx.get("sample_index")
        if task and sample_index is not None:
            cached = frozen_all.get(f"{task}:{int(sample_index)}")
    if not isinstance(cached, dict) or bool(cached.get("fallback_used")):
        return None
    raw = cached.get("mcu_selected_block_ids") or cached.get("selected")
    if not raw:
        return None
    return {int(x) for x in raw}


def _mcu_selected_jaccard(left: Set[int], right: Set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _mcu_block_token_lengths(cfg: Dict[str, Any], num_blocks: int) -> List[int]:
    cached = cfg.get("_mcu_block_token_lengths")
    if isinstance(cached, list) and len(cached) >= num_blocks:
        return [int(x) for x in cached[:num_blocks]]
    block_size = int(cfg.get("block_size", _BLOCK_SIZE) or _BLOCK_SIZE)
    context_len = int((cfg.get("audit_context") or {}).get("context_length", 0) or 0)
    if context_len <= 0:
        return [block_size for _ in range(num_blocks)]
    return [min(block_size, max(0, context_len - start)) for start in range(0, context_len, block_size)]


def _mcu_store_selected(selected: Set[int]) -> List[int]:
    return sorted(int(x) for x in selected)


def _mcu_selected_from_result(result: Dict[str, Any], default: Set[int]) -> Set[int]:
    raw = result.get("selected", default)
    if isinstance(raw, set):
        return {int(x) for x in raw}
    if isinstance(raw, (list, tuple)):
        return {int(x) for x in raw}
    return set(default)


def _mcu_sliminfer_apply(cfg: Dict[str, Any], original_selected: Set[int]) -> Dict[str, Any]:
    cached = cfg.get("_mcu_cached_result")
    if not isinstance(cached, dict):
        frozen_all = cfg.get("mcu_frozen_selections") or {}
        ctx = cfg.get("audit_context") or {}
        task = str(ctx.get("task", ""))
        sample_index = ctx.get("sample_index")
        if task and sample_index is not None:
            cached = frozen_all.get(f"{task}:{int(sample_index)}")
    if not isinstance(cached, dict):
        return {
            "selected": _mcu_store_selected(original_selected),
            "fallback_used": True,
            "mcu_triggered": False,
            "mcu_trigger_reason": "missing_mcu_cached_result",
        }
    if bool(cached.get("fallback_used")):
        return {
            "selected": _mcu_store_selected(original_selected),
            "fallback_used": True,
            "mcu_triggered": False,
            "mcu_trigger_reason": str(cached.get("fallback_reason", "fallback_used")),
            **{k: cached.get(k) for k in (
                "original_selected_block_ids", "mcu_selected_block_ids",
                "removed_block_ids", "added_block_ids", "utility_margin",
                "selected_jaccard_vs_original", "probe_runtime_ms",
            )},
        }
    mcu_ids = cached.get("mcu_selected_block_ids") or cached.get("selected") or []
    mcu_selected = {int(x) for x in mcu_ids}
    if not mcu_selected:
        return {
            "selected": _mcu_store_selected(original_selected),
            "fallback_used": True,
            "mcu_triggered": False,
            "mcu_trigger_reason": "empty_mcu_selection",
        }
    token_lengths = _mcu_block_token_lengths(cfg, max(max(mcu_selected), max(original_selected), 0) + 1)
    original_token_count = sum(token_lengths[bid] for bid in original_selected if 0 <= bid < len(token_lengths))
    mcu_token_count = sum(token_lengths[bid] for bid in mcu_selected if 0 <= bid < len(token_lengths))
    budget_preserved = bool(cached.get("budget_preserved", True))
    if len(mcu_selected) != len(original_selected):
        return {
            "selected": _mcu_store_selected(original_selected),
            "fallback_used": True,
            "mcu_triggered": False,
            "mcu_trigger_reason": "budget_mismatch",
            "selected_token_count_original": original_token_count,
            "selected_token_count_mcu": mcu_token_count,
        }
    if not budget_preserved and mcu_token_count != original_token_count:
        return {
            "selected": _mcu_store_selected(original_selected),
            "fallback_used": True,
            "mcu_triggered": False,
            "mcu_trigger_reason": "budget_mismatch",
            "selected_token_count_original": original_token_count,
            "selected_token_count_mcu": mcu_token_count,
        }
    block_texts = cfg.get("block_texts") or (cfg.get("audit_context") or {}).get("block_texts") or []
    query_text = str(cfg.get("query_text") or (cfg.get("audit_context") or {}).get("query_text") or "")
    token_map = {idx: _gate_content_tokens(text) for idx, text in enumerate(block_texts)} if block_texts else {}
    query_tokens = _gate_content_tokens(query_text)
    query_overlaps = {
        idx: (len(tokens.intersection(query_tokens)) / len(query_tokens) if query_tokens and tokens else 0.0)
        for idx, tokens in token_map.items()
    }
    original_redundancy = _dc_selected_redundancy(original_selected, token_map) if token_map else 0.0
    mcu_redundancy = _dc_selected_redundancy(mcu_selected, token_map) if token_map else 0.0
    original_query_overlap = float(
        sum(query_overlaps.get(bid, 0.0) for bid in original_selected) / max(1, len(original_selected))
    ) if original_selected else 0.0
    mcu_query_overlap = float(
        sum(query_overlaps.get(bid, 0.0) for bid in mcu_selected) / max(1, len(mcu_selected))
    ) if mcu_selected else 0.0
    removed = sorted(original_selected - mcu_selected)
    added = sorted(mcu_selected - original_selected)
    return {
        "selected": _mcu_store_selected(mcu_selected),
        "fallback_used": False,
        "mcu_triggered": bool(removed or added),
        "mcu_trigger_reason": str(cached.get("mcu_select_reason", cached.get("fallback_reason", "mcu_swap_applied"))),
        "original_selected_block_ids": sorted(int(x) for x in original_selected),
        "mcu_selected_block_ids": sorted(int(x) for x in mcu_selected),
        "removed_block_ids": removed,
        "added_block_ids": added,
        "removed_block_utility": cached.get("removed_block_utility", []),
        "added_block_utility": cached.get("added_block_utility", []),
        "utility_margin": cached.get("utility_margin", 0.0),
        "hidden_change_mean": cached.get("hidden_change_mean", 0.0),
        "logit_kl_mean": cached.get("logit_kl_mean", 0.0),
        "entropy_change_mean": cached.get("entropy_change_mean", 0.0),
        "top_logit_margin_change_mean": cached.get("top_logit_margin_change_mean", 0.0),
        "probe_runtime_ms": cached.get("probe_runtime_ms", 0.0),
        "selected_jaccard_vs_original": _mcu_selected_jaccard(original_selected, mcu_selected),
        "original_redundancy": original_redundancy,
        "mcu_redundancy": mcu_redundancy,
        "redundancy_delta": mcu_redundancy - original_redundancy,
        "original_query_overlap": original_query_overlap,
        "mcu_query_overlap": mcu_query_overlap,
        "query_overlap_delta": mcu_query_overlap - original_query_overlap,
        "budget_preserved": True,
        "selected_token_count_original": original_token_count,
        "selected_token_count_mcu": mcu_token_count,
        "over_budget": False,
    }


def _write_mcu_selection_audit(
    cfg: Dict[str, Any],
    original_selected: Set[int],
    mcu_selected: Set[int],
    mcu_result: Dict[str, Any],
) -> None:
    audit_path = cfg.get("mcu_selection_audit_path")
    if not audit_path:
        return
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    record = {
        "task": context.get("task", ""),
        "sample_id": context.get("sample_id", ""),
        "sample_index": context.get("sample_index", ""),
        "method": cfg.get("variant_method", "mcu_sliminfer"),
        "layer_id": int(cfg.get("layer_idx", -1)),
        "phase": cfg.get("phase", ""),
        "original_selected_block_ids": sorted(int(x) for x in original_selected),
        "mcu_selected_block_ids": sorted(int(x) for x in mcu_selected),
        "removed_block_ids": mcu_result.get("removed_block_ids", []),
        "added_block_ids": mcu_result.get("added_block_ids", []),
        "mcu_triggered": bool(mcu_result.get("mcu_triggered")),
        "mcu_trigger_reason": str(mcu_result.get("mcu_trigger_reason", "")),
        "fallback_used": bool(mcu_result.get("fallback_used")),
        "fallback_reason": str(mcu_result.get("mcu_trigger_reason", "")),
        "budget_preserved": bool(mcu_result.get("budget_preserved", True)),
        "selected_jaccard_vs_original": mcu_result.get("selected_jaccard_vs_original"),
        "utility_margin": mcu_result.get("utility_margin"),
        "hidden_change_mean": mcu_result.get("hidden_change_mean"),
        "logit_kl_mean": mcu_result.get("logit_kl_mean"),
        "entropy_change_mean": mcu_result.get("entropy_change_mean"),
        "top_logit_margin_change_mean": mcu_result.get("top_logit_margin_change_mean"),
        "probe_runtime_ms": mcu_result.get("probe_runtime_ms"),
        "original_redundancy": mcu_result.get("original_redundancy"),
        "mcu_redundancy": mcu_result.get("mcu_redundancy"),
        "redundancy_delta": mcu_result.get("redundancy_delta"),
        "original_query_overlap": mcu_result.get("original_query_overlap"),
        "mcu_query_overlap": mcu_result.get("mcu_query_overlap"),
        "query_overlap_delta": mcu_result.get("query_overlap_delta"),
        "selected_token_count_original": mcu_result.get("selected_token_count_original"),
        "selected_token_count_mcu": mcu_result.get("selected_token_count_mcu"),
    }
    try:
        with open(str(audit_path), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _mcu_maybe_apply_page_selection(cfg: Dict[str, Any], original_selected: Set[int]) -> Set[int]:
    if cfg.get("variant_method") != "mcu_sliminfer" or not _mcu_is_gate_layer(cfg):
        return set(original_selected)
    frozen_selected = _mcu_frozen_selected_block_ids(cfg)
    if frozen_selected is not None:
        mcu_result = cfg.get("_mcu_last")
        if not isinstance(mcu_result, dict):
            mcu_result = _mcu_sliminfer_apply(cfg, set(original_selected))
            cfg["_mcu_last"] = mcu_result
        mcu_selected = set(frozen_selected)
        audit_key = (str(cfg.get("phase", "")), int(cfg.get("layer_idx", -1)))
        written = cfg.setdefault("_mcu_audit_written_keys", set())
        if audit_key not in written:
            _write_mcu_selection_audit(cfg, set(original_selected), mcu_selected, mcu_result)
            written.add(audit_key)
        cfg["_mcu_runtime_selected_block_ids"] = _mcu_store_selected(mcu_selected)
        return mcu_selected
    mcu_result = _mcu_sliminfer_apply(cfg, set(original_selected))
    cfg["_mcu_last"] = mcu_result
    mcu_selected = _mcu_selected_from_result(mcu_result, set(original_selected))
    _write_mcu_selection_audit(cfg, original_selected, mcu_selected, mcu_result)
    cfg["_mcu_runtime_selected_block_ids"] = _mcu_store_selected(mcu_selected)
    return mcu_selected
