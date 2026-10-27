import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from .bridge_score import compute_bridge_scores
from .semantic_edges import build_semantic_edge_weights, compute_node_idf_scores, extract_block_features
from .structural_risk import compute_structural_scores


def _normalize_tensor(vec: torch.Tensor, mode: str = "per_layer") -> torch.Tensor:
    if mode == "none":
        return torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))
    if vec.numel() == 0:
        return vec
    finite = torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))
    if mode == "minmax" or mode == "per_layer":
        min_v = finite.min()
        max_v = finite.max()
        denom = (max_v - min_v).clamp_min(1e-6)
        return (finite - min_v) / denom
    if mode == "zscore":
        mean = finite.mean()
        std = finite.std().clamp_min(1e-6)
        return (finite - mean) / std
    return finite


def build_bridge_graph(
    block_texts: List[str],
    query_text: str,
    bridge_graph_edges: str = "query_semantic_pos",
    bridge_pos_edge_weight: float = 0.05,
    bridge_topk_sem_edges: int = 16,
    qk_scores: Optional[torch.Tensor] = None,
    bridge_query_edge_weight: float = 1.0,
    bridge_edge_weighting: str = "raw",
    bridge_hub_penalty: bool = False,
    bridge_semantic_threshold: float = 0.0,
) -> Dict[str, Any]:
    start = time.perf_counter()
    num_blocks = len(block_texts)
    token_features = [extract_block_features(text) for text in block_texts]
    query_tokens = set(extract_block_features(query_text)["tokens"])

    edge_weights = [[0.0 for _ in range(num_blocks)] for _ in range(num_blocks)]
    query_node_ids = [num_blocks - 1] if num_blocks > 0 else []
    num_query_edges = 0
    num_semantic_edges = 0
    num_position_edges = 0
    num_attention_edges = 0

    qk_norm = None
    if qk_scores is not None and qk_scores.numel() > 0:
        qk_norm = qk_scores.detach().float()
        qk_norm = (qk_norm - qk_norm.min()).clamp_min(0.0)
        qk_denom = qk_norm.max().clamp_min(1e-6)
        qk_norm = qk_norm / qk_denom

    if bridge_graph_edges in {"random_graph", "random_graph_control"}:
        for i in range(num_blocks):
            for mul, add in ((37, 17), (53, 5), (97, 29)):
                j = (i * mul + add) % max(1, num_blocks)
                if i != j:
                    edge_weights[i][j] += 0.25
    if bridge_graph_edges in {"query_only", "query_semantic", "query_semantic_pos", "query_semantic_pos_attention", "random_graph", "random_graph_control"}:
        for idx, features in enumerate(token_features):
            overlap = 0.0
            if query_tokens:
                overlap = len(features["tokens"] & query_tokens) / max(1, len(query_tokens))
            qk_val = float(qk_norm[idx]) if qk_norm is not None else 0.0
            weight = max(overlap * 0.5, qk_val * bridge_query_edge_weight)
            edge_weights[num_blocks - 1][idx] = float(weight)
            if weight > 0.0:
                num_query_edges += 1

    if bridge_graph_edges in {"query_semantic", "query_semantic_pos", "query_semantic_pos_attention"}:
        semantic_weights = build_semantic_edge_weights(
            token_features,
            topk=bridge_topk_sem_edges,
            edge_weighting=bridge_edge_weighting,
            semantic_threshold=bridge_semantic_threshold,
        )
        for i, neighbors in enumerate(semantic_weights):
            for j, w in neighbors.items():
                edge_weights[i][j] += w
                if w > 0.0:
                    num_semantic_edges += 1

    if bridge_graph_edges in {"query_semantic_pos", "query_semantic_pos_attention"}:
        for i in range(num_blocks - 1):
            edge_weights[i][i + 1] += bridge_pos_edge_weight
            edge_weights[i + 1][i] += bridge_pos_edge_weight
            if bridge_pos_edge_weight > 0.0:
                num_position_edges += 2

    degrees = [sum(1 for weight in row if weight > 0.0) for row in edge_weights]
    num_edges_total = sum(degrees)
    query_overlap_scores = []
    for features in token_features:
        query_overlap_scores.append(len(features["tokens"] & query_tokens) / max(1, len(query_tokens)) if query_tokens else 0.0)
    idf_node_scores = compute_node_idf_scores(token_features)

    graph_summary = {
        "num_blocks": num_blocks,
        "num_edges_total": num_edges_total,
        "num_query_edges": num_query_edges,
        "num_semantic_edges": num_semantic_edges,
        "num_position_edges": num_position_edges,
        "num_attention_edges": num_attention_edges,
        "avg_degree": (sum(degrees) / num_blocks) if num_blocks else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "ppr_alpha": None,
        "pseudo_evidence_count": 0,
        "graph_build_time_ms": (time.perf_counter() - start) * 1000.0,
        "ppr_time_ms": 0.0,
        "bridge_graph_edges": bridge_graph_edges,
        "bridge_pos_edge_weight": bridge_pos_edge_weight,
        "bridge_topk_sem_edges": bridge_topk_sem_edges,
        "bridge_query_edge_weight": bridge_query_edge_weight,
        "bridge_edge_weighting": bridge_edge_weighting,
        "bridge_hub_penalty": bool(bridge_hub_penalty),
        "bridge_semantic_threshold": bridge_semantic_threshold,
        "control_type": "random_graph" if bridge_graph_edges in {"random_graph", "random_graph_control"} else "none",
    }
    return {
        "num_blocks": num_blocks,
        "edge_weights": edge_weights,
        "query_node_ids": query_node_ids,
        "graph_summary": graph_summary,
        "query_text": query_text,
        "block_texts": block_texts,
        "query_overlap_scores": query_overlap_scores,
        "idf_node_scores": idf_node_scores,
        "degree_scores": degrees,
    }


def select_pseudo_evidence_nodes(
    qk_scores: torch.Tensor,
    delta_o_scores: Optional[torch.Tensor],
    pseudo_evidence_ratio: float = 0.10,
    anchor_mode: str = "union_weighted",
    lexical_scores: Optional[torch.Tensor] = None,
    structural_scores: Optional[torch.Tensor] = None,
) -> List[int]:
    if qk_scores.numel() == 0:
        return []
    qk_norm = _normalize_tensor(qk_scores, "minmax")
    do_norm = _normalize_tensor(delta_o_scores, "minmax") if delta_o_scores is not None and delta_o_scores.numel() == qk_scores.numel() else torch.zeros_like(qk_norm)
    lex_norm = _normalize_tensor(lexical_scores, "minmax") if lexical_scores is not None and lexical_scores.numel() == qk_scores.numel() else torch.zeros_like(qk_norm)
    struct_norm = _normalize_tensor(structural_scores, "minmax") if structural_scores is not None and structural_scores.numel() == qk_scores.numel() else torch.zeros_like(qk_norm)
    k = max(1, math.ceil(qk_scores.numel() * pseudo_evidence_ratio))
    if anchor_mode == "delta_o_top":
        combined = do_norm
    elif anchor_mode == "qk_top":
        combined = qk_norm
    elif anchor_mode == "lexical_overlap_top":
        combined = lex_norm
    elif anchor_mode == "retrieval_struct_top":
        combined = 0.5 * lex_norm + 0.5 * struct_norm
    elif anchor_mode == "intersection_delta_o_qk":
        top_do = set(torch.topk(do_norm, k=min(k * 2, do_norm.numel())).indices.detach().cpu().tolist())
        top_qk = set(torch.topk(qk_norm, k=min(k * 2, qk_norm.numel())).indices.detach().cpu().tolist())
        inter = list(top_do & top_qk)
        if len(inter) >= k:
            scores = (do_norm + qk_norm) / 2.0
            inter.sort(key=lambda idx: float(scores[idx]), reverse=True)
            return [int(idx) for idx in inter[:k]]
        combined = 0.5 * do_norm + 0.5 * qk_norm
    elif anchor_mode == "random_anchor_control":
        perm = sorted(range(qk_scores.numel()), key=lambda idx: ((idx * 37 + 11) % max(1, qk_scores.numel())))
        return [int(idx) for idx in perm[:k]]
    else:
        combined = 0.45 * do_norm + 0.35 * qk_norm + 0.10 * lex_norm + 0.10 * struct_norm
    _, indices = torch.topk(combined, k=k)
    return [int(i) for i in indices.tolist()]


def _gather_state_scores(state: Dict[str, Any], name: str, block_ids: Optional[List[int]], length: int, device: torch.device) -> torch.Tensor:
    full = torch.tensor(state.get(name, [0.0] * int(state.get("num_blocks", length))), dtype=torch.float32, device=device)
    if block_ids and full.numel() > max(block_ids):
        out = full[torch.tensor(block_ids, dtype=torch.long, device=device)]
    else:
        out = full
    if out.numel() != length:
        return torch.zeros(length, dtype=torch.float32, device=device)
    return out


def compute_bridge_utilities(
    state: Dict[str, Any],
    qk_scores: torch.Tensor,
    delta_o_values: Optional[torch.Tensor],
    args: Any,
    block_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if state is None:
        return {}
    device = qk_scores.device
    local_scores = torch.zeros_like(qk_scores)
    if args.local_score == "delta_o" and delta_o_values is not None:
        local_scores = _normalize_tensor(delta_o_values, args.bridge_norm)
    elif args.local_score == "qk":
        local_scores = _normalize_tensor(qk_scores, args.bridge_norm)
    local_scores = local_scores.to(device=device, dtype=torch.float32)

    structural_scores = _gather_state_scores(state, "structural_scores", block_ids, len(qk_scores), device)
    structural_scores = _normalize_tensor(structural_scores, args.bridge_norm)
    lexical_scores = _normalize_tensor(_gather_state_scores(state, "query_overlap_scores", block_ids, len(qk_scores), device), args.bridge_norm)
    idf_node_scores = _normalize_tensor(_gather_state_scores(state, "idf_node_scores", block_ids, len(qk_scores), device), args.bridge_norm)
    degree_scores = _gather_state_scores(state, "degree_scores", block_ids, len(qk_scores), device)

    local_evidence_ids = select_pseudo_evidence_nodes(
        qk_scores=qk_scores,
        delta_o_scores=delta_o_values,
        pseudo_evidence_ratio=getattr(args, "bridge_anchor_ratio", args.bridge_pseudo_evidence_ratio),
        anchor_mode=getattr(args, "bridge_anchor_mode", "union_weighted"),
        lexical_scores=lexical_scores,
        structural_scores=structural_scores,
    )
    if block_ids:
        evidence_ids = [int(block_ids[idx]) for idx in local_evidence_ids if idx < len(block_ids)]
    else:
        evidence_ids = local_evidence_ids
    bridge_result = compute_bridge_scores(
        edge_weights=state["edge_weights"],
        query_node_ids=state["query_node_ids"],
        evidence_node_ids=evidence_ids,
        bridge_score_type=args.bridge_score,
        ppr_alpha=args.bridge_ppr_alpha,
        num_evidence_anchors=int(getattr(args, "num_evidence_anchors", 16)),
        seed=int(getattr(args, "seed", 0) or 0),
    )

    full_bridge_scores = bridge_result["bridge_scores"].to(device=device, dtype=torch.float32)
    if block_ids and full_bridge_scores.numel() > max(block_ids):
        bridge_scores = full_bridge_scores[torch.tensor(block_ids, dtype=torch.long, device=device)]
    else:
        bridge_scores = full_bridge_scores
    if bridge_scores.numel() != qk_scores.numel():
        bridge_scores = torch.zeros_like(qk_scores, dtype=torch.float32)
    hub_lambda = float(getattr(args, "bridge_hub_lambda", 0.0) or 0.0)
    if bool(getattr(args, "bridge_hub_penalty", False)) or hub_lambda > 0.0:
        penalty = 1.0 + max(hub_lambda, 1.0) * torch.log1p(degree_scores.float()).to(device)
        bridge_scores = bridge_scores / penalty.clamp_min(1e-6)
        bridge_scores = _normalize_tensor(bridge_scores, args.bridge_norm)
    if bool(getattr(args, "bridge_idf_node", False)):
        bridge_scores = bridge_scores * idf_node_scores
        bridge_scores = _normalize_tensor(bridge_scores, args.bridge_norm)
    future_scores = torch.zeros_like(bridge_scores)
    bridge_gamma = float(getattr(args, "bridge_gamma", 0.0))
    final_utility = (
        args.bridge_alpha * local_scores
        + args.bridge_beta * bridge_scores
        + bridge_gamma * future_scores
        + args.bridge_eta * structural_scores
    )
    final_utility = _normalize_tensor(final_utility, args.bridge_norm)

    graph_summary = state.setdefault("graph_summary", {})
    graph_summary["ppr_alpha"] = float(args.bridge_ppr_alpha)
    graph_summary["pseudo_evidence_count"] = len(evidence_ids)
    graph_summary["ppr_time_ms"] = float(bridge_result["ppr_time_ms"])
    graph_summary["anchor_mode"] = getattr(args, "bridge_anchor_mode", "union_weighted")
    graph_summary["anchor_ratio"] = float(getattr(args, "bridge_anchor_ratio", args.bridge_pseudo_evidence_ratio))
    graph_summary["control_type"] = state.get("control_type", "none")

    return {
        "local_scores": local_scores,
        "bridge_scores": bridge_scores,
        "future_scores": future_scores,
        "structural_scores": structural_scores,
        "lexical_scores": lexical_scores,
        "idf_node_scores": idf_node_scores,
        "degree_scores": degree_scores,
        "final_utility": final_utility,
        "bridge_ppr_time_ms": bridge_result["ppr_time_ms"],
        "bridge_evidence_node_ids": evidence_ids,
        "bridge_local_evidence_node_ids": local_evidence_ids,
        "bridge_score_type": args.bridge_score,
        "bridge_summary": bridge_result,
    }
