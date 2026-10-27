import heapq
import time
from typing import Any, Dict, List, Optional

import torch


def _normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    if vec.numel() == 0:
        return vec
    finite = torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))
    min_v = finite.min()
    max_v = finite.max()
    denom = (max_v - min_v).clamp_min(1e-6)
    return (finite - min_v) / denom


def _build_transition_matrix(edge_weights: List[List[float]]) -> torch.Tensor:
    mat = torch.tensor(edge_weights, dtype=torch.float32)
    if mat.numel() == 0:
        return mat
    row_sum = mat.sum(dim=1, keepdim=True)
    dangling = row_sum.squeeze(-1) <= 1e-9
    if dangling.any():
        mat[dangling] = 1.0 / max(1, mat.shape[0])
        row_sum = mat.sum(dim=1, keepdim=True)
    out_sum = row_sum.clamp_min(1e-9)
    return mat / out_sum


def _degree_scores(edge_weights: List[List[float]]) -> torch.Tensor:
    mat = torch.tensor(edge_weights, dtype=torch.float32)
    if mat.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    return mat.sum(dim=0) + mat.sum(dim=1)


def _path_coverage_scores(
    edge_weights: List[List[float]],
    query_node_ids: List[int],
    evidence_node_ids: Optional[List[int]],
    num_evidence_anchors: int = 16,
) -> torch.Tensor:
    num_nodes = len(edge_weights)
    scores = torch.zeros(num_nodes, dtype=torch.float32)
    if num_nodes == 0:
        return scores
    sources = query_node_ids or [num_nodes - 1]
    anchors = (evidence_node_ids or [])[: max(1, int(num_evidence_anchors))]
    if not anchors:
        return scores

    adjacency: List[List[tuple[float, int]]] = [[] for _ in range(num_nodes)]
    for i, row in enumerate(edge_weights):
        for j, weight in enumerate(row):
            if weight > 0:
                adjacency[i].append((1.0 / max(float(weight), 1e-6), j))

    for source in sources:
        dist = [float("inf")] * num_nodes
        prev = [-1] * num_nodes
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            d_cur, node = heapq.heappop(heap)
            if d_cur != dist[node]:
                continue
            for cost, nxt in adjacency[node]:
                cand = d_cur + cost
                if cand < dist[nxt]:
                    dist[nxt] = cand
                    prev[nxt] = node
                    heapq.heappush(heap, (cand, nxt))
        for anchor in anchors:
            if anchor < 0 or anchor >= num_nodes or not torch.isfinite(torch.tensor(dist[anchor])):
                continue
            node = anchor
            guard = 0
            while node != -1 and guard <= num_nodes:
                scores[node] += 1.0
                if node == source:
                    break
                node = prev[node]
                guard += 1
    return scores


def personalized_pagerank(
    transition_matrix: torch.Tensor,
    personalization: torch.Tensor,
    alpha: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> torch.Tensor:
    n = transition_matrix.shape[0]
    x = personalization.clone().float()
    teleport = personalization.clone().float()
    for _ in range(max_iter):
        x_next = alpha * (transition_matrix.t() @ x) + (1.0 - alpha) * teleport
        if torch.norm(x_next - x, p=1) < tol:
            return x_next
        x = x_next
    return x


def compute_bridge_scores(
    edge_weights: List[List[float]],
    query_node_ids: List[int],
    evidence_node_ids: Optional[List[int]],
    bridge_score_type: str = "ppr",
    ppr_alpha: float = 0.85,
    num_evidence_anchors: int = 16,
    seed: int = 0,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    start = time.perf_counter()
    num_nodes = len(edge_weights)
    if num_nodes == 0:
        return {
            "bridge_scores": torch.zeros(0, dtype=torch.float32),
            "ppr_time_ms": 0.0,
            "evidence_node_ids": evidence_node_ids or [],
        }

    if bridge_score_type == "degree_centrality":
        ppr_query = torch.zeros(num_nodes, dtype=torch.float32)
        bridge_scores = _degree_scores(edge_weights)
    elif bridge_score_type == "random":
        ppr_query = torch.zeros(num_nodes, dtype=torch.float32)
        # Deterministic pseudo-random control: stable across processes for the same seed.
        values = [((idx * 1103515245 + 12345 + int(seed) * 2654435761) % 2147483647) for idx in range(num_nodes)]
        bridge_scores = torch.tensor(values, dtype=torch.float32)
    elif bridge_score_type == "path_coverage":
        ppr_query = torch.zeros(num_nodes, dtype=torch.float32)
        bridge_scores = _path_coverage_scores(edge_weights, query_node_ids, evidence_node_ids, num_evidence_anchors)
    else:
        transition = _build_transition_matrix(edge_weights)
        query_personalization = torch.zeros(num_nodes, dtype=torch.float32)
        if query_node_ids:
            query_personalization[query_node_ids] = 1.0 / len(query_node_ids)
        else:
            query_personalization[:] = 1.0 / num_nodes

        ppr_query = personalized_pagerank(transition, query_personalization, alpha=ppr_alpha, max_iter=max_iter, tol=tol)
        if bridge_score_type == "ppr_product" and evidence_node_ids:
            evidence_personalization = torch.zeros(num_nodes, dtype=torch.float32)
            evidence_personalization[evidence_node_ids] = 1.0 / len(evidence_node_ids)
            ppr_evidence = personalized_pagerank(transition, evidence_personalization, alpha=ppr_alpha, max_iter=max_iter, tol=tol)
            bridge_scores = torch.sqrt(ppr_query * ppr_evidence)
        else:
            bridge_scores = ppr_query

    if bridge_score_type == "shuffled_bridge_scores_control":
        bridge_scores = torch.roll(bridge_scores, shifts=max(1, num_nodes // 3))

    bridge_scores = _normalize_vector(bridge_scores)
    result = {
        "bridge_scores": bridge_scores,
        "ppr_time_ms": (time.perf_counter() - start) * 1000.0,
        "ppr_query": ppr_query,
        "evidence_node_ids": evidence_node_ids or [],
    }
    return result
