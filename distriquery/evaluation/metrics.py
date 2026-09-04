"""Retrieval evaluation metrics (Theory Primer, Part 6.2)."""

import math
from typing import Dict, List, Sequence, Set


def precision_at_k(retrieved: Sequence[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[int], relevant: Set[int]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_lists: List[Sequence[int]], relevant_sets: List[Set[int]]
) -> float:
    if not retrieved_lists:
        return 0.0
    scores = [reciprocal_rank(r, rel) for r, rel in zip(retrieved_lists, relevant_sets)]
    return sum(scores) / len(scores)


def dcg_at_k(retrieved: Sequence[int], relevance_scores: Dict[int, int], k: int) -> float:
    top_k = retrieved[:k]
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        rel = relevance_scores.get(item, 0)
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: Sequence[int], relevance_scores: Dict[int, int], k: int) -> float:
    dcg = dcg_at_k(retrieved, relevance_scores, k)
    ideal_order = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_order, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg