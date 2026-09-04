"""Evaluation runner — generic over any retrieval strategy."""

from dataclasses import dataclass
from typing import Callable, List

from distriquery.evaluation.golden_dataset import GoldenExample
from distriquery.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RetrievalFn = Callable[[str, int], List[int]]


@dataclass
class EvaluationResult:
    query: str
    retrieved_positions: List[int]
    relevant_positions: set
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


@dataclass
class EvaluationSummary:
    strategy_name: str
    k: int
    results: List[EvaluationResult]
    mean_precision_at_k: float
    mean_recall_at_k: float
    mrr: float


def evaluate_retrieval(
    retrieval_fn: RetrievalFn,
    golden_set: List[GoldenExample],
    k: int = 4,
    strategy_name: str = "unnamed",
) -> EvaluationSummary:
    results: List[EvaluationResult] = []
    all_retrieved: List[List[int]] = []
    all_relevant: List[set] = []

    for example in golden_set:
        retrieved_positions = retrieval_fn(example.query, k)

        results.append(
            EvaluationResult(
                query=example.query,
                retrieved_positions=retrieved_positions,
                relevant_positions=example.relevant_positions,
                precision_at_k=precision_at_k(retrieved_positions, example.relevant_positions, k),
                recall_at_k=recall_at_k(retrieved_positions, example.relevant_positions, k),
                reciprocal_rank=reciprocal_rank(retrieved_positions, example.relevant_positions),
            )
        )
        all_retrieved.append(retrieved_positions)
        all_relevant.append(example.relevant_positions)

    n = len(results) or 1
    return EvaluationSummary(
        strategy_name=strategy_name,
        k=k,
        results=results,
        mean_precision_at_k=sum(r.precision_at_k for r in results) / n,
        mean_recall_at_k=sum(r.recall_at_k for r in results) / n,
        mrr=mean_reciprocal_rank(all_retrieved, all_relevant),
    )