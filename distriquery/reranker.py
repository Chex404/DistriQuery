"""Reranking.

Two-stage retrieval pattern (Theory Primer, Part 1.6): hybrid retrieval
(Part 1.5) is optimized for speed over the whole corpus and returns a
broader shortlist; a reranker then re-scores just that shortlist with a
more expensive-but-more-accurate model, and only the final top-k survive
to reach the LLM. A cross-encoder can't scale to a full corpus (it scores
query+document jointly, so nothing can be precomputed) but is far more
accurate at judging fine-grained relevance once the candidate set is small.
"""

from abc import ABC, abstractmethod
from typing import List

from distriquery.sparse_retrieval import tokenize
from distriquery.vectorstore import SearchResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        raise NotImplementedError


class OverlapReranker(Reranker):
    """Scores each candidate by how many distinct query terms appear in its
    text. Crude compared to a real cross-encoder — it can't judge meaning,
    just raw word overlap — but deterministic, free, and exercises the
    real two-stage interface end to end.
    """

    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        if not results:
            return []

        query_terms = set(tokenize(query))

        def overlap_score(result: SearchResult) -> int:
            chunk_terms = set(tokenize(result.chunk.text))
            return len(query_terms & chunk_terms)

        reranked = sorted(results, key=overlap_score, reverse=True)
        return reranked[:top_k]


class CrossEncoderReranker(Reranker):
    """Real cross-encoder reranking via sentence-transformers' CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy import

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        if not results:
            return []

        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)

        reranked = sorted(zip(results, scores), key=lambda pair: -pair[1])
        return [
            SearchResult(chunk=r.chunk, score=float(score)) for r, score in reranked[:top_k]
        ]


def get_reranker(backend: str, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    if backend == "overlap":
        return OverlapReranker()
    if backend == "cross-encoder":
        return CrossEncoderReranker(model_name=model_name)
    raise ValueError(f"Unknown reranker backend: {backend}")