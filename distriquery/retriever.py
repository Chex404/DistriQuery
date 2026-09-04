"""Retrieval.

DenseRetriever: Phase A's original, dense-only retrieval — embed the
query, search the vector store, return the top-k chunks. Kept as-is,
still used directly by tests that want to isolate dense-only behavior.

HybridRetriever (Phase E): fuses dense and BM25 rankings via Reciprocal
Rank Fusion (Theory Primer, Part 1.5). Each method has a distinct blind
spot — BM25 misses semantically-related-but-lexically-different text;
dense embeddings miss exact rare terms (error codes, proper nouns) the
embedding model undertrained on (Part 1.2/1.3). Fusing both, rather than
picking one, is the standard production pattern for exactly this reason.
This is what Pipeline now uses by default.
"""

from typing import List

from distriquery.embedder import Embedder
from distriquery.sparse_retrieval import BM25Index
from distriquery.vectorstore import InMemoryVectorStore, SearchResult, VectorStore

RRF_K = 60  # standard constant from the RRF literature


class DenseRetriever:
    def __init__(self, embedder: Embedder, vector_store: InMemoryVectorStore):
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        query_vector = self._embedder.embed([query])[0]
        return self._vector_store.search(query_vector, top_k=top_k)


def _reciprocal_rank_fusion(ranked_id_lists: List[List[str]], k: int = RRF_K) -> List[str]:
    scores: dict = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda cid: -scores[cid])


class HybridRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, candidate_multiplier: int = 5):
        self._embedder = embedder
        self._vector_store = vector_store
        self._candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        candidate_k = max(top_k * self._candidate_multiplier, 20)

        query_vector = self._embedder.embed([query])[0]
        dense_results = self._vector_store.search(query_vector, top_k=candidate_k)
        dense_by_id = {r.chunk.chunk_id: r for r in dense_results}
        dense_ranked_ids = [r.chunk.chunk_id for r in dense_results]

        all_chunks = self._vector_store.get_all_chunks()
        bm25_index = BM25Index(all_chunks)
        bm25_results = bm25_index.search(query, top_k=candidate_k)
        bm25_by_id = {r.chunk.chunk_id: r for r in bm25_results}
        bm25_ranked_ids = [r.chunk.chunk_id for r in bm25_results]

        fused_ids = _reciprocal_rank_fusion([dense_ranked_ids, bm25_ranked_ids])[:top_k]

        results = []
        for chunk_id in fused_ids:
            if chunk_id in dense_by_id:
                results.append(dense_by_id[chunk_id])
            else:
                bm25_result = bm25_by_id[chunk_id]
                results.append(SearchResult(chunk=bm25_result.chunk, score=bm25_result.score))
        return results