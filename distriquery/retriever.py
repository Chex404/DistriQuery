"""Retrieval.

Phase A's retriever is intentionally dense-only — embed the query, search
the vector store, return the top-k chunks. Hybrid retrieval (fusing this
with BM25) is scoped for Phase E, not here.
"""

from typing import List

from distriquery.embedder import Embedder
from distriquery.vectorstore import InMemoryVectorStore, SearchResult


class DenseRetriever:
    def __init__(self, embedder: Embedder, vector_store: InMemoryVectorStore):
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        query_vector = self._embedder.embed([query])[0]
        return self._vector_store.search(query_vector, top_k=top_k)