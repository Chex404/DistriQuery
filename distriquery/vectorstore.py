"""Vector storage.

An in-memory vector store — the deliberate stand-in for Qdrant during
Phase A. Search is brute-force cosine similarity, exact rather than
approximate, which is completely fine at Phase A's scale.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from distriquery.chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self):
        self._chunks: List[Chunk] = []
        self._vectors: Optional[np.ndarray] = None  # (N, dim), rows correspond to self._chunks

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Number of chunks must match number of vectors")

        self._chunks.extend(chunks)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        if self._vectors is None or len(self._chunks) == 0:
            return []

        # Vectors are already L2-normalized, so a plain dot product IS
        # cosine similarity here — no need to renormalize.
        scores = self._vectors @ query_vector
        top_indices = np.argsort(-scores)[:top_k]

        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self._chunks)