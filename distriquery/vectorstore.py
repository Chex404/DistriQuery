"""Vector storage.

Two implementations behind the same interface — same pluggable-backend
pattern as embedder.py and generator.py:

- InMemoryVectorStore: a Python list + numpy array. Works fine as long as
  ingestion and querying happen in the SAME process. Once Phase D splits
  ingestion into a separate Kafka consumer process, this stops working —
  the worker's copy and the API server's copy would never see each other.

- QdrantVectorStore: a real external vector database both the API server
  and ingestion workers can independently connect to, the same way MySQL
  already is. This is what makes cross-process ingestion actually work.

Search in InMemoryVectorStore is brute-force cosine similarity (Theory
Primer, Part 1.3/1.4) — exact, not approximate, fine at small scale, and
exactly the O(N) approach Qdrant's ANN index (HNSW) replaces once a
tenant's corpus gets large.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from distriquery.chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self._chunks: List[Chunk] = []
        self._vectors: Optional[np.ndarray] = None

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Number of chunks must match number of vectors")

        self._chunks.extend(chunks)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        if self._vectors is None or len(self._chunks) == 0:
            return []

        scores = self._vectors @ query_vector
        top_indices = np.argsort(-scores)[:top_k]

        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self._chunks)


class QdrantVectorStore(VectorStore):
    """Real vector persistence via Qdrant — one COLLECTION PER TENANT (Phase 4
    design decision: simplest correctness story at small tenant counts,
    vs. one shared collection with metadata filtering at large scale).

    Not exercised in this sandbox (no Docker access here) — this is written
    carefully against the documented qdrant-client API but genuinely
    untested until you run it against your real Qdrant container.
    """

    def __init__(self, collection_name: str, dim: int, url: str = "http://localhost:6333"):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url)
        self._collection_name = collection_name

        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=vector.tolist(),
                payload={"text": chunk.text, "source": chunk.source, "position": chunk.position},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector.tolist(),
            limit=top_k,
        ).points

        return [
            SearchResult(
                chunk=Chunk(
                    chunk_id=str(point.id),
                    text=point.payload["text"],
                    source=point.payload["source"],
                    position=point.payload["position"],
                ),
                score=point.score,
            )
            for point in results
        ]

    def __len__(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count