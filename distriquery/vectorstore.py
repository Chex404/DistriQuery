"""Vector storage.

Two implementations behind the same interface — same pluggable-backend
pattern as embedder.py and generator.py.
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
    """Keyed by chunk_id, upsert semantics — adding a chunk_id that already
    exists REPLACES it rather than duplicating it. This matches Qdrant's
    real upsert() behavior, which matters now that chunk IDs are
    deterministic (chunker.py): re-ingesting the same source should
    overwrite existing points, not accumulate duplicates, regardless of
    which vector store backend is in use.
    """

    def __init__(self):
        self._chunks_by_id: dict = {}
        self._vectors_by_id: dict = {}

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Number of chunks must match number of vectors")

        for chunk, vector in zip(chunks, vectors):
            self._chunks_by_id[chunk.chunk_id] = chunk
            self._vectors_by_id[chunk.chunk_id] = vector

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        if not self._chunks_by_id:
            return []

        ids = list(self._chunks_by_id.keys())
        vectors = np.array([self._vectors_by_id[chunk_id] for chunk_id in ids])

        scores = vectors @ query_vector
        top_indices = np.argsort(-scores)[:top_k]

        return [
            SearchResult(chunk=self._chunks_by_id[ids[i]], score=float(scores[i]))
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._chunks_by_id)


class QdrantVectorStore(VectorStore):
    """Real vector persistence via Qdrant — one COLLECTION PER TENANT."""

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
                id=chunk.chunk_id,  # deterministic uuid5 (chunker.py) — a valid Qdrant point ID,
                                    # and upserting the same ID again overwrites rather than duplicates
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