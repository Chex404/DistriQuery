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

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        """Return every stored chunk (text + metadata, no vectors needed).

        This is what lets hybrid retrieval (Phase E) build a BM25 index
        without needing a separate persistent store — Qdrant already
        holds every chunk's text in its payload, so we can reuse it.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Remove every chunk belonging to a given source.

        Phase I found a real bug this fixes: deterministic chunk IDs
        (source + position) make upsert-based re-ingestion correctly
        overwrite positions that still exist in a new version — but if a
        document SHRINKS, positions beyond the new chunk count were never
        touched, leaving orphaned stale chunks silently retrievable.
        Calling this before every ingest_document() gives correct
        "replace" semantics instead of "upsert-only."
        """
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

    def get_all_chunks(self) -> List[Chunk]:
        return list(self._chunks_by_id.values())

    def delete_by_source(self, source: str) -> None:
        stale_ids = [
            chunk_id for chunk_id, chunk in self._chunks_by_id.items() if chunk.source == source
        ]
        for chunk_id in stale_ids:
            del self._chunks_by_id[chunk_id]
            del self._vectors_by_id[chunk_id]


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

    def get_all_chunks(self) -> List[Chunk]:
        chunks = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                chunks.append(
                    Chunk(
                        chunk_id=str(point.id),
                        text=point.payload["text"],
                        source=point.payload["source"],
                        position=point.payload["position"],
                    )
                )
            if offset is None:
                break
        return chunks

    def delete_by_source(self, source: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )