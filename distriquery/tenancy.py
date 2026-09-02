"""Per-tenant pipeline registry.

As of Phase D, this registry is used by BOTH the FastAPI query path and
the Kafka ingestion worker — building a QdrantVectorStore pointed at the
same URL/collection naming scheme in both places is what lets a worker's
ingestion actually become visible to the API's queries.
"""

from typing import Dict

from distriquery.config import settings
from distriquery.embedder import get_embedder
from distriquery.pipeline import Pipeline
from distriquery.vectorstore import InMemoryVectorStore, VectorStore

_pipelines: Dict[int, Pipeline] = {}


def _build_vector_store(tenant_id: int, dim: int) -> VectorStore:
    if settings.vector_store_backend == "qdrant":
        from distriquery.vectorstore import QdrantVectorStore

        return QdrantVectorStore(
            collection_name=f"tenant_{tenant_id}", dim=dim, url=settings.qdrant_url
        )
    return InMemoryVectorStore()


def get_pipeline_for_tenant(tenant_id: int) -> Pipeline:
    if tenant_id not in _pipelines:
        embedder = get_embedder(
            settings.embedding_backend,
            dim=settings.embedding_dim,
            model_name=settings.sentence_transformer_model,
        )
        vector_store = _build_vector_store(tenant_id, embedder.dim)
        _pipelines[tenant_id] = Pipeline(
            settings=settings, embedder=embedder, vector_store=vector_store
        )
    return _pipelines[tenant_id]


def reset_registry() -> None:
    """Test-only helper: clears all tenant pipelines between test runs."""
    _pipelines.clear()