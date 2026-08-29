"""Per-tenant pipeline registry.

Phase A/B used one shared Pipeline — and therefore one shared in-memory
vector store — for everyone. That's exactly the isolation failure mode
Theory Primer Part 5.2 warns about: two tenants' chunks sitting in the
same searchable index with nothing preventing a query from crossing
between them.

This mirrors the "collection-per-tenant" pattern the Phase 4 design
recommended for Qdrant at small tenant counts — one Pipeline (and
therefore one InMemoryVectorStore) per tenant_id, created lazily. When
Phase D swaps in real Qdrant, this registry is what gets replaced by
per-tenant collections — api/main.py won't need to change.
"""

from typing import Dict

from distriquery.config import settings
from distriquery.pipeline import Pipeline

_pipelines: Dict[int, Pipeline] = {}


def get_pipeline_for_tenant(tenant_id: int) -> Pipeline:
    if tenant_id not in _pipelines:
        _pipelines[tenant_id] = Pipeline(settings=settings)
    return _pipelines[tenant_id]


def reset_registry() -> None:
    """Test-only helper: clears all tenant pipelines between test runs."""
    _pipelines.clear()