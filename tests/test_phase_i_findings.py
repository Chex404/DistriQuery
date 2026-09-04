"""Phase I findings, tracked as permanent tests."""

import tempfile
from pathlib import Path

from distriquery.experiments.harness import (
    experiment_settings,
    setup_tenant_and_document,
    sqlite_session,
)
from distriquery.ingestion_worker import process_event
from distriquery.pipeline import Pipeline
from distriquery.tenancy import get_pipeline_for_tenant, reset_registry


def test_finding_duplicate_delivery_does_not_change_corpus_size():
    reset_registry()
    db = sqlite_session()
    tenant, document = setup_tenant_and_document(db)

    event = {
        "event_id": "regression-test-fixed-event-id",
        "tenant_id": tenant.id,
        "document_id": document.id,
        "path": document.source,
    }

    process_event(event, db=db)
    pipeline = get_pipeline_for_tenant(tenant.id)
    size_after_first = len(pipeline.vector_store)

    for _ in range(4):
        process_event(event, db=db)

    size_after_five = len(pipeline.vector_store)

    assert size_after_first == size_after_five
    reset_registry()


def test_finding_shrinking_reingestion_no_longer_leaves_orphaned_stale_chunks():
    """UPDATED FINDING: this test originally documented a confirmed bug —
    re-ingesting a shorter document left orphaned chunks from the old,
    longer version. That bug has since been fixed: ingest_document() now
    calls vector_store.delete_by_source() before adding new chunks.
    """
    settings = experiment_settings()
    settings.chunk_size = 80
    settings.chunk_overlap = 10
    pipeline = Pipeline(settings=settings)

    with tempfile.TemporaryDirectory() as tmp_dir:
        doc_path = Path(tmp_dir) / "shrinking_doc.txt"

        long_version = (
            "Section one is the introduction to the whole system.\n\n"
            "Section two covers the ingestion pipeline in detail.\n\n"
            "Section three covers the query pipeline in detail.\n\n"
            "Section four covers reranking and generation in detail."
        )
        doc_path.write_text(long_version)
        pipeline.ingest_document(str(doc_path))
        stored_after_v1 = len(pipeline.vector_store)
        assert stored_after_v1 > 1

        short_version = "Just one short sentence now."
        doc_path.write_text(short_version)
        chunk_count_v2 = pipeline.ingest_document(str(doc_path))
        stored_after_v2 = len(pipeline.vector_store)

    assert chunk_count_v2 == 1
    assert stored_after_v2 == 1
    assert stored_after_v2 == chunk_count_v2