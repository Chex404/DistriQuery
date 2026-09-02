"""Core ingestion-worker logic — the actual work done for one Kafka event.

Lives inside the package (not in scripts/) so it's a normal, directly-
importable, directly-testable function. scripts/ingestion_worker.py is
the thin CLI wrapper that actually runs a Kafka consumer loop calling
this — same pattern as scripts/ask.py wrapping pipeline.py.
"""

from distriquery.db.models import Document
from distriquery.db.session import SessionLocal
from distriquery.tenancy import get_pipeline_for_tenant


def process_event(event: dict, db=None) -> int:
    """Do the actual ingestion for one event. Returns the chunk count.

    On any failure, marks the Document row "failed" before re-raising.
    Note: does NOT yet handle being called twice for the same event
    (Kafka's at-least-once delivery means that WILL happen under real
    failure conditions) — true idempotency is the next hardening step.
    """
    tenant_id = event["tenant_id"]
    document_id = event["document_id"]
    path = event["path"]

    owns_session = db is None
    db = db or SessionLocal()
    try:
        pipeline = get_pipeline_for_tenant(tenant_id)
        chunk_count = pipeline.ingest_document(path)

        document = db.query(Document).filter(Document.id == document_id).one_or_none()
        if document is not None:
            document.status = "ingested"
            document.chunk_count = chunk_count
            db.commit()

        return chunk_count
    except Exception:
        document = db.query(Document).filter(Document.id == document_id).one_or_none()
        if document is not None:
            document.status = "failed"
            db.commit()
        raise
    finally:
        if owns_session:
            db.close()