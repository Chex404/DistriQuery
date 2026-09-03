"""Core ingestion-worker logic.

Two separate concerns, deliberately kept as two functions:

- process_event(): does the actual ingestion for ONE attempt. Raises on
  failure. Idempotent: if this exact event_id has already been
  successfully processed, it skips straight to returning the existing
  chunk count instead of re-embedding and re-storing.

- handle_event_with_retries(): owns retry-with-backoff and, after
  exhausting retries, publishes to the dead-letter topic and marks the
  document "failed" for real. This is what scripts/ingestion_worker.py's
  consumer loop actually calls.
"""

import time

from distriquery.db.models import Document, ProcessedEvent
from distriquery.db.session import SessionLocal
from distriquery.tenancy import get_pipeline_for_tenant


def process_event(event: dict, db=None) -> int:
    event_id = event["event_id"]
    tenant_id = event["tenant_id"]
    document_id = event["document_id"]
    path = event["path"]

    owns_session = db is None
    db = db or SessionLocal()
    try:
        already_processed = (
            db.query(ProcessedEvent).filter(ProcessedEvent.event_id == event_id).one_or_none()
        )
        if already_processed is not None:
            document = db.query(Document).filter(Document.id == document_id).one_or_none()
            return document.chunk_count if document is not None else 0

        pipeline = get_pipeline_for_tenant(tenant_id)
        chunk_count = pipeline.ingest_document(path)

        document = db.query(Document).filter(Document.id == document_id).one_or_none()
        if document is not None:
            document.status = "ingested"
            document.chunk_count = chunk_count

        db.add(ProcessedEvent(event_id=event_id))
        db.commit()

        return chunk_count
    finally:
        if owns_session:
            db.close()


def mark_document_failed(document_id: int, db=None) -> None:
    owns_session = db is None
    db = db or SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).one_or_none()
        if document is not None:
            document.status = "failed"
            db.commit()
    finally:
        if owns_session:
            db.close()


def handle_event_with_retries(
    event: dict,
    dlq_producer,
    dlq_topic: str = "document-ingestion-dlq",
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    db=None,
) -> bool:
    """Retries process_event up to max_retries times with exponential
    backoff. On exhausting retries: marks the document "failed" and
    publishes the event (plus the error) to the DLQ topic. Returns True
    on eventual success, False if it went to DLQ.
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            process_event(event, db=db)
            return True
        except Exception as e:
            last_exception = e
            if attempt < max_retries and backoff_seconds > 0:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    mark_document_failed(event["document_id"], db=db)

    dlq_event = {**event, "error": str(last_exception)}
    future = dlq_producer.send(dlq_topic, dlq_event)
    future.get(timeout=10)
    dlq_producer.flush()

    return False