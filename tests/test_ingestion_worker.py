"""Tests for the hardening pieces added on top of Phase D's basic flow:
idempotency (safe against duplicate Kafka delivery) and retry/DLQ handling.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from distriquery.db.models import Document, ProcessedEvent, Tenant
from distriquery.db.session import Base
from distriquery.ingestion_worker import handle_event_with_retries, process_event
from distriquery.tenancy import get_pipeline_for_tenant, reset_registry


class FakeFuture:
    def get(self, timeout=None):
        return None


class FakeProducer:
    def __init__(self):
        self.sent_events = []

    def send(self, topic, value):
        self.sent_events.append((topic, value))
        return FakeFuture()

    def flush(self):
        pass


def _sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _setup_tenant_and_document(db, tmp_path, content=b"Kafka partitions events across brokers."):
    tenant = Tenant(name="acme", api_key="key-123")
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    file_path = tmp_path / "doc.txt"
    file_path.write_bytes(content)

    document = Document(tenant_id=tenant.id, source=str(file_path), status="pending")
    db.add(document)
    db.commit()
    db.refresh(document)

    return tenant, document


def test_process_event_is_idempotent_for_duplicate_event_id(tmp_path):
    reset_registry()
    db = _sqlite_session()
    tenant, document = _setup_tenant_and_document(db, tmp_path)

    event = {
        "event_id": "same-event-id-both-times",
        "tenant_id": tenant.id,
        "document_id": document.id,
        "path": document.source,
    }

    first_count = process_event(event, db=db)
    pipeline = get_pipeline_for_tenant(tenant.id)
    size_after_first = len(pipeline.vector_store)

    second_count = process_event(event, db=db)
    size_after_second = len(pipeline.vector_store)

    assert first_count == second_count
    assert size_after_first == size_after_second

    processed_rows = db.query(ProcessedEvent).filter(
        ProcessedEvent.event_id == "same-event-id-both-times"
    ).count()
    assert processed_rows == 1

    reset_registry()


def test_reingesting_same_source_overwrites_via_deterministic_chunk_ids(tmp_path):
    reset_registry()
    db = _sqlite_session()
    tenant, document = _setup_tenant_and_document(db, tmp_path)

    event_1 = {
        "event_id": "event-1",
        "tenant_id": tenant.id,
        "document_id": document.id,
        "path": document.source,
    }
    event_2 = {
        "event_id": "event-2",
        "tenant_id": tenant.id,
        "document_id": document.id,
        "path": document.source,
    }

    process_event(event_1, db=db)
    pipeline = get_pipeline_for_tenant(tenant.id)
    size_after_first = len(pipeline.vector_store)

    process_event(event_2, db=db)
    size_after_second = len(pipeline.vector_store)

    assert size_after_first == size_after_second

    reset_registry()


def test_handle_event_with_retries_succeeds_without_touching_dlq(tmp_path):
    reset_registry()
    db = _sqlite_session()
    tenant, document = _setup_tenant_and_document(db, tmp_path)
    dlq_producer = FakeProducer()

    event = {
        "event_id": "will-succeed",
        "tenant_id": tenant.id,
        "document_id": document.id,
        "path": document.source,
    }

    succeeded = handle_event_with_retries(
        event, dlq_producer, max_retries=3, backoff_seconds=0, db=db
    )

    assert succeeded is True
    assert dlq_producer.sent_events == []

    db.refresh(document)
    assert document.status == "ingested"

    reset_registry()


def test_handle_event_with_retries_sends_to_dlq_after_exhausting_attempts(tmp_path):
    reset_registry()
    db = _sqlite_session()
    tenant, document = _setup_tenant_and_document(db, tmp_path)
    dlq_producer = FakeProducer()

    event = {
        "event_id": "will-fail",
        "tenant_id": tenant.id,
        "document_id": document.id,
        "path": str(tmp_path / "does_not_exist.txt"),
    }

    succeeded = handle_event_with_retries(
        event, dlq_producer, dlq_topic="my-dlq-topic", max_retries=3, backoff_seconds=0, db=db
    )

    assert succeeded is False
    assert len(dlq_producer.sent_events) == 1

    topic, dlq_event = dlq_producer.sent_events[0]
    assert topic == "my-dlq-topic"
    assert dlq_event["event_id"] == "will-fail"
    assert "error" in dlq_event

    db.refresh(document)
    assert document.status == "failed"

    reset_registry()