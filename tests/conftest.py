"""Shared pytest fixtures."""

import secrets

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from distriquery.api.main import app, get_kafka_producer
from distriquery.db.models import Tenant
from distriquery.db.session import Base, get_db
from distriquery.tenancy import reset_registry


@pytest.fixture(autouse=True)
def _use_fast_test_backends(monkeypatch):
    """Forces every test to use fast, isolated backends (hashing embedder,
    in-memory vector store, fake LLM) regardless of what's in the
    developer's real .env file.

    Without this, tests silently used the REAL settings object — meaning
    real sentence-transformers model loads (slow) and, more seriously,
    the REAL persistent Qdrant instance, so leftover data from manual
    testing bled into automated test assertions ("empty index" tests
    weren't actually empty). monkeypatch.setattr on the shared settings
    object affects every module that imports it (tenancy.py, pipeline.py),
    and is automatically undone after each test.
    """
    from distriquery.config import settings

    monkeypatch.setattr(settings, "embedding_backend", "hashing")
    monkeypatch.setattr(settings, "embedding_dim", 128)
    monkeypatch.setattr(settings, "llm_backend", "fake")
    monkeypatch.setattr(settings, "vector_store_backend", "in-memory")

@pytest.fixture
def db_session_factory(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


class FakeKafkaFuture:
    def get(self, timeout=None):
        return None


class FakeKafkaProducer:
    def __init__(self):
        self.sent_events = []

    def send(self, topic, value):
        self.sent_events.append(value)
        return FakeKafkaFuture()

    def flush(self):
        pass


@pytest.fixture
def fake_kafka_producer():
    return FakeKafkaProducer()


@pytest.fixture
def client(tmp_path, monkeypatch, db_session_factory, fake_kafka_producer):
    def override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_kafka_producer] = lambda: fake_kafka_producer
    reset_registry()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    reset_registry()


@pytest.fixture
def create_tenant_helper(db_session_factory):
    def _create(name: str = "tenant") -> tuple:
        db = db_session_factory()
        try:
            api_key = secrets.token_hex(16)
            tenant = Tenant(name=name, api_key=api_key)
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
            return tenant.id, tenant.api_key
        finally:
            db.close()

    return _create


@pytest.fixture
def process_worker_events(db_session_factory, fake_kafka_producer):
    """Simulates the ingestion worker processing every event a test's
    upload(s) published — WITHOUT needing a real Kafka broker or a real
    running worker process.

    Uses handle_event_with_retries with backoff_seconds=0 (no real
    sleeping) and the same fake producer as the DLQ target, so tests can
    also inspect fake_kafka_producer.sent_events for DLQ'd events.
    """

    def _process_all():
        from distriquery.ingestion_worker import handle_event_with_retries

        db = db_session_factory()
        results = []
        try:
            events = list(fake_kafka_producer.sent_events)
            fake_kafka_producer.sent_events.clear()
            for event in events:
                succeeded = handle_event_with_retries(
                    event,
                    fake_kafka_producer,
                    dlq_topic="document-ingestion-dlq",
                    max_retries=3,
                    backoff_seconds=0,
                    db=db,
                )
                results.append(succeeded)
        finally:
            db.close()
        return results

    return _process_all