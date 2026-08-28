import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from distriquery.api.main import app, get_db, get_pipeline
from distriquery.config import Settings
from distriquery.db.session import Base
from distriquery.pipeline import Pipeline


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    test_settings = Settings(embedding_backend="hashing", embedding_dim=64, llm_backend="fake")
    test_pipeline = Pipeline(settings=test_settings)

    def override_get_pipeline():
        return test_pipeline

    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_pipeline] = override_get_pipeline

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_document(client):
    file_content = b"Kafka partitions events across a distributed cluster."
    response = client.post(
        "/documents", files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 1
    assert body["chunk_count"] > 0


def test_reuploading_same_filename_bumps_version(client):
    file_content = b"Some content about distributed systems."
    client.post("/documents", files={"file": ("doc.txt", io.BytesIO(file_content), "text/plain")})
    response = client.post(
        "/documents", files={"file": ("doc.txt", io.BytesIO(file_content), "text/plain")}
    )

    assert response.json()["version"] == 2


def test_query_after_upload_returns_answer(client):
    file_content = b"The query planner decides between direct, hybrid, and multi-hop retrieval."
    client.post("/documents", files={"file": ("doc.txt", io.BytesIO(file_content), "text/plain")})

    response = client.post("/query", json={"question": "What does the query planner decide?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert len(body["citations"]) > 0


def test_query_with_empty_question_returns_400(client):
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 400


def test_query_on_empty_index_does_not_crash(client):
    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 200
    assert response.json()["citations"] == []