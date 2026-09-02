import io
from pathlib import Path
import pytest


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_documents_without_api_key_is_rejected(client):
    response = client.post(
        "/documents", files={"file": ("t.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert response.status_code == 401


def test_query_with_invalid_api_key_returns_401(client):
    response = client.post(
        "/query", json={"question": "hi"}, headers={"X-API-Key": "not-a-real-key"}
    )
    assert response.status_code == 401


def test_upload_document_with_valid_key(client, create_tenant_helper):
    tenant_id, api_key = create_tenant_helper(name="acme")

    file_content = b"Kafka partitions events across a distributed cluster."
    response = client.post(
        "/documents",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == tenant_id
    assert body["version"] == 1
    assert body["status"] == "pending"


def test_document_status_becomes_ingested_after_worker_processes_it(
    client, create_tenant_helper, process_worker_events
):
    _, api_key = create_tenant_helper(name="acme")
    headers = {"X-API-Key": api_key}

    upload_response = client.post(
        "/documents",
        files={"file": ("doc.txt", io.BytesIO(b"Some content here."), "text/plain")},
        headers=headers,
    )
    document_id = upload_response.json()["document_id"]

    status_response = client.get(f"/documents/{document_id}", headers=headers)
    assert status_response.json()["status"] == "pending"

    process_worker_events()

    status_response = client.get(f"/documents/{document_id}", headers=headers)
    body = status_response.json()
    assert body["status"] == "ingested"
    assert body["chunk_count"] > 0


def test_document_status_returns_404_for_another_tenants_document(client, create_tenant_helper):
    _, key_a = create_tenant_helper(name="acme")
    _, key_b = create_tenant_helper(name="globex")

    upload_response = client.post(
        "/documents",
        files={"file": ("doc.txt", io.BytesIO(b"tenant A's document"), "text/plain")},
        headers={"X-API-Key": key_a},
    )
    document_id = upload_response.json()["document_id"]

    response = client.get(f"/documents/{document_id}", headers={"X-API-Key": key_b})
    assert response.status_code == 404


def test_reuploading_same_filename_bumps_version(client, create_tenant_helper):
    _, api_key = create_tenant_helper(name="acme")
    headers = {"X-API-Key": api_key}

    file_content = b"Some content about distributed systems."
    client.post(
        "/documents",
        files={"file": ("doc.txt", io.BytesIO(file_content), "text/plain")},
        headers=headers,
    )
    response = client.post(
        "/documents",
        files={"file": ("doc.txt", io.BytesIO(file_content), "text/plain")},
        headers=headers,
    )

    assert response.json()["version"] == 2


def test_query_after_upload_returns_answer(client, create_tenant_helper, process_worker_events):
    _, api_key = create_tenant_helper(name="acme")
    headers = {"X-API-Key": api_key}

    file_content = b"The query planner decides between direct, hybrid, and multi-hop retrieval."
    client.post(
        "/documents",
        files={"file": ("doc.txt", io.BytesIO(file_content), "text/plain")},
        headers=headers,
    )
    process_worker_events()

    response = client.post(
        "/query", json={"question": "What does the query planner decide?"}, headers=headers
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert len(body["citations"]) > 0

def test_query_with_empty_question_returns_400(client, create_tenant_helper):
    _, api_key = create_tenant_helper(name="acme")

    response = client.post("/query", json={"question": "   "}, headers={"X-API-Key": api_key})

    assert response.status_code == 400


def test_query_on_empty_index_does_not_crash(client, create_tenant_helper):
    _, api_key = create_tenant_helper(name="acme")

    response = client.post("/query", json={"question": "anything"}, headers={"X-API-Key": api_key})

    assert response.status_code == 200
    assert response.json()["citations"] == []

def test_worker_marks_document_failed_on_processing_error(
    client, create_tenant_helper, process_worker_events
):
    _, api_key = create_tenant_helper(name="acme")
    headers = {"X-API-Key": api_key}

    upload_response = client.post(
        "/documents",
        files={"file": ("doc.txt", io.BytesIO(b"some content"), "text/plain")},
        headers=headers,
    )
    document_id = upload_response.json()["document_id"]
    source_path = upload_response.json()["source"]

    Path(source_path).unlink()

    with pytest.raises(FileNotFoundError):
        process_worker_events()

    status_response = client.get(f"/documents/{document_id}", headers=headers)
    assert status_response.json()["status"] == "failed"


def test_uploading_unsupported_file_type_returns_400_not_500(client, create_tenant_helper):
    _, api_key = create_tenant_helper(name="acme")

    response = client.post(
        "/documents",
        files={"file": ("archive.zip", io.BytesIO(b"not a real zip either"), "application/zip")},
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]