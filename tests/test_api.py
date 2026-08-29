import io


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
    assert body["chunk_count"] > 0


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


def test_query_after_upload_returns_answer(client, create_tenant_helper):
    _, api_key = create_tenant_helper(name="acme")
    headers = {"X-API-Key": api_key}

    file_content = b"The query planner decides between direct, hybrid, and multi-hop retrieval."
    client.post(
        "/documents",
        files={"file": ("doc.txt", io.BytesIO(file_content), "text/plain")},
        headers=headers,
    )

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