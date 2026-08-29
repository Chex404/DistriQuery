"""Tests that specifically try to break tenant isolation."""

import io

from distriquery.tenancy import get_pipeline_for_tenant, reset_registry


def test_query_never_returns_another_tenants_content(client, create_tenant_helper):
    _, key_a = create_tenant_helper(name="acme")
    _, key_b = create_tenant_helper(name="globex")

    secret_content = b"Acme's confidential merger price is $42 per share."
    client.post(
        "/documents",
        files={"file": ("secret.txt", io.BytesIO(secret_content), "text/plain")},
        headers={"X-API-Key": key_a},
    )

    response = client.post(
        "/query",
        json={"question": "What is the confidential merger price?"},
        headers={"X-API-Key": key_b},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert "42" not in body["answer"]


def test_each_tenant_only_sees_their_own_uploaded_documents(client, create_tenant_helper):
    _, key_a = create_tenant_helper(name="acme")
    _, key_b = create_tenant_helper(name="globex")

    client.post(
        "/documents",
        files={"file": ("a.txt", io.BytesIO(b"Content belonging to tenant A only."), "text/plain")},
        headers={"X-API-Key": key_a},
    )
    client.post(
        "/documents",
        files={"file": ("b.txt", io.BytesIO(b"Content belonging to tenant B only."), "text/plain")},
        headers={"X-API-Key": key_b},
    )

    response_a = client.post(
        "/query", json={"question": "tenant content"}, headers={"X-API-Key": key_a}
    )
    response_b = client.post(
        "/query", json={"question": "tenant content"}, headers={"X-API-Key": key_b}
    )

    sources_a = {c["source"] for c in response_a.json()["citations"]}
    sources_b = {c["source"] for c in response_b.json()["citations"]}

    assert all("a.txt" in s for s in sources_a)
    assert all("b.txt" in s for s in sources_b)
    assert sources_a.isdisjoint(sources_b)


def test_uploading_identically_named_files_does_not_collide_on_disk(client, create_tenant_helper):
    _, key_a = create_tenant_helper(name="acme")
    _, key_b = create_tenant_helper(name="globex")

    response_a = client.post(
        "/documents",
        files={"file": ("report.txt", io.BytesIO(b"Tenant A's report content here."), "text/plain")},
        headers={"X-API-Key": key_a},
    )
    response_b = client.post(
        "/documents",
        files={"file": ("report.txt", io.BytesIO(b"Tenant B's totally different report."), "text/plain")},
        headers={"X-API-Key": key_b},
    )

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert response_a.json()["version"] == 1
    assert response_b.json()["version"] == 1
    assert response_a.json()["source"] != response_b.json()["source"]


def test_pipeline_registry_gives_distinct_instances_per_tenant():
    reset_registry()

    pipeline_1 = get_pipeline_for_tenant(1)
    pipeline_2 = get_pipeline_for_tenant(2)
    pipeline_1_again = get_pipeline_for_tenant(1)

    assert pipeline_1 is not pipeline_2
    assert pipeline_1 is pipeline_1_again

    reset_registry()