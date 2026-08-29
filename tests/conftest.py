"""Shared pytest fixtures.

pytest auto-discovers this file — test_api.py and test_multitenancy.py
use `client` and `create_tenant_helper` without importing anything from
here directly.
"""

import secrets

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from distriquery.api.main import app
from distriquery.db.models import Tenant
from distriquery.db.session import Base, get_db
from distriquery.tenancy import reset_registry


@pytest.fixture
def db_session_factory(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def client(tmp_path, monkeypatch, db_session_factory):
    def override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    app.dependency_overrides[get_db] = override_get_db
    reset_registry()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    reset_registry()


@pytest.fixture
def create_tenant_helper(db_session_factory):
    """Returns a callable: create_tenant_helper(name="x") -> (tenant_id, api_key)."""

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