from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from distriquery.db.models import Document, Tenant
from distriquery.db.session import Base


def _sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _make_tenant(db, name="acme", api_key="key-123"):
    tenant = Tenant(name=name, api_key=api_key)
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


def test_create_document_defaults():
    db = _sqlite_session()
    tenant = _make_tenant(db)

    doc = Document(tenant_id=tenant.id, source="uploads/sample.txt", chunk_count=4)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    assert doc.id is not None
    assert doc.tenant_id == tenant.id
    assert doc.version == 1
    assert doc.chunk_count == 4
    assert doc.created_at is not None


def test_document_without_tenant_id_is_rejected():
    db = _sqlite_session()

    db.add(Document(source="uploads/orphan.txt", chunk_count=1))
    try:
        db.commit()
        assert False, "expected IntegrityError for missing tenant_id"
    except IntegrityError:
        db.rollback()


def test_source_must_be_unique_per_tenant():
    db = _sqlite_session()
    tenant = _make_tenant(db)

    db.add(Document(tenant_id=tenant.id, source="uploads/dup.txt", chunk_count=1))
    db.commit()

    db.add(Document(tenant_id=tenant.id, source="uploads/dup.txt", chunk_count=2))
    try:
        db.commit()
        assert False, "expected IntegrityError for duplicate (tenant_id, source)"
    except IntegrityError:
        db.rollback()


def test_same_source_allowed_across_different_tenants():
    db = _sqlite_session()
    tenant_a = _make_tenant(db, name="acme", api_key="key-a")
    tenant_b = _make_tenant(db, name="globex", api_key="key-b")

    db.add(Document(tenant_id=tenant_a.id, source="uploads/report.pdf", chunk_count=3))
    db.commit()

    db.add(Document(tenant_id=tenant_b.id, source="uploads/report.pdf", chunk_count=5))
    db.commit()

    assert db.query(Document).count() == 2


def test_version_bump_on_reingest():
    db = _sqlite_session()
    tenant = _make_tenant(db)

    doc = Document(tenant_id=tenant.id, source="uploads/sample.txt", chunk_count=4)
    db.add(doc)
    db.commit()

    doc.version += 1
    doc.chunk_count = 6
    db.commit()
    db.refresh(doc)

    assert doc.version == 2
    assert doc.chunk_count == 6


def test_tenant_api_key_must_be_unique():
    db = _sqlite_session()
    _make_tenant(db, name="acme", api_key="shared-key")

    db.add(Tenant(name="globex", api_key="shared-key"))
    try:
        db.commit()
        assert False, "expected IntegrityError for duplicate api_key"
    except IntegrityError:
        db.rollback()