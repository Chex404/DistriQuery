from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from distriquery.db.models import Document
from distriquery.db.session import Base


def _sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_create_document_defaults():
    db = _sqlite_session()

    doc = Document(source="uploads/sample.txt", chunk_count=4)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    assert doc.id is not None
    assert doc.version == 1
    assert doc.chunk_count == 4
    assert doc.created_at is not None


def test_source_must_be_unique():
    from sqlalchemy.exc import IntegrityError

    db = _sqlite_session()
    db.add(Document(source="uploads/dup.txt", chunk_count=1))
    db.commit()

    db.add(Document(source="uploads/dup.txt", chunk_count=2))
    try:
        db.commit()
        assert False, "expected IntegrityError for duplicate source"
    except IntegrityError:
        db.rollback()


def test_version_bump_on_reingest():
    db = _sqlite_session()

    doc = Document(source="uploads/sample.txt", chunk_count=4)
    db.add(doc)
    db.commit()

    doc.version += 1
    doc.chunk_count = 6
    db.commit()
    db.refresh(doc)

    assert doc.version == 2
    assert doc.chunk_count == 6