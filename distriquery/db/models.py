"""Database models.

Just one table for Phase B: Document metadata. Deliberately does NOT have
a tenant_id column yet — that's Phase C's job, along with the isolation
logic that actually enforces it. Adding an unenforced column now would be
worse than not having it at all: it would look like it provides isolation
without actually doing so.
"""

from sqlalchemy import Column, DateTime, Integer, String, func

from distriquery.db.session import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(255), unique=True, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())