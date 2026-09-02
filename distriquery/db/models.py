"""Database models.

Phase C adds Tenant and scopes Document to a tenant_id. Two things worth
flagging explicitly:

1. `source` was globally unique in Phase B; it's now unique PER TENANT
   (via the composite constraint below) — two tenants must be able to
   upload a file called "report.pdf" without colliding.

2. MySQL has no native row-level security (unlike Postgres, which is what
   the Phase 4 design assumed). That means tenant isolation here is
   enforced entirely in application code — every query in api/main.py
   must explicitly filter by tenant_id, with no database-level backstop
   if one ever forgets to. This is a real, documented tradeoff of using
   MySQL instead of Postgres, not a hidden gap — see tests/test_multitenancy.py
   for tests that specifically try to break isolation.
"""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint, func

from distriquery.db.session import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    api_key = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("tenant_id", "source", name="uq_tenant_source"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    source = Column(String(255), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    # "pending" -> published to Kafka, not yet processed by a worker
    # "ingested" -> a worker successfully chunked/embedded/stored it
    # "failed" -> a worker gave up (arrives with DLQ, next step)
    status = Column(String(20), nullable=False, default="pending")
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())