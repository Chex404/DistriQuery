"""Database session management.

DATABASE_URL defaults to the real Postgres instance from docker-compose.
For tests, we don't touch this default at all — instead, tests build their
own SQLite-backed engine/session and override the get_db dependency in
FastAPI. Same pattern as embedder.py/generator.py in Phase A: production
uses the real thing, tests use a fast, dependency-free stand-in.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "mysql+pymysql://distriquery:distriquery@localhost:3307/distriquery"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a session, always closes it afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables() -> None:
    from distriquery.db import models  # noqa: F401 - import registers Tenant/Document on Base.metadata

    Base.metadata.create_all(bind=engine)