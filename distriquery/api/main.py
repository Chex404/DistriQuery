"""FastAPI application — Phase D.

/documents no longer does the actual parsing/chunking/embedding — it
saves the file, creates a "pending" Document row, and publishes an event
to Kafka. A separate process (scripts/ingestion_worker.py) consumes that
event and does the real work, writing to the same Qdrant collection this
API's /query endpoint reads from.
"""

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from distriquery.api.auth import get_current_tenant
from distriquery.config import settings
from distriquery.db.models import Document, Tenant
from distriquery.db.session import get_db
from distriquery.loader import SUPPORTED_EXTENSIONS
from distriquery.pipeline import Pipeline
from distriquery.tenancy import get_pipeline_for_tenant

app = FastAPI(title="DistriQuery API", version="phase-d")

_kafka_producer = None


def get_kafka_producer():
    global _kafka_producer
    if _kafka_producer is None:
        from kafka import KafkaProducer

        _kafka_producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    return _kafka_producer


def get_tenant_pipeline(tenant: Tenant = Depends(get_current_tenant)) -> Pipeline:
    return get_pipeline_for_tenant(tenant.id)


def _upload_dir() -> Path:
    path = Path(os.environ.get("UPLOAD_DIR", "uploads"))
    path.mkdir(parents=True, exist_ok=True)
    return path

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    rerank: Optional[bool] = None
    use_agent: Optional[bool] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents")
def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    producer=Depends(get_kafka_producer),
):
    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    tenant_dir = _upload_dir() / str(tenant.id)
    tenant_dir.mkdir(parents=True, exist_ok=True)
    dest_path = tenant_dir / file.filename
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    existing = (
        db.query(Document)
        .filter(Document.tenant_id == tenant.id, Document.source == str(dest_path))
        .one_or_none()
    )
    if existing:
        existing.version += 1
        existing.status = "pending"
        document = existing
    else:
        document = Document(tenant_id=tenant.id, source=str(dest_path), version=1, status="pending")
        db.add(document)

    db.commit()
    db.refresh(document)

    event = {
        "event_id": str(uuid.uuid4()),
        "tenant_id": tenant.id,
        "document_id": document.id,
        "path": str(dest_path),
    }
    future = producer.send(settings.kafka_ingestion_topic, event)
    future.get(timeout=10)

    return {
        "document_id": document.id,
        "tenant_id": tenant.id,
        "source": document.source,
        "version": document.version,
        "status": document.status,
    }


@app.get("/documents/{document_id}")
def get_document_status(
    document_id: int,
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.tenant_id == tenant.id)
        .one_or_none()
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": document.id,
        "source": document.source,
        "version": document.version,
        "status": document.status,
        "chunk_count": document.chunk_count,
    }


@app.post("/query")
def query(
    request: QueryRequest,
    tenant: Tenant = Depends(get_current_tenant),
    pipeline: Pipeline = Depends(get_tenant_pipeline),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")

    payload = pipeline.answer(
        request.question, top_k=request.top_k, rerank=request.rerank, use_agent=request.use_agent
    )
    return payload.to_dict()