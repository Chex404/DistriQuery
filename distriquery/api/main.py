"""FastAPI application — Phase C.

Every endpoint except /health now requires a valid X-API-Key header
(auth.py), and every document/query operation is scoped to that tenant's
own Pipeline (tenancy.py) and own rows in the documents table. See
models.py's docstring for the MySQL-vs-Postgres row-level-security
tradeoff this design has to compensate for in application code.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from distriquery.api.auth import get_current_tenant
from distriquery.db.models import Document, Tenant
from distriquery.db.session import get_db
from distriquery.pipeline import Pipeline
from distriquery.tenancy import get_pipeline_for_tenant

app = FastAPI(title="DistriQuery API", version="phase-c")


def get_tenant_pipeline(tenant: Tenant = Depends(get_current_tenant)) -> Pipeline:
    return get_pipeline_for_tenant(tenant.id)


def _upload_dir() -> Path:
    path = Path(os.environ.get("UPLOAD_DIR", "uploads"))
    path.mkdir(parents=True, exist_ok=True)
    return path


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents")
def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    pipeline: Pipeline = Depends(get_tenant_pipeline),
):
    tenant_dir = _upload_dir() / str(tenant.id)
    tenant_dir.mkdir(parents=True, exist_ok=True)
    dest_path = tenant_dir / file.filename
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunk_count = pipeline.ingest_document(str(dest_path))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    existing = (
        db.query(Document)
        .filter(Document.tenant_id == tenant.id, Document.source == str(dest_path))
        .one_or_none()
    )
    if existing:
        existing.version += 1
        existing.chunk_count = chunk_count
        document = existing
    else:
        document = Document(
            tenant_id=tenant.id, source=str(dest_path), version=1, chunk_count=chunk_count
        )
        db.add(document)

    db.commit()
    db.refresh(document)

    return {
        "document_id": document.id,
        "tenant_id": tenant.id,
        "source": document.source,
        "version": document.version,
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

    payload = pipeline.answer(request.question, top_k=request.top_k)
    return payload.to_dict()