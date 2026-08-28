"""FastAPI application — Phase B.

Wraps the Phase A Pipeline in real HTTP endpoints and makes document
*metadata* durable in Postgres. Note what this does NOT do yet:

- Vectors are still in-memory only (lost on restart) — real persistence
  arrives in Phase D when Qdrant + Kafka-based ingestion replace this.
- There's no tenant scoping or auth — that's Phase C.
- Retrieval is still dense-only, one shared Pipeline instance for
  everyone — hybrid retrieval and per-tenant isolation both come later.

Table creation is deliberately NOT done automatically on app startup —
run `python scripts/init_db.py` once before starting the API for the
first time.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from distriquery.db.models import Document
from distriquery.db.session import get_db
from distriquery.pipeline import Pipeline

app = FastAPI(title="DistriQuery API", version="phase-b")

_default_pipeline = Pipeline()


def get_pipeline() -> Pipeline:
    return _default_pipeline


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
    pipeline: Pipeline = Depends(get_pipeline),
):
    dest_path = _upload_dir() / file.filename
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    chunk_count = pipeline.ingest_document(str(dest_path))

    existing = db.query(Document).filter(Document.source == str(dest_path)).one_or_none()
    if existing:
        existing.version += 1
        existing.chunk_count = chunk_count
        document = existing
    else:
        document = Document(source=str(dest_path), version=1, chunk_count=chunk_count)
        db.add(document)

    db.commit()
    db.refresh(document)

    return {
        "document_id": document.id,
        "source": document.source,
        "version": document.version,
        "chunk_count": document.chunk_count,
    }


@app.post("/query")
def query(request: QueryRequest, pipeline: Pipeline = Depends(get_pipeline)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")

    payload = pipeline.answer(request.question, top_k=request.top_k)
    return payload.to_dict()