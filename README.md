# DistriQuery

A distributed, multi-tenant RAG platform with agentic query orchestration.
This repo is being built incrementally, phase by phase. This README covers
what's built so far: **Phase A, the basic RAG MVP**.

## Setup

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests/ -v

## Running the MVP

python scripts/ask.py sample_data/sample.txt "What does the query planner decide between?"

## What's implemented (Phase A)

- distriquery/loader.py — reads .txt / .md / .pdf into plain text
- distriquery/chunker.py — recursive, structure-aware text splitting
- distriquery/embedder.py — pluggable embedding backends
- distriquery/vectorstore.py — in-memory cosine-similarity search
- distriquery/retriever.py — dense retrieval only
- distriquery/generator.py — pluggable LLM backends
- distriquery/pipeline.py — ties it all together

## Not implemented yet (by design)

api/, workers/, db/, agents/ are placeholders for later phases. No Kafka,
Postgres, real vector DB, multi-tenancy, reranker, or query planner yet.