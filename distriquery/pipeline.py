"""Pipeline.

Ties the whole MVP together: ingest_document() runs load -> chunk -> embed
-> store, and answer() runs embed-query -> retrieve -> generate.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import List

from docx import settings

from distriquery.chunker import chunk_text
from distriquery.config import Settings, settings as default_settings
from distriquery.embedder import Embedder, get_embedder
from distriquery.generator import FakeLLMClient, LLMClient, build_prompt, get_llm_client
from distriquery.loader import load_document
from distriquery.vectorstore import InMemoryVectorStore, SearchResult, VectorStore


@dataclass
class Citation:
    chunk_id: str
    source: str
    position: int
    snippet: str


@dataclass
class RetrievalTrace:
    strategy: str
    chunks_considered: int
    chunks_used: int


@dataclass
class AnswerPayload:
    answer: str
    citations: List[Citation]
    retrieval: RetrievalTrace
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Pipeline:
    def __init__(
        self,
        settings: Settings = None,
        embedder: Embedder = None,
        llm_client: LLMClient = None,
        vector_store: VectorStore = None,
    ):
        self.settings = settings or default_settings
        self.embedder = embedder or get_embedder(
            self.settings.embedding_backend,
            dim=self.settings.embedding_dim,
            model_name=self.settings.sentence_transformer_model,
        )
        self.llm_client = llm_client or get_llm_client(
            self.settings.llm_backend, model=self.settings.anthropic_model
        )
        self.vector_store = vector_store if vector_store is not None else InMemoryVectorStore()
    
    def ingest_document(self, path: str) -> int:
        """Load, chunk, embed, and store a document. Returns the number of chunks created."""
        document = load_document(path)
        chunks = chunk_text(
            document.text,
            source=document.source,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            return 0

        vectors = self.embedder.embed([c.text for c in chunks])
        self.vector_store.add(chunks, vectors)
        return len(chunks)

    def answer(self, question: str, top_k: int = None) -> AnswerPayload:
        top_k = top_k or self.settings.top_k
        start = time.perf_counter()

        query_vector = self.embedder.embed([question])[0]
        results: List[SearchResult] = self.vector_store.search(query_vector, top_k=top_k)

        retrieval_ms = (time.perf_counter() - start) * 1000
        generation_start = time.perf_counter()

        if isinstance(self.llm_client, FakeLLMClient):
            answer_text = FakeLLMClient.extractive_answer(question, results)
        else:
            prompt = build_prompt(question, results)
            answer_text = self.llm_client.generate(prompt)

        generation_ms = (time.perf_counter() - generation_start) * 1000

        citations = [
            Citation(
                chunk_id=r.chunk.chunk_id,
                source=r.chunk.source,
                position=r.chunk.position,
                snippet=r.chunk.text[:200],
            )
            for r in results
        ]

        return AnswerPayload(
            answer=answer_text,
            citations=citations,
            retrieval=RetrievalTrace(
                strategy="dense",
                chunks_considered=len(self.vector_store),
                chunks_used=len(results),
            ),
            metrics={
                "latency_ms": {"retrieval": round(retrieval_ms, 2), "generation": round(generation_ms, 2)}
            },
        )