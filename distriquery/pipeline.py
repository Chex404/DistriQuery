"""Pipeline.

Ties the whole system together: ingest_document() runs load -> chunk ->
embed -> store, and answer() runs plan -> retrieve -> rerank -> generate.

Phase F adds an opt-in query planner (Theory Primer, Part 3.4) that routes
each query to a strategy before any retrieval happens:
- "tool": skip retrieval and generation entirely — a tool answers directly.
- "multi_hop": chain evidence across multiple retrieval hops.
- "direct": the existing hybrid+rerank pipeline from Phase E, unchanged.

use_agent defaults to False (via settings.agent_enabled), so every
existing call site and test keeps working exactly as before unless this
is explicitly turned on — same toggle pattern as Phase E's `rerank` flag.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import List

from distriquery.agents.multi_hop import MultiHopRetriever
from distriquery.agents.planner import HeuristicQueryPlanner, PlanDecision
from distriquery.agents.tools import get_tool
from distriquery.chunker import chunk_text
from distriquery.config import Settings, settings as default_settings
from distriquery.embedder import Embedder, get_embedder
from distriquery.generator import FakeLLMClient, LLMClient, build_prompt, get_llm_client
from distriquery.loader import load_document
from distriquery.reranker import Reranker, get_reranker
from distriquery.retriever import HybridRetriever
from distriquery.vectorstore import InMemoryVectorStore, SearchResult, VectorStore


@dataclass
class Citation:
    chunk_id: str
    source: str
    position: int
    snippet: str


@dataclass
class RetrievalTrace:
    strategy: str  # "hybrid" | "multi_hop" | "tool"
    reranked: bool
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
        reranker: Reranker = None,
        query_planner: HeuristicQueryPlanner = None,
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
        self.retriever = HybridRetriever(self.embedder, self.vector_store)
        self.reranker = reranker or get_reranker(
            self.settings.reranker_backend, model_name=self.settings.cross_encoder_model
        )
        self.query_planner = query_planner or HeuristicQueryPlanner()
        self.multi_hop_retriever = MultiHopRetriever(self.retriever, max_hops=self.settings.max_hops)

    def ingest_document(self, path: str) -> int:
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

    def answer(
        self,
        question: str,
        top_k: int = None,
        rerank: bool = None,
        use_agent: bool = None,
    ) -> AnswerPayload:
        top_k = top_k or self.settings.top_k
        rerank = self.settings.reranking_enabled if rerank is None else rerank
        use_agent = self.settings.agent_enabled if use_agent is None else use_agent

        start = time.perf_counter()

        decision = self.query_planner.plan(question) if use_agent else PlanDecision(strategy="direct")

        if decision.strategy == "tool":
            tool = get_tool(decision.tool_name)
            answer_text = tool.run(question)
            total_ms = (time.perf_counter() - start) * 1000
            return AnswerPayload(
                answer=answer_text,
                citations=[],
                retrieval=RetrievalTrace(
                    strategy="tool", reranked=False, chunks_considered=0, chunks_used=0
                ),
                metrics={"latency_ms": {"retrieval": 0.0, "rerank": 0.0, "generation": round(total_ms, 2)}},
            )

        if decision.strategy == "multi_hop":
            shortlist_k = max(top_k * self.settings.rerank_shortlist_multiplier, 20) if rerank else top_k
            shortlist = self.multi_hop_retriever.retrieve(question, top_k=shortlist_k)
            strategy_label = "multi_hop"
        else:
            shortlist_k = max(top_k * self.settings.rerank_shortlist_multiplier, 20) if rerank else top_k
            shortlist = self.retriever.retrieve(question, top_k=shortlist_k)
            strategy_label = "hybrid"

        retrieval_ms = (time.perf_counter() - start) * 1000

        if rerank:
            rerank_start = time.perf_counter()
            results: List[SearchResult] = self.reranker.rerank(question, shortlist, top_k=top_k)
            rerank_ms = (time.perf_counter() - rerank_start) * 1000
        else:
            results = shortlist[:top_k]
            rerank_ms = 0.0

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
                strategy=strategy_label,
                reranked=rerank,
                chunks_considered=len(self.vector_store),
                chunks_used=len(results),
            ),
            metrics={
                "latency_ms": {
                    "retrieval": round(retrieval_ms, 2),
                    "rerank": round(rerank_ms, 2),
                    "generation": round(generation_ms, 2),
                }
            },
        )