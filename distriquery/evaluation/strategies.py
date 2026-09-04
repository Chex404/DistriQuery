"""Named retrieval strategies for the naive vs. hybrid vs. reranked vs. agentic comparison."""

from distriquery.pipeline import Pipeline
from distriquery.retriever import DenseRetriever


def naive_strategy(pipeline: Pipeline):
    dense = DenseRetriever(pipeline.embedder, pipeline.vector_store)

    def retrieve(query: str, top_k: int):
        results = dense.retrieve(query, top_k=top_k)
        return [r.chunk.position for r in results]

    return retrieve


def hybrid_strategy(pipeline: Pipeline):
    def retrieve(query: str, top_k: int):
        results = pipeline.retriever.retrieve(query, top_k=top_k)
        return [r.chunk.position for r in results]

    return retrieve


def reranked_strategy(pipeline: Pipeline):
    def retrieve(query: str, top_k: int):
        shortlist_k = max(top_k * pipeline.settings.rerank_shortlist_multiplier, 20)
        shortlist = pipeline.retriever.retrieve(query, top_k=shortlist_k)
        results = pipeline.reranker.rerank(query, shortlist, top_k=top_k)
        return [r.chunk.position for r in results]

    return retrieve


def agentic_strategy(pipeline: Pipeline):
    def retrieve(query: str, top_k: int):
        payload = pipeline.answer(query, top_k=top_k, use_agent=True)
        return [c.position for c in payload.citations]

    return retrieve


STRATEGIES = {
    "naive": naive_strategy,
    "hybrid": hybrid_strategy,
    "reranked": reranked_strategy,
    "agentic": agentic_strategy,
}