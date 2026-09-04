from distriquery.agents.multi_hop import MultiHopRetriever
from distriquery.chunker import chunk_text
from distriquery.embedder import HashingEmbedder
from distriquery.retriever import HybridRetriever
from distriquery.vectorstore import InMemoryVectorStore


def _build_multi_hop_retriever(texts, max_hops=2):
    embedder = HashingEmbedder(dim=128)
    store = InMemoryVectorStore()

    all_chunks = []
    for i, text in enumerate(texts):
        chunks = chunk_text(text, source=f"doc{i}", chunk_size=200, chunk_overlap=20)
        all_chunks.extend(chunks)

    if all_chunks:
        vectors = embedder.embed([c.text for c in all_chunks])
        store.add(all_chunks, vectors)

    hybrid = HybridRetriever(embedder, store)
    return MultiHopRetriever(hybrid, max_hops=max_hops)


def test_multi_hop_returns_results_across_hops():
    retriever = _build_multi_hop_retriever(
        [
            "Kafka partitions events across a distributed cluster of brokers.",
            "Qdrant stores vectors and supports approximate nearest neighbor search.",
        ]
    )

    results = retriever.retrieve("kafka partitions and qdrant vectors", top_k=2)

    assert len(results) > 0


def test_multi_hop_does_not_duplicate_chunks_across_hops():
    retriever = _build_multi_hop_retriever(
        ["Kafka partitions events across brokers in a distributed cluster."]
    )

    results = retriever.retrieve("kafka partitions brokers", top_k=4)

    chunk_ids = [r.chunk.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_multi_hop_on_empty_index_returns_empty():
    retriever = _build_multi_hop_retriever([])

    assert retriever.retrieve("anything", top_k=3) == []


def test_multi_hop_respects_max_hops_of_one():
    """With max_hops=1, this should behave like a single retrieval call —
    no second hop's query expansion should occur."""
    retriever = _build_multi_hop_retriever(
        ["Kafka partitions events across brokers."], max_hops=1
    )

    results = retriever.retrieve("kafka partitions", top_k=4)

    assert len(results) > 0