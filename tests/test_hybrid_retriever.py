from distriquery.chunker import chunk_text
from distriquery.embedder import HashingEmbedder
from distriquery.retriever import HybridRetriever
from distriquery.vectorstore import InMemoryVectorStore


def _build_hybrid_retriever(texts):
    embedder = HashingEmbedder(dim=128)
    store = InMemoryVectorStore()

    all_chunks = []
    for i, text in enumerate(texts):
        chunks = chunk_text(text, source=f"doc{i}", chunk_size=200, chunk_overlap=20)
        all_chunks.extend(chunks)

    if all_chunks:
        vectors = embedder.embed([c.text for c in all_chunks])
        store.add(all_chunks, vectors)

    return HybridRetriever(embedder, store)


def test_hybrid_finds_exact_term_match():
    retriever = _build_hybrid_retriever(
        [
            "Kafka partitions events across a distributed cluster of brokers.",
            "A sourdough starter needs to be fed with flour and water daily.",
        ]
    )

    results = retriever.retrieve("kafka partitions", top_k=2)

    assert len(results) >= 1
    assert "kafka" in results[0].chunk.text.lower()


def test_hybrid_respects_top_k():
    retriever = _build_hybrid_retriever(["one two three. " * 50])

    results = retriever.retrieve("one two three", top_k=1)

    assert len(results) <= 1


def test_hybrid_on_empty_index_returns_empty():
    retriever = _build_hybrid_retriever([])

    assert retriever.retrieve("anything", top_k=3) == []


def test_hybrid_deduplicates_chunks_found_by_both_methods():
    retriever = _build_hybrid_retriever(
        ["Kafka partitions events across brokers in a distributed cluster."]
    )

    results = retriever.retrieve("kafka distributed brokers", top_k=4)

    chunk_ids = [r.chunk.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_hybrid_recovers_a_case_the_hashing_embedder_alone_gets_wrong():
    document = (
        "Section one is about general project background and history.\n\n"
        "Section two covers unrelated topics like weather and travel plans.\n\n"
        "Your most-used AI tool? ChatGPT, used mainly for final checks.\n\n"
        "Section four discusses budgets and scheduling for next quarter."
    )
    retriever = _build_hybrid_retriever([document])

    results = retriever.retrieve("What is the most-used AI tool?", top_k=2)

    assert any("ChatGPT" in r.chunk.text for r in results)