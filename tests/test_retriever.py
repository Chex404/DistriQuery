from distriquery.chunker import chunk_text
from distriquery.embedder import HashingEmbedder
from distriquery.retriever import DenseRetriever
from distriquery.vectorstore import InMemoryVectorStore


def _build_retriever(texts):
    embedder = HashingEmbedder(dim=128)
    store = InMemoryVectorStore()

    all_chunks = []
    for i, text in enumerate(texts):
        chunks = chunk_text(text, source=f"doc{i}", chunk_size=200, chunk_overlap=20)
        all_chunks.extend(chunks)

    if all_chunks:
        vectors = embedder.embed([c.text for c in all_chunks])
        store.add(all_chunks, vectors)

    return DenseRetriever(embedder, store)


def test_retrieve_returns_relevant_chunk_first():
    retriever = _build_retriever(
        [
            "Kafka partitions events across a distributed cluster of brokers.",
            "A sourdough starter needs to be fed with flour and water daily.",
        ]
    )

    results = retriever.retrieve("how does kafka partition events", top_k=2)

    assert len(results) == 2
    assert "kafka" in results[0].chunk.text.lower() or "Kafka" in results[0].chunk.text


def test_retrieve_respects_top_k():
    retriever = _build_retriever(["one two three. " * 50])

    results = retriever.retrieve("one two three", top_k=1)

    assert len(results) <= 1


def test_retrieve_on_empty_index_returns_empty():
    retriever = _build_retriever([])

    results = retriever.retrieve("anything", top_k=3)

    assert results == []