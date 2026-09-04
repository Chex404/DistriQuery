from distriquery.config import Settings
from distriquery.pipeline import Pipeline
from distriquery.vectorstore import InMemoryVectorStore

def _test_settings(tmp_chunk_size: int = 300) -> Settings:
    return Settings(
        chunk_size=tmp_chunk_size,
        chunk_overlap=30,
        top_k=3,
        embedding_backend="hashing",
        embedding_dim=128,
        llm_backend="fake",
        vector_store_backend="in-memory",
        reranker_backend="overlap",
    )


def test_ingest_document_returns_chunk_count(tmp_path):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Kafka is a distributed event streaming platform. " * 20)

    pipeline = Pipeline(settings=_test_settings())
    chunk_count = pipeline.ingest_document(str(file_path))

    assert chunk_count > 0
    assert len(pipeline.vector_store) == chunk_count


def test_answer_returns_full_payload_shape(tmp_path):
    file_path = tmp_path / "doc.txt"
    file_path.write_text(
        "The query planner decides between direct, hybrid, multi-hop, and tool retrieval."
    )

    pipeline = Pipeline(settings=_test_settings())
    pipeline.ingest_document(str(file_path))

    payload = pipeline.answer("What does the query planner decide between?")

    assert payload.answer
    assert isinstance(payload.citations, list)
    assert len(payload.citations) > 0
    assert payload.retrieval.strategy == "hybrid"
    assert payload.retrieval.reranked is True
    assert "latency_ms" in payload.metrics
    assert "rerank" in payload.metrics["latency_ms"]

def test_answer_with_rerank_false_skips_reranking(tmp_path):
    file_path = tmp_path / "doc.txt"
    file_path.write_text(
        "The query planner decides between direct, hybrid, multi-hop, and tool retrieval."
    )

    pipeline = Pipeline(settings=_test_settings())
    pipeline.ingest_document(str(file_path))

    payload = pipeline.answer("What does the query planner decide between?", rerank=False)

    assert payload.retrieval.reranked is False
    assert payload.metrics["latency_ms"]["rerank"] == 0.0

def test_answer_on_empty_pipeline_does_not_crash():
    pipeline = Pipeline(settings=_test_settings())

    payload = pipeline.answer("anything at all")

    assert payload.citations == []
    assert "couldn't find" in payload.answer.lower()


def test_to_dict_produces_json_serializable_structure(tmp_path):
    import json

    file_path = tmp_path / "doc.txt"
    file_path.write_text("Reranking uses a cross-encoder to re-score retrieved chunks.")

    pipeline = Pipeline(settings=_test_settings())
    pipeline.ingest_document(str(file_path))
    payload = pipeline.answer("What does reranking use?")

    json.dumps(payload.to_dict())

def test_explicitly_passed_empty_vector_store_is_not_silently_replaced():
    """Regression test for a real bug: `vector_store or InMemoryVectorStore()`
    looks correct but isn't — Python's `or` checks truthiness, and any
    VectorStore with 0 items is falsy (because VectorStore defines
    __len__). That silently discarded a perfectly valid, explicitly-passed
    empty store (e.g. a fresh QdrantVectorStore for a new tenant) and
    replaced it with a brand new InMemoryVectorStore instead.
    """
    explicit_store = InMemoryVectorStore()
    assert len(explicit_store) == 0

    pipeline = Pipeline(settings=_test_settings(), vector_store=explicit_store)

    assert pipeline.vector_store is explicit_store