from distriquery.config import Settings
from distriquery.pipeline import Pipeline


def _test_settings(tmp_chunk_size: int = 300) -> Settings:
    return Settings(
        chunk_size=tmp_chunk_size,
        chunk_overlap=30,
        top_k=3,
        embedding_backend="hashing",
        embedding_dim=128,
        llm_backend="fake",
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
    assert payload.retrieval.strategy == "dense"
    assert "latency_ms" in payload.metrics


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