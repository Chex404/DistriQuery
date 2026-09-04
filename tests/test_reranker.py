import pytest

from distriquery.chunker import Chunk
from distriquery.reranker import OverlapReranker, get_reranker
from distriquery.vectorstore import SearchResult


def _make_result(chunk_id, text, score=0.5):
    chunk = Chunk(chunk_id=chunk_id, text=text, source="doc1", position=0)
    return SearchResult(chunk=chunk, score=score)


def test_overlap_reranker_promotes_higher_term_overlap():
    results = [
        _make_result("a", "completely unrelated content about cooking"),
        _make_result("b", "Kafka partitions events across distributed brokers"),
    ]
    reranker = OverlapReranker()

    reranked = reranker.rerank("kafka distributed partitions", results, top_k=2)

    assert reranked[0].chunk.chunk_id == "b"


def test_overlap_reranker_respects_top_k():
    results = [_make_result(str(i), f"apple banana cherry {i}") for i in range(10)]
    reranker = OverlapReranker()

    reranked = reranker.rerank("apple banana", results, top_k=3)

    assert len(reranked) == 3


def test_overlap_reranker_on_empty_results_returns_empty():
    reranker = OverlapReranker()
    assert reranker.rerank("anything", [], top_k=4) == []


def test_get_reranker_factory_overlap():
    reranker = get_reranker("overlap")
    assert isinstance(reranker, OverlapReranker)


def test_get_reranker_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_reranker("not-a-real-backend")