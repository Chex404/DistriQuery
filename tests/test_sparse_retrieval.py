from distriquery.chunker import Chunk
from distriquery.sparse_retrieval import BM25Index


def _make_chunk(chunk_id, text, position=0):
    return Chunk(chunk_id=chunk_id, text=text, source="doc1", position=position)


def test_exact_term_match_scores_highest():
    chunks = [
        _make_chunk("a", "Kafka partitions events across a distributed cluster of brokers."),
        _make_chunk("b", "A sourdough starter needs to be fed with flour and water daily."),
    ]
    index = BM25Index(chunks)

    results = index.search("kafka partitions", top_k=2)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "a"
    assert results[0].score > 0


def test_ubiquitous_term_contributes_far_less_than_a_distinguishing_term():
    chunks = [
        _make_chunk("a", "the quick brown fox jumps"),
        _make_chunk("b", "the lazy dog sleeps"),
    ]
    index = BM25Index(chunks)

    ubiquitous_results = index.search("the", top_k=2)
    distinguishing_results = index.search("fox", top_k=2)

    assert ubiquitous_results[0].score < distinguishing_results[0].score


def test_higher_term_frequency_scores_higher():
    chunks = [
        _make_chunk("a", "kafka kafka kafka is a distributed system"),
        _make_chunk("b", "kafka is mentioned once here"),
    ]
    index = BM25Index(chunks)

    results = index.search("kafka", top_k=2)

    assert results[0].chunk.chunk_id == "a"


def test_empty_index_returns_empty():
    index = BM25Index([])
    assert index.search("anything", top_k=4) == []


def test_query_with_no_matching_terms_returns_empty():
    chunks = [_make_chunk("a", "completely unrelated content here")]
    index = BM25Index(chunks)

    assert index.search("zzz nonexistent qqq", top_k=4) == []


def test_respects_top_k():
    chunks = [_make_chunk(str(i), f"apple banana cherry number {i}") for i in range(10)]
    index = BM25Index(chunks)

    results = index.search("apple banana cherry", top_k=3)

    assert len(results) == 3