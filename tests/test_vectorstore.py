import numpy as np
import pytest

from distriquery.chunker import Chunk
from distriquery.vectorstore import InMemoryVectorStore


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source="doc1", position=0)


def test_search_returns_closest_vector_first():
    store = InMemoryVectorStore()
    chunks = [_make_chunk("a", "about kafka"), _make_chunk("b", "about cats")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.add(chunks, vectors)

    query = np.array([0.9, 0.1], dtype=np.float32)
    results = store.search(query, top_k=2)

    assert results[0].chunk.chunk_id == "a"
    assert results[0].score > results[1].score


def test_search_respects_top_k():
    store = InMemoryVectorStore()
    chunks = [_make_chunk(str(i), f"chunk {i}") for i in range(5)]
    vectors = np.eye(5, dtype=np.float32)
    store.add(chunks, vectors)

    results = store.search(np.array([1, 0, 0, 0, 0], dtype=np.float32), top_k=2)

    assert len(results) == 2


def test_search_on_empty_store_returns_empty_list():
    store = InMemoryVectorStore()
    results = store.search(np.array([1.0, 0.0]), top_k=3)

    assert results == []


def test_add_mismatched_lengths_raises():
    store = InMemoryVectorStore()
    chunks = [_make_chunk("a", "text")]
    vectors = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        store.add(chunks, vectors)


def test_len_reflects_number_of_stored_chunks():
    store = InMemoryVectorStore()
    assert len(store) == 0

    chunks = [_make_chunk("a", "x"), _make_chunk("b", "y")]
    store.add(chunks, np.zeros((2, 3), dtype=np.float32))

    assert len(store) == 2

def test_adding_same_chunk_id_again_overwrites_not_duplicates():
    store = InMemoryVectorStore()
    store.add([_make_chunk("a", "original text")], np.array([[1.0, 0.0]], dtype=np.float32))
    assert len(store) == 1

    store.add([_make_chunk("a", "updated text")], np.array([[0.0, 1.0]], dtype=np.float32))

    assert len(store) == 1
    results = store.search(np.array([0.0, 1.0], dtype=np.float32), top_k=1)
    assert results[0].chunk.text == "updated text"

def test_multiple_add_calls_accumulate():
    store = InMemoryVectorStore()
    store.add([_make_chunk("a", "x")], np.zeros((1, 3), dtype=np.float32))
    store.add([_make_chunk("b", "y")], np.zeros((1, 3), dtype=np.float32))

    assert len(store) == 2