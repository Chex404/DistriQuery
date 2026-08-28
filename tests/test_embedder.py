import numpy as np

from distriquery.embedder import HashingEmbedder, get_embedder


def test_hashing_embedder_returns_correct_shape():
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed(["hello world", "distributed systems"])

    assert vectors.shape == (2, 64)


def test_hashing_embedder_vectors_are_l2_normalized():
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed(["some reasonably long piece of text here"])

    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-5


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dim=64)
    v1 = embedder.embed(["distriquery is a rag platform"])
    v2 = embedder.embed(["distriquery is a rag platform"])

    assert np.allclose(v1, v2)


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = HashingEmbedder(dim=256)
    query = embedder.embed(["kafka partitions and consumer groups"])[0]
    related = embedder.embed(["kafka topics are split into partitions"])[0]
    unrelated = embedder.embed(["a recipe for chocolate chip cookies"])[0]

    assert np.dot(query, related) > np.dot(query, unrelated)


def test_empty_text_does_not_crash():
    embedder = HashingEmbedder(dim=32)
    vectors = embedder.embed([""])

    assert vectors.shape == (1, 32)


def test_get_embedder_factory_hashing():
    embedder = get_embedder("hashing", dim=16)
    assert embedder.dim == 16


def test_get_embedder_factory_unknown_backend_raises():
    import pytest

    with pytest.raises(ValueError):
        get_embedder("not-a-real-backend")