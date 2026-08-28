"""Embedding.

Defines an Embedder interface plus two implementations:

- HashingEmbedder: dependency-free, deterministic, no model download.
  Uses the "hashing trick" (feature hashing): hash each word into one of
  `dim` buckets and accumulate counts, then L2-normalize. This is NOT a
  semantic embedding — two synonyms won't end up close together — but it's
  enough to prove that chunking, storage, cosine-similarity ranking, and
  retrieval all work correctly, without needing network access or a
  multi-hundred-MB model. Used in tests and in this demo.

- SentenceTransformerEmbedder: the real thing. Encodes text into a genuine
  semantic embedding space using a pretrained bi-encoder. This is what you
  should actually use once you're running the project somewhere with
  internet access — swap EMBEDDING_BACKEND to "sentence-transformers" in
  your .env, nothing else changes, because both classes implement the same
  embed() signature.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (N, dim) array of L2-normalized embeddings, one row per input text."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 256):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in _WORD_RE.findall(text.lower()):
                bucket = self._hash_bucket(word)
                vectors[i, bucket] += 1.0
        return _l2_normalize(vectors)

    def _hash_bucket(self, word: str) -> int:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self._dim


class SentenceTransformerEmbedder(Embedder):
    """Real semantic embeddings via sentence-transformers.

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return _l2_normalize(vectors.astype(np.float32))


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for an all-zero (e.g. empty) text
    return vectors / norms


def get_embedder(backend: str, dim: int = 256, model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    if backend == "hashing":
        return HashingEmbedder(dim=dim)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"Unknown embedding backend: {backend}")