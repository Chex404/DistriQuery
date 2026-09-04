"""Sparse (BM25) retrieval.

Hand-rolled BM25 (Theory Primer, Part 1.2), same reasoning as chunker.py's
hand-rolled splitter: understand the mechanism, don't just import
rank_bm25 and move on.

    score(D,Q) = sum over query terms of:
        IDF(term) * (f(term,D) * (k1+1)) / (f(term,D) + k1*(1-b+b*|D|/avgdl))

Built fresh from a tenant's chunks at query time (see
VectorStore.get_all_chunks()), rather than maintained as a persistent
index. This avoids needing new infrastructure — Qdrant already stores
every chunk's text in its payload — at the cost of rebuilding the index
on every query (O(n) in corpus size). A real production system would
cache this per tenant with invalidation on ingestion; that's a reasonable
future optimization, not something this phase needs yet.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

from distriquery.chunker import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")

K1 = 1.5  # term-frequency saturation
B = 0.75  # document-length normalization


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Result:
    chunk: Chunk
    score: float

class BM25Index:
    def __init__(self, chunks: List[Chunk]):
        self._chunks = chunks
        self._n_docs = len(chunks)

        doc_tokens = [tokenize(c.text) for c in chunks]
        self._doc_lengths = [len(tokens) for tokens in doc_tokens]
        self._avg_doc_length = (
            sum(self._doc_lengths) / self._n_docs if self._n_docs else 0.0
        )
        self._doc_term_counts = [Counter(tokens) for tokens in doc_tokens]

        doc_freq: Counter = Counter()
        for term_counts in self._doc_term_counts:
            for term in term_counts:
                doc_freq[term] += 1
        self._doc_freq = doc_freq

    def _idf(self, term: str) -> float:
        n = self._doc_freq.get(term, 0)
        return math.log((self._n_docs - n + 0.5) / (n + 0.5) + 1)

    def search(self, query: str, top_k: int = 4) -> List[BM25Result]:
        if self._n_docs == 0:
            return []

        query_terms = tokenize(query)
        scores = [0.0] * self._n_docs

        for term in query_terms:
            idf = self._idf(term)
            if idf <= 0:
                continue  # defensive only — the "+1" smoothing above means this
                          # practically never triggers; IDF stays small-but-positive
                          # even for a term in every document, rather than hitting zero
            for i, term_counts in enumerate(self._doc_term_counts):
                f = term_counts.get(term, 0)
                if f == 0:
                    continue
                doc_len = self._doc_lengths[i]
                denom = f + K1 * (1 - B + B * doc_len / (self._avg_doc_length or 1))
                scores[i] += idf * (f * (K1 + 1)) / denom

        ranked_indices = sorted(range(self._n_docs), key=lambda i: -scores[i])
        return [
            BM25Result(chunk=self._chunks[i], score=scores[i])
            for i in ranked_indices[:top_k]
            if scores[i] > 0
        ]