"""Multi-hop retrieval (Theory Primer, Part 2.6 / 3.3).

Be honest about what this is and isn't: a real multi-hop agent would use
an LLM to READ hop 1's evidence and REASON about what's still missing,
generating a genuinely new hop-2 query. That needs a real, capable LLM —
not FakeLLMClient. What's implemented here is the mechanism (retrieve,
use results to inform another retrieval, merge across hops) without the
reasoning step — a crude but real approximation, not a fake one.
"""

from typing import List

from distriquery.retriever import HybridRetriever
from distriquery.vectorstore import SearchResult


class MultiHopRetriever:
    def __init__(self, retriever: HybridRetriever, max_hops: int = 2):
        self._retriever = retriever
        self._max_hops = max_hops

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        seen_ids = set()
        merged: List[SearchResult] = []
        hop_query = query

        for _ in range(self._max_hops):
            hop_results = self._retriever.retrieve(hop_query, top_k=top_k)

            for result in hop_results:
                if result.chunk.chunk_id not in seen_ids:
                    seen_ids.add(result.chunk.chunk_id)
                    merged.append(result)

            if not hop_results:
                break

            top_snippet = hop_results[0].chunk.text[:200]
            hop_query = f"{query} {top_snippet}"

        return merged[: top_k * self._max_hops]