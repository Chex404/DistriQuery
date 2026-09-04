"""Golden dataset (Theory Primer, Part 6.4)."""

from dataclasses import dataclass
from typing import Set

SAMPLE_TXT_SOURCE = "sample_data/sample.txt"


@dataclass
class GoldenExample:
    query: str
    relevant_positions: Set[int]
    notes: str = ""


SAMPLE_TXT_GOLDEN_SET = [
    GoldenExample(
        query="What does the query planner decide between?",
        relevant_positions={2},
        notes="Position 2 explicitly says the query planner decides on a retrieval strategy.",
    ),
    GoldenExample(
        query="What happens when a tenant uploads a document?",
        relevant_positions={1},
        notes="Position 1 describes validation, storage, and publishing to Kafka.",
    ),
    GoldenExample(
        query="What does the reranker do after retrieval?",
        relevant_positions={3},
        notes="Position 3 describes the cross-encoder reranking step.",
    ),
    GoldenExample(
        query="What kind of platform is DistriQuery?",
        relevant_positions={0},
        notes="Position 0 is the project overview describing it as distributed and multi-tenant.",
    ),
]