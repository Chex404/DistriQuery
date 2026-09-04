"""Runs the naive vs. hybrid vs. reranked vs. agentic comparison against
the sample.txt golden dataset, and prints a results table.

Usage:
    python scripts/run_evaluation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distriquery.config import Settings  # noqa: E402
from distriquery.evaluation.golden_dataset import (  # noqa: E402
    SAMPLE_TXT_GOLDEN_SET,
    SAMPLE_TXT_SOURCE,
)
from distriquery.evaluation.runner import evaluate_retrieval  # noqa: E402
from distriquery.evaluation.strategies import STRATEGIES  # noqa: E402
from distriquery.pipeline import Pipeline  # noqa: E402


def main():
    settings = Settings(
        chunk_size=500,
        chunk_overlap=50,
        top_k=4,
        embedding_backend="hashing",
        embedding_dim=128,
        llm_backend="fake",
        vector_store_backend="in-memory",
        reranker_backend="overlap",
        agent_enabled=False,
    )

    pipeline = Pipeline(settings=settings)
    chunk_count = pipeline.ingest_document(SAMPLE_TXT_SOURCE)
    print(f"Ingested {SAMPLE_TXT_SOURCE} -> {chunk_count} chunks")
    print(f"Golden set: {len(SAMPLE_TXT_GOLDEN_SET)} queries\n")

    print(f"{'Strategy':<12} {'Precision@4':<13} {'Recall@4':<11} {'MRR':<8}")
    print("-" * 46)

    for name, builder in STRATEGIES.items():
        retrieval_fn = builder(pipeline)
        summary = evaluate_retrieval(retrieval_fn, SAMPLE_TXT_GOLDEN_SET, k=4, strategy_name=name)
        print(
            f"{summary.strategy_name:<12} "
            f"{summary.mean_precision_at_k:<13.3f} "
            f"{summary.mean_recall_at_k:<11.3f} "
            f"{summary.mrr:<8.3f}"
        )

    print(
        "\nNote: this is the HASHING embedder + OVERLAP reranker (fast, free, "
        "deterministic) — set EMBEDDING_BACKEND=sentence-transformers and "
        "RERANKER_BACKEND=cross-encoder in .env for a real quality comparison, "
        "not just a mechanism check."
    )


if __name__ == "__main__":
    main()