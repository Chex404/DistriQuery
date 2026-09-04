from distriquery.config import Settings
from distriquery.evaluation.golden_dataset import SAMPLE_TXT_GOLDEN_SET, SAMPLE_TXT_SOURCE
from distriquery.evaluation.runner import evaluate_retrieval
from distriquery.evaluation.strategies import (
    agentic_strategy,
    hybrid_strategy,
    naive_strategy,
    reranked_strategy,
)
from distriquery.pipeline import Pipeline


def _test_settings() -> Settings:
    return Settings(
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


def _build_ingested_pipeline() -> Pipeline:
    pipeline = Pipeline(settings=_test_settings())
    pipeline.ingest_document(SAMPLE_TXT_SOURCE)
    return pipeline


def test_naive_strategy_runs_against_real_golden_set():
    pipeline = _build_ingested_pipeline()
    retrieval_fn = naive_strategy(pipeline)

    summary = evaluate_retrieval(retrieval_fn, SAMPLE_TXT_GOLDEN_SET, k=4, strategy_name="naive")

    assert summary.strategy_name == "naive"
    assert len(summary.results) == len(SAMPLE_TXT_GOLDEN_SET)
    assert 0.0 <= summary.mrr <= 1.0


def test_hybrid_strategy_runs_against_real_golden_set():
    pipeline = _build_ingested_pipeline()
    retrieval_fn = hybrid_strategy(pipeline)

    summary = evaluate_retrieval(retrieval_fn, SAMPLE_TXT_GOLDEN_SET, k=4, strategy_name="hybrid")

    assert 0.0 <= summary.mrr <= 1.0


def test_reranked_strategy_runs_against_real_golden_set():
    pipeline = _build_ingested_pipeline()
    retrieval_fn = reranked_strategy(pipeline)

    summary = evaluate_retrieval(retrieval_fn, SAMPLE_TXT_GOLDEN_SET, k=4, strategy_name="reranked")

    assert 0.0 <= summary.mrr <= 1.0


def test_agentic_strategy_runs_against_real_golden_set():
    pipeline = _build_ingested_pipeline()
    retrieval_fn = agentic_strategy(pipeline)

    summary = evaluate_retrieval(retrieval_fn, SAMPLE_TXT_GOLDEN_SET, k=4, strategy_name="agentic")

    assert 0.0 <= summary.mrr <= 1.0


def test_all_four_strategies_are_independently_comparable():
    """Not asserting one strategy beats another (that would be asserting a
    specific empirical result, which is what Phase 6's real experiments
    are for) — just proving all four run against the identical corpus and
    golden set, and produce genuinely independent, comparable summaries.
    """
    pipeline = _build_ingested_pipeline()

    summaries = {
        name: evaluate_retrieval(builder(pipeline), SAMPLE_TXT_GOLDEN_SET, k=4, strategy_name=name)
        for name, builder in [
            ("naive", naive_strategy),
            ("hybrid", hybrid_strategy),
            ("reranked", reranked_strategy),
            ("agentic", agentic_strategy),
        ]
    }

    assert len(summaries) == 4
    for name, summary in summaries.items():
        assert summary.strategy_name == name
        assert len(summary.results) == len(SAMPLE_TXT_GOLDEN_SET)