from distriquery.evaluation.golden_dataset import GoldenExample
from distriquery.evaluation.runner import evaluate_retrieval


def test_evaluate_retrieval_with_perfect_retrieval_fn():
    """A retrieval_fn that always returns exactly the relevant positions
    first should score perfectly across every metric."""
    golden_set = [
        GoldenExample(query="q1", relevant_positions={2}),
        GoldenExample(query="q2", relevant_positions={0}),
    ]

    def perfect_fn(query, top_k):
        example = next(e for e in golden_set if e.query == query)
        return list(example.relevant_positions) + [99, 98, 97][: top_k - 1]

    summary = evaluate_retrieval(perfect_fn, golden_set, k=4, strategy_name="perfect")

    assert summary.mean_precision_at_k == 0.25  # 1 relevant out of 4 retrieved, per query
    assert summary.mean_recall_at_k == 1.0  # found the (only) relevant item every time
    assert summary.mrr == 1.0  # relevant item always ranked first


def test_evaluate_retrieval_with_useless_retrieval_fn():
    golden_set = [GoldenExample(query="q1", relevant_positions={2})]

    def useless_fn(query, top_k):
        return [99, 98, 97, 96][:top_k]

    summary = evaluate_retrieval(useless_fn, golden_set, k=4, strategy_name="useless")

    assert summary.mean_precision_at_k == 0.0
    assert summary.mean_recall_at_k == 0.0
    assert summary.mrr == 0.0


def test_evaluate_retrieval_records_per_query_results():
    golden_set = [
        GoldenExample(query="q1", relevant_positions={2}),
        GoldenExample(query="q2", relevant_positions={0}),
    ]

    def fn(query, top_k):
        return [0, 1, 2, 3][:top_k]

    summary = evaluate_retrieval(fn, golden_set, k=4)

    assert len(summary.results) == 2
    assert summary.results[0].query == "q1"
    assert summary.results[1].query == "q2"