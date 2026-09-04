import pytest

from distriquery.evaluation.metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k_basic():
    assert precision_at_k([1, 2, 3, 4], {2, 4}, k=4) == 0.5


def test_precision_at_k_with_smaller_k():
    assert precision_at_k([1, 2, 3, 4], {2, 4}, k=2) == 0.5  # top-2: [1,2] -> 1 hit / 2


def test_precision_at_k_no_hits():
    assert precision_at_k([1, 2], {99}, k=2) == 0.0


def test_precision_at_k_zero_k_returns_zero():
    assert precision_at_k([1, 2], {1}, k=0) == 0.0


def test_recall_at_k_basic():
    assert recall_at_k([1, 2, 3, 4], {2, 4}, k=4) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k([1, 2], {2, 4}, k=2) == 0.5  # found 1 of 2 relevant


def test_recall_at_k_no_relevant_items_returns_zero():
    assert recall_at_k([1, 2], set(), k=2) == 0.0


def test_reciprocal_rank_first_hit_at_rank_two():
    assert reciprocal_rank([1, 2, 3], {2}) == 0.5


def test_reciprocal_rank_no_hit_returns_zero():
    assert reciprocal_rank([1, 2, 3], {99}) == 0.0


def test_reciprocal_rank_hit_at_rank_one():
    assert reciprocal_rank([5, 1, 2], {5}) == 1.0


def test_mean_reciprocal_rank_averages_across_queries():
    retrieved_lists = [[1, 2, 3], [5, 1, 2]]
    relevant_sets = [{2}, {5}]  # RR = 0.5 and 1.0 -> mean = 0.75

    assert mean_reciprocal_rank(retrieved_lists, relevant_sets) == pytest.approx(0.75)


def test_dcg_and_ndcg_worked_example():
    relevance_scores = {1: 2, 2: 1, 3: 0}
    retrieved = [3, 1, 2]

    assert dcg_at_k(retrieved, relevance_scores, 3) == pytest.approx(2.3927892607143724)
    assert ndcg_at_k(retrieved, relevance_scores, 3) == pytest.approx(0.6590018048024133)


def test_ndcg_perfect_ranking_equals_one():
    relevance_scores = {1: 2, 2: 1, 3: 0}
    perfect_order = [1, 2, 3]  # already sorted by relevance, best first

    assert ndcg_at_k(perfect_order, relevance_scores, 3) == pytest.approx(1.0)


def test_ndcg_with_no_relevant_items_returns_zero():
    assert ndcg_at_k([1, 2, 3], {}, 3) == 0.0