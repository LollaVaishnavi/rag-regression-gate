"""Properties of the per-query retrieval metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ragscore.metrics import (
    dcg_at_k,
    ndcg_at_k,
    per_query_metric,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_is_one_when_all_relevant_are_in_top_k():
    assert recall_at_k([0, 1, 2, 3], {0, 1}, k=4) == 1.0


def test_recall_is_the_fraction_found_when_some_are_missed():
    assert recall_at_k([0, 9, 9, 9], {0, 1}, k=4) == 0.5


def test_recall_ignores_documents_below_the_cut_off():
    assert recall_at_k([9, 9, 9, 0], {0}, k=3) == 0.0


def test_recall_of_an_empty_relevant_set_is_zero():
    assert recall_at_k([0, 1, 2], set(), k=3) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([0, 1, 2], {0, 1, 2}, k=3) == pytest.approx(1.0)


def test_ndcg_rewards_relevant_documents_ranked_higher():
    better = ndcg_at_k([0, 9, 9], {0}, k=3)
    worse = ndcg_at_k([9, 9, 0], {0}, k=3)
    assert better > worse


def test_ndcg_of_an_empty_relevant_set_is_zero():
    assert ndcg_at_k([0, 1], set(), k=2) == 0.0


def test_ndcg_lies_between_zero_and_one():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ranking = rng.permutation(20).tolist()
        relevant = set(rng.choice(20, size=5, replace=False).tolist())
        value = ndcg_at_k(ranking, relevant, k=10)
        assert 0.0 <= value <= 1.0


def test_dcg_uses_the_log2_discount():
    # One relevant document at rank 2 gives 1/log2(3).
    assert dcg_at_k([9, 0], {0}, k=2) == pytest.approx(1.0 / np.log2(3))


def test_reciprocal_rank_is_one_over_the_first_hit():
    assert reciprocal_rank([9, 9, 0, 1], {0, 1}) == pytest.approx(1.0 / 3.0)


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_retrieved():
    assert reciprocal_rank([9, 8, 7], {0}) == 0.0


def test_per_query_metric_returns_one_value_per_query():
    rankings = [[0, 1], [1, 0], [0, 1]]
    relevants = [{0}, {0}, {1}]
    values = per_query_metric(rankings, relevants, k=2, metric="recall")
    assert values.shape == (3,)


def test_per_query_metric_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="disagree in length"):
        per_query_metric([[0]], [{0}, {1}], k=1, metric="ndcg")


def test_per_query_metric_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        per_query_metric([[0]], [{0}], k=1, metric="made-up")


@pytest.mark.parametrize("metric", ["recall", "ndcg", "mrr"])
def test_every_metric_is_supported_by_the_dispatch(metric):
    values = per_query_metric([[0, 1, 2]], [{1}], k=3, metric=metric)
    assert values.shape == (1,)


@pytest.mark.parametrize("k", [0, -1])
def test_a_non_positive_cut_off_is_refused(k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        recall_at_k([0], {0}, k=k)
