"""Properties of the shared shipped scenario and the evaluation seam."""

from __future__ import annotations

import numpy as np
import pytest

from ragscore.evaluate import PairedScores, score_pair
from ragscore.scenario import (
    SHIPPED_GOLDEN_SIZE,
    build_scenario,
    shipped_candidate,
)


def test_paired_scores_reject_misaligned_vectors():
    with pytest.raises(ValueError, match="must align"):
        PairedScores(
            baseline=np.zeros(3),
            candidate=np.zeros(4),
            metric="ndcg",
            k=10,
        )


def test_the_scenario_builds_the_shipped_golden_size():
    scenario = build_scenario()
    assert scenario.golden.n_queries == SHIPPED_GOLDEN_SIZE


def test_the_scenario_is_deterministic():
    a = build_scenario()
    b = build_scenario()
    assert a.golden.queries == b.golden.queries


def test_the_shipped_candidate_carries_a_fault():
    assert shipped_candidate().blend > 0.0


def test_score_pair_returns_aligned_vectors():
    scenario = build_scenario(golden_size=30)
    scores = score_pair(
        corpus=scenario.corpus,
        golden=scenario.golden,
        baseline_config=scenario.baseline_config,
        candidate_config=scenario.candidate_config,
        metric="ndcg",
        k=10,
        baseline_seed=5,
        candidate_seed=6,
    )
    assert scores.baseline.shape == scores.candidate.shape == (30,)


def test_the_shipped_candidate_is_worse_on_average_in_the_population():
    scenario = build_scenario(golden_size=1000)
    scores = score_pair(
        corpus=scenario.corpus,
        golden=scenario.golden,
        baseline_config=scenario.baseline_config,
        candidate_config=scenario.candidate_config,
        metric="ndcg",
        k=10,
        baseline_seed=5,
        candidate_seed=6,
    )
    assert scores.candidate.mean() < scores.baseline.mean()
