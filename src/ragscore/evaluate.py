"""Turn two retriever configs and a golden set into the paired per-query vectors the gate needs.

This module is the seam between the retrieval half of the project and the statistics half. It exists
so that the CLI, the experiments and the tests all build their inputs the same way, because the one
thing that must never vary between a published number and a test is how the number was produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragscore.corpus import Corpus, GoldenSet
from ragscore.metrics import per_query_metric
from ragscore.retrieval import Retriever, RetrieverConfig


@dataclass(frozen=True)
class PairedScores:
    """Per-query metric vectors for two retrievers over the same golden set."""

    baseline: np.ndarray
    candidate: np.ndarray
    metric: str
    k: int

    def __post_init__(self) -> None:
        if self.baseline.shape != self.candidate.shape:
            raise ValueError("baseline and candidate score vectors must align")


def score_pair(
    corpus: Corpus,
    golden: GoldenSet,
    baseline_config: RetrieverConfig,
    candidate_config: RetrieverConfig,
    metric: str,
    k: int,
    baseline_seed: int,
    candidate_seed: int,
) -> PairedScores:
    """Rank the golden set with both configs and reduce each to a per-query metric vector.

    The two seeds drive the jitter independently, so passing the same config twice with two seeds
    models two runs of one system (the null), while passing a degraded candidate config models a
    real change. The caller chooses which, and that choice is the only difference between measuring
    a false alarm and measuring a detection.
    """
    baseline_ranker = Retriever(corpus, baseline_config)
    candidate_ranker = Retriever(corpus, candidate_config)
    baseline_rankings = baseline_ranker.rank(golden.queries, seed=baseline_seed)
    candidate_rankings = candidate_ranker.rank(golden.queries, seed=candidate_seed)
    baseline_values = per_query_metric(baseline_rankings, golden.relevants, k, metric)
    candidate_values = per_query_metric(candidate_rankings, golden.relevants, k, metric)
    return PairedScores(
        baseline=baseline_values,
        candidate=candidate_values,
        metric=metric,
        k=k,
    )
