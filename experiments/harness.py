"""Shared Monte Carlo machinery for the experiments.

The experiments repeatedly draw a fresh golden set, score a baseline and a candidate retriever on
it, and look at the resulting verdict. The expensive step is fitting the TF-IDF space, so the
retrievers are fitted once per configuration and reused across every golden-set draw; only the query
transform and the ranking repeat. Keeping this in one place means every experiment draws its samples
the same way, which is the point of having a harness at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ragscore.config import GatePolicy
from ragscore.corpus import Corpus, make_corpus, make_golden_set
from ragscore.metrics import per_query_metric
from ragscore.retrieval import BASELINE, Retriever, RetrieverConfig
from ragscore.scenario import CORPUS_SEED, shipped_candidate

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "experiments"

# A seed offset for the Monte Carlo golden sets, kept clear of the shipped golden seed (2) so the
# sampled sets never coincide with the one the shipped scenario reports on.
MC_SEED_BASE = 1000


@dataclass(frozen=True)
class TrialScores:
    baseline: np.ndarray
    candidate: np.ndarray


def fixed_corpus() -> Corpus:
    """The one corpus every experiment shares."""
    return make_corpus(seed=CORPUS_SEED)


def run_trials(
    corpus: Corpus,
    candidate_config: RetrieverConfig,
    trials: int,
    n_queries: int,
    k: int,
    metric: str,
    baseline_seed: int = 5,
    candidate_seed: int = 6,
) -> list[TrialScores]:
    """Draw `trials` golden sets and score baseline against `candidate_config` on each.

    The two retrievers are fitted once here, not per trial, which is the whole reason the
    experiments
    finish in seconds rather than minutes.
    """
    baseline_retriever = Retriever(corpus, BASELINE)
    candidate_retriever = Retriever(corpus, candidate_config)
    out: list[TrialScores] = []
    for t in range(trials):
        golden = make_golden_set(seed=MC_SEED_BASE + t, corpus=corpus, n_queries=n_queries)
        base_rankings = baseline_retriever.rank(golden.queries, seed=baseline_seed)
        cand_rankings = candidate_retriever.rank(golden.queries, seed=candidate_seed)
        base = per_query_metric(base_rankings, golden.relevants, k, metric)
        cand = per_query_metric(cand_rankings, golden.relevants, k, metric)
        out.append(TrialScores(baseline=base, candidate=cand))
    return out


def write_json(name: str, payload: dict) -> Path:
    """Write an experiment's result to docs/experiments/<name>.json and return the path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def default_policy_for_experiments() -> GatePolicy:
    """The shipped policy, used so experiment numbers match the gate a reviewer runs."""
    return GatePolicy()


def candidate_for_blend(blend: float) -> RetrieverConfig:
    """A candidate retriever carrying a corrupted-shard fault of the given size."""
    if blend == 0.0:
        return BASELINE
    return shipped_candidate(blend)
