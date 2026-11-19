"""The one shipped scenario, defined once so every number traces to the same setup.

The CLI, the experiments and the tests all build their corpus and golden set from here. If these
constants lived in each experiment, a reviewer could never be sure that the golden-set size the
README quotes is the one the gate actually ran on. Keeping them in a single module is the cheapest
way to make "the same input" literally true across the whole repository.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragscore.corpus import Corpus, GoldenSet, make_corpus, make_golden_set
from ragscore.retrieval import BASELINE, RetrieverConfig

# The corpus a reviewer reproduces. Fine-grained topics (five documents each) make ndcg@10 sensitive
# to a corrupted shard, which is what gives the gate something to measure.
CORPUS_SEED = 1
GOLDEN_SEED = 2
SHIPPED_GOLDEN_SIZE = 50

# The shipped candidate: a corrupted shard demoting three percent of documents. Its true population
# drop is small enough that a fifty-query set cannot resolve it, which is the whole point.
SHIPPED_BLEND = 0.03

# Seeds for the two runs being compared. Different, so the comparison carries realistic run jitter.
BASELINE_RUN_SEED = 5
CANDIDATE_RUN_SEED = 6


@dataclass(frozen=True)
class Scenario:
    corpus: Corpus
    golden: GoldenSet
    baseline_config: RetrieverConfig
    candidate_config: RetrieverConfig


def shipped_candidate(blend: float = SHIPPED_BLEND) -> RetrieverConfig:
    """The candidate retriever: baseline settings plus a corrupted-shard fault of size `blend`."""
    return RetrieverConfig(
        name="candidate",
        use_idf=BASELINE.use_idf,
        min_df=BASELINE.min_df,
        jitter_scale=BASELINE.jitter_scale,
        blend=blend,
    )


def build_scenario(
    golden_size: int = SHIPPED_GOLDEN_SIZE, blend: float = SHIPPED_BLEND
) -> Scenario:
    """Materialise the shipped corpus, golden set and the two retriever configs."""
    corpus = make_corpus(seed=CORPUS_SEED)
    golden = make_golden_set(seed=GOLDEN_SEED, corpus=corpus, n_queries=golden_size)
    return Scenario(
        corpus=corpus,
        golden=golden,
        baseline_config=BASELINE,
        candidate_config=shipped_candidate(blend),
    )
