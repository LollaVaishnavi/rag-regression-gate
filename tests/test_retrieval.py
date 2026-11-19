"""Properties of the TF-IDF retriever, its jitter, and its shard fault."""

from __future__ import annotations

import numpy as np

from ragscore.corpus import make_corpus, make_golden_set
from ragscore.metrics import per_query_metric
from ragscore.retrieval import BASELINE, Retriever, RetrieverConfig, rank_golden_set
from ragscore.scenario import shipped_candidate


def _ndcg(corpus, golden, config, seed):
    rankings = rank_golden_set(corpus, golden, config, seed=seed)
    return per_query_metric(rankings, golden.relevants, k=10, metric="ndcg")


def test_a_clean_retriever_without_jitter_is_a_pure_function():
    corpus = make_corpus(seed=0)
    golden = make_golden_set(seed=1, corpus=corpus, n_queries=30)
    clean = RetrieverConfig(name="clean", jitter_scale=0.0)
    a = rank_golden_set(corpus, golden, clean, seed=10)
    b = rank_golden_set(corpus, golden, clean, seed=99)
    assert a == b


def test_jitter_makes_two_runs_disagree_on_at_least_one_query():
    corpus = make_corpus(seed=0)
    golden = make_golden_set(seed=1, corpus=corpus, n_queries=60)
    a = rank_golden_set(corpus, golden, BASELINE, seed=10)
    b = rank_golden_set(corpus, golden, BASELINE, seed=11)
    assert a != b


def test_every_ranking_is_a_permutation_of_the_corpus():
    corpus = make_corpus(seed=0, n_docs=120, n_topics=24)
    golden = make_golden_set(seed=1, corpus=corpus, n_queries=10)
    rankings = rank_golden_set(corpus, golden, BASELINE, seed=5)
    for ranking in rankings:
        assert sorted(ranking) == list(range(corpus.n_docs))


def test_the_shard_fault_lowers_mean_ndcg_in_the_population():
    corpus = make_corpus(seed=1)
    golden = make_golden_set(seed=2, corpus=corpus, n_queries=800)
    clean = _ndcg(corpus, golden, BASELINE, seed=5).mean()
    faulty = _ndcg(corpus, golden, shipped_candidate(0.08), seed=6).mean()
    assert faulty < clean


def test_a_larger_fault_lowers_ndcg_further():
    corpus = make_corpus(seed=1)
    golden = make_golden_set(seed=2, corpus=corpus, n_queries=800)
    small = _ndcg(corpus, golden, shipped_candidate(0.03), seed=6).mean()
    large = _ndcg(corpus, golden, shipped_candidate(0.15), seed=6).mean()
    assert large < small


def test_the_shard_is_fixed_across_runs_not_resampled_per_query():
    # Two runs of the same faulty config differ only by jitter, so the mean drop is stable.
    corpus = make_corpus(seed=1)
    golden = make_golden_set(seed=2, corpus=corpus, n_queries=400)
    a = _ndcg(corpus, golden, shipped_candidate(0.1), seed=6).mean()
    b = _ndcg(corpus, golden, shipped_candidate(0.1), seed=7).mean()
    assert abs(a - b) < 0.02


def test_a_query_with_no_signal_still_returns_a_full_ranking():
    corpus = make_corpus(seed=0, n_docs=50, n_topics=10)
    retriever = Retriever(corpus, RetrieverConfig(jitter_scale=0.2))
    # A query of pure out-of-vocabulary tokens has an all-zero score vector.
    rankings = retriever.rank(["zzz_unknown yyy_unknown"], seed=3)
    assert sorted(rankings[0]) == list(range(corpus.n_docs))


def test_baseline_config_is_clean():
    assert BASELINE.blend == 0.0
    assert BASELINE.use_idf is True


def test_retriever_exposes_its_config():
    corpus = make_corpus(seed=0, n_docs=40, n_topics=8)
    retriever = Retriever(corpus, BASELINE)
    assert retriever.config is BASELINE
