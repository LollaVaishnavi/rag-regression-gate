"""Properties of the synthetic corpus and golden set."""

from __future__ import annotations

import pytest

from ragscore.corpus import make_corpus, make_golden_set


def test_corpus_has_the_requested_number_of_documents():
    corpus = make_corpus(seed=0, n_docs=120, n_topics=12)
    assert corpus.n_docs == 120


def test_corpus_spreads_documents_evenly_across_topics():
    corpus = make_corpus(seed=0, n_docs=120, n_topics=12)
    counts = {int(t): 0 for t in set(corpus.doc_topics.tolist())}
    for topic in corpus.doc_topics:
        counts[int(topic)] += 1
    assert set(counts.values()) == {10}


def test_corpus_is_deterministic_in_its_seed():
    a = make_corpus(seed=7)
    b = make_corpus(seed=7)
    assert a.documents == b.documents


def test_different_seeds_give_different_corpora():
    a = make_corpus(seed=1)
    b = make_corpus(seed=2)
    assert a.documents != b.documents


def test_golden_set_has_the_requested_number_of_queries():
    corpus = make_corpus(seed=0)
    golden = make_golden_set(seed=1, corpus=corpus, n_queries=30)
    assert golden.n_queries == 30


def test_every_query_has_at_least_one_relevant_document():
    corpus = make_corpus(seed=0)
    golden = make_golden_set(seed=1, corpus=corpus, n_queries=40)
    assert all(len(rel) > 0 for rel in golden.relevants)


def test_relevant_ids_are_valid_document_indices():
    corpus = make_corpus(seed=0)
    golden = make_golden_set(seed=1, corpus=corpus, n_queries=40)
    for rel in golden.relevants:
        assert all(0 <= doc_id < corpus.n_docs for doc_id in rel)


def test_golden_set_is_deterministic_in_its_seed():
    corpus = make_corpus(seed=0)
    a = make_golden_set(seed=3, corpus=corpus, n_queries=20)
    b = make_golden_set(seed=3, corpus=corpus, n_queries=20)
    assert a.queries == b.queries


def test_overlap_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="overlap"):
        make_corpus(seed=0, overlap=1.5)


def test_a_noise_ratio_of_one_is_refused():
    with pytest.raises(ValueError, match="noise_ratio"):
        make_corpus(seed=0, noise_ratio=1.0)


def test_zero_overlap_is_allowed_and_produces_private_vocabularies():
    corpus = make_corpus(seed=0, overlap=0.0, n_topics=4, n_docs=40)
    assert "shared_" not in " ".join(corpus.documents)
