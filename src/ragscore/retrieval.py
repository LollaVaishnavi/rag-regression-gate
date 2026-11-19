"""A deterministic TF-IDF retriever, with modelled run-to-run jitter and a tunable fault.

Three design decisions carry most of the project's weight and all three live here.

First, retrieval is real TF-IDF over the corpus rather than simulated scores, because a reviewer
should be able to see that the rankings come from cosine similarity they could recompute, not from a
distribution chosen to make the point. scikit-learn does the vectorising; this module adds only the
jitter, the fault, and the ranking.

Second, the jitter. Re-running "the same" retrieval system does not return identical rankings: float
non-determinism, shard ordering, and near-tie breaks all move borderline documents. That movement is
the entire reason a golden set cannot tell a real regression from noise, so it has to be present in
the null. It is modelled as mean-zero Gaussian noise added to the similarity scores before ranking,
driven by the per-run seed, so two runs of one config disagree exactly as two runs of a real system
would. The magnitude is a declared input, and ADR 004 records why that boundary is honest.

Third, the fault. A real regression is a systematic loss of ranking quality, not more noise, so it
is modelled as a corrupted shard: a fixed fraction `blend` of documents, chosen once by `fault_seed`
and identical for every query, are demoted below the rest. When a relevant document lands in that
fraction it drops out of the top k, which lowers ndcg systematically rather than noisily. `blend` is
continuous, so the experiments can dial a known effect size and ask a gate to find it. This is a
faithful model of a real failure mode: one bad ingest shard, a truncated field, a tokeniser bug that
touches part of the index and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ragscore.corpus import Corpus, GoldenSet


@dataclass(frozen=True)
class RetrieverConfig:
    """A retriever variant.

    use_idf:        inverse-document-frequency weighting. Off is a coarse, large degradation.
    min_df:         minimum document frequency for a token to enter the vocabulary.
    jitter_scale:   standard deviation of the mean-zero run jitter, as a fraction of the per-query
                    score spread. Present in every config, this is what makes two runs disagree.
    blend:          fraction of documents demoted by the corrupted-shard fault. Zero for a clean
                    retriever; small positive values are small systematic regressions.
    fault_seed:     seed choosing which documents the shard fault demotes. Fixed across runs so the
                    fault is a bias, not noise.
    """

    name: str = "baseline"
    use_idf: bool = True
    min_df: int = 1
    jitter_scale: float = 0.08
    blend: float = 0.0
    fault_seed: int = 2024


BASELINE = RetrieverConfig(name="baseline")


class Retriever:
    """Fits one TF-IDF space on a corpus and ranks documents for queries within it."""

    def __init__(self, corpus: Corpus, config: RetrieverConfig) -> None:
        self._config = config
        self._vectorizer = TfidfVectorizer(
            use_idf=config.use_idf,
            min_df=config.min_df,
            token_pattern=r"[^\s]+",
        )
        self._doc_matrix = self._vectorizer.fit_transform(corpus.documents)
        self._n_docs = corpus.n_docs
        # The corrupted shard is chosen once and applied to every query, because a bad shard does
        # not pick a fresh set of documents per request. A large penalty rather than removal keeps
        # the demoted documents in the ranking (so recall at a very large k is unchanged), which is
        # the realistic behaviour: the documents are still indexed, just scored far too low.
        if self._config.blend > 0.0:
            fault_rng = np.random.default_rng(self._config.fault_seed)
            self._demoted = fault_rng.random(self._n_docs) < self._config.blend
        else:
            self._demoted = None

    @property
    def config(self) -> RetrieverConfig:
        return self._config

    def rank(self, queries: list[str], seed: int) -> list[list[int]]:
        """Return a ranked list of document ids for every query.

        The ranking is cosine similarity, demoted by a fixed corrupted shard (if `blend` is
        positive) and then perturbed by mean-zero run jitter (if `jitter_scale` is positive). With
        blend and jitter both zero the ranking is a pure function of the corpus and the queries.
        """
        query_matrix = self._vectorizer.transform(queries)
        scores = cosine_similarity(query_matrix, self._doc_matrix)
        run_rng = np.random.default_rng(seed)
        rankings: list[list[int]] = []
        for row in scores:
            spread = row.std()
            if spread == 0.0:
                spread = 1.0
            corrupted = row.astype(float, copy=True)
            if self._demoted is not None:
                corrupted[self._demoted] -= 1.0e6
            if self._config.jitter_scale > 0.0:
                corrupted = corrupted + run_rng.normal(
                    0.0, self._config.jitter_scale * spread, size=row.shape
                )
            order = np.argsort(-corrupted, kind="stable")
            rankings.append(order.tolist())
        return rankings


def rank_golden_set(
    corpus: Corpus,
    golden: GoldenSet,
    config: RetrieverConfig,
    seed: int,
) -> list[list[int]]:
    """Convenience wrapper: build a retriever for `config` and rank every query in `golden`."""
    retriever = Retriever(corpus, config)
    return retriever.rank(golden.queries, seed=seed)
