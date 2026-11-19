"""A seeded synthetic corpus with ground-truth relevance.

Real relevance judgements are expensive and non-deterministic, and neither property helps a reviewer
who wants to reproduce a number. So relevance here is defined by construction: every document
belongs to exactly one topic, every query belongs to one topic, and the relevant set for a query is
the set of documents sharing its topic. That gives an oracle the experiments can measure a gate
against, which a real golden set cannot provide because its own labels are the thing under doubt.

The generator exposes two difficulty knobs, vocabulary overlap between topics and the amount of
background noise per document, because a corpus where retrieval is perfect measures nothing: there
has to be headroom for a change to help or hurt before a gate on it means anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Corpus:
    """Documents with hidden topic labels, ready to hand to a vectoriser."""

    documents: list[str]
    doc_topics: np.ndarray  # shape (n_docs,), integer topic id per document

    @property
    def n_docs(self) -> int:
        return len(self.documents)


@dataclass(frozen=True)
class GoldenSet:
    """Queries with the relevant document ids implied by shared topic membership."""

    queries: list[str]
    relevants: list[set[int]]

    @property
    def n_queries(self) -> int:
        return len(self.queries)


def _topic_vocabularies(
    rng: np.random.Generator,
    n_topics: int,
    signature_size: int,
    overlap: float,
) -> list[list[str]]:
    """Give each topic a list of signature tokens, sharing a fraction `overlap` with a shared pool.

    A token is written as "t{topic}_{i}" for a private token and "shared_{i}" for a shared one, so a
    reader of a generated document can see at a glance why it was retrieved.
    """
    n_shared = int(round(signature_size * overlap))
    n_private = signature_size - n_shared
    shared_pool = [f"shared_{i}" for i in range(max(n_shared * 2, 1))]
    vocabularies: list[list[str]] = []
    for topic in range(n_topics):
        private = [f"t{topic}_{i}" for i in range(n_private)]
        shared = list(rng.choice(shared_pool, size=n_shared, replace=False)) if n_shared else []
        vocabularies.append(private + shared)
    return vocabularies


def make_corpus(
    seed: int,
    n_docs: int = 600,
    n_topics: int = 120,
    doc_length: int = 24,
    signature_size: int = 12,
    overlap: float = 0.4,
    noise_ratio: float = 0.6,
    noise_vocab: int = 400,
) -> Corpus:
    """Build a corpus of `n_docs` documents spread evenly across `n_topics` topics.

    Each document is a bag of tokens: a sample from its topic's signature vocabulary, plus a share
    of background noise tokens drawn from a large generic pool. `noise_ratio` is the fraction of a
    document's tokens that are noise, which is the main lever on how hard retrieval is.
    """
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap must lie in [0, 1], got {overlap}")
    if not 0.0 <= noise_ratio < 1.0:
        raise ValueError(f"noise_ratio must lie in [0, 1), got {noise_ratio}")
    rng = np.random.default_rng(seed)
    vocabularies = _topic_vocabularies(rng, n_topics, signature_size, overlap)
    noise_pool = [f"bg_{i}" for i in range(noise_vocab)]

    n_noise = int(round(doc_length * noise_ratio))
    n_signal = doc_length - n_noise
    documents: list[str] = []
    doc_topics = np.empty(n_docs, dtype=int)
    for doc_id in range(n_docs):
        topic = doc_id % n_topics
        doc_topics[doc_id] = topic
        signal = rng.choice(vocabularies[topic], size=n_signal, replace=True)
        noise = rng.choice(noise_pool, size=n_noise, replace=True)
        tokens = np.concatenate([signal, noise])
        rng.shuffle(tokens)
        documents.append(" ".join(tokens.tolist()))
    return Corpus(documents=documents, doc_topics=doc_topics)


def make_golden_set(
    seed: int,
    corpus: Corpus,
    n_queries: int = 50,
    query_length: int = 5,
    signature_size: int = 12,
    overlap: float = 0.4,
) -> GoldenSet:
    """Sample `n_queries` queries, each a short bag of one topic's signature tokens.

    The vocabularies are regenerated from the same parameters rather than stored on the corpus, so a
    golden set drawn with a different seed is statistically comparable but not identical, which is
    what the experiments need when they draw many independent sets.
    """
    rng = np.random.default_rng(seed)
    n_topics = int(corpus.doc_topics.max()) + 1
    vocabularies = _topic_vocabularies(rng, n_topics, signature_size, overlap)
    by_topic: list[list[int]] = [[] for _ in range(n_topics)]
    for doc_id, topic in enumerate(corpus.doc_topics):
        by_topic[int(topic)].append(doc_id)

    queries: list[str] = []
    relevants: list[set[int]] = []
    for _ in range(n_queries):
        topic = int(rng.integers(n_topics))
        tokens = rng.choice(vocabularies[topic], size=query_length, replace=True)
        queries.append(" ".join(tokens.tolist()))
        relevants.append(set(by_topic[topic]))
    return GoldenSet(queries=queries, relevants=relevants)
