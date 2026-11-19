"""Per-query retrieval metrics with binary relevance.

These are deliberately computed one query at a time and returned per query, never pre-averaged,
because the whole point of the project is that the spread across queries is the thing a mean throws
away. Everything downstream (the bootstrap, the power arithmetic, the gate) consumes the per-query
vector, so the mean is only ever taken at the very end where its uncertainty can travel with it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def recall_at_k(ranked_ids: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k.

    Returns 0.0 when there are no relevant documents, which is the conservative choice: a query with
    nothing to find cannot contribute evidence that retrieval improved.
    """
    _check_k(k)
    if not relevant:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for doc_id in topk if doc_id in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked_ids: Sequence[int], relevant: set[int], k: int) -> float:
    """Discounted cumulative gain with binary gains and the standard log2 discount."""
    _check_k(k)
    gain = 0.0
    for position, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            gain += 1.0 / np.log2(position + 1)
    return gain


def ndcg_at_k(ranked_ids: Sequence[int], relevant: set[int], k: int) -> float:
    """DCG normalised by the best achievable DCG for this relevant set at this cut-off.

    Returns 0.0 for an empty relevant set so that a query with nothing to find scores the same under
    every retriever and therefore contributes zero to any difference.
    """
    _check_k(k)
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(position + 1) for position in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_ids, relevant, k) / idcg


def reciprocal_rank(ranked_ids: Sequence[int], relevant: set[int]) -> float:
    """Reciprocal of the rank of the first relevant document, or 0.0 if none is retrieved."""
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def per_query_metric(
    rankings: Sequence[Sequence[int]],
    relevants: Sequence[set[int]],
    k: int,
    metric: str,
) -> np.ndarray:
    """Vector of one metric value per query, in query order.

    rankings[i] is the ranked document id list the retriever returned for query i, and relevants[i]
    is that query's relevant set. The two sequences must align; a length mismatch is a caller error
    and is refused here rather than silently truncated.
    """
    if len(rankings) != len(relevants):
        raise ValueError(
            f"rankings and relevants disagree in length: {len(rankings)} vs {len(relevants)}"
        )
    if metric == "recall":
        values = [recall_at_k(r, rel, k) for r, rel in zip(rankings, relevants, strict=True)]
    elif metric == "ndcg":
        values = [ndcg_at_k(r, rel, k) for r, rel in zip(rankings, relevants, strict=True)]
    elif metric == "mrr":
        values = [reciprocal_rank(r, rel) for r, rel in zip(rankings, relevants, strict=True)]
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return np.asarray(values, dtype=float)
