# ADR 001: A paired, query-level bootstrap rather than a t-test

## Status

Accepted.

## Context

The gate needs an interval on the difference in a mean retrieval metric between two retrievers over a
golden set. The obvious parametric choice is a paired t-test on the per-query differences, which is
one line and has a closed form.

Two facts about the data make the t-test the wrong default. First, ndcg@10 and recall@10 per query
are bounded in [0, 1] and, on an easy corpus, pile up near the top of that range, so the per-query
difference is not close to normal. Second, the quantity the gate cares about is a small mean near
zero where the shape of the tail decides the verdict, which is exactly where a normal approximation
is least safe.

## Decision

Estimate the interval with a bootstrap that resamples queries with replacement, pairing the two
retrievers by query so the between-query variance cancels. The unit of resampling is the query
because the question is whether the observed difference would survive a different sample of queries,
and the query is the thing that was sampled.

## Alternatives considered

- **Paired t-test.** Rejected as the default for the non-normality above, though it would agree with
  the bootstrap in the large-sample limit.
- **Two-sample bootstrap.** Rejected because it discards the pairing and so keeps the between-query
  variance the paired design removes, widening every interval for no gain.

## Consequences

The interval is distribution-free and honest about skew, at the cost of being seeded and therefore
needing a convergence check on the resample count (ADR 005 and experiment 00). The bootstrap also
exposes a second decision, which interval to read off the resamples, that a t-test would not have
forced. That decision turned out to matter and is ADR 005.
