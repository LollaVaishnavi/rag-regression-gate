# ADR 003: A synthetic TF-IDF corpus, and what that boundary costs

## Status

Accepted.

## Context

A retrieval regression gate could be demonstrated on a real embedding model and a real vector
database. Doing so would make the absolute metric values realistic. It would also make every number
in the repository depend on an API key, a model version, and a network, none of which a reviewer can
reproduce a year from now, and it would move the measurement off the one thing the project is about.

## Decision

Retrieve with TF-IDF cosine over a seeded synthetic corpus whose relevance is defined by topic
membership. The corpus is deterministic, keyless, offline, and ships in the repository through its
generator. A reviewer reproduces every number with one command and no credentials.

## What transfers and what does not

This is the boundary, stated plainly because it is the most senior thing in the repository.

**Transfers unchanged** to a real embedding-plus-vector-database setup: the entire statistical layer.
The paired bootstrap, the calibration result in ADR 005, the three-valued verdict, and the
sample-size arithmetic all operate on per-query metric vectors and know nothing about where the
rankings came from. Swap TF-IDF for a real retriever and the gate behaves identically.

**Does not transfer**: the absolute metric values, the specific per-query difference spread, and
therefore the exact required-sample-size numbers. Those depend on the retriever and the corpus. A
team using this on a real system would re-measure the spread on their own golden set, which the
`power` command does, and the published numbers here would not apply to them.

**Genuinely needs the real thing**: a claim that a particular embedding change helps or harms on a
particular production corpus. That is a claim about absolute quality, and no fixture can stand in for
it. This repository makes the narrower claim, that a difference is or is not resolvable at a given
sample size, and that claim survives the fixture.

## Consequences

The repository trades realism for reproducibility, on purpose. The experiments measure properties of
the gate, not of any real retriever, and the README says so in its own section rather than in a
footnote.
