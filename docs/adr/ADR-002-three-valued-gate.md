# ADR 002: A three-valued verdict, because "inconclusive" is the point

## Status

Accepted.

## Context

A regression gate in CI is almost always a boolean: the mean metric dropped by more than a threshold,
or it did not. That boolean is computable on any sample of any size, which is precisely the problem
this repository exists to name. A boolean gate on fifty queries returns a confident verdict that the
fifty queries cannot support, and it has no way to say so.

## Decision

Return one of three verdicts, computed from the bootstrap interval on the difference
D = mean(candidate) minus mean(baseline):

- **regression**: the interval lies entirely below zero, so a drop is real at the chosen confidence.
- **pass**: the interval excludes a drop as large as delta, so any regression is below the threshold
  the team cares about and the sample was large enough to establish that.
- **inconclusive**: the interval still admits a delta-sized drop, so this golden set cannot tell.

The CLI maps these to distinct exit codes, with 2 reserved for inconclusive, so a CI job can
distinguish "the tool answered no" from "the tool could not answer".

## Alternatives considered

- **Boolean with a fixed threshold.** The status quo. Rejected because it cannot express the one
  thing the project measures, and because experiment 03 shows it flips between pass and regression
  near the threshold depending only on which queries were sampled.
- **Boolean plus a separate warning.** A pass or fail with an advisory note. Rejected because a note
  is not an exit code, and CI acts on exit codes.

## Consequences

Callers have to decide what to do with inconclusive, which is more work than a boolean but is the
honest amount of work. The recommended response is to collect more queries, and the `power` command
prints how many. A team that would rather block on inconclusive can treat exit 2 as failure; a team
that would rather ship can treat it as success. The gate refuses to make that risk decision silently,
which is the whole intent.
