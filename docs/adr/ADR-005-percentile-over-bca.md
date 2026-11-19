# ADR 005: Percentile interval over BCa, because BCa was not calibrated here

## Status

Accepted. Supersedes the initial choice of BCa.

## Context

The paired bootstrap (ADR 001) still leaves a choice of how to turn the resampled means into an
interval. The literature's default for a skewed, bounded statistic is BCa, bias-corrected and
accelerated, and that is what the gate shipped with first. It was a reasonable default and it was
wrong for this data.

## The measurement

Experiment 01 runs the gate under the null: the candidate is the baseline config run a second time,
so the true difference is zero and only run jitter separates the two score vectors. A 95 percent
interval should call a regression on about 5 percent of null golden sets. Over 400 null golden sets
of 50 queries:

| interval method | null false-positive rate |
| --- | --- |
| nominal | 5.0% |
| BCa | 14.0% |
| percentile | 5.8% |

BCa fired nearly three times as often as it promised. The naive fixed threshold, for comparison,
fired on 0% of the same null, which is its one virtue: it is not trigger-happy on pure jitter, it is
blind near the threshold instead (experiment 03).

## Root cause

The paired-difference vector is zero-inflated. On most queries the two runs rank the relevant
documents identically, so the difference is exactly zero, and only a handful of queries move. BCa's
acceleration term is estimated from the jackknife of that vector, and the jackknife of a vector that
is constant in all but a few positions is badly behaved, so the acceleration swings and drags the
interval endpoints with it. The percentile interval estimates no acceleration and so has nothing to
destabilise.

## Decision

Default to the percentile interval. Keep BCa reachable through the policy and the bootstrap API, so
the calibration experiment can keep measuring both, and so a future user on differently shaped data
can re-check and switch back if their own null says BCa is calibrated for them.

## Consequences

The gate is calibrated under its own null, which is the minimum a reviewer should demand before
trusting any verdict it gives. The cost is that the percentile interval is slightly less efficient
than a well-behaved BCa would be, so it needs marginally more queries for the same power. On this
data that cost is invisible next to a three-fold false-positive inflation. A regression test pins the
finding so a later change that reaches for BCa again fails before it ships.
