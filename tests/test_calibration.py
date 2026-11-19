"""The calibration bug, pinned.

Story: the gate first shipped with a BCa interval, the textbook choice for a skewed statistic. The
calibration experiment then showed BCa calling a regression on about one null golden set in seven
at a
nominal one in twenty. The paired-difference vector is zero-inflated, most queries score identically
under two runs, and BCa's acceleration term is estimated from a jackknife of a near-constant vector,
which makes it unstable. The fix was to default to the percentile interval. This test pins both
halves
of that finding at a smaller scale so a future change that reaches for BCa again fails here first.
"""

from __future__ import annotations

import numpy as np

from ragscore.bootstrap import paired_bootstrap_diff

TRIALS = 200
N = 50
RESAMPLES = 2000


def _null_difference_vector(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A zero-inflated paired sample: mostly identical, a few queries move by a small amount.

    This reproduces the shape of the real null, where two runs of one retriever agree on most
    queries
    and disagree on a handful, without needing to fit a retriever inside a unit test.
    """
    baseline = rng.uniform(0.5, 1.0, size=N)
    candidate = baseline.copy()
    movers = rng.choice(N, size=5, replace=False)
    candidate[movers] += rng.normal(0.0, 0.03, size=5)
    return baseline, candidate


def _false_positive_rate(method: str) -> float:
    rng = np.random.default_rng(0)
    fired = 0
    for _ in range(TRIALS):
        base, cand = _null_difference_vector(rng)
        interval = paired_bootstrap_diff(base, cand, RESAMPLES, 0.05, seed=7, method=method)
        if interval.high < 0.0:
            fired += 1
    return fired / TRIALS


def test_bca_false_positive_rate_exceeds_percentile_on_the_null():
    bca = _false_positive_rate("bca")
    percentile = _false_positive_rate("percentile")
    assert bca > percentile


def test_percentile_stays_near_the_nominal_level():
    percentile = _false_positive_rate("percentile")
    assert percentile <= 0.10
