"""The gate: pass, regression, or inconclusive.

A boolean gate answers "did the mean drop by more than delta". That question is answerable on any
sample, which is the problem: it returns a confident verdict even when the sample is too small to
support one. This gate answers a better question built from the bootstrap interval on the paired
difference D = mean(candidate) - mean(baseline):

  regression    the interval lies entirely below zero, so a drop is real at this confidence.
  pass          the interval excludes a drop as large as delta, so any regression is below the
                threshold the team cares about, and the sample was large enough to say so.
  inconclusive  the interval still admits a delta-sized drop, so the honest answer is that this
                golden set cannot tell. This is the verdict a boolean gate structurally cannot give.

The naive gate is kept here too, not as a straw man but because it is what almost every team ships,
and the experiments measure how often it is wrong so the README can put a number on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ragscore.bootstrap import Interval, paired_bootstrap_diff
from ragscore.config import GatePolicy


class Verdict(StrEnum):
    """The three verdicts, plus the naive gate's two, with stable string values for reports."""

    PASS = "pass"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"


# Exit codes. The split that matters is between "answered no" and "could not answer": a regression
# and a pass are answers, inconclusive is the tool reporting that the evidence is insufficient.
EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_INCONCLUSIVE = 2
EXIT_ERROR = 3

VERDICT_EXIT = {
    Verdict.PASS: EXIT_PASS,
    Verdict.REGRESSION: EXIT_REGRESSION,
    Verdict.INCONCLUSIVE: EXIT_INCONCLUSIVE,
}


@dataclass(frozen=True)
class GateVerdict:
    """A verdict with the evidence that produced it, so a report can explain the decision."""

    verdict: Verdict
    naive_verdict: Verdict
    interval: Interval
    delta: float

    @property
    def exit_code(self) -> int:
        return VERDICT_EXIT[self.verdict]

    @property
    def disagrees_with_naive(self) -> bool:
        return self.verdict is not self.naive_verdict


def naive_gate(baseline: np.ndarray, candidate: np.ndarray, delta: float) -> Verdict:
    """What almost everyone ships: fail if the mean dropped by more than delta, else pass."""
    diff = float(np.mean(candidate) - np.mean(baseline))
    return Verdict.REGRESSION if diff < -delta else Verdict.PASS


def _classify(interval: Interval, delta: float) -> Verdict:
    if interval.high < 0.0:
        return Verdict.REGRESSION
    if interval.low > -delta:
        return Verdict.PASS
    return Verdict.INCONCLUSIVE


def gate(
    baseline: np.ndarray,
    candidate: np.ndarray,
    policy: GatePolicy,
    seed: int,
) -> GateVerdict:
    """Compute the three-valued verdict and the naive verdict on the same paired metric vectors."""
    interval = paired_bootstrap_diff(
        baseline=baseline,
        candidate=candidate,
        n_resamples=policy.n_resamples,
        alpha=policy.alpha,
        seed=seed,
        method=policy.bootstrap_method,
    )
    verdict = _classify(interval, policy.delta)
    naive = naive_gate(baseline, candidate, policy.delta)
    return GateVerdict(verdict=verdict, naive_verdict=naive, interval=interval, delta=policy.delta)
