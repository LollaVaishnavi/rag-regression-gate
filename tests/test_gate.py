"""Properties of the three-valued gate."""

from __future__ import annotations

import numpy as np

from ragscore.config import GatePolicy
from ragscore.gate import (
    EXIT_INCONCLUSIVE,
    EXIT_PASS,
    EXIT_REGRESSION,
    Verdict,
    gate,
    naive_gate,
)

POLICY = GatePolicy(delta=0.02, alpha=0.05, n_resamples=2000)


def test_a_large_real_drop_is_called_a_regression():
    base = np.full(80, 0.9)
    cand = np.full(80, 0.6)
    verdict = gate(base, cand, POLICY, seed=1)
    assert verdict.verdict is Verdict.REGRESSION
    assert verdict.exit_code == EXIT_REGRESSION


def test_an_identical_candidate_is_a_pass():
    rng = np.random.default_rng(0)
    base = rng.normal(0.8, 0.1, size=80)
    verdict = gate(base, base.copy(), POLICY, seed=1)
    assert verdict.verdict is Verdict.PASS
    assert verdict.exit_code == EXIT_PASS


def test_a_tiny_ambiguous_drop_is_inconclusive():
    # A drop well under delta with enough spread that the interval still admits a delta drop.
    rng = np.random.default_rng(1)
    base = rng.normal(0.8, 0.2, size=40)
    cand = base - 0.01 + rng.normal(0, 0.2, size=40)
    verdict = gate(base, cand, POLICY, seed=2)
    assert verdict.verdict is Verdict.INCONCLUSIVE
    assert verdict.exit_code == EXIT_INCONCLUSIVE


def test_the_naive_gate_fires_on_any_mean_drop_past_delta():
    base = np.array([0.9, 0.9, 0.9, 0.9])
    cand = np.array([0.9, 0.9, 0.9, 0.5])  # mean drop 0.1 > delta
    assert naive_gate(base, cand, delta=0.02) is Verdict.REGRESSION


def test_the_naive_gate_passes_a_drop_under_delta():
    base = np.array([0.9, 0.9, 0.9, 0.9])
    cand = np.array([0.9, 0.9, 0.9, 0.88])
    assert naive_gate(base, cand, delta=0.02) is Verdict.PASS


def test_the_gate_reports_when_it_disagrees_with_the_naive_threshold():
    rng = np.random.default_rng(2)
    base = rng.normal(0.8, 0.2, size=40)
    cand = base - 0.01 + rng.normal(0, 0.2, size=40)
    verdict = gate(base, cand, POLICY, seed=3)
    # naive passes a 0.01 drop under delta 0.02; the gate finds it inconclusive.
    assert verdict.naive_verdict is Verdict.PASS
    assert verdict.disagrees_with_naive is (verdict.verdict is not Verdict.PASS)


def test_exit_codes_are_distinct_across_verdicts():
    assert len({EXIT_PASS, EXIT_REGRESSION, EXIT_INCONCLUSIVE}) == 3


def test_verdict_values_are_stable_strings():
    assert Verdict.PASS == "pass"
    assert Verdict.REGRESSION == "regression"
    assert Verdict.INCONCLUSIVE == "inconclusive"
