"""Properties of the paired bootstrap interval."""

from __future__ import annotations

import numpy as np
import pytest

from ragscore.bootstrap import paired_bootstrap_diff


def test_the_interval_brackets_the_observed_difference():
    rng = np.random.default_rng(0)
    base = rng.normal(0.8, 0.1, size=60)
    cand = base - 0.05 + rng.normal(0, 0.02, size=60)
    iv = paired_bootstrap_diff(base, cand, n_resamples=3000, alpha=0.05, seed=1)
    assert iv.low <= iv.point <= iv.high


def test_a_clear_drop_gives_an_interval_entirely_below_zero():
    base = np.full(80, 0.9)
    cand = np.full(80, 0.6)
    iv = paired_bootstrap_diff(base, cand, n_resamples=2000, alpha=0.05, seed=1)
    assert iv.high < 0.0


def test_a_clear_gain_gives_an_interval_entirely_above_zero():
    base = np.full(80, 0.6)
    cand = np.full(80, 0.9)
    iv = paired_bootstrap_diff(base, cand, n_resamples=2000, alpha=0.05, seed=1)
    assert iv.low > 0.0


def test_the_point_estimate_is_the_mean_paired_difference():
    rng = np.random.default_rng(2)
    base = rng.normal(0.5, 0.1, size=40)
    cand = rng.normal(0.5, 0.1, size=40)
    iv = paired_bootstrap_diff(base, cand, n_resamples=1500, alpha=0.05, seed=3)
    assert iv.point == pytest.approx(float((cand - base).mean()))


def test_the_interval_is_deterministic_in_its_seed():
    rng = np.random.default_rng(4)
    base = rng.normal(0.5, 0.1, size=40)
    cand = base - 0.03
    a = paired_bootstrap_diff(base, cand, n_resamples=2000, alpha=0.05, seed=5)
    b = paired_bootstrap_diff(base, cand, n_resamples=2000, alpha=0.05, seed=5)
    assert (a.low, a.high) == (b.low, b.high)


def test_a_wider_confidence_level_gives_a_wider_interval():
    rng = np.random.default_rng(6)
    base = rng.normal(0.7, 0.15, size=70)
    cand = base - 0.04 + rng.normal(0, 0.05, size=70)
    narrow = paired_bootstrap_diff(base, cand, n_resamples=3000, alpha=0.20, seed=7)
    wide = paired_bootstrap_diff(base, cand, n_resamples=3000, alpha=0.01, seed=7)
    assert wide.width > narrow.width


def test_both_methods_are_available():
    rng = np.random.default_rng(8)
    base = rng.normal(0.7, 0.1, size=50)
    cand = base - 0.03 + rng.normal(0, 0.03, size=50)
    bca = paired_bootstrap_diff(base, cand, 2000, 0.05, seed=9, method="bca")
    perc = paired_bootstrap_diff(base, cand, 2000, 0.05, seed=9, method="percentile")
    assert bca.method == "bca"
    assert perc.method == "percentile"


def test_an_unknown_method_is_refused():
    base = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="unknown bootstrap method"):
        paired_bootstrap_diff(base, base, 1000, 0.05, seed=0, method="jackknife")


def test_a_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="disagree in length"):
        paired_bootstrap_diff(np.arange(5.0), np.arange(6.0), 1000, 0.05, seed=0)


def test_a_single_observation_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        paired_bootstrap_diff(np.array([1.0]), np.array([2.0]), 1000, 0.05, seed=0)


def test_bca_falls_back_when_every_difference_is_identical():
    # A constant difference vector has no jackknife spread, so BCa must degrade gracefully.
    base = np.zeros(30)
    cand = np.full(30, -0.1)
    iv = paired_bootstrap_diff(base, cand, 2000, 0.05, seed=1, method="bca")
    assert iv.method == "bca-degenerate"
    assert iv.point == pytest.approx(-0.1)
