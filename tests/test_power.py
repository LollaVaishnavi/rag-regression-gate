"""Properties of the sample-size arithmetic."""

from __future__ import annotations

import pytest

from ragscore.power import minimum_detectable_effect, required_sample_size


def test_a_smaller_effect_needs_more_queries():
    big = required_sample_size(effect=0.05, sd_diff=0.1, alpha=0.05, power=0.8)
    small = required_sample_size(effect=0.01, sd_diff=0.1, alpha=0.05, power=0.8)
    assert small > big


def test_more_variance_needs_more_queries():
    quiet = required_sample_size(effect=0.02, sd_diff=0.05, alpha=0.05, power=0.8)
    noisy = required_sample_size(effect=0.02, sd_diff=0.15, alpha=0.05, power=0.8)
    assert noisy > quiet


def test_required_sample_size_rounds_up():
    n = required_sample_size(effect=0.02, sd_diff=0.066, alpha=0.05, power=0.8)
    assert isinstance(n, int)
    assert n >= 1


def test_mde_shrinks_as_the_sample_grows():
    small = minimum_detectable_effect(n=50, sd_diff=0.1, alpha=0.05, power=0.8)
    large = minimum_detectable_effect(n=500, sd_diff=0.1, alpha=0.05, power=0.8)
    assert large < small


def test_mde_and_required_n_are_consistent():
    # The MDE at the required n for a target effect should be close to that effect.
    effect = 0.02
    n = required_sample_size(effect=effect, sd_diff=0.08, alpha=0.05, power=0.8)
    mde = minimum_detectable_effect(n=n, sd_diff=0.08, alpha=0.05, power=0.8)
    assert mde == pytest.approx(effect, rel=0.05)


def test_a_non_positive_effect_is_refused():
    with pytest.raises(ValueError, match="effect must be positive"):
        required_sample_size(effect=0.0, sd_diff=0.1, alpha=0.05, power=0.8)


def test_a_non_positive_sd_is_refused():
    with pytest.raises(ValueError, match="sd_diff must be positive"):
        required_sample_size(effect=0.02, sd_diff=0.0, alpha=0.05, power=0.8)


def test_mde_refuses_a_sample_below_two():
    with pytest.raises(ValueError, match="n must be at least 2"):
        minimum_detectable_effect(n=1, sd_diff=0.1, alpha=0.05, power=0.8)


def test_mde_refuses_a_non_positive_sd():
    with pytest.raises(ValueError, match="sd_diff must be positive"):
        minimum_detectable_effect(n=50, sd_diff=0.0, alpha=0.05, power=0.8)
