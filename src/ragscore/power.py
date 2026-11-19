"""Sample-size arithmetic: how many queries a verdict would actually need.

This is the half of the project that a confidence interval alone does not give you. The interval
says whether the current sample can resolve the observed difference; the power calculation says how
large a sample would have to be to resolve a difference the team cares about, and conversely the
smallest difference the current sample could resolve. Both come from the same normal approximation
for a paired mean, which is stated plainly in the README as an approximation rather than sold as
exact, because near the boundary a bootstrap and this formula can disagree by a query or two.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def required_sample_size(
    effect: float,
    sd_diff: float,
    alpha: float,
    power: float,
) -> int:
    """Smallest paired sample size to detect a mean difference of `effect` at `alpha` and `power`.

    `effect` is the mean paired difference to detect, in metric units, and `sd_diff` is the standard
    deviation of the per-query paired differences. The result is rounded up, because a fractional
    query cannot be collected and rounding down would under-power the test.
    """
    if effect <= 0.0:
        raise ValueError(f"effect must be positive, got {effect}")
    if sd_diff <= 0.0:
        raise ValueError(f"sd_diff must be positive, got {sd_diff}")
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_power = norm.ppf(power)
    n = ((z_alpha + z_power) * sd_diff / effect) ** 2
    return int(np.ceil(n))


def minimum_detectable_effect(
    n: int,
    sd_diff: float,
    alpha: float,
    power: float,
) -> float:
    """The smallest mean difference a paired sample of size `n` can detect at `alpha` and `power`.

    This is the inverse of `required_sample_size` and is the honest way to report what a golden set
    of a given size can and cannot see, which is the number the gate turns into an "inconclusive"
    verdict.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if sd_diff <= 0.0:
        raise ValueError(f"sd_diff must be positive, got {sd_diff}")
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_power = norm.ppf(power)
    return float((z_alpha + z_power) * sd_diff / np.sqrt(n))
