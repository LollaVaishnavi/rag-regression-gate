"""Paired bootstrap confidence interval on a difference in mean metric.

The unit of resampling is the query, not the document and not the ranking, because the question the
gate answers is "would this difference survive a different sample of queries". Pairing matters: the
same query is scored by both retrievers, so the two per-query vectors are correlated and resampling
them together removes the between-query variance that a two-sample interval would wrongly keep.

Two interval methods are implemented. BCa (bias-corrected and accelerated) is the default, because
the per-query metric difference is bounded and skewed and BCa is the textbook correction for exactly
that. The percentile interval is kept as a plainer comparison point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Interval:
    """A two-sided confidence interval on mean(candidate) minus mean(baseline)."""

    point: float
    low: float
    high: float
    alpha: float
    method: str

    @property
    def width(self) -> float:
        return self.high - self.low


def _resample_means(diffs: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    return diffs[idx].mean(axis=1)


def _percentile(diffs: np.ndarray, resamples: np.ndarray, alpha: float) -> Interval:
    lo = float(np.quantile(resamples, alpha / 2.0))
    hi = float(np.quantile(resamples, 1.0 - alpha / 2.0))
    return Interval(point=float(diffs.mean()), low=lo, high=hi, alpha=alpha, method="percentile")


def _bca(diffs: np.ndarray, resamples: np.ndarray, alpha: float) -> Interval:
    observed = float(diffs.mean())
    n = diffs.size
    proportion_less = float(np.mean(resamples < observed))
    if proportion_less <= 0.0 or proportion_less >= 1.0:
        out = _percentile(diffs, resamples, alpha)
        return Interval(out.point, out.low, out.high, out.alpha, method="bca-degenerate")
    z0 = norm.ppf(proportion_less)

    jackknife = (diffs.sum() - diffs) / (n - 1)
    centred = jackknife.mean() - jackknife
    denom = 6.0 * (np.sum(centred**2) ** 1.5)
    if denom == 0.0:
        out = _percentile(diffs, resamples, alpha)
        return Interval(out.point, out.low, out.high, out.alpha, method="bca-degenerate")
    acceleration = float(np.sum(centred**3) / denom)

    def adjusted(z: float) -> float:
        value = z0 + (z0 + z) / (1.0 - acceleration * (z0 + z))
        return float(norm.cdf(value))

    lo_q = adjusted(norm.ppf(alpha / 2.0))
    hi_q = adjusted(norm.ppf(1.0 - alpha / 2.0))
    low = float(np.quantile(resamples, lo_q))
    high = float(np.quantile(resamples, hi_q))
    return Interval(point=observed, low=low, high=high, alpha=alpha, method="bca")


def paired_bootstrap_diff(
    baseline: np.ndarray,
    candidate: np.ndarray,
    n_resamples: int,
    alpha: float,
    seed: int,
    method: str = "bca",
) -> Interval:
    """Confidence interval for the mean paired difference candidate minus baseline.

    `method` selects "bca" or "percentile". BCa is the default; both are kept so a calibration
    experiment can measure them on the same data.
    """
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"baseline and candidate disagree in length: {baseline.shape} vs {candidate.shape}"
        )
    if baseline.size < 2:
        raise ValueError("a bootstrap needs at least two paired observations")

    diffs = candidate - baseline
    resamples = _resample_means(diffs, n_resamples, seed)
    if method == "percentile":
        return _percentile(diffs, resamples, alpha)
    if method == "bca":
        return _bca(diffs, resamples, alpha)
    raise ValueError(f"unknown bootstrap method {method!r}")
