"""Gate policy as data, not as constants scattered through the code.

Every number that decides a verdict lives in a policy file so a reviewer can change the gate's
behaviour without reading the source, and so the defaults are visible in one commented place. The
loader is deliberately small: it reads a TOML file with the standard library and validates the
fields it cares about, because an unreadable policy must fail loudly at the boundary rather than
producing a quietly wrong verdict later.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path


class ConfigError(ValueError):
    """Raised when a policy file is missing a field or carries an impossible value."""


@dataclass(frozen=True)
class GatePolicy:
    """The knobs that decide a verdict.

    k:                 cut-off for the ranking metrics (recall@k, ndcg@k).
    delta:             the smallest mean drop in the primary metric that counts as a regression
                       worth failing a build over. Smaller drops are noise the team tolerates.
    alpha:             two-sided significance level for the bootstrap interval.
    power:             target power used only by the required-sample-size arithmetic.
    n_resamples:       bootstrap resamples. Fixed by a convergence experiment, not by habit.
    primary_metric:    the metric the verdict is computed from. The others are reported, not gated.
    bootstrap_method:  "bca" or "percentile". The interval method whose calibration ADR 005 reports.
    """

    k: int = 10
    delta: float = 0.01
    alpha: float = 0.05
    power: float = 0.80
    n_resamples: int = 5000
    primary_metric: str = "ndcg"
    bootstrap_method: str = "percentile"

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ConfigError(f"k must be at least 1, got {self.k}")
        if not 0.0 < self.delta < 1.0:
            raise ConfigError(f"delta must lie in (0, 1), got {self.delta}")
        if not 0.0 < self.alpha < 0.5:
            raise ConfigError(f"alpha must lie in (0, 0.5), got {self.alpha}")
        if not 0.0 < self.power < 1.0:
            raise ConfigError(f"power must lie in (0, 1), got {self.power}")
        if self.n_resamples < 1000:
            raise ConfigError(f"n_resamples below 1000 is too noisy, got {self.n_resamples}")
        if self.primary_metric not in {"ndcg", "recall", "mrr"}:
            raise ConfigError(f"unknown primary_metric {self.primary_metric!r}")
        if self.bootstrap_method not in {"bca", "percentile"}:
            raise ConfigError(f"unknown bootstrap_method {self.bootstrap_method!r}")


DEFAULT_POLICY = GatePolicy()


def load_policy(path: str | Path) -> GatePolicy:
    """Load a policy from a TOML file, falling back to the default for any omitted field."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"policy file not found: {p}")
    with p.open("rb") as fh:
        raw = tomllib.load(fh)
    section = raw.get("gate", raw)
    known = set(GatePolicy.__dataclass_fields__)
    unknown = set(section) - known
    if unknown:
        raise ConfigError(f"policy file has unknown keys: {sorted(unknown)}")
    return replace(DEFAULT_POLICY, **section)
