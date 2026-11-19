"""Render the shipped scenario as a self-contained HTML report.

The report exists so there is something a screenshot tool can photograph that a reviewer trusts: it
is generated from the same gate the CLI runs, not drawn by hand. Every CSS class used in the markup
has a rule in the stylesheet, which a test asserts, because a class with no rule is how a verdict
badge once rendered invisible. Run `python -m ragscore.report` to write it to reports/report.html.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ragscore.config import DEFAULT_POLICY, GatePolicy
from ragscore.evaluate import score_pair
from ragscore.gate import GateVerdict, gate
from ragscore.power import minimum_detectable_effect, required_sample_size
from ragscore.scenario import build_scenario

VERDICT_CLASS = {
    "pass": "badge badge-pass",
    "regression": "badge badge-regression",
    "inconclusive": "badge badge-inconclusive",
}

STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
       max-width: 760px; color: #1a1a1a; background: #ffffff; }
h1 { font-size: 1.5rem; }
h2 { font-size: 1.05rem; margin-top: 1.6rem; border-bottom: 1px solid #e3e3e3;
     padding-bottom: 0.2rem; }
table { border-collapse: collapse; width: 100%; margin-top: 0.6rem; }
td, th { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid #ededed;
         font-variant-numeric: tabular-nums; }
th { color: #555; font-weight: 600; }
.badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 0.4rem;
         font-weight: 700; color: #ffffff; }
.badge-pass { background: #1b7f37; }
.badge-regression { background: #b3261e; }
.badge-inconclusive { background: #8a6d00; }
.note { color: #555; font-size: 0.9rem; margin-top: 0.5rem; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
"""


def _rows(verdict: GateVerdict, policy: GatePolicy, mde: float, needed: int, golden: int) -> str:
    iv = verdict.interval
    pairs = [
        ("primary metric", f"{policy.primary_metric}@{policy.k}"),
        ("golden set size", f"{golden} queries"),
        ("observed difference", f"{iv.point:+.4f} (candidate minus baseline)"),
        (f"{int((1 - policy.alpha) * 100)}% interval", f"[{iv.low:+.4f}, {iv.high:+.4f}]"),
        ("interval method", iv.method),
        ("delta (policy)", f"{policy.delta:.3f}"),
        ("smallest resolvable drop", f"{mde:.3f} (population spread)"),
        ("queries to resolve delta", f"{needed} (has {golden})"),
    ]
    return "\n".join(
        f"<tr><td>{label}</td><td class='mono'>{value}</td></tr>" for label, value in pairs
    )


def _population_sd(policy: GatePolicy, sample: int = 4000) -> float:
    """Per-query difference spread estimated on a large draw, matching experiment 02.

    The resolution rows use this rather than the fifty-query set's own spread, because an sd
    estimated from fifty queries is itself noisy and would disagree with the headline numbers. The
    verdict and interval above stay on the fifty queries a team actually holds.
    """
    big = build_scenario(golden_size=sample)
    scores = score_pair(
        corpus=big.corpus,
        golden=big.golden,
        baseline_config=big.baseline_config,
        candidate_config=big.candidate_config,
        metric=policy.primary_metric,
        k=policy.k,
        baseline_seed=5,
        candidate_seed=6,
    )
    return float(np.std(scores.candidate - scores.baseline, ddof=1))


def render_report() -> str:
    """Build the HTML string for the shipped scenario."""
    policy = DEFAULT_POLICY
    scenario = build_scenario()
    scores = score_pair(
        corpus=scenario.corpus,
        golden=scenario.golden,
        baseline_config=scenario.baseline_config,
        candidate_config=scenario.candidate_config,
        metric=policy.primary_metric,
        k=policy.k,
        baseline_seed=5,
        candidate_seed=6,
    )
    verdict = gate(scores.baseline, scores.candidate, policy, seed=7)
    sd_diff = _population_sd(policy)
    mde = minimum_detectable_effect(scenario.golden.n_queries, sd_diff, policy.alpha, policy.power)
    needed = required_sample_size(policy.delta, sd_diff, policy.alpha, policy.power)

    badge_class = VERDICT_CLASS[verdict.verdict.value]
    naive = verdict.naive_verdict.value
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>ragscore report</title><style>{STYLE}</style></head>
<body>
<h1>rag-regression-gate: shipped scenario</h1>
<p>Gate verdict: <span class="{badge_class}">{verdict.verdict.value}</span>
   &nbsp; naive threshold verdict: <span class="mono">{naive}</span></p>
<h2>Evidence</h2>
<table>{_rows(verdict, policy, mde, needed, scenario.golden.n_queries)}</table>
<p class="note">The naive threshold reads the observed drop as "{naive}".
The interval still admits a drop as large as delta, so the gate returns
"{verdict.verdict.value}": this golden set cannot resolve the change.
Collect {needed} queries to decide it.</p>
</body>
</html>
"""


def write_report(path: str | Path = "reports/report.html") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report())
    return out


def main() -> int:  # pragma: no cover - thin IO wrapper exercised by the screenshot tool
    path = write_report()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
