"""Re-measure every published figure and write docs/metrics.json.

Nothing in this file types a number by hand. It reads the test count and coverage from the machine
readable reports pytest writes, runs every experiment and reads the JSON each one emits, and turns
those into the exact display strings the README uses, each paired with an anchor phrase.
check_numbers
then verifies the README against this file, so a figure that drifts from the code fails the build.

Run it two ways. In CI the test suite runs as its own step and this is called with --skip-tests,
which
reads the reports that step produced. Locally, without the flag, it runs pytest first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EXPERIMENTS = ROOT / "experiments"
DOCS_EXPERIMENTS = ROOT / "docs" / "experiments"
METRICS_PATH = ROOT / "docs" / "metrics.json"

EXPERIMENT_SCRIPTS = [
    "exp00_resamples.py",
    "exp01_calibration.py",
    "exp02_resolution.py",
    "exp03_gate_comparison.py",
    "exp04_shipped.py",
]


def _pct(value: float) -> str:
    """Format a percentage the way the prose does: no trailing .0 for a whole number."""
    rounded = round(value, 1)
    if rounded == int(rounded):
        return f"{int(rounded)}%"
    return f"{rounded}%"


def run_tests() -> None:
    REPORTS.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "pytest",
            f"--junitxml={REPORTS / 'junit.xml'}",
            "--cov=ragscore",
            f"--cov-report=json:{REPORTS / 'coverage.json'}",
            "-q",
        ],
        cwd=ROOT,
        check=True,
    )


def read_test_count() -> int:
    tree = ET.parse(REPORTS / "junit.xml")
    root = tree.getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    tests = int(suite.get("tests"))
    failures = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    if failures:
        raise SystemExit(
            f"collect_metrics refuses to publish metrics with {failures} failing tests"
        )
    return tests


def read_coverage() -> float:
    data = json.loads((REPORTS / "coverage.json").read_text())
    return float(data["totals"]["percent_covered"])


def run_experiments() -> None:
    for script in EXPERIMENT_SCRIPTS:
        subprocess.run([sys.executable, str(EXPERIMENTS / script)], cwd=ROOT, check=True)


def load_experiment(name: str) -> dict:
    return json.loads((DOCS_EXPERIMENTS / f"{name}.json").read_text())


def build_metrics(tests_total: int, coverage: float) -> tuple[dict, dict, dict]:
    calib = load_experiment("exp01_calibration")
    resolution = load_experiment("exp02_resolution")
    comparison = load_experiment("exp03_gate_comparison")
    shipped = load_experiment("exp04_shipped")
    resamples = load_experiment("exp00_resamples")

    coinflip = next(r for r in comparison["rows"] if r["blend"] == 0.015)
    null_row = next(r for r in comparison["rows"] if r["blend"] == 0.0)

    precise = {
        "tests_total": tests_total,
        "coverage_line_pct": coverage,
        "bca_null_fp_pct": calib["bca_false_positive_pct"],
        "percentile_null_fp_pct": calib["percentile_false_positive_pct"],
        "naive_null_fp_pct": calib["naive_false_positive_pct"],
        "nominal_fp_pct": calib["nominal_false_positive_pct"],
        "calibration_trials": calib["trials"],
        "gate_null_regression_pct": null_row["gate_regression_pct"],
        "coinflip_naive_pass_pct": coinflip["naive_pass_pct"],
        "coinflip_naive_regression_pct": coinflip["naive_regression_pct"],
        "coinflip_gate_inconclusive_pct": coinflip["gate_inconclusive_pct"],
        "required_queries": resolution["required_queries_for_delta"],
        "golden_size": resolution["golden_size"],
        "min_detectable_drop": resolution["min_detectable_drop"],
        "per_query_diff_sd": resolution["per_query_diff_sd"],
        "shipped_population_drop": abs(shipped["population_true_drop"]),
        "shipped_observed_drop": shipped["observed_drop"],
        "shipped_interval_low": shipped["interval_low"],
        "shipped_interval_high": shipped["interval_high"],
        "chosen_resamples": resamples["chosen_resamples"],
    }

    display = {
        "tests_total": str(tests_total),
        "coverage_line_pct": f"{int(round(coverage))}%",
        "bca_null_fp_pct": _pct(precise["bca_null_fp_pct"]),
        "percentile_null_fp_pct": _pct(precise["percentile_null_fp_pct"]),
        "naive_null_fp_pct": _pct(precise["naive_null_fp_pct"]),
        "nominal_fp_pct": _pct(precise["nominal_fp_pct"]),
        "calibration_trials": str(precise["calibration_trials"]),
        "gate_null_regression_pct": _pct(precise["gate_null_regression_pct"]),
        "coinflip_naive_pass_pct": _pct(precise["coinflip_naive_pass_pct"]),
        "coinflip_naive_regression_pct": _pct(precise["coinflip_naive_regression_pct"]),
        "coinflip_gate_inconclusive_pct": _pct(precise["coinflip_gate_inconclusive_pct"]),
        "required_queries": str(precise["required_queries"]),
        "golden_size": str(precise["golden_size"]),
        "min_detectable_drop": f"{precise['min_detectable_drop']:.3f}",
        "per_query_diff_sd": f"{precise['per_query_diff_sd']:.3f}",
        "shipped_population_drop": f"{precise['shipped_population_drop']:.3f}",
        "shipped_observed_drop": f"{precise['shipped_observed_drop']:.3f}",
        "shipped_interval_low": f"{precise['shipped_interval_low']:.3f}",
        "shipped_interval_high": f"{precise['shipped_interval_high']:.3f}",
        "chosen_resamples": str(precise["chosen_resamples"]),
    }

    anchors = {
        "tests_total": ["{} tests"],
        "coverage_line_pct": ["{} of lines", "{} line coverage"],
        "bca_null_fp_pct": ["BCa fired on {}", "BCa called a regression on {}"],
        "percentile_null_fp_pct": ["percentile stayed at {}", "percentile interval held at {}"],
        "naive_null_fp_pct": ["naive threshold fired on {}", "naive fired on {}"],
        "nominal_fp_pct": ["nominal {}"],
        "calibration_trials": ["{} null golden sets", "over {} null"],
        "gate_null_regression_pct": ["gate fires on {}", "calibrated to {}"],
        "coinflip_naive_pass_pct": ["passed it on {}", "ships it on {}"],
        "coinflip_naive_regression_pct": ["failed it on {}", "fails it on {}"],
        "coinflip_gate_inconclusive_pct": ["inconclusive on {}", "{} of those sets"],
        "required_queries": ["needs {} queries", "{} queries to resolve"],
        "golden_size": ["{}-query golden set", "set has {} queries", "{} queries it holds"],
        "min_detectable_drop": ["resolve a drop of {}", "resolve about {}"],
        "per_query_diff_sd": ["difference spread of {}", "spread of {}"],
        "shipped_population_drop": ["a real {} drop", "carries a {} drop"],
        "shipped_observed_drop": ["measured drop is {}", "observed drop of {}"],
        "shipped_interval_low": ["[{},", "interval of [{}"],
        "shipped_interval_high": [", {}]", "{}] still includes"],
        "chosen_resamples": ["{} resamples"],
    }
    return precise, display, anchors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="read existing test reports")
    args = parser.parse_args(argv)

    if not args.skip_tests:
        run_tests()
    tests_total = read_test_count()
    coverage = read_coverage()
    run_experiments()
    precise, display, anchors = build_metrics(tests_total, coverage)

    payload = {
        "metrics": display,
        "precise": precise,
        "anchors": anchors,
        "checked_documents": ["README.md"],
        "note": "every value here is produced by running the suite and the experiments",
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {METRICS_PATH.relative_to(ROOT)} with {len(display)} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
