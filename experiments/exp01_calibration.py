"""Is the gate calibrated under the null, and which interval method keeps it there?

This is the experiment that decides the gate's default. Under the null the candidate is the baseline
config run a second time, so the true difference is zero and nothing but run jitter separates the
two
score vectors. A 95 percent interval should then call a regression no more than about 5 percent of
the
time. BCa does not: the paired-difference vector is zero-inflated, most queries score identically,
and
BCa's acceleration is estimated from a jackknife of a near-constant vector. The percentile interval,
which does not estimate an acceleration, stays close to nominal. The naive threshold is measured
too,
for the comparison the README leads with.
"""

from __future__ import annotations

from harness import default_policy_for_experiments, fixed_corpus, run_trials, write_json

from ragscore.bootstrap import paired_bootstrap_diff
from ragscore.gate import naive_gate
from ragscore.retrieval import BASELINE

TRIALS = 400
N_QUERIES = 50


def main() -> None:
    policy = default_policy_for_experiments()
    corpus = fixed_corpus()
    trials = run_trials(
        corpus,
        candidate_config=BASELINE,  # the null: same config, different run seed
        trials=TRIALS,
        n_queries=N_QUERIES,
        k=policy.k,
        metric=policy.primary_metric,
    )

    bca_fp = 0
    percentile_fp = 0
    naive_fp = 0
    for ts in trials:
        bca = paired_bootstrap_diff(
            ts.baseline, ts.candidate, policy.n_resamples, policy.alpha, seed=7, method="bca"
        )
        perc = paired_bootstrap_diff(
            ts.baseline, ts.candidate, policy.n_resamples, policy.alpha, seed=7, method="percentile"
        )
        if bca.high < 0.0:
            bca_fp += 1
        if perc.high < 0.0:
            percentile_fp += 1
        if naive_gate(ts.baseline, ts.candidate, policy.delta).value == "regression":
            naive_fp += 1

    payload = {
        "trials": TRIALS,
        "n_queries": N_QUERIES,
        "nominal_false_positive_pct": round(policy.alpha * 100, 1),
        "bca_false_positive_pct": round(100 * bca_fp / TRIALS, 1),
        "percentile_false_positive_pct": round(100 * percentile_fp / TRIALS, 1),
        "naive_false_positive_pct": round(100 * naive_fp / TRIALS, 1),
    }
    write_json("exp01_calibration", payload)

    print(f"null false-positive (regression) rate over {TRIALS} golden sets of {N_QUERIES} queries")
    print(f"  nominal     : {payload['nominal_false_positive_pct']}%")
    print(f"  BCa         : {payload['bca_false_positive_pct']}%")
    print(f"  percentile  : {payload['percentile_false_positive_pct']}%")
    print(f"  naive       : {payload['naive_false_positive_pct']}%")


if __name__ == "__main__":
    main()
