"""The shipped scenario: one golden set, one candidate, the verdict a team would actually get.

The candidate carries a real corrupted-shard fault whose population drop is measured here on a large
draw. On the single fifty-query golden set the team happens to hold, the observed drop is smaller,
and
the naive threshold reads it as a pass: it ships the regression. The gate reports an interval that
still admits a delta-sized drop and returns inconclusive instead. This is the whole argument in one
run, so the README leads with it.
"""

from __future__ import annotations

from harness import (
    candidate_for_blend,
    default_policy_for_experiments,
    fixed_corpus,
    run_trials,
    write_json,
)

from ragscore.evaluate import score_pair
from ragscore.gate import gate
from ragscore.scenario import SHIPPED_BLEND, build_scenario

LARGE_SAMPLE = 4000


def main() -> None:
    policy = default_policy_for_experiments()
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

    corpus = fixed_corpus()
    pop = run_trials(
        corpus, candidate_for_blend(SHIPPED_BLEND), 1, LARGE_SAMPLE, policy.k, policy.primary_metric
    )
    true_drop = float((pop[0].candidate - pop[0].baseline).mean())

    iv = verdict.interval
    payload = {
        "golden_size": scenario.golden.n_queries,
        "population_true_drop": round(true_drop, 4),
        "observed_drop": round(-iv.point, 4) if iv.point < 0 else round(iv.point, 4),
        "observed_diff": round(iv.point, 4),
        "interval_low": round(iv.low, 4),
        "interval_high": round(iv.high, 4),
        "interval_method": iv.method,
        "naive_verdict": verdict.naive_verdict.value,
        "gate_verdict": verdict.verdict.value,
        "naive_ships_the_regression": verdict.naive_verdict.value == "pass",
    }
    write_json("exp04_shipped", payload)

    print(f"population true drop : {true_drop:+.4f} ndcg@{policy.k}")
    print(f"observed on {scenario.golden.n_queries} queries: {iv.point:+.4f}")
    print(f"95% interval         : [{iv.low:+.4f}, {iv.high:+.4f}] via {iv.method}")
    print(f"naive verdict        : {verdict.naive_verdict.value}")
    print(f"gate verdict         : {verdict.verdict.value}")
    if payload["naive_ships_the_regression"]:
        print("the naive threshold ships a candidate that carries a real regression")


if __name__ == "__main__":
    main()
