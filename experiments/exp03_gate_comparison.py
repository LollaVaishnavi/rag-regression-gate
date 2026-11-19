"""How do the naive threshold and the gate behave as the true drop grows from zero?

For a range of fault sizes this draws many fifty-query golden sets and tallies what each gate says.
Two things show up. Near a true drop the size of the policy's delta the naive verdict is close to a
coin flip across golden sets: the same change is failed on some samples and passed on others, decided
by which queries were drawn. The gate replaces most of those confident-but-wrong verdicts with
"inconclusive", which is the honest reading of a sample that cannot resolve the effect.
"""

from __future__ import annotations

from harness import (
    candidate_for_blend,
    default_policy_for_experiments,
    fixed_corpus,
    population_sd_diff,
    run_trials,
    write_json,
)

from ragscore.gate import gate, naive_gate

TRIALS = 200
N_QUERIES = 50
BLENDS = [0.0, 0.015, 0.03, 0.05, 0.08]
LARGE_SAMPLE = 4000


def _true_drop(corpus, candidate, policy) -> float:
    scores = run_trials(
        corpus, candidate, trials=1, n_queries=LARGE_SAMPLE, k=policy.k, metric=policy.primary_metric
    )
    return float((scores[0].candidate - scores[0].baseline).mean())


def main() -> None:
    policy = default_policy_for_experiments()
    corpus = fixed_corpus()
    rows = []
    for blend in BLENDS:
        candidate = candidate_for_blend(blend)
        true_drop = _true_drop(corpus, candidate, policy)
        trials = run_trials(
            corpus, candidate, TRIALS, N_QUERIES, policy.k, policy.primary_metric
        )
        naive = {"pass": 0, "regression": 0}
        gated = {"pass": 0, "regression": 0, "inconclusive": 0}
        for ts in trials:
            naive[naive_gate(ts.baseline, ts.candidate, policy.delta).value] += 1
            gated[gate(ts.baseline, ts.candidate, policy, seed=7).verdict.value] += 1
        rows.append(
            {
                "blend": blend,
                "true_drop": round(true_drop, 4),
                "naive_pass_pct": round(100 * naive["pass"] / TRIALS, 1),
                "naive_regression_pct": round(100 * naive["regression"] / TRIALS, 1),
                "gate_pass_pct": round(100 * gated["pass"] / TRIALS, 1),
                "gate_regression_pct": round(100 * gated["regression"] / TRIALS, 1),
                "gate_inconclusive_pct": round(100 * gated["inconclusive"] / TRIALS, 1),
            }
        )

    payload = {
        "trials": TRIALS,
        "n_queries": N_QUERIES,
        "delta": policy.delta,
        "method": policy.bootstrap_method,
        "rows": rows,
    }
    write_json("exp03_gate_comparison", payload)

    header = (
        f"{'blend':>6} {'drop':>8} | {'naive_pass':>10} {'naive_reg':>9} | "
        f"{'g_pass':>7} {'g_reg':>6} {'g_inconc':>9}"
    )
    print(header)
    for r in rows:
        print(
            f"{r['blend']:>6.3f} {r['true_drop']:>8.4f} | "
            f"{r['naive_pass_pct']:>9.1f}% {r['naive_regression_pct']:>8.1f}% | "
            f"{r['gate_pass_pct']:>6.1f}% {r['gate_regression_pct']:>5.1f}% "
            f"{r['gate_inconclusive_pct']:>8.1f}%"
        )


if __name__ == "__main__":
    main()
