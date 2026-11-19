"""What can a fifty-query golden set actually resolve, and how many queries would it take?

This is the resolution arithmetic. Using the per-query difference spread measured on a large draw of
the shipped candidate, it reports the smallest drop a fifty-query set can detect at the policy's
alpha
and power, and the number of queries the policy's delta would actually need. The gap between the
set a
team has and the set the claim needs is the number this project exists to make visible.
"""

from __future__ import annotations

import numpy as np
from harness import default_policy_for_experiments, write_json

from ragscore.evaluate import score_pair
from ragscore.power import minimum_detectable_effect, required_sample_size
from ragscore.scenario import SHIPPED_GOLDEN_SIZE, build_scenario

LARGE_SAMPLE = 4000


def main() -> None:
    policy = default_policy_for_experiments()
    # Estimate the per-query difference spread on a large draw of the shipped scenario, the same
    # draw the report uses, so the published resolution numbers and the report agree exactly.
    big = build_scenario(golden_size=LARGE_SAMPLE)
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
    sd_diff = float(np.std(scores.candidate - scores.baseline, ddof=1))
    mde = minimum_detectable_effect(SHIPPED_GOLDEN_SIZE, sd_diff, policy.alpha, policy.power)
    required = required_sample_size(policy.delta, sd_diff, policy.alpha, policy.power)

    payload = {
        "golden_size": SHIPPED_GOLDEN_SIZE,
        "per_query_diff_sd": round(sd_diff, 4),
        "delta": policy.delta,
        "alpha": policy.alpha,
        "power": policy.power,
        "min_detectable_drop": round(mde, 4),
        "required_queries_for_delta": required,
    }
    write_json("exp02_resolution", payload)

    print(f"per-query difference sd        : {sd_diff:.4f}")
    print(f"golden set size                : {SHIPPED_GOLDEN_SIZE} queries")
    print(f"smallest drop it can resolve   : {mde:.4f} ndcg@{policy.k}")
    print(f"delta the policy cares about   : {policy.delta:.4f}")
    print(f"queries to resolve that delta  : {required}")


if __name__ == "__main__":
    main()
