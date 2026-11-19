"""How many bootstrap resamples does the interval need before it stops moving?

The policy fixes n_resamples at 5000. This experiment is the justification: it recomputes the
shipped
interval at a range of resample counts and reports how far the endpoints still move. A number chosen
by habit is a number a reviewer discounts; a number chosen by a printed convergence curve is not.
"""

from __future__ import annotations

from harness import write_json

from ragscore.bootstrap import paired_bootstrap_diff
from ragscore.evaluate import score_pair
from ragscore.scenario import build_scenario


def main() -> None:
    scenario = build_scenario()
    scores = score_pair(
        corpus=scenario.corpus,
        golden=scenario.golden,
        baseline_config=scenario.baseline_config,
        candidate_config=scenario.candidate_config,
        metric="ndcg",
        k=10,
        baseline_seed=5,
        candidate_seed=6,
    )
    counts = [1000, 2000, 4000, 8000, 16000]
    rows = []
    previous = None
    max_shift = 0.0
    for n in counts:
        iv = paired_bootstrap_diff(scores.baseline, scores.candidate, n, 0.05, seed=7, method="bca")
        shift = None
        if previous is not None:
            shift = max(abs(iv.low - previous[0]), abs(iv.high - previous[1]))
            max_shift = max(max_shift, shift)
        rows.append({"resamples": n, "low": iv.low, "high": iv.high, "shift_from_prev": shift})
        previous = (iv.low, iv.high)

    # The largest endpoint move once we are at or above the chosen 4000 resamples.
    tail_shift = max(
        r["shift_from_prev"] for r in rows if r["resamples"] >= 4000 and r["shift_from_prev"]
    )
    payload = {
        "chosen_resamples": 5000,
        "endpoint_shift_at_or_above_4000": round(tail_shift, 5),
        "rows": rows,
    }
    write_json("exp00_resamples", payload)

    print("resamples   low       high      shift")
    for r in rows:
        shift = "-" if r["shift_from_prev"] is None else f"{r['shift_from_prev']:.5f}"
        print(f"{r['resamples']:>8}  {r['low']:+.4f}  {r['high']:+.4f}  {shift}")
    print(f"\nendpoint shift at or above 4000 resamples: {tail_shift:.5f}")


if __name__ == "__main__":
    main()
