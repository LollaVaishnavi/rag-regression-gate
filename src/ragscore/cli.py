"""Command line entry point.

Two subcommands. `gate` runs the shipped scenario (or a user-supplied golden-set size and fault)
and prints the verdict with the interval that produced it. `power` answers the sizing question on
its own: the smallest drop a set of a given size can resolve, and the size a target drop would need.

The exit codes are the contract and are documented in the README. The one that matters is 2,
"inconclusive": it is the tool reporting that the evidence is insufficient, which a caller must not
confuse with 0 (a clean pass) or 1 (a real regression). A CI job that treats 2 as success ships on
no evidence; a job that treats it as failure blocks on no evidence. The right response is to collect
more queries, and that is why `power` prints how many.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ragscore.config import DEFAULT_POLICY, GatePolicy, load_policy
from ragscore.evaluate import score_pair
from ragscore.gate import EXIT_ERROR, GateVerdict, gate
from ragscore.power import minimum_detectable_effect, required_sample_size
from ragscore.scenario import SHIPPED_BLEND, build_scenario


def _policy_from_args(args: argparse.Namespace) -> GatePolicy:
    if args.policy:
        return load_policy(args.policy)
    return DEFAULT_POLICY


def _run_gate(scenario_scores, policy: GatePolicy, seed: int) -> GateVerdict:
    return gate(scenario_scores.baseline, scenario_scores.candidate, policy, seed=seed)


def _print_gate(verdict: GateVerdict, policy: GatePolicy) -> None:
    iv = verdict.interval
    print("ragscore gate")
    print(f"  primary metric : {policy.primary_metric}@{policy.k}")
    print(f"  observed diff  : {iv.point:+.4f}  (candidate minus baseline)")
    print(f"  {int((1 - policy.alpha) * 100)}% interval  : [{iv.low:+.4f}, {iv.high:+.4f}]"
          f"  via {iv.method}")
    print(f"  delta (policy) : {policy.delta:.4f}")
    print(f"  naive verdict  : {verdict.naive_verdict.value}")
    print(f"  gate verdict   : {verdict.verdict.value}")
    if verdict.disagrees_with_naive:
        print("  note           : the gate and the naive threshold disagree on this sample")


def cmd_gate(args: argparse.Namespace) -> int:
    policy = _policy_from_args(args)
    scenario = build_scenario(golden_size=args.golden_size, blend=args.blend)
    scores = score_pair(
        corpus=scenario.corpus,
        golden=scenario.golden,
        baseline_config=scenario.baseline_config,
        candidate_config=scenario.candidate_config,
        metric=policy.primary_metric,
        k=policy.k,
        baseline_seed=args.baseline_seed,
        candidate_seed=args.candidate_seed,
    )
    verdict = _run_gate(scores, policy, seed=args.seed)
    _print_gate(verdict, policy)
    return verdict.exit_code


def cmd_power(args: argparse.Namespace) -> int:
    policy = _policy_from_args(args)
    scenario = build_scenario(golden_size=args.golden_size, blend=args.blend)
    scores = score_pair(
        corpus=scenario.corpus,
        golden=scenario.golden,
        baseline_config=scenario.baseline_config,
        candidate_config=scenario.candidate_config,
        metric=policy.primary_metric,
        k=policy.k,
        baseline_seed=args.baseline_seed,
        candidate_seed=args.candidate_seed,
    )
    sd_diff = float(np.std(scores.candidate - scores.baseline, ddof=1))
    mde = minimum_detectable_effect(args.golden_size, sd_diff, policy.alpha, policy.power)
    needed = required_sample_size(policy.delta, sd_diff, policy.alpha, policy.power)
    print("ragscore power")
    print(f"  golden set size    : {args.golden_size} queries")
    print(f"  per-query diff sd  : {sd_diff:.4f}")
    print(f"  delta (policy)     : {policy.delta:.4f}")
    print(f"  min detectable drop: {mde:.4f}  at this size")
    print(f"  queries to resolve : {needed}  to call a {policy.delta:.3f} drop at "
          f"{int((1 - policy.alpha) * 100)}%/{int(policy.power * 100)}%")
    if mde > policy.delta:
        print("  note               : this set cannot resolve the drop the policy cares about")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragscore", description=__doc__)
    parser.add_argument("--policy", help="path to a policy TOML file; defaults are used otherwise")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--golden-size", type=int, default=50, help="number of golden queries")
    common.add_argument("--blend", type=float, default=SHIPPED_BLEND, help="candidate fault size")
    common.add_argument("--baseline-seed", type=int, default=5, help="baseline run jitter seed")
    common.add_argument("--candidate-seed", type=int, default=6, help="candidate run jitter seed")

    g = sub.add_parser("gate", parents=[common], help="run the gate and print the verdict")
    g.add_argument("--seed", type=int, default=7, help="bootstrap seed")
    g.set_defaults(func=cmd_gate)

    p = sub.add_parser("power", parents=[common], help="print resolution and required sample size")
    p.set_defaults(func=cmd_power)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        print(f"ragscore: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
