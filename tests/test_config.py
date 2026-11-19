"""Properties of the policy and its loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragscore.config import DEFAULT_POLICY, ConfigError, GatePolicy, load_policy


def test_the_shipped_policy_file_loads():
    policy = load_policy(Path(__file__).resolve().parents[1] / "policy.toml")
    assert isinstance(policy, GatePolicy)
    assert policy.primary_metric == "ndcg"


def test_an_omitted_field_falls_back_to_the_default(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text("[gate]\ndelta = 0.05\n")
    policy = load_policy(path)
    assert policy.delta == 0.05
    assert policy.k == DEFAULT_POLICY.k


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_policy(tmp_path / "absent.toml")


def test_an_unknown_key_is_refused(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text("[gate]\nwobble = 3\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_policy(path)


def test_a_file_without_a_gate_table_is_read_at_the_top_level(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text("delta = 0.03\n")
    assert load_policy(path).delta == 0.03


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k": 0}, "k must be at least 1"),
        ({"delta": 0.0}, "delta must lie"),
        ({"delta": 1.5}, "delta must lie"),
        ({"alpha": 0.9}, "alpha must lie"),
        ({"power": 0.0}, "power must lie"),
        ({"n_resamples": 10}, "n_resamples"),
        ({"primary_metric": "bleu"}, "unknown primary_metric"),
        ({"bootstrap_method": "jack"}, "unknown bootstrap_method"),
    ],
)
def test_impossible_policies_are_refused(kwargs, message):
    with pytest.raises(ConfigError, match=message):
        GatePolicy(**kwargs)


def test_the_default_policy_is_valid():
    assert DEFAULT_POLICY.k >= 1
