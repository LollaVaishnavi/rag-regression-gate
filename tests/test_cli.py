"""Properties of the command line interface and its exit codes."""

from __future__ import annotations

import numpy as np

from ragscore.cli import main
from ragscore.gate import EXIT_ERROR, EXIT_INCONCLUSIVE, EXIT_REGRESSION


def test_the_shipped_gate_run_is_inconclusive(capsys):
    code = main(["gate"])
    out = capsys.readouterr().out
    assert code == EXIT_INCONCLUSIVE
    assert "inconclusive" in out


def test_a_large_fault_exits_as_a_regression():
    code = main(["gate", "--blend", "0.12"])
    assert code == EXIT_REGRESSION


def test_the_power_command_reports_a_required_sample_size(capsys):
    code = main(["power"])
    out = capsys.readouterr().out
    assert code == 0
    assert "queries to resolve" in out


def test_a_bad_golden_size_is_reported_as_a_tool_error(capsys):
    code = main(["gate", "--golden-size", "1"])
    err = capsys.readouterr().err
    assert code == EXIT_ERROR
    assert "error" in err


def test_the_gate_output_names_both_verdicts(capsys):
    main(["gate"])
    out = capsys.readouterr().out
    assert "naive verdict" in out
    assert "gate verdict" in out


def test_a_custom_policy_file_is_honoured(tmp_path, capsys):
    path = tmp_path / "p.toml"
    path.write_text("[gate]\ndelta = 0.5\n")
    # With an enormous delta nothing counts as a regression, so the shipped run passes.
    code = main(["--policy", str(path), "gate"])
    out = capsys.readouterr().out
    assert "gate verdict   : pass" in out
    assert code == 0


def test_the_power_note_appears_when_the_set_is_too_small(capsys):
    main(["power", "--golden-size", "20"])
    out = capsys.readouterr().out
    assert "cannot resolve" in out


def test_main_returns_an_int_exit_code():
    rng = np.random.default_rng(0)
    assert isinstance(rng, np.random.Generator)  # guard: numpy import is live
    assert isinstance(main(["power"]), int)
