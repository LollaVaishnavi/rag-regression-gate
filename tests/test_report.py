"""Properties of the HTML report, including the invariant that caught an invisible badge."""

from __future__ import annotations

import re

from ragscore.report import STYLE, render_report, write_report


def test_the_report_shows_the_inconclusive_verdict_for_the_shipped_scenario():
    html = render_report()
    assert "badge-inconclusive" in html
    assert ">inconclusive<" in html


def test_every_css_class_in_the_markup_has_a_rule_in_the_stylesheet():
    # The bug this pins: a class used in markup but absent from the stylesheet rendered the verdict
    # badge invisible, green on green, while every test still passed. A class with no rule is a bug.
    html = render_report()
    body = html.split("</style>", 1)[1]
    used = set()
    for attr in re.findall(r'class="([^"]+)"', body):
        used.update(attr.split())
    for cls in used:
        assert f".{cls}" in STYLE, f"class {cls!r} is used in the markup but has no CSS rule"


def test_the_report_names_both_the_gate_and_the_naive_verdict():
    html = render_report()
    assert "Gate verdict" in html
    assert "naive threshold verdict" in html


def test_write_report_creates_a_file(tmp_path):
    path = write_report(tmp_path / "report.html")
    assert path.is_file()
    assert path.read_text().startswith("<!DOCTYPE html>")
