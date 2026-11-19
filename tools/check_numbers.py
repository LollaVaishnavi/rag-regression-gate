"""Fail the build when the README quotes a figure the code no longer produces.

It loads docs/metrics.json, then for every metric checks that at least one of its anchor phrases,
with the value substituted in, appears in each checked document. Anchors are alternatives because
the
same figure reads differently in a table and a sentence, and each anchor is a number plus a few
words
rather than a whole clause so it survives markdown line wrapping.

It also runs the reverse check: numbers in the README that match no metric are reported, because
those
are numbers nothing re-measures. Fenced and inline code are excluded first, so an example invocation
with a made-up row count is not flagged. The reverse check prints; only the forward check fails the
build, because a false positive on the reverse check trains a reader to ignore it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "docs" / "metrics.json"


def strip_code(text: str) -> str:
    """Remove the parts of a document a number check must not read.

    Fenced blocks and inline code spans hold example invocations with made-up values. HTML tags hold
    attributes such as an image width. Markdown link and image targets hold ADR filenames with
    numbers in them. None of these is a published figure, so flagging them would only teach a reader
    to ignore the reverse check.
    """
    out = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    out = re.sub(r"`[^`]*`", " ", out)
    out = re.sub(r"<[^>]+>", " ", out)
    out = re.sub(r"\]\([^)]*\)", "] ", out)
    out = re.sub(r"ADR[\s-]\d+", " ", out)
    return out


def forward_check(metrics: dict, anchors: dict, documents: list[str]) -> list[str]:
    failures: list[str] = []
    for doc_name in documents:
        doc_path = ROOT / doc_name
        text = doc_path.read_text()
        for key, value in metrics.items():
            templates = anchors.get(key, [])
            if not templates:
                failures.append(f"{doc_name}: metric {key} has no anchor phrase")
                continue
            phrases = [t.format(value) for t in templates]
            if not any(phrase in text for phrase in phrases):
                failures.append(
                    f"{doc_name}: none of the anchors for {key} ({value!r}) were found; "
                    f"tried {phrases}"
                )
    return failures


def reverse_check(metrics: dict, documents: list[str]) -> list[str]:
    known = set(metrics.values())
    known_bare = {v.rstrip("%") for v in known}
    unverified: list[str] = []
    number_pattern = re.compile(r"-?\d+(?:\.\d+)?%?")
    for doc_name in documents:
        text = strip_code((ROOT / doc_name).read_text())
        for token in number_pattern.findall(text):
            if token in known or token.rstrip("%") in known_bare:
                continue
            unverified.append(f"{doc_name}: unverified number {token!r}")
    return unverified


def main() -> int:
    if not METRICS_PATH.is_file():
        print("docs/metrics.json is missing; run tools/collect_metrics.py first", file=sys.stderr)
        return 2
    payload = json.loads(METRICS_PATH.read_text())
    metrics = payload["metrics"]
    anchors = payload["anchors"]
    documents = payload["checked_documents"]

    failures = forward_check(metrics, anchors, documents)
    unverified = reverse_check(metrics, documents)

    if unverified:
        print("reverse check, numbers in prose that no metric re-measures (informational):")
        for item in unverified:
            print(f"  {item}")

    if failures:
        print("\nnumber check FAILED:")
        for item in failures:
            print(f"  {item}")
        return 1
    print(f"number check passed: {len(metrics)} metrics anchored in {documents}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
