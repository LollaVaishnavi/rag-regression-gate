"""Photograph the real HTML report, and fail if the verdict badge is not legible.

The screenshot in the README has to be of the report the tool actually produces, so this builds the
report first (rather than photographing whatever is on disk) and then captures it. The one check
that
only a screenshot tool can do is pixels: it reads back the computed style of the verdict badge and
fails if the badge has collapsed to invisible, which is how a real badge once rendered green on
green
while every test still passed.

Dev-time tool. Needs a browser and is skipped in the pure-Python CI job.
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from ragscore.report import write_report

ROOT = Path(__file__).resolve().parents[1]
REPORT_HTML = ROOT / "reports" / "report.html"
OUT_PNG = ROOT / "docs" / "img" / "report.png"
MANIFEST = ROOT / "docs" / "img" / "screenshot-manifest.json"
CHROME = "/usr/bin/google-chrome"


def _badge_reading(page) -> dict:
    return page.evaluate(
        """() => {
            const badge = document.querySelector('.badge');
            if (!badge) return null;
            const style = getComputedStyle(badge);
            return {
                color: style.color,
                background: style.backgroundColor,
                visible: badge.offsetWidth > 0 && badge.offsetHeight > 0,
            };
        }"""
    )


def main() -> int:
    write_report(REPORT_HTML)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 820, "height": 900}, device_scale_factor=2)
        page.goto(REPORT_HTML.as_uri())
        page.wait_for_selector(".badge")
        reading = _badge_reading(page)
        if reading is None or not reading["visible"]:
            browser.close()
            raise SystemExit("verdict badge is not visible in the rendered report")
        if reading["color"] == reading["background"]:
            browser.close()
            colour = reading["color"]
            raise SystemExit(f"verdict badge is illegible: text and background are both {colour}")
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(OUT_PNG), full_page=True)
        browser.close()

    MANIFEST.write_text(
        json.dumps(
            {
                "output": str(OUT_PNG.relative_to(ROOT)),
                "source_report": str(REPORT_HTML.relative_to(ROOT)),
                "badge": reading,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT_PNG.relative_to(ROOT)}, badge {reading['color']} on {reading['background']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
