"""Render the README's mermaid source to a committed SVG that GitHub displays reliably.

GitHub's own mermaid rendering fails intermittently from inside its lazily loaded renderer, so the
diagram is rendered here instead and embedded with an img tag. Three post-processing steps turn the
mermaid output into an SVG that loads as an image rather than only inside a live page: html labels
are
turned off so the file is valid XML, the root element is given intrinsic width and height from the
viewBox so a browser has an aspect ratio, and a light background rectangle is painted so edge labels
are legible in a dark theme. See the rendering traps in the project notes.

This is a dev-time tool, not part of make verify. It needs @mermaid-js/mermaid-cli installed with
npm
and drives it against the system Chrome, so it runs where a browser is available and is skipped in
the
pure-Python CI job.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT_SVG = ROOT / "docs" / "img" / "architecture.svg"
MANIFEST = ROOT / "docs" / "img" / "diagram-manifest.json"
MMDC = ROOT / "node_modules" / ".bin" / "mmdc"
CHROME = "/usr/bin/google-chrome"


def extract_source(markdown: str) -> str:
    match = re.search(r"```mermaid-source\n(.*?)```", markdown, flags=re.DOTALL)
    if not match:
        raise SystemExit("no mermaid-source fence found in README.md")
    return match.group(1).strip()


def give_intrinsic_dimensions(svg: str) -> str:
    """Replace the percentage width with explicit pixel width and height from the viewBox."""
    viewbox = re.search(r'viewBox="([\d.\s-]+)"', svg)
    if not viewbox:
        return svg
    _, _, width, height = viewbox.group(1).split()
    svg = re.sub(r'\bwidth="[^"]*"', f'width="{float(width):.0f}"', svg, count=1)
    if re.search(r"<svg[^>]*\bheight=", svg):
        svg = re.sub(r'\bheight="[^"]*"', f'height="{float(height):.0f}"', svg, count=1)
    else:
        svg = svg.replace("<svg", f'<svg height="{float(height):.0f}"', 1)
    return re.sub(r"max-width:\s*[\d.]+px", f"max-width: {float(width):.0f}px", svg)


def paint_background(svg: str) -> str:
    """Insert one light rectangle sized to the viewBox, right after the opening svg tag."""
    viewbox = re.search(r'viewBox="([\d.\s-]+)"', svg)
    if not viewbox:
        return svg
    min_x, min_y, width, height = viewbox.group(1).split()
    rect = (
        f'<rect x="{min_x}" y="{min_y}" width="{width}" height="{height}" fill="#ffffff"/>'
    )
    return re.sub(r"(<svg[^>]*>)", r"\1" + rect, svg, count=1)


def main() -> int:
    source = extract_source(README.read_text())
    mermaid_config = {
        "htmlLabels": False,
        "flowchart": {"htmlLabels": False, "useMaxWidth": False},
        "theme": "neutral",
    }
    puppeteer_config = {"executablePath": CHROME, "args": ["--no-sandbox"]}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "in.mmd").write_text(source)
        (tmp_path / "mermaid.json").write_text(json.dumps(mermaid_config))
        (tmp_path / "puppeteer.json").write_text(json.dumps(puppeteer_config))
        raw_svg = tmp_path / "out.svg"
        subprocess.run(
            [
                str(MMDC),
                "-i", str(tmp_path / "in.mmd"),
                "-o", str(raw_svg),
                "-c", str(tmp_path / "mermaid.json"),
                "-p", str(tmp_path / "puppeteer.json"),
                "-b", "white",
            ],
            check=True,
            cwd=ROOT,
        )
        svg = raw_svg.read_text()

    svg = give_intrinsic_dimensions(svg)
    svg = paint_background(svg)
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUT_SVG.write_text(svg)
    MANIFEST.write_text(
        json.dumps(
            {"source": "README.md mermaid-source fence", "output": str(OUT_SVG.relative_to(ROOT))},
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT_SVG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
