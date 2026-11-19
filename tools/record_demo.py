"""Record a replay of the headline commands as a GIF, paced by their measured wall time.

This is a replay, not a screen recording, and the README says so. Each command is run once for real,
its stdout, exit code and duration captured, and then a self-contained HTML terminal types the
command
and reveals the captured output paced by that command's measured duration. Playwright records the
page
to webm and ffmpeg transcodes to a GIF and an MP4. Every line of terminal text is real captured
output,
never hand written, and every command with its exit code and duration is written to a manifest.

The segments are ordered as the argument, not the feature list: the naive pass first, then the
gate's
inconclusive verdict on the same scenario, then the sizing that explains it, then a clear
regression,
then the receipts. Dev-time tool, needs a browser and ffmpeg, skipped in the pure-Python CI job.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "docs" / "video"
CHROME = "/usr/bin/google-chrome"

SEGMENTS = [
    ("ragscore gate", ["ragscore", "gate"]),
    ("ragscore power", ["ragscore", "power"]),
    ("ragscore gate --blend 0.12", ["ragscore", "gate", "--blend", "0.12"]),
    ("python experiments/exp01_calibration.py", ["python", "experiments/exp01_calibration.py"]),
]


def capture_segments() -> list[dict]:
    """Run each command once, capturing stdout, exit code and wall time."""
    # Warm the import so the first segment is not charged for interpreter start-up.
    subprocess.run(["ragscore", "power"], cwd=ROOT, capture_output=True, text=True)
    captured = []
    for label, argv in SEGMENTS:
        start = time.perf_counter()
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        duration = time.perf_counter() - start
        captured.append(
            {
                "label": label,
                "exit_code": result.returncode,
                "duration_s": round(duration, 3),
                "stdout": result.stdout.rstrip("\n"),
            }
        )
    return captured


def build_player(segments: list[dict]) -> str:
    data = json.dumps(segments)
    return """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    body{background:#0c0c0c;margin:0;}
    #t{font-family:ui-monospace,Menlo,monospace;font-size:15px;line-height:1.5;color:#e6e6e6;
       padding:18px;white-space:pre-wrap;}
    .p{color:#5fb3ff;} .c{color:#ffffff;} .o{color:#b8b8b8;}
    </style></head><body><div id="t"></div><script>
    const segs = __DATA__;
    const t = document.getElementById('t');
    function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
    async function typeCmd(cmd){
      t.innerHTML += '<span class="p">$ </span>';
      for(const ch of cmd){ t.innerHTML += '<span class="c">'+ch.replace('<','&lt;')+'</span>';
        await sleep(28);}
      t.innerHTML += '\\n';
    }
    async function run(){
      for(const s of segs){
        await typeCmd(s.label);
        const pace = Math.max(300, Math.min(2200, s.duration_s*1000));
        await sleep(pace);
        t.innerHTML += '<span class="o">'+s.stdout.replace(/</g,'&lt;')+'</span>\\n\\n';
        window.scrollTo(0, document.body.scrollHeight);
        await sleep(650);
      }
      document.title = 'done';
    }
    run();
    </script></body></html>""".replace("__DATA__", data)


def main() -> int:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to build the demo")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    segments = capture_segments()
    player = VIDEO_DIR / "_player.html"
    player.write_text(build_player(segments))

    webm_dir = VIDEO_DIR / "_rec"
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 900, "height": 620},
            record_video_dir=str(webm_dir),
            record_video_size={"width": 900, "height": 620},
        )
        page = context.new_page()
        page.goto(player.as_uri())
        total_ms = sum(max(300, min(2200, s["duration_s"] * 1000)) + 650 for s in segments)
        page.wait_for_function("() => document.title === 'done'", timeout=int(total_ms) + 15000)
        page.wait_for_timeout(600)
        context.close()
        browser.close()

    webm = next(webm_dir.glob("*.webm"))
    mp4 = VIDEO_DIR / "demo.mp4"
    gif = VIDEO_DIR / "demo.gif"
    palette = webm_dir / "palette.png"
    subprocess.run(["ffmpeg", "-y", "-i", str(webm), "-movflags", "+faststart",
                    "-pix_fmt", "yuv420p", str(mp4)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(webm), "-vf",
                    "fps=12,scale=900:-1:flags=lanczos,palettegen", str(palette)],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(webm), "-i", str(palette), "-lavfi",
                    "fps=12,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse", str(gif)],
                   check=True, capture_output=True)

    (VIDEO_DIR / "manifest.json").write_text(
        json.dumps({"segments": segments, "gif": "demo.gif", "mp4": "demo.mp4"}, indent=2) + "\n"
    )
    shutil.rmtree(webm_dir, ignore_errors=True)
    player.unlink(missing_ok=True)
    print(f"wrote {gif.relative_to(ROOT)} ({gif.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
