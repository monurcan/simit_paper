"""Renderers for SVG, Mermaid, Vega-Lite, and HTML diagram types.

Each renderer takes the model's raw markup/spec and converts it to a PNG:
  - SVG:       cairosvg (Python library, no subprocess needed)
  - Mermaid:   mmdc CLI (Puppeteer-based, requires --no-sandbox on HPC)
  - Vega-Lite: vl-convert-python (Rust library, no subprocess needed)
  - HTML:      chromium-browser --headless --no-sandbox
"""

import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[2])
import json
import re
import subprocess
import tempfile
from pathlib import Path

# mmdc is installed in a local nodeenv (no system-wide node/npm on this HPC
# environment). The installed mmdc entry point is a symlink to a JS file with
# a `#!/usr/bin/env node` shebang -- invoking that path directly still fails
# for any process whose PATH doesn't include the nodeenv's bin/ (true for
# jobs launched from a plain shell), since `env node` can't resolve. Invoking
# node with the script as an explicit argument sidesteps PATH entirely.
_NODEENV_DIR = Path(_os.environ.get("SIMIT_NODEENV") or SIMIT_ROOT + "/.nodeenv")
_NODE_BIN = _NODEENV_DIR / "bin" / "node"
_MMDC_JS = _NODEENV_DIR / "lib" / "node_modules" / "@mermaid-js" / "mermaid-cli" / "src" / "cli.js"
if _NODE_BIN.exists() and _MMDC_JS.exists():
    MMDC_CMD = [str(_NODE_BIN), str(_MMDC_JS)]
else:
    MMDC_CMD = ["mmdc"]  # fall back to PATH search

# Puppeteer on HPC login nodes requires --no-sandbox (no user namespaces).
_PUPPETEER_CONFIG = '{"args":["--no-sandbox"]}'

# Regex to extract a fenced code block for a given language (or any language).
_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def extract_fenced_block(text: str) -> str | None:
    """Return the content of the first fenced code block found."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

_SVG_SCRIPT_RE = re.compile(r"<script[\s>]", re.IGNORECASE)
_SVG_HANDLER_RE = re.compile(r"\bon\w+\s*=", re.IGNORECASE)


def is_safe_svg(svg_code: str) -> bool:
    return not (_SVG_SCRIPT_RE.search(svg_code) or _SVG_HANDLER_RE.search(svg_code))


def render_svg(svg_code: str, out_path: str, output_size: int = 600) -> None:
    """Render SVG markup to PNG via cairosvg with a white background."""
    import cairosvg
    if not is_safe_svg(svg_code):
        raise ValueError("SVG contains script elements or event handlers")
    cairosvg.svg2png(
        bytestring=svg_code.encode(),
        write_to=out_path,
        output_width=output_size,
        output_height=output_size,
        background_color="white",
    )


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------

def render_mermaid(mermaid_code: str, out_path: str) -> None:
    """Render Mermaid diagram to PNG via mmdc CLI."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_file = tmp_path / "diagram.mmd"
        puppeteer_cfg = tmp_path / "puppeteer.json"
        input_file.write_text(mermaid_code, encoding="utf-8")
        puppeteer_cfg.write_text(_PUPPETEER_CONFIG)
        proc = subprocess.run(
            MMDC_CMD + ["-i", str(input_file), "-o", out_path,
                        "-p", str(puppeteer_cfg), "-q", "-w", "900"],
            capture_output=True, text=True, timeout=60,
        )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        # mmdc wraps the mermaid parse error in a long stack trace;
        # pull out just the first relevant line.
        for line in msg.splitlines():
            if line and not line.startswith("at ") and "Error" in line or "error" in line:
                msg = line
                break
        raise ValueError(msg or "mmdc failed with no error message")


# ---------------------------------------------------------------------------
# Vega-Lite
# ---------------------------------------------------------------------------

def render_vegalite(spec: dict, out_path: str) -> None:
    """Render a Vega-Lite v5 JSON spec to PNG via vl-convert."""
    import vl_convert as vlc
    png_bytes = vlc.vegalite_to_png(vl_spec=spec, scale=2)
    Path(out_path).write_bytes(png_bytes)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

# Any headless-capable Chromium/Chrome binary; override with SIMIT_CHROMIUM=/path/to/chrome.
_CHROMIUM_PATH = _os.environ.get("SIMIT_CHROMIUM", "chromium-browser")
# Script/handler injection safeguard: reject any HTML that could run JS.
_HTML_SCRIPT_RE = re.compile(r"<script[\s>]", re.IGNORECASE)
_HTML_HANDLER_RE = re.compile(r"\bon\w+\s*=", re.IGNORECASE)


def is_safe_html(html_code: str) -> bool:
    return not (_HTML_SCRIPT_RE.search(html_code) or _HTML_HANDLER_RE.search(html_code))


def render_html(html_code: str, out_path: str, width: int = 900, height: int = 700) -> None:
    """Render HTML to PNG using chromium-browser --headless --no-sandbox."""
    import os
    if not is_safe_html(html_code):
        raise ValueError("HTML contains script elements or event handlers")
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w",
                                     encoding="utf-8", delete=False) as f:
        f.write(html_code)
        html_path = f.name
    try:
        # Without explicit --user-data-dir/--disk-cache-dir, Chromium leaks a
        # scoped profile dir under ~/.config/chromium-headless (on any exit
        # other than clean) and a separate disk-cache dir under
        # ~/.cache/chromium-headless (on every run, clean or not, since the
        # cache dir isn't nested under --user-data-dir by default).
        with tempfile.TemporaryDirectory(prefix="chromium-profile-") as profile_dir:
            proc = subprocess.run(
                [_CHROMIUM_PATH, "--headless", "--no-sandbox",
                 f"--user-data-dir={profile_dir}",
                 f"--disk-cache-dir={profile_dir}/cache",
                 f"--screenshot={out_path}",
                 f"--window-size={width},{height}",
                 f"file://{html_path}"],
                capture_output=True, text=True, timeout=30,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"chromium-browser failed: {(proc.stderr or proc.stdout)[:400]}")
        if not Path(out_path).exists():
            raise RuntimeError("chromium-browser exited cleanly but wrote no screenshot")
    finally:
        os.unlink(html_path)
