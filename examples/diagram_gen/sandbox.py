"""Execution sandbox: safety checks and isolated subprocess execution for
model-written matplotlib code (the FIGURE/GEOMETRY pipelines)."""

import re
import subprocess
import sys
from pathlib import Path

from .helpers import GEOMETRY_HELPERS_SRC, TEXT_OVERLAP_FIX_SRC

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


# BAGEL's text output includes the literal Qwen-style chat special tokens
# (e.g. "<|im_end|>") as trailing text rather than stripping them server-side.
# Code-block extraction is naturally bounded by the closing ``` so this never
# bit that path, but unbounded text (like the DSL extraction below) needs an
# explicit strip or the artifact leaks into the last parsed value.
SPECIAL_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>")


def clean_response(text: str) -> str:
    return SPECIAL_TOKEN_RE.sub("", text)


# Refuse to execute generated code that touches anything beyond plotting --
# the model is only ever asked for matplotlib/numpy, so anything else is a
# red flag rather than something we should silently allow.
DENYLIST = (
    "import os", "import sys", "import subprocess", "import socket",
    "import shutil", "__import__", "eval(", "exec(", "open(", "os.", "sys.",
)


def extract_code(text: str) -> str | None:
    match = CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else None


def is_safe(code: str) -> bool:
    return not any(pattern in code for pattern in DENYLIST)


# Occasionally the model's decoding degenerates into mixing CJK characters
# into identifiers/calls (e.g. "ax.plot共同点列表,..."). The resulting Python
# SyntaxError doesn't make the actual cause obvious to a repair prompt, so
# detect it directly and say so -- a specific diagnosis is far more
# actionable for the next attempt than a generic traceback.
CJK_RE = re.compile(r"[一-鿿]")


def find_non_ascii_issue(code: str) -> str | None:
    if CJK_RE.search(code):
        return (
            "The code mixes Chinese characters into Python identifiers/calls "
            "(e.g. `ax.plot共同点列表,...`), which is invalid syntax. Rewrite the "
            "entire script using only English words and standard ASCII "
            "characters for every variable name, function call, and argument."
        )
    return None


# Specific, known matplotlib API pitfalls the model keeps reproducing even
# after seeing the corrected version once -- caught statically (regardless of
# whether this particular attempt happens to trigger the runtime error) so
# the repair message is exact rather than relying on the bare traceback.
def find_known_api_issues(code: str) -> str | None:
    if "boxplot(" in code and "tick_labels" not in code and re.search(r"\blabels\s*=", code):
        return (
            "ax.boxplot's label argument is called tick_labels in this matplotlib "
            "version, not labels. Replace labels= with tick_labels= in the boxplot() call."
        )
    if re.search(r"\bArc\(", code) and re.search(r"\b(angle1|angle2|start_angle|end_angle)\s*=", code):
        return (
            "matplotlib's Arc patch takes theta1 and theta2 for its sweep angles -- "
            "angle1/angle2/start_angle/end_angle are not real Arc parameters. Either fix the "
            "Arc(...) call to use theta1=/theta2=, or better, use the pre-defined "
            "angle_arc(ax, vertex, theta1, theta2, radius=..., label=...) helper instead of "
            "constructing Arc directly."
        )
    return None


MAX_DIM = 2000  # pixels -- tightened from 3000 after a real case slipped through: a


# parallelogram with vertices spanning [0,5]x[0,3] rendered at 2963x1962px because
# set_xlim/set_ylim were never called, leaving autoscale to expand to a degenerate
# view. No legitimate output observed in this pipeline has approached 2000px on
# either side, so this stays a meaningful backstop without false-positiving on
# genuinely complex content.
# Legitimately oblong content (a number line, a timeline) can reach ~7:1 once
# bbox_inches="tight" crops to the actual drawn extent -- verified against a
# correct, correctly-bounded number-line render that hit 959x131px (7.3:1).
# The genuine "stray artist" failure mode this guards against instead blows
# one side up far past MAX_DIM, so it's already caught by that check above;
# this ratio only needs to catch a blow-up that stays just under MAX_DIM.
MAX_ASPECT_RATIO = 9.0  # longest side / shortest side


def run_script(code: str, workdir: Path) -> tuple[bool, str]:
    script_path = workdir / "script.py"
    script_path.write_text(GEOMETRY_HELPERS_SRC + "\n" + TEXT_OVERLAP_FIX_SRC + "\n" + code)
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=workdir, capture_output=True, text=True, timeout=20,
            env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        return False, "Execution timed out after 20s"
    if proc.returncode != 0:
        return False, proc.stderr[-2000:]
    png_path = workdir / "output.png"
    if not png_path.exists():
        return False, "Script exited cleanly but did not produce output.png"

    # A pathologically huge or sliver-shaped image means the figure's axis
    # limits/data extents were inconsistent (e.g. ax.add_patch without
    # explicit set_xlim/set_ylim) -- catch it here rather than silently
    # accepting an unusable render.
    from PIL import Image
    width, height = Image.open(png_path).size
    if width > MAX_DIM or height > MAX_DIM or max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        return False, (
            f"Rendered image has pathological dimensions {width}x{height}px. This usually "
            "means a shape was added with ax.add_patch(...) without also calling "
            "ax.set_xlim(...)/ax.set_ylim(...) explicitly, or bbox_inches='tight' picked up "
            "an artist placed far outside the intended view. Fix the axis limits."
        )
    return True, ""
