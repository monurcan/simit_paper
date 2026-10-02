"""Establishes the theoretical floor: what does the SAME suffix alone cost
with NO ~21k-char skill-prompt prefix at all? If prefix caching were truly
skipping recomputation of a shared prefix, repeat-prefix calls should
approach this floor. If they don't, caching isn't meaningfully active.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import io
import json
import sys
import time

sys.path.insert(0, SIMIT_ROOT)
from PIL import Image
from wait_ready import wait_ready

N = 6
MAX_OUT_TOKENS = 60
IMAGE_PATH = SIMIT_ROOT + "/Bagel/test_images/women.jpg"
SUFFIXES = [
    "Draw a bar chart of monthly rainfall.",
    "Draw a scatter plot of height vs weight.",
    "Draw a pie chart of budget allocation.",
    "Draw a line chart of stock price over a year.",
    "Draw a histogram of exam scores.",
    "Draw a box plot comparing three groups.",
]


def load_image_bytes():
    img = Image.open(IMAGE_PATH).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    client = wait_ready()
    image_bytes = load_image_bytes()
    print(f"\n=== No-prefix baseline: {N} calls, short prompt only (no ~21k-char prefix) ===")
    latencies = []
    for s in SUFFIXES:
        t0 = time.perf_counter()
        client.chat(s, images=image_bytes, think_mode=False, max_output_tokens=MAX_OUT_TOKENS, temperature=0.2)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        print(f"  call took {dt:.2f}s")
    result = {"latencies_s": latencies, "avg_s": sum(latencies) / N}
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/bench_no_prefix_baseline_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
