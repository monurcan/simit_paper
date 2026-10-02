"""Measures whether the existing mstar server reuses KV cache across
independent chat() calls that share a long prompt prefix -- the "skill
prompt" pattern (e.g. Bagel/../examples/diagram_gen/prompts.py's
MASTER_PROMPT, ~21k chars, reused verbatim for every query in a diagram
category). If prefix caching is active, later calls sharing that long
prefix should get cheaper prompt-processing time than calls with an
equally-long but never-before-seen prefix.

Does NOT touch the existing pipeline code -- standalone server (port 8010).
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import io
import json
import sys
import time

sys.path.insert(0, SIMIT_ROOT)
sys.path.insert(0, SIMIT_ROOT + "/examples/diagram_gen")
from PIL import Image
from wait_ready import wait_ready
import prompts as skill_prompts

N = 6
MAX_OUT_TOKENS = 60  # keep output short: isolate prompt-processing (prefill) time, not decode time
IMAGE_PATH = SIMIT_ROOT + "/Bagel/test_images/women.jpg"

SHARED_SKILL_PREFIX = skill_prompts.MASTER_PROMPT  # ~21k chars, real skill prompt from the pipeline
SUFFIXES = [
    "\n\nNow apply the above to this request: draw a bar chart of monthly rainfall.",
    "\n\nNow apply the above to this request: draw a scatter plot of height vs weight.",
    "\n\nNow apply the above to this request: draw a pie chart of budget allocation.",
    "\n\nNow apply the above to this request: draw a line chart of stock price over a year.",
    "\n\nNow apply the above to this request: draw a histogram of exam scores.",
    "\n\nNow apply the above to this request: draw a box plot comparing three groups.",
]
assert len(SUFFIXES) == N

# Control: N "fresh" prefixes of matched length (~21k chars) built by
# repeating a filler paragraph with a unique seed per call, so the model
# has never seen this exact prefix before -- isolates prefix-*reuse*
# specifically, not just "long prompt" effects.
FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank "
          "while the weather remains unpredictable this time of year. ")


def fresh_prefix(seed: int) -> str:
    text = f"[unique session {seed}] " + FILLER * (len(SHARED_SKILL_PREFIX) // len(FILLER) + 1)
    return text[: len(SHARED_SKILL_PREFIX)]


def load_image_bytes():
    img = Image.open(IMAGE_PATH).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def one_call(client, prompt, image_bytes):
    t0 = time.perf_counter()
    client.chat(prompt, images=image_bytes, think_mode=False, max_output_tokens=MAX_OUT_TOKENS, temperature=0.2)
    return time.perf_counter() - t0


def main():
    client = wait_ready()
    image_bytes = load_image_bytes()

    print(f"\n=== Shared long skill-prompt prefix, {N} calls (same {len(SHARED_SKILL_PREFIX)}-char prefix each time) ===")
    shared_latencies = []
    for i, suffix in enumerate(SUFFIXES):
        dt = one_call(client, SHARED_SKILL_PREFIX + suffix, image_bytes)
        shared_latencies.append(dt)
        print(f"  call {i} (shared prefix) took {dt:.2f}s")

    print(f"\n=== Control: {N} calls, each with a NEW never-seen prefix of matched length ===")
    fresh_latencies = []
    for i in range(N):
        prompt = fresh_prefix(seed=1000 + i) + SUFFIXES[i]
        dt = one_call(client, prompt, image_bytes)
        fresh_latencies.append(dt)
        print(f"  call {i} (fresh prefix) took {dt:.2f}s")

    result = {
        "n": N,
        "prefix_char_len": len(SHARED_SKILL_PREFIX),
        "max_output_tokens": MAX_OUT_TOKENS,
        "shared_prefix_latencies_s": shared_latencies,
        "fresh_prefix_latencies_s": fresh_latencies,
        "shared_prefix_call0_vs_rest": {
            "call0_s": shared_latencies[0],
            "avg_calls_1plus_s": sum(shared_latencies[1:]) / (N - 1),
        },
        "avg_shared_s": sum(shared_latencies) / N,
        "avg_fresh_s": sum(fresh_latencies) / N,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/bench_prefix_cache_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
