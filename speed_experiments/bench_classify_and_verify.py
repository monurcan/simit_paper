"""Measures the REAL classify_request-style call (CLASSIFY_PROMPT, ~19k
chars, only 20 output tokens -- almost pure prefill, the step where caching
should matter most) and a verify()-style call (short _CRITIC_PROMPT, 300
output tokens), to properly weight caching benefit across pipeline stages
instead of assuming the triplet-generation step represents everything.
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
from verify import _CRITIC_PROMPT

IMAGE_PATH = SIMIT_ROOT + "/Bagel/test_images/women.jpg"
CLASSIFY_PROMPT = skill_prompts.CLASSIFY_PROMPT
N = 5

REQUESTS = [
    "a bar chart of monthly rainfall",
    "a labeled diagram of a right triangle with sides 3, 4, 5",
    "a circuit diagram with a battery, resistor, and switch in series",
    "a photo of a golden retriever running on a beach",
    "a Venn diagram showing overlap between set A and set B",
]


def load_image_bytes():
    img = Image.open(IMAGE_PATH).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    client = wait_ready()

    print(f"\n=== classify_request-style: real CLASSIFY_PROMPT ({len(CLASSIFY_PROMPT)} chars), max_output_tokens=20 ===")
    classify_latencies = []
    for r in REQUESTS:
        t0 = time.perf_counter()
        client.chat(CLASSIFY_PROMPT.format(request=r), think_mode=False, max_output_tokens=20, temperature=0.0)
        dt = time.perf_counter() - t0
        classify_latencies.append(dt)
        print(f"  '{r[:40]}...' took {dt:.3f}s")

    print(f"\n=== classify floor: SAME 20-token output, NO big prefix (just the request text) ===")
    classify_floor_latencies = []
    for r in REQUESTS:
        t0 = time.perf_counter()
        client.chat(f"Classify this in one word: {r}", think_mode=False, max_output_tokens=20, temperature=0.0)
        dt = time.perf_counter() - t0
        classify_floor_latencies.append(dt)
        print(f"  '{r[:40]}...' took {dt:.3f}s")

    print(f"\n=== verify()-style: real _CRITIC_PROMPT ({len(_CRITIC_PROMPT)} chars) + image, max_output_tokens=300 ===")
    image_bytes = load_image_bytes()
    verify_latencies = []
    for r in REQUESTS:
        prompt = _CRITIC_PROMPT.format(request=r)
        t0 = time.perf_counter()
        client.chat(prompt, images=image_bytes, think_mode=True, max_output_tokens=300, temperature=0.0)
        dt = time.perf_counter() - t0
        verify_latencies.append(dt)
        print(f"  '{r[:40]}...' took {dt:.3f}s")

    result = {
        "classify_real_prompt_s": classify_latencies,
        "classify_no_prefix_floor_s": classify_floor_latencies,
        "classify_avg_real": sum(classify_latencies) / N,
        "classify_avg_floor": sum(classify_floor_latencies) / N,
        "classify_prefix_overhead_s": sum(classify_latencies) / N - sum(classify_floor_latencies) / N,
        "verify_latencies_s": verify_latencies,
        "verify_avg_s": sum(verify_latencies) / N,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/bench_classify_and_verify_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
