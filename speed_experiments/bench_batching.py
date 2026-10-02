"""Measures the real speedup available from issuing concurrent requests to
the existing mstar server instead of the synthesis pipeline's current
sequential-one-at-a-time pattern (similar_samples_llms_eval_lite/
synthesize_vqa.py's `for idx in range(...): process_sample(...)` loop).

Does NOT touch the existing pipeline code -- talks to a standalone server
(port 8010) via the same MStarClient interface, just called differently.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, SIMIT_ROOT)
from PIL import Image
from wait_ready import wait_ready

N = 6
MAX_OUT_TOKENS = 300
IMAGE_PATH = SIMIT_ROOT + "/Bagel/test_images/women.jpg"

# N distinct prompts (no shared prefix) -- isolates batching benefit from
# any prefix-caching confound (that's bench_prefix_cache.py's job).
PROMPTS = [
    "Describe this image in detail, focusing on the people present.",
    "What is the dominant color palette in this image? Explain why.",
    "Write three plausible questions a visitor might ask about this image.",
    "Estimate the setting/location of this image and justify your reasoning.",
    "List five distinct objects visible in this image with their approximate positions.",
    "Write a short caption for this image suitable for a photo archive.",
]
assert len(PROMPTS) == N


def load_image_bytes():
    img = Image.open(IMAGE_PATH).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def one_call(client, prompt, image_bytes):
    t0 = time.perf_counter()
    resp = client.chat(prompt, images=image_bytes, think_mode=False, max_output_tokens=MAX_OUT_TOKENS, temperature=0.7)
    dt = time.perf_counter() - t0
    return dt, len(resp.text)


def main():
    client = wait_ready()
    image_bytes = load_image_bytes()

    print(f"\n=== Sequential: {N} calls, one at a time (current pipeline's pattern) ===")
    seq_latencies = []
    t_seq_start = time.perf_counter()
    for p in PROMPTS:
        dt, tlen = one_call(client, p, image_bytes)
        seq_latencies.append(dt)
        print(f"  call took {dt:.2f}s, response {tlen} chars")
    seq_total = time.perf_counter() - t_seq_start

    print(f"\n=== Concurrent: {N} calls issued at once (ThreadPoolExecutor) ===")
    conc_latencies = []
    t_conc_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N) as ex:
        futures = [ex.submit(one_call, client, p, image_bytes) for p in PROMPTS]
        for f in futures:
            dt, tlen = f.result()
            conc_latencies.append(dt)
            print(f"  call took {dt:.2f}s, response {tlen} chars")
    conc_total = time.perf_counter() - t_conc_start

    result = {
        "n": N,
        "max_output_tokens": MAX_OUT_TOKENS,
        "sequential_total_s": seq_total,
        "sequential_per_call_latencies_s": seq_latencies,
        "concurrent_total_s": conc_total,
        "concurrent_per_call_latencies_s": conc_latencies,
        "speedup_total_wallclock": seq_total / conc_total,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/bench_batching_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
