"""The real production logs show image-generation (Bagel's diffusion-based
natural-image head) dominates total synthesis time (~84% for ai2d), not any
text LLM call. Tests whether THIS step batches the same way text chat()
calls did (~1.95x measured earlier), since that would be the far bigger
lever for a "more optimal implementation" than any text-side caching.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, SIMIT_ROOT)
from wait_ready import wait_ready

N = 5
PROMPTS = [
    "a golden retriever running on a beach at sunset",
    "a red apple sitting on a wooden table",
    "a mountain landscape with snow-capped peaks",
    "a city street at night with neon signs",
    "a cup of coffee on a marble countertop",
]


def one_call(client, prompt):
    t0 = time.perf_counter()
    img_bytes = client.generate_image(prompt)
    return time.perf_counter() - t0, len(img_bytes)


def main():
    client = wait_ready()

    print(f"\n=== Sequential: {N} image-gen calls, one at a time ===")
    seq_latencies = []
    t_seq0 = time.perf_counter()
    for p in PROMPTS:
        dt, n_bytes = one_call(client, p)
        seq_latencies.append(dt)
        print(f"  took {dt:.2f}s, {n_bytes} bytes")
    seq_total = time.perf_counter() - t_seq0

    print(f"\n=== Concurrent: {N} image-gen calls issued at once ===")
    conc_latencies = []
    t_conc0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N) as ex:
        futures = [ex.submit(one_call, client, p) for p in PROMPTS]
        for f in futures:
            dt, n_bytes = f.result()
            conc_latencies.append(dt)
            print(f"  took {dt:.2f}s, {n_bytes} bytes")
    conc_total = time.perf_counter() - t_conc0

    result = {
        "sequential_total_s": seq_total,
        "sequential_latencies_s": seq_latencies,
        "concurrent_total_s": conc_total,
        "concurrent_latencies_s": conc_latencies,
        "speedup": seq_total / conc_total,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/bench_image_gen_batching_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
