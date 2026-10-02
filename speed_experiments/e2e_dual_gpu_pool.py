"""Bigger, more realistic test than the earlier N=4/N=12 "launch everything
at once" runs: a BOUNDED worker pool (not unlimited concurrency) pulling from
a queue of real samples -- this is what an actual production implementation
would look like (a fixed number of concurrent slots continuously fed from a
queue, so one slow retry-heavy sample doesn't stall the whole batch) -- run
independently on BOTH GPUs (two separate mstar servers, one per GPU, real
process-level parallelism) to use all available hardware.

Real, unmodified pipeline functions only (process_sample etc., imported from
synthesize_vqa.py). Own OUT_ROOT under speed_experiments/, never touches the
real generated_samples/ directory.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, SIMIT_ROOT)
sys.path.insert(0, SIMIT_ROOT + "/similar_samples_llms_eval_lite")
import synthesize_vqa as sv
from mstar import MStarClient
from datasets import load_dataset

SUBSET = "ai2d"
N_PER_GPU = 24
POOL_SIZE_PER_GPU = 12  # bounded -- well under the engine's demonstrated 64-batch ceiling, gives real queueing since pool < N
SAMPLE_START = 200  # fresh range, avoids reusing indices from earlier smaller tests

sv._THINK_MODE = False
sv._VERIFY_THINK_MODE = True
sv._DISCARD_UNVERIFIED = True
sv._CONFIDENCE_THRESHOLD = 50
sv._VERIFY_CRITERION = "score"
sv._MAX_TRIPLET_ATTEMPTS = 16
sv._STRATEGY = "bfs"
CFG = sv.SUBSET_CONFIG["ai2d"]

GPU0_URL = "http://localhost:8000"
GPU1_URL = "http://localhost:8001"


def wait_health(url, timeout_s=600):
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def run_pool(label, server_url, out_root, sample_indices, ds, pool_size):
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    sv.OUT_ROOT = out_root

    def do_one(idx):
        client = MStarClient(server_url, timeout=120.0)
        t0 = time.perf_counter()
        sv.process_sample(client, SUBSET, idx, ds[idx], CFG)
        return idx, time.perf_counter() - t0

    print(f"\n=== {label}: {len(sample_indices)} samples, pool={pool_size}, server={server_url} ===")
    t_start = time.perf_counter()
    per_sample = {}
    with ThreadPoolExecutor(max_workers=pool_size) as ex:
        futures = {ex.submit(do_one, idx): idx for idx in sample_indices}
        for f in as_completed(futures):
            idx, dt = f.result()
            per_sample[idx] = dt
            print(f"  [{label}] sample{idx} done in {dt:.1f}s ({len(per_sample)}/{len(sample_indices)})")
    total = time.perf_counter() - t_start
    print(f"  {label} TOTAL: {total:.1f}s for {len(sample_indices)} samples ({total/len(sample_indices):.1f}s/sample avg)")
    return total, per_sample


def main():
    print("Checking both GPU servers are healthy...")
    ok0 = wait_health(GPU0_URL)
    ok1 = wait_health(GPU1_URL)
    print(f"GPU0 (port 8000) healthy: {ok0}, GPU1 (port 8001) healthy: {ok1}")
    if not (ok0 and ok1):
        print("ABORT: not both servers healthy")
        return

    # Confirm GPU1 server's model is actually loaded (health != warm) via a real warmup chat.
    print("Warming up GPU1 server (chat, may block if still loading weights)...")
    c1 = MStarClient(GPU1_URL, timeout=600.0)
    t0 = time.perf_counter()
    c1.chat("Hi.", think_mode=False, max_output_tokens=5)
    print(f"  GPU1 warmup chat done in {time.perf_counter()-t0:.1f}s")

    print(f"Loading real {SUBSET} dataset...")
    ds = load_dataset("lmms-lab/LMMs-Eval-Lite", SUBSET)["lite"]

    gpu0_indices = list(range(SAMPLE_START, SAMPLE_START + N_PER_GPU))
    gpu1_indices = list(range(SAMPLE_START + N_PER_GPU, SAMPLE_START + 2 * N_PER_GPU))

    # Run both GPU pools CONCURRENTLY (real process-level parallelism across
    # the two independent servers) via top-level threads, each internally
    # running its own bounded worker pool.
    results = {}
    with ThreadPoolExecutor(max_workers=2) as top_ex:
        f0 = top_ex.submit(run_pool, "GPU0", GPU0_URL, Path(SIMIT_ROOT + "/speed_experiments/output_gpu0"), gpu0_indices, ds, POOL_SIZE_PER_GPU)
        f1 = top_ex.submit(run_pool, "GPU1", GPU1_URL, Path(SIMIT_ROOT + "/speed_experiments/output_gpu1"), gpu1_indices, ds, POOL_SIZE_PER_GPU)
        gpu0_total, gpu0_per_sample = f0.result()
        gpu1_total, gpu1_per_sample = f1.result()

    wall_clock_total = max(gpu0_total, gpu1_total)  # both ran concurrently
    n_total = N_PER_GPU * 2

    # Sequential-equivalent baseline: reuse the already-measured real
    # sequential rate (109.3s/sample avg from the earlier N=4 original run)
    # rather than re-running a long sequential baseline at this N.
    SEQUENTIAL_RATE_S_PER_SAMPLE = 109.27755975845503
    sequential_equivalent_total = SEQUENTIAL_RATE_S_PER_SAMPLE * n_total

    result = {
        "n_per_gpu": N_PER_GPU,
        "pool_size_per_gpu": POOL_SIZE_PER_GPU,
        "gpu0_total_s": gpu0_total,
        "gpu0_per_sample_s": gpu0_per_sample,
        "gpu1_total_s": gpu1_total,
        "gpu1_per_sample_s": gpu1_per_sample,
        "wall_clock_total_s": wall_clock_total,
        "n_total_samples": n_total,
        "wall_clock_avg_s_per_sample": wall_clock_total / n_total,
        "sequential_baseline_rate_s_per_sample": SEQUENTIAL_RATE_S_PER_SAMPLE,
        "sequential_equivalent_total_s": sequential_equivalent_total,
        "speedup_vs_sequential": sequential_equivalent_total / wall_clock_total,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/e2e_dual_gpu_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
