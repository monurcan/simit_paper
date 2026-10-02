"""Same real pipeline, same bounded-pool idea as e2e_single_gpu_high_concurrency.py,
but using PROCESSES instead of threads. ThreadPoolExecutor only gives true
concurrency for I/O-bound work (waiting on HTTP requests to the mstar
server) -- any CPU-bound work in the pipeline (renderers.py's matplotlib
renderers run in-process, no subprocess isolation, unlike sandbox.py /
renderers_web.py / renderers_latex.py which DO use subprocess.run) would
serialize across threads under the GIL regardless of pool size. This would
exactly explain why pool=6/10/16 all converged to ~1.76x with threads.
Processes bypass the GIL entirely for that CPU-bound portion.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, SIMIT_ROOT)
sys.path.insert(0, SIMIT_ROOT + "/similar_samples_llms_eval_lite")

SUBSET = "ai2d"
N_SAMPLES = 24
POOL_SIZE = 6
SAMPLE_START = 100
SERVER_URL = "http://localhost:8000"
SEQUENTIAL_RATE_S_PER_SAMPLE = 109.27755975845503


def _configure_sv():
    """Each child process imports synthesize_vqa fresh and sets the same
    production config -- must be called inside the worker, not just the
    parent, since each process gets its own module state."""
    import synthesize_vqa as sv
    sv._THINK_MODE = False
    sv._VERIFY_THINK_MODE = True
    sv._DISCARD_UNVERIFIED = True
    sv._CONFIDENCE_THRESHOLD = 50
    sv._VERIFY_CRITERION = "score"
    sv._MAX_TRIPLET_ATTEMPTS = 16
    sv._STRATEGY = "bfs"
    sv.OUT_ROOT = Path(SIMIT_ROOT + "/speed_experiments/output_procpool")
    return sv


def do_one(idx):
    from mstar import MStarClient
    from datasets import load_dataset
    sv = _configure_sv()
    ds = load_dataset("lmms-lab/LMMs-Eval-Lite", SUBSET)["lite"]
    cfg = sv.SUBSET_CONFIG["ai2d"]
    client = MStarClient(SERVER_URL, timeout=120.0)
    t0 = time.perf_counter()
    sv.process_sample(client, SUBSET, idx, ds[idx], cfg)
    return idx, time.perf_counter() - t0


def main():
    out_root = Path(SIMIT_ROOT + "/speed_experiments/output_procpool")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    sample_indices = list(range(SAMPLE_START, SAMPLE_START + N_SAMPLES))
    print(f"\n=== PROCESS POOL: {N_SAMPLES} samples, pool={POOL_SIZE} (bypasses GIL) ===")
    t_start = time.perf_counter()
    per_sample = {}
    with ProcessPoolExecutor(max_workers=POOL_SIZE) as ex:
        futures = {ex.submit(do_one, idx): idx for idx in sample_indices}
        for f in as_completed(futures):
            idx, dt = f.result()
            per_sample[idx] = dt
            print(f"  sample{idx} done in {dt:.1f}s ({len(per_sample)}/{N_SAMPLES})")
    total = time.perf_counter() - t_start

    sequential_equivalent_total = SEQUENTIAL_RATE_S_PER_SAMPLE * N_SAMPLES
    result = {
        "n_samples": N_SAMPLES,
        "pool_size": POOL_SIZE,
        "mode": "process",
        "total_s": total,
        "per_sample_s": per_sample,
        "avg_s_per_sample": total / N_SAMPLES,
        "sequential_baseline_rate_s_per_sample": SEQUENTIAL_RATE_S_PER_SAMPLE,
        "sequential_equivalent_total_s": sequential_equivalent_total,
        "speedup_vs_sequential": sequential_equivalent_total / total,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/e2e_procpool_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
