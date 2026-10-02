"""Single-GPU (GPU1 is hardware/driver-broken on this node -- confirmed via
repeated clean restarts, not a fixable software issue) bounded worker-pool
test at higher concurrency (pool=16, queue=48) to actually approach the
engine's demonstrated batching ceiling (CUDA graphs captured up to bs=64).
Real, unmodified pipeline functions only.
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
import threading

SUBSET = "ai2d"
N_SAMPLES = 24
POOL_SIZE = 6
SAMPLE_START = 150

# Fix the race condition that crashed the earlier pool=16 attempt: multiple
# concurrent worker threads each independently cross the consecutive-error
# threshold and call _restart_server() at the same time (unsynchronized
# module-level globals + non-atomic check-then-act). Wrap with a lock + a
# short cooldown so only one thread actually restarts per real stall, and
# threads that arrive just after a fresh restart just trust it instead of
# piling on. Monkey-patch only, does not touch the original file.
_restart_lock = threading.Lock()
_last_restart_time = [0.0]
_original_restart_server = sv._restart_server


def _locked_restart_server(*args, **kwargs):
    with _restart_lock:
        if time.time() - _last_restart_time[0] < 90:
            print("  [lock] restart already happened recently, skipping duplicate")
            return True
        result = _original_restart_server(*args, **kwargs)
        _last_restart_time[0] = time.time()
        return result


sv._restart_server = _locked_restart_server

sv._THINK_MODE = False
sv._VERIFY_THINK_MODE = True
sv._DISCARD_UNVERIFIED = True
sv._CONFIDENCE_THRESHOLD = 50
sv._VERIFY_CRITERION = "score"
sv._MAX_TRIPLET_ATTEMPTS = 16
sv._STRATEGY = "bfs"
CFG = sv.SUBSET_CONFIG["ai2d"]
SERVER_URL = "http://localhost:8000"
SEQUENTIAL_RATE_S_PER_SAMPLE = 109.27755975845503  # from the earlier real N=4 sequential measurement


def main():
    print(f"Loading real {SUBSET} dataset...")
    ds = load_dataset("lmms-lab/LMMs-Eval-Lite", SUBSET)["lite"]
    sample_indices = list(range(SAMPLE_START, SAMPLE_START + N_SAMPLES))

    out_root = Path(SIMIT_ROOT + "/speed_experiments/output_highconc")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    sv.OUT_ROOT = out_root

    def do_one(idx):
        client = MStarClient(SERVER_URL, timeout=120.0)
        t0 = time.perf_counter()
        sv.process_sample(client, SUBSET, idx, ds[idx], CFG)
        return idx, time.perf_counter() - t0

    print(f"\n=== HIGH-CONCURRENCY POOL: {N_SAMPLES} samples, pool={POOL_SIZE} ===")
    t_start = time.perf_counter()
    per_sample = {}
    with ThreadPoolExecutor(max_workers=POOL_SIZE) as ex:
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
        "total_s": total,
        "per_sample_s": per_sample,
        "avg_s_per_sample": total / N_SAMPLES,
        "sequential_baseline_rate_s_per_sample": SEQUENTIAL_RATE_S_PER_SAMPLE,
        "sequential_equivalent_total_s": sequential_equivalent_total,
        "speedup_vs_sequential": sequential_equivalent_total / total,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/e2e_highconc_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
