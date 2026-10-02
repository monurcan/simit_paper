"""Real end-to-end comparison: the ACTUAL synthesis pipeline's real
functions (process_sample, generate_synthetic_examples, generate_visualization,
classify_request, verify, generate_with_verification -- all imported, none
copy-pasted or modified) run on a small slice of real ai2d data, once
"original" (sequential, matching synthesize_vqa.py's real for-loop) and once
"optimized" (same functions, concurrent per-sample dispatch via a thread
pool -- the only change is HOW they're called, not the pipeline logic).

Does not touch the existing codebase: writes to its own OUT_ROOT overrides
under speed_experiments/, never to similar_samples_llms_eval_lite/generated_samples/.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, SIMIT_ROOT)
sys.path.insert(0, SIMIT_ROOT + "/similar_samples_llms_eval_lite")
import synthesize_vqa as sv
from datasets import load_dataset

N_SAMPLES = 12
PRIME_IDX = 450  # throwaway index for warmup, outside the 0..N_SAMPLES-1 range used for real timing (ai2d lite has 500 samples, 0-499)
SUBSET = "ai2d"

# Match the real production discard4_ai2d job's config exactly (from its own
# log header: "K=4 think_mode=False verify_think_mode=True verify_all=True
# discard_unverified=True confidence_threshold=50 verify_criterion=score
# max_triplet_attempts=16").
sv._THINK_MODE = False
sv._VERIFY_THINK_MODE = True
sv._DISCARD_UNVERIFIED = True
sv._CONFIDENCE_THRESHOLD = 50
sv._VERIFY_CRITERION = "score"
sv._MAX_TRIPLET_ATTEMPTS = 16
sv._STRATEGY = "bfs"

CFG = sv.SUBSET_CONFIG["ai2d"]


def run_condition(label: str, out_root: Path, sample_indices: list[int], ds, concurrent: bool):
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    sv.OUT_ROOT = out_root  # redirect module global -- own dir, never touches real generated_samples/

    def do_one(idx):
        t0 = time.perf_counter()
        client = sv._make_client()
        sv.process_sample(client, SUBSET, idx, ds[idx], CFG)
        return idx, time.perf_counter() - t0

    print(f"\n=== {label}: {len(sample_indices)} samples, concurrent={concurrent} ===")
    t_start = time.perf_counter()
    per_sample = {}
    if concurrent:
        with ThreadPoolExecutor(max_workers=len(sample_indices)) as ex:
            futures = [ex.submit(do_one, idx) for idx in sample_indices]
            for f in futures:
                idx, dt = f.result()
                per_sample[idx] = dt
                print(f"  sample{idx} done in {dt:.1f}s")
    else:
        for idx in sample_indices:
            idx, dt = do_one(idx)
            per_sample[idx] = dt
            print(f"  sample{idx} done in {dt:.1f}s")
    total = time.perf_counter() - t_start
    print(f"  {label} TOTAL: {total:.1f}s for {len(sample_indices)} samples ({total/len(sample_indices):.1f}s/sample avg)")
    return total, per_sample


def main():
    # Use the pipeline's OWN real startup function (not a separate server on
    # a different port) -- avoids the port-8010-vs-8000 GPU-memory conflict
    # that OOM'd a second server instance and caused a 57-minute stall.
    print("Ensuring the real pipeline's server (port 8000) is up via sv._ensure_server_running()...")
    sv._ensure_server_running()

    print(f"Loading real {SUBSET} dataset...")
    ds = load_dataset("lmms-lab/LMMs-Eval-Lite", SUBSET)["lite"]
    sample_indices = list(range(N_SAMPLES))

    # Priming pass: one throwaway real sample through the full real pipeline
    # (triplet-gen + classify + render/image-gen + verify), discarded, so any
    # first-encounter CUDA-graph-capture/JIT cost for these specific
    # prompt/image shapes lands here instead of skewing either official
    # condition -- at production N (500/benchmark) this one-time cost is
    # negligible anyway, so excluding it from the N=12 measurement is the
    # fairer analog of the real full-scale regime, per the user's request.
    print("\n=== PRIMING (discarded, warms up first-encounter JIT/graph-capture costs) ===")
    prime_root = Path(SIMIT_ROOT + "/speed_experiments/output_prime")
    if prime_root.exists():
        shutil.rmtree(prime_root)
    prime_root.mkdir(parents=True)
    sv.OUT_ROOT = prime_root
    t0 = time.perf_counter()
    sv.process_sample(sv._make_client(), SUBSET, PRIME_IDX, ds[PRIME_IDX], CFG)
    print(f"  priming sample done in {time.perf_counter() - t0:.1f}s (discarded)")

    original_total, original_per_sample = run_condition(
        "ORIGINAL (sequential, current pipeline pattern)",
        Path(SIMIT_ROOT + "/speed_experiments/output_original"),
        sample_indices, ds, concurrent=False,
    )
    optimized_total, optimized_per_sample = run_condition(
        "OPTIMIZED (concurrent dispatch, same real pipeline functions)",
        Path(SIMIT_ROOT + "/speed_experiments/output_optimized"),
        sample_indices, ds, concurrent=True,
    )

    result = {
        "n_samples": N_SAMPLES,
        "original_total_s": original_total,
        "original_per_sample_s": original_per_sample,
        "original_avg_s_per_sample": original_total / N_SAMPLES,
        "optimized_total_s": optimized_total,
        "optimized_per_sample_s": optimized_per_sample,
        "optimized_avg_s_per_sample": optimized_total / N_SAMPLES,
        "speedup": original_total / optimized_total,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/e2e_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
