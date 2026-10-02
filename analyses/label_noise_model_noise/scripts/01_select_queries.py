"""Step 1: pick the audited original test queries, same set for every method.

Selects N_PER_BENCH doc_ids (0-indexed, == sampleN == lmms-eval doc_id,
positional join verified across baseline eval / generated_samples /
LFRL rollout logs) per benchmark, seeded and reproducible. Tracks a
used-ids registry so a future, larger draw can add NEW queries without
re-selecting ones already audited.

Usage:
    python 01_select_queries.py [--seed 42] [--n-per-bench 10] [--draw-tag draw1]
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BENCHMARKS, LITE_TASK, BASELINE_EVAL_DIR, GENERATED_SAMPLES_ROOT,
    CACHE_DIR, DEFAULT_SEED, DEFAULT_N_PER_BENCH,
)


def available_doc_ids(bench: str) -> set:
    pattern = os.path.join(BASELINE_EVAL_DIR, f"*_samples_{LITE_TASK[bench]}.jsonl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no baseline eval jsonl for {bench}: {pattern}")
    baseline_file = matches[-1]  # latest timestamp
    baseline_ids = set()
    with open(baseline_file) as f:
        for line in f:
            baseline_ids.add(json.loads(line)["doc_id"])

    bench_dir = os.path.join(GENERATED_SAMPLES_ROOT, bench)
    synth_ids = set()
    for d in os.listdir(bench_dir):
        if not d.startswith("sample"):
            continue
        idx = int(d[len("sample"):])
        sample_dir = os.path.join(bench_dir, d)
        required = ["original_sample.png", "original_sample_prompt.txt", "original_sample_answers.txt"]
        if all(os.path.isfile(os.path.join(sample_dir, r)) for r in required):
            synth_ids.add(idx)

    return baseline_ids & synth_ids, baseline_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-per-bench", type=int, default=DEFAULT_N_PER_BENCH)
    ap.add_argument("--draw-tag", type=str, default="draw1", help="Label for this selection batch (future draws use a new tag).")
    args = ap.parse_args()

    registry_path = os.path.join(CACHE_DIR, "used_ids_registry.json")
    registry = json.load(open(registry_path)) if os.path.exists(registry_path) else {b: [] for b in BENCHMARKS}

    selected_path = os.path.join(CACHE_DIR, "selected_queries.jsonl")
    existing_rows = []
    if os.path.exists(selected_path):
        existing_rows = [json.loads(l) for l in open(selected_path)]

    new_rows = []
    for bench in BENCHMARKS:
        avail, baseline_file = available_doc_ids(bench)
        already_used = set(registry.get(bench, []))
        pool = sorted(avail - already_used)
        if len(pool) < args.n_per_bench:
            raise RuntimeError(f"{bench}: only {len(pool)} unused eligible doc_ids left, need {args.n_per_bench}")
        rng = random.Random(f"{args.seed}:{bench}:{args.draw_tag}")
        chosen = sorted(rng.sample(pool, args.n_per_bench))
        for doc_id in chosen:
            new_rows.append({
                "benchmark": bench,
                "doc_id": doc_id,
                "sample_id": f"sample{doc_id}",
                "seed": args.seed,
                "draw_tag": args.draw_tag,
                "baseline_eval_file": baseline_file,
            })
        registry.setdefault(bench, []).extend(chosen)

    with open(selected_path, "w") as f:
        for row in existing_rows + new_rows:
            f.write(json.dumps(row) + "\n")
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Selected {len(new_rows)} new queries ({args.n_per_bench}/benchmark x {len(BENCHMARKS)} benchmarks).")
    print(f"Total queries on file: {len(existing_rows) + len(new_rows)} -> {selected_path}")
    print(f"Registry updated -> {registry_path}")


if __name__ == "__main__":
    main()
