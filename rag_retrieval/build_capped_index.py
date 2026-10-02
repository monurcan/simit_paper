#!/usr/bin/env python
"""Build a "capped" copy of each benchmark's FAISS index, restricted to the
exact same N=2000 (N=1000 for chartqa) real-neighbor samples used by the
real-data finetuning study (Bagel/prepare_vqa_data_real_rag.py), so the
RAG-ICL k-sweep can be repeated on the identical retrieval pool the
finetuning experiment trained on.

Reuses prepare_vqa_data_real_rag.py's exact selection: SEED=0,
random.Random(SEED).shuffle(list(range(pool_size))), first N indices. Since
each benchmark's index.faiss is IndexFlatIP (uncompressed), the selected
vectors are reconstructed losslessly from the existing index -- no
re-embedding, no GPU. query_embed_cache.npz (query-side embeddings, keyed by
eval doc_id) doesn't depend on the pool at all, so it's copied over verbatim
to skip re-embedding queries too.

nocaps_val_lite/textcaps_val_lite/textvqa_val_lite: their deployed index.faiss
has fewer vectors than samples.json/metadata.jsonl (a pre-existing bug in
build_faiss_index.py -- one bad image in a batch of 16 collapses that whole
batch to a single dummy embedding, silently desyncing every row after it), so
index row i doesn't reliably match samples[i] for these three. Not worth
re-embedding to fix (no GPU needed for this task) -- instead this just draws a
*different* random N from the still-valid overlapping range
[0, min(index.ntotal, len(samples))), with a different seed (SEED_FALLBACK=1)
so it's clearly not claimed to be the finetuning study's exact sample set.

Run under rag_retrieval/.venv (has faiss):
    ./.venv/bin/python build_capped_index.py --all
    ./.venv/bin/python build_capped_index.py --benchmark ai2d_lite
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import faiss
import numpy as np

DATA_ROOT = Path(__file__).parent / "data"
CAPPED_ROOT = Path(__file__).parent / "data_capped2000"
SEED = 0
SEED_FALLBACK = 1
MILESTONES = [25, 50, 75, 100, 125, 150, 250, 500, 1000, 2000]
# chartqa_lite's pool (1946) falls between the 1000 and 2000 milestones, so
# the finetuning study only ever trained on 1000 of its 1946 samples. For
# this ICL experiment (not required to match that milestone list) use the
# whole pool instead of leaving ~946 samples unused.
FULL_POOL_BENCHES = {"chartqa_lite"}

# lmms-eval lite task name -> our short bench name used by
# prepare_vqa_data_real_rag.py (kept only for the printed summary).
TASKS = [
    "ai2d_lite", "chartqa_lite", "coco2017_cap_val_lite", "docvqa_val_lite",
    "flickr30k_test_lite", "gqa_lite", "infovqa_val_lite", "mmbench_cn_dev_lite",
    "mmbench_en_dev_lite", "nocaps_val_lite", "ok_vqa_val2014_lite",
    "refcoco_bbox_val_lite", "seedbench_lite", "textcaps_val_lite",
    "textvqa_val_lite", "vizwiz_vqa_val_lite", "vqav2_val_lite",
    # mathvista_testmini: empty retrieval pool, no real-data corpus -- skip.
]


def cap_one(task: str) -> dict:
    src_dir = DATA_ROOT / task
    meta_path = src_dir / "retrieval_pool" / "metadata.jsonl"
    index_path = src_dir / "index" / "index.faiss"
    samples_path = src_dir / "index" / "samples.json"

    pool_size = sum(1 for l in meta_path.read_text().splitlines() if l.strip())
    if pool_size == 0:
        print(f"{task}: EMPTY pool, skipping")
        return {"skipped": True}

    index = faiss.read_index(str(index_path))
    with open(samples_path) as f:
        samples = json.load(f)

    aligned = index.ntotal == pool_size and len(samples) == pool_size
    if aligned:
        rng = random.Random(SEED)
        order = list(range(pool_size))
        rng.shuffle(order)
        if task in FULL_POOL_BENCHES:
            n = pool_size
        else:
            milestones = sorted(set(m for m in MILESTONES if m <= pool_size))
            n = milestones[-1] if milestones else pool_size
        selected = order[:n]
    else:
        # Misaligned index (see module docstring) -- fall back to a
        # different random draw over the range both arrays agree on.
        valid_n = min(index.ntotal, len(samples))
        rng = random.Random(SEED_FALLBACK)
        order = list(range(valid_n))
        rng.shuffle(order)
        milestones = sorted(set(m for m in MILESTONES if m <= valid_n))
        n = milestones[-1] if milestones else valid_n
        selected = order[:n]
        print(f"{task}: index/pool misaligned (index.ntotal={index.ntotal}, samples.json={len(samples)}, pool_size={pool_size}) "
              f"-- using {n} different random samples (seed={SEED_FALLBACK}) instead of the finetuning study's exact set")

    vecs = np.vstack([index.reconstruct(i) for i in selected]).astype(np.float32)
    new_index = faiss.IndexFlatIP(index.d)
    new_index.add(vecs)
    new_samples = [samples[i] for i in selected]

    out_dir = CAPPED_ROOT / task
    out_index_dir = out_dir / "index"
    out_index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(new_index, str(out_index_dir / "index.faiss"))
    with open(out_index_dir / "samples.json", "w") as f:
        json.dump(new_samples, f)

    query_cache_src = src_dir / "query_embed_cache.npz"
    if query_cache_src.exists():
        shutil.copy2(query_cache_src, out_dir / "query_embed_cache.npz")

    print(f"{task}: pool_size={pool_size} capped_n={n} aligned={aligned} -> {out_index_dir}")
    return {"pool_size": pool_size, "capped_n": n, "aligned": aligned}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--benchmark", type=str)
    args = parser.parse_args()

    tasks = TASKS if args.all else [args.benchmark]
    summary = {}
    for task in tasks:
        summary[task] = cap_one(task)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
