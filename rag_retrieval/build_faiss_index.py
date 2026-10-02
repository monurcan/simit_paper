#!/usr/bin/env python
"""Build a per-benchmark FAISS index (Qwen3-VL-Embedding) from a retrieval
pool built by build_retrieval_sets.py.

Embedding text recipe: row["question"] if present and non-empty, else a fixed
generic caption instruction (only nocaps_val_lite/flickr30k_test_lite lack a
"question" field entirely -- confirmed by inspecting their schemas; every
other benchmark, including refcoco_bbox_val_lite, carries a real usable
question-ish string even where the task's own doc_to_text ignores it).

This is intentionally a *simpler* text than the fully lmms-eval-formatted
prompt used to render the chosen demo at query time (bagel_icl.py reuses
lmms_eval's own Task.doc_to_text/doc_to_target/doc_to_visual for that, native
to the Bagel venv where it already runs) -- retrieval matching only needs a
reasonable text signal for similarity search, and what actually matters is
that the offline pool side and the online query side
(bagel_icl.py::RealSampleRetriever) use the *same* recipe. Keeping that
recipe simple also means this script never needs lmms_eval itself, which
would drag in its full task registry (and per-task metric-library deps) into
this venv for no real benefit.

Run under this project's venv (needs transformers>=4.57, faiss):
    ./.venv/bin/python build_faiss_index.py --benchmark ai2d_lite
    ./.venv/bin/python build_faiss_index.py --all
"""
import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

from embedder import embed_batch, load_embedder

DATA_ROOT = Path(__file__).parent / "data"
BATCH_SIZE = 16

FIXED_CAPTION_TEXT = {
    "nocaps_val_lite": "Describe the image in one sentence.",
    "flickr30k_test_lite": "Describe the image in one sentence.",
}


def log(msg: str) -> None:
    print(f"[build_faiss_index] {msg}", file=sys.stderr, flush=True)


def embed_text_for(task: str, row: dict) -> str:
    q = row.get("question")
    if isinstance(q, str) and q.strip():
        return q
    return FIXED_CAPTION_TEXT.get(task, "Describe the image.")


def first_image_path(pool_dir: Path, image_path) -> str:
    """image_path is a single relative path, or a list of them for
    multi-image tasks (e.g. seedbench) -- embed only the first image,
    matching build_retrieval_sets.py's small_arr/image_dedup_hash convention."""
    rel = image_path[0] if isinstance(image_path, list) else image_path
    return str(pool_dir / rel)


def absolute_image_path(pool_dir: Path, image_path):
    if isinstance(image_path, list):
        return [str(pool_dir / rel) for rel in image_path]
    return str(pool_dir / image_path)


def build_index(task: str, embedder, batch_size: int = BATCH_SIZE) -> None:
    pool_dir = DATA_ROOT / task / "retrieval_pool"
    meta_path = pool_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing -- run build_retrieval_sets.py --benchmark {task} first")

    with open(meta_path) as f:
        rows = [json.loads(line) for line in f]
    log(f"'{task}': {len(rows)} pool rows")

    all_embs = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        items = [{"image": first_image_path(pool_dir, r["image_path"]), "text": embed_text_for(task, r)} for r in batch]
        embs = embed_batch(embedder, items)
        all_embs.append(embs)
        if (i // batch_size) % 20 == 0:
            log(f"  embedded {min(i + batch_size, len(rows))}/{len(rows)}")

    embeddings = np.ascontiguousarray(np.vstack(all_embs), dtype=np.float32)
    faiss.normalize_L2(embeddings)  # idempotent: embed_batch already L2-normalizes

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir = DATA_ROOT / task / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    # Store absolute image paths so embed_worker's IndexCache / bagel_icl.py's
    # RealSampleRetriever can load a retrieved neighbor's image directly,
    # without needing to know the data root.
    samples = []
    for r in rows:
        s = dict(r)
        s["image_path"] = absolute_image_path(pool_dir, r["image_path"])
        samples.append(s)
    with open(index_dir / "samples.json", "w") as f:
        json.dump(samples, f)

    log(f"'{task}': wrote index with {index.ntotal} vectors, dim={index.d} -> {index_dir}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark", help="single task name, e.g. ai2d_lite")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.all:
        tasks = sorted(p.name for p in DATA_ROOT.iterdir() if (p / "retrieval_pool" / "metadata.jsonl").exists())
    else:
        tasks = [args.benchmark]

    embedder = load_embedder()
    for task in tasks:
        log(f"=== building index for '{task}' ===")
        build_index(task, embedder, args.batch_size)


if __name__ == "__main__":
    main()
