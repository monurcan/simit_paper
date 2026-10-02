#!/usr/bin/env python
"""Long-lived retrieval worker for the Bagel-ICL RAG evaluation.

Loads Qwen3-VL-Embedding once, then serves embedding+FAISS-retrieval requests
over stdin/stdout, one JSON object per line. Spawned as a subprocess by
`lmms_eval/models/simple/bagel_icl.py::RealSampleRetriever`, which runs in a
*different* venv (transformers==4.49.0, pinned for Bagel's own vendored
modeling code) that cannot import this model (needs transformers>=4.57). Must
be run with this project's venv: $SIMIT_ROOT/rag_retrieval/.venv/bin/python

Protocol
--------
Request (one JSON object per stdin line):
    {"benchmark": "ai2d_lite", "doc_id": 3, "image_path": "/abs/path.png", "text": "...", "k": 4}
`image_path` and `text` are each optional (but not both absent). `benchmark`
must match a folder under --data-root/<benchmark>/index/{index.faiss,samples.json}
built by build_faiss_index.py. `doc_id` is optional (falls back to no caching
for that request) but should always be passed by callers with a stable eval
doc_id -- see query embedding cache below.

Response (one JSON object per stdout line):
    {"results": [{...sample fields..., "_retrieval_score": float}, ...]}
    {"error": "..."}   -- this request failed, worker stays alive for the next one

On startup, prints {"status": "ready", "t_load_s": 0} immediately -- the model
is NOT loaded yet. Query embedding cache: the eval set's queries are fixed
across a k/resize sweep (only num_icl_examples and post-retrieval demo-image
resizing differ; the query embedding itself never changes for a given
(benchmark, doc_id)), so each request's query embedding is looked up in a
persistent on-disk cache at --data-root/<benchmark>/query_embed_cache.npz
first. The (GPU-resident) embedding model is lazily loaded on the *first cache
miss only* -- if a benchmark's cache is already fully warm (e.g. a later run
in the same sweep), this process may never touch the GPU at all. This is safe
for the fixed eval sets this project sweeps over, and remains fully general/
correct for any new benchmark or eval set: a cache miss just costs one real
embedding call and populates the cache for next time.
Exits cleanly when stdin is closed (parent process end / pipe close).
"""
import argparse
import json
import os
import sys
import time
from typing import Optional

import faiss
import numpy as np

from embedder import DEFAULT_MODEL, embed_one, load_embedder


def log(msg: str) -> None:
    print(f"[embed_worker] {msg}", file=sys.stderr, flush=True)


class IndexCache:
    """Lazily loads and caches each benchmark's FAISS index + sample metadata."""

    def __init__(self, data_root: str):
        self.data_root = data_root
        self._cache = {}

    def get(self, benchmark: str):
        if benchmark not in self._cache:
            index_dir = os.path.join(self.data_root, benchmark, "index")
            index_path = os.path.join(index_dir, "index.faiss")
            samples_path = os.path.join(index_dir, "samples.json")
            index = faiss.read_index(index_path)
            with open(samples_path) as f:
                samples = json.load(f)
            log(f"loaded index for '{benchmark}': {index.ntotal} vectors, dim={index.d}")
            self._cache[benchmark] = (index, samples)
        return self._cache[benchmark]


class QueryEmbedCache:
    """Persistent, on-disk cache of query embeddings, keyed by (benchmark,
    doc_id). The eval-set queries a k/resize sweep runs over are fixed, so
    once a benchmark's cache is fully warm from one run, every later run for
    that benchmark can skip the embedding model entirely for every query --
    see module docstring."""

    def __init__(self, data_root: str):
        self.data_root = data_root
        self._mem: dict = {}  # benchmark -> {doc_id_str: np.ndarray}

    def _path(self, benchmark: str) -> str:
        return os.path.join(self.data_root, benchmark, "query_embed_cache.npz")

    def _load(self, benchmark: str) -> dict:
        if benchmark not in self._mem:
            path = self._path(benchmark)
            if os.path.exists(path):
                with np.load(path) as npz:
                    self._mem[benchmark] = {k: npz[k] for k in npz.files}
                log(f"loaded query embed cache for '{benchmark}': {len(self._mem[benchmark])} entries")
            else:
                self._mem[benchmark] = {}
        return self._mem[benchmark]

    def get(self, benchmark: str, doc_id) -> Optional[np.ndarray]:
        if doc_id is None:
            return None
        return self._load(benchmark).get(str(doc_id))

    def put(self, benchmark: str, doc_id, emb: np.ndarray) -> None:
        if doc_id is None:
            return
        self._load(benchmark)[str(doc_id)] = emb
        self._save(benchmark)

    def _save(self, benchmark: str) -> None:
        # Atomic write (temp file + rename) so a killed job never corrupts an
        # already-warm cache another job might be reading concurrently.
        # tmp_path must itself end in ".npz" -- np.savez silently appends
        # ".npz" to any path that doesn't already have it, which would
        # otherwise desync the name actually written from the name we
        # os.replace() (confirmed: this broke every save with FileNotFoundError).
        path = self._path(benchmark)
        tmp_path = f"{path}.tmp{os.getpid()}.npz"
        np.savez(tmp_path, **self._mem[benchmark])
        os.replace(tmp_path, path)


def handle_request(req: dict, embedder_holder: dict, index_cache: IndexCache, query_cache: QueryEmbedCache) -> dict:
    benchmark = req["benchmark"]
    doc_id = req.get("doc_id")
    k = int(req.get("k", 4))
    image_path = req.get("image_path")
    text = req.get("text")
    if not image_path and not text:
        raise ValueError("request must include 'image_path' and/or 'text'")

    index, samples = index_cache.get(benchmark)

    query_emb = query_cache.get(benchmark, doc_id)
    if query_emb is None:
        if embedder_holder["model"] is None:
            log("query embed cache miss -- lazily loading embedding model onto GPU now")
            t0 = time.time()
            embedder_holder["model"] = load_embedder(embedder_holder["model_name"], **embedder_holder["load_kwargs"])
            log(f"model loaded in {time.time() - t0:.1f}s")
        query_emb = embed_one(embedder_holder["model"], image_path=image_path, text=text)
        query_cache.put(benchmark, doc_id, query_emb)

    query_emb = np.ascontiguousarray(query_emb[None, :], dtype=np.float32)
    faiss.normalize_L2(query_emb)  # idempotent: embed_one already L2-normalizes

    k = min(k, index.ntotal)
    scores, indices = index.search(query_emb, k)

    results = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0:
            continue
        sample = dict(samples[idx])
        sample["_retrieval_score"] = score
        results.append(sample)
    return {"results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-root", required=True, help="rag_retrieval/data root containing <benchmark>/index/")
    parser.add_argument("--attn-implementation", default=None)
    args = parser.parse_args()

    load_kwargs = {"attn_implementation": args.attn_implementation} if args.attn_implementation else {}
    # "model" stays None until the first query-cache miss actually needs it
    # (see module docstring / QueryEmbedCache) -- this dict is mutated in
    # place by handle_request so the loaded model persists across requests.
    embedder_holder = {"model": None, "model_name": args.model, "load_kwargs": load_kwargs}

    index_cache = IndexCache(args.data_root)
    query_cache = QueryEmbedCache(args.data_root)

    print(json.dumps({"status": "ready", "t_load_s": 0}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req, embedder_holder, index_cache, query_cache)
        except Exception as e:
            log(f"error handling request {line[:200]!r}: {e!r}")
            resp = {"error": str(e)}
        print(json.dumps(resp), flush=True)

    log("stdin closed, exiting")


if __name__ == "__main__":
    main()
