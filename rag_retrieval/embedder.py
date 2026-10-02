"""Shared Qwen3-VL-Embedding loading/embedding helpers.

Used by both `embed_worker.py` (the long-lived online worker that
`lmms_eval/models/simple/bagel_icl.py::RealSampleRetriever` talks to over a
subprocess pipe) and `build_faiss_index.py` (the offline per-benchmark index
builder). Keeping the loading/embedding recipe in one place guarantees the
online query embedding and the offline retrieval-pool embeddings are computed
identically.

Must be run under this project's dedicated venv
($SIMIT_ROOT/rag_retrieval/.venv, transformers>=4.57) -- NOT the Bagel
eval venv (transformers==4.49.0, pinned because Bagel vendors modeling code
that reaches into transformers internals that changed since then).
"""
from typing import Optional

import numpy as np

from qwen3_vl_embedding_official import Qwen3VLEmbedder

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def load_embedder(model_name_or_path: str = DEFAULT_MODEL, **kwargs) -> Qwen3VLEmbedder:
    return Qwen3VLEmbedder(model_name_or_path=model_name_or_path, **kwargs)


def embed_one(embedder: Qwen3VLEmbedder, image_path: Optional[str] = None, text: Optional[str] = None) -> np.ndarray:
    """Embed a single (image, text) pair -- either may be omitted, not both."""
    return embed_batch(embedder, [{"image": image_path, "text": text}])[0]


def embed_batch(embedder: Qwen3VLEmbedder, items: list) -> np.ndarray:
    """items: list of {"image": path_or_None, "text": str_or_None}.

    Returns (N, D) float32, L2-normalized (Qwen3VLEmbedder.process default).
    """
    embeddings = embedder.process(items)
    return embeddings.detach().cpu().float().numpy()
