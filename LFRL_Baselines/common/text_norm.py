from __future__ import annotations

import string

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Standard VQA-style normalization: lowercase, strip punctuation, drop
    articles, collapse whitespace. Used only to compare the model's own sampled
    completions against each other for self-consistency voting -- never against
    ground truth.

    Articles are only stripped from multi-word answers ("a dog" -> "dog"): a
    bare single-word answer is left alone, since several benchmarks' correct
    answer format IS a single letter/word that happens to collide with an
    article -- e.g. multiple-choice "A" is an answer choice, not the English
    article "a", and must not be normalized away to an empty string."""
    text = str(text or "").strip().lower()
    text = text.translate(_PUNCT_TABLE)
    words = text.split()
    if len(words) > 1:
        words = [w for w in words if w not in _ARTICLES]
    return " ".join(words)
