from __future__ import annotations

from collections import Counter
from typing import List

from .text_norm import normalize_answer


def naive_majority_vote_rewards(completions: List[str]) -> List[float]:
    """TTRL's original majority-vote reward: exact string match (after light
    normalization) against the plurality answer. This is degenerate (near-all-
    zero) for free-form text like captions -- that's expected, it's exactly
    what TTRL's own reference implementation does (no clustering)."""
    normalized = [normalize_answer(c) for c in completions]
    counts = Counter(n for n in normalized if n)
    if not counts:
        return [0.0 for _ in completions]
    majority_answer, _ = counts.most_common(1)[0]
    return [1.0 if n == majority_answer and n else 0.0 for n in normalized]


def _word_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def lexical_cluster_vote_rewards(completions: List[str], similarity_threshold: float = 0.6) -> List[float]:
    """EMPO-style extension beyond TTRL's exact-match voting for free-form
    text: groups completions into clusters by pairwise word-set Jaccard
    similarity (self-contained, no external/embedding model, matching the
    test-time self-improvement constraint -- EMPO itself uses an external
    1.5B-param verifier model for this, which our setting can't use), and
    rewards membership in the largest cluster. Word-Jaccard was chosen over
    character-level difflib ratio (the original implementation) after an
    offline comparison scored candidate consensus answers from real saved
    rollouts against ground-truth references (CIDEr, monitoring-only, never
    the reward): word-Jaccard was the best or tied-best of {naive exact
    match, char-difflib, word-difflib, word-Jaccard} on both benchmarks
    tested (coco2017_cap_val, refcoco_bbox_val), while char-difflib
    underperformed naive exact match on refcoco_bbox_val."""
    normalized = [normalize_answer(c) for c in completions]
    n = len(normalized)
    cluster_of = [-1] * n
    clusters: List[List[int]] = []
    for i in range(n):
        if not normalized[i] or cluster_of[i] != -1:
            continue
        cluster_of[i] = len(clusters)
        clusters.append([i])
        for j in range(i + 1, n):
            if cluster_of[j] != -1 or not normalized[j]:
                continue
            ratio = _word_jaccard(normalized[i], normalized[j])
            if ratio >= similarity_threshold:
                cluster_of[j] = cluster_of[i]
                clusters[cluster_of[i]].append(j)
    if not clusters:
        return [0.0 for _ in completions]
    largest_cluster_id = max(range(len(clusters)), key=lambda c: len(clusters[c]))
    return [1.0 if cluster_of[i] == largest_cluster_id else 0.0 for i in range(n)]


def naive_majority_vote_answer(completions: List[str]) -> str:
    """Returns the raw (un-normalized) text of the completion matching the
    plurality normalized answer -- the consensus pseudo-label naive_majority_vote_rewards
    scores against. "" if no completion has a non-empty normalized answer.
    Separate from the rewards function (small duplicated Counter computation,
    cheap for K<=32) to avoid touching that already-verified code path."""
    normalized = [normalize_answer(c) for c in completions]
    counts = Counter(n for n in normalized if n)
    if not counts:
        return ""
    majority_answer, _ = counts.most_common(1)[0]
    for c, n in zip(completions, normalized):
        if n == majority_answer:
            return c
    return ""


def lexical_cluster_vote_answer(completions: List[str], similarity_threshold: float = 0.6) -> str:
    """Returns a representative completion (the cluster's first member) from
    the largest word-Jaccard cluster -- the consensus pseudo-label
    lexical_cluster_vote_rewards scores against."""
    normalized = [normalize_answer(c) for c in completions]
    n = len(normalized)
    cluster_of = [-1] * n
    clusters: List[List[int]] = []
    for i in range(n):
        if not normalized[i] or cluster_of[i] != -1:
            continue
        cluster_of[i] = len(clusters)
        clusters.append([i])
        for j in range(i + 1, n):
            if cluster_of[j] != -1 or not normalized[j]:
                continue
            ratio = _word_jaccard(normalized[i], normalized[j])
            if ratio >= similarity_threshold:
                cluster_of[j] = cluster_of[i]
                clusters[cluster_of[i]].append(j)
    if not clusters:
        return ""
    largest_cluster_id = max(range(len(clusters)), key=lambda c: len(clusters[c]))
    return completions[clusters[largest_cluster_id][0]]
