from __future__ import annotations

import math
from typing import List, Tuple

from common.text_norm import normalize_answer

VOTING_MODES = ("naive", "lexical")


def _word_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def _cluster_completions(
    completions: List[str],
    voting_mode: str,
    lexical_similarity_threshold: float,
) -> Tuple[List[List[int]], List[int]]:
    """Buckets completions into unique-output clusters. TTRV's Eq. 3 defines
    uniqueness via literal string equality, which -- exactly like TTRL's own
    naive_majority_vote_rewards -- is degenerate for free-form text. We reuse
    this project's existing naive (exact-match) / lexical (word-Jaccard)
    choice as the equality relation, the same disclosed substitution already
    used for EMPO."""
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
            if voting_mode == "naive":
                same = normalized[i] == normalized[j]
            else:
                same = _word_jaccard(normalized[i], normalized[j]) >= lexical_similarity_threshold
            if same:
                cluster_of[j] = cluster_of[i]
                clusters[cluster_of[i]].append(j)
    return clusters, cluster_of


def compute_ttrv_rewards(
    completions: List[str],
    voting_mode: str = "naive",
    lexical_similarity_threshold: float = 0.6,
) -> Tuple[List[float], float]:
    """TTRV's two reward terms (paper Eq. 3-6):

      r1[i] = p(cluster of completion i) -- frequency-based reward, Eq. 4.
              Soft/continuous (unlike TTRL's hard 0/1 majority reward):
              every completion gets nonzero credit proportional to how many
              of the K samples landed in the same cluster, not just the
              single largest one.
      r2     = -H(P), the diversity-control term, Eq. 5-6 -- one scalar
              shared by the whole group of K completions (it depends only
              on the empirical cluster-size distribution, not on which
              completion i we're scoring).

    Returns (r1_per_completion, r2_scalar) rather than a single combined
    per-completion reward -- see train_ttrv.py for why they must be combined
    AFTER GRPO's per-group advantage normalization, not before.
    """
    clusters, cluster_of = _cluster_completions(completions, voting_mode, lexical_similarity_threshold)
    n = len(completions)
    if not clusters:
        return [0.0 for _ in completions], 0.0
    probs = [len(c) / n for c in clusters]
    r1 = [probs[cluster_of[i]] if cluster_of[i] != -1 else 0.0 for i in range(n)]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    r2 = -entropy
    return r1, r2
