from __future__ import annotations

from typing import List

from common.voting import lexical_cluster_vote_rewards, naive_majority_vote_rewards

VOTING_MODES = ("naive", "lexical")


def compute_ttrl_rewards(
    completions: List[str],
    voting_mode: str = "naive",
    lexical_similarity_threshold: float = 0.6,
) -> List[float]:
    if voting_mode == "naive":
        return naive_majority_vote_rewards(completions)
    if voting_mode == "lexical":
        return lexical_cluster_vote_rewards(completions, similarity_threshold=lexical_similarity_threshold)
    raise ValueError(f"Unknown voting_mode={voting_mode!r}, expected one of {VOTING_MODES}")
