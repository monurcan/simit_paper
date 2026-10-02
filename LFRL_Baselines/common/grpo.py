from __future__ import annotations

import math
from typing import List


def group_normalized_advantages(rewards: List[float], eps: float = 1e-6) -> List[float]:
    """GRPO advantage: per-group (r - mean) / std. Falls back to 0.0 when the
    group has ~zero reward variance (every sampled completion agreed or
    disagreed identically), which is the correct no-op: there's no relative
    signal to learn from within that group."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std > eps:
        return [(r - mean) / std for r in rewards]
    return [0.0 for _ in rewards]
