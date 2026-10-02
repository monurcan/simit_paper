from __future__ import annotations

import re
from typing import List, Optional

from PIL import Image

from common.rollout import generate_k_completions

# Verbatim from the original SRLM paper (Yuan et al. 2024, Figure 2) -- MMUPT's
# own SRLM baseline description explicitly says "we... use the prompt provided
# in the original paper", so this is copied exactly, including the "utilizing
# web search knowledge as necessary" clause even though our model has no such
# capability (disclosed, not silently dropped).
JUDGE_PROMPT_TEMPLATE = """Review the user's question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

User: {instruction}

<response>{response}</response>

After examining the user's instruction and the response:

- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Score: <total points>"

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we'll systematically attribute points based on the outlined criteria."""

_SCORE_RE = re.compile(r"Score:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_score(judge_output: str) -> Optional[float]:
    """Extracts the final "Score: X" the judge prompt asks for. Takes the
    LAST match in case the justification text itself mentions a criterion's
    point number before the concluding line. Returns None (not 0) when
    unparseable, so callers can tell "judge gave a real 0" apart from
    "judge output was malformed" -- treating a malformed judge response as a
    zero score would silently bias every such completion toward "rejected"."""
    matches = _SCORE_RE.findall(judge_output or "")
    if not matches:
        return None
    try:
        score = float(matches[-1])
    except ValueError:
        return None
    if not (0.0 <= score <= 5.0):
        return None
    return score


def score_completions(
    runtime,
    *,
    image: Image.Image,
    instruction: str,
    completions: List[str],
    max_new_tokens: int = 200,
) -> List[Optional[float]]:
    """Self-rewarding step: the SAME model judges each of its own candidate
    completions via LLM-as-a-Judge prompting (Section 2.2 of the SRLM paper).
    One independent judge call per completion (greedy decoding -- the score
    is a self-generated training signal, not a benchmark metric, so a
    deterministic single pass is both cheaper and simpler to reason about
    than the original paper's average-of-3-sampled-evaluations; disclosed
    simplification, not a silent deviation)."""
    scores: List[Optional[float]] = []
    for completion in completions:
        if not completion:
            scores.append(None)
            continue
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(instruction=instruction, response=completion)
        judge_output = generate_k_completions(
            runtime,
            image=image,
            prompt=judge_prompt,
            k=1,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )[0]
        scores.append(parse_score(judge_output))
    return scores


def select_preference_pair(
    completions: List[str],
    scores: List[Optional[float]],
):
    """Highest-scored completion is the chosen (preferred) response, lowest-
    scored is the rejected one -- exactly MM-UPT's stated SRLM baseline
    recipe. Unparseable-score completions are excluded from selection
    entirely (neither chosen nor rejected -- their score is unknown, not
    low). Returns None if fewer than 2 scoreable completions remain, or if
    the best and worst score tie (discarded, per the original SRLM paper
    Section 2.3: "discarding the pair if their scores are the same")."""
    scored = [(c, s) for c, s in zip(completions, scores) if s is not None]
    if len(scored) < 2:
        return None
    scored.sort(key=lambda cs: cs[1])
    worst_completion, worst_score = scored[0]
    best_completion, best_score = scored[-1]
    if best_score == worst_score:
        return None
    return best_completion, worst_completion, best_score, worst_score
