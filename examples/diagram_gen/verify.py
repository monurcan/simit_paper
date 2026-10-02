"""Optional post-hoc visual verification for NATURAL-category images: ask
BAGEL's own text/vision head whether the generated photo actually matches
the request, and regenerate if not.

Scoped to NATURAL only -- benchmarked on 20 prompts spanning every other
category (GEOMETRY/FIGURE/TABLE/VENN/SCENE/PUZZLE/DIAGRAM/GRAPH/MOLECULE/
CIRCUIT/VECTOR): applying this same loop there *lowered* average output
quality (manual 1.0/0.5/0/0 scoring: 0.775 -> 0.675), because the strict
critic doesn't generalize to line-diagram/chart content (it false-failed
several already-correct renders, burning the retry budget for nothing) and
because re-extracting a DSL from a revised prompt can drift into a worse
result (wrong category, hallucinated data, leaked placeholder text) instead
of fixing the actual problem. Those categories already render
deterministically from a parsed spec -- correct once parsed -- so a photo
critic has nothing useful to check.
"""
import re
from pathlib import Path

from mstar import MStarClient

_CRITIC_PROMPT = (
    'You are a strict quality-control critic reviewing an AI-generated image against this brief: "{request}"\n\n'
    "Go through the brief clause by clause. For each specific, checkable detail "
    "(counts, colors, species, objects, setting, text), state whether the image "
    "satisfies it. List every discrepancy you find, however small, with no "
    "tolerance for approximation. End your answer with a line that says either "
    "'VERDICT: PASS' or 'VERDICT: FAIL'."
)

# Same brief as _CRITIC_PROMPT, plus a self-reported confidence score so
# callers can threshold on it (see generate_with_verification's
# confidence_threshold param) instead of trusting a bare PASS/FAIL. This is a
# model self-report, not a real probability -- mstar's serving stack doesn't
# expose token logprobs anywhere, so this is the cheapest available proxy.
_CRITIC_PROMPT_CONFIDENCE = (
    'You are a strict quality-control critic reviewing an AI-generated image against this brief: "{request}"\n\n'
    "Go through the brief clause by clause. For each specific, checkable detail "
    "(counts, colors, species, objects, setting, text), state whether the image "
    "satisfies it. List every discrepancy you find, however small, with no "
    "tolerance for approximation. End your answer with exactly two lines:\n"
    "'VERDICT: PASS' or 'VERDICT: FAIL'\n"
    "'CONFIDENCE: <integer 0-100>' -- how confident you are in that verdict "
    "(100 = certain beyond doubt, 0 = pure guess)."
)

# Rubric-based quality score instead of a binary verdict + confidence-in-that-
# verdict: an experiment on 96 already-generated images (all categories) found
# CONFIDENCE from _CRITIC_PROMPT_CONFIDENCE barely discriminates -- PASS
# clustered at 95-100 and FAIL spanned 80-100 with real overlap, because it
# measures "how sure am I about my verdict" (usually very sure either way),
# not "how good is this image". A direct 0-100 quality score with an explicit
# rubric gives the same critic pass a continuous target to land on instead.
_CRITIC_PROMPT_SCORE = (
    'You are a strict quality-control critic scoring an AI-generated image against this brief: "{request}"\n\n'
    "Go through the brief clause by clause. For each specific, checkable detail "
    "(counts, colors, species, objects, setting, text, spatial relations), state "
    "whether the image satisfies it, with no tolerance for approximation. List "
    "every discrepancy you find, however small.\n\n"
    "Then assign a single quality SCORE from 0 to 100 using this rubric:\n"
    "  90-100: every checkable detail in the brief is satisfied, no discrepancies\n"
    "  70-89:  matches the brief overall, only minor/cosmetic discrepancies "
    "(e.g. slightly off shade, imprecise count in a busy/cluttered scene)\n"
    "  40-69:  matches part of the brief but has at least one clear factual "
    "discrepancy (wrong count, wrong object, wrong text, wrong spatial relation)\n"
    "  0-39:   mostly or entirely fails to match the brief\n\n"
    "End your answer with exactly one line: 'SCORE: <integer 0-100>'."
)

# For DSL/code-rendered categories (charts, diagrams, tables, geometry, ...):
# an experiment generating 20 fresh chartqa-inspired FIGURE charts found this
# exact-detail rubric scores 0 or 40 on charts that are visually clean, well-
# labeled, and structurally correct -- because the chart's description often
# contains specific invented numbers (e.g. "Lamb's price index is 103") that
# an LLM writing matplotlib code from scratch can never be expected to
# reproduce exactly. That mirrors this module's own original NATURAL-only
# scoping rationale (see module docstring): a strict critic doesn't
# generalize to this content. This variant explicitly asks the critic to
# ignore exact numeric/data mismatches and judge structural correctness
# instead -- the actual thing a deterministic DSL renderer can get right or
# wrong.
_CRITIC_PROMPT_SCORE_STRUCTURED = (
    'You are a quality-control critic scoring an AI-generated chart/diagram/figure '
    'against this brief: "{request}"\n\n'
    "This image is being generated as training data by first asking a model for a "
    "detailed description (which may invent specific numbers/labels/statistics), then "
    "asking a separate model to write code that renders that description. The exact "
    "numeric values, colors, or specific labels in the brief are NOT required to match "
    "precisely -- do not penalize the image for a different number, value, or label than "
    "the brief states. Judge STRUCTURAL correctness only:\n"
    "- Is it the right TYPE of visualization (e.g. bar chart vs line chart vs pie chart vs "
    "diagram vs table), matching what the brief describes?\n"
    "- Does it have the general elements the brief mentions (e.g. axis labels, a title, "
    "the right number of categories/bars/nodes, a legend if described)?\n"
    "- Is the image coherent and readable, with no garbled text, broken layout, or "
    "rendering artifacts?\n\n"
    "Then assign a single quality SCORE from 0 to 100 using this rubric:\n"
    "  90-100: right visualization type, all described structural elements present, clean and readable\n"
    "  70-89:  right visualization type with the described elements, but missing one minor "
    "structural element (e.g. no legend when one was implied) or slightly cluttered\n"
    "  40-69:  right general visualization type but missing a clearly-described structural "
    "element, or somewhat hard to read\n"
    "  0-39:   wrong visualization type entirely, or broken/garbled/unreadable rendering\n\n"
    "End your answer with exactly one line: 'SCORE: <integer 0-100>'."
)

_REVISE_PROMPT = (
    'This image description was used to generate an image: "{request}"\n'
    'A critic reviewed the generated image and found these problems:\n"{critique}"\n\n'
    "Rewrite the description, keeping everything that was already correct, but "
    "adding or changing whatever is needed so the next generation avoids these "
    "exact problems. Output ONLY the revised description text, nothing else."
)


def _parse_confidence(text: str) -> int | None:
    m = re.search(r"CONFIDENCE:\s*(\d{1,3})", text, re.IGNORECASE)
    if not m:
        return None
    return max(0, min(100, int(m.group(1))))


def _parse_score(text: str) -> int | None:
    m = re.search(r"SCORE:\s*(\d{1,3})", text, re.IGNORECASE)
    if not m:
        return None
    return max(0, min(100, int(m.group(1))))


_CRITERIA = ("verdict", "verdict_confidence", "score", "score_structured")


def verify(
    client: MStarClient, request: str, image_path: str,
    think_mode: bool = True, criterion: str = "verdict",
) -> tuple[bool, str, int | None]:
    """Ask the model whether `image_path` satisfies `request`. Returns
    (passed, critique_text, metric).

    criterion:
      - "verdict":            plain VERDICT: PASS/FAIL (_CRITIC_PROMPT). metric
                               is always None; this is the original behavior.
      - "verdict_confidence": VERDICT plus a self-reported CONFIDENCE 0-100 in
                               that verdict (_CRITIC_PROMPT_CONFIDENCE). Found
                               to barely discriminate quality (PASS clustered
                               95-100, FAIL spanned 80-100, overlapping) --
                               kept for completeness, "score" is preferred.
      - "score":               a single rubric-based quality SCORE 0-100
                               (_CRITIC_PROMPT_SCORE, see its docstring).
                               `passed` is True whenever a score was parsed;
                               the actual accept/reject decision is the
                               caller's via confidence_threshold against
                               `metric` (the score).
      - "score_structured":    same 0-100 SCORE mechanics as "score", but with
                               a critic that explicitly ignores exact numeric/
                               data/label mismatches and grades structural
                               correctness instead (_CRITIC_PROMPT_SCORE_STRUCTURED,
                               see its docstring) -- use for DSL/code-rendered
                               categories (charts, diagrams, tables, ...), where
                               the strict "score" rubric was found to score
                               visually-correct renders at 0-40 just because an
                               invented specific number didn't match exactly.
    """
    # matplotlib's savefig() defaults to RGBA; BAGEL's vision preprocessing
    # expects 3 channels and 500s on 4 -- seen on every DSL-rendered PNG,
    # never on BAGEL's own RGB photo output. Harmless no-op for RGB inputs.
    from PIL import Image
    img = Image.open(image_path)
    if img.mode != "RGB":
        rgb_path = str(Path(image_path).with_suffix(".rgb.png"))
        img.convert("RGB").save(rgb_path)
        image_path = rgb_path
    prompt_template = {
        "verdict": _CRITIC_PROMPT,
        "verdict_confidence": _CRITIC_PROMPT_CONFIDENCE,
        "score": _CRITIC_PROMPT_SCORE,
        "score_structured": _CRITIC_PROMPT_SCORE_STRUCTURED,
    }[criterion]
    resp = client.chat(
        prompt_template.format(request=request), images=image_path, think_mode=think_mode,
        max_output_tokens=300, temperature=0.0,
    )
    text = resp.text.strip()
    if criterion in ("score", "score_structured"):
        score = _parse_score(text)
        return score is not None, text, score
    passed = bool(re.search(r"VERDICT:\s*PASS", text, re.IGNORECASE))
    confidence = _parse_confidence(text) if criterion == "verdict_confidence" else None
    return passed, text, confidence


def revise_request(
    client: MStarClient, original_request: str, critique: str, think_mode: bool = True,
) -> str:
    """Rewrite `original_request` to address the critic's specific complaints."""
    resp = client.chat(
        _REVISE_PROMPT.format(request=original_request, critique=critique),
        think_mode=think_mode, max_output_tokens=200, temperature=0.0,
    )
    return resp.text.strip()


def generate_with_verification(
    client: MStarClient,
    request: str,
    out_path: str,
    generate_once,
    strategy: str = "informed",
    max_attempts: int = 5,
    think_mode: bool = True,
    confidence_threshold: int | None = None,
    discard_unverified: bool = False,
    criterion: str = "verdict",
) -> bool:
    """Repeatedly call `generate_once(prompt) -> bool` (writes to out_path on
    success) until `verify()` passes or `max_attempts` is reached.

    strategy:
      - "none":     single shot, no verification (current pipeline default
                    before this existed).
      - "plain":    on failure, retry with the *same* request -- pure
                    resampling.
      - "informed": on failure, rewrite the request around the critic's
                    specific complaint before retrying.

    criterion ("verdict" / "verdict_confidence" / "score", default "verdict"):
    which critic prompt/parsing verify() uses -- see its docstring. Only
    matters when confidence_threshold is set (with plain "verdict" there's no
    metric to threshold on, so confidence_threshold has no effect).

    confidence_threshold (None by default): when set, an attempt is only
    accepted once verify() both passes *and* its metric (confidence or score,
    per `criterion`) is >= confidence_threshold -- a bare PASS is not enough.

    discard_unverified (False by default, preserves the original behavior):
    when False, always leaves the *last successfully generated* image at
    out_path even if every attempt failed verification -- a wasted-but-
    harmless retry budget is better than no image. When True, an unaccepted
    image after max_attempts is deleted and this returns False instead, so
    the caller can skip saving that sample entirely.
    """
    if strategy == "none":
        return generate_once(request)

    current_request = request
    last_ok = False
    consecutive_gen_failures = 0
    for attempt in range(max_attempts):
        last_ok = generate_once(current_request)
        if not last_ok:
            consecutive_gen_failures += 1
            if consecutive_gen_failures >= 2:
                break  # server likely stuck; let watchdog handle it
            continue
        consecutive_gen_failures = 0
        try:
            passed, critique, confidence = verify(
                client, request, out_path, think_mode=think_mode, criterion=criterion,
            )
        except Exception as e:
            # Verification is best-effort QC: the image itself just generated
            # fine, so keep it rather than losing the synth to a transient
            # critic-chat error (e.g. a timeout right after this image) --
            # unless the caller demands a confident verified pass, in which
            # case a skipped check can't count as one.
            if discard_unverified:
                print(f"  [verify attempt {attempt}] verify error, discarding: {e}")
                continue
            print(f"  [verify attempt {attempt}] verify error, accepting image: {e}")
            return True
        accepted = passed and (confidence is None or confidence >= confidence_threshold)
        conf_str = f" {criterion}={confidence}" if criterion != "verdict" else ""
        print(f"  [verify attempt {attempt}] passed={passed}{conf_str}")
        if accepted:
            return True
        if attempt < max_attempts - 1 and strategy == "informed":
            try:
                current_request = revise_request(client, current_request, critique, think_mode=think_mode)
            except Exception as e:
                print(f"  [verify attempt {attempt}] revise error, resampling with same request: {e}")
    if discard_unverified:
        Path(out_path).unlink(missing_ok=True)
        return False
    return last_ok
