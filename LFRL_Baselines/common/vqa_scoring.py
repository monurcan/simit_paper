from __future__ import annotations

from typing import List

from .text_norm import normalize_answer

# Benchmarks scored by CIDEr (via pycocoevalcap's real TF-IDF corpus
# statistics) in the official lmms-eval harness -- vqa_soft_accuracy below is
# not a substitute for that (comparing full free-form sentences for exact
# equality is structurally near-always-zero, same failure mode as naive
# majority voting on captions). Training-time monitoring accuracy is skipped
# entirely for these; the official lmms-eval numbers are the only trustworthy
# source for them.
CIDER_SCORED_BENCHMARKS = {"coco2017_cap_val", "nocaps_val", "textcaps_val", "flickr30k_test", "refcoco_bbox_val"}


def vqa_soft_accuracy(prediction: str, gt_answers: List[str]) -> float:
    """Standard VQA soft-accuracy formula (from the original VQA paper): for
    >=2 reference answers (multi-annotator VQA-style benchmarks), the
    leave-one-out average of min(#other-annotators-matching / 3, 1). For a
    single reference (e.g. GQA), falls back to plain normalized exact match.

    This is a training-time MONITORING proxy using a well-documented public
    formula -- it is not guaranteed bit-identical to lmms-eval's own
    per-benchmark scorer (which may apply additional, benchmark-specific
    normalization rules), and is never used in the reward/loss. The official
    lmms-eval evaluation remains the authoritative number for reporting."""
    pred = normalize_answer(prediction)
    gts = [normalize_answer(a) for a in gt_answers if a]
    if not gts:
        return 0.0
    if len(gts) == 1:
        return 1.0 if pred and pred == gts[0] else 0.0
    accs = []
    for i in range(len(gts)):
        other = gts[:i] + gts[i + 1 :]
        matches = sum(1 for g in other if g == pred)
        accs.append(min(matches / 3.0, 1.0))
    return sum(accs) / len(accs)
