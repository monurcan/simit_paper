"""GQA retrieval-pool row loader.

GQA is a special case among the benchmarks: `lmms-lab/LMMs-Eval-Lite`'s "gqa"
rows carry only an `imageId` reference (no embedded image bytes), and
`lmms-lab-encoder/GQA` mirrors this same images/instructions split rather
than one self-contained table -- confirmed via
`similar_samples_llms_eval_lite/fetch_gqa_images.py`, which resolves the same
Lite imageIds against `lmms-lab/GQA`'s per-split image configs.

We use the "testdev_all_*" configs (the superset of "testdev_balanced_*",
which is what the 12,578-row distillation split was drawn from -- matches
GQA's well-known testdev-balanced size exactly) so the retrieval pool covers
all testdev questions, not just the balanced subset.
"""
import sys

from datasets import load_dataset

GQA_REPO = "lmms-lab-encoder/GQA"


def log(msg: str) -> None:
    print(f"[build_gqa_retrieval_set] {msg}", file=sys.stderr, flush=True)


def iter_gqa_rows():
    log(f"loading '{GQA_REPO}' testdev_all_images ...")
    images_ds = load_dataset(GQA_REPO, "testdev_all_images", split="testdev")
    id2image = {row["id"]: row["image"] for row in images_ds}
    log(f"  {len(id2image)} images loaded")

    log(f"loading '{GQA_REPO}' testdev_all_instructions ...")
    instructions_ds = load_dataset(GQA_REPO, "testdev_all_instructions", split="testdev")
    log(f"  {len(instructions_ds)} instructions loaded")

    n_missing_image = 0
    for row in instructions_ds:
        img = id2image.get(row["imageId"])
        if img is None:
            n_missing_image += 1
            continue
        merged = dict(row)
        merged["image"] = img
        yield merged
    if n_missing_image:
        log(f"  {n_missing_image} instructions had no matching image, skipped")
