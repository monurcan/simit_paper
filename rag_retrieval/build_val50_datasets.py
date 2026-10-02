#!/usr/bin/env python
"""Turn the 50-sample validation splits into lmms-eval-loadable datasets.

build_retrieval_sets.py carves, per benchmark, a 50-sample validation split
out of the full benchmark (disjoint from both the 500 LMMs-Eval-Lite eval
samples and the retrieval/training pool) and writes it as
    rag_retrieval/data/<task>/validation/{metadata.jsonl, images/}.
This script converts each of those into a local HF dataset with exactly the
LMMs-Eval-Lite schema (same columns and feature types), saved with
`save_to_disk` under $SIMIT_DATA/val50/<task>/ (single split "val"). The
`<task>_val50` lmms-eval tasks load it with `load_from_disk` and otherwise
reuse the `<task>` Lite task unchanged (prompts, filters, metrics), and
similar_samples_llms_eval_lite/synthesize_vqa.py --split=val50 synthesizes
practice samples for these validation queries.

Two benchmarks need a small adapter because their validation rows come from a
repo whose schema differs from Lite's:
  * gqa_lite: Lite rows carry only an imageId (lmms-eval attaches the image in
    gqa_process_docs from GQA's testdev_balanced images). Validation rows come
    from the larger testdev_all pool, so the image is stored in an extra
    `image` column instead and the val50 task disables process_docs.
  * vqav2_val_lite: the pool source (merve/vqav2-small) only ships the
    consensus `multiple_choice_answer`, not the 10 human answers. We store it
    as 10 identical `answers` entries, so the VQA accuracy of a prediction is
    1 if it matches the consensus answer and 0 otherwise.

Usage (any env with `datasets` + Pillow, e.g. Bagel/.venv):
    python rag_retrieval/build_val50_datasets.py --all
    python rag_retrieval/build_val50_datasets.py --benchmark ai2d_lite
"""
import argparse
import json
import os
from pathlib import Path

import datasets
from PIL import Image

LITE_REPO = "lmms-lab/LMMs-Eval-Lite"
DATA_ROOT = Path(__file__).resolve().parent / "data"
SIMIT_ROOT = os.environ.get("SIMIT_ROOT") or str(Path(__file__).resolve().parents[1])
OUT_ROOT = Path(os.environ.get("SIMIT_DATA") or os.path.join(SIMIT_ROOT, "workdir")) / "val50"

TASKS = [
    "ai2d_lite", "chartqa_lite", "coco2017_cap_val_lite", "docvqa_val_lite", "flickr30k_test_lite",
    "gqa_lite", "infovqa_val_lite", "mmbench_cn_dev_lite", "mmbench_en_dev_lite", "nocaps_val_lite",
    "ok_vqa_val2014_lite", "refcoco_bbox_val_lite", "seedbench_lite", "textcaps_val_lite",
    "textvqa_val_lite", "vizwiz_vqa_val_lite", "vqav2_val_lite",
]


def load_images(split_dir: Path, rel):
    if isinstance(rel, list):
        return [Image.open(split_dir / r).convert("RGB") for r in rel]
    return Image.open(split_dir / rel).convert("RGB")


def build(task: str) -> None:
    split_dir = DATA_ROOT / task / "validation"
    rows = [json.loads(l) for l in open(split_dir / "metadata.jsonl") if l.strip()]
    features = datasets.load_dataset(LITE_REPO, task[: -len("_lite")], split="lite").features.copy()
    if task == "gqa_lite":
        features["image"] = datasets.Image()

    out_rows = []
    for i, row in enumerate(rows):
        rel = row.pop("image_path")
        row.pop("_dedup_hash", None)
        row["image"] = load_images(split_dir, rel)
        if task == "vqav2_val_lite":
            mca = row["multiple_choice_answer"]
            row["answers"] = [{"answer": mca, "answer_confidence": "yes", "answer_id": j + 1} for j in range(10)]
            row.setdefault("question_id", 900_000_000 + i)
            row.setdefault("image_id", 900_000_000 + i)
            row.setdefault("answer_type", "other")
            row.setdefault("question_type", "none of the above")
        if "image_path" in features:  # textcaps: Lite also has an (unused) image_path string column
            row["image_path"] = str(rel)
        out_rows.append({k: row.get(k) for k in features})

    ds = datasets.Dataset.from_list(out_rows, features=features)
    out_dir = OUT_ROOT / task
    datasets.DatasetDict({"val": ds}).save_to_disk(str(out_dir))
    print(f"{task}: {len(ds)} validation rows -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--benchmark")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()
    for task in TASKS if args.all else [args.benchmark]:
        build(task)


if __name__ == "__main__":
    main()
