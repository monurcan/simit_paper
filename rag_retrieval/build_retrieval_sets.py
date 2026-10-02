#!/usr/bin/env python
"""Build per-benchmark real-sample retrieval pools for the RAG-ICL evaluation.

For each benchmark, the pool is "the big/undistilled HF dataset (all splits
combined) minus the 500 samples used in lmms-lab/LMMs-Eval-Lite", with 50
samples additionally carved out as a held-out validation set (unused by this
project, just saved for later). mathvista_testmini is the one exception: no
Lite dedup (MathVista's official testmini/test splits are already disjoint),
retrieval pool = MathVista's `test` split directly.

Run under this project's venv (needs `datasets`; does NOT need transformers/
faiss, so the Bagel eval venv would also work, but we keep everything under
rag_retrieval/.venv for a single consistent environment):
    ./.venv/bin/python build_retrieval_sets.py --benchmark ai2d_lite
    ./.venv/bin/python build_retrieval_sets.py --all

Output layout (per benchmark task name, e.g. "ai2d_lite"):
    data/<task>/retrieval_pool/{images/*.png, metadata.jsonl}
    data/<task>/validation/{images/*.png, metadata.jsonl}
Each metadata.jsonl line is the original HF row's fields (JSON-safe), plus
"image_path" (relative to that split dir) and "_dedup_hash" (for debugging).
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from datasets import get_dataset_split_names, load_dataset
from PIL import Image

DATA_ROOT = Path(__file__).parent / "data"
LITE_REPO = "lmms-lab/LMMs-Eval-Lite"
VAL_SIZE = 50
SEED = 0
SMALL_SIZE = (32, 32)
# Mean-absolute-pixel-difference (0-255 scale) threshold for "same image".
# Confirmed empirically (MMBench_EN): the Lite and lmms-lab-encoder/lmms-lab
# repos can store the *same* underlying photo re-encoded slightly
# differently (identical question/answer/dimensions, MD5-different pixels) --
# an exact hash silently under-matched there (35/500 instead of 500/500).
# Observed MAD for that true-duplicate pair was ~0.04, vs 50-224 for 20
# genuinely different images from the same dataset -- huge margin either side
# of this threshold.
MAD_THRESHOLD = 10.0

# benchmark task name (matches lmms-eval task strings, e.g. "ai2d_lite") -> big-repo spec.
# Lite's `dataset_name` kwarg is the task name with the "_lite" suffix stripped
# (mirrors the convention already used by bagel_icl.py::SyntheticSampleRetriever).
#
# id_field: when present, this column is a stable identifier shared verbatim
# between the Lite split and the big repo (confirmed empirically for every
# entry below: all 500 Lite ids found in the big repo's id set) -- dedup uses
# an exact match on this field, which is 100% reliable. Where absent (ai2d,
# chartqa), no such id survives into Lite, so dedup falls back to
# LiteDedupIndex (tolerant image match + exact question match) -- see its
# docstring for why *both* signals are needed together there.
BIG_REPO = {
    "refcoco_bbox_val_lite": dict(repo="lmms-lab-encoder/RefCOCO", id_field="question_id"),
    # test split (test2017) is COCO's official held-out set: answer=['None']
    # placeholder for every row, no public captions -- skip it outright.
    "coco2017_cap_val_lite": dict(repo="lmms-lab-encoder/COCO-Caption2017", id_field="question_id", splits=["val"]),
    # test split's reference_strs are all the literal placeholder "None"
    # (TextCaps' held-out challenge-leaderboard split, no public captions) --
    # skip loading/iterating it at all rather than relying on
    # has_usable_answer() to filter it out row-by-row after the fact.
    "textcaps_val_lite": dict(repo="lmms-lab-encoder/TextCaps", id_field="question_id", splits=["train", "val"]),
    "flickr30k_test_lite": dict(repo="lmms-lab-encoder/flickr30k", id_field="img_id"),
    "ai2d_lite": dict(repo="lmms-lab-encoder/ai2d"),
    "nocaps_val_lite": dict(repo="lmms-lab-encoder/NoCaps", id_field="image_id"),
    # Full lmms-lab-encoder/VQAv2 validation+testdev+test is ~770k rows and
    # blew well past 100GB (84GB staged + 113GB raw HF cache, still mid-run)
    # -- switched to this much smaller pre-sampled mirror instead. It has no
    # question_id column, so dedup falls back to LiteDedupIndex (tolerant
    # image + exact question match); confirmed 60/500 distilled samples
    # actually overlap with this smaller pool (the rest of the 500 simply
    # aren't in this subsample, nothing to exclude for those).
    "vqav2_val_lite": dict(repo="merve/vqav2-small"),
    "chartqa_lite": dict(repo="lmms-lab-encoder/ChartQA"),
    "seedbench_lite": dict(repo="lmms-lab-encoder/SEED-Bench", id_field="question_id"),
    "vizwiz_vqa_val_lite": dict(repo="lmms-lab-encoder/VizWiz-VQA", id_field="question_id"),
    "infovqa_val_lite": dict(repo="lmms-lab-encoder/DocVQA", config="InfographicVQA", id_field="questionId"),
    # lmms-lab/MMBench_EN's own column is lowercase "l2-category", but the
    # Lite dataset (and lmms_eval/tasks/mmbench/en_utils.py::mmbench_doc_to_text,
    # which RealSampleRetriever reuses for demo construction) expect
    # "L2-category" -- confirmed by a live KeyError during verification.
    "mmbench_en_dev_lite": dict(
        repo="lmms-lab/MMBench_EN",
        text_fields=("question", "A", "B", "C", "D", "answer"),
        rename_fields={"l2-category": "L2-category"},
    ),
    "mmbench_cn_dev_lite": dict(repo="lmms-lab/MMBench_CN", text_fields=("question", "A", "B", "C", "D", "answer")),
    # test split's answers are 10x blank strings (VQA-Challenge held-out
    # split, no public ground truth) -- skip it outright.
    "textvqa_val_lite": dict(repo="lmms-lab-encoder/textvqa", id_field="question_id", splits=["train", "validation"]),
    "ok_vqa_val2014_lite": dict(repo="lmms-lab-encoder/OK-VQA", id_field="question_id"),
    "docvqa_val_lite": dict(repo="lmms-lab-encoder/DocVQA", config="DocVQA", id_field="questionId"),
    # gqa_lite and mathvista_testmini are special-cased below (see load_gqa_rows /
    # is_mathvista): gqa's Lite rows carry only an imageId (image looked up from a
    # separate images config); mathvista draws from a different repo entirely with
    # no Lite dedup needed.
}

ANSWER_FIELDS = ["answer", "answers", "annotations_captions", "caption", "reference_strs", "caption_str", "multiple_choice_answer"]


def log(msg: str) -> None:
    print(f"[build_retrieval_sets] {msg}", file=sys.stderr, flush=True)


def lite_name_for(task: str) -> str:
    return task[: -len("_lite")] if task.endswith("_lite") else task


def _is_real_answer_text(v: str) -> bool:
    """Rejects both blank strings and the literal placeholder "None" that
    several big repos (COCO-Caption2017's test2017 split, TextCaps' held-out
    test split) use to mark rows with no public ground truth -- confirmed via
    build_prompt_viewer.py's dump: this string was silently ending up as a
    RAG-ICL demo's "Answer:" text."""
    s = v.strip()
    return bool(s) and s.lower() != "none"


def has_usable_answer(row: dict) -> bool:
    for f in ANSWER_FIELDS:
        v = row.get(f)
        if isinstance(v, str) and _is_real_answer_text(v):
            return True
        if isinstance(v, list) and any(
            _is_real_answer_text(x) if isinstance(x, str) else (isinstance(x, (int, float)) and not isinstance(x, bool)) for x in v
        ):
            # A non-empty list isn't enough on its own -- textvqa's held-out
            # test split has answers=['', '', ...] (10 blank strings), which
            # passed the old "len(v) > 0" check despite having no real content.
            return True
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return True
    return False


def _sanitize_filename(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:200]


def as_image_list(img) -> list:
    """Some tasks (e.g. seedbench) store multiple images per row as a list
    (matching that task's own doc_to_visual, which expects a list). Normalize
    to a list either way so downstream code has one code path."""
    return list(img) if isinstance(img, list) else [img]


def small_arr(img) -> np.ndarray:
    first = as_image_list(img)[0]
    return np.asarray(first.convert("RGB").resize(SMALL_SIZE, Image.BICUBIC), dtype=np.float32)


class LiteDedupIndex:
    """Tolerant (image, text-signature) duplicate detector for one
    benchmark's 500 Lite rows. Images are compared via mean-absolute-
    difference on a 32x32 thumbnail (see MAD_THRESHOLD), not an exact hash --
    required because the Lite and big/-encoder repos don't always store
    byte-identical pixels for the same underlying sample (confirmed on
    MMBench). The text signature (default: just "question") must still match
    *exactly*: benchmarks like ai2d ask many different questions about the
    same image, so image similarity alone would wrongly exclude legitimate
    non-distilled rows that merely happen to share an image with a
    distilled one.

    text_fields lets a benchmark use a richer signature than bare question
    text where needed -- MMBench reuses generic question templates (e.g.
    "Extract text from the image") across many genuinely different images,
    so (question, A, B, C, D, answer) is used there instead of question
    alone (confirmed empirically: question-only gave 1915-1956 false
    "duplicates" on MMBench_EN/CN; the fuller tuple gives 485-501, in line
    with the expected 500)."""

    def __init__(self, lite_rows: list, text_fields=("question",)):
        self.text_fields = text_fields
        self.by_text: Dict[tuple, list] = {}
        no_text = []
        for row in lite_rows:
            arr = small_arr(row["image"])
            key = self._text_key(row)
            if key is not None:
                self.by_text.setdefault(key, []).append(arr)
            else:
                no_text.append(arr)
        self.no_text_arrs = np.stack(no_text) if no_text else None

    def _text_key(self, row: dict):
        q = row.get(self.text_fields[0])
        if not (isinstance(q, str) and q):
            return None
        return tuple(row.get(f) for f in self.text_fields)

    def is_duplicate(self, row: dict) -> bool:
        img = row.get("image")
        if img is None:
            return False
        arr = small_arr(img)
        key = self._text_key(row)
        if key is not None:
            candidates = self.by_text.get(key)
            if not candidates:
                return False
            return any(float(np.abs(arr - c).mean()) < MAD_THRESHOLD for c in candidates)
        if self.no_text_arrs is None:
            return False
        diffs = np.abs(self.no_text_arrs - arr[None]).mean(axis=(1, 2, 3))
        return bool((diffs < MAD_THRESHOLD).any())


def build_dedup_fn(task: str, spec: dict):
    """Returns is_duplicate_fn(row) -> bool for this benchmark: exact match on
    spec['id_field'] when the repo has one (100% reliable, no image
    comparison needed), else LiteDedupIndex's tolerant image+text match
    (spec['text_fields'], default just "question").

    NOTE on MMBench: its "index" column looked like a promising id_field
    (MMBench_EN: raw index found 500/500 in dev) but turned out unreliable --
    MMBench_CN's dev split index is a locally-reassigned dense 0..4328
    sequence, NOT the original master benchmark index, so "matches" there
    were pure coincidental range-overlap (confirmed by direct question-text
    comparison: only 2/500 "matches" actually had matching question text).
    MMBench also reuses generic question templates (e.g. "Extract text from
    the image") across many genuinely different images, so plain
    LiteDedupIndex with text_fields=("question",) alone over-matched too
    (1915-1956 "duplicates" instead of ~500). The fix used here:
    text_fields=("question","A","B","C","D","answer") -- the full MCQ
    content, not just the stem -- which is discriminative enough in
    practice (confirmed: 501/500 for CN, 485/500 for EN; the small EN gap is
    fine, it just means those Lite rows' specific rotation isn't literally
    stored as a dev/test row to exclude, so there's nothing to leak)."""
    name = lite_name_for(task)
    lite = load_dataset(LITE_REPO, name, split="lite")
    id_field = spec.get("id_field")
    if id_field:
        ids = {row[id_field] for row in lite}
        log(f"'{task}': {len(ids)} dedup ids from {len(lite)} Lite rows (exact '{id_field}' match)")
        return lambda row: row.get(id_field) in ids

    text_fields = spec.get("text_fields", ("question",))
    index = LiteDedupIndex(list(lite), text_fields=text_fields)
    n_no_text = 0 if index.no_text_arrs is None else len(index.no_text_arrs)
    log(f"'{task}': no id_field -- dedup index from {len(lite)} Lite rows ({len(index.by_text)} unique {text_fields} signatures, {n_no_text} without one)")
    return index.is_duplicate


def json_safe(value):
    if isinstance(value, Image.Image):
        raise TypeError("image field must be popped before json_safe()")
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)  # fallback for exotic HF feature types


def iter_big_rows(spec: dict):
    repo, config = spec["repo"], spec.get("config")
    splits = spec.get("splits") or (get_dataset_split_names(repo, config) if config else get_dataset_split_names(repo))
    rename_fields = spec.get("rename_fields")
    log(f"loading '{repo}' config={config} splits={splits}")
    for split in splits:
        ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        log(f"  split '{split}': {len(ds)} rows")
        for row in ds:
            if rename_fields:
                row = dict(row)
                for old, new in rename_fields.items():
                    if old in row:
                        row[new] = row.pop(old)
            yield row


def stage_and_split(task: str, row_iter, is_duplicate_fn, val_size: int = VAL_SIZE, seed: int = SEED, image_key_fn=None):
    """is_duplicate_fn(row) -> bool: True if this big-repo row is one of the
    500 Lite-distilled samples (see LiteDedupIndex.is_duplicate / GQA's exact
    imageId+question check in build_benchmark).

    image_key_fn(row) -> hashable key: when given, rows sharing the same key
    (e.g. GQA's imageId -- many questions per image) reuse one staged image
    file instead of each writing their own copy. Without this, a
    172k-row/398-unique-image benchmark like GQA writes 172k near-duplicate
    PNGs, which is what actually tripped an inode-quota failure during this
    project's build (see conversation/incident notes) well before disk space
    itself was a concern."""
    task_dir = DATA_ROOT / task
    staging = task_dir / "_staging"
    staging_images = staging / "images"
    if staging.exists():
        shutil.rmtree(staging)
    staging_images.mkdir(parents=True)

    n_seen = n_no_answer = n_dup = n_kept = n_bad = 0
    staged_by_image_key: Dict[Any, Any] = {}
    meta_path = staging / "metadata.jsonl"
    row_iterator = iter(row_iter)
    with open(meta_path, "w") as meta_f:
        while True:
            # A corrupt/truncated image in the source dataset can raise
            # *during* row decoding itself (datasets' own Arrow->Python
            # formatting), i.e. while advancing the iterator -- not just
            # while we're processing an already-yielded row. A plain `for
            # row in row_iter` can't catch that from inside the loop body,
            # so iterate manually and guard the advance step too (confirmed
            # necessary: this is what killed the NoCaps build even with the
            # try/except below already in place).
            try:
                row = next(row_iterator)
            except StopIteration:
                break
            except Exception as e:
                n_bad += 1
                n_seen += 1
                log(f"  WARNING: skipping unreadable row during iteration ({e!r})")
                continue
            n_seen += 1
            try:
                if not has_usable_answer(row):
                    n_no_answer += 1
                    continue
                img = row.get("image")
                if img is None:
                    continue
                images = as_image_list(img)

                if is_duplicate_fn(row):
                    n_dup += 1
                    continue

                img_key = image_key_fn(row) if image_key_fn is not None else None
                if img_key is not None and img_key in staged_by_image_key:
                    rel_path = staged_by_image_key[img_key]
                else:
                    base = _sanitize_filename(str(img_key)) if img_key is not None else f"{n_kept:07d}"
                    # Preserve list-vs-scalar based on the *source* field type,
                    # not len(images) -- some schemas (seedbench) always wrap
                    # "image" in a list even for a single image, because their
                    # doc_to_visual unconditionally iterates it. Collapsing a
                    # length-1 list to a bare path here broke that at eval
                    # time (confirmed: TypeError, 'PngImageFile' not iterable).
                    if isinstance(img, list):
                        rel_path = [f"images/{base}_{j}.png" for j in range(len(images))]
                        for j, im in enumerate(images):
                            im.convert("RGB").save(staging_images / f"{base}_{j}.png")
                    else:
                        rel_path = f"images/{base}.png"
                        images[0].convert("RGB").save(staging_images / f"{base}.png")
                    if img_key is not None:
                        staged_by_image_key[img_key] = rel_path
            except Exception as e:
                # e.g. a truncated/corrupt image in the source dataset --
                # confirmed to happen for real (one bad row killed the whole
                # NoCaps build). Skip this row rather than lose the entire
                # benchmark's build to one bad upstream record.
                n_bad += 1
                log(f"  WARNING: skipping row {n_seen} ({e!r})")
                continue

            record = {k: v for k, v in row.items() if k != "image"}
            record = json_safe(record)
            record["image_path"] = rel_path
            meta_f.write(json.dumps(record) + "\n")
            n_kept += 1
            if n_kept % 2000 == 0:
                log(f"  ... {n_kept} kept so far ({n_seen} seen, {n_dup} dup, {n_no_answer} no-answer, {n_bad} bad, {len(staged_by_image_key)} unique images staged)")

    log(f"'{task}': seen={n_seen} no_answer={n_no_answer} dup_with_lite={n_dup} bad={n_bad} kept={n_kept} unique_images={len(staged_by_image_key) or n_kept}")

    with open(meta_path) as f:
        records = [json.loads(line) for line in f]
    rng = random.Random(seed)
    rng.shuffle(records)

    val_records, pool_records = records[:val_size], records[val_size:]
    # Global (across both splits) map of staged-path -> already-materialized
    # destination path(s). Each unique staged image is MOVED to its first
    # destination (no duplication on disk at all) and only ever COPIED for
    # genuine additional references to that same image (e.g. GQA rows
    # sharing an imageId, or the rare case where one image's rows land in
    # both validation and retrieval_pool) -- never duplicated wholesale.
    materialized: Dict[Any, Any] = {}
    for split_name, split_records in [("validation", val_records), ("retrieval_pool", pool_records)]:
        split_dir = task_dir / split_name
        split_images = split_dir / "images"
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_images.mkdir(parents=True)
        seen_in_split: Dict[Any, Any] = {}
        with open(split_dir / "metadata.jsonl", "w") as out_f:
            for rec in split_records:
                old_rel = rec["image_path"]
                cache_key = tuple(old_rel) if isinstance(old_rel, list) else old_rel
                if cache_key in seen_in_split:
                    new_rel = seen_in_split[cache_key]
                else:
                    idx = len(seen_in_split)
                    if isinstance(old_rel, list):
                        new_rel = [f"images/{idx:07d}_{j}.png" for j in range(len(old_rel))]
                    else:
                        new_rel = f"images/{idx:07d}.png"
                    new_rel_list = new_rel if isinstance(new_rel, list) else [new_rel]
                    old_rel_list = old_rel if isinstance(old_rel, list) else [old_rel]
                    dest_paths = [str(split_images / Path(r).name) for r in new_rel_list]

                    if cache_key in materialized:
                        for prev_dest, dest in zip(materialized[cache_key], dest_paths):
                            shutil.copy2(prev_dest, dest)
                    else:
                        for old_r, dest in zip(old_rel_list, dest_paths):
                            shutil.move(str(staging_images / Path(old_r).name), dest)
                        materialized[cache_key] = dest_paths
                    seen_in_split[cache_key] = new_rel
                rec["image_path"] = new_rel
                out_f.write(json.dumps(rec) + "\n")
        log(f"'{task}': wrote {len(split_records)} rows ({len(seen_in_split)} unique images) to {split_dir}")

    shutil.rmtree(staging, ignore_errors=True)


def build_benchmark(task: str):
    if task == "mathvista_testmini":
        row_iter = iter_big_rows(dict(repo="AI4Math/MathVista", splits=["test"]))
        stage_and_split(task, row_iter, is_duplicate_fn=lambda row: False)
        return
    elif task == "gqa_lite":
        from build_gqa_retrieval_set import iter_gqa_rows

        name = lite_name_for(task)
        lite = load_dataset(LITE_REPO, name, split="lite")
        # GQA's Lite rows carry no embedded image (only an imageId reference,
        # see build_gqa_retrieval_set.py) -- exact (imageId, question) match
        # is reliable here, no image comparison needed/possible.
        gqa_keys = {(row["imageId"], row.get("question")) for row in lite}
        log(f"'{task}': {len(gqa_keys)} dedup keys from {len(lite)} Lite rows (exact imageId+question)")
        row_iter = iter_gqa_rows()
        stage_and_split(
            task,
            row_iter,
            is_duplicate_fn=lambda row: (row["imageId"], row.get("question")) in gqa_keys,
            image_key_fn=lambda row: row["imageId"],
        )
        return
    else:
        if task not in BIG_REPO:
            raise ValueError(f"Unknown benchmark task '{task}'. Known: {sorted(BIG_REPO) + ['gqa_lite', 'mathvista_testmini']}")
        spec = BIG_REPO[task]
        is_duplicate_fn = build_dedup_fn(task, spec)
        row_iter = iter_big_rows(spec)
        stage_and_split(task, row_iter, is_duplicate_fn=is_duplicate_fn)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark", help="single task name, e.g. ai2d_lite")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    tasks = sorted(BIG_REPO) + ["gqa_lite", "mathvista_testmini"] if args.all else [args.benchmark]
    for task in tasks:
        log(f"=== building '{task}' ===")
        build_benchmark(task)


if __name__ == "__main__":
    main()
