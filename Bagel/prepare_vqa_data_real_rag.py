"""
Build BAGEL vlm_sft JSONL fine-tuning corpora from the real ICL retrieval
pools (rag_retrieval/data/<lite_task>/retrieval_pool), using each
benchmark's own lmms-eval doc_to_text/doc_to_target/doc_to_visual -- the same
functions lmms_eval/models/simple/bagel_icl.py's RealSampleRetriever uses to
format RAG demo examples at eval time -- so the fine-tuning prompt format
matches exactly what the model sees during ICL/eval (e.g. "Answer the
question using a single word or phrase." suffixes, MCQ option strings, etc.),
avoiding train/eval prompt-format drift.

One fixed-random-order (seeded shuffle) JSONL per benchmark, images
materialized post doc_to_visual (e.g. refcoco's drawn bbox) into
data_root/vqa_benchmarks_real_rag/<bench>/images/. Only the first
max(milestones) samples of the shuffled order are materialized, not the
whole pool, since some pools (gqa: 171623) are far larger than any milestone
we train on.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, SIMIT_ROOT + "/lmms-eval")
sys.path.insert(0, SIMIT_ROOT + "/Bagel")

from PIL import Image  # noqa: E402
from lmms_eval.tasks import TaskManager, get_task_dict  # noqa: E402
from lmms_eval.models.simple.bagel_icl import _rag_fallback_answer  # noqa: E402
from data.data_utils import pil_img2rgb  # noqa: E402

RAG_ROOT = Path(SIMIT_ROOT + "/rag_retrieval/data")
OUT_ROOT = Path(SIMIT_ROOT + "/Bagel/data_root/vqa_benchmarks_real_rag")
SEED = 0

# our short bench name -> lmms-eval lite task name
BENCH_TASK = {
    "ai2d": "ai2d_lite",
    "chartqa": "chartqa_lite",
    "coco2017_cap_val": "coco2017_cap_val_lite",
    "docvqa_val": "docvqa_val_lite",
    "flickr30k_test": "flickr30k_test_lite",
    "gqa": "gqa_lite",
    "infovqa_val": "infovqa_val_lite",
    "mmbench_cn_dev": "mmbench_cn_dev_lite",
    "mmbench_en_dev": "mmbench_en_dev_lite",
    "nocaps_val": "nocaps_val_lite",
    "ok_vqa_val2014": "ok_vqa_val2014_lite",
    "refcoco_bbox_val": "refcoco_bbox_val_lite",
    "seedbench": "seedbench_lite",
    "textcaps_val": "textcaps_val_lite",
    "textvqa_val": "textvqa_val_lite",
    "vizwiz_vqa_val": "vizwiz_vqa_val_lite",
    "vqav2_val": "vqav2_val_lite",
    # mathvista_testmini: retrieval_pool is empty (0 rows), no real-data
    # corpus exists for it -- intentionally excluded.
}

MILESTONES = [25, 50, 75, 100, 125, 150, 250, 500, 1000, 2000]


def resolve_answer(task_obj, doc):
    cfg_target = task_obj.config.doc_to_target
    if isinstance(cfg_target, str) and cfg_target not in task_obj.features:
        return _rag_fallback_answer(doc)
    target = task_obj.doc_to_target(doc)
    if isinstance(target, list):
        return str(target[0]) if target else ""
    return str(target)


def load_images(pool_dir, image_path_field):
    paths = image_path_field if isinstance(image_path_field, list) else [image_path_field]
    imgs = [Image.open(pool_dir / p) for p in paths]
    for im in imgs:
        im.load()
    return imgs, isinstance(image_path_field, list)


def build_bench(bench_name, task_name, task_manager):
    pool_dir = RAG_ROOT / task_name / "retrieval_pool"
    meta_path = pool_dir / "metadata.jsonl"
    rows = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
    pool_size = len(rows)
    if pool_size == 0:
        print(f"{bench_name}: EMPTY pool, skipping")
        return None

    rng = random.Random(SEED)
    order = list(range(pool_size))
    rng.shuffle(order)

    milestones = sorted(set(m for m in MILESTONES if m <= pool_size))
    if not milestones:
        milestones = [pool_size]
    max_n = milestones[-1]

    task_dict = get_task_dict([task_name], task_manager)
    obj = task_dict[task_name]
    if isinstance(obj, tuple):
        _, obj = obj
    obj.model_name = "bagel_icl"
    obj._prepare_model_specific_config()

    out_dir = OUT_ROOT / bench_name
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{bench_name}_real_rag.jsonl"

    n_written = 0
    n_failed = 0
    with jsonl_path.open("w") as f:
        for out_idx, pool_idx in enumerate(order[:max_n]):
            row = rows[pool_idx]
            try:
                loaded_imgs, was_list = load_images(pool_dir, row["image_path"])
                doc = dict(row)
                doc["image"] = loaded_imgs if was_list else loaded_imgs[0]

                question = obj.doc_to_text(doc)
                answer = resolve_answer(obj, doc)
                if not answer:
                    n_failed += 1
                    continue
                final_imgs = [pil_img2rgb(im) for im in obj.doc_to_visual(doc)]
                if not final_imgs:
                    n_failed += 1
                    continue
            except Exception as e:
                print(f"{bench_name}: failed pool row {pool_idx}: {e}")
                n_failed += 1
                continue

            rel_paths = []
            for j, im in enumerate(final_imgs):
                out_img_path = img_dir / f"{out_idx:07d}_{j}.png"
                im.save(out_img_path)
                rel_paths.append(str(out_img_path.relative_to(OUT_ROOT)))

            image_field = rel_paths[0] if len(rel_paths) == 1 else rel_paths
            image_tags = "<image>" * len(rel_paths)
            record = {
                "image": image_field,
                "conversations": [
                    {"from": "human", "value": f"{image_tags}\n{question}"},
                    {"from": "gpt", "value": str(answer)},
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            n_written += 1

    print(f"{bench_name}: pool_size={pool_size} milestones={milestones} written={n_written} failed={n_failed} -> {jsonl_path}")
    return {"pool_size": pool_size, "milestones": milestones, "written": n_written}


def main():
    task_manager = TaskManager()
    summary = {}
    only = sys.argv[1:] if len(sys.argv) > 1 else list(BENCH_TASK.keys())
    for bench_name in only:
        task_name = BENCH_TASK[bench_name]
        summary[bench_name] = build_bench(bench_name, task_name, task_manager)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
