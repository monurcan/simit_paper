"""Cached demo examples: real runs of the demo pipeline on the paper's figure
queries (LMMs-Eval-Lite rows), stored with both answers, the imagined demos and
the reference answer. ``--keep improved`` keeps only runs where SIMIT-ICL fixed
the base answer.

CUDA_VISIBLE_DEVICES=0 python tools/make_examples.py bagel 60 [--keep improved]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

from datasets import load_dataset  # noqa: E402
from PIL import Image  # noqa: E402

from demo_models import MODELS, run  # noqa: E402
from general_qa import _list  # noqa: E402
from simit.metrics import get_metric  # noqa: E402

VQA = "Answer the question using a single word or phrase."
UNANSWERABLE = "When the provided information is insufficient, respond with 'Unanswerable'."
# (subset, row, prompt suffix, metric, where it appears in the paper)
CANDIDATES = [
    ("ok_vqa_val2014", 492, f"{UNANSWERABLE}\n{VQA}", "vqa_accuracy", "paper Fig. (qualitative samples)"),
    ("infovqa_val", 449, VQA, "anls", "paper Fig. (qualitative samples)"),
    ("vizwiz_vqa_val", 345, f"{UNANSWERABLE}\n{VQA}", "vqa_accuracy", "paper teaser"),
    ("vizwiz_vqa_val", 47, f"{UNANSWERABLE}\n{VQA}", "vqa_accuracy", "paper appendix"),
    ("vizwiz_vqa_val", 87, f"{UNANSWERABLE}\n{VQA}", "vqa_accuracy", "paper appendix"),
] + [("ok_vqa_val2014", i, VQA, "vqa_accuracy", "paper appendix") for i in (115, 151, 210, 22, 253, 272, 291, 381,
                                                                           477, 8)]
SUBSET_NAME = {"ok_vqa_val2014": "OK-VQA", "infovqa_val": "InfographicVQA", "vizwiz_vqa_val": "VizWiz"}

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("budget", type=int)
ap.add_argument("--keep", choices=["all", "improved"], default="all")
ap.add_argument("--always", action="store_true", help="imagine even when the model is confident (UI checkbox)")
ap.add_argument("--out", default=str(HERE / "examples"))
ap.add_argument("--only", nargs="*", default=None, help="subset:row items to run, e.g. ok_vqa_val2014:492")
ap.add_argument("--image", help="a custom query image (instead of the dataset candidates)")
ap.add_argument("--question", help="the question for --image")
ap.add_argument("--reference", nargs="+", help="reference answer(s) for --image")
ap.add_argument("--metric", default="contains", help="metric for --image (free-form answers: 'contains')")
ap.add_argument("--id", help="example id (folder name) for --image")
ap.add_argument("--source", default="custom example", help="source line shown in the UI for --image")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
index_file = out / "index.json"
index = json.loads(index_file.read_text()) if index_file.exists() else []
spec = MODELS[args.model]
spec.load()
suffix_id = "_always" if args.always else ""


def jobs():
    """(example id, image, question, references, metric name, source, keep the folder's own files)"""
    if args.image:
        img = Image.open(args.image)
        img.load()
        yield (args.id + suffix_id, img, args.question, args.reference, args.metric, args.source, True)
        return
    for sub, row, suffix, metric_name, where in CANDIDATES:
        if args.only is not None and f"{sub}:{row}" not in args.only:
            continue
        r = load_dataset("lmms-lab/LMMs-Eval-Lite", sub)["lite"][row]
        yield (f"{args.model}_{sub}_{row}{suffix_id}", r["image"].convert("RGB"), f"{r['question']}\n{suffix}",
               [str(a) for a in _list(r.get("answers", r.get("answer")))], metric_name,
               f"{SUBSET_NAME.get(sub, sub)} row {row}, {where}", False)


for ex_id, image, question, refs, metric_name, source, custom in jobs():
    metric = get_metric(metric_name)
    t0 = time.time()
    greedy, demos, final, info = None, [], None, {}
    for ev in run(spec, image, question, args.budget, args.always):
        if ev[0] == "greedy":
            greedy = ev[1]
        elif ev[0] == "demo":
            demos.append(ev[1])
        elif ev[0] == "final":
            final, info = ev[1], ev[2]
    g_ok, s_ok = metric(greedy.answer, refs) >= 0.5, metric(final, refs) >= 0.5
    print(f"{ex_id}: greedy={greedy.answer!r} ({g_ok}) simit={final!r} ({s_ok}) demos={len(demos)} "
          f"{time.time() - t0:.0f}s refs={refs[:3]}", flush=True)
    if args.keep == "improved" and not (s_ok and not g_ok):
        continue
    d = out / ex_id
    if custom and d.exists():          # the user's own files stay; only earlier demo images are replaced
        for f in d.glob("demo_*.png"):
            f.unlink()
    else:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
    if custom and Path(args.image).resolve().parent == d.resolve():
        query_file = Path(args.image).name
    else:
        shown = image.convert("RGB")
        shown.thumbnail((1024, 1024))      # display copy (the run used the original)
        shown.save(d / "query.jpg", quality=90)
        query_file = "query.jpg"
    stored = []
    for i, dm in enumerate(demos):
        dm.image.save(d / f"demo_{i}.png")
        stored.append({"image": f"demo_{i}.png", "question": dm.question, "answer": dm.answer, "skill": dm.skill,
                       "verify_score": dm.verify_score, "confidence": dm.confidence})
    entry = {"id": ex_id, "query_file": query_file, "model": args.model, "budget": args.budget,
             "always": args.always, "question": question, "reference": refs[0],
             "greedy": greedy.answer, "p0": greedy.confidence, "simit": final, "greedy_ok": g_ok, "simit_ok": s_ok,
             "demos": stored, "elapsed": info.get("elapsed", time.time() - t0), "source": source}
    index = [e for e in index if e["id"] != ex_id] + [entry]
    index_file.write_text(json.dumps(index, indent=1))
