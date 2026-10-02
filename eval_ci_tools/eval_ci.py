#!/usr/bin/env python3
"""
Eval-set bootstrap confidence intervals for BAGEL lmms-eval outputs.

Scans one or more eval-output directories (each containing a
`models__BAGEL-7B-MoT/*_samples_<bench>_lite.jsonl` file per benchmark run in
that directory), computes a bootstrap mean/std/95%CI per benchmark from the
already-saved per-example predictions (no reruns needed), and aggregates to
the project's standard 5-domain grouping.

Usage:
    python eval_ci.py <dir1> [<dir2> ...]
    python eval_ci.py --glob "benchmark_eval_outputs/ok_vqa_val2014_nodf4scale_*"
    python eval_ci.py <dir> --json out.json
    python eval_ci.py <dir> --mmbench-include-rotations   # see note below

Each <dir> is one checkpoint/run (e.g. a single LoRA checkpoint's eval output
folder) and may contain any subset of the 17 benchmarks; only the ones found
are scored and only their domains get a domain-level number.

What this does NOT capture: training-run-to-run variance (different LoRA
init/data-shuffle seed). Every number here is from ONE training run; this
tool only quantifies eval-set sampling variance (how much a score would
wobble depending on which examples landed in the eval subset), via bootstrap
resampling of the saved per-example predictions. Treat these CIs as a lower
bound on true uncertainty.

mmbench (en/cn) note: the official MMBench scorer uses circular evaluation —
each question may appear at multiple rotated option-orderings encoded as
index, index+1_000_000, index+2_000_000, ... and only counts index<1_000_000
rows (~26% of a 500-row "lite" sample) toward the official score, treating
rotations only as a consistency check. By default this tool reproduces that
official convention. Pass --mmbench-include-rotations to instead score every
row (every rotation) independently as its own question -- useful if you
specifically want to include VLM rotation-sensitivity as signal rather than
as a discarded consistency check. The two modes give visibly different
numbers; know which one you're using.

CIDEr note: computed via the same PTBTokenizer + pycocoevalcap pipeline
lmms-eval itself uses (verified to reproduce official cached scores to full
float precision). The per-image CIDEr array is computed once per benchmark
run (fixing corpus-level IDF weights), then bootstrap-resampled -- the
standard "fixed corpus statistics" approximation, not a full IDF refit per
resample.
"""
import argparse
import glob as globmod
import json
import math
import random
import string
import sys

# ---------------------------------------------------------------------------
# Benchmark registry: metric type, per-example field, and domain membership.
# ---------------------------------------------------------------------------

ACC_FIELD = {
    "ok_vqa_val2014": "exact_match",
    "vizwiz_vqa_val": "exact_match",
    "gqa": "exact_match",
    "vqav2_val": "exact_match",
    "textvqa_val": "exact_match",
    "ai2d": "exact_match",
    "chartqa": "relaxed_overall",
    "docvqa_val": "anls",
    "infovqa_val": "anls",
    "seedbench": "seed_all",  # per-row {"pred":..., "answer":...}, not a plain float
}

CIDER_FIELD = {
    "nocaps_val": "nocaps_CIDEr",
    "flickr30k_test": "flickr_CIDEr",
    "textcaps_val": "textcaps_CIDEr",
    "refcoco_bbox_val": "refcoco_CIDEr",
    "coco2017_cap_val": "coco_CIDEr",
}

MMBENCH = ["mmbench_en_dev", "mmbench_cn_dev"]

ALL_BENCHES = list(ACC_FIELD) + list(CIDER_FIELD) + MMBENCH

DOMAINS = {
    "Visual Question Answering": ["ok_vqa_val2014", "vizwiz_vqa_val", "gqa", "vqav2_val", "textvqa_val"],
    "Doc & Infographic Understanding": ["chartqa", "docvqa_val", "infovqa_val"],
    "Image Understanding & Captioning": list(CIDER_FIELD),
    "Multi-discipline": ["mmbench_en_dev", "mmbench_cn_dev", "seedbench"],
    "Math & Science": ["ai2d"],
}

# Domains whose member scores are averaged raw (no x100 rescale). Everything
# else is a [0,1]-bounded metric rescaled to a 0-100 scale before averaging,
# per this project's established convention (see Bagel/BENCHMARK_METRIC_BOUNDS.md).
RAW_DOMAINS = {"Image Understanding & Captioning"}


# ---------------------------------------------------------------------------
# Loading + scoring
# ---------------------------------------------------------------------------

def find_jsonl(directory, bench):
    """Locate this benchmark's per-example predictions file inside `directory`."""
    task = f"{bench}_lite"
    matches = globmod.glob(f"{directory}/**/models__BAGEL-7B-MoT/*_samples_{task}.jsonl", recursive=True)
    if not matches:
        matches = globmod.glob(f"{directory}/**/*_samples_{task}.jsonl", recursive=True)
    return matches[0] if matches else None


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def can_infer_option(answer, num_choice=5):
    """MMBench's own answer-letter extractor (mmbench_evals.py), reimplemented
    to score directly from saved raw prediction text. Verified to reproduce
    lmms-eval's official cached mmbench score to full float precision."""
    choices = string.ascii_uppercase[:num_choice]
    if "Failed to obtain answer via API" in answer:
        return False
    # Normalize full-width punctuation (e.g. Chinese "。") to ASCII so
    # mmbench_cn responses aren't spuriously unparseable -- confirmed this
    # session as the actual cause of a real discrepancy vs the cached score.
    answer = answer.translate(str.maketrans("。，：）（", ".,:)("))

    def count(splits, choices="ABCD", prefix="", suffix=""):
        return sum(1 for c in choices if prefix + c + suffix in splits)

    splits = [x.strip() for x in answer.split()]
    if count(splits, choices) == 1:
        for ch in choices:
            if "A" in splits and len(splits) > 3:
                break
            if ch in splits:
                return ch
    tups = [("", "."), ("", ","), ("", ":"), ("", ")"), ("", ")."), ("(", ")"), ("(", ")."),
            (":", ""), (":", ","), (":", "."), (":", ")"), (":", ").")]
    for prefix, suffix in tups:
        if count(splits, choices, prefix=prefix, suffix=suffix) == 1:
            for ch in choices:
                if prefix + ch + suffix in splits:
                    return ch
    return False


def bootstrap_std(scores, B=1000, seed=0):
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += scores[rng.randrange(n)]
        means.append(s / n)
    mean_of_means = sum(means) / B
    var = sum((m - mean_of_means) ** 2 for m in means) / (B - 1)
    return sum(scores) / n, math.sqrt(var)


def acc_scores(rows, field):
    if field == "seed_all":
        return [1.0 if r[field]["pred"] == r[field]["answer"] else 0.0 for r in rows]
    return [float(r[field]) for r in rows]


def mmbench_scores(rows, include_rotations):
    if not include_rotations:
        rows = [r for r in rows if r["gpt_eval_score"]["index"] < 1_000_000]
    out = []
    for r in rows:
        g = r["gpt_eval_score"]
        pred = can_infer_option(g["prediction"])
        out.append(1.0 if pred == g["answer"] else 0.0)
    return out


def cider_per_image_scores(rows, field):
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import Cider, COCOEvalCap
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    results = []
    for r in rows:
        c = r[field]
        iid = c.get("image_id", c.get("ann_id", c.get("id")))
        results.append({"answer": c["answer"], "pred": c["pred"], "image_id": iid})

    dataset = {"annotations": [], "images": []}
    stored = []
    idx = 0
    for result in results:
        stored.append({"image_id": result["image_id"], "caption": result["pred"]})
        for a in result["answer"]:
            dataset["annotations"].append({"image_id": result["image_id"], "caption": a, "id": idx})
            idx += 1
        dataset["images"].append({"id": result["image_id"]})

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    res_coco = coco.loadRes(stored)
    ev = COCOEvalCap(coco, res_coco)
    img_ids = ev.params["image_id"]
    gts = {i: ev.coco.imgToAnns[i] for i in img_ids}
    res = {i: ev.cocoRes.imgToAnns[i] for i in img_ids}
    tok = PTBTokenizer()
    gts = tok.tokenize(gts)
    res = tok.tokenize(res)
    _, per_image = Cider().compute_score(gts, res)
    return list(per_image)


def score_benchmark(directory, bench, mmbench_include_rotations=False, B=1000):
    """Returns (mean, std, n_examples) or (None, None, None) if not found."""
    path = find_jsonl(directory, bench)
    if path is None:
        return None, None, None
    rows = load_rows(path)
    if bench in ACC_FIELD:
        scores = acc_scores(rows, ACC_FIELD[bench])
    elif bench in MMBENCH:
        scores = mmbench_scores(rows, mmbench_include_rotations)
    elif bench in CIDER_FIELD:
        scores = cider_per_image_scores(rows, CIDER_FIELD[bench])
    else:
        raise ValueError(f"unknown benchmark: {bench}")
    if not scores:
        return None, None, 0
    mean, std = bootstrap_std(scores, B=B)
    return mean, std, len(scores)


# ---------------------------------------------------------------------------
# Directory-level + domain-level aggregation
# ---------------------------------------------------------------------------

def scale(bench, value):
    if bench in MMBENCH:
        return value * 100
    if bench in CIDER_FIELD:
        return value  # raw, no rescale
    return value * 100  # [0,1]-bounded accuracy-style metrics


def score_directory(directory, benches=None, mmbench_include_rotations=False, B=1000):
    """Scores every known benchmark found in `directory`. Returns
    {bench: {"mean":..., "std":..., "n":...}} for benches actually present."""
    benches = benches or ALL_BENCHES
    out = {}
    for bench in benches:
        mean, std, n = score_benchmark(directory, bench, mmbench_include_rotations, B)
        if mean is not None:
            out[bench] = {"mean": mean, "std": std, "n": n}
    return out


def domain_aggregate(bench_results):
    """bench_results: {bench: {"mean":..., "std":...}}. Returns
    {domain: {"mean":..., "std":..., "n_benchmarks":..., "benchmarks": [...]}}
    for domains with at least one present benchmark."""
    out = {}
    for domain, benches in DOMAINS.items():
        present = [b for b in benches if b in bench_results]
        if not present:
            continue
        means = [scale(b, bench_results[b]["mean"]) for b in present]
        stds = [scale(b, bench_results[b]["std"]) for b in present]
        dmean = sum(means) / len(means)
        dstd = math.sqrt(sum(s ** 2 for s in stds)) / len(means)
        out[domain] = {
            "mean": dmean,
            "std": dstd,
            "ci95": 1.96 * dstd,
            "n_benchmarks": len(present),
            "benchmarks": present,
            "missing": [b for b in benches if b not in bench_results],
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="Eval-output directories to score (one per checkpoint/run)")
    ap.add_argument("--glob", help="Shell glob pattern for directories, alternative to listing them")
    ap.add_argument("--benches", nargs="*", help="Restrict to these benchmarks (default: all 17 found)")
    ap.add_argument("--mmbench-include-rotations", action="store_true",
                     help="Score every mmbench row independently instead of the official index<1e6-only convention")
    ap.add_argument("--bootstrap-n", type=int, default=1000, help="Bootstrap resamples per benchmark (default 1000)")
    ap.add_argument("--json", help="Write full results (per-benchmark + per-domain) to this JSON file")
    args = ap.parse_args()

    dirs = list(args.dirs)
    if args.glob:
        dirs += sorted(globmod.glob(args.glob))
    if not dirs:
        ap.error("provide at least one directory or --glob")

    all_out = {}
    for d in dirs:
        print(f"\n=== {d} ===")
        bench_results = score_directory(d, args.benches, args.mmbench_include_rotations, args.bootstrap_n)
        if not bench_results:
            print("  (no known benchmark eval logs found here)")
            all_out[d] = {"benchmarks": {}, "domains": {}}
            continue

        print(f"  {'benchmark':22s} {'mean':>10s} {'std':>8s} {'95% CI':>16s} {'n':>6s}")
        for bench, r in bench_results.items():
            v = scale(bench, r["mean"])
            s = scale(bench, r["std"])
            print(f"  {bench:22s} {v:10.3f} {s:8.3f}  ±{1.96*s:<8.3f}     {r['n']:>4d}")

        domains = domain_aggregate(bench_results)
        print(f"\n  {'domain':35s} {'mean':>10s} {'std':>8s} {'95% CI':>16s} {'n_bench':>8s}")
        for domain, r in domains.items():
            miss = f"  (missing: {r['missing']})" if r["missing"] else ""
            print(f"  {domain:35s} {r['mean']:10.3f} {r['std']:8.3f}  ±{r['ci95']:<8.3f}     {r['n_benchmarks']:>6d}{miss}")

        all_out[d] = {
            "benchmarks": {b: {"mean": scale(b, r["mean"]), "std": scale(b, r["std"]), "n": r["n"]} for b, r in bench_results.items()},
            "domains": domains,
        }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_out, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
