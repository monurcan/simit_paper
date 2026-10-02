"""Step 2: extract the frozen base model's greedy answer for every selected
original query, from the already-cached zero-shot baseline eval
(lmms-eval pretrained_baseline run) -- no inference is run here.

M itself (binary correctness) is NOT assigned in this script: the automatic
per-task metric is inconsistent across benchmarks (hard 0/1 for some, soft
ANLS for others, entirely absent for the 4 captioning benchmarks), so M goes
through the same external/agentic evaluator as the supervision audits. This
script only pulls the greedy-decoded answer text + question/image, i.e.
everything needed to build that evaluation packet.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import CACHE_DIR, GENERATED_SAMPLES_ROOT


def load_original_query(bench, doc_id):
    d = os.path.join(GENERATED_SAMPLES_ROOT, bench, f"sample{doc_id}")
    with open(os.path.join(d, "original_sample_prompt.txt")) as f:
        question = f.read().strip()
    answers_path = os.path.join(d, "original_sample_answers.txt")
    gt_answers = open(answers_path).read().strip().splitlines() if os.path.exists(answers_path) else []
    return {
        "image_path": os.path.join(d, "original_sample.png"),
        "question": question,
        "gt_answers": gt_answers,
    }


def load_baseline_by_doc_id(baseline_eval_file):
    by_doc_id = {}
    with open(baseline_eval_file) as f:
        for line in f:
            row = json.loads(line)
            by_doc_id[row["doc_id"]] = row
    return by_doc_id


def main():
    selected_path = os.path.join(CACHE_DIR, "selected_queries.jsonl")
    selected = [json.loads(l) for l in open(selected_path)]

    baseline_cache = {}  # baseline_eval_file -> {doc_id: row}
    out_rows = []
    for row in selected:
        bef = row["baseline_eval_file"]
        if bef not in baseline_cache:
            baseline_cache[bef] = load_baseline_by_doc_id(bef)
        base_row = baseline_cache[bef][row["doc_id"]]
        query = load_original_query(row["benchmark"], row["doc_id"])
        out_rows.append(
            {
                "benchmark": row["benchmark"],
                "doc_id": row["doc_id"],
                "sample_id": row["sample_id"],
                "image_path": query["image_path"],
                "question": query["question"],
                "gt_answers": query["gt_answers"],
                "baseline_answer": base_row["filtered_resps"],
                "checkpoint": "BAGEL-7B-MoT (frozen base, greedy decoding)",
                "source_eval_run": bef,
            }
        )

    out_path = os.path.join(CACHE_DIR, "baseline_answers.jsonl")
    with open(out_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Extracted {len(out_rows)} baseline answers -> {out_path}")


if __name__ == "__main__":
    main()
