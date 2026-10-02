"""Step 3: reconstruct rollout-level advantages for TTRL, EMPO, TTRV (SRLM
is handled separately in 04_regenerate_srlm.py -- it has no cached rollout
data at all) and sample one supervision answer per query per method,
proportional to positive advantage:

    P(j | i, m) = max(A_ij, 0) / sum_k max(A_ik, 0)

Each method's own reward is read as-is from train_log.jsonl (never
recomputed) -- only the GRPO advantage normalization is reapplied, using
LFRL_Baselines/common/grpo.py's own function (imported, not reimplemented):

  - TTRL / EMPO: A_ij = group_normalized_advantages(rewards)   [hard 0/1 reward]
  - TTRV:        A_ij = group_normalized_advantages(r1) + entropy_coef * r2
                 (entropy_coef read from that run's own job log header --
                 it varies per run, e.g. 0.25 for the ai2d winner vs the
                 0.75 code default elsewhere)

If every advantage in a query's group is <=0, there is no positive-advantage
answer to sample -- recorded as availability="no_positive_supervision", per
instruction (never invent an answer).
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[3])
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, SIMIT_ROOT + "/LFRL_Baselines")
from common.grpo import group_normalized_advantages  # noqa: E402 (read-only import)

from config import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINT_SUBDIR,
    GENERATED_SAMPLES_ROOT,
    LFRL_CHECKPOINTS_ROOT,
    LFRL_ROOT,
    run_name as canonical_run_name,
)
from lfrl_log_utils import find_canonical_entry, load_run_records, parse_ttrv_header  # noqa: E402

JOB_OUTPUTS_DIR = os.path.join(LFRL_ROOT, "job_outputs")
BASE_SAMPLING_SEED = 20260913


def real_image_and_question(bench, sample_id):
    doc_id = int(sample_id.replace("sample", ""))
    d = os.path.join(GENERATED_SAMPLES_ROOT, bench, f"sample{doc_id}")
    with open(os.path.join(d, "original_sample_prompt.txt")) as f:
        question = f.read().strip()
    return os.path.join(d, "original_sample.png"), question


def advantages_for(method, entry, ttrv_header):
    if method in ("ttrl", "empo"):
        rewards = entry["rewards"]
        return group_normalized_advantages(rewards), rewards
    if method == "ttrv":
        r1 = entry["r1"]
        base = group_normalized_advantages(r1)
        entropy_coef = ttrv_header["entropy_coef"]
        r2 = entry["r2_entropy_term"]
        adv = [b + entropy_coef * r2 for b in base]
        return adv, r1
    raise ValueError(method)


def sample_positive_advantage(advantages, rng):
    positive = [max(a, 0.0) for a in advantages]
    total = sum(positive)
    if total <= 0.0:
        return None, None, [0.0] * len(advantages)
    probs = [p / total for p in positive]
    idx = rng.choices(range(len(advantages)), weights=probs, k=1)[0]
    return idx, probs[idx], probs


def process_run(method, bench, selected_sample_ids):
    run_name = canonical_run_name(method, bench)
    checkpoint_subdir = CHECKPOINT_SUBDIR[method]
    train_log_path = os.path.join(LFRL_CHECKPOINTS_ROOT, checkpoint_subdir, run_name, "train_log.jsonl")

    records, used_out_files = load_run_records(JOB_OUTPUTS_DIR, train_log_path, run_name)
    ttrv_header = parse_ttrv_header(JOB_OUTPUTS_DIR, used_out_files[0]) if method == "ttrv" else None
    if method == "ttrv" and ttrv_header is None:
        raise RuntimeError(f"{run_name}: could not parse entropy_coef from job header ({used_out_files[0]})")

    image_path, question = None, None
    out_rows = []
    for sample_id in selected_sample_ids:
        entry, completions, n_dup = find_canonical_entry(records, sample_id)
        row = {
            "benchmark": bench,
            "method": method,
            "sample_id": sample_id,
            "run_name": run_name,
            "checkpoint_config": (
                {"lora_rank": ttrv_header["lora_rank"], "temperature": ttrv_header["temperature"],
                 "entropy_coef": ttrv_header["entropy_coef"], "voting_mode": ttrv_header["voting_mode"], "k": ttrv_header["k"]}
                if method == "ttrv" else {}
            ),
            "log_source_out_files": used_out_files,
        }
        if entry is None:
            row.update({"availability": "sample_id_not_found_in_log", "step": None})
            out_rows.append(row)
            continue

        image_path, question = real_image_and_question(bench, sample_id)
        advantages, raw_rewards = advantages_for(method, entry, ttrv_header)

        rng = random.Random(f"{BASE_SAMPLING_SEED}:{method}:{bench}:{sample_id}")
        idx, prob, all_probs = sample_positive_advantage(advantages, rng)

        row.update(
            {
                "step": entry["step"],
                "epoch": entry["epoch"],
                "resume_duplicate_count": n_dup,
                "image_path": image_path,
                "question": question,
                "rollout_answers": completions,
                "raw_rewards_or_r1": raw_rewards,
                "advantages": advantages,
                "sampling_seed": f"{BASE_SAMPLING_SEED}:{method}:{bench}:{sample_id}",
            }
        )
        if idx is None:
            row.update({"availability": "no_positive_supervision", "selected_answer": None,
                        "selected_index": None, "selection_probability": None})
        else:
            row.update(
                {
                    "availability": "available",
                    "selected_index": idx,
                    "selected_answer": completions[idx],
                    "selection_probability": prob,
                    "selection_probability_all": all_probs,
                }
            )
        out_rows.append(row)
    return out_rows


def main():
    selected = [json.loads(l) for l in open(os.path.join(CACHE_DIR, "selected_queries.jsonl"))]
    by_bench = {}
    for row in selected:
        by_bench.setdefault(row["benchmark"], []).append(row["sample_id"])

    for method in ["ttrl", "empo", "ttrv"]:
        all_rows = []
        for bench, sample_ids in by_bench.items():
            print(f"[{method}] {bench}: extracting {len(sample_ids)} queries from run "
                  f"{canonical_run_name(method, bench)} ...", flush=True)
            rows = process_run(method, bench, sample_ids)
            all_rows.extend(rows)
            n_avail = sum(1 for r in rows if r["availability"] == "available")
            print(f"[{method}] {bench}: {n_avail}/{len(rows)} available", flush=True)

        out_path = os.path.join(CACHE_DIR, f"lfrl_{method}_selected.jsonl")
        with open(out_path, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{method}] wrote {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
