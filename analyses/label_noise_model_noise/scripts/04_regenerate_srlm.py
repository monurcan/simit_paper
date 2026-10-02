"""Step 5 (per plan): SRLM has no cached rollout data anywhere -- confirmed
by investigation: job_outputs/*.out never prints completions for SRLM, and
train_log.jsonl only stores aggregates (n_candidates, n_scored, mean_score,
pair_found), never per-candidate text or scores. Per the user's decision,
we regenerate SRLM's rollouts live: "initial supervision before parameter
updates" means the FROZEN BASE MODEL (LoRA at initialization has lora_B=0,
so it is behaviorally identical to the base model -- this is also exactly
what TTRL/TTRV/EMPO's own step-1 rollouts reflect), so this loads BAGEL-7B
with NO LoRA at all (lora_rank=0) rather than any partially-trained
checkpoint -- the historical exact rollout draw is unrecoverable either way,
so this is the closest faithful reconstruction of "what SRLM would have
proposed as candidates on this query, before any update".

For each selected query: generate K candidates (matching that benchmark's
own canonical run's K/temperature, read from its job script) via
common/rollout.py's generate_k_completions (imported, not reimplemented),
score each with srlm/judge.py's own self-judge prompt (imported verbatim),
then apply the SAME group_normalized_advantages GRPO-style centering used
for TTRL/EMPO/TTRV to the raw self-judge scores -- consistent with the
paper's own framing ("for SRLM, [supervision] is the candidate response with
the highest score from the self-judge"): centering by the group mean makes
the top-scored response get positive advantage (and P(j|i,m) concentrates
there) while preserving the general advantage-weighted-sampling procedure
used uniformly across all 4 LFRL methods, rather than a special-cased argmax.
Unparseable judge scores are excluded from the group entirely (matching
srlm/judge.py's own select_preference_pair behavior), never treated as 0.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[3])
import json
import os
import random
import re
import sys

sys.path.insert(0, SIMIT_ROOT + "/lmms-eval/pylibs")
sys.path.insert(0, SIMIT_ROOT + "/Bagel")
sys.path.insert(0, SIMIT_ROOT + "/LFRL_Baselines")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from common.dataset import load_real_samples  # noqa: E402 (read-only import)
from common.grpo import group_normalized_advantages  # noqa: E402
from common.paths import BAGEL_MODEL_PATH  # noqa: E402
from common.rollout import generate_k_completions  # noqa: E402
from common.runtime import load_bagel_runtime  # noqa: E402
from srlm.judge import JUDGE_PROMPT_TEMPLATE, score_completions  # noqa: E402 (unused import kept for provenance clarity)

sys.path.insert(0, SIMIT_ROOT + "/analyses/label_noise_model_noise/scripts")
from config import CACHE_DIR, LFRL_ROOT, run_name as canonical_run_name  # noqa: E402

BASE_SAMPLING_SEED = 20260913
JOB_SCRIPTS_DIR = os.path.join(LFRL_ROOT, "job_scripts")


def parse_srlm_job_args(bench):
    run_name = canonical_run_name("srlm", bench)
    path = os.path.join(JOB_SCRIPTS_DIR, f"{run_name}.sh")
    text = open(path).read()

    def grab(flag, default, cast):
        m = re.search(rf"{re.escape(flag)} (\S+)", text)
        return cast(m.group(1)) if m else default

    return {
        "run_name": run_name,
        "k": grab("--k", 16, int),
        "temperature": grab("--temperature", 1.0, float),
        "max_new_tokens": grab("--max_new_tokens", 128, int),
        "judge_max_new_tokens": grab("--judge_max_new_tokens", 200, int),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-bench", type=str, default=None, help="restrict to one benchmark, for testing")
    args = ap.parse_args()

    selected = [json.loads(l) for l in open(os.path.join(CACHE_DIR, "selected_queries.jsonl"))]
    by_bench = {}
    for row in selected:
        by_bench.setdefault(row["benchmark"], []).append(row["sample_id"])
    if args.only_bench:
        by_bench = {args.only_bench: by_bench[args.only_bench]}

    print("Loading frozen BAGEL-7B runtime (lora_rank=0, i.e. no adapter -- pure base model, "
          "matching 'before any parameter update')...", flush=True)
    runtime = load_bagel_runtime(model_path=BAGEL_MODEL_PATH, device="cuda:0", lora_rank=0)
    print("Runtime loaded.", flush=True)

    out_rows = []
    for bench, sample_ids in by_bench.items():
        job_args = parse_srlm_job_args(bench)
        print(f"[srlm] {bench}: run={job_args['run_name']} k={job_args['k']} "
              f"temp={job_args['temperature']} -- regenerating {len(sample_ids)} queries", flush=True)

        real_samples = {s.sample_id: s for s in load_real_samples(benchmarks=[bench], load_answers=True)}

        for sample_id in sample_ids:
            sample = real_samples[sample_id]
            image = Image.open(sample.image_path).convert("RGB")

            completions = generate_k_completions(
                runtime, image=image, prompt=sample.prompt, k=job_args["k"],
                max_new_tokens=job_args["max_new_tokens"], temperature=job_args["temperature"],
            )
            scores = score_completions(
                runtime, image=image, instruction=sample.prompt, completions=completions,
                max_new_tokens=job_args["judge_max_new_tokens"],
            )

            scoreable = [(i, s) for i, (s, c) in enumerate(zip(scores, completions)) if s is not None and c]
            row = {
                "benchmark": bench,
                "method": "srlm",
                "sample_id": sample_id,
                "run_name": job_args["run_name"],
                "checkpoint_config": {"k": job_args["k"], "temperature": job_args["temperature"],
                                       "lora_rank_regen": 0, "note": "regenerated from frozen base model, see module docstring"},
                "image_path": sample.image_path,
                "question": sample.prompt,
                "rollout_answers": completions,
                "raw_judge_scores": scores,
                "sampling_seed": f"{BASE_SAMPLING_SEED}:srlm:{bench}:{sample_id}",
            }

            if len(scoreable) < 2:
                row.update({"availability": "no_positive_supervision",
                            "availability_reason": "fewer_than_2_scoreable_completions",
                            "advantages": None, "selected_answer": None,
                            "selected_index": None, "selection_probability": None})
                out_rows.append(row)
                continue

            idxs = [i for i, _ in scoreable]
            sub_scores = [s for _, s in scoreable]
            sub_advantages = group_normalized_advantages(sub_scores)
            advantages_full = [None] * len(completions)
            for i, a in zip(idxs, sub_advantages):
                advantages_full[i] = a

            positive = [max(a, 0.0) for a in sub_advantages]
            total = sum(positive)
            row["advantages"] = advantages_full
            if total <= 0.0:
                row.update({"availability": "no_positive_supervision",
                            "availability_reason": "zero_reward_variance",
                            "selected_answer": None, "selected_index": None, "selection_probability": None})
            else:
                probs = [p / total for p in positive]
                rng = random.Random(row["sampling_seed"])
                chosen_local = rng.choices(range(len(idxs)), weights=probs, k=1)[0]
                chosen_idx = idxs[chosen_local]
                row.update({
                    "availability": "available",
                    "selected_index": chosen_idx,
                    "selected_answer": completions[chosen_idx],
                    "selection_probability": probs[chosen_local],
                })
            out_rows.append(row)

        n_avail = sum(1 for r in out_rows if r["benchmark"] == bench and r["availability"] == "available")
        print(f"[srlm] {bench}: {n_avail}/{len(sample_ids)} available", flush=True)

        # persist incrementally so partial progress survives any crash
        out_path = os.path.join(CACHE_DIR, "lfrl_srlm_selected.jsonl")
        with open(out_path, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[srlm] wrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
