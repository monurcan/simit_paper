"""
Build filtered/capped training JSONLs for the difficulty-filtering (DF) +
adaptive-sample-count (ASC) LoRA data-selection mechanisms, from
a synthetic pool (--pool v2: generated_samples_discard_unverified_v2, 6
candidates per query; --pool nodf4: ..._no_diff_filtering_4_samples, 4) plus the
BAGEL confidence scores written by compute_confidence.py
(original_sample_confidence.json -> p0, synthesized_sample_{i}_confidence.json -> c[i]).

Per real sample:
  stage 1 (DF):  kept = [c_i for i in slot order if T_low <= c_i <= T_high]
  stage 2 (ASC): k = min{K in 0..Kmax : p0 >= tau[K]}, else k=Kmax
  used = kept[:k]

--disable-df makes stage 1 a no-op (kept = all available slots).
--disable-adaptive-k makes stage 2 a no-op (k = Kmax, i.e. no cap).

Writes, per benchmark, under data_root/vqa_benchmarks_v2_dfadaptive/{tag}/:
  {bench}.jsonl      -- selected (image, question, answer) records
  registry.json       -- DATASET_INFO-compatible entries for every bench processed (merged in)
and per benchmark under data/configs/:
  vqa_{bench}_v2_dfadaptive_{tag}.yaml
"""
import argparse
import json
import math
import os
import re
from pathlib import Path

SIMIT_ROOT = os.environ.get("SIMIT_ROOT") or str(Path(__file__).resolve().parents[1])
SIMIT_DATA = os.environ.get("SIMIT_DATA") or os.path.join(SIMIT_ROOT, "workdir")
from simit_pools import POOLS

SRC_ROOT = Path(SIMIT_DATA) / POOLS["v2"][0]  # overridden by --pool
OUT_ROOT = Path(SIMIT_ROOT) / "Bagel" / "data_root" / "vqa_benchmarks_v2_dfadaptive"
CONFIG_DIR = Path(SIMIT_ROOT) / "Bagel" / "data" / "configs"
NUM_GENERATIONS = POOLS["v2"][1]  # Kmax, overridden by --pool


def clean_question(text: str) -> str:
    text = text.strip()
    if "\n" not in text:
        return text
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    q_lines = [l for l in lines if l.lower().startswith("question:")]
    if q_lines:
        return re.sub(r"(?i)^question:\s*", "", q_lines[-1]).strip()
    return " ".join(lines)


def compute_tau(epsilon, A0, A1, B):
    O_0 = A0 / (1 - A0)
    tau = {0: 1 - epsilon}
    for K in range(1, NUM_GENERATIONS + 1):
        A_K = min(max(A1 + B * math.log(K), 1e-3), 1 - 1e-3)
        O_K = A_K / (1 - A_K)
        tau[K] = (1 - epsilon) * O_0 / (epsilon * O_K + (1 - epsilon) * O_0)
    return tau


def adaptive_k(p0, tau):
    for K in range(0, NUM_GENERATIONS + 1):
        if p0 >= tau[K]:
            return K
    return NUM_GENERATIONS


def load_prob(path: Path):
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    return d.get("avg_answer_prob")


def build_records(bench_name, epsilon, A0, A1, B, T_low, T_high, disable_df, disable_adaptive_k, with_description=False):
    bench_dir = SRC_ROOT / bench_name
    tau = compute_tau(epsilon, A0, A1, B)
    records = []
    k_hist = {K: 0 for K in range(NUM_GENERATIONS + 1)}
    n_samples = 0
    sum_kept, sum_used = 0, 0

    for sample_dir in sorted(
        (p for p in bench_dir.iterdir() if p.is_dir()), key=lambda p: int(re.search(r"\d+", p.name).group())
    ):
        p0 = load_prob(sample_dir / "original_sample_confidence.json")
        if p0 is None:
            continue
        n_samples += 1

        candidates = []  # (slot_index, question, answer, image_rel_path, c_i)
        for i in range(NUM_GENERATIONS):
            img = sample_dir / f"synthesized_sample_{i}.png"
            prompt_f = sample_dir / f"synthesized_sample_{i}_prompt.txt"
            ans_f = sample_dir / f"synthesized_sample_{i}_answers.txt"
            conf_f = sample_dir / f"synthesized_sample_{i}_confidence.json"
            if not (img.exists() and prompt_f.exists() and ans_f.exists()):
                continue
            c_i = load_prob(conf_f)
            if c_i is None:
                continue
            question = clean_question(prompt_f.read_text())
            answer = ans_f.read_text().strip()
            if not question or not answer:
                continue
            if with_description:
                desc_f = sample_dir / f"synthesized_sample_{i}_description.txt"
                if desc_f.exists():
                    description = desc_f.read_text().strip()
                    if description:
                        question = f"The figure shows: {description}\n\n{question}"
            candidates.append((i, question, answer, img, c_i))

        if disable_df:
            kept = candidates
        else:
            kept = [c for c in candidates if T_low <= c[4] <= T_high]

        if disable_adaptive_k:
            k = NUM_GENERATIONS
        else:
            k = adaptive_k(p0, tau)
        k_hist[k] += 1

        used = kept[:k]
        sum_kept += len(kept)
        sum_used += len(used)

        for _, question, answer, img, _ in used:
            records.append(
                {
                    "image": str(img.relative_to(SRC_ROOT)),
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{question}"},
                        {"from": "gpt", "value": answer},
                    ],
                }
            )

    stats = {
        "n_real_samples": n_samples,
        "k_histogram": k_hist,
        "avg_kept_before_cap": sum_kept / n_samples if n_samples else 0.0,
        "avg_used_after_cap": sum_used / n_samples if n_samples else 0.0,
        "total_records": len(records),
    }
    return records, stats


def write_yaml_config(dataset_name, num_records, out_path):
    out_path.write_text(
        "vlm_sft:\n"
        f"  dataset_names:\n  - {dataset_name}\n"
        "  image_transform_args:\n"
        "    image_stride: 14\n"
        "    max_image_size: 980\n"
        "    min_image_size: 378\n"
        "    max_pixels: 2_007_040\n"
        "  frame_sampler_args:\n"
        "    max_num_frames: 12\n"
        "    min_num_frames: 8\n"
        "  is_mandatory: true\n"
        "  shuffle_lines: False\n"
        "  shuffle_seed: 0\n"
        "  preserve_order: True\n"
        f"  num_used_data:\n  - {num_records}\n"
        "  weight: 1\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pool", choices=sorted(POOLS), default="v2", help="synthetic pool (Kmax = 6 for v2, 4 for the others)")
    ap.add_argument("--benches", required=True, help="comma-separated benchmark names")
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--A0", type=float, required=True)
    ap.add_argument("--A1", type=float, required=True)
    ap.add_argument("--B", type=float, required=True)
    ap.add_argument("--T-low", type=float, required=True, dest="T_low")
    ap.add_argument("--T-high", type=float, required=True, dest="T_high")
    ap.add_argument("--disable-df", action="store_true")
    ap.add_argument("--disable-adaptive-k", action="store_true")
    ap.add_argument("--with-description", action="store_true",
                     help="Prepend the synthetic sample's generation-time description "
                          "text ('The figure shows: ...') to the question.")
    args = ap.parse_args()
    global SRC_ROOT, NUM_GENERATIONS
    SRC_ROOT = Path(SIMIT_DATA) / POOLS[args.pool][0]
    NUM_GENERATIONS = POOLS[args.pool][1]

    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = {}

    for bench_name in args.benches.split(","):
        records, stats = build_records(
            bench_name, args.epsilon, args.A0, args.A1, args.B, args.T_low, args.T_high,
            args.disable_df, args.disable_adaptive_k, with_description=args.with_description,
        )
        jsonl_path = out_dir / f"{bench_name}.jsonl"
        with jsonl_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        dataset_name = f"vqa_{bench_name}_v2_dfadaptive_{args.tag}"
        registry[dataset_name] = {
            "data_dir": str(SRC_ROOT),
            "jsonl_path": f"data_root/vqa_benchmarks_v2_dfadaptive/{args.tag}/{bench_name}.jsonl",
            "num_total_samples": len(records),
        }
        write_yaml_config(dataset_name, len(records), CONFIG_DIR / f"vqa_{bench_name}_v2_dfadaptive_{args.tag}.yaml")

        print(f"{bench_name}: {stats}")

    with (out_dir / "registry.json").open("w") as f:
        json.dump(registry, f, indent=2)


if __name__ == "__main__":
    main()
