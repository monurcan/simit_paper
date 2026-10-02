"""Training data without adaptive budget allocation or difficulty filtering: every
synthetic sample of a pool, for the ablations of Tab. 2 / 3 and the N_train
scaling curve (Fig. 9). Replaces one near-identical script per pool.

For each benchmark, writes under data_root/vqa_benchmarks_pools/<pool>/:
  <bench>.jsonl             sample-major order (all candidates of query 0, then query 1, ...)
  <bench>_roundrobin.jsonl  generation-major order (candidate 0 of every query, then candidate 1, ...),
                            so the first N records spread N samples over as many queries as possible
  registry.json             dataset_info.py entries vqa_<bench>_<pool>[_roundrobin]
and data/configs/vqa_<bench>_<pool>[_roundrobin].yaml using all records.

    python prepare_vqa_data_pool.py --pool force_natural [--benches ai2d,gqa]
"""
import argparse
import json
from pathlib import Path

from prepare_vqa_data_v2_dfadaptive import CONFIG_DIR, SIMIT_DATA, SIMIT_ROOT, clean_question, write_yaml_config
from simit_pools import POOLS
from simit_sweep import ALL_BENCHES

OUT_ROOT = Path(SIMIT_ROOT) / "Bagel" / "data_root" / "vqa_benchmarks_pools"


def build_records(src_root: Path, bench: str, num_generations: int):
    """by_gen[i] = records of candidate slot i, in query order."""
    by_gen = {i: [] for i in range(num_generations)}
    sample_dirs = sorted((p for p in (src_root / bench).iterdir() if p.is_dir() and p.name.startswith("sample")),
                         key=lambda p: int(p.name[len("sample"):]))
    for sample_dir in sample_dirs:
        for i in range(num_generations):
            img = sample_dir / f"synthesized_sample_{i}.png"
            prompt_f = sample_dir / f"synthesized_sample_{i}_prompt.txt"
            ans_f = sample_dir / f"synthesized_sample_{i}_answers.txt"
            if not (img.exists() and prompt_f.exists() and ans_f.exists()):
                continue
            question = clean_question(prompt_f.read_text())
            answer = ans_f.read_text().strip()
            if not question or not answer:
                continue
            by_gen[i].append({
                "image": str(img.relative_to(src_root)),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            })
    return by_gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=sorted(POOLS))
    ap.add_argument("--benches", default=",".join(ALL_BENCHES))
    args = ap.parse_args()
    pool_dir, num_generations = POOLS[args.pool]
    src_root = Path(SIMIT_DATA) / pool_dir
    out_dir = OUT_ROOT / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_path = out_dir / "registry.json"
    registry = json.loads(reg_path.read_text()) if reg_path.exists() else {}

    for bench in args.benches.split(","):
        by_gen = build_records(src_root, bench, num_generations)
        sample_major = [by_gen[i][s] for s in range(len(by_gen[0])) for i in range(num_generations) if s < len(by_gen[i])]
        roundrobin = [r for i in range(num_generations) for r in by_gen[i]]
        for suffix, records in (("", sample_major), ("_roundrobin", roundrobin)):
            jsonl = out_dir / f"{bench}{suffix}.jsonl"
            jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
            name = f"vqa_{bench}_{args.pool}{suffix}"
            registry[name] = {"data_dir": str(src_root), "jsonl_path": str(jsonl.relative_to(Path(SIMIT_ROOT) / "Bagel")),
                              "num_total_samples": len(records)}
            write_yaml_config(name, len(records), CONFIG_DIR / f"{name}.yaml")
        print(f"{bench}: {len(sample_major)} records, per candidate slot {[len(by_gen[i]) for i in range(num_generations)]}")
    reg_path.write_text(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
