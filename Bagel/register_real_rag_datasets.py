"""
Register vqa_<bench>_real_rag datasets (built by prepare_vqa_data_real_rag.py)
via data_root/vqa_benchmarks_real_rag/registry.json (picked up by
data/dataset_info.py) and write matching training-config yamls under
data/configs/, mirroring the vqa_<bench>_v2_roundrobin convention (shuffle
already baked into the jsonl's fixed order, so shuffle_lines=False +
preserve_order=True here too).
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import json
import time
from pathlib import Path

# Skip any jsonl whose mtime is too recent -- prepare_vqa_data_real_rag.py may
# still be actively writing it (its 'with open(...) as f' only flushes/closes
# once that benchmark finishes), and registering a still-growing file would
# freeze num_used_data at whatever partial line count happened to be on disk.
STALE_SECONDS = 120

BAGEL_ROOT = Path(SIMIT_ROOT + "/Bagel")
OUT_ROOT = BAGEL_ROOT / "data_root" / "vqa_benchmarks_real_rag"
CONFIGS_DIR = BAGEL_ROOT / "data" / "configs"

YAML_TEMPLATE = """vlm_sft:
  dataset_names:
  - vqa_{bench}_real_rag
  image_transform_args:
    image_stride: 14
    max_image_size: 980
    min_image_size: 378
    max_pixels: 2_007_040
  frame_sampler_args:
    max_num_frames: 12
    min_num_frames: 8
  is_mandatory: true
  shuffle_lines: False
  shuffle_seed: 0
  preserve_order: True
  num_used_data:
  - {n}
  weight: 1
"""


def main():
    entries = []
    for bench_dir in sorted(OUT_ROOT.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench = bench_dir.name
        jsonl_path = bench_dir / f"{bench}_real_rag.jsonl"
        if not jsonl_path.exists():
            continue
        age_s = time.time() - jsonl_path.stat().st_mtime
        if age_s < STALE_SECONDS:
            print(f"{bench}: jsonl modified {age_s:.0f}s ago (<{STALE_SECONDS}s), still being written -- skipping for now")
            continue
        n = sum(1 for l in jsonl_path.open() if l.strip())
        if n == 0:
            print(f"{bench}: 0 records, skipping registration")
            continue
        entries.append((bench, n))

        yaml_path = CONFIGS_DIR / f"vqa_{bench}_real_rag.yaml"
        yaml_path.write_text(YAML_TEMPLATE.format(bench=bench, n=n))

    registry = {
        f"vqa_{bench}_real_rag": {
            "data_dir": "data_root/vqa_benchmarks_real_rag",
            "jsonl_path": f"data_root/vqa_benchmarks_real_rag/{bench}/{bench}_real_rag.jsonl",
            "num_total_samples": n,
        }
        for bench, n in entries
    }
    (OUT_ROOT / "registry.json").write_text(json.dumps(registry, indent=2))
    print(f"Registered {len(registry)} real_rag datasets in {OUT_ROOT / 'registry.json'}")
    print(json.dumps(dict(entries), indent=2))


if __name__ == "__main__":
    main()
