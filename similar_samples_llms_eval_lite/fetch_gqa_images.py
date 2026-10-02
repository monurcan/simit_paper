"""Download only the GQA images needed for the LMMs-Eval-Lite GQA subset.

Streams lmms-lab/GQA (testdev_balanced_images then testdev_all_images if any
remain) and saves each matching image to gqa_images/<imageId>.jpg without
ever holding the full dataset in memory.
"""

import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import sys
from pathlib import Path

sys.path.insert(0, SIMIT_ROOT)

from datasets import load_dataset, load_dataset_builder

OUT_DIR = Path(SIMIT_ROOT + "/similar_samples_llms_eval_lite/gqa_images")
OUT_DIR.mkdir(exist_ok=True)

# Step 1: collect the 500 imageIds we need
print("Loading GQA lite imageIds...")
lite = load_dataset("lmms-lab/LMMs-Eval-Lite", "gqa")["lite"]
needed = {ex["imageId"] for ex in lite}
print(f"  {len(needed)} unique imageIds across {len(lite)} samples")

# Step 2: stream GQA images configs until all found
already = {p.stem for p in OUT_DIR.glob("*.jpg")}
remaining = needed - already
print(f"  {len(already)} already cached, {len(remaining)} to fetch")

CONFIGS = [
    ("testdev_balanced_images", "testdev"),
    ("testdev_all_images", "testdev"),
    ("val_balanced_images", "val"),
    ("val_all_images", "val"),
]

for config, split in CONFIGS:
    if not remaining:
        break
    print(f"\nStreaming lmms-lab/GQA / {config} (split={split}) ...")
    ds = load_dataset("lmms-lab/GQA", config, split=split, streaming=True)
    found_this_pass = 0
    for ex in ds:
        img_id = ex["id"]
        if img_id in remaining:
            path = OUT_DIR / f"{img_id}.jpg"
            ex["image"].convert("RGB").save(path, format="JPEG", quality=95)
            remaining.discard(img_id)
            found_this_pass += 1
            if found_this_pass % 50 == 0:
                print(f"  found {found_this_pass} so far, {len(remaining)} still needed")
        if not remaining:
            break
    print(f"  fetched {found_this_pass} from {config}")

if remaining:
    print(f"\nWARNING: {len(remaining)} imageIds not found in any GQA config: {list(remaining)[:10]}")
else:
    print(f"\nDone. All {len(needed)} images saved to {OUT_DIR}")
