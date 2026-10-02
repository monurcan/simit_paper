"""Single calls at the REAL pipeline's max_output_tokens=1800 (matching
synthesize_vqa.py's actual generate_synthetic_examples() call) to see how
much the skill-prompt prefix overhead actually matters once decode time
(proportional to output length) dominates, rather than the short 60-token
test used to isolate prefill cost.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import io
import json
import sys
import time

sys.path.insert(0, SIMIT_ROOT)
sys.path.insert(0, SIMIT_ROOT + "/examples/diagram_gen")
from PIL import Image
from wait_ready import wait_ready
import prompts as skill_prompts

IMAGE_PATH = SIMIT_ROOT + "/Bagel/test_images/women.jpg"
MASTER_PROMPT = skill_prompts.MASTER_PROMPT
SUFFIX = "\n\nNow apply the above to this request: draw a bar chart of monthly rainfall."
NO_PREFIX_PROMPT = "Draw a bar chart of monthly rainfall. Describe it in detail with axis labels and data values."


def load_image_bytes():
    img = Image.open(IMAGE_PATH).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    client = wait_ready()
    image_bytes = load_image_bytes()

    print("\n=== With ~21k-char skill prompt prefix, max_output_tokens=1800 ===")
    t0 = time.perf_counter()
    resp = client.chat(MASTER_PROMPT + SUFFIX, images=image_bytes, think_mode=False, max_output_tokens=1800, temperature=0.2)
    dt_with = time.perf_counter() - t0
    n_out_chars_with = len(resp.text)
    print(f"  took {dt_with:.2f}s, {n_out_chars_with} output chars")

    print("\n=== No prefix, same output budget, max_output_tokens=1800 ===")
    t0 = time.perf_counter()
    resp = client.chat(NO_PREFIX_PROMPT, images=image_bytes, think_mode=False, max_output_tokens=1800, temperature=0.2)
    dt_without = time.perf_counter() - t0
    n_out_chars_without = len(resp.text)
    print(f"  took {dt_without:.2f}s, {n_out_chars_without} output chars")

    result = {
        "with_prefix_s": dt_with, "with_prefix_out_chars": n_out_chars_with,
        "without_prefix_s": dt_without, "without_prefix_out_chars": n_out_chars_without,
        "prefix_overhead_s": dt_with - dt_without,
        "prefix_overhead_fraction_of_with": (dt_with - dt_without) / dt_with,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/bench_realistic_length_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
