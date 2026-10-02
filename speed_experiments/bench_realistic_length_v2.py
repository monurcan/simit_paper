"""Better-controlled version of bench_realistic_length.py: multiple trials
per condition, and normalizes for the fact that output length varies call to
call (EOS-triggered), by estimating decode time from a measured tokens/sec
rate and backing out prefill time = total - decode, rather than comparing
raw wall-clock across calls that generated different amounts of text.
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
N_TRIALS = 5
CHARS_PER_TOKEN = 4.0  # rough approximation, consistent across both conditions so it cancels in the comparison

WITH_PREFIX_SUFFIXES = [
    "\n\nNow apply the above to this request: draw a bar chart of monthly rainfall.",
    "\n\nNow apply the above to this request: draw a scatter plot of height vs weight.",
    "\n\nNow apply the above to this request: draw a pie chart of budget allocation.",
    "\n\nNow apply the above to this request: draw a line chart of stock price over a year.",
    "\n\nNow apply the above to this request: draw a histogram of exam scores.",
]
NO_PREFIX_PROMPTS = [
    "Draw a bar chart of monthly rainfall. Describe it in detail with axis labels and data values.",
    "Draw a scatter plot of height vs weight. Describe it in detail with axis labels and data values.",
    "Draw a pie chart of budget allocation. Describe it in detail with axis labels and data values.",
    "Draw a line chart of stock price over a year. Describe it in detail with axis labels and data values.",
    "Draw a histogram of exam scores. Describe it in detail with axis labels and data values.",
]


def load_image_bytes():
    img = Image.open(IMAGE_PATH).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_trials(client, image_bytes, prompts, label):
    trials = []
    for i, p in enumerate(prompts):
        t0 = time.perf_counter()
        resp = client.chat(p, images=image_bytes, think_mode=False, max_output_tokens=1800, temperature=0.2)
        dt = time.perf_counter() - t0
        out_tokens_est = len(resp.text) / CHARS_PER_TOKEN
        trials.append({"total_s": dt, "out_chars": len(resp.text), "out_tokens_est": out_tokens_est})
        print(f"  [{label}] trial {i}: {dt:.2f}s, {len(resp.text)} chars (~{out_tokens_est:.0f} tok)")
    return trials


def main():
    client = wait_ready()
    image_bytes = load_image_bytes()

    print("\n=== WITH skill-prompt prefix (~21k chars), 5 trials, max_output_tokens=1800 ===")
    with_trials = run_trials(client, image_bytes, [MASTER_PROMPT + s for s in WITH_PREFIX_SUFFIXES], "with-prefix")

    print("\n=== WITHOUT prefix, 5 trials, max_output_tokens=1800 ===")
    without_trials = run_trials(client, image_bytes, NO_PREFIX_PROMPTS, "no-prefix")

    # Fit decode rate (tokens/sec) via simple linear regression across BOTH
    # conditions pooled: total_s = prefill_s + out_tokens_est / rate.
    # With only two conditions (different prefill, different token counts)
    # we can't do a full regression robustly with n=10, so instead: use the
    # no-prefix condition's own spread (different out_tokens_est values) to
    # fit a per-condition rate, assuming its own prefill_s is roughly constant
    # across its 5 trials (same prompt length each time).
    def fit_rate_and_prefill(trials):
        # total_s = a + b * out_tokens_est ; b = 1/rate, a = prefill_s
        n = len(trials)
        xs = [t["out_tokens_est"] for t in trials]
        ys = [t["total_s"] for t in trials]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        if den == 0:
            return None, None
        b = num / den
        a = mean_y - b * mean_x
        rate = 1 / b if b > 0 else None
        return a, rate

    prefill_with, rate_with = fit_rate_and_prefill(with_trials)
    prefill_without, rate_without = fit_rate_and_prefill(without_trials)

    result = {
        "with_prefix_trials": with_trials,
        "without_prefix_trials": without_trials,
        "with_prefix_fit": {"prefill_s": prefill_with, "decode_rate_tok_per_s": rate_with},
        "without_prefix_fit": {"prefill_s": prefill_without, "decode_rate_tok_per_s": rate_without},
        "avg_total_with": sum(t["total_s"] for t in with_trials) / len(with_trials),
        "avg_total_without": sum(t["total_s"] for t in without_trials) / len(without_trials),
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open(SIMIT_ROOT + "/speed_experiments/bench_realistic_length_v2_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
