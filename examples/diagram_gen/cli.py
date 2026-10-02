"""Test-harness entry point: runs every prompt in TEST_PROMPTS through
generate_visualization(), captions each output with its source prompt, and
prints a final pass/fail summary."""

import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mstar import MStarClient

from .pipeline import generate_visualization
from .test_prompts import TEST_PROMPTS
from .test_prompts_hard import HARD_TEST_PROMPTS
from .test_prompts_web import WEB_TEST_PROMPTS

OUTPUT_DIR = Path(__file__).parent.parent / "diagram_outputs"
HARD_OUTPUT_DIR = Path(__file__).parent.parent / "diagram_outputs_hard"
WEB_OUTPUT_DIR = Path(__file__).parent.parent / "diagram_outputs_web"


def add_caption(image_path: str, request_text: str) -> None:
    """Append the original request text as a wrapped caption strip below the
    image -- a test-review convenience so a reviewer can see what prompt
    produced a given output without cross-referencing the log. Kept out of
    generate_visualization() itself since it's a reporting concern for this
    test harness, not part of the pipeline's actual output contract."""
    import textwrap

    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    try:
        import matplotlib
        font_path = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
        font = ImageFont.truetype(font_path, 16)
    except Exception:
        font = ImageFont.load_default()

    chars_per_line = max(20, img.width // 8)
    lines = textwrap.wrap(f"Prompt: {request_text}", width=chars_per_line)
    line_height = font.getbbox("Ay")[3] + 6
    caption_height = line_height * len(lines) + 16

    canvas = Image.new("RGB", (img.width, img.height + caption_height), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((10, img.height + 8 + i * line_height), line, fill="black", font=font)
    canvas.save(image_path)


def main() -> None:
    # --hard: run the separate, harder/longer-description benchmark suite
    # (test_prompts_hard.py) into its own output directory.
    # --web: run the SVG/Mermaid/Vega-Lite benchmark suite (test_prompts_web.py).
    hard = "--hard" in sys.argv
    web = "--web" in sys.argv
    if web:
        prompts = WEB_TEST_PROMPTS
        out_dir = WEB_OUTPUT_DIR
    elif hard:
        prompts = HARD_TEST_PROMPTS
        out_dir = HARD_OUTPUT_DIR
    else:
        prompts = TEST_PROMPTS
        out_dir = OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)
    # --resume: skip prompts that already have an output PNG, so a run that
    # was interrupted (e.g. by a server crash) can continue from where it
    # left off instead of regenerating everything from scratch.
    resume = "--resume" in sys.argv
    # --concurrency=N: submit N requests at once instead of one at a time.
    # Measured ~18% faster wall-clock on a mixed 16-prompt batch, because
    # fast classify/DSL-extraction/render calls for one prompt overlap with
    # another prompt's slow NATURAL image-generation call. It does NOT
    # parallelize image generation itself -- the server has one GPU/worker,
    # so concurrent image-gen calls queue rather than run side by side.
    concurrency = 1
    for arg in sys.argv:
        if arg.startswith("--concurrency="):
            concurrency = int(arg.split("=", 1)[1])
    # --verify=none|plain|informed: verification strategy for NATURAL-category
    # output (default "informed"); ignored by every other category -- see
    # verify.py for why it's scoped to NATURAL only.
    verify_strategy = "informed"
    for arg in sys.argv:
        if arg.startswith("--verify="):
            verify_strategy = arg.split("=", 1)[1]

    todo = []
    results = {}
    for name, request in prompts.items():
        out_path = str(out_dir / f"{name}.png")
        if resume and Path(out_path).exists():
            print(f"\n=== {name} === (already done, skipping)")
            results[name] = True
            continue
        todo.append((name, request, out_path))

    def run_one(name: str, request: str, out_path: str) -> bool:
        print(f"\n=== {name} ===\nRequest: {request}")
        client = MStarClient("http://localhost:8000")
        try:
            # Single general entry point -- it classifies the request itself
            # and picks the right pipeline; the test harness no longer needs
            # to know in advance which category a request belongs to.
            ok = generate_visualization(client, request, out_path, verify_strategy=verify_strategy)
            if ok:
                add_caption(out_path, request)
            return ok
        except Exception as e:
            # A transient server/network error on one prompt (e.g. a 500
            # from the inference server) shouldn't abort a 200-prompt run --
            # record it as a failure and keep going.
            print(f"  unhandled error: {e}")
            return False

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(run_one, name, request, out_path): name for name, request, out_path in todo}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
    else:
        for name, request, out_path in todo:
            results[name] = run_one(name, request, out_path)

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} figures rendered successfully")


if __name__ == "__main__":
    main()
