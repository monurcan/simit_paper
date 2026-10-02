"""For each sample in a benchmark subset: save the original sample
(image/prompt/answer), ask the model for K diverse VQA examples inspired by
the image, then render each example's description through the skill
library (common/skills/pipeline.py) unchanged.

Ported from similar_samples_llms_eval_lite/synthesize_vqa.py, simplified per
project scope (SIMIT-ICL only, no ablations):
  - Only the BFS fill-to-capacity strategy (the original's own measurements
    found it 1.9x-3.5x fewer LLM calls than the legacy DFS strategy for the
    same K slots filled -- no reason to carry DFS forward).
  - No --no-diversity-prompt / --fixed-question-prompt / --fixed-image-prompt
    / --force-natural ablation flags.
  - No mstar HTTP server / watchdog (SenseNovaClient loads the model
    in-process via vLLM-omni's Omni; there is no separate server process to
    stall or restart).

Defaults match the original pipeline's own "best config"
(submit_discard_unverified.sh): verify_all=True, discard_unverified=True,
verify_criterion="score", confidence_threshold=50, verify_think_mode=True.
Note this "confidence" is the critic's own self-reported 0-100 quality
SCORE (verify.py), unrelated to the avg_answer_prob teacher-forced
confidence used for difficulty filtering / adaptive budget allocation
(confidence/ -- a separate, later stage, same as in the original pipeline).
"""
import argparse
import os
import re
import sys
from pathlib import Path

SIMIT_ROOT = os.environ.get("SIMIT_ROOT") or str(Path(__file__).resolve().parents[2])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset, load_from_disk

from common.benchmark_config import SUBSET_CONFIG, build_answer_text, build_prompt, get_image
from common.model_clients import get_client
from common.skills.pipeline import generate_visualization
from common.skills.sandbox import clean_response

GEN_WIDTH = 400
GEN_HEIGHT = 400

_ALL_GEN_BATCH_PROMPT = (
    "I will give you an image. "
    "Generate exactly {K} diverse visual question-answering training examples.\n\n"
    "Each example must have:\n"
    "- A visual question (about any visual concept — NOT tied to the given image)\n"
    "- A short correct answer\n"
    "- A detailed description of a new image where your question and answer are correct, your description should be detailed and specific, one should be able to visualize the image perfectly from the description\n\n"
    "DIVERSITY RULES:\n"
    "- Cover as many different visual reasoning types as possible across the {K} examples:\n"
    "  counting, color/attribute, spatial position, object recognition, action/activity,\n"
    "  text/OCR, yes/no judgment, comparison, scene type, etc.\n"
    "- Use the provided image only as inspiration for difficulty level and visual domain.\n"
    "  The generated examples do NOT need to match the provided image.\n"
    "- Answers must be short (1-5 words).\n\n"
    "Output format (blank line between examples):\n\n"
    "Question: [your visual question]\n"
    "Answer: [short correct answer]\n"
    "Description: [detailed description of a new image where question+answer are correct]\n\n"
    "Question: [another diverse visual question]\n"
    "Answer: [short correct answer]\n"
    "Description: [description for this example]\n\n"
    "... (exactly {K} Question/Answer/Description triples)"
)

_ALL_GEN_RE = re.compile(
    r"Question:\s*(.+?)\nAnswer:\s*(.+?)\nDescription:\s*(.+?)"
    r"(?=\n\n(?:\d+[.)]\s*|Example\s+\d+:?\s*\n)?Question:|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_FORMAT_REMINDER = (
    "\n\nIMPORTANT: your previous response did not follow the required output "
    "format. Output ONLY plain text in exactly this form, nothing else -- no "
    "markdown bold, no numbering, no preamble or summary:\n\n"
    "Question: ...\nAnswer: ...\nDescription: ...\n\n"
    "Question: ...\nAnswer: ...\nDescription: ...\n\n(blank line between each of the {K} examples)"
)

# --- Lance-specific triplet synthesis -------------------------------------
# Lance (common/model_clients/lance_client.py) cannot follow
# _ALL_GEN_BATCH_PROMPT: it is a much smaller, narrowly-specialized model
# whose only reliably-trained text-generation task is single-turn, direct
# question->short-answer (see its own bytedance/Lance config/examples/
# x2t_image_example.json -- every example is exactly that shape). Confirmed
# empirically: 8+ independent one-shot attempts at this batch/multi-field
# prompt (K=1 and K=3, across sampling settings, with and without the format-
# reminder retry above) produced either a plain caption of the input image or
# degenerate token repetition, never one parseable triple -- see
# environment/vllm_omni_patches/README.md for the investigation. Decomposing
# into three separate single-turn calls, each shaped like Lance's own native
# task, reliably produces good, diverse, correctly-shaped triples instead.
# This is Lance-only: SensenovaClient callers still use
# generate_synthetic_examples/_ALL_GEN_BATCH_PROMPT above, unchanged.
_LANCE_DIVERSITY_TOPICS = [
    "counting", "color or attribute", "spatial position", "object recognition",
    "action or activity", "text or OCR", "a yes/no judgment", "comparison", "scene type",
]

_LANCE_QUESTION_PROMPT_TMPL = (
    "Here are examples of short visual questions used to test other pictures:\n"
    "- How many birds are on the wire?\n"
    "- What color is the umbrella?\n"
    "- Is the door open or closed?\n\n"
    "Write ONE new short visual question about {topic}, on a completely "
    "different topic than this image -- the question does NOT need to match "
    "this image, use it only as inspiration for difficulty level and domain. "
    "Output only the question, nothing else."
)
_LANCE_ANSWER_PROMPT_TMPL = (
    'Give a short 1-5 word answer to this question, as if answering about '
    'some other picture: "{question}"\n{type_hint}Output only the answer, nothing else.'
)
_LANCE_DESCRIPTION_PROMPT_TMPL = (
    'Imagine a different picture. In that picture, the question "{question}" '
    'has the answer "{answer}". Describe that picture in detail: what it '
    "shows, the setting, colors, and objects."
)

# Rejects the malformed-question/description artifacts observed empirically
# (bounding-box-looking coordinate lists like "[0.04, 0.32, 0.16, 0.39]") --
# a light sanity gate, not a correctness check; _fill_slots_bfs's existing
# retry loop resamples a fresh question on rejection.
_LANCE_COORD_LIST_RE = re.compile(r"^[\d.,\[\]\s-]+$")

# Rejects questions where the model talks ABOUT the question/task instead of
# just asking it -- also observed empirically (e.g. "One sharp short question
# about counting ... is: ...", "... that's an excellent question! As I
# mentioned earlier ..."), which _LANCE_QUESTION_PROMPT_TMPL's few-shot demo
# alone doesn't fully prevent.
_LANCE_META_LEAK_RE = re.compile(
    r"\bquestion (about|is)\b|excellent question|as i mentioned|short visual question|different topic than this image",
    re.IGNORECASE,
)

# (question-prefix regex, answer-type hint) -- a light heuristic so the
# separate answer-generation call (which has no other signal tying it to the
# question's expected answer shape) doesn't produce a type-mismatched answer
# (e.g. "No" to "What color ..."), confirmed to happen empirically without
# this hint.
_LANCE_ANSWER_TYPE_HINTS = [
    (re.compile(r"^how many\b", re.IGNORECASE), "Answer with just a number. "),
    (re.compile(r"^(what|which) color", re.IGNORECASE), "Answer with just a color name. "),
    (re.compile(r"^(is|are|do|does|can|will|was|were|has|have)\b", re.IGNORECASE), "Answer with Yes or No. "),
]


def _lance_answer_type_hint(question: str) -> str:
    for pattern, hint in _LANCE_ANSWER_TYPE_HINTS:
        if pattern.match(question):
            return hint
    return ""


def _lance_clean_line(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    line = text.splitlines()[0].strip().strip('"')
    line = re.sub(r"^question:\s*", "", line, flags=re.IGNORECASE)
    return re.sub(r"^\d+[.)]\s*", "", line)


def _lance_question_ok(question: str) -> bool:
    if len(question.split()) < 3 or not question.endswith("?"):
        return False
    if _LANCE_COORD_LIST_RE.match(question) or _LANCE_META_LEAK_RE.search(question):
        return False
    return True


def _lance_description_ok(description: str) -> bool:
    return len(description) >= 20 and not _LANCE_COORD_LIST_RE.match(description)


def generate_synthetic_examples_lance(client, image, k, max_attempts=None, think_mode=False):
    """Lance-specific drop-in replacement for generate_synthetic_examples:
    same return shape (list of up to k (question, answer, description)
    triples), built from three separate single-turn Lance calls per example
    -- see the comment above _LANCE_QUESTION_PROMPT_TMPL."""
    max_attempts = max_attempts or k * 6
    png = _png_bytes(image)
    results = []
    attempts = 0
    while len(results) < k and attempts < max_attempts:
        topic = _LANCE_DIVERSITY_TOPICS[attempts % len(_LANCE_DIVERSITY_TOPICS)]
        attempts += 1
        try:
            q_resp = client.chat(
                _LANCE_QUESTION_PROMPT_TMPL.format(topic=topic), images=png,
                think_mode=think_mode, max_output_tokens=40, temperature=0.8,
            )
            question = _lance_clean_line(q_resp.text)
            if not _lance_question_ok(question):
                print(f"  [lance-synth attempt {attempts}] rejected question: {question!r}")
                continue

            a_resp = client.chat(
                _LANCE_ANSWER_PROMPT_TMPL.format(question=question, type_hint=_lance_answer_type_hint(question)),
                images=png,
                think_mode=think_mode, max_output_tokens=16, temperature=0.8,
            )
            answer = _lance_clean_line(a_resp.text)
            if not answer:
                continue

            d_resp = client.chat(
                _LANCE_DESCRIPTION_PROMPT_TMPL.format(question=question, answer=answer), images=png,
                think_mode=think_mode, max_output_tokens=150, temperature=0.8,
            )
            description = d_resp.text.strip()
            if not _lance_description_ok(description):
                print(f"  [lance-synth attempt {attempts}] rejected description: {description[:80]!r}")
                continue
        except Exception as e:
            print(f"  [lance-synth attempt {attempts}] chat error: {e}")
            continue
        results.append((question, answer, description))
    return results


def generate_synthetic_examples(client, image, k, max_attempts=3, think_mode=False):
    prompt = _ALL_GEN_BATCH_PROMPT.format(K=k)
    best = []
    for attempt in range(max_attempts):
        try:
            resp = client.chat(
                prompt, images=_png_bytes(image), think_mode=think_mode,
                max_output_tokens=1800, temperature=0.2 if attempt == 0 else 0.7,
            )
        except Exception as e:
            print(f"  [attempt {attempt}] chat error: {e}")
            continue
        text = clean_response(resp.text).replace("**", "")

        def _clean_desc(d: str) -> str:
            sep = d.find("\n---")
            return (d[:sep] if sep != -1 else d).strip()

        matches = [(q.strip(), a.strip(), _clean_desc(d)) for q, a, d in _ALL_GEN_RE.findall(text)]
        if len(matches) > len(best):
            best = matches
        if len(best) >= k:
            break
        prompt = _ALL_GEN_BATCH_PROMPT.format(K=k) + _FORMAT_REMINDER.format(K=k)
    return best


def _png_bytes(image) -> bytes:
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _sample_accepted_count(sample_dir: Path) -> int:
    n = 0
    while (sample_dir / f"synthesized_sample_{n}.png").exists():
        n += 1
    return n


def _sample_done(out_root: Path, subset: str, idx: int, k: int) -> bool:
    sample_dir = out_root / subset / f"sample{idx}"
    if (sample_dir / "SHORTFALL.txt").exists():
        return True
    return _sample_accepted_count(sample_dir) >= k


def _generate_examples(client, image, k, backend, think_mode):
    """Dispatch to the backend-appropriate triplet-generation function --
    generate_synthetic_examples (SensenovaClient, unchanged) or
    generate_synthetic_examples_lance (LanceClient, see its docstring for why
    Lance needs a different prompting strategy)."""
    if backend == "lance":
        return generate_synthetic_examples_lance(client, image, k, think_mode=think_mode)
    return generate_synthetic_examples(client, image, k, think_mode=think_mode)


def _fill_slots_bfs(client, image, sample_dir, triples_queue, accepted, k, max_triplet_attempts,
                     verify_all, think_mode, verify_criterion, confidence_threshold,
                     verify_think_mode, discard_unverified, backend):
    """Phase 1 (breadth): try each triplet once, no self-refine/revise.
    Phase 2+ (repair rounds): round-robin the retry pool with full
    generate_with_verification treatment. See the original synthesize_vqa.py
    for the measurements motivating BFS-over-DFS."""
    total_attempts = 0
    retry_pool = []

    def try_one(question, answer, description, *, single_shot):
        nonlocal accepted, total_attempts
        total_attempts += 1
        slot = accepted
        out_png = sample_dir / f"synthesized_sample_{slot}.png"
        (sample_dir / f"synthesized_sample_{slot}_prompt.txt").write_text(question)
        (sample_dir / f"synthesized_sample_{slot}_answers.txt").write_text(answer)
        (sample_dir / f"synthesized_sample_{slot}_description.txt").write_text(description)
        kwargs = dict(
            width=GEN_WIDTH, height=GEN_HEIGHT, verify_all=verify_all, think_mode=think_mode,
            confidence_threshold=confidence_threshold, discard_unverified=discard_unverified,
            criterion=verify_criterion, verify_think_mode=verify_think_mode,
            # Lance-only: skip classify_request/DSL-extraction entirely and
            # always render through the native image-generation head.
            # generate_visualization's force_natural flag already exists for
            # exactly this (originally for a classifier ablation) -- Lance
            # has near-zero code-generation ability (FIGURE) and unreliable
            # structured-DSL extraction (DIAGRAM/GRAPH/etc., see
            # common/model_clients/lance_client.py's module docstring), but
            # excellent native t2i quality, and this project's synthesis
            # prompts are photo-domain-inspired anyway (see
            # generate_synthetic_examples_lance), so NATURAL is the right
            # category regardless of what classify_request would guess.
            # SensenovaClient is unaffected (force_natural=False, unchanged).
            force_natural=(backend == "lance"),
            # NATURAL's own critic loop runs regardless of verify_all (see
            # generate_visualization's docstring: "verify_strategy/width/
            # height only affect NATURAL by default") and asks for the same
            # kind of structured verdict/score output Lance can't reliably
            # produce (confirmed empirically: every attempt came back
            # score=None -> treated as failed, even though the underlying
            # image generated fine) -- skip it for Lance the same way
            # --no-verify-all skips it for the other categories.
            verify_strategy="none" if backend == "lance" else "informed",
        )
        if single_shot:
            kwargs["retries"] = 0
            kwargs["verify_max_attempts"] = 1
        try:
            ok = generate_visualization(client, description, str(out_png), **kwargs)
        except Exception as e:
            print(f"    slot{slot} (attempt {total_attempts}/{max_triplet_attempts}): ERROR {e}")
            ok = False
        print(f"    slot{slot} (attempt {total_attempts}/{max_triplet_attempts}): gen_ok={ok}")
        if ok:
            accepted += 1
            return True
        for suffix in ("_prompt.txt", "_answers.txt", "_description.txt"):
            (sample_dir / f"synthesized_sample_{slot}{suffix}").unlink(missing_ok=True)
        out_png.unlink(missing_ok=True)
        return False

    phase1_budget = max(k, max_triplet_attempts - k)
    while accepted < k and total_attempts < phase1_budget:
        if not triples_queue:
            fresh = _generate_examples(client, image, 1, backend, think_mode)
            if not fresh:
                total_attempts += 1
                continue
            triples_queue.extend(fresh)
        q, a, d = triples_queue.pop(0)
        if not try_one(q, a, d, single_shot=True):
            retry_pool.append((q, a, d))

    while accepted < k and total_attempts < max_triplet_attempts and retry_pool:
        next_pool = []
        for q, a, d in retry_pool:
            if accepted >= k or total_attempts >= max_triplet_attempts:
                next_pool.append((q, a, d))
                continue
            if not try_one(q, a, d, single_shot=False):
                next_pool.append((q, a, d))
        retry_pool = next_pool

    return accepted, total_attempts


def process_sample(client, out_root, subset, idx, example, cfg, k, max_triplet_attempts,
                    verify_all, think_mode, verify_criterion, confidence_threshold,
                    verify_think_mode, discard_unverified, backend):
    sample_dir = out_root / subset / f"sample{idx}"
    if _sample_done(out_root, subset, idx, k):
        print(f"  sample{idx}: already done, skipping")
        return
    sample_dir.mkdir(parents=True, exist_ok=True)

    image = get_image(example, cfg)
    question_text = build_prompt(example, cfg)
    if not (sample_dir / "original_sample.png").exists():
        image.save(sample_dir / "original_sample.png")
        (sample_dir / "original_sample_prompt.txt").write_text(question_text)
        (sample_dir / "original_sample_answers.txt").write_text(build_answer_text(example, cfg))

    accepted = _sample_accepted_count(sample_dir)
    triples_queue = list(_generate_examples(client, image, k, backend, think_mode))
    print(f"  sample{idx}: parsed {len(triples_queue)}/{k} synthesized examples, {accepted}/{k} already accepted")

    accepted, total_attempts = _fill_slots_bfs(
        client, image, sample_dir, triples_queue, accepted, k, max_triplet_attempts,
        verify_all, think_mode, verify_criterion, confidence_threshold, verify_think_mode, discard_unverified,
        backend,
    )

    if accepted < k:
        print(f"  sample{idx}: only {accepted}/{k} accepted after {total_attempts} triplet attempts, shipping short")
        (sample_dir / "SHORTFALL.txt").write_text(
            f"only {accepted}/{k} synths accepted after {total_attempts} triplet attempts\n"
        )


def process_subset(client, out_root, subset, cfg, k, max_triplet_attempts, limit, start_idx,
                    verify_all, think_mode, verify_criterion, confidence_threshold,
                    verify_think_mode, discard_unverified, backend, split="lite"):
    if split == "val50":
        # 50-sample validation split (rag_retrieval/build_val50_datasets.py), written to <bench>_val50/
        ds = load_from_disk(f"{os.environ.get('SIMIT_DATA') or os.path.join(SIMIT_ROOT, 'workdir')}/val50/{subset}_lite")["val"]
    elif "hf_repo" in cfg:
        ds = load_dataset(cfg["hf_repo"], split=cfg["hf_split"])
    else:
        ds = load_dataset("lmms-lab/LMMs-Eval-Lite", subset)["lite"]
    folder = f"{subset}_val50" if split == "val50" else subset
    n = len(ds) if limit is None else min(limit, len(ds))
    print(f"\n=== {folder} ({n}/{len(ds)} samples, start={start_idx}) ===")
    for idx in range(start_idx, n):
        if _sample_done(out_root, folder, idx, k):
            continue
        print(f"-- {folder} sample{idx} --")
        try:
            process_sample(
                client, out_root, folder, idx, ds[idx], cfg, k, max_triplet_attempts,
                verify_all, think_mode, verify_criterion, confidence_threshold,
                verify_think_mode, discard_unverified, backend,
            )
        except Exception as e:
            print(f"  ERROR on sample{idx}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subsets", nargs="+", help="benchmark subset name(s), e.g. docvqa_val ok_vqa_val2014")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--split", default="lite", choices=["lite", "val50"],
                    help="lite: the 500 eval queries; val50: the 50 validation queries (hyperparameter tuning)")
    ap.add_argument("--k", type=int, default=6, help="synthetic samples per real sample (K_max=6 for full-scale runs)")
    ap.add_argument("--max-triplet-attempts", type=int, default=None, help="default: K*4")
    ap.add_argument("--think", action="store_true", help="enable think_mode on generation calls (default off, matching the original pipeline's best config)")
    ap.add_argument("--no-verify-all", action="store_true")
    ap.add_argument("--no-discard-unverified", action="store_true")
    ap.add_argument("--verify-criterion", default="score", choices=["verdict", "verdict_confidence", "score", "score_structured"])
    ap.add_argument("--confidence-threshold", type=int, default=50)
    ap.add_argument("--no-verify-think", action="store_true")
    ap.add_argument("--model", default="sensenova", choices=["sensenova", "lance"],
                     help="UMM backend (default: sensenova, matching the original pipeline)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    max_triplet_attempts = args.max_triplet_attempts or args.k * 4

    print(f"model={args.model} K={args.k} think_mode={args.think} verify_all={not args.no_verify_all} "
          f"discard_unverified={not args.no_discard_unverified} verify_criterion={args.verify_criterion} "
          f"confidence_threshold={args.confidence_threshold} verify_think_mode={not args.no_verify_think} "
          f"max_triplet_attempts={max_triplet_attempts}")

    client = get_client(args.model)

    for subset in args.subsets:
        if subset not in SUBSET_CONFIG:
            print(f"WARNING: unknown subset '{subset}', skipping (known: {list(SUBSET_CONFIG)})")
            continue
        process_subset(
            client, out_root, subset, SUBSET_CONFIG[subset], args.k, max_triplet_attempts,
            args.limit, args.start_idx,
            not args.no_verify_all, args.think, args.verify_criterion, args.confidence_threshold,
            not args.no_verify_think, not args.no_discard_unverified, args.model, split=args.split,
        )
    print("\nDONE")


if __name__ == "__main__":
    main()
