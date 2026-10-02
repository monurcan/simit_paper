"""SIMIT demo: a vision-language model imagines its own practice examples for
your question, then answers again with them in context.

Runs on Hugging Face ZeroGPU (each request uses the visitor's own GPU quota)
and on any machine with a CUDA GPU (``python app.py``).
"""

import os

# ZeroGPU starts a fresh process per request: keep Triton's autotuning results on disk
# (otherwise every request re-tunes the FP8 / linear-attention kernels, ~20 s).
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

import spaces  # before anything CUDA-related: ZeroGPU patches torch  # noqa: E402  # isort: skip

import base64  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from demo_models import BUDGETS, ENABLED, MODELS, code_snippet, run  # noqa: E402

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
EXAMPLES_DIR = HERE / "examples"


def _data_uri(path: Path) -> str:
    return "data:image/webp;base64," + base64.b64encode(path.read_bytes()).decode()


MASCOT = {p.stem: _data_uri(p) for p in ASSETS.glob("*.webp")}
STAGES = {  # stage -> (mascot, message)
    "idle": ("initial", "Ask me anything about an image"),
    "loading": ("initial", "Loading the model weights (first request only)"),
    "warmup": ("initial", "Waking up the model on the GPU"),
    "solving": ("initial", "Solving it on its own first"),
    "synthesizing": ("synthesizer", "Writing practice questions it already knows the answers to"),
    "routing": ("router", "Choosing how to draw each practice image"),
    "drawing": ("artist", "Painting a practice image"),
    "coding": ("architect", "Writing code for a chart, diagram or document"),
    "checking": ("critic", "Checking that each picture matches its question"),
    "improving": ("improved", "Answering again, with its own practice examples in context"),
}
LABELS = {MODELS[k].label: k for k in ENABLED}
BUDGET_LABELS = [f"{b} s" for b in BUDGETS]
GPU_OVERHEAD = 20  # seconds requested on top of the budget (worker start-up, final answer); weight transfer counts inside it


def _seconds(label: str) -> int:
    return int(str(label).split()[0])


# --------------------------------------------------------------------- HTML
def status_html(stage: str, elapsed: float = 0.0, limit: float = 0.0, n_demos: int = 0, note: str = "") -> str:
    mascot, message = STAGES.get(stage, STAGES["solving"])
    dots = "" if stage == "idle" else '<span class="simit-dots"></span>'
    bar = ""
    if limit:
        pct = max(2.0, min(100.0, 100.0 * elapsed / limit))
        bar = (f'<div class="simit-bar"><div style="width:{pct:.0f}%"></div></div>'
               f'<div class="simit-sub">{elapsed:.0f} s of {limit:.0f} s'
               + (f' · {n_demos} practice example{"s" if n_demos != 1 else ""} ready' if n_demos else "")
               + "</div>")
    return (f'<div class="simit-status"><img class="simit-mascot" src="{MASCOT[mascot]}" alt="">'
            f'<div><div class="simit-msg">{html.escape(message)}{dots}</div>'
            f'{bar}{f"<div class=simit-sub>{html.escape(note)}</div>" if note else ""}</div></div>')


def done_html(note: str) -> str:
    return (f'<div class="simit-status done"><img class="simit-mascot still" src="{MASCOT["improved"]}" alt="">'
            f'<div><div class="simit-msg">Done</div><div class="simit-sub">{html.escape(note)}</div></div></div>')


def error_html(note: str) -> str:
    return (f'<div class="simit-status error"><img class="simit-mascot still" src="{MASCOT["critic"]}" alt="">'
            f'<div><div class="simit-msg">Something went wrong</div><div class="simit-sub">{html.escape(note)}'
            '</div></div></div>')


def answer_card(kind: str, answer: str = "", sub: str = "", mark: str = "") -> str:
    title = "Base model (greedy)" if kind == "base" else "SIMIT-ICL (self-improved)"
    body = html.escape(answer) if answer else '<span class="simit-wait">…</span>'
    size = " long" if len(answer) > 60 else ""
    badge = {"right": '<span class="simit-ok">✓</span>', "wrong": '<span class="simit-no">✗</span>'}.get(mark, "")
    return (f'<div class="simit-card {kind}"><div class="simit-card-title">{title}{badge}</div>'
            f'<div class="simit-answer{size}">{body}</div><div class="simit-card-sub">{html.escape(sub)}</div></div>')


def demo_caption(d) -> str:
    bits = [f"Q: {d.question}", f"A: {d.answer}"]
    meta = [d.skill] if d.skill else []
    if d.verify_score is not None:
        meta.append(f"critic {d.verify_score}/100")
    if d.confidence is not None:
        meta.append(f"confidence {d.confidence:.2f}")
    return "\n".join(bits) + (f"\n({', '.join(meta)})" if meta else "")


# ------------------------------------------------------------- GPU requests
def _duration(image, question, key, budget_label, always, *args, **kwargs):
    return _seconds(budget_label) + GPU_OVERHEAD


def _session(image, question, key, budget_label, always):
    """Runs inside the GPU call: turns pipeline events into UI updates."""
    spec, budget = MODELS[key], _seconds(budget_label)
    try:
        yield from _events(spec, image, question, budget, always)
    except Exception as e:  # show it in the page instead of a bare error toast
        print(f"[demo] request failed: {type(e).__name__}: {e}", flush=True)
        yield (error_html(f"{type(e).__name__}: {str(e)[:300]}"), answer_card("base"), answer_card("simit"), [], "")


def _events(spec, image, question, budget, always):
    greedy, k_budget, demos, gallery = None, 0, [], []
    base, improved = answer_card("base"), answer_card("simit")
    status = status_html("warmup")
    for event in run(spec, image, question, budget, always):
        kind = event[0]
        if kind == "stage":
            _, stage, info = event
            status = status_html(stage, info.get("elapsed", 0), info.get("limit", 0), len(demos))
        elif kind == "greedy":
            _, zs, k_budget = event
            greedy = zs
            sub = f"confidence p0 = {zs.confidence:.2f}" if zs.confidence is not None else ""
            base = answer_card("base", zs.answer, sub)
        elif kind == "demo":
            demos.append(event[1])
            gallery = [(d.image, demo_caption(d)) for d in demos]
        else:  # final
            _, answer, info = event
            reason = info.get("skipped")
            if reason == "confident":
                sub = "The model was already confident, so SIMIT kept its answer (adaptive budget: 0 examples)."
            elif reason == "policy":
                sub = ("On general questions, our tuning found no gain from imagination for this model at this "
                       "budget, so SIMIT keeps the base answer. Tick 'Imagine even when confident' to try it anyway.")
            elif reason == "no_time":
                sub = "Not enough time budget left to imagine examples; kept the base answer."
            elif reason == "none_passed":
                sub = "No imagined example passed the checks in time; kept the base answer."
            else:
                changed = "changed" if answer.strip() != greedy.answer.strip() else "kept"
                sub = f"{changed} the answer after {info['demos']} imagined example{'s' if info['demos'] > 1 else ''}"
            improved = answer_card("simit", answer, sub)
            status = done_html(f"{info['elapsed']:.0f} s on the GPU")
        yield status, base, improved, gallery, _details(greedy, k_budget, demos, spec, budget)


@spaces.GPU(duration=_duration, size="large")
def gpu_large(image, question, key, budget_label, always):
    yield from _session(image, question, key, budget_label, always)


@spaces.GPU(duration=_duration, size="xlarge")
def gpu_xlarge(image, question, key, budget_label, always):
    yield from _session(image, question, key, budget_label, always)


def _details(greedy, k_budget, demos, spec, budget) -> str:
    if greedy is None:
        return ""
    p0 = f"{greedy.confidence:.2f}" if greedy.confidence is not None else "n/a"
    return (f"**{spec.label}**, {budget} s budget. Zero-shot confidence p0 = {p0}, so the adaptive budget asked "
            f"for **{k_budget}** example{'s' if k_budget != 1 else ''}; **{len(demos)}** passed the checks. "
            "Each example is a question the model wrote, whose answer it decided first, with an image made "
            "to fit that answer.")


def submit(image, question, model_label, budget_label, always):
    if image is None:
        raise gr.Error("Please upload an image.")
    if not question or not question.strip():
        raise gr.Error("Please type a question about the image.")
    key = LABELS[model_label]
    spec = MODELS[key]
    image = image.convert("RGB")
    if spec._weights is None:  # main process, CPU: no GPU time is spent here
        yield (status_html("loading"), answer_card("base"), answer_card("simit"), [], "")
        spec.load()
    fn = gpu_xlarge if spec.gpu_size == "xlarge" else gpu_large
    yield from fn(image, question.strip(), key, budget_label, always)


# ------------------------------------------------------------------ cached
def _load_examples():
    index = EXAMPLES_DIR / "index.json"
    return json.loads(index.read_text()) if index.exists() else []


EXAMPLES = [e for e in _load_examples() if e["model"] in ENABLED]


def show_cached(image, question, model_label, budget_label, always=False):
    """A cached example: the stored outputs of a real run, shown without using the GPU."""
    entry = next((e for e in EXAMPLES if e["question"] == question and MODELS[e["model"]].label == model_label
                  and f"{e['budget']} s" == budget_label and bool(e.get("always")) == bool(always)), None)
    if entry is None:
        return status_html("idle", note="Press Submit to run this example."), answer_card("base"), \
            answer_card("simit"), [], ""
    d = EXAMPLES_DIR / entry["id"]
    mark = lambda ok: "right" if ok else "wrong"  # noqa: E731
    gt = f"reference answer: {entry['reference']}"
    base = answer_card("base", entry["greedy"], f"confidence p0 = {entry['p0']:.2f} · {gt}", mark(entry["greedy_ok"]))
    n = len(entry["demos"])
    improved = answer_card("simit", entry["simit"], f"after {n} imagined example{'s' if n != 1 else ''} · {gt}",
                           mark(entry["simit_ok"]))
    gallery = [(str(d / x["image"]), "\n".join([f"Q: {x['question']}", f"A: {x['answer']}",
                                                f"({x['skill']}" + (f", critic {x['verify_score']}/100" if
                                                                    x.get("verify_score") is not None else "") + ")"]))
               for x in entry["demos"]]
    note = (f"Cached result from a real run ({entry['elapsed']:.0f} s on an H100, {entry['budget']} s budget"
            + (", imagining even when confident" if entry.get("always") else "") + f"). Source: {entry['source']}.")
    details = (f"**{MODELS[entry['model']].label}**, {entry['budget']} s budget. Zero-shot confidence "
               f"p0 = {entry['p0']:.2f}; {n} imagined example{'s' if n != 1 else ''} passed the checks.")
    return done_html(note), base, improved, gallery, details


# ---------------------------------------------------------------------- UI
CSS = (ASSETS / "style.css").read_text()
PAPER_URL = "https://monurcan.github.io/simit"
INTRO = f"""
<div class="simit-hero">
  <div class="simit-hero-top">
    <h1>SIMIT: models that imagine their own practice examples</h1>
    <a class="simit-paper" href="{PAPER_URL}" target="_blank" rel="noopener">Check our paper ↗</a>
  </div>
  <p>Ask any question about an image. The model first answers on its own. Then, without any labels, it writes
  similar practice questions whose answers it decides first, makes an image for each one, keeps only those it
  can verify, and answers your question again with them in context.</p>
</div>
"""

with gr.Blocks(title="SIMIT demo") as demo:
    gr.HTML(INTRO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(type="pil", label="Image", height=360)
            question = gr.Textbox(label="Question", placeholder="e.g. In what country would you find this hat?",
                                  lines=2)
            model = gr.Dropdown(list(LABELS), value=next(iter(LABELS)), label="Model")
            budget = gr.Radio(BUDGET_LABELS, value="60 s", label="Max GPU time per request",
                              info="More time lets the model imagine and check more practice examples.")
            with gr.Accordion("Advanced", open=False):
                always = gr.Checkbox(False, label="Imagine even when the model is already confident",
                                     info="By default SIMIT skips imagination for confident answers.")
            run_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=6):
            status = gr.HTML(status_html("idle", note="Upload an image and type a question, or pick an example below."))
            with gr.Row():
                base_out = gr.HTML(answer_card("base"), min_width=260)
                simit_out = gr.HTML(answer_card("simit"), min_width=260)
            with gr.Accordion("See the imagined practice examples", open=False):
                details = gr.Markdown()
                gallery = gr.Gallery(columns=4, height=300, object_fit="contain", show_label=False)
            with gr.Accordion("Use SIMIT in your own code", open=False):
                code = gr.Code(code_snippet(MODELS[next(iter(LABELS.values()))], 60, ""), language="python",
                               interactive=False)

    outputs = [status, base_out, simit_out, gallery, details]
    # On ZeroGPU each request gets its own GPU process; run locally, requests share this process and GPU.
    from spaces.config import Config as _SpacesConfig
    run_btn.click(submit, [image, question, model, budget, always], outputs,
                  concurrency_limit=4 if _SpacesConfig.zero_gpu else 1)

    def _code(model_label, budget_label, q, force):
        return code_snippet(MODELS[LABELS[model_label]], _seconds(budget_label), q or "", force)
    for comp in (model, budget, question, always):
        comp.change(_code, [model, budget, question, always], code, queue=False)

    if EXAMPLES:
        gr.Markdown("### Examples\nCached results from real runs of this demo (click one; press Submit to "
                    "run it again live).")
        examples = gr.Gallery(
            [(str(EXAMPLES_DIR / e["id"] / e["query_file"]),
              f"{e['question'].splitlines()[0]}  ({MODELS[e['model']].label})") for e in EXAMPLES],
            columns=4, height=300 * ((len(EXAMPLES) + 3) // 4) + 20, object_fit="cover", allow_preview=False,
            show_label=False)

        def pick(evt: gr.SelectData):
            e = EXAMPLES[evt.index]
            inputs = [Image.open(EXAMPLES_DIR / e["id"] / e["query_file"]), e["question"],
                      MODELS[e["model"]].label, f"{e['budget']} s", bool(e.get("always"))]
            return (*inputs, *show_cached(*inputs))
        examples.select(pick, None, [image, question, model, budget, always] + outputs)
    gr.Markdown("Models: " + ", ".join(f"[{MODELS[k].label}](https://huggingface.co/{MODELS[k].repo})"
                                      for k in ENABLED)
                + ". Each request uses your own ZeroGPU quota for the selected time budget.")

if os.environ.get("SIMIT_DEMO_PRELOAD", "1") == "1":
    for key in ENABLED:  # load every model on CPU before serving (main process, no CUDA)
        MODELS[key].load()

if __name__ == "__main__":
    demo.queue(max_size=32).launch(css=CSS, theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber"))
