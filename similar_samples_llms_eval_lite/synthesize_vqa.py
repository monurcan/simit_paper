"""For each sample in each LMMs-Eval-Lite subset: save the original
sample (image/prompt/answer), ask BAGEL for K diverse VQA examples inspired
by the image, then render each example's description through the existing
diagram_gen pipeline unchanged.

--split=lite (default) processes the 500 eval queries of each subset and writes
<out-root>/<subset>/sample{i}; --split=val50 processes the 50 validation queries
(built by rag_retrieval/build_val50_datasets.py, used only for hyperparameter
tuning) and writes <out-root>/<subset>_val50/sample{i}."""

import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SIMIT_ROOT = os.environ.get("SIMIT_ROOT") or str(Path(__file__).resolve().parents[1])
SIMIT_DATA = os.environ.get("SIMIT_DATA") or os.path.join(SIMIT_ROOT, "workdir")

sys.path.insert(0, SIMIT_ROOT)
sys.path.insert(0, SIMIT_ROOT + "/similar_samples_llms_eval_lite")

from datasets import load_dataset, load_from_disk
from mstar import MStarClient
from examples.diagram_gen.pipeline import generate_visualization
from examples.diagram_gen.sandbox import clean_response
from examples.diagram_gen.verify import verify
from vqa_synth_config import SUBSET_CONFIG, get_image, build_prompt, build_answer_text

_SERVER_URL = "http://localhost:8000"  # overridden by --port= arg in main()
_SERVER_GPU: str | None = None         # --gpu= arg; None = keep the CUDA_VISIBLE_DEVICES the
                                       # scheduler assigned (never force a device under LSF/SLURM)
_SOCKET_PATH_PREFIX: str | None = None # overridden by --socket-prefix= arg in main()
_MSTAR_BIN = SIMIT_ROOT + "/.venv/bin/mstar"
_consecutive_errors = 0
# Two consecutive 120s chat timeouts mean the server is stalled (healthy chats
# answer in <30s); waiting for a third just wastes another 2 minutes.
_ERROR_RESTART_THRESHOLD = 2
# Image-gen errors are caught inside generate_natural and return ok=False rather
# than raising. Track consecutive gen failures separately and probe the server
# with a real generate_image call to distinguish a broken image-gen path from
# a legitimately hard sample where all categories fail for content reasons.
_consecutive_gen_failures = 0
# Probe after 2 consecutive gen failures. Threshold 1 was tried and reverted:
# DSL-category content failures (parse/render, common on chart/document-heavy
# subsets) are indistinguishable from infra failures here, and probing every
# one of them cost 31 probes/hour on a healthy server. With fail-fast timeouts
# a genuinely stalled server is still detected within ~2 failed synths.
_GEN_FAILURE_PROBE_THRESHOLD = 2


def _probe_image_gen() -> bool:
    """Return True if the server can actually generate an image right now.
    Must use the actual generation size — the worker corruption only manifests
    when generating many image tokens; a tiny 64x64 probe passes even when
    400x400 is broken, causing false 'content-related' diagnoses."""
    try:
        c = MStarClient(_SERVER_URL, timeout=120.0)
        c.generate_image("a red circle on white background", width=GEN_WIDTH, height=GEN_HEIGHT)
        return True
    except Exception:
        return False


def _kill_pid_tree(root_pid: int, exclude_pid: int) -> None:
    """Kill root_pid and all its descendants, skipping exclude_pid (ourselves)."""
    # Build parent→children map from ps
    result = subprocess.run(
        ["ps", "axo", "pid,ppid", "--no-headers"],
        capture_output=True, text=True,
    )
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            pid, ppid = int(parts[0]), int(parts[1])
            children.setdefault(ppid, []).append(pid)

    to_kill: list[int] = []
    queue = [root_pid]
    while queue:
        p = queue.pop()
        if p != exclude_pid:
            to_kill.append(p)
            queue.extend(children.get(p, []))

    for p in to_kill:
        try:
            os.kill(p, 9)
        except ProcessLookupError:
            pass


def _sweep_stale_mstar_processes(port: str) -> None:
    """Kill stale mstar server processes for the given port from previous runs.
    Port-specific so two concurrent servers on different ports don't interfere.
    Also kills orphaned resource_tracker/spawn_main from any previous mstar run.
    Protects the current process's entire subtree from accidental killing."""
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["ps", "axo", "pid,ppid,command", "--no-headers"],
            capture_output=True, text=True, timeout=10,
        )

        pid_to_ppid: dict[int, int] = {}
        pid_to_cmd: dict[int, str] = {}
        for line in result.stdout.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            try:
                pid, ppid, cmd = int(parts[0]), int(parts[1]), parts[2]
                pid_to_ppid[pid] = ppid
                pid_to_cmd[pid] = cmd
            except (ValueError, IndexError):
                pass

        # Build parent→children map
        parent_to_children: dict[int, list[int]] = {}
        for pid, ppid in pid_to_ppid.items():
            parent_to_children.setdefault(ppid, []).append(pid)

        # Protect the entire subtree rooted at my_pid
        my_subtree: set[int] = {my_pid}
        queue: list[int] = [my_pid]
        while queue:
            p = queue.pop()
            for child in parent_to_children.get(p, []):
                if child not in my_subtree:
                    my_subtree.add(child)
                    queue.append(child)

        to_kill: set[int] = set()
        for pid, cmd in pid_to_cmd.items():
            if pid in my_subtree or "grep" in cmd:
                continue
            # Only kill the mstar server on OUR SPECIFIC PORT (prevents killing a
            # sibling server running on a different port on the same node)
            if "mstar serve" in cmd and f"--port {port}" in cmd:
                to_kill.add(pid)
            # Orphaned resource_tracker/spawn_main from mstar venv (parent died → ppid=1)
            # These are port-agnostic but definitively stale since parent is gone
            elif "mstar" in cmd and ("resource_tracker" in cmd or "spawn_main" in cmd):
                ppid = pid_to_ppid.get(pid, 0)
                if ppid == 1 or ppid not in pid_to_ppid:
                    to_kill.add(pid)

        # Expand to all descendants of found processes (excluding my subtree)
        queue = list(to_kill)
        while queue:
            p = queue.pop()
            for child in parent_to_children.get(p, []):
                if child not in my_subtree and child not in to_kill:
                    to_kill.add(child)
                    queue.append(child)

        pids_killed = []
        for pid in to_kill:
            try:
                os.kill(pid, 9)
                pids_killed.append(pid)
            except (ProcessLookupError, PermissionError):
                pass

        if pids_killed:
            print(f"  [startup] swept stale mstar PIDs (port {port}): {pids_killed}")
            time.sleep(5)
        else:
            print(f"  [startup] no stale mstar processes found (port {port})")

        # Clean stale IPC socket files so next mstar can bind without ZMQ errors
        import glob as _glob, getpass as _getpass
        _prefix = _SOCKET_PATH_PREFIX or _getpass.getuser()
        for _ipc in _glob.glob(f"/tmp/mstar_{_prefix}/*.ipc"):
            try:
                os.unlink(_ipc)
            except Exception:
                pass
    except Exception as e:
        print(f"  [startup] stale sweep failed: {e}")


def _wait_gpu_released(timeout_s: int = 120) -> None:
    """Block until no compute process holds our GPU (or timeout). Starting the
    new server while the killed worker still holds ~50GB causes the new worker
    to boot into a degraded state that passes warmup but stalls on real load."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
                + (["-i", _SERVER_GPU] if _SERVER_GPU else []),
                capture_output=True, text=True, timeout=15,
            )
            if not result.stdout.strip():
                return
        except Exception:
            return  # nvidia-smi unavailable — don't block the restart on it
        time.sleep(5)
    print(f"  [watchdog] WARN: GPU {_SERVER_GPU} still busy after {timeout_s}s, proceeding anyway")


def _restart_server(max_attempts: int = 3) -> bool:
    """Kill and restart the server. Returns True if server came back up, False otherwise.
    Never raises — all exceptions are caught internally."""
    print("  [watchdog] server stalled — killing and restarting...")
    port = _SERVER_URL.rsplit(":", 1)[-1]
    my_pid = os.getpid()

    for attempt in range(max_attempts):
        # Sweep stale mstar processes on our specific port
        _sweep_stale_mstar_processes(port)

        # Also find+kill by port in case sweep missed anything
        try:
            result = subprocess.run(
                f"ps aux | grep 'mstar serve.*--port {port}' | grep -v grep | awk '{{print $2}}'",
                shell=True, capture_output=True, text=True,
            )
            for spid in result.stdout.split():
                try:
                    _kill_pid_tree(int(spid), my_pid)
                except Exception:
                    pass
        except Exception as e:
            print(f"  [watchdog] kill failed: {e}")

        _wait_gpu_released()
        time.sleep(5)
        import socket as _socket
        _hostname = _socket.gethostname()
        _log_dir = Path(SIMIT_DATA) / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(_log_dir / f"mstar_server_{_hostname}_{port}.log")
        cmd = [_MSTAR_BIN, "serve", "bagel", "--port", port] + (["--gpus", _SERVER_GPU] if _SERVER_GPU else [])
        if _SOCKET_PATH_PREFIX:
            cmd += ["--socket-path-prefix", _SOCKET_PATH_PREFIX]
        try:
            subprocess.Popen(cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT)
        except Exception as e:
            print(f"  [watchdog] Popen failed: {e}")
            time.sleep(30)
            continue

        # Wait for /health
        import urllib.request
        healthy = False
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{_SERVER_URL}/health", timeout=5)
                healthy = True
                break
            except Exception:
                time.sleep(5)
        if not healthy:
            print(f"  [watchdog] /health never responded (attempt {attempt + 1}), retrying...")
            continue

        time.sleep(10)

        # Verify chat is alive. Must allow full model-loading time: on a
        # cold-NFS node the first chat blocks until weights finish loading
        # (10+ min), and a short timeout here kills servers that are merely
        # loading — producing an endless kill/reload cycle.
        try:
            c = MStarClient(_SERVER_URL, timeout=600.0)
            c.chat("Hi.", think_mode=False, max_output_tokens=5)
        except Exception as e:
            print(f"  [watchdog] chat warmup failed: {e} (attempt {attempt + 1}), retrying...")
            continue

        # Warm up image gen with the actual generation size — the stall bug only
        # manifests at full resolution (64x64 probes pass even when broken).
        img_ok = False
        for _ in range(30):
            try:
                c.generate_image("a red circle on white background", width=GEN_WIDTH, height=GEN_HEIGHT)
                img_ok = True
                break
            except Exception as e:
                print(f"  [watchdog] image warmup error: {e}")
                time.sleep(10)
        if not img_ok:
            print(f"  [watchdog] image warmup failed (attempt {attempt + 1}), retrying...")
            continue

        print("  [watchdog] server back up")
        return True

    print("  [watchdog] WARN: server did not recover after all restart attempts")
    return False


def _make_client() -> MStarClient:
    return MStarClient(_SERVER_URL, timeout=120.0)


def _is_infra_error(e: Exception) -> bool:
    """Only network/server failures indicate a hung server; code-level errors
    (e.g. AttributeError on a malformed model response) are content problems —
    counting them toward the restart threshold caused healthy-server restarts."""
    s = str(e).lower()
    return "timed out" in s or "connection" in s or "500" in s


def _record_error():
    global _consecutive_errors
    _consecutive_errors += 1
    if _consecutive_errors >= _ERROR_RESTART_THRESHOLD:
        _restart_server()  # always returns, never raises
        _consecutive_errors = 0


def _record_success():
    global _consecutive_errors, _consecutive_gen_failures
    _consecutive_errors = 0
    _consecutive_gen_failures = 0


def _record_gen_failure() -> bool:
    """Call when generate_visualization returns False for a NATURAL image.
    Returns True when the server is verified healthy afterwards (probe passed
    or restart succeeded) — i.e. retrying the failed synth is worthwhile."""
    global _consecutive_gen_failures
    _consecutive_gen_failures += 1
    if _consecutive_gen_failures >= _GEN_FAILURE_PROBE_THRESHOLD:
        print(f"  [watchdog] {_consecutive_gen_failures} consecutive gen failures — probing image gen...")
        _consecutive_gen_failures = 0
        if not _probe_image_gen():
            print("  [watchdog] image gen broken, restarting server")
            return _restart_server()
        print("  [watchdog] probe passed, failures were content-related")
        return True
    return False

OUT_ROOT = Path(SIMIT_DATA) / "generated_samples"  # overridden by --out-root=
_SPLIT = "lite"  # overridden by --split=lite|val50


def _subset_dir(subset: str) -> Path:
    return OUT_ROOT / (f"{subset}_val50" if _SPLIT == "val50" else subset)

_VERIFY_ALL = False  # overridden by --verify-all arg: critic-verify EVERY category
_THINK_MODE = True  # overridden by --no-think arg: disable think_mode on every Bagel call
_VERIFY_THINK_MODE: bool | None = None  # overridden by --verify-think: keep the critic/revise
    # calls thinking independently of _THINK_MODE. An A/B test (12 real samples, manual
    # 1/0.5/0 scoring) found "think everywhere" and "think only in verify" tie on output
    # quality (0.50 avg both) while the latter is far cheaper -- see pipeline.py's
    # generate_natural docstring. None = fall back to _THINK_MODE (old uniform behavior).
_DISCARD_UNVERIFIED = False  # overridden by --discard-unverified: drop synths that never pass verification
                              # instead of keeping the last-generated-but-unverified image; implies verify_all
_CONFIDENCE_THRESHOLD: int | None = None  # overridden by --confidence-threshold=N: require the critic's
                                           # metric (per _VERIFY_CRITERION) >= N to accept a synth
_VERIFY_CRITERION = "verdict"  # overridden by --verify-criterion=verdict|verdict_confidence|score
                                # see examples/diagram_gen/verify.py::verify() docstring
K = 4  # overridden by --k=N
_MAX_TRIPLET_ATTEMPTS: int | None = None  # overridden by --max-triplet-attempts=N: total
    # fresh-triplet attempts per sample (fill-to-capacity: keep pulling new triplets, not
    # just revising the same one, until K are accepted or this cap is hit). Defaults to
    # K*4 (computed in main() once K is final) if left unset.
_STRATEGY = "bfs"  # overridden by --strategy=bfs|dfs -- see _fill_slots_bfs/_fill_slots_dfs.
    # bfs (default): breadth before depth -- try every triplet once (no self-refine, no
    # revise) before spending retry budget on any one of them. dfs (legacy): exhaust one
    # triplet's full retry/revise budget before moving to the next.
_FORCE_NATURAL = False  # overridden by --force-natural: skips classify_request/deterministic
    # renderers entirely, forcing every triplet through the NATURAL (model image-gen head)
    # path -- ablation for measuring the classifier + structured-renderer components' value.
_VERIFY_STRATEGY = "informed"  # overridden by --no-verify: sets this to "none", which makes
    # generate_with_verification a single-shot passthrough (see verify.py) -- NATURAL images
    # are kept as generated with no critic call at all, and combined with verify_all=False
    # (the default) non-NATURAL categories never call the critic either, so nothing in a
    # sample ever gets verified. --discard-unverified/--verify-all/--confidence-threshold
    # are then no-ops (nothing left to discard/threshold), so don't combine them with this.
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

_ALL_GEN_BATCH_PROMPT_NO_DIVERSITY = (
    "I will give you an image. "
    "Generate exactly {K} visual question-answering training examples.\n\n"
    "Each example must have:\n"
    "- A visual question (about any visual concept — NOT tied to the given image)\n"
    "- A short correct answer\n"
    "- A detailed description of a new image where your question and answer are correct, your description should be detailed and specific, one should be able to visualize the image perfectly from the description\n\n"
    "  The generated examples do NOT need to match the provided image.\n"
    "- Answers must be short (1-5 words).\n\n"
    "Output format (blank line between examples):\n\n"
    "Question: [your visual question]\n"
    "Answer: [short correct answer]\n"
    "Description: [detailed description of a new image where question+answer are correct]\n\n"
    "Question: [another visual question]\n"
    "Answer: [short correct answer]\n"
    "Description: [description for this example]\n\n"
    "... (exactly {K} Question/Answer/Description triples)"
)
_NO_DIVERSITY_PROMPT = False  # overridden by --no-diversity-prompt: drops the DIVERSITY
    # RULES block and "diverse" wording from _ALL_GEN_BATCH_PROMPT -- ablation for measuring
    # the diversity-prompting component's effect on triplet generation.

_ALL_GEN_BATCH_PROMPT_FIXED_QUESTION = (
    "I will give you an image and a fixed question: \"{QUESTION}\"\n\n"
    "Generate exactly {K} diverse visual question-answering training examples that "
    "all use this EXACT question, but each with a different plausible answer.\n\n"
    "Each example must have:\n"
    "- A plausible short correct answer to the fixed question above\n"
    "- A detailed description of a new image where the fixed question, asked about that image, "
    "would have this answer\n\n"
    "CORRECTNESS RULE (most important):\n"
    "- The description must fully justify the answer: someone reading ONLY the description, "
    "without ever seeing any image, must be able to answer the fixed question correctly and "
    "unambiguously just from the text.\n"
    "- If the question refers to labels, letters, options (e.g. A/B/C/D), numbers, or "
    "positions/stages that name specific things in the image (e.g. \"the item labeled D\", "
    "\"Option A) c\", \"Stage 2\"), your description must explicitly spell out what each "
    "referenced label/option corresponds to in the new image -- do not just describe generic "
    "scene content and expect the label mapping to be assumed or reused from the original image.\n"
    "- Your description should be detailed and specific enough that one could visualize the "
    "image perfectly and independently re-derive the answer from it alone.\n\n"
    "DIVERSITY RULES:\n"
    "- Cover as many different plausible answers as possible across the {K} examples -- avoid "
    "repeating the same or near-identical answer twice.\n"
    "- Use the provided image only as inspiration for difficulty level and visual domain.\n"
    "  The generated examples do NOT need to match the provided image.\n"
    "- Answers must be short (1-5 words).\n\n"
    "Output format (blank line between examples):\n\n"
    "Answer: [short plausible answer]\n"
    "Description: [detailed description of a new image where the fixed question above has this "
    "answer, explicitly grounding any labels/options/positions referenced in the question]\n\n"
    "Answer: [another plausible, different answer]\n"
    "Description: [description for this example]\n\n"
    "... (exactly {K} Answer/Description pairs)"
)
_FIXED_QUESTION_PROMPT = False  # overridden by --fixed-question-prompt: the triplet-synthesis
    # question is fixed to the real test sample's own question (counterfactual ablation) --
    # the model only generates (answer, description) pairs, and the code enriches them with
    # the fixed question to form full triplets.

_ALL_GEN_BATCH_PROMPT_FIXED_IMAGE = (
    "I will give you an image. The original question someone asked about this image was: "
    "\"{QUESTION}\"\n\n"
    "Generate {K} questions about this image and their answers. Each question should be "
    "similar in style and topic to the original question above, but the {K} questions "
    "should still be diverse from each other -- don't just repeat the original question "
    "verbatim, and avoid repeating the same or near-identical question twice.\n\n"
    "CORRECTNESS RULE (most important):\n"
    "- Each answer must be verifiably correct by looking at the image alone -- do not invent "
    "objects, text, counts, or details that are not actually present in the image.\n\n"
    "Output format (blank line between examples):\n\n"
    "Question: [your question about this image]\n"
    "Answer: [correct answer]\n\n"
    "Question: [another question about this image]\n"
    "Answer: [correct answer]\n\n"
    "... (exactly {K} Question/Answer pairs)"
)
_FIXED_IMAGE_PROMPT = False  # overridden by --fixed-image-prompt: the triplet-synthesis
    # image is fixed to the real test sample's own image (counterfactual ablation) -- the
    # model only generates (question, answer) pairs about that image, and the "image
    # realization" stage is replaced by a mock that just copies the original image back out
    # unchanged (no renderers/image-gen head involved at all).

_QUESTION_ANSWER_RE = re.compile(
    # \n+ (not \n\n): with free-form (now unbounded-length) answers, the model
    # sometimes packs consecutive Question/Answer pairs with only a single
    # newline between them (no blank line) -- requiring a literal blank line
    # here made the Answer group swallow the next pair whole when that
    # happened, corrupting both.
    r"Question:\s*(.+?)\nAnswer:\s*(.+?)"
    r"(?=\n+(?:\d+[.)]\s*|Example\s+\d+:?\s*\n)?Question:|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_FORMAT_REMINDER_FIXED_IMAGE = (
    "\n\nIMPORTANT: your previous response did not follow the required output "
    "format. Output ONLY plain text in exactly this form, nothing else -- no "
    "markdown bold, no numbering, no preamble or summary:\n\n"
    "Question: ...\nAnswer: ...\n\n"
    "Question: ...\nAnswer: ...\n\n(blank line between each of the {K} examples)"
)


def _fixed_image_brief(question: str, answer: str) -> str:
    """The 'brief' checked against the (unchanged) image by the critic in
    fixed-image mode -- stands in for the textual image description used
    everywhere else in the pipeline."""
    return f'The image supports answering the question "{question}" with the answer "{answer}".'


_ANSWER_DESC_RE = re.compile(
    r"Answer:\s*(.+?)\nDescription:\s*(.+?)"
    r"(?=\n\n(?:\d+[.)]\s*|Example\s+\d+:?\s*\n)?Answer:|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_FORMAT_REMINDER_FIXED_QUESTION = (
    "\n\nIMPORTANT: your previous response did not follow the required output "
    "format. Output ONLY plain text in exactly this form, nothing else -- no "
    "markdown bold, no numbering, no preamble or summary:\n\n"
    "Answer: ...\nDescription: ...\n\n"
    "Answer: ...\nDescription: ...\n\n(blank line between each of the {K} examples)"
)

_ALL_GEN_RE = re.compile(
    # BAGEL doesn't always follow the literal template -- observed variants
    # include a leading "Example N:" line and "1. " numbering before
    # "Question:". Tolerating these in the boundary lookahead (not just the
    # leading match) matters most: without it, example 1's Description
    # swallows the rest of the response, since the lookahead never finds a
    # bare "Question:" to stop at, silently dropping examples 2..K instead
    # of just losing the formatting on one field. Markdown bold markers
    # ("**Question:**", "**Question**:", ...) are stripped separately before
    # matching -- they showed up wrapping the keyword in inconsistent
    # positions, so stripping "**" outright is far more robust than trying
    # to enumerate every position they can appear in.
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


def generate_synthetic_examples(
    client: MStarClient, image, k: int = K, max_attempts: int = 3, think_mode: bool = True,
    question: str | None = None,
):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    if _FIXED_IMAGE_PROMPT:
        base_prompt = _ALL_GEN_BATCH_PROMPT_FIXED_IMAGE
        prompt = base_prompt.format(K=k, QUESTION=question)
    elif _FIXED_QUESTION_PROMPT:
        base_prompt = _ALL_GEN_BATCH_PROMPT_FIXED_QUESTION
        prompt = base_prompt.format(K=k, QUESTION=question)
    else:
        base_prompt = _ALL_GEN_BATCH_PROMPT_NO_DIVERSITY if _NO_DIVERSITY_PROMPT else _ALL_GEN_BATCH_PROMPT
        prompt = base_prompt.format(K=k)
    best = []
    for attempt in range(max_attempts):
        try:
            # Retries need MORE randomness, not less: a low-temperature retry
            # with the same context reproduces the same malformed output (see
            # the identical finding documented in generate_figure).
            resp = client.chat(
                prompt, images=buf.getvalue(), think_mode=think_mode,
                max_output_tokens=1800, temperature=0.2 if attempt == 0 else 0.7,
            )
            _record_success()
        except Exception as e:
            print(f"  [attempt {attempt}] chat error: {e}")
            _record_error()
            continue
        text = clean_response(resp.text).replace("**", "")
        def _clean_desc(d: str) -> str:
            sep = d.find("\n---")
            return (d[:sep] if sep != -1 else d).strip()
        if _FIXED_IMAGE_PROMPT:
            matches = [(q.strip(), a.strip(), _fixed_image_brief(q.strip(), a.strip()))
                       for q, a in _QUESTION_ANSWER_RE.findall(text)]
        elif _FIXED_QUESTION_PROMPT:
            matches = [(question, a.strip(), _clean_desc(d)) for a, d in _ANSWER_DESC_RE.findall(text)]
        else:
            matches = [(q.strip(), a.strip(), _clean_desc(d)) for q, a, d in _ALL_GEN_RE.findall(text)]
        if len(matches) > len(best):
            best = matches
        if len(best) >= k:
            break
        if _FIXED_IMAGE_PROMPT:
            prompt = base_prompt.format(K=k, QUESTION=question) + _FORMAT_REMINDER_FIXED_IMAGE.format(K=k)
        elif _FIXED_QUESTION_PROMPT:
            prompt = base_prompt.format(K=k, QUESTION=question) + _FORMAT_REMINDER_FIXED_QUESTION.format(K=k)
        else:
            prompt = base_prompt.format(K=k) + _FORMAT_REMINDER.format(K=k)
    return best


def _sample_accepted_count(sample_dir: Path) -> int:
    """Slots are filled in order as triplets get accepted (fill-to-capacity:
    see process_sample) -- count the contiguous run of accepted PNGs from 0."""
    n = 0
    while (sample_dir / f"synthesized_sample_{n}.png").exists():
        n += 1
    return n


def _sample_done(subset: str, idx: int) -> bool:
    # Done when K synths were accepted, or when a SHORTFALL marker says we
    # already exhausted _MAX_TRIPLET_ATTEMPTS trying to fill the rest (written
    # by process_sample) -- avoids re-spending the full attempt budget on an
    # already-exhausted sample every multi-pass sweep.
    sample_dir = _subset_dir(subset) / f"sample{idx}"
    if (sample_dir / "SHORTFALL.txt").exists():
        return True
    return _sample_accepted_count(sample_dir) >= K


def generate_fixed_image(
    client: MStarClient, image, request: str, out_path: str,
    verify_strategy: str = "informed", think_mode: bool = True,
    confidence_threshold: int | None = None, discard_unverified: bool = False,
    criterion: str = "verdict", verify_think_mode: bool | None = None,
) -> bool:
    """--fixed-image-prompt entry point: mirrors generate_visualization's role
    in the pipeline but skips classify_request and every renderer/image-gen
    call -- the output is always the original test image, unchanged; only the
    (question, answer) varies. `request` is the critic brief built from that
    pair (see _fixed_image_brief). Since the image can never change, a failed
    verification can't be fixed by resampling/revising like
    generate_with_verification does for real generation, so this calls
    verify() only ONCE and reports failure as an ordinary generation failure
    -- the caller's fill-to-capacity retry then pulls a fresh (question,
    answer) pair for the slot instead."""
    image.save(out_path, format="PNG")
    if verify_strategy == "none":
        return True
    try:
        passed, critique, metric = verify(
            client, request, out_path,
            think_mode=verify_think_mode if verify_think_mode is not None else think_mode,
            criterion=criterion,
        )
    except Exception as e:
        if discard_unverified:
            print(f"  [verify] verify error, discarding: {e}")
            Path(out_path).unlink(missing_ok=True)
            return False
        print(f"  [verify] verify error, accepting: {e}")
        return True
    accepted = passed and (metric is None or confidence_threshold is None or metric >= confidence_threshold)
    metric_str = f" {criterion}={metric}" if criterion != "verdict" else ""
    print(f"  [verify] passed={passed}{metric_str}")
    if accepted:
        return True
    if discard_unverified:
        Path(out_path).unlink(missing_ok=True)
        return False
    return True  # keep the (always identical) image even though it failed verification


def _generate_slot(client: MStarClient, image, description: str, out_path: str, verify_all: bool, *, single_shot: bool) -> bool:
    """Dispatch one slot's generation+verification to either the normal
    classify/render/verify pipeline or (--fixed-image-prompt) the fixed-image
    mock, depending on mode."""
    if _FIXED_IMAGE_PROMPT:
        return generate_fixed_image(
            client, image, description, out_path,
            verify_strategy=_VERIFY_STRATEGY, think_mode=_THINK_MODE,
            confidence_threshold=_CONFIDENCE_THRESHOLD, discard_unverified=_DISCARD_UNVERIFIED,
            criterion=_VERIFY_CRITERION, verify_think_mode=_VERIFY_THINK_MODE,
        )
    kwargs = dict(
        width=GEN_WIDTH, height=GEN_HEIGHT, verify_all=verify_all,
        think_mode=_THINK_MODE, confidence_threshold=_CONFIDENCE_THRESHOLD,
        discard_unverified=_DISCARD_UNVERIFIED, criterion=_VERIFY_CRITERION,
        verify_think_mode=_VERIFY_THINK_MODE, verify_strategy=_VERIFY_STRATEGY,
        force_natural=_FORCE_NATURAL,
    )
    if single_shot:
        kwargs["retries"] = 0
        kwargs["verify_max_attempts"] = 1
    return generate_visualization(client, description, out_path, **kwargs)


def _fill_slots_dfs(client: MStarClient, image, sample_dir: Path, triples_queue: list, accepted: int, verify_all: bool, question: str | None = None) -> tuple[int, int]:
    """Legacy strategy (--strategy=dfs): each triplet gets generate_visualization's
    full retry/revise budget (default retries=2, generate_with_verification's
    default max_attempts=5) before we give up on it and pull a fresh one. See
    _fill_slots_bfs's docstring for why this is no longer the default -- most
    of that budget is spent on triplets that were going to fail anyway."""
    total_attempts = 0
    while accepted < K and total_attempts < _MAX_TRIPLET_ATTEMPTS:
        if not triples_queue:
            fresh = generate_synthetic_examples(client, image, 1, think_mode=_THINK_MODE, question=question)
            if not fresh:
                total_attempts += 1
                continue
            triples_queue.extend(fresh)
        question, answer, description = triples_queue.pop(0)
        total_attempts += 1
        slot = accepted
        out_png = sample_dir / f"synthesized_sample_{slot}.png"
        # Save metadata before image gen so a restart picks up per-slot state.
        (sample_dir / f"synthesized_sample_{slot}_prompt.txt").write_text(question)
        (sample_dir / f"synthesized_sample_{slot}_answers.txt").write_text(answer)
        (sample_dir / f"synthesized_sample_{slot}_description.txt").write_text(description)
        # Synths are independent: a chat error (e.g. in classify_request) must
        # not abort the sample and skip the remaining synths.
        try:
            ok = _generate_slot(client, image, description, str(out_png), verify_all, single_shot=False)
        except Exception as e:
            print(f"    slot{slot} (triplet attempt {total_attempts}/{_MAX_TRIPLET_ATTEMPTS}): ERROR {e}")
            if _is_infra_error(e):
                _record_error()
            ok = False
        if not ok and _record_gen_failure():
            # Server verified healthy (probe passed or restart succeeded) —
            # retry once instead of leaving a hole that needs a manual rerun.
            print(f"    slot{slot}: retrying after recovery")
            try:
                ok = _generate_slot(client, image, description, str(out_png), verify_all, single_shot=False)
            except Exception as e:
                print(f"    slot{slot}: retry ERROR {e}")
                if _is_infra_error(e):
                    _record_error()
                ok = False
        print(f"    slot{slot} (triplet attempt {total_attempts}/{_MAX_TRIPLET_ATTEMPTS}): gen_ok={ok}")
        if ok:
            _record_success()
            accepted += 1
        else:
            # This triplet didn't pan out -- clear its files so the next
            # triplet can reuse the same slot cleanly.
            for suffix in ("_prompt.txt", "_answers.txt", "_description.txt"):
                (sample_dir / f"synthesized_sample_{slot}{suffix}").unlink(missing_ok=True)
            out_png.unlink(missing_ok=True)
    return accepted, total_attempts


def _fill_slots_bfs(client: MStarClient, image, sample_dir: Path, triples_queue: list, accepted: int, verify_all: bool, question: str | None = None) -> tuple[int, int]:
    """Default strategy: breadth before depth.

    Measured on real generation logs (2026-08-19, ai2d/docvqa_val/infovqa_val/
    mmbench_cn_dev): 32-80% of triplets that ever get accepted are accepted on
    their very FIRST render+verify attempt, and P(fully accepted on the first
    try alone) already gets K=4 slots filled within the existing
    _MAX_TRIPLET_ATTEMPTS budget for the median sample. The expensive part of
    the old (dfs) strategy -- generate_visualization's self-refine retries and
    generate_with_verification's revise-on-failure retries -- was mostly being
    spent on triplets that were going to fail regardless, instead of trying
    more triplets. Measured speedup on those 4 subsets: 1.9x-3.5x fewer LLM
    calls per K=4 slots filled.

    Phase 1: try each triplet exactly once (retries=0, verify_max_attempts=1
    -- no self-refine repair, no revise-on-failed-verification). A triplet
    that fails keeps its (question, answer, description) in a retry pool
    instead of being discarded outright -- these are the "intermediate
    outputs" phase 2+ retries.
    Phase 2, 3, ...: once phase 1's budget is spent (or exhausted of fresh
    triplets) and slots remain unfilled, round-robin over the retry pool
    giving each one attempt with the FULL retry/revise treatment per round;
    survivors carry over to the next round. This still gives every failed
    triplet a real shot at self-refine/revise recovery -- it's just spread
    across many candidates per round instead of exhausted on one candidate
    before ever trying the next (which is what --strategy=dfs does)."""
    total_attempts = 0
    retry_pool: list[tuple[str, str, str]] = []

    def try_one(question: str, answer: str, description: str, *, single_shot: bool) -> bool:
        nonlocal accepted, total_attempts
        total_attempts += 1
        slot = accepted
        out_png = sample_dir / f"synthesized_sample_{slot}.png"
        (sample_dir / f"synthesized_sample_{slot}_prompt.txt").write_text(question)
        (sample_dir / f"synthesized_sample_{slot}_answers.txt").write_text(answer)
        (sample_dir / f"synthesized_sample_{slot}_description.txt").write_text(description)
        phase = "phase1" if single_shot else "retry"
        try:
            ok = _generate_slot(client, image, description, str(out_png), verify_all, single_shot=single_shot)
        except Exception as e:
            print(f"    slot{slot} (bfs {phase}, attempt {total_attempts}/{_MAX_TRIPLET_ATTEMPTS}): ERROR {e}")
            if _is_infra_error(e):
                _record_error()
            ok = False
        print(f"    slot{slot} (bfs {phase}, attempt {total_attempts}/{_MAX_TRIPLET_ATTEMPTS}): gen_ok={ok}")
        if ok:
            _record_success()
            accepted += 1
            return True
        # Same watchdog protection as dfs -- consecutive real failures still
        # trigger the image-gen probe/restart, we just don't act on the
        # "worth an immediate manual retry" return value here since the retry
        # pool (phase 2+) already provides that.
        _record_gen_failure()
        for suffix in ("_prompt.txt", "_answers.txt", "_description.txt"):
            (sample_dir / f"synthesized_sample_{slot}{suffix}").unlink(missing_ok=True)
        out_png.unlink(missing_ok=True)
        return False

    # Phase 1: breadth. Reserve at least K attempts' worth of budget for the
    # repair rounds so phase 1 can't spend the entire budget on ever-fresh
    # triplets and leave nothing for phase 2+.
    phase1_budget = max(K, _MAX_TRIPLET_ATTEMPTS - K)
    while accepted < K and total_attempts < phase1_budget:
        if not triples_queue:
            fresh = generate_synthetic_examples(client, image, 1, think_mode=_THINK_MODE, question=question)
            if not fresh:
                total_attempts += 1
                continue
            triples_queue.extend(fresh)
        q, answer, description = triples_queue.pop(0)
        if not try_one(q, answer, description, single_shot=True):
            retry_pool.append((q, answer, description))

    # Phase 2+: repair rounds, one full-strength attempt per candidate per
    # round, round-robin over whatever's still unfilled.
    while accepted < K and total_attempts < _MAX_TRIPLET_ATTEMPTS and retry_pool:
        next_pool = []
        for question, answer, description in retry_pool:
            if accepted >= K or total_attempts >= _MAX_TRIPLET_ATTEMPTS:
                next_pool.append((question, answer, description))
                continue
            if not try_one(question, answer, description, single_shot=False):
                next_pool.append((question, answer, description))
        retry_pool = next_pool

    return accepted, total_attempts


def process_sample(client: MStarClient, subset: str, idx: int, example: dict, cfg: dict) -> None:
    sample_dir = _subset_dir(subset) / f"sample{idx}"
    if _sample_done(subset, idx):
        print(f"  sample{idx}: already done, skipping")
        return
    sample_dir.mkdir(parents=True, exist_ok=True)

    image = get_image(example, cfg)
    question_text = build_prompt(example, cfg)
    if not (sample_dir / "original_sample.png").exists():
        image.save(sample_dir / "original_sample.png")
        (sample_dir / "original_sample_prompt.txt").write_text(question_text)
        (sample_dir / "original_sample_answers.txt").write_text(build_answer_text(example, cfg))

    # discard_unverified implies every category gets critic-verified -- discarding
    # unverified synths without ever verifying them would be a no-op.
    verify_all = _VERIFY_ALL or _DISCARD_UNVERIFIED

    # Fill-to-capacity: slots are filled in order as triplets get ACCEPTED, not
    # pre-assigned to the K triplets generated up front. If a triplet's own
    # concept is the problem (e.g. an overly specific invented number/label a
    # renderer can't hit), no amount of retrying that same triplet helps --
    # pull a fresh one from generate_synthetic_examples and keep going, up to
    # _MAX_TRIPLET_ATTEMPTS total tries, rather than shipping a slot's worth
    # short the moment one triplet concept turns out to be hard.
    accepted = _sample_accepted_count(sample_dir)
    triples_queue = list(generate_synthetic_examples(
        client, image, K, think_mode=_THINK_MODE,
        question=question_text if (_FIXED_QUESTION_PROMPT or _FIXED_IMAGE_PROMPT) else None,
    ))
    print(f"  sample{idx}: parsed {len(triples_queue)}/{K} synthesized examples, {accepted}/{K} already accepted")

    fill_fn = _fill_slots_dfs if _STRATEGY == "dfs" else _fill_slots_bfs
    accepted, total_attempts = fill_fn(
        client, image, sample_dir, triples_queue, accepted, verify_all,
        question=question_text if (_FIXED_QUESTION_PROMPT or _FIXED_IMAGE_PROMPT) else None,
    )

    if accepted < K:
        print(f"  sample{idx}: only {accepted}/{K} accepted after {total_attempts} triplet attempts, shipping short")
        (sample_dir / "SHORTFALL.txt").write_text(
            f"only {accepted}/{K} synths accepted after {total_attempts} triplet attempts\n"
        )


def process_subset(subset: str, cfg: dict, limit: int | None = None, start_idx: int = 0) -> int:
    """Run one pass over the subset. Returns the number of still-incomplete samples."""
    if _SPLIT == "val50":
        ds = load_from_disk(f"{SIMIT_DATA}/val50/{subset}_lite")["val"]
    elif "hf_repo" in cfg:
        ds = load_dataset(cfg["hf_repo"], split=cfg["hf_split"])
    else:
        ds = load_dataset("lmms-lab/LMMs-Eval-Lite", subset)["lite"]
    n = len(ds) if limit is None else min(limit, len(ds))
    print(f"\n=== {subset} [{_SPLIT}] ({n}/{len(ds)} samples, start={start_idx}) ===")
    for idx in range(start_idx, n):
        # Check before ds[idx]: decoding a dataset row costs ~0.5s, which
        # multiplied by 8500 mostly-done samples makes each sweep pass slow.
        if _sample_done(subset, idx):
            continue
        print(f"-- {subset} sample{idx} --")
        try:
            # Re-create client each sample so a post-restart client picks up cleanly
            process_sample(_make_client(), subset, idx, ds[idx], cfg)
        except Exception as e:
            print(f"  ERROR on sample{idx}: {e}")
            try:
                _record_error()
            except Exception as re:
                print(f"  ERROR in _record_error: {re} — continuing")
    return sum(1 for idx in range(start_idx, n) if not _sample_done(subset, idx))


def _ensure_server_running() -> None:
    """Start the server if not up, then warmup chat+image to ensure the model is loaded.
    Blocks until fully ready. Never raises."""
    import urllib.request
    already_up = False
    try:
        urllib.request.urlopen(f"{_SERVER_URL}/health", timeout=5)
        already_up = True
    except Exception:
        pass

    if not already_up:
        # Sweep stale mstar processes before starting — leftover resource_tracker/spawn_main
        # from previous jobs on this node cause "Deferring cleanup of tensor" hangs that
        # make every inference return 500 indefinitely.
        port = _SERVER_URL.rsplit(":", 1)[-1]
        print(f"  [startup] sweeping stale mstar processes (port {port})...")
        _sweep_stale_mstar_processes(port)
        print(f"  [startup] server not up at {_SERVER_URL}, starting...")
        import socket as _socket
        _hostname = _socket.gethostname()
        _log_dir = Path(SIMIT_DATA) / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(_log_dir / f"mstar_server_{_hostname}_{port}.log")
        cmd = [_MSTAR_BIN, "serve", "bagel", "--port", port] + (["--gpus", _SERVER_GPU] if _SERVER_GPU else [])
        if _SOCKET_PATH_PREFIX:
            cmd += ["--socket-path-prefix", _SOCKET_PATH_PREFIX]
        try:
            subprocess.Popen(cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT)
        except Exception as e:
            print(f"  [startup] Popen failed: {e}")
            return
        healthy = False
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{_SERVER_URL}/health", timeout=5)
                healthy = True
                break
            except Exception:
                time.sleep(5)
        if not healthy:
            print(f"  [startup] WARNING: server never responded to /health")
            return
        print(f"  [startup] server healthy at {_SERVER_URL}")
        time.sleep(10)

    # Warmup: verify chat works (triggers and waits for model loading).
    # Use a long timeout (600s) since NFS model loading can take 10+ minutes.
    print(f"  [startup] warming up chat (model loading may take 10+ min)...")
    c = MStarClient(_SERVER_URL, timeout=600.0)
    _consec_500s = 0
    _consec_errors = 0
    for attempt in range(30):
        try:
            c.chat("Hi.", think_mode=False, max_output_tokens=5)
            print(f"  [startup] chat warmup ok (attempt {attempt})")
            _consec_500s = 0
            _consec_errors = 0
            break
        except Exception as e:
            _consec_errors += 1
            print(f"  [startup] chat warmup error (attempt {attempt}): {e}, retrying in 15s...")
            if "500" in str(e):
                _consec_500s += 1
                if _consec_500s >= 5:
                    print(f"  [startup] 5 consecutive 500s — sweeping and restarting server...")
                    _restart_server(max_attempts=2)
                    _consec_500s = 0
                    _consec_errors = 0
            elif "timed out" in str(e).lower():
                # The 600s budget exists only for cold NFS model loading; a chat
                # that times out after 600s means the server is hung, full stop.
                # Waiting for 3 such timeouts stalled startup for over 30 min.
                print(f"  [startup] warmup chat timed out after 600s — server hung, restarting...")
                _restart_server(max_attempts=2)
                _consec_500s = 0
                _consec_errors = 0
            elif _consec_errors >= 3:
                # 3 consecutive connection errors — server is hung, restart
                print(f"  [startup] 3 consecutive non-500 errors — restarting server...")
                _restart_server(max_attempts=2)
                _consec_500s = 0
                _consec_errors = 0
            else:
                _consec_500s = 0
            time.sleep(15)

    # Warmup image gen with actual size — the stall bug manifests at 400x400
    print(f"  [startup] warming up image gen ({GEN_WIDTH}x{GEN_HEIGHT})...")
    for attempt in range(30):
        try:
            c.generate_image("a red circle on white background", width=GEN_WIDTH, height=GEN_HEIGHT)
            print(f"  [startup] image warmup ok (attempt {attempt})")
            break
        except Exception as e:
            print(f"  [startup] image warmup error (attempt {attempt}): {e}, retrying in 15s...")
            time.sleep(15)
    print(f"  [startup] server ready at {_SERVER_URL}")


def main():
    global _SERVER_URL, _SERVER_GPU, _SOCKET_PATH_PREFIX, OUT_ROOT, _VERIFY_ALL, _SPLIT
    global _THINK_MODE, _VERIFY_THINK_MODE, _DISCARD_UNVERIFIED, _CONFIDENCE_THRESHOLD, _VERIFY_CRITERION, K, _MAX_TRIPLET_ATTEMPTS, _STRATEGY, _VERIFY_STRATEGY, _FORCE_NATURAL, _NO_DIVERSITY_PROMPT, _FIXED_QUESTION_PROMPT, _FIXED_IMAGE_PROMPT
    subsets = sys.argv[1:] or list(SUBSET_CONFIG)
    limit = None
    start_idx = 0
    for arg in list(subsets):
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
            subsets.remove(arg)
        elif arg.startswith("--start-idx="):
            start_idx = int(arg.split("=", 1)[1])
            subsets.remove(arg)
        elif arg.startswith("--port="):
            _SERVER_URL = f"http://localhost:{arg.split('=', 1)[1]}"
            subsets.remove(arg)
        elif arg.startswith("--gpu="):
            _SERVER_GPU = arg.split("=", 1)[1]
            subsets.remove(arg)
        elif arg.startswith("--socket-prefix="):
            _SOCKET_PATH_PREFIX = arg.split("=", 1)[1]
            subsets.remove(arg)
        elif arg.startswith("--split="):
            _SPLIT = arg.split("=", 1)[1]
            assert _SPLIT in ("lite", "val50"), f"--split must be 'lite' or 'val50', got {_SPLIT!r}"
            subsets.remove(arg)
        elif arg.startswith("--out-root="):
            OUT_ROOT = Path(arg.split("=", 1)[1])
            subsets.remove(arg)
        elif arg == "--verify-all":
            _VERIFY_ALL = True
            subsets.remove(arg)
        elif arg == "--no-think":
            _THINK_MODE = False
            subsets.remove(arg)
        elif arg == "--verify-think":
            _VERIFY_THINK_MODE = True
            subsets.remove(arg)
        elif arg == "--discard-unverified":
            _DISCARD_UNVERIFIED = True
            subsets.remove(arg)
        elif arg == "--no-verify":
            _VERIFY_STRATEGY = "none"
            subsets.remove(arg)
        elif arg.startswith("--confidence-threshold="):
            _CONFIDENCE_THRESHOLD = int(arg.split("=", 1)[1])
            subsets.remove(arg)
        elif arg.startswith("--verify-criterion="):
            _VERIFY_CRITERION = arg.split("=", 1)[1]
            subsets.remove(arg)
        elif arg.startswith("--k="):
            K = int(arg.split("=", 1)[1])
            subsets.remove(arg)
        elif arg.startswith("--max-triplet-attempts="):
            _MAX_TRIPLET_ATTEMPTS = int(arg.split("=", 1)[1])
            subsets.remove(arg)
        elif arg.startswith("--strategy="):
            _STRATEGY = arg.split("=", 1)[1]
            assert _STRATEGY in ("bfs", "dfs"), f"--strategy must be 'bfs' or 'dfs', got {_STRATEGY!r}"
            subsets.remove(arg)
        elif arg == "--force-natural":
            _FORCE_NATURAL = True
            subsets.remove(arg)
        elif arg == "--no-diversity-prompt":
            _NO_DIVERSITY_PROMPT = True
            subsets.remove(arg)
        elif arg == "--fixed-question-prompt":
            _FIXED_QUESTION_PROMPT = True
            subsets.remove(arg)
        elif arg == "--fixed-image-prompt":
            _FIXED_IMAGE_PROMPT = True
            subsets.remove(arg)
    if _MAX_TRIPLET_ATTEMPTS is None:
        _MAX_TRIPLET_ATTEMPTS = K * 4
    print(f"Using server: {_SERVER_URL} (GPU {_SERVER_GPU})")
    print(
        f"K={K} think_mode={_THINK_MODE} verify_think_mode={_VERIFY_THINK_MODE} verify_all={_VERIFY_ALL} "
        f"discard_unverified={_DISCARD_UNVERIFIED} confidence_threshold={_CONFIDENCE_THRESHOLD} "
        f"verify_criterion={_VERIFY_CRITERION} max_triplet_attempts={_MAX_TRIPLET_ATTEMPTS} strategy={_STRATEGY} "
        f"verify_strategy={_VERIFY_STRATEGY} force_natural={_FORCE_NATURAL} no_diversity_prompt={_NO_DIVERSITY_PROMPT} "
        f"fixed_question_prompt={_FIXED_QUESTION_PROMPT} fixed_image_prompt={_FIXED_IMAGE_PROMPT}"
    )
    _ensure_server_running()
    # Multi-pass: a single pass leaves holes (server stalls, <K parsed triples,
    # failed gens) that previously required manual reruns. Repeat until every
    # sample is complete, giving up only when 3 consecutive passes make no
    # progress (retries are stochastic, so stagnation needs a few chances).
    prev_incomplete = None
    stagnant = 0
    for pass_num in range(1, 31):
        incomplete = 0
        for subset in subsets:
            incomplete += process_subset(subset, SUBSET_CONFIG[subset], limit=limit, start_idx=start_idx)
        print(f"\n[pass {pass_num}] incomplete samples: {incomplete}")
        if incomplete == 0:
            break
        if prev_incomplete is not None and incomplete >= prev_incomplete:
            stagnant += 1
            if stagnant >= 3:
                print(f"WARN: no progress for {stagnant} passes, giving up with {incomplete} incomplete")
                break
        else:
            stagnant = 0
        prev_incomplete = incomplete
    print("\nDONE")


if __name__ == "__main__":
    main()
