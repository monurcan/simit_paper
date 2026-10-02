"""Models of the demo and how one request runs on ZeroGPU.

ZeroGPU forks the Space process for every ``@spaces.GPU`` call and kills the
fork afterwards. So the weights are loaded once, on CPU, in the main process
(no CUDA there), and each call moves only the chosen model to the GPU and
builds a fresh SIMIT engine around it.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from simit import SIMIT, SIMITConfig

HERE = Path(__file__).parent
PRESETS = json.loads((HERE / "presets.json").read_text())
BUDGETS = [20, 40, 60, 90, 180]           # seconds of GPU time per request
ANSWER_TOKENS = 128
ANSWER_RESERVE = 5.0                      # seconds kept for the final SIMIT-ICL answer


@dataclass
class ModelSpec:
    key: str
    label: str
    repo: str
    kind: str                               # "bagel" | "lance" | "hf"
    gpu_size: str                           # ZeroGPU size: "large" (48 GB) or "xlarge" (96 GB)
    image_generator: Optional[str] = None
    blurb: str = ""
    _weights: object = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ----------------------------------------------------- main process (CPU)
    def load(self):
        """Load the weights on CPU (main process; nothing touches CUDA)."""
        with self._lock:
            if self._weights is None:
                t = time.time()
                self._weights = _LOADERS[self.kind](self)
                print(f"[demo] loaded {self.label} on CPU in {time.time() - t:.0f}s", flush=True)
        return self._weights

    # ------------------------------------------------- inside @spaces.GPU call
    def build(self, preset: dict) -> SIMIT:
        """Move the weights to the GPU and wrap them in a SIMIT engine."""
        w = self.load()
        # Without ZeroGPU every request runs in this process: keep one model on the GPU at a time.
        # (On ZeroGPU each request is a fresh fork in which nothing is on the GPU yet.)
        if _ON_GPU[0] not in (None, self.key):
            MODELS[_ON_GPU[0]]._to("cpu")
            torch.cuda.empty_cache()
        _ON_GPU[0] = self.key
        cfg = SIMITConfig(**{k: v for k, v in preset.items() if k in SIMITConfig.__dataclass_fields__})
        if self.kind == "bagel":
            from simit.backends.bagel import BagelBackend
            backend = BagelBackend(w, device="cuda", image_steps=preset.get("image_steps"), use_cuda_graphs=False)
            return SIMIT(backend, config=cfg, model_id=self.repo)
        if self.kind == "lance":
            from simit.backends.lance import LanceBackend
            backend = LanceBackend(w, device="cuda", image_steps=preset.get("image_steps"), use_cuda_graphs=False)
            return SIMIT(backend, config=cfg, model_id=self.repo)
        from simit.backends.hf import HFBackend
        from simit.backends.imagegen import DiffusersImageGenerator
        model, processor, pipe = w
        model.to("cuda")
        pipe.to("cuda")
        backend = HFBackend(model, processor=processor, max_batch_size=8, max_batch_tokens=32768)
        return SIMIT(backend, image_generator=DiffusersImageGenerator(pipe), config=cfg, model_id=self.repo)

    def _to(self, device: str):
        if self.kind in ("bagel", "lance"):
            self._weights.to(device)
        else:
            model, _, pipe = self._weights
            model.to(device)
            pipe.to(device)

    def preset(self, budget: int) -> dict:
        return dict(PRESETS[self.key][str(budget)])


_ON_GPU = [None]   # key of the model whose weights are on the GPU in this process


@contextlib.contextmanager
def _fp8_without_gpu():
    """transformers dequantizes FP8 checkpoints to bf16 when it sees no GPU at load
    time. FP8 weights only need a GPU to *run*, so let the load keep them in FP8
    on CPU; they are moved to the GPU inside each request."""
    saved = torch.cuda.is_available, torch.cuda.get_device_capability
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
    try:
        yield
    finally:
        torch.cuda.is_available, torch.cuda.get_device_capability = saved


def _load_bagel(spec):
    from simit.backends.bagel import BagelModel
    return BagelModel.load(spec.repo, device="cpu")


def _load_lance(spec):
    from simit.backends.lance import LanceModel
    return LanceModel.load(spec.repo, device="cpu")


def _load_hf(spec):
    from diffusers import DiffusionPipeline
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from simit.backends.hf import _dense_fp8_fix
    os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")  # Triton FP8 kernels: no nvcc JIT at runtime
    processor = AutoProcessor.from_pretrained(spec.repo)
    kw = dict(device_map="cpu", dtype=torch.bfloat16)
    _dense_fp8_fix(spec.repo, kw)  # transformers would skip FP8 conversion of gate_proj for this checkpoint
    with _fp8_without_gpu():
        model = AutoModelForMultimodalLM.from_pretrained(spec.repo, **kw)
    model.eval()
    pipe = DiffusionPipeline.from_pretrained(spec.image_generator, dtype=torch.bfloat16)
    return model, processor, pipe


_LOADERS = {"bagel": _load_bagel, "lance": _load_lance, "hf": _load_hf}

MODELS = {m.key: m for m in [
    ModelSpec("bagel", "BAGEL-7B-MoT", "ByteDance-Seed/BAGEL-7B-MoT", "bagel", "large",
              blurb="Unified model: draws its own photos and writes code for charts and diagrams."),
    ModelSpec("lance", "Lance", "bytedance-research/Lance", "lance", "large",
              blurb="Small unified model: fast native image generation."),
    ModelSpec("qwen", "Qwen3.8-27B (FP8) + FLUX.2-klein", "Qwen/Qwen3.8-27B-FP8", "hf", "xlarge",
              image_generator="black-forest-labs/FLUX.2-klein-4B",
              blurb="Standard VLM: FLUX.2-klein draws the photos; it writes code for charts and diagrams."),
]}
ENABLED = [k for k in os.environ.get("SIMIT_DEMO_MODELS", "bagel,lance,qwen").split(",") if k in MODELS]


# ---------------------------------------------------------------- a request
# Pipeline call tags (``on_progress``) -> what the loading screen shows.
STAGE_OF_TAG = {"synthesis": "synthesizing", "synthesis_fresh": "synthesizing", "revise": "synthesizing",
                "route": "routing", "skill_spec": "coding", "render": "coding", "image": "drawing",
                "critic": "checking", "critic_score": "checking"}


def _refresh_triton_driver():
    """ZeroGPU forks this GPU worker from a main process whose torch.cuda queries are faked.
    Triton's driver copies torch.cuda.get_device_capability / current_device when it is first
    created, so a driver created in the main process (e.g. by fla's import-time device probe)
    keeps reporting the fake GPU, and kernels get compiled for the wrong architecture ("no kernel
    image is available for execution on the device"). Recreate it here, with the real GPU."""
    if not torch.cuda.is_available():
        return
    try:
        from triton.backends.nvidia.driver import CudaDriver
        from triton.runtime import driver
    except Exception:
        return
    driver.set_active(CudaDriver())


def run(spec: ModelSpec, image, question: str, budget: int, always: bool):
    """Generator of progress events for one request (runs inside the GPU call):
    ("stage", name, info) / ("greedy", ZeroShot, k) / ("demo", Demo) / ("final", answer, info)."""
    t_start = time.time()
    preset = spec.preset(budget)
    yield ("stage", "warmup", {"elapsed": 0.0})
    _refresh_triton_driver()
    sim = spec.build(preset)
    try:
        yield ("stage", "solving", {"elapsed": time.time() - t_start})
        zs = sim.zero_shot(image, question, max_new_tokens=ANSWER_TOKENS)   # the greedy answer + p0 (ABA input)
        k_budget = sim.config.budget(zs.confidence)
        yield ("greedy", zs, k_budget)

        # what is left of the budget, keeping room for the final answer (measured: ~1-1.5 s after the last demo)
        time_left = budget - (time.time() - t_start) - ANSWER_RESERVE
        if (k_budget == 0 and not always) or time_left < 3:
            if time_left < 3:
                reason = "no_time"
            else:
                reason = "confident" if preset.get("imagine_by_default", True) else "policy"
            yield ("final", zs.answer, {"demos": 0, "skipped": reason, "elapsed": time.time() - t_start})
            return

        events: "queue.Queue" = queue.Queue()
        current = ["synthesis"]
        result = {}

        def work():
            try:
                result["det"] = sim.imagine(image, question, k=preset["k_max"] if always else None,
                                            return_details=True, max_new_tokens=ANSWER_TOKENS, time_limit=time_left,
                                            on_demo=events.put, on_progress=lambda tag: current.__setitem__(0, tag))
            except Exception as e:  # surfaced below
                result["error"] = e
        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive() or not events.empty():
            try:
                yield ("demo", events.get(timeout=0.4))
                continue
            except queue.Empty:
                pass
            stage = STAGE_OF_TAG.get(current[0].split(":")[0], "synthesizing")
            yield ("stage", stage, {"elapsed": time.time() - t_start, "limit": budget})
        if "error" in result:
            raise result["error"]
        det = result["det"]
        if not det.demos:
            yield ("final", zs.answer, {"demos": 0, "skipped": "none_passed", "elapsed": time.time() - t_start,
                                        "stats": det.stats})
            return
        yield ("stage", "improving", {"elapsed": time.time() - t_start})
        answer = sim.answer(image, question, det.demos, max_new_tokens=ANSWER_TOKENS)
        yield ("final", answer, {"demos": len(det.demos), "elapsed": time.time() - t_start, "stats": det.stats})
    finally:
        sim.close()


def code_snippet(spec: ModelSpec, budget: int, question: str, always: bool = False) -> str:
    """Copy-paste code reproducing this request with the library."""
    p = spec.preset(budget)
    force = always or not p.get("imagine_by_default", True)
    keys = ["k_max", "epsilon", "A0", "A1", "B", "t_low", "t_high", "verify", "verify_think", "use_skills",
            "attempts_per_slot", "verify_rounds"]
    args = [f"{k}={round(p[k], 3) if isinstance(p[k], float) else p[k]!r}" for k in keys if k in p]
    lines, line = [], ""
    for a in args:  # wrap the config arguments, a few per line
        if line and len(line) + len(a) > 62:
            lines.append(line.rstrip())
            line = ""
        line += a + ", "
    lines.append(line.rstrip(", "))
    cfg = "\n        " + "\n        ".join(lines) + ",\n    "
    gen = f',\n    image_generator="{spec.image_generator}",' if spec.image_generator else ","
    q = question.strip().replace('"""', "'''") or "What is shown in this image?"
    steps = f"\n    image_steps={p['image_steps']}," if p.get("image_steps") else ""
    return f'''# pip install simit
from simit import SIMIT, SIMITConfig

model = SIMIT.from_pretrained(
    "{spec.repo}"{gen}
    config=SIMITConfig({cfg}),{steps}
)

image = "my_image.jpg"
question = """{q}"""

greedy = model.greedy(image, question)                          # base model, zero-shot
demos = model.imagine(image, question{f", k={p['k_max']}" if force else ""}, time_limit={max(5, budget - 15)})   # imagined examples
improved = model.answer(image, question, demos)                 # SIMIT-ICL
print(greedy, "->", improved)

from simit import save_demos
save_demos(demos, "imagined/")                                  # PNGs + jsonl, e.g. to fine-tune on
'''
