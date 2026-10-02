"""UMMClient backend for SenseNova-U1-8B-MoT via vLLM-omni's offline Omni API.

Confirmed empirically (scripts/smoke_test_omni*.py, verify_routing.py,
verify_scoring.py) against vllm-omni==0.24.0:
  - text2text / img2text / text2img all work through Omni.generate().
  - Multiple images in one request need an explicit "<image>\\n" placeholder
    per image, interleaved in the prompt text at the right position (the
    model's chat template convention) -- a bare prompt + multiple images
    with no placeholders crashes the worker (tensor shape mismatch) and
    that crash is FATAL to the whole Omni instance (no per-request recovery
    observed -- every later call in the same process also failed with
    "ShutDown" afterward). A single image needs no placeholder (auto-inserted).
  - sampling_params_list must be a single shared OmniDiffusionSamplingParams
    object when `prompts` is a list (a list of length == len(prompts) is
    rejected: "Expected 1 sampling params, got N").
  - The model emits <think>...</think> reasoning by default on every
    text/understanding call, with no per-call flag to suppress it (the
    "think" extra_arg is a no-op for text calls -- confirmed byte-identical
    output/timing with think_mode=True vs False). The only way to skip it is
    to pre-seed an already-closed empty <think></think> block into the
    prompt itself before generating (a pattern this model's own image-gen
    path already relies on for think_mode=False, so it's a trained-in
    behavior, not a hack) -- patched into vllm_omni's installed
    pipeline_sensenova_u1.py as the force_empty_think extra_arg (see
    environment/vllm_omni_patches/). This client sets it on every text call:
    every prompt this pipeline sends (classification, verification, triplet
    synthesis) was designed and tuned assuming a direct, non-reasoning
    answer, exactly BAGEL's behavior -- and letting it think instead measurably
    hurt correctness (classify_request scored 4/10 with truncated mid-think
    garbage vs 10/10 with force_empty_think) and reliability (verify()'s
    score criterion was silently getting truncated mid-think -- score=None --
    even at a generous 2048-token pad).
  - Teacher-forced/greedy token log-probs ARE available, but only via the
    same vllm_omni patch (score_teacher_forced/score_greedy below) -- the
    public Omni API never exposes per-token logits on its own.

NOT thread-safe for concurrent calls (one running Omni instance); callers
should serialize calls to one SensenovaClient the same way the BAGEL
pipeline serialized calls through one mstar server.
"""

import io
import re
from typing import List, Optional, Union

from PIL import Image

from .base import ChatResponse, ConfidenceRecord, UMMClient

MODEL_ID = "SenseNova/SenseNova-U1-8B-MoT"

# Matches synthesize_vqa.py's GEN_WIDTH/GEN_HEIGHT for the BAGEL pipeline --
# use the same generation resolution here rather than vLLM-omni's own much
# larger default (2048x2048).
DEFAULT_GEN_WIDTH = 400
DEFAULT_GEN_HEIGHT = 400

_THINK_TAG_RE = re.compile(r"</think>", re.IGNORECASE)

# With force_empty_think=True (see _text_sampling_params), the model never
# reasons before answering, so max_tokens no longer needs to absorb a
# variable (sometimes 1000+ token) <think> block -- just a small margin for
# answers slightly longer than the caller's nominal budget (e.g. multi-word
# category names, "PASS" vs a longer critique sentence).
_ANSWER_TOKEN_SAFETY_MARGIN = 64


def _strip_think(text: str) -> str:
    """Defensive no-op in the normal (force_empty_think=True) case -- kept in
    case some caller ever passes force_empty_think=False and the model emits
    a real <think> block again; take the text after the last </think>."""
    parts = _THINK_TAG_RE.split(text)
    return parts[-1].strip() if len(parts) > 1 else text.strip()


def _to_pil(image: Union[bytes, str, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.open(image).convert("RGB")


class SensenovaClient(UMMClient):
    def __init__(self, model: str = MODEL_ID, tensor_parallel_size: int = 1):
        from vllm_omni.entrypoints.omni import Omni

        self._omni = Omni(model=model, tensor_parallel_size=tensor_parallel_size, enforce_eager=False)

    def _text_sampling_params(self, max_output_tokens: int, temperature: float, think_mode: bool):
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        return OmniDiffusionSamplingParams(
            height=64, width=64, seed=0, num_inference_steps=1,
            extra_args={
                # force_empty_think=True: every prompt this pipeline sends
                # (classification, verification, triplet synthesis) was
                # designed and tuned assuming a direct, non-reasoning answer
                # -- exactly BAGEL's behavior. This model instead always
                # emits a <think> block first, with no per-call way to
                # suppress it except pre-seeding an already-closed empty
                # think block in the prompt (see pipeline_sensenova_u1.py's
                # patched _forward_text). This is strictly better than
                # padding max_tokens and hoping the (variable-length,
                # sometimes 1000+ token) reasoning finishes in time --
                # confirmed empirically: verify()'s score criterion was
                # silently getting truncated mid-think (score=None) even at
                # the old 2048-token pad. "think" extra_arg is left in for
                # documentation/back-compat but is a no-op on this pipeline's
                # text path (confirmed: byte-identical output/timing with
                # think_mode=True vs False, see verify_routing.py).
                "max_tokens": max_output_tokens + _ANSWER_TOKEN_SAFETY_MARGIN,
                "do_sample": temperature > 0,
                "temperature": temperature,
                "think": think_mode,
                "force_empty_think": True,
            },
        )

    def _run_text(self, prompt_dict: dict, max_output_tokens: int, temperature: float, think_mode: bool) -> str:
        sp = self._text_sampling_params(max_output_tokens, temperature, think_mode)
        outs = list(self._omni.generate(prompts=prompt_dict, sampling_params_list=sp))
        raw = outs[0].custom_output.get("text_output", "")
        return _strip_think(raw)

    def chat(
        self,
        prompt: str,
        images: Optional[Union[bytes, str]] = None,
        think_mode: bool = False,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ChatResponse:
        if images is not None:
            prompt_dict = {
                "prompt": prompt,
                "multi_modal_data": {"image": [_to_pil(images)]},
                "modalities": ["text"],
            }
        else:
            prompt_dict = {"prompt": prompt, "modalities": ["text"]}
        text = self._run_text(prompt_dict, max_output_tokens, temperature, think_mode)
        return ChatResponse(text=text)

    def chat_multi(
        self,
        parts: List[Union[str, Image.Image]],
        think_mode: bool = False,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        text_chunks = []
        images: List[Image.Image] = []
        for part in parts:
            if isinstance(part, Image.Image):
                images.append(part.convert("RGB"))
                text_chunks.append("<image>\n")
            else:
                text_chunks.append(part)
        prompt = "".join(text_chunks)
        prompt_dict = {"prompt": prompt, "modalities": ["text"]}
        if images:
            prompt_dict["multi_modal_data"] = {"image": images}
        temp = temperature if do_sample else 0.0
        return self._run_text(prompt_dict, max_output_tokens, temp, think_mode)

    def generate_image(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        think_mode: bool = False,
    ) -> bytes:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        sp = OmniDiffusionSamplingParams(
            height=height or DEFAULT_GEN_HEIGHT,
            width=width or DEFAULT_GEN_WIDTH,
            seed=None,
            num_inference_steps=50,
            extra_args={"cfg_scale": 4.0, "timestep_shift": 3.0, "batch_size": 1, "think": think_mode},
        )
        outs = list(self._omni.generate(prompts={"prompt": prompt, "modalities": ["image"]}, sampling_params_list=sp))
        images = outs[0].images or outs[0].request_output.images
        if not images:
            raise RuntimeError("generate_image: no image returned")
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        return buf.getvalue()

    def _score_prompt_dict(self, image, prompt_text: str) -> dict:
        if image is not None:
            return {
                "prompt": prompt_text,
                "multi_modal_data": {"image": [image.convert("RGB")]},
                "modalities": ["text"],
            }
        return {"prompt": prompt_text, "modalities": ["text"]}

    def _run_score(self, prompt_dict: dict, extra: dict) -> ConfidenceRecord:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        sp = OmniDiffusionSamplingParams(height=64, width=64, seed=0, num_inference_steps=1, extra_args=extra)
        outs = list(self._omni.generate(prompts=prompt_dict, sampling_params_list=sp))
        co = outs[0].custom_output
        return ConfidenceRecord(
            answer_text=co["answer_text"],
            answer_tokens_str=co["answer_tokens_str"],
            token_logprobs=co["token_logprobs"],
            token_probs=co["token_probs"],
            num_answer_tokens=co["num_answer_tokens"],
            avg_answer_logprob=co["avg_answer_logprob"],
            avg_answer_prob=co["avg_answer_prob"],
            min_token_logprob=co["min_token_logprob"],
            min_token_prob=co["min_token_prob"],
        )

    def score_teacher_forced(self, image, prompt_text: str, answer_text: str) -> ConfidenceRecord:
        prompt_dict = self._score_prompt_dict(image, prompt_text)
        return self._run_score(prompt_dict, {"score_answer": answer_text})

    def score_greedy(self, image, prompt_text: str, max_new_tokens: int) -> ConfidenceRecord:
        prompt_dict = self._score_prompt_dict(image, prompt_text)
        return self._run_score(prompt_dict, {"score_greedy": True, "max_tokens": max_new_tokens})
