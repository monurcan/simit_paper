"""UMMClient backend for Lance-3B (ByteDance) via vLLM-omni's offline Omni API.

Lance is BAGEL-lineage (LancePipeline subclasses vllm-omni's BagelPipeline),
loaded the same way as common/model_clients/sensenova_client.py's
SensenovaClient (in-process via Omni(), no separate server), but with a
different engine-construction signature and prompt/chat-template convention,
confirmed against the model's own reference example
(vllm-omni examples/offline_inference/lance/end2end.py, vllm-omni==0.24.1):

  - Omni() needs Lance's own single-stage engine kwargs (pipeline="lance",
    enforce_eager=True, trust_remote_code=True, max_num_seqs=1, ...) --
    SensenovaClient's plain Omni(model=..., tensor_parallel_size=...) is not
    enough (the Lance pipeline is only selected via pipeline="lance"; its
    HF config.json carries no model_type for auto-detection).
  - Prompts must use Lance's own Qwen-chat-style template
    (<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...) with a
    per-task system prompt; image content is marked inline with
    <|vision_start|><|video_pad|><|vision_end|>, immediately before the
    question text -- unlike SenseNova's own <image> placeholder convention.
    Confirmed empirically load-bearing even with NO image at all: routing
    plain text-only calls (classification, DSL/code extraction -- see
    common/skills/pipeline.py) through BagelPipeline's inherited plain-
    text2text path (no system prompt at all) measurably hurt instruction-
    following, so every chat()/chat_multi() call -- 0, 1, or N images alike
    -- is routed through the SAME system-prompt-wrapped path (_run_x2t,
    backed by the patched pipeline's _forward_x2t_image_multi), with
    _X2T_SYSTEM_PROMPT (QA-style) when an image is present and
    _TEXT_SYSTEM_PROMPT (generic) otherwise. A missing/wrong system prompt
    on an image+question (x2t_image) call makes the model caption the image
    instead of answering the question (confirmed in vllm-omni's own example
    script docstring).
  - Lance's text generation frequently degenerates under greedy decoding --
    either an immediate EOS (image-conditioned calls; documented upstream
    default do_sample=True, text_temperature=0.8, "greedy frequently emits
    an immediate EOS for Lance") or repetitive token loops (confirmed
    empirically here on text-only calls too, e.g. plain classification
    prompts) -- unlike SenseNova (fine at temperature=0.0 across the board).
    This client (a) floors the sampling temperature at
    _X2T_MIN_TEMPERATURE and forces do_sample=True regardless of the
    caller-requested value, and (b) applies a repetition penalty
    (_DEFAULT_REPETITION_PENALTY, matching Lance's own generation_config.json
    recommendation -- upstream vllm-omni's decode loop doesn't apply one at
    all) for every chat()/chat_multi() call.
  - Even with the above, Lance's complex/structured-output instructions
    (this project's original one-shot "generate K diverse triples in a
    specific multi-field format" synthesis prompt, DSL extraction,
    17-way classification) are NOT reliable -- Lance is a much smaller,
    narrowly-specialized model whose only reliably-trained text task is
    single-turn direct question->short-answer (see synthesis/
    synthesize_vqa.py's generate_synthetic_examples_lance and its
    docstring for the decomposed, few-shot workaround this project uses
    instead).
  - Multi-image interleaved chat (SIMIT-ICL's K demonstration images + 1
    query image, common/model_clients/base.py's chat_multi) and teacher-
    forced/greedy per-token log-prob scoring (score_teacher_forced/
    score_greedy) are NOT covered by vllm-omni's public Lance example or its
    own Omni.generate() API (same limitation SensenovaClient has -- see its
    docstring); both need a vllm-omni patch, same idea as
    pipeline_sensenova_u1.py.patched's score_teacher_forced/score_greedy
    support -- see environment/vllm_omni_patches/pipeline_lance.py.patched
    and its README entry.

NOT thread-safe for concurrent calls (one running Omni instance), same as
SensenovaClient.
"""

import io
import math
import os
from typing import List, Optional, Union

from PIL import Image

from .base import ChatResponse, ConfidenceRecord, UMMClient

MODEL_ID = "bytedance-research/Lance"

# Matches synthesize_vqa.py's GEN_WIDTH/GEN_HEIGHT for the BAGEL pipeline --
# same fallback default as SensenovaClient, for cross-backend size parity
# when a caller doesn't pass explicit width/height. synthesize_vqa.py always
# does, so this only matters for other/future callers.
DEFAULT_GEN_WIDTH = 400
DEFAULT_GEN_HEIGHT = 400

# Lance's own documented t2i defaults (vllm-omni examples/offline_inference/
# lance/end2end.py): 30 denoising steps, cfg_text_scale=4.0, timestep_shift=3.5.
GEN_STEPS = 30
GEN_CFG_TEXT_SCALE = 4.0
GEN_TIMESTEP_SHIFT = 3.5

# x2t_image system prompt: without an explicit QA-style override, Lance's
# own default system prompt for this task is caption-style ("Generate a
# detailed and accurate description of the image...") and the model
# describes the image instead of answering the question (see vllm-omni's
# end2end.py --system-prompt help text). Every prompt this pipeline sends
# through chat()/chat_multi() with an image is a question expecting a direct
# answer (classification/VQA/ICL), never a plain captioning request.
_X2T_SYSTEM_PROMPT = "Look at the image carefully and answer the question."

# System prompt for chat()/chat_multi() calls with no image (classification,
# DSL/code extraction -- see common/skills/pipeline.py -- and the text-only
# fallback for chat_multi's plain-text case). Confirmed empirically load-
# bearing here too, same as _X2T_SYSTEM_PROMPT above: BagelPipeline's
# inherited plain-text2text path (no system prompt at all) measurably hurt
# instruction-following even with zero images involved, so this project's
# own text-only calls are routed through the same system-prompt-wrapped
# _run_x2t path as image calls (with an empty image list) rather than that
# inherited path -- see module docstring / environment/vllm_omni_patches/
# README.md.
_TEXT_SYSTEM_PROMPT = "You are a helpful assistant."

# See module docstring: Lance's x2t_image generation is unreliable (immediate
# EOS) under greedy/low-temperature decoding. Confirmed empirically for this
# project too -- see environment/vllm_omni_patches/README.md.
_X2T_MIN_TEMPERATURE = 0.8

# Lance's own generation_config.json ships repetition_penalty=1.05 as its
# recommended default; upstream vllm-omni's generate_text loop doesn't apply
# it at all (see the patched pipeline's _generate_text_rp). See module
# docstring / environment/vllm_omni_patches/README.md for why this matters
# here and how this value was picked empirically.
_DEFAULT_REPETITION_PENALTY = 1.05

# extra_args key LanceClient's own multi-image prompts split on (matches
# LancePipeline.ICL_VISION_MARKER in the patched pipeline).
_VISION_MARKER = "<|vision_start|><|video_pad|><|vision_end|>"


def _to_pil(image: Union[bytes, str, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.open(image).convert("RGB")


class LanceClient(UMMClient):
    def __init__(self, model: str = MODEL_ID, tensor_parallel_size: int = 1):
        # Lance aligns more closely with upstream reference outputs using
        # FLASH_ATTN (SDPA accumulates measurably more numerical drift
        # through the 36-layer Qwen2 stack per vllm-omni's own example
        # script comment) -- set it before Omni() builds the model, but
        # don't clobber a value the caller/environment already set.
        os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN")

        from vllm_omni.entrypoints.omni import Omni

        # Lance-specific single-stage engine kwargs (not SensenovaClient's
        # plain Omni(model=..., tensor_parallel_size=...)) -- confirmed
        # required by vllm-omni's own end2end.py example (formerly
        # vllm_omni/deploy/lance.yaml, now passed as flat Omni() kwargs).
        self._omni = Omni(
            model=model,
            pipeline="lance",
            max_num_batched_tokens=32768,
            max_num_seqs=1,
            enforce_eager=True,
            trust_remote_code=True,
            enable_prefix_caching=False,
            async_chunk=False,
            tensor_parallel_size=tensor_parallel_size,
        )

    # -- prompt building ---------------------------------------------------

    def _render_multi(self, parts: List[Union[str, Image.Image]], system_prompt: str):
        """parts is BagelUMM/SensenovaICL's own ordered text/PIL.Image
        convention (see base.py's chat_multi docstring); turns it into a
        single Lance chat-template string with one _VISION_MARKER per image
        (immediately before that image's own text, matching Lance's
        vision-block-before-text convention), plus the image list in the
        same order. No trailing "\\n" after "assistant" -- see
        LancePipeline._split_x2t_prompt's docstring (patched pipeline
        expects to inject that token itself)."""
        text_chunks = []
        images: List[Image.Image] = []
        for part in parts:
            if isinstance(part, Image.Image):
                images.append(part.convert("RGB"))
                text_chunks.append(_VISION_MARKER)
            else:
                text_chunks.append(part)
        user_msg = "".join(text_chunks)
        rendered = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant"
        )
        return rendered, images

    # -- generation ----------------------------------------------------------

    def _run_x2t(self, rendered: str, images: List[Image.Image], max_output_tokens: int,
                 temperature: float, force_sample: bool = True,
                 repetition_penalty: float = _DEFAULT_REPETITION_PENALTY) -> str:
        """Single text-generation entrypoint for chat()/chat_multi(), used for
        0, 1, or N images alike -- all routed through the patched pipeline's
        _forward_x2t_image_multi/_x2t_multi_setup (environment/
        vllm_omni_patches/pipeline_lance.py.patched), NOT BagelPipeline's
        inherited plain-text2text path. That inherited path has no system
        prompt at all, which measurably hurts instruction-following on this
        project's non-trivial prompts (DSL extraction, classification,
        triplet synthesis) even with no image involved -- see module
        docstring / environment/vllm_omni_patches/README.md."""
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        do_sample = force_sample or temperature > 0
        text_temperature = max(temperature, _X2T_MIN_TEMPERATURE) if do_sample else temperature
        sp = OmniDiffusionSamplingParams(
            height=64, width=64, seed=None, num_inference_steps=1,
            extra_args={
                "max_think_tokens": max_output_tokens,
                "do_sample": do_sample,
                "text_temperature": text_temperature,
                "repetition_penalty": repetition_penalty,
            },
        )
        prompt_dict = {"prompt": rendered, "multi_modal_data": {"image": images}, "modalities": ["text"]}
        outs = list(self._omni.generate(prompts=prompt_dict, sampling_params_list=sp))
        return outs[0].custom_output.get("text_output", "").strip()

    def chat(
        self,
        prompt: str,
        images: Optional[Union[bytes, str]] = None,
        think_mode: bool = False,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ChatResponse:
        # think_mode: accepted for UMMClient interface parity (SensenovaClient
        # takes it too) but a no-op here -- unlike SenseNova, Lance's
        # tokenizer has no <think> token and its checkpoint was never trained
        # with a think-then-answer split (confirmed: no think-related special
        # tokens in its tokenizer.json/generation_config.json), so there is
        # no equivalent mode to force on or off.
        parts = [_to_pil(images), prompt] if images is not None else [prompt]
        rendered, imgs = self._render_multi(parts, _X2T_SYSTEM_PROMPT if images is not None else _TEXT_SYSTEM_PROMPT)
        text = self._run_x2t(rendered, imgs, max_output_tokens, temperature)
        return ChatResponse(text=text)

    def chat_multi(
        self,
        parts: List[Union[str, Image.Image]],
        think_mode: bool = False,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        has_image = any(isinstance(p, Image.Image) for p in parts)
        system_prompt = _X2T_SYSTEM_PROMPT if has_image else _TEXT_SYSTEM_PROMPT
        rendered, imgs = self._render_multi(parts, system_prompt)
        # See module docstring: Lance's decoding needs do_sample=True
        # regardless of the caller's own do_sample/temperature request
        # (SIMIT-ICL's eval driver calls this with temperature=0.0/do_sample
        # defaulted False expecting greedy, which empirically breaks Lance's
        # x2t path -- see environment/vllm_omni_patches/README.md).
        return self._run_x2t(rendered, imgs, max_output_tokens, temperature)

    def generate_image(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        think_mode: bool = False,
    ) -> bytes:
        from vllm_omni.diffusion.models.lance.prompts import render_lance_prompt
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        sp = OmniDiffusionSamplingParams(
            height=height or DEFAULT_GEN_HEIGHT, width=width or DEFAULT_GEN_WIDTH,
            seed=None, num_inference_steps=GEN_STEPS,
            extra_args={"cfg_text_scale": GEN_CFG_TEXT_SCALE, "timestep_shift": GEN_TIMESTEP_SHIFT},
        )
        prompt_dict = {"prompt": render_lance_prompt("t2i", prompt), "modalities": ["image"]}
        outs = list(self._omni.generate(prompts=prompt_dict, sampling_params_list=sp))
        images = outs[0].images or outs[0].request_output.images
        if not images:
            raise RuntimeError("generate_image: no image returned")
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        return buf.getvalue()

    # -- confidence scoring --------------------------------------------------

    def _run_score(self, rendered: str, images: List[Image.Image], extra: dict) -> ConfidenceRecord:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        sp = OmniDiffusionSamplingParams(height=64, width=64, seed=0, num_inference_steps=1, extra_args=extra)
        prompt_dict = {"prompt": rendered, "modalities": ["text"]}
        if images:
            prompt_dict["multi_modal_data"] = {"image": images}
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
        parts = [_to_pil(image), prompt_text] if image is not None else [prompt_text]
        rendered, imgs = self._render_multi(parts, _X2T_SYSTEM_PROMPT)
        return self._run_score(rendered, imgs, {"score_answer": answer_text})

    def score_greedy(self, image, prompt_text: str, max_new_tokens: int) -> ConfidenceRecord:
        parts = [_to_pil(image), prompt_text] if image is not None else [prompt_text]
        rendered, imgs = self._render_multi(parts, _X2T_SYSTEM_PROMPT)
        return self._run_score(rendered, imgs, {"score_greedy": True, "max_tokens": max_new_tokens})
