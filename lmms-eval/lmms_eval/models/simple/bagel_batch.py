"""Fast-rollout variant of the bagel_umm lmms-eval model, for self-consistency
(repeats>1) evaluation specifically. Registers as a NEW model ("bagel_batch"),
separate from bagel_umm in bagel.py -- that file is not touched.

Reuses 100% of the official lmms-eval harness (Task loading, filters,
process_results, aggregation all stay exactly as they are for bagel_umm) --
the ONLY thing this changes is generate_until's generation loop: instead of
one sequential InterleaveInferencer call per repeat (K calls, each redoing the
image ViT encoding + prompt prefill from scratch), it batches all K repeats
of a doc into one shared-prefix, batched decode using the same rollout
machinery already verified in LFRL_Baselines/common/rollout.py (see that
module's docstring for the correctness verification: greedy-decode exact
match against the untouched original BAGEL API, confirmed batching-order
divergence only on long free-form completions is a known bf16 property, not
a masking bug).

Usage: --model bagel_batch --model_args pretrained=...,lora_ckpt_path=...
Same task/tags as bagel_umm (e.g. --tasks gqa_lite_sc), same model_args shape.
Only "understanding" mode is supported (no image generation, no "thinking")
-- that's all this project's self-consistency evals need.
"""

from __future__ import annotations
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[4])

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

_LFRL_BASELINES_ROOT = SIMIT_ROOT + "/LFRL_Baselines"
if _LFRL_BASELINES_ROOT not in sys.path:
    sys.path.insert(0, _LFRL_BASELINES_ROOT)

from common.rollout import generate_k_completions  # noqa: E402
from common.runtime import load_bagel_runtime  # noqa: E402


@register_model("bagel_batch")
class BagelUMMBatch(lmms):
    def __init__(
        self,
        pretrained: str = "",
        lora_ckpt_path: Optional[str] = None,
        device: str = "cuda:0",
        text_temperature: float = 0.7,
        max_think_token_n: int = 1000,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__()
        self._device = device
        self.text_temperature = float(text_temperature)
        self.max_new_tokens = int(max_think_token_n)

        self.runtime = load_bagel_runtime(
            model_path=pretrained,
            device=device,
            lora_rank=int(lora_rank) if lora_ckpt_path else 0,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_checkpoint_path=lora_ckpt_path,
        )
        eval_logger.info(f"BagelUMMBatch initialized (lora_ckpt_path={lora_ckpt_path})")

        self._rank = 0
        self._world_size = 1

    @property
    def eot_token_id(self):
        return self.runtime.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 4096

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]):
        raise NotImplementedError("loglikelihood is not implemented for bagel_batch")

    def generate_until_multi_round(self, requests: List[Instance]):
        raise NotImplementedError("multi-round generation is not implemented for bagel_batch")

    def _build_image(self, contexts: str, visuals: Optional[List[Any]]) -> Optional[Image.Image]:
        if not visuals:
            return None
        v = visuals[0]
        if isinstance(v, Image.Image):
            return v.convert("RGB")
        if isinstance(v, str) and os.path.exists(v):
            return Image.open(v).convert("RGB")
        return None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Groups consecutive requests sharing the same (task, doc_id) -- the
        harness emits a doc's `repeats` requests consecutively, same
        convention bagel_umm's own generate_until relies on via a
        rollout-index counter -- and generates that whole group in ONE
        batched call instead of `repeats` sequential ones. For repeats>1,
        replicates bagel_umm's own greedy-first + sampled-remainder
        convention exactly (first rollout greedy, so self-consistency can
        never do worse than plain greedy; remainder sampled at
        text_temperature) -- this is the one piece of generation-time
        behavior that must match bit-for-bit for the majority-vote filter
        downstream to see the same distribution bagel_umm would have produced."""
        res: List[Optional[str]] = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (batched)")

        i = 0
        n = len(requests)
        while i < n:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = requests[i].args
            group_idxs = [i]
            j = i + 1
            while j < n and requests[j].task_name == task and requests[j].doc_id == doc_id:
                group_idxs.append(j)
                j += 1
            k = len(group_idxs)

            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            image = self._build_image(contexts, visuals)

            if image is None:
                eval_logger.warning(f"[bagel_batch] no image for task={task} doc_id={doc_id}, emitting empty responses.")
                outputs = ["" for _ in range(k)]
            elif k > 1:
                # bagel_umm's own convention: rollout 0 greedy, rest sampled.
                greedy = generate_k_completions(
                    self.runtime, image=image, prompt=contexts, k=1,
                    max_new_tokens=self.max_new_tokens, temperature=0.0,
                )
                sampled = generate_k_completions(
                    self.runtime, image=image, prompt=contexts, k=k - 1,
                    max_new_tokens=self.max_new_tokens, temperature=self.text_temperature,
                )
                outputs = greedy + sampled
            else:
                outputs = generate_k_completions(
                    self.runtime, image=image, prompt=contexts, k=1,
                    max_new_tokens=self.max_new_tokens, temperature=0.0,
                )

            for idx, text in zip(group_idxs, outputs):
                res[idx] = text
            pbar.update(k)
            i = j

        pbar.close()
        return res
