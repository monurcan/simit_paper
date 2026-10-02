"""Model-agnostic client interface the rest of Other_UMMs is written against.

Mirrors two roles the BAGEL pipeline splits across two different access
paths (mstar's MStarClient HTTP server for synthesize_vqa.py/diagram_gen,
and BAGEL's raw InterleaveInferencer for compute_confidence.py) as ONE
interface, so a single client object covers synthesis, verification,
skill-library rendering calls, ICL evaluation, and confidence scoring.

Adding a second backbone (e.g. MammothModa2) later means writing one more
class implementing this interface -- nothing above this layer (synthesis,
skills, eval, sweep) needs to change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union

from PIL import Image


@dataclass
class ChatResponse:
    text: str


@dataclass
class ConfidenceRecord:
    """Same fields as compute_confidence.py's build_confidence_record, so
    downstream difficulty-filtering/adaptive-budget code (which reads
    avg_answer_prob out of a JSON dict) needs no changes."""
    answer_text: str
    answer_tokens_str: List[dict]
    token_logprobs: List[float]
    token_probs: List[float]
    num_answer_tokens: int
    avg_answer_logprob: Optional[float]
    avg_answer_prob: Optional[float]
    min_token_logprob: Optional[float]
    min_token_prob: Optional[float]


class UMMClient(ABC):
    """A unified multimodal understanding+generation model backend."""

    @abstractmethod
    def chat(
        self,
        prompt: str,
        images: Optional[Union[bytes, str]] = None,
        think_mode: bool = False,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ChatResponse:
        """Single-image (or no-image) text chat. `images` is PNG bytes or a
        file path, matching MStarClient.chat's calling convention used
        throughout the synthesis pipeline and skill library."""
        ...

    @abstractmethod
    def chat_multi(
        self,
        parts: List[Union[str, Image.Image]],
        think_mode: bool = False,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """Interleaved multi-turn chat: `parts` is an ordered list of text
        strings and PIL images (BAGEL's InterleaveInferencer input_list
        convention), used for ICL evaluation where K demonstration images +
        their Q/A text precede the real query."""
        ...

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        think_mode: bool = False,
    ) -> bytes:
        """Native text-to-image generation (NATURAL category). Returns PNG bytes."""
        ...

    @abstractmethod
    def score_teacher_forced(
        self, image: Optional[Image.Image], prompt_text: str, answer_text: str
    ) -> ConfidenceRecord:
        """Single forward pass: teacher-force `answer_text` after the
        image+prompt_text context, score with shifted next-token log-probs.
        Used for synthesized samples, whose GT answer is known by
        construction (see compute_confidence.py::score_synthetic_teacher_forced)."""
        ...

    @abstractmethod
    def score_greedy(
        self, image: Optional[Image.Image], prompt_text: str, max_new_tokens: int
    ) -> ConfidenceRecord:
        """Greedy-decode an answer with no GT, scoring each generated token
        under the model's own predicted distribution. Used for the original
        (real) test sample's zero-shot confidence p0 (see
        compute_confidence.py::generate_and_score_real)."""
        ...
