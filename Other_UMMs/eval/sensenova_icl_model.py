"""SenseNova-U1 + in-context learning (SIMIT-ICL) lmms-eval model.

Port of lmms-eval/lmms_eval/models/simple/bagel_icl.py's BagelICL onto the
new backbone: same prompt-building convention ("Here are N examples...",
"Example i:", "Question: ...\\nAnswer: ..."), same DFASCSyntheticSampleRetriever
demo-selection logic (eval/icl_retriever.py), but calling
common/model_clients/sensenova_client.py's chat_multi() instead of BAGEL's
InterleaveInferencer, and with no real-data RAG retriever / prompt-dump
debugging / ablation options -- this project only needs the synthetic
(SIMIT-generated) retrieval strategy.

Usage (direct Python API, see run_icl_eval.py):
    from eval.sensenova_icl_model import SensenovaICL
    model = SensenovaICL(icl_samples_root="...", num_icl_examples=4)
    lmms_eval.evaluator.simple_evaluate(model=model, tasks=["ok_vqa_val2014_lite"], limit=5)
"""
import sys
from pathlib import Path
from typing import List, Tuple, Union

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lmms_eval.api.instance import Instance  # noqa: E402
from lmms_eval.api.model import lmms  # noqa: E402
from lmms_eval.api.registry import register_model  # noqa: E402
from loguru import logger as eval_logger  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common.model_clients.lance_client import LanceClient  # noqa: E402
from common.model_clients.sensenova_client import SensenovaClient  # noqa: E402
from eval.icl_retriever import DFASCSyntheticSampleRetriever, ICLExample  # noqa: E402


@register_model("sensenova_icl")
class SensenovaICL(lmms):
    #: Which UMMClient backend to construct in __init__ -- overridden by
    #: LanceICL below. A class attribute (not a constructor branch) so each
    #: subclass constructs only its own client, never both.
    _CLIENT_CLASS = SensenovaClient

    def __init__(
        self,
        icl_samples_root: str = "",
        num_icl_examples: int = 4,
        num_generations: int = 6,
        df_enabled: bool = False,
        asc_enabled: bool = False,
        df_asc_epsilon: float = 0.14,
        df_asc_A0: float = 0.35,
        df_asc_A1: float = 0.60,
        df_asc_B: float = 0.2,
        df_asc_T_low: float = 0.2,
        df_asc_T_high: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__()
        if not icl_samples_root:
            raise ValueError("sensenova_icl requires icl_samples_root=<path to synthesized samples>")
        self.client = self._CLIENT_CLASS()
        self.num_icl_examples = int(num_icl_examples)
        self.retriever = DFASCSyntheticSampleRetriever(
            icl_samples_root,
            num_generations=num_generations,
            df_enabled=df_enabled,
            asc_enabled=asc_enabled,
            epsilon=df_asc_epsilon,
            A0=df_asc_A0,
            A1=df_asc_A1,
            B=df_asc_B,
            T_low=df_asc_T_low,
            T_high=df_asc_T_high,
        )
        eval_logger.info(
            f"SensenovaICL initialized: num_icl_examples={self.num_icl_examples}, "
            f"icl_samples_root={icl_samples_root}, df_enabled={df_enabled}, asc_enabled={asc_enabled}"
        )

    def _build_input_list(self, contexts: str, visuals) -> List[Union[str, Image.Image]]:
        """Same simple case as BagelUMM._build_input_list (no <image> tags in
        contexts -- true for every task this project uses): images first,
        then the text context."""
        processed = []
        if visuals:
            for v in visuals:
                if isinstance(v, Image.Image):
                    processed.append(v.convert("RGB"))
                elif isinstance(v, str) and Path(v).exists():
                    processed.append(Image.open(v).convert("RGB"))
        return (processed + [contexts.strip()]) if processed else [contexts.strip()]

    def _build_icl_input_list(self, examples: List[ICLExample], query_input_list: List[Union[str, Image.Image]]):
        if not examples:
            return query_input_list
        n = len(examples)
        input_list: List[Union[str, Image.Image]] = [
            f"Here {'is' if n == 1 else 'are'} {n} example{'s' if n > 1 else ''} to help you answer the question:\n"
        ]
        for i, ex in enumerate(examples, 1):
            input_list.append(f"\nExample {i}:")
            input_list.append(Image.open(ex.image_path).convert("RGB"))
            input_list.append(f"Question: {ex.question}\nAnswer: {ex.answer}")
        input_list.append("\nNow, answer this question:")
        input_list.extend(query_input_list)
        return input_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (SIMIT-ICL)")
        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)
            if visuals is not None and not isinstance(visuals, list):
                visuals = [visuals]

            query_input_list = self._build_input_list(contexts, visuals)
            icl_examples = self.retriever.retrieve(task, doc_id, doc, self.num_icl_examples)
            input_list = self._build_icl_input_list(icl_examples, query_input_list)

            max_new_tokens = gen_kwargs.get("max_new_tokens", 64) if gen_kwargs else 64
            output_text = self.client.chat_multi(input_list, max_output_tokens=max_new_tokens, temperature=0.0)

            eval_logger.debug(f"[generate_until] {len(icl_examples)} example(s), doc_id={doc_id}: {output_text!r}")
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "sensenova_icl only supports generate_until-based tasks for now "
            "(no MCQ/loglikelihood-based benchmark support in this simple port)."
        )

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        return self.generate_until(requests)


@register_model("lance_icl")
class LanceICL(SensenovaICL):
    """Same SIMIT-ICL model as SensenovaICL, backed by LanceClient instead of
    SensenovaClient (see common/model_clients/lance_client.py's module
    docstring for the backend differences -- QA-style system prompt,
    sampling temperature floor, repetition penalty -- chat_multi() applies
    internally; no changes needed here). Everything else (prompt building,
    DF/ASC retrieval) is inherited unchanged."""

    _CLIENT_CLASS = LanceClient
