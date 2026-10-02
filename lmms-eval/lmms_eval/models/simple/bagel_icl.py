"""Bagel + in-context learning (ICL) demonstrations, for the "synthetic RAG"
evaluation: instead of retrieving real neighbors from an index, the K
demonstrations for a doc are the (image, question, answer) triplets that
Bagel itself synthesized as "similar" to that doc (see
similar_samples_llms_eval_lite/synthesize_vqa.py). The example-selection
step is behind an ICLRetriever interface so a real FAISS-backed retriever can
be dropped in later without touching the prompt-building or generation code.
"""
import json
import os
import re
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.bagel import BagelUMM, BAGEL_AVAILABLE

if BAGEL_AVAILABLE:
    from bagel.data.data_utils import pil_img2rgb


@dataclass
class ICLExample:
    image_path: Optional[str] = None
    question: str = ""
    answer: str = ""
    # Already-loaded (and, where relevant, already-processed -- e.g. refcoco's
    # bbox drawn on) demo image(s), used instead of re-opening image_path.
    # Only set by RealSampleRetriever, which gets these directly from the
    # task's own doc_to_visual(); left None (default) for SyntheticSampleRetriever,
    # which is unaffected by this field and keeps opening image_path exactly as before.
    images: Optional[List[Image.Image]] = None


class ICLRetriever(ABC):
    """Selects the in-context demonstrations for a given doc."""

    @abstractmethod
    def retrieve(self, task: str, doc_id: Any, doc: Dict[str, Any], k: int) -> List[ICLExample]:
        ...


class SyntheticSampleRetriever(ICLRetriever):
    """Stand-in for real retrieval: returns the BAGEL-synthesized examples
    generated for this exact doc (folder layout produced by
    similar_samples_llms_eval_lite/synthesize_vqa.py):

        <root>/<benchmark>/sample<doc_id>/synthesized_sample_{0..3}.png
                                            synthesized_sample_{0..3}_prompt.txt
                                            synthesized_sample_{0..3}_answers.txt

    <benchmark> is the lmms-eval task name with the "_lite" suffix (and, for
    mathvista, the task-variant suffix) stripped, e.g. "ai2d_lite" -> "ai2d",
    "docvqa_val_lite" -> "docvqa_val", "mathvista_testmini_format" -> "mathvista".
    doc_id is used directly as the sample index (confirmed to match: the
    samples were synthesized by indexing the same lmms-lab/LMMs-Eval-Lite
    "lite" split that these lmms-eval tasks load, in the same row order).
    """

    def __init__(self, root: str):
        self.root = Path(root)

    def _benchmark_dir_for_task(self, task: str) -> Optional[Path]:
        name = task
        if name.startswith("mathvista"):
            name = "mathvista"
        elif name.endswith("_lite_val50"):
            # 50-sample validation split: samples synthesized with
            # synthesize_vqa.py --split=val50 live in <root>/<benchmark>_val50/.
            name = name[: -len("_lite_val50")] + "_val50"
        elif name.endswith("_lite"):
            name = name[: -len("_lite")]
        bench_dir = self.root / name
        return bench_dir if bench_dir.is_dir() else None

    @staticmethod
    def _clean_question(text: str) -> str:
        """Same parsing as Bagel/prepare_vqa_data.py: a few synthesized prompt
        files contain leftover batch-generation text before the real question;
        take the last "Question: ..." line when present."""
        text = text.strip()
        if "\n" not in text:
            return text
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        q_lines = [l for l in lines if l.lower().startswith("question:")]
        if q_lines:
            return re.sub(r"(?i)^question:\s*", "", q_lines[-1]).strip()
        return " ".join(lines)

    def retrieve(self, task: str, doc_id: Any, doc: Dict[str, Any], k: int) -> List[ICLExample]:
        bench_dir = self._benchmark_dir_for_task(task)
        if bench_dir is None:
            eval_logger.warning(f"[SyntheticSampleRetriever] No sample folder for task '{task}' under {self.root}")
            return []

        sample_dir = bench_dir / f"sample{doc_id}"
        if not sample_dir.is_dir():
            eval_logger.warning(f"[SyntheticSampleRetriever] No sample dir {sample_dir} for task '{task}' doc_id {doc_id}")
            return []

        examples = []
        for i in range(4):
            if len(examples) >= k:
                break
            img_f = sample_dir / f"synthesized_sample_{i}.png"
            prompt_f = sample_dir / f"synthesized_sample_{i}_prompt.txt"
            ans_f = sample_dir / f"synthesized_sample_{i}_answers.txt"
            if not (img_f.exists() and prompt_f.exists() and ans_f.exists()):
                continue
            question = self._clean_question(prompt_f.read_text())
            answer = ans_f.read_text().strip()
            if not question or not answer:
                continue
            examples.append(ICLExample(image_path=str(img_f), question=question, answer=answer))
        return examples


def _compute_tau(epsilon: float, A0: float, A1: float, B: float, num_generations: int) -> Dict[int, float]:
    """Identical formula to Bagel/prepare_vqa_data_v2_dfadaptive.py::compute_tau,
    duplicated here (not imported) since this process's venv is pinned separately
    from Bagel's training-side scripts. tau[K] is the zero-shot-confidence
    threshold above which a real sample is judged to need at most K demos."""
    import math
    O_0 = A0 / (1 - A0)
    tau = {0: 1 - epsilon}
    for K in range(1, num_generations + 1):
        A_K = min(max(A1 + B * math.log(K), 1e-3), 1 - 1e-3)
        O_K = A_K / (1 - A_K)
        tau[K] = (1 - epsilon) * O_0 / (epsilon * O_K + (1 - epsilon) * O_0)
    return tau


def _adaptive_k(p0: float, tau: Dict[int, float], num_generations: int) -> int:
    for K in range(0, num_generations + 1):
        if p0 >= tau[K]:
            return K
    return num_generations


def _load_avg_answer_prob(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        return json.load(open(path)).get("avg_answer_prob")
    except Exception:
        return None


class DFASCSyntheticSampleRetriever(SyntheticSampleRetriever):
    """SyntheticSampleRetriever, generalized to an arbitrary number of
    generated variants per sample (generated_samples_discard_unverified_v2 has
    6, not the older pool's 4) and extended with the same Difficulty Filtering
    (DF) + Adaptive Sample Count (ASC) demo-selection mechanisms used for LoRA
    finetuning data selection (Bagel/prepare_vqa_data_v2_dfadaptive.py) --
    applied here to *which ICL demos get shown*, not which training records
    get used. Both mechanisms default to disabled, reproducing the parent
    class's plain "first k available" behavior exactly (just with
    num_generations=6 instead of the hardcoded 4).

    DF: keep only variants whose own teacher-forced confidence c_i falls in
    [T_low, T_high]. ASC: cap how many kept variants are shown, based on how
    confident the model already is on the real sample zero-shot (p0) vs a
    tau[K] curve shaped by epsilon/A0/A1/B -- the same formula as finetuning,
    just deciding a demo count instead of a training-record count.
    """

    def __init__(
        self,
        root: str,
        num_generations: int = 6,
        df_enabled: bool = False,
        asc_enabled: bool = False,
        epsilon: float = 0.14,
        A0: float = 0.35,
        A1: float = 0.60,
        B: float = 0.2,
        T_low: float = 0.2,
        T_high: float = 0.9,
    ):
        super().__init__(root)
        self.num_generations = int(num_generations)
        self.df_enabled = bool(df_enabled)
        self.asc_enabled = bool(asc_enabled)
        self.epsilon, self.A0, self.A1, self.B = float(epsilon), float(A0), float(A1), float(B)
        self.T_low, self.T_high = float(T_low), float(T_high)
        self._tau = _compute_tau(self.epsilon, self.A0, self.A1, self.B, self.num_generations) if self.asc_enabled else None
        eval_logger.info(
            f"DFASCSyntheticSampleRetriever: num_generations={self.num_generations}, df_enabled={self.df_enabled}, "
            f"asc_enabled={self.asc_enabled}, epsilon={self.epsilon}, A0={self.A0}, A1={self.A1}, B={self.B}, "
            f"T_low={self.T_low}, T_high={self.T_high}"
        )

    def retrieve(self, task: str, doc_id: Any, doc: Dict[str, Any], k: int) -> List[ICLExample]:
        bench_dir = self._benchmark_dir_for_task(task)
        if bench_dir is None:
            eval_logger.warning(f"[DFASCSyntheticSampleRetriever] No sample folder for task '{task}' under {self.root}")
            return []
        sample_dir = bench_dir / f"sample{doc_id}"
        if not sample_dir.is_dir():
            eval_logger.warning(f"[DFASCSyntheticSampleRetriever] No sample dir {sample_dir} for task '{task}' doc_id {doc_id}")
            return []

        # slot order (index, question, answer, image_path, c_i)
        candidates = []
        for i in range(self.num_generations):
            img_f = sample_dir / f"synthesized_sample_{i}.png"
            prompt_f = sample_dir / f"synthesized_sample_{i}_prompt.txt"
            ans_f = sample_dir / f"synthesized_sample_{i}_answers.txt"
            if not (img_f.exists() and prompt_f.exists() and ans_f.exists()):
                continue
            question = self._clean_question(prompt_f.read_text())
            answer = ans_f.read_text().strip()
            if not question or not answer:
                continue
            c_i = _load_avg_answer_prob(sample_dir / f"synthesized_sample_{i}_confidence.json") if self.df_enabled else None
            candidates.append((i, question, answer, img_f, c_i))

        if self.df_enabled:
            kept = [c for c in candidates if c[4] is not None and self.T_low <= c[4] <= self.T_high]
        else:
            kept = candidates

        if self.asc_enabled:
            p0 = _load_avg_answer_prob(sample_dir / "original_sample_confidence.json")
            k_adaptive = _adaptive_k(p0, self._tau, self.num_generations) if p0 is not None else self.num_generations
        else:
            k_adaptive = self.num_generations

        used = kept[:k_adaptive][:k]
        return [ICLExample(image_path=str(img_f), question=q, answer=a) for (_, q, a, img_f, _) in used]


# Fields duplicated (not imported) from rag_retrieval/build_faiss_index.py and
# rag_retrieval/build_retrieval_sets.py: rag_retrieval/ lives in a separate
# venv (transformers>=4.57, needed for Qwen3-VL-Embedding) that this process
# -- pinned to transformers==4.49.0 for Bagel's own vendored modeling code --
# cannot import. Both lists are short and stable enough that duplicating them
# here (rather than adding a cross-venv dependency) is the simpler tradeoff.
_RAG_ANSWER_FIELDS = ["answer", "answers", "annotations_captions", "caption", "reference_strs", "caption_str", "multiple_choice_answer"]
_RAG_FIXED_CAPTION_TEXT = {
    "nocaps_val_lite": "Describe the image in one sentence.",
    "flickr30k_test_lite": "Describe the image in one sentence.",
}


def _rag_embed_text(task: str, doc: Dict[str, Any]) -> str:
    """Must match rag_retrieval/build_faiss_index.py::embed_text_for exactly --
    the online query embedding and the offline retrieval-pool embeddings have
    to use the identical text recipe for retrieval to make sense."""
    q = doc.get("question")
    if isinstance(q, str) and q.strip():
        return q
    return _RAG_FIXED_CAPTION_TEXT.get(task, "Describe the image.")


def _rag_fallback_answer(doc: Dict[str, Any]) -> str:
    for f in _RAG_ANSWER_FIELDS:
        v = doc.get(f)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and v:
            return str(v[0])
    return ""


def _rag_as_image_list(img) -> List[Image.Image]:
    if img is None:
        return []
    return list(img) if isinstance(img, list) else [img]


class RealSampleRetriever(ICLRetriever):
    """Retrieves real neighbor samples from a per-benchmark FAISS index built
    by rag_retrieval/build_faiss_index.py (Qwen3-VL-Embedding), instead of
    SyntheticSampleRetriever's Bagel-synthesized examples.

    Demo text/images are built by reusing each task's *own*
    doc_to_visual/doc_to_text/doc_to_target -- the same functions the real
    query prompt for that task is built from -- so a RAG demo is formatted
    exactly like a real query would be for that benchmark (e.g. full MCQ
    options + lettered answer for ai2d/mmbench/seedbench), not the
    free-text-only convention SyntheticSampleRetriever's synthesized samples
    happen to use. This is deliberately a different convention between the
    two retrievers; the prompt *wrapper* built by _build_icl_input_list
    ("Example N:", "Question: ...\\nAnswer: ...", etc.) is identical either way.

    Retrieval matching itself (the FAISS search) never sees any ground-truth
    answer, query or neighbor: both sides are embedded from (image, question)
    only (see _rag_embed_text / rag_retrieval/build_faiss_index.py). A
    neighbor's answer is only used afterwards, to render that neighbor's own
    (already-known, real) label as the shown demo answer.

    The embedding model needs transformers>=4.57 (this venv is pinned to
    4.49.0 for Bagel's own vendored modeling code), so the actual embedding +
    FAISS search runs in a subprocess under a separate venv
    (rag_retrieval/.venv), talked to over stdin/stdout JSON lines -- see
    rag_retrieval/embed_worker.py for the other end of this protocol.
    """

    def __init__(
        self,
        index_root: str,
        embed_python: str,
        embed_worker_script: Optional[str] = None,
        embed_gpu: Optional[str] = None,
    ):
        self.index_root = index_root
        self.embed_python = embed_python
        # embed_python is expected to be <rag_retrieval>/.venv/bin/python (a
        # symlink to a system interpreter) -- don't .resolve() it, that would
        # follow the symlink away from rag_retrieval/ entirely.
        self.embed_worker_script = embed_worker_script or str(Path(embed_python).parent.parent.parent / "embed_worker.py")
        self.embed_gpu = embed_gpu
        self._proc: Optional[subprocess.Popen] = None
        self._task_manager = None
        self._task_cache: Dict[str, Any] = {}

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        env = os.environ.copy()
        if self.embed_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.embed_gpu)
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [self.embed_python, "-u", self.embed_worker_script, "--data-root", self.index_root]

        # When Bagel and the embed worker share one physical GPU (single-GPU
        # jobs, embed_gpu pointing at the same device Bagel already loaded
        # onto), the worker's own CUDA init can transiently race Bagel's
        # concurrent GPU activity: "CUDA-capable device(s) is/are busy or
        # unavailable" during embed_worker's model.to(device) call, which
        # kills it before it prints its ready line. Confirmed in production:
        # 2 of the first 3 full-scale sweep jobs died this way within ~2
        # minutes of starting (0 results, silently -- lmms_eval's own
        # cli_evaluate catches the resulting RuntimeError and exits 0, so LSF
        # reported "Successfully completed"), while a 3rd, otherwise
        # identical job did not hit it -- i.e. transient contention, not a
        # permanent incompatibility. Retry with backoff rather than failing
        # the whole 17-benchmark job on one bad race.
        max_attempts = 5
        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            eval_logger.info(
                f"[RealSampleRetriever] spawning embed worker (attempt {attempt}/{max_attempts}): {' '.join(cmd)} "
                f"(CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<inherited>')})"
            )
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1, env=env)
            ready_line = self._proc.stdout.readline()
            if ready_line:
                status = json.loads(ready_line)
                eval_logger.info(f"[RealSampleRetriever] embed worker ready: {status}")
                return
            self._proc.wait()
            self._proc = None
            last_err = RuntimeError("embed_worker died before signaling ready (check its stderr in this process's own stderr stream)")
            if attempt < max_attempts:
                backoff_s = 10 * attempt
                eval_logger.warning(f"[RealSampleRetriever] embed worker failed to start (attempt {attempt}/{max_attempts}), retrying in {backoff_s}s: {last_err}")
                time.sleep(backoff_s)
        raise last_err

    def _query_embed_worker(self, task: str, doc_id: Any, image_path: Optional[str], text: Optional[str], k: int) -> List[Dict[str, Any]]:
        self._ensure_worker()
        # doc_id lets embed_worker's on-disk query-embedding cache key on a
        # stable identity: the eval set's queries are fixed across a whole
        # k/resize sweep, so this doc's embedding never changes between
        # sweep runs (see embed_worker.py's QueryEmbedCache).
        req = {"benchmark": task, "doc_id": doc_id, "image_path": image_path, "text": text, "k": k}
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("embed_worker closed its stdout unexpectedly (crashed?)")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"embed_worker error: {resp['error']}")
        return resp["results"]

    def _get_task_obj(self, task: str):
        if task not in self._task_cache:
            from lmms_eval.tasks import TaskManager, get_task_dict

            if self._task_manager is None:
                self._task_manager = TaskManager()
            task_dict = get_task_dict([task], self._task_manager)
            obj = task_dict[task]
            if isinstance(obj, tuple):
                _, obj = obj
            # Re-resolve lmms_eval_specific_kwargs/model_specific_target_kwargs
            # for our model name (get_task_dict doesn't take one, so the Task
            # was built with model_name=None -- always falls back to "default"
            # anyway since no task yaml lists "bagel_icl" specifically, but do
            # this properly rather than relying on that coincidence).
            obj.model_name = "bagel_icl"
            obj._prepare_model_specific_config()
            self._task_cache[task] = obj
        return self._task_cache[task]

    def _resolve_answer(self, task_obj, neighbor_doc: Dict[str, Any]) -> str:
        cfg_target = task_obj.config.doc_to_target
        if isinstance(cfg_target, str) and cfg_target not in task_obj.features:
            # Dangling reference to a non-existent column -- e.g.
            # vqav2_val_lite/textvqa_val_lite/ok_vqa_val2014_lite/
            # vizwiz_vqa_val_lite's yaml all say doc_to_target: "answer" but
            # the real column is "answers"; lmms-eval's own Task.doc_to_target
            # silently falls through its template path and returns the
            # literal string "answer" in that case (confirmed empirically).
            # Use the same answer-field-priority fallback build_retrieval_sets.py
            # uses instead of trusting that degenerate return value.
            return _rag_fallback_answer(neighbor_doc)
        target = task_obj.doc_to_target(neighbor_doc)
        if isinstance(target, list):
            return str(target[0]) if target else ""
        return str(target)

    def retrieve(self, task: str, doc_id: Any, doc: Dict[str, Any], k: int) -> List[ICLExample]:
        if k <= 0:
            return []
        task_obj = self._get_task_obj(task)

        query_images = _rag_as_image_list(doc.get("image"))
        tmp_path = None
        try:
            if query_images:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img2rgb(query_images[0]).save(tmp_path)
            query_text = _rag_embed_text(task, doc)
            results = self._query_embed_worker(task, doc_id, tmp_path, query_text, k)
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        examples = []
        for sample in results:
            neighbor_doc = {k2: v for k2, v in sample.items() if not k2.startswith("_")}
            neighbor_paths = sample["image_path"]
            neighbor_paths = neighbor_paths if isinstance(neighbor_paths, list) else [neighbor_paths]
            neighbor_images = [Image.open(p) for p in neighbor_paths]
            neighbor_doc["image"] = neighbor_images if isinstance(sample["image_path"], list) else neighbor_images[0]

            question = task_obj.doc_to_text(neighbor_doc)
            answer = self._resolve_answer(task_obj, neighbor_doc)
            if not answer:
                eval_logger.warning(f"[RealSampleRetriever] task '{task}': retrieved neighbor had no usable answer, skipping it")
                continue
            demo_images = [pil_img2rgb(im) for im in task_obj.doc_to_visual(neighbor_doc)]
            examples.append(ICLExample(question=question, answer=answer, images=demo_images))
        return examples


RETRIEVAL_STRATEGIES = {
    "synthetic": SyntheticSampleRetriever,
    "rag": RealSampleRetriever,
}


@register_model("bagel_icl")
class BagelICL(BagelUMM):
    """Bagel VQA evaluation with in-context demonstrations prepended to each
    query, e.g.:

        accelerate launch -m lmms_eval \\
            --model bagel_icl \\
            --model_args pretrained=/path/to/BAGEL-7B-MoT,icl_samples_root=$SIMIT_ROOT/similar_samples_llms_eval_lite/generated_samples_small_fullset_allverified,num_icl_examples=4 \\
            --tasks ai2d_lite \\
            --batch_size 1 \\
            --output_path ./logs/

    Only "understanding" and "think_understanding" modes are supported (ICL
    demonstrations for the image/edit-generation modes are out of scope here).
    """

    def __init__(
        self,
        icl_samples_root: str = "",
        num_icl_examples: int = 4,
        retrieval_strategy: str = "synthetic",
        icl_image_max_long_side: Optional[int] = None,
        rag_index_root: str = "",
        rag_embed_python: str = "",
        rag_embed_worker_script: str = "",
        rag_embed_gpu: Optional[str] = None,
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
        mode = kwargs.get("mode", "understanding")
        if mode not in ("understanding", "think_understanding"):
            raise ValueError(f"bagel_icl only supports mode 'understanding' or 'think_understanding', got '{mode}'")
        if retrieval_strategy not in RETRIEVAL_STRATEGIES:
            raise ValueError(f"Unknown retrieval_strategy '{retrieval_strategy}'. Must be one of: {list(RETRIEVAL_STRATEGIES)}")
        if retrieval_strategy == "synthetic" and not icl_samples_root:
            raise ValueError("bagel_icl requires icl_samples_root=<path to generated samples> for retrieval_strategy='synthetic'")
        if retrieval_strategy == "rag" and (not rag_index_root or not rag_embed_python):
            raise ValueError("bagel_icl requires rag_index_root=<path to rag_retrieval/data> and rag_embed_python=<path to rag_retrieval/.venv/bin/python> for retrieval_strategy='rag'")

        super().__init__(**kwargs)

        self.num_icl_examples = int(num_icl_examples)
        if retrieval_strategy == "synthetic":
            # Strict generalization of the old SyntheticSampleRetriever (num_generations=6
            # instead of hardcoded 4; extra unfilled slots are skipped harmlessly on
            # pools that only have 4). DF/ASC default to disabled, reproducing the
            # old "first k available, no filtering" behavior exactly.
            self.retriever: ICLRetriever = DFASCSyntheticSampleRetriever(
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
        else:
            self.retriever = RealSampleRetriever(
                index_root=rag_index_root,
                embed_python=rag_embed_python,
                embed_worker_script=rag_embed_worker_script or None,
                embed_gpu=rag_embed_gpu,
            )
        # Adaptive downscale for ICL demonstration images only (never the query
        # image): caps the long edge at this many pixels, preserving aspect
        # ratio, and never upscales. Default None = unlimited (no resize,
        # matches prior behavior exactly). Vision-token count scales with
        # resized pixel area (see bagel's ImageTransform/MaxLongEdgeMinShortEdgeResize),
        # so this exists to test whether trimming demo-image token cost helps.
        self.icl_image_max_long_side = int(icl_image_max_long_side) if icl_image_max_long_side is not None else None
        eval_logger.info(
            f"BagelICL initialized: retrieval_strategy='{retrieval_strategy}', num_icl_examples={self.num_icl_examples}, "
            f"icl_samples_root={icl_samples_root}, rag_index_root={rag_index_root}, rag_embed_python={rag_embed_python}, "
            f"icl_image_max_long_side={self.icl_image_max_long_side}"
        )

    def _resize_demo_image(self, img: Image.Image) -> Image.Image:
        """Downscale-only, aspect-ratio-preserving cap on the long edge for ICL
        demonstration images. No-op when icl_image_max_long_side is None, or
        when the image's long edge is already <= the cap (never upscales)."""
        if self.icl_image_max_long_side is None:
            return img
        w, h = img.size
        long_side = max(w, h)
        if long_side <= self.icl_image_max_long_side:
            return img
        scale = self.icl_image_max_long_side / long_side
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        return img.resize((new_w, new_h), Image.BICUBIC)

    def _build_icl_input_list(self, examples: List[ICLExample], query_input_list: List[Union[str, Image.Image]]) -> List[Union[str, Image.Image]]:
        if not examples:
            return query_input_list

        n = len(examples)
        input_list: List[Union[str, Image.Image]] = [f"Here {'is' if n == 1 else 'are'} {n} example{'s' if n > 1 else ''} to help you answer the question:\n"]

        for i, ex in enumerate(examples, 1):
            input_list.append(f"\nExample {i}:")
            # RealSampleRetriever supplies already-loaded (and, where relevant,
            # already-processed, e.g. refcoco's bbox drawn on) demo image(s)
            # directly via ex.images; SyntheticSampleRetriever leaves it None
            # and this falls back to the original image_path-based loading.
            imgs = ex.images if ex.images is not None else [pil_img2rgb(Image.open(ex.image_path))]
            for img in imgs:
                input_list.append(self._resize_demo_image(img))
            input_list.append(f"Question: {ex.question}\nAnswer: {ex.answer}")

        input_list.append("\nNow, answer this question:")
        input_list.extend(query_input_list)
        return input_list

    def _dump_target(self, task: str, doc: Dict[str, Any]) -> Optional[str]:
        """Best-effort ground-truth resolution for BAGEL_ICL_PROMPT_DUMP_DIR
        (visual-inspection only, never used for scoring): reuses
        RealSampleRetriever's dangling-doc_to_target-column fallback when
        that's the active retriever, since several *_lite.yaml configs point
        doc_to_target at a nonexistent column (see _resolve_answer)."""
        try:
            if isinstance(self.retriever, RealSampleRetriever):
                task_obj = self.retriever._get_task_obj(task)
                return self.retriever._resolve_answer(task_obj, doc)
            target = self.task_dict[task].doc_to_target(doc)
            return str(target[0]) if isinstance(target, list) and target else str(target)
        except Exception as e:
            return f"<target resolution failed: {e}>"

    def _dump_icl_prompt(
        self,
        dump_dir: str,
        task: str,
        doc_id: Any,
        icl_examples: List["ICLExample"],
        input_list: List[Union[str, Image.Image]],
        output_text: str,
        target: Optional[str],
    ) -> None:
        """Writes the exact ordered (text, image) sequence passed to
        interleave_inference for this query -- i.e. the real prompt, not a
        reconstruction -- to <dump_dir>/<task>/doc<doc_id>/prompt.json (+
        sibling img*.png files), for manual verification of ICL prompt
        building (see rag_retrieval/build_prompt_viewer.py)."""
        sample_dir = Path(dump_dir) / task / f"doc{doc_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        img_idx = 0
        for part in input_list:
            if isinstance(part, Image.Image):
                img_idx += 1
                img_name = f"img{img_idx}.png"
                part.save(sample_dir / img_name)
                parts.append({"type": "image", "path": img_name})
            else:
                parts.append({"type": "text", "text": part})
        record = {
            "task": task,
            "doc_id": doc_id,
            "num_icl_examples": len(icl_examples),
            "parts": parts,
            "model_output": output_text,
            "target": target,
        }
        with open(sample_dir / "prompt.json", "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (ICL)")
        dump_dir = os.environ.get("BAGEL_ICL_PROMPT_DUMP_DIR")

        rollout_counts: Dict[Any, int] = {}
        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            rollout_key = (task, doc_id)
            rollout_idx = rollout_counts.get(rollout_key, 0)
            rollout_counts[rollout_key] = rollout_idx + 1
            force_greedy = request.repeats is not None and request.repeats > 1 and rollout_idx == 0

            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)

            query_input_list = self._build_input_list(contexts, visuals)
            icl_examples = self.retriever.retrieve(task, doc_id, doc, self.num_icl_examples)
            input_list = self._build_icl_input_list(icl_examples, query_input_list)

            inference_params = self.inference_params.copy()
            inference_params["text_temperature"] = self.text_temperature
            if force_greedy:
                inference_params["do_sample"] = False

            eval_logger.debug(f"[generate_until ICL] {len(icl_examples)} example(s), input_list: {input_list}")

            with torch.autocast(device_type="cuda", enabled=True, dtype=self._torch_dtype):
                output_list = self.inferencer.interleave_inference(input_list, **inference_params)

            output_text = ""
            for output in output_list:
                if isinstance(output, str):
                    output_text = output

            if dump_dir:
                self._dump_icl_prompt(dump_dir, task, doc_id, icl_examples, input_list, output_text, self._dump_target(task, doc))

            res.append(output_text)
            pbar.update(1)

        pbar.close()
        return res
