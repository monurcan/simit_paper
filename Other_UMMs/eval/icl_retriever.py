"""Selects the SIMIT-ICL in-context demonstrations for a given doc: the
synthesized (image, question, answer) triplets synthesis/synthesize_vqa.py
generated for that exact doc, optionally filtered by difficulty (DF) and
capped by an adaptive per-doc budget (ASC) computed from the real doc's own
zero-shot confidence. Port of
lmms-eval/lmms_eval/models/simple/bagel_icl.py's DFASCSyntheticSampleRetriever
+ _compute_tau/_adaptive_k -- same formula, same folder-layout convention
(now guaranteed to match exactly, since it's this project's own
synthesis/synthesize_vqa.py + confidence/compute_confidence.py that produced
it), same consumption order (DF filter, then ASC cap, then caller's
requested k).
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass
class ICLExample:
    image_path: str
    question: str
    answer: str


def _compute_tau(epsilon: float, A0: float, A1: float, B: float, num_generations: int) -> Dict[int, float]:
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


class DFASCSyntheticSampleRetriever:
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
        self.root = Path(root)
        self.num_generations = int(num_generations)
        self.df_enabled = bool(df_enabled)
        self.asc_enabled = bool(asc_enabled)
        self.epsilon, self.A0, self.A1, self.B = float(epsilon), float(A0), float(A1), float(B)
        self.T_low, self.T_high = float(T_low), float(T_high)
        self._tau = _compute_tau(self.epsilon, self.A0, self.A1, self.B, self.num_generations) if self.asc_enabled else None

    @staticmethod
    def _benchmark_dir_name(task: str) -> str:
        """Map an lmms-eval task name (e.g. "ok_vqa_val2014_lite") to the
        subset-name directory synthesis/synthesize_vqa.py wrote to (e.g.
        "ok_vqa_val2014", no "_lite" suffix) -- same mapping as the original
        bagel_icl.py's SyntheticSampleRetriever._benchmark_dir_for_task."""
        if task.startswith("mathvista"):
            return "mathvista"
        if task.endswith("_lite_val50"):  # 50-sample validation split -> <bench>_val50/
            return task[: -len("_lite_val50")] + "_val50"
        if task.endswith("_lite"):
            return task[: -len("_lite")]
        return task

    def retrieve(self, task: str, doc_id: Any, doc: Dict[str, Any], k: int) -> List[ICLExample]:
        bench_dir = self.root / self._benchmark_dir_name(task)
        if not bench_dir.is_dir():
            return []
        sample_dir = bench_dir / f"sample{doc_id}"
        if not sample_dir.is_dir():
            return []

        candidates = []
        for i in range(self.num_generations):
            img_f = sample_dir / f"synthesized_sample_{i}.png"
            prompt_f = sample_dir / f"synthesized_sample_{i}_prompt.txt"
            ans_f = sample_dir / f"synthesized_sample_{i}_answers.txt"
            if not (img_f.exists() and prompt_f.exists() and ans_f.exists()):
                continue
            question = prompt_f.read_text().strip()
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
