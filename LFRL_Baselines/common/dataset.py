from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional

from .paths import EXCLUDED_BENCHMARKS, REAL_SAMPLES_ROOT


@dataclass
class RealSample:
    benchmark: str
    sample_id: str
    image_path: str
    prompt: str
    answers: List[str]  # GT reference answers; empty unless load_answers=True (never used for training).


def list_benchmarks() -> List[str]:
    return sorted(
        d
        for d in os.listdir(REAL_SAMPLES_ROOT)
        if os.path.isdir(os.path.join(REAL_SAMPLES_ROOT, d)) and d not in EXCLUDED_BENCHMARKS
    )


def _read_answers(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_real_samples(
    benchmarks: Optional[List[str]] = None,
    max_samples_per_benchmark: Optional[int] = None,
    seed: int = 0,
    load_answers: bool = False,
) -> List[RealSample]:
    """Loads only the real (`original_sample_*`) files, never the synthesized ones
    living alongside them in the same sample directories. The methods are label-free:
    eval-split answers are not loaded unless load_answers=True (analysis only), so the
    trainers' optional GT "monitor_accuracy" logging stays off."""
    benchmarks = benchmarks or list_benchmarks()
    rng = random.Random(seed)
    samples: List[RealSample] = []
    for benchmark in benchmarks:
        bench_dir = os.path.join(REAL_SAMPLES_ROOT, benchmark)
        sample_dirs = sorted(d for d in os.listdir(bench_dir) if d.startswith("sample"))
        if max_samples_per_benchmark is not None and len(sample_dirs) > max_samples_per_benchmark:
            sample_dirs = rng.sample(sample_dirs, max_samples_per_benchmark)
        for sample_id in sample_dirs:
            sample_dir = os.path.join(bench_dir, sample_id)
            image_path = os.path.join(sample_dir, "original_sample.png")
            prompt_path = os.path.join(sample_dir, "original_sample_prompt.txt")
            answers_path = os.path.join(sample_dir, "original_sample_answers.txt")
            if not (os.path.isfile(image_path) and os.path.isfile(prompt_path) and os.path.isfile(answers_path)):
                continue
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            # The saved prompt files always carry a "Question: " prefix, but
            # lmms-eval's own doc_to_text (confirmed directly from a saved
            # eval transcript: input='Is it overcast?\n...', no prefix) does
            # NOT include it -- the model is evaluated without this prefix,
            # so training must match, or every run trains on a prompt format
            # it's never evaluated on.
            if prompt.startswith("Question:"):
                prompt = prompt[len("Question:") :].lstrip()
            samples.append(
                RealSample(
                    benchmark=benchmark,
                    sample_id=sample_id,
                    image_path=image_path,
                    prompt=prompt,
                    answers=_read_answers(answers_path) if load_answers else [],
                )
            )
    return samples
