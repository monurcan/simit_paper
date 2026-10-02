"""Per-subset field mapping for lmms-lab/LMMs-Eval-Lite -> human-readable
prompt + answer text. Excludes the answer field from the prompt itself.

gqa has no `image` field in the lite dataset (only `imageId`). Images are
pre-fetched by fetch_gqa_images.py and stored in gqa_images/<imageId>.jpg;
the `image_dir` key points get_image() to that cache instead of example["image"].

answer_instruction appends the standard lmms-eval-style answering
instruction for that task type (MCQ / short-answer VQA / vizwiz's
unanswerable-aware variant); omitted for captioning-style subsets, whose
`question` field is already a full instruction, not a question expecting a
short answer.
"""

import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[2])
_MCQ_INSTRUCTION = "Answer with a single letter."
_VQA_INSTRUCTION = "Answer the question using a single word or phrase."
_VIZWIZ_INSTRUCTION = (
    "When the provided information is insufficient, respond with 'Unanswerable'.\n"
    "Answer the question using a single word or phrase."
)

_GQA_IMAGE_DIR = SIMIT_ROOT + "/similar_samples_llms_eval_lite/gqa_images"

SUBSET_CONFIG = {
    "ai2d": dict(text="question", choices_list="options", answer="answer", answer_is_index=True,
                 answer_instruction=_MCQ_INSTRUCTION),
    "gqa": dict(text="question", answer="answer", answer_instruction=_VQA_INSTRUCTION,
                image_dir=_GQA_IMAGE_DIR, image_id_field="imageId"),
    "chartqa": dict(text="question", answer="answer", answer_instruction=_VQA_INSTRUCTION),
    "coco2017_cap_val": dict(text="question", answer="answer"),
    "docvqa_val": dict(text="question", answer="answers", answer_instruction=_VQA_INSTRUCTION),
    "flickr30k_test": dict(default_text="Describe this image in detail.", answer="caption"),
    "infovqa_val": dict(text="question", answer="answers", answer_instruction=_VQA_INSTRUCTION),
    "mmbench_cn_dev": dict(text="question", hint="hint", choices_named=["A", "B", "C", "D"], answer="answer",
                            answer_instruction=_MCQ_INSTRUCTION),
    "mmbench_en_dev": dict(text="question", hint="hint", choices_named=["A", "B", "C", "D"], answer="answer",
                            answer_instruction=_MCQ_INSTRUCTION),
    "nocaps_val": dict(default_text="Describe this image in detail.", answer="annotations_captions"),
    "ok_vqa_val2014": dict(text="question", answer="answers", answer_instruction=_VQA_INSTRUCTION),
    "refcoco_bbox_val": dict(text="question", answer="answer"),
    "seedbench": dict(
        text="question",
        choices_named_src=[("A", "choice_a"), ("B", "choice_b"), ("C", "choice_c"), ("D", "choice_d")],
        answer="answer", image_is_list=True, answer_instruction=_MCQ_INSTRUCTION,
    ),
    "textcaps_val": dict(text="question", answer="caption_str"),
    "textvqa_val": dict(text="question", answer="answers", answer_instruction=_VQA_INSTRUCTION),
    "vizwiz_vqa_val": dict(text="question", answer="answers", answer_instruction=_VIZWIZ_INSTRUCTION),
    "vqav2_val": dict(text="question", answer="multiple_choice_answer", answer_instruction=_VQA_INSTRUCTION),
    "mathvista": dict(full_query="query", answer="answer", image_field="decoded_image",
                       hf_repo="AI4Math/MathVista", hf_split="testmini"),
}


def _is_missing(v) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s == "" or s == "nan"


def get_image(example: dict, cfg: dict):
    if "image_dir" in cfg and example.get("image") is None:  # gqa validation rows embed their image
        from PIL import Image
        img_id = example[cfg["image_id_field"]]
        return Image.open(f"{cfg['image_dir']}/{img_id}.jpg").convert("RGB")
    img = example[cfg.get("image_field", "image")]
    if cfg.get("image_is_list"):
        img = img[0]
    return img.convert("RGB")


def build_prompt(example: dict, cfg: dict) -> str:
    if "full_query" in cfg:
        return example[cfg["full_query"]]
    lines = []
    if "hint" in cfg and not _is_missing(example.get(cfg["hint"])):
        lines.append(f"Context: {example[cfg['hint']]}")
    text = example.get(cfg["text"]) if "text" in cfg else cfg.get("default_text", "Describe this image.")
    lines.append(f"Question: {text}")

    if "choices_list" in cfg:
        choices = example[cfg["choices_list"]]
        letters = "ABCDEFGHIJ"
        opts = [f"{letters[i]}) {c}" for i, c in enumerate(choices) if not _is_missing(c)]
        if opts:
            lines.append("Options:\n" + "\n".join(opts))
    elif "choices_named" in cfg:
        opts = [f"{letter}) {example[letter]}" for letter in cfg["choices_named"] if not _is_missing(example.get(letter))]
        if opts:
            lines.append("Options:\n" + "\n".join(opts))
    elif "choices_named_src" in cfg:
        opts = [f"{letter}) {example[field]}" for letter, field in cfg["choices_named_src"] if not _is_missing(example.get(field))]
        if opts:
            lines.append("Options:\n" + "\n".join(opts))

    if "answer_instruction" in cfg:
        lines.append(cfg["answer_instruction"])

    return "\n".join(lines)


def build_answer_text(example: dict, cfg: dict) -> str:
    val = example.get(cfg["answer"])
    if cfg.get("answer_is_index") and "choices_list" in cfg:
        choices = example[cfg["choices_list"]]
        try:
            val = choices[int(val)]
        except (ValueError, IndexError, TypeError):
            pass
    if isinstance(val, list):
        return "\n".join(str(v) for v in val)
    return str(val)
