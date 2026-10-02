"""A mixed "general visual QA" set from LMMs-Eval-Lite, close to what demo
visitors ask: everyday photos (VizWiz, VQAv2), knowledge (OK-VQA), reading text
(TextVQA), charts and documents (ChartQA, DocVQA) and multiple-choice reasoning
(MMBench, AI2D). Each example carries its benchmark's own metric.
"""

import ast

from datasets import load_dataset

VQA = "Answer the question using a single word or phrase."


def _list(v):
    if isinstance(v, list):
        return v
    try:
        v = ast.literal_eval(v)
    except Exception:
        pass
    return v if isinstance(v, list) else [v]


def _options(row, keys):
    opts = [(k, str(row[k])) for k in keys if str(row.get(k, "nan")) not in ("nan", "None", "")]
    return "\n".join(f"{k}. {v}" for k, v in opts)


def _mmbench(r):
    hint = str(r.get("hint", "nan"))
    q = (hint + "\n" if hint not in ("nan", "None", "") else "") + r["question"]
    return (q + "\n" + _options(r, "ABCD") + "\nAnswer with the option's letter from the given choices directly.",
            [str(r["answer"])], "multiple_choice")


def _ai2d(r):
    opts = _list(r["options"])
    q = r["question"] + "\n" + "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
    return q + "\nAnswer with the option's letter from the given choices directly.", [chr(65 + int(r["answer"]))], \
        "multiple_choice"


SUBSETS = {
    "vizwiz_vqa_val": lambda r: (f"{r['question']}\nWhen the provided information is insufficient, respond with "
                                 f"'Unanswerable'.\n{VQA}", [str(a) for a in _list(r["answers"])], "vqa_accuracy"),
    "vqav2_val": lambda r: (f"{r['question']}\n{VQA}",
                            [a["answer"] if isinstance(a, dict) else str(a) for a in _list(r["answers"])],
                            "vqa_accuracy"),
    "ok_vqa_val2014": lambda r: (f"{r['question']}\n{VQA}", [str(a) for a in _list(r["answers"])], "vqa_accuracy"),
    "textvqa_val": lambda r: (f"{r['question']}\n{VQA}", [str(a) for a in _list(r["answers"])], "vqa_accuracy"),
    "chartqa": lambda r: (f"{r['question']}\nAnswer the question with a single word.",
                          [str(a) for a in _list(r["answer"])], "relaxed_accuracy"),
    "docvqa_val": lambda r: (f"{r['question']}\n{VQA}", [str(a) for a in _list(r["answers"])], "anls"),
    "mmbench_en_dev": _mmbench,
    "ai2d": _ai2d,
}


def general_qa(per_subset: int = 8, offset: int = 300) -> list[dict]:
    """``per_subset`` examples from each subset, starting at row ``offset``
    (rows below 300 were used for the package benchmarks)."""
    out = []
    for sub, fn in SUBSETS.items():
        ds = load_dataset("lmms-lab/LMMs-Eval-Lite", sub)["lite"]
        for i in range(offset, offset + per_subset):
            r = ds[i]
            question, answers, metric = fn(r)
            out.append({"image": r["image"].convert("RGB"), "question": question, "answers": answers,
                        "metric": metric, "subset": sub})
    return out
