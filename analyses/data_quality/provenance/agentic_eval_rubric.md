# Agentic evaluation rubric (external evaluator protocol)

This is the rubric used by each per-benchmark evaluation subagent (step 8 of
the pipeline). Kept here for provenance / reproducibility -- this exact text
is what each subagent is instructed to apply.

## Task

You will judge a list of (image, question, answer) triplets, one at a time.
For each, decide: **is this triplet correct?** — does the image actually
support answering the question, and does the given answer correctly answer
that question about that image? You are NOT told which method or system
produced any answer, and you should not try to guess — judge every triplet
on its own merits, using the same standard for all of them.

## How to judge correctness (apply per question type)

- **Multiple choice** (question lists lettered options): the answer is
  correct only if it selects the option that is actually true of the image.
- **Short-answer / single-word-or-phrase VQA**: the answer is correct if it
  is a reasonable, factually accurate response to the question given the
  image. Minor differences in phrasing, capitalization, singular/plural, or
  a synonym that means the same thing do NOT make an answer incorrect.
- **Captioning** ("provide a caption", "describe the image"): correct if the
  caption accurately describes salient, verifiable content of the image. It
  does not need to match any specific reference wording, and does not need
  to describe every detail — it needs to not be wrong or fabricated.
- **Document / chart / infographic QA**: the answer must match what is
  actually shown in or computable from the image (a number, label, or
  short text span). Treat this like short-answer VQA otherwise.
- **"Unanswerable" / "cannot be determined" as the given answer**: correct
  ONLY if the image genuinely does not contain enough information to answer
  the question. If the image clearly does support an answer, "Unanswerable"
  is INCORRECT.

## What to ignore

Cosmetic image imperfections (slightly off rendering, minor visual
artifacts, mild blur/upscaling, imperfect but legible text) are irrelevant
UNLESS they actually obscure or change the information needed to answer the
question. Judge the content, not the production quality.

## When to flag "ambiguous" instead of deciding

Use `flagged_ambiguous: true` sparingly — only when a careful, reasonable
person genuinely could not decide correct-vs-incorrect (e.g. the image
failed to load / is corrupted, the question itself is nonsensical or
untranslatable, or the answer is a genuine judgment call with real
ambiguity). Do NOT use it just because a question is hard, or because you
are not 100% certain — make your best judgment call in those cases instead
and explain why in the rationale. Ambiguous items are excluded from the
statistics and reported separately, so overusing this flag would throw away
real signal.

## Output format

Exactly one JSON object per line (JSONL), one line per input triplet, in
any order, written to the exact output path you're given:

```json
{"audit_id": "<copied from input>", "verdict": true, "flagged_ambiguous": false, "rationale": "<one short sentence>"}
```

`verdict`: `true` = correct triplet, `false` = incorrect triplet.
`rationale`: one short sentence explaining the call (or the ambiguity, if
flagged). Every input triplet must get exactly one output line.
