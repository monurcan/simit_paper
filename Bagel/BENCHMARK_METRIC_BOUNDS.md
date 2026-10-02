# Benchmark metric bounds & group-averaging rules

Reference table for the 18-benchmark eval suite used across the synthetic-data
and real-data LoRA studies. Compiled 2026-08-19 by directly inspecting each
benchmark's `results.json` metric keys and its lmms-eval scorer implementation
(`lmms_eval/tasks/<bench>/utils.py`), not assumed from benchmark names.

Retrieval/Training and Val. columns are the synthetic-data-study numbers
(unrelated to the real-data study's pool sizes, which are much larger and
listed separately in `prepare_vqa_data_real_rag.py`).

| Task Domain | Dataset | Retrieval/Training | Val. | Eval. | Metric | Lower | Upper |
|---|---|---:|---:|---:|---|---:|---:|
| Doc & Infographic Understanding | ChartQA | 1946 | 50 | 500 | Relaxed Accuracy | 0 | 1 |
| Doc & Infographic Understanding | DocVQA | 4799 | 50 | 500 | ANLS | 0 | 1 |
| Doc & Infographic Understanding | InfoVQA | 2251 | 50 | 500 | ANLS | 0 | 1 |
| Image Understanding & Captioning | Flickr30k | 31233 | 50 | 500 | CIDEr | 0 | unbounded† |
| Image Understanding & Captioning | NoCaps | 3950 | 50 | 500 | CIDEr | 0 | unbounded† |
| Image Understanding & Captioning | TextCaps | 24569 | 50 | 500 | CIDEr | 0 | unbounded† |
| Image Understanding & Captioning | RefCOCO | 16988 | 50 | 500 | CIDEr‡ | 0 | unbounded† |
| Image Understanding & Captioning | COCO | 4450 | 50 | 500 | CIDEr | 0 | unbounded† |
| Visual Question Answering | GQA | 171623 | 50 | 500 | Exact Match (soft VQA-acc) | 0 | 1 |
| Visual Question Answering | OKVQA | 4496 | 50 | 500 | Exact Match (soft VQA-acc) | 0 | 1 |
| Visual Question Answering | VizWiz-VQA | 3769 | 50 | 500 | Exact Match (soft VQA-acc) | 0 | 1 |
| Visual Question Answering | VQA-V2(-Small) | 21325 | 50 | 500 | Exact Match (soft VQA-acc) | 0 | 1 |
| Visual Question Answering | TextVQA | 39052 | 50 | 500 | Exact Match (soft VQA-acc) | 0 | 1 |
| Math & Science | MathVista | 0 | 0 | 1000 | LLM-as-Judge Accuracy | 0 | 100 |
| Math & Science | AI2D | 2538 | 50 | 500 | Exact Match | 0 | 1 |
| Multi-discipline | MM-Bench (CN) | 3778 | 50 | 500 | GPT-Eval Score | 0 | 100 |
| Multi-discipline | MM-Bench (EN) | 3842 | 50 | 500 | GPT-Eval Score | 0 | 100 |
| Multi-discipline | Seed-Bench | 17440 | 50 | 500 | Accuracy | 0 | 1 |

† CIDEr has no fixed ceiling — TF-IDF-weighted n-gram cosine similarity, not a
percentage. This project's (unscaled, ×1 not the ×100 convention some papers
use) values topped out around 1.3–1.4 in the studies run so far.

‡ RefCOCO in this pipeline is *not* scored the standard way (IoU/box
accuracy). The task is reframed as "describe the circled region," scored via
CIDEr against 1–3 short reference phrases — structurally different reference
sets than COCO's 5 full captions, which matters for the CIDEr-comparability
caveat below.

## Group-averaging rules

**Before averaging any group, normalize every metric in it to 0–100** (i.e.
multiply the `[0,1]`-bounded metrics by 100). Only after that rescale do the
domain groups below become arithmetically valid to combine — mixing raw 0–1
and 0–100 values in one mean is a scale bug, not a judgment call.

- **Visual Question Answering**: safe to average, even unweighted. All five
  benchmarks use the identical metric (soft VQA-accuracy exact-match) on the
  identical scale. Standard practice in the VLM literature. Caveat: an
  unweighted mean treats GQA (171k pool) and VizWiz (3.7k pool) as equally
  important — a modeling choice, not a bug, but worth stating if reported.

- **Doc & Infographic Understanding**: plausible, but mixes metric
  *semantics*, not just numbers on the same scale. Relaxed Accuracy is
  essentially binary per-sample; ANLS is continuous edit-distance partial
  credit. Both live in [0,1] so the arithmetic doesn't break, but a 0.85 in
  one isn't calibrated the same way as a 0.85 in the other. Common practice,
  defensible, not perfectly apples-to-apples.

- **Image Understanding & Captioning**: the shakiest group, even though every
  entry is nominally "CIDEr." CIDEr's TF-IDF term-weighting is computed from
  each dataset's own reference corpus, so the same caption quality produces
  different raw CIDEr numbers on COCO vs. Flickr30k vs. NoCaps independent of
  model quality — averaging raw CIDEr across datasets is a summary heuristic,
  not a principled combination. RefCOCO makes this worse specifically in this
  pipeline (see ‡ above: 1–3 short referring phrases vs. COCO's 5 full
  captions). Since it's unbounded, there's no rescale step needed here (no
  0–100 conversion applies to CIDEr) — average it as-is, understanding it's a
  looser combination than the bounded-metric groups.

- **Math & Science**: MathVista (0–100) + AI2D (0–1, → rescale to 0–100).
  Must rescale AI2D before averaging with MathVista, or the mean is wrong.

- **Multi-discipline**: MM-Bench CN/EN (0–100) + Seed-Bench (0–1, → rescale
  to 0–100). Must rescale Seed-Bench before averaging with the MM-Bench pair.

**Bottom line**: safe to average unweighted when metric *and* scale match
exactly (VQA); defensible-with-a-caveat when scale matches but semantics
differ (Doc & Infographic); a summary heuristic, not strictly principled, for
the CIDEr group (Image Understanding & Captioning) — but no rescale needed
there since it's unbounded either way; a **hard rescale-to-0–100 requirement**
whenever raw scales differ outright (Math & Science, Multi-discipline).
