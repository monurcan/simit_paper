# eval_ci_tools

Bootstrap confidence intervals for BAGEL lmms-eval outputs, computed from
per-example predictions already saved to disk — no reruns needed.

## Usage

```bash
# Score one directory (one checkpoint/run), all 17 benchmarks it contains
python eval_ci.py /path/to/some_eval_output_dir

# Score several
python eval_ci.py dir1 dir2 dir3

# Or a glob (e.g. every checkpoint of one training run)
python eval_ci.py --glob "benchmark_eval_outputs/ok_vqa_val2014_nodf4scale_*"

# Restrict to specific benchmarks
python eval_ci.py some_dir --benches ok_vqa_val2014 refcoco_bbox_val

# Save full results (per-benchmark + per-domain) to JSON
python eval_ci.py some_dir --json out.json
```

Each directory should contain (anywhere under it) a
`models__BAGEL-7B-MoT/*_samples_<bench>_lite.jsonl` file per benchmark run
there — that's lmms-eval's standard per-example output location. The script
finds whichever of the 17 known benchmarks are present, scores each one, and
aggregates to the project's standard 5-domain grouping (VQA, Doc &
Infographic, Captioning, Multi-discipline, Math & Science).

## What this does and doesn't capture

**Does:** eval-set sampling variance — how much a benchmark's score would
wobble depending on which examples happened to land in the ~500-example
eval subset. Computed via bootstrap resampling (default B=1000) of the saved
per-example predictions.

**Does not:** training-run-to-run variance. Every score in this project
comes from exactly one training run (no repeated seeds anywhere), so these
CIs are a lower bound on true uncertainty, not the full picture. Label plots
accordingly ("eval-set bootstrap CI", not "seed variance").

## Metric-specific handling (the parts that are easy to get wrong)

- **Accuracy-style metrics** (exact_match, ANLS, relaxed_overall, seed_all):
  plain per-example bootstrap, straightforward.
- **mmbench (en/cn):** the raw `prediction` field is unextracted model text
  (e.g. `"A."` not `"A"`) — scored via a reimplementation of MMBench's own
  `can_infer_option` extractor, verified to reproduce lmms-eval's official
  cached score to full float precision. By default this also reproduces the
  official circular-eval convention (only `index < 1_000_000` rows count,
  ~26% of a 500-row sample — the rest are rotated variants of *other*
  questions used only as consistency checks). Pass
  `--mmbench-include-rotations` to instead score every row independently
  (useful if you want rotation-sensitivity itself as a signal, not something
  discarded) — this gives a visibly different, larger-n number; know which
  mode you're using before comparing across runs.
- **CIDEr** (flickr30k, nocaps, textcaps, refcoco, coco2017_cap): corpus-level
  metric, not a simple per-example average. Computed via the same
  PTBTokenizer + pycocoevalcap pipeline lmms-eval itself uses (needs `java`
  on PATH — confirmed available). Per-image CIDEr scores are computed once
  (fixing corpus-level IDF weights against the full ~500-image set), then
  that per-image array is bootstrap-resampled — the standard "fixed corpus
  statistics" approximation, not a full IDF refit per resample.

## Domain aggregation

`std_domain = sqrt(sum of per-benchmark variances) / n_benchmarks` (valid
since benchmarks have disjoint eval sets — independent). [0,1]-bounded
accuracy metrics are rescaled ×100 before averaging into a domain; CIDEr is
averaged raw (unbounded, corpus-relative); mmbench's gpt_eval_score-style
fraction is also ×100. This matches the rescale convention already
established in `Bagel/BENCHMARK_METRIC_BOUNDS.md`.

## Validated against

Reproduces every known-correct cached score exactly: pretrained baseline
domain averages (VQA 64.64, Doc&Info 78.40, Captioning 1.076, Multi-discipline
82.16 with official mmbench scoring, Math&Sci 86.40), nocaps CIDEr
(0.7550864730537687 to full precision), mmbench_en official score
(84.0909090909091 to full precision).
