# Table 2 data-quality audit

Fills in Table 2's "Data Quality" column (component ablation of TT-SI) via
the same agentic-evaluation protocol as `analyses/label_noise_model_noise`,
on the same 170 queries (10/benchmark x 17 benchmarks), for each of the 6
non-baseline rows of Table 2 (row 1, the BAGEL-7B baseline, has no
synthesized data and is excluded).

Unlike the label-noise audit, this is a single number per setting: the
average binary rubric score (fraction of audited triplets judged correct),
no c/c_wrong/c_correct decomposition.

## Setting -> Table 2 row mapping

Confirmed with the user 2026-09-15 (folder name doesn't always self-evidently
map to a component -- `force_natural` ablates the Skill Library, not a
"category" setting, since without SL the pipeline falls back to BAGEL's
native/natural image generation instead of a category-specific render
skill).

| setting key | Table 2 row | generated_samples root | disable_df | disable_adaptive_k |
|---|---|---|---|---|
| `full` | Full pipeline | `..._discard_unverified_v2` | per-bench config | False |
| `no_aba` | w/o ABA | `..._discard_unverified_v2` | per-bench config | True |
| `no_aba_no_df` | w/o ABA, DF | `..._discard_unverified_no_diff_filtering_4_samples` | True | True |
| `no_aba_no_df_no_dp` | w/o ABA, DP, DF | `..._no_diversity_prompt_no_diff_filtering_4_samples` | True | True |
| `no_aba_no_df_no_sl` | w/o ABA, SL, DF | `..._force_natural_no_diff_filtering_4_samples` | True | True |
| `no_aba_no_df_no_v` | w/o ABA, V, DF | `..._keep_unverified_no_diff_filtering_4_samples` | True | True |

`full`/`no_aba` reuse `Bagel/prepare_vqa_data_v2_dfadaptive.py`'s
`build_records()` (imported read-only) with the real per-benchmark DF+ASC
hyperparameters from `analyses/label_noise_model_noise/provenance/
simit_df_asc_configs.json` (same config already used/approved for the
label-noise audit's SIMIT extraction -- `full`'s 72/170 availability count
matches that audit's `simit_selected.jsonl` exactly).

The other 4 settings fix the sample budget (disable_df=True,
disable_adaptive_k=True), matching the paper's stated ablation design. Three
of their four raw-generation folders have no confidence JSONs at all (never
run through confidence scoring), which `build_records()` requires
unconditionally even though it never reads them on this code path -- so
these 4 settings use `collect_fixed_budget_candidates()` in
`01_extract_settings.py` instead: the same candidate-collection loop minus
the confidence gate.

## Pipeline (run in order)

| script | what it does | output |
|---|---|---|
| `01_extract_settings.py` | pool-samples up to 10 synthesized triplets per benchmark per setting (see below) | `cache/<setting>_selected.jsonl` |
| `02_build_eval_packets.py` | builds blinded (image, question, answer) packets per benchmark, across all 6 settings | `eval_packets/<bench>_packets.jsonl`, `provenance/audit_manifest.jsonl` (de-anonymizing map, NOT for the evaluator) |
| *(external)* | one agent per benchmark judges every packet against `provenance/agentic_eval_rubric.md` (unmodified copy of the label-noise audit's rubric), writes `eval_packets/<bench>_verdicts.jsonl` | -- |
| `03_aggregate.py` | joins verdicts with the manifest, computes the average score per setting (+ per-benchmark breakdown) | `results/summary.json` |

## Final sampling design (2026-09-16)

Went through two prior designs before landing here (see git history of this
file if curious); both had real flaws the user caught:

1. **One retained candidate per query, uniformly at random.** Wrong: settings
   retain very different numbers of candidates per query (`full` ~0.9/query
   since real DF+ASC discards most of them, vs. ~3.6-4.0/query for the
   fixed-budget ablations), so sampling one per query regardless massively
   under-represents denser settings' actual synthesized-corpus composition.
2. **Pool all candidates per benchmark, weighted-sample up to 10.** Better,
   but still let each setting draw from a *different* query subset (whatever
   it happened to retain data for), which isn't a fair comparison -- e.g. a
   benchmark where `full` retains data for only 3 of its 10 queries would
   still contribute up to 10 samples for the loosest ablation settings.

**Current design**, per explicit user instructions (2026-09-16):

- **Fairness mask**: `compute_full_eligible_mask()` in `01_extract_settings.py`
  runs the real full pipeline (real per-benchmark DF+ASC config, ABA on) and
  records which of the 10 originally-selected queries it retains >=1
  candidate for. A query full retains nothing for is dropped from EVERY
  setting's audit -- never scored as a "0" for full, and never given to the
  looser settings either, since there's no full-pipeline data to compare
  against there. (DF's confidence-band filtering is identical between
  `full` and `no_aba` -- only the ASC/adaptive-k cap differs -- so this one
  mask, computed from `full`, is valid for both.) 3 of the 17 benchmarks
  (docvqa_val, chartqa, mmbench_en_dev) turn out to have ZERO
  full-pipeline-eligible queries among the original 10 selected, and so
  contribute 0 triplets to every setting -- not a bug, genuine full-pipeline
  data sparsity there.
- **No subsampling within a query**: for each eligible query, EVERY retained
  candidate under a given setting's own selection logic is used (all of
  them, not a random subset) -- "all the labels used in that variant for the
  same real sample" are the real supervision signal that setting
  contributes for that query.
- **Speed**: only `N_QUERIES_PER_BENCH=2` eligible queries per benchmark
  (seeded random pick) are audited, keeping the agentic judging pass small.

## Query selection

Reuses the exact same 170 query IDs already drawn in
`analyses/label_noise_model_noise/cache/selected_queries.jsonl` (per user
decision, for a paired comparison across all 6 settings -- no new randomness
introduced). Verified: all 170 sample directories exist in all 5 relevant
`generated_samples_*` roots.
