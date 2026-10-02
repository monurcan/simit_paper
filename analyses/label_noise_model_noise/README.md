# Label-noise / model-noise audit (Sec 4.5)

Audits whether each method's supervision is correct for original test queries
the base model gets wrong, per the protocol in the paper draft's Section 4.5
("Why SIMIT Outperforms LFRL") and the user's refined methodology (positive-
advantage-weighted sampling instead of naive majority-vote/argmax auditing).

Pilot scale: 10 queries/benchmark x 17 benchmarks = 170 queries, one sampled
pseudolabel per method per query. `01_select_queries.py` tracks a used-ids
registry so a larger future draw adds NEW queries rather than resampling.

## Pipeline (run in order)

| script | what it does | output |
|---|---|---|
| `01_select_queries.py` | seeded random query selection | `cache/selected_queries.jsonl`, `cache/used_ids_registry.json` |
| `02_extract_baseline.py` | pulls cached zero-shot greedy answers (no inference) | `cache/baseline_answers.jsonl` |
| `03_extract_lfrl_rollouts.py` | reconstructs (rollout, reward) pairs for TTRL/EMPO/TTRV from `job_outputs/*.out` + `train_log.jsonl` (no cache existed for this before -- see `lfrl_log_utils.py`), recomputes each method's own advantage formula, samples per P(j\|i,m) | `cache/lfrl_{ttrl,empo,ttrv}_selected.jsonl` |
| `04_regenerate_srlm.py` | SRLM has NO cached rollout data anywhere -- regenerates live on GPU from the frozen base model (lora_rank=0, matching "before any parameter update") using the real self-judge prompt | `cache/lfrl_srlm_selected.jsonl` |
| `05_extract_simit.py` | reuses `Bagel/prepare_vqa_data_v2_dfadaptive.py`'s own `build_records()` (imported, unmodified) with per-benchmark DF+ASC hyperparameters | `cache/simit_selected.jsonl` |
| `06_build_eval_packets.py` | builds blinded (image, question, answer) packets, one per available M/L record, no method/reward labels | `eval_packets/<bench>_packets.jsonl`, `provenance/audit_manifest.jsonl` (de-anonymizing map, NOT for the evaluator) |
| *(external)* | one agent per benchmark judges every packet against `provenance/agentic_eval_rubric.md`, writes `eval_packets/<bench>_verdicts.jsonl` | -- |
| *(merge)* | `provenance/eval_verdicts.jsonl` = concatenation of all `*_verdicts.jsonl`, after validating ID coverage/schema | -- |
| `07_aggregate.py` | joins L+M, computes c/c_wrong/c_correct/Delta per method (pooled + macro-per-benchmark + common-eligible-subset), bootstrap 95% CIs | `results/summary.json`, `results/lm_realizations.jsonl` |

## Key design decisions (see script docstrings for full rationale)

- **Canonical run per (method, benchmark)**: `provenance/canonical_lfrl_runs.json`, ground truth given directly by the user (best full-benchmark eval score across every hyperparameter variant tried) -- NOT a naive `f"{method}_{bench}"` formula, which is wrong for ttrv/ai2d and 3 srlm benchmarks.
- **Resumed-run join**: `.out` files can be a partial-resume-and-append pattern OR a full-restart-and-overwrite pattern depending on the run; `lfrl_log_utils.load_run_records()` tries growing suffixes from the most recent `.out` file backward and takes the first (smallest) suffix whose completions count matches `train_log.jsonl`'s line count exactly -- validated against the training script's own `n_updated` bookkeeping (independent cross-check, matches exactly).
- **"Initial supervision before parameter updates"** = a sample_id's minimum-epoch occurrence in `train_log.jsonl`; if a resume re-ran epoch 1 more than once, the chronologically LAST such occurrence is used (the one whose update actually survived into the persisted checkpoint trajectory).
- **SIMIT DF+ASC hyperparameters**: `provenance/simit_df_asc_configs.json` -- 12 benchmarks from the user's own winning-config table, ai2d with DF disabled (per that table), 5 benchmarks missing from the table filled from the best Optuna/hand-search trial in `Bagel/df_asc_sweep_results.jsonl` / `Bagel/icl_dfasc_sweep_results.jsonl` (per user decision: "use best available DF+ASC config for all 17", never round-robin). Validation: every benchmark with a "records" count in the user's table reproduces that exact count.
- **P(j\|i,m) sampling**: per-occurrence (not deduplicated by answer text), `max(A_ij,0)/sum_k max(A_ik,0)`, seeded per (base_seed, method, benchmark, sample_id) for reproducibility.
- **Missing supervision**: `no_positive_supervision` (all advantages <=0) or `no_retained_supervision` (SIMIT keeps nothing) -- never an invented answer.
- **Blinding**: evaluator packets carry only {audit_id, image_path, question, answer} -- no method label, no reward/advantage, no sibling candidates. De-anonymization happens only after verdicts are collected, via `provenance/audit_manifest.jsonl`.

## Rerunning / extending

`01_select_queries.py --n-per-bench N --draw-tag <name>` APPENDS N new,
never-before-drawn doc_ids per benchmark to `cache/selected_queries.jsonl`
and updates `cache/used_ids_registry.json` accordingly (the registry is what
guarantees non-overlap with the existing 170, not the draw-tag, which is
just a label carried per-row for provenance). Every downstream script (02-07)
always operates on the FULL current contents of `cache/selected_queries.jsonl`,
so rerunning 02-06 after growing it will reprocess the original 170 queries
too (cheap: 02/05/06 are pure file reads, 03 rebuilds each run's records
from scratch, 04 needs a fresh GPU pass only for the newly added queries --
edit it to skip sample_ids already present in `cache/lfrl_srlm_selected.jsonl`
if you want to avoid regenerating the original 170). New packets will need
new agentic-eval passes for just the new audit_ids before rerunning 07.
