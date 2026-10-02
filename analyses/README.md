# Analyses

| Folder | Paper | What it does |
|---|---|---|
| [`stratified_gain/`](stratified_gain) | Fig. 6 | splits eval queries by whether Self-Consistency (K=32) answers them correctly and measures P(wrong→correct) / P(correct→wrong) per method |
| [`label_noise_model_noise/`](label_noise_model_noise) | Fig. 7, 13, Tab. 10 | samples one positively-encouraged pseudo-label per query for SIMIT and each LFRL baseline and audits it with an external, blind evaluator |
| [`data_quality/`](data_quality) | Tab. 2 (Data Quality) | the same audit on the synthetic training data of each ablation variant |

Each folder's README describes its pipeline. The external evaluator (Claude Sonnet 5, rubric in
`provenance/agentic_eval_rubric.md`) only audits the data and never feeds back into synthesis or
adaptation; its verdicts for the paper are kept in `provenance/eval_verdicts.jsonl` together with the
audited items (`provenance/audit_manifest.jsonl`), so the reported statistics can be recomputed with the
`*_aggregate.py` scripts without re-running the audit.

These scripts post-process the outputs of the experiments in [`../REPRODUCE.md`](../REPRODUCE.md). Their
input locations are set at the top of each `config.py` / script. `provenance/simit_df_asc_configs.json` and
`provenance/canonical_lfrl_runs.json` record the SIMIT configurations and LFRL runs analyzed for the paper;
to analyze new runs, take them from the validation-selected configurations in
`$SIMIT_DATA/hparams/<sweep>/<bench>.json` instead.
