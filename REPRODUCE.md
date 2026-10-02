# Reproducing the paper

Commands assume `source simit_env.sh` and the environments from the [README](README.md#-installation).
Large outputs go to `$SIMIT_DATA` (default `./workdir`). The sweeps and grid runners submit LSF jobs
(`SIMIT_LSF_SLOTS="gpuh100,p1,..."`); set `SIMIT_LAUNCHER=local` to run everything on the current GPU.
`run_synthesis.sh`, `run_baselines.sh` and `Other_UMMs/run_simit_icl.sh` run locally by default and use
`SIMIT_LAUNCHER=bsub` to submit instead. All runtimes in the paper are for one H100.

**Hyperparameter protocol.** Everything that is tuned is tuned on the 50-sample validation split
(`<task>_val50`); every reported number comes from one evaluation on the 500-sample eval split
(`<task>`) with the selected configuration. Sweeps store their Optuna studies in
`$SIMIT_DATA/sweeps/<name>/` and, after `--finalize`, the selected configurations in
`$SIMIT_DATA/hparams/<name>/<bench>.json`, which downstream experiments read.

## 0. Data
```bash
Bagel/.venv/bin/python rag_retrieval/build_retrieval_sets.py --all      # 50 validation + retrieval pool per benchmark
Bagel/.venv/bin/python rag_retrieval/build_val50_datasets.py --all      # $SIMIT_DATA/val50/<task> for the *_val50 tasks
.venv/bin/python similar_samples_llms_eval_lite/fetch_gqa_images.py      # GQA Lite images
```

## 1. Synthetic data (Sec. 3.2, App. G.1)
For each pool, synthesize for the eval queries (`lite`) and the validation queries (`val50`), then score
the confidences used by adaptive budget allocation (ABA, zero-shot confidence p0) and difficulty
filtering (DF, teacher-forced confidence of each intended answer):
```bash
for pool in v2 nodf4; do                         # Kmax = 6 / 4 candidates per query
  for split in lite val50; do similar_samples_llms_eval_lite/run_synthesis.sh $pool $split; done
done
Bagel/.venv/bin/python similar_samples_llms_eval_lite/compute_confidence.py --mode full \
    --datasets $SIMIT_DATA/generated_samples_discard_unverified_v2 \
               $SIMIT_DATA/generated_samples_discard_unverified_no_diff_filtering_4_samples
```
The ablation pools of Tab. 2/3 are made the same way (`lite` split, no confidence needed except
`nodf4`): `run_synthesis.sh {no_diversity,force_natural,keep_unverified,fixed_question,fixed_image} lite`.

## 2. Tab. 1 / Tab. 8: SIMIT and test-time scaling baselines
| Row | Command |
|---|---|
| BAGEL-7B zero-shot | `Bagel/run_baselines.sh zero_shot` |
| Thinking (1,000 / 10,000 tokens) | `Bagel/run_baselines.sh think1k`, `think10k` |
| Self-Refine (1 / 3 / 5 iters) | `Bagel/run_baselines.sh self_refine1`, `self_refine3`, `self_refine5` |
| Self-Consistency (K = 8 / 16 / 32) | `Bagel/run_baselines.sh sc8`, `sc16`, `sc32` |
| SIMIT-ICL | `cd Bagel && python run_icl_sweep.py && python run_icl_sweep.py --finalize` |
| SIMIT-FT, Single-Query | `cd Bagel && python run_tta_sweep.py && python run_tta_sweep.py --finalize` |
| SIMIT-FT, Test-Set | `cd Bagel && python run_ft_sweep.py && python run_ft_sweep.py --finalize` |
| SIMIT-FT, Test-Set → ICL | `cd Bagel && python run_ft_icl.py` (after the ICL and Test-Set finalizes) |
| TTRL / EMPO / TTRV / SRLM | `cd LFRL_Baselines && python sweep_lfrl.py --method {ttrl,empo,ttrv,srlm}`, then the same with `--finalize` |

Use `Bagel/.venv/bin/python` for the Python scripts. Results of zero-shot and TTS baselines are in
`$SIMIT_DATA/baselines/<method>/<bench>/`; the finalize steps print per-benchmark scores, domain means and
the mean relative gain ("Av. Gain").

## 3. Tab. 2 / Tab. 3: ablations (SIMIT-FT, Test-Set)
Each ablation reuses the validation-selected LoRA settings of SIMIT-FT (Test-Set), 150 steps:
```bash
cd Bagel
for v in no_aba no_aba_df no_dp no_sl no_v fixed_question fixed_image; do .venv/bin/python run_ablations.py --variant $v; done
```
`no_aba_df` is also the "generate both" row of Tab. 3. Data-quality scores: `analyses/data_quality/` (needs the
external Claude evaluator, see its README).

## 4. Sec. 4.2, Fig. 4, 9, 10 / App. I: synthetic vs. real data
Build the real-data references (needs `rag_retrieval/.venv`):
```bash
rag_retrieval/.venv/bin/python rag_retrieval/build_faiss_index.py --all                   # Qwen3-VL-Embedding-2B index
rag_retrieval/.venv/bin/python rag_retrieval/build_capped_index.py --all                 # 2000-sample pools
Bagel/.venv/bin/python Bagel/prepare_vqa_data_real_rag.py && Bagel/.venv/bin/python Bagel/register_real_rag_datasets.py
```
Fixed-K comparisons (ABA and DF disabled):
```bash
cd Bagel
.venv/bin/python run_scaling_k.py --setting icl --pool nodf4     # SIMIT-ICL, K = 1..4
.venv/bin/python run_scaling_k.py --setting rices                # RICES
.venv/bin/python run_scaling_k.py --setting ft_sq --pool nodf4   # SIMIT-FT (Single-Query)
.venv/bin/python run_scaling_k.py --setting ttt_nn               # TTT-NN
.venv/bin/python run_ablations.py --variant ntrain --pool nodf4  # SIMIT-FT (Test-Set), N_train = 500..2000
.venv/bin/python run_ablations.py --variant real_sft             # SFT on the real pool
```

## 5. App. J (Fig. 11 / 12): cross-dataset generalization
`cd Bagel && .venv/bin/python run_cross_dataset.py` (uses the SIMIT-FT Test-Set checkpoints).

## 6. App. K (Tab. 9): other unified models
```bash
Other_UMMs/run_simit_icl.sh sensenova all     # SenseNova-U1-8B
Other_UMMs/run_simit_icl.sh lance all         # Lance-3B
```

## 7. Analyses and figures
| Paper | Code |
|---|---|
| Fig. 6 (stratified gains) | `analyses/stratified_gain/stratified_gain.py`, plotted by `visualization_scripts/stratified_gain_tornado.py` |
| Fig. 7, 13, Tab. 10 (label-noise / model-noise audit) | `analyses/label_noise_model_noise/` (see README), plotted by `visualization_scripts/label_noise_*.py` |
| Tab. 2 data quality | `analyses/data_quality/` |
| Fig. 1 right | `visualization_scripts/fig1_right_runtime_vs_gain.py` |
| Fig. 9 / 10 | `visualization_scripts/scaling_k_subplots.py`, `real_vs_synthetic_scatter.py` |
| Fig. 11 / 12 | `visualization_scripts/cross_dataset_generalization{,_compact}.py` |
| Fig. 3, 5, 14 (skill library samples / ablation) | `examples/generate_skill_ablation.py` (needs a running `mstar serve bagel`) |
| Speed-ups in App. G.1 | `speed_experiments/` (see `RUNTIME_OPTIMIZATION_FINDINGS.md`) |
| Confidence intervals | `eval_ci_tools/eval_ci.py` |

The analysis scripts read the outputs of the runs above; their input locations are set at the top of each
`config.py` / script. The plotting scripts contain the paper's numbers inline and write to
`visualization_scripts/figures/`.
