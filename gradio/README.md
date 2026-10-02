---
title: SIMIT - models that imagine their own practice examples
emoji: 🥯
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
startup_duration_timeout: 1h
short_description: A VLM imagines practice examples for your question
models:
  - ByteDance-Seed/BAGEL-7B-MoT
  - bytedance-research/Lance
  - Qwen/Qwen3.8-27B-FP8
  - black-forest-labs/FLUX.2-klein-4B
---

# SIMIT demo

Upload an image, ask a question, and compare the model's plain answer with its
SIMIT-ICL answer: before answering again, the model imagines a few practice
examples (question, answer decided first, an image made to fit), checks them,
and keeps them in context.

Built on the [`simit`](https://github.com/monurcan/simit) library. See `tools/` for how the per-budget
hyperparameters were chosen and how the cached examples were produced.

## Hardware notes (ZeroGPU)

* Each request runs in a fresh `@spaces.GPU` process and uses the visitor's own
  ZeroGPU quota for the selected time budget (20-180 s).
* BAGEL-7B-MoT and Lance run on the default 48 GB size; Qwen3.8-27B-FP8 +
  FLUX.2-klein needs the 96 GB size (2x quota).
* All weights stay on CPU in the main process (~91 GB of RAM, ~92 GB of disk for
  the model files). If the Space's disk is too small, attach persistent storage
  and set `HF_HOME=/data/.huggingface`, or limit the models with
  `SIMIT_DEMO_MODELS=bagel,lance`.
* The first request after start-up compiles and autotunes GPU kernels (Qwen:
  ~25 s); later requests reuse the on-disk cache.

## How the time budgets map to settings

Each budget (20 / 40 / 60 / 90 / 180 s of GPU time per request) uses its own settings per model
(`presets.json`, built by `tools/build_presets.py`):

1. **Accuracy per K_max** (`tools/tune_presets.py`): one synthesis cache with 4 candidates per query on a
   mixed 64-question general-QA validation set: 8 each from VizWiz, VQAv2, OK-VQA, TextVQA, ChartQA,
   DocVQA, MMBench and AI2D (LMMs-Eval-Lite rows 300-307), each scored with its own metric. Then 800
   Optuna trials over the ABA/DF thresholds and K_max ∈ {1,2,3,4}. Per K_max, the pick is regularized:
   within one validation query of the best score, the widest difficulty-filter band.
2. **Latency** (`tools/latency.py`): per-request timelines on the demo's own code path in fresh forked
   processes (as on ZeroGPU, weight transfer included). Measured on an H100 and scaled by spaces'
   own factor for the ZeroGPU GPU sizes (1.5× on `large`, 1× on `xlarge`).
3. **Choice**: the tuned (best-quality) speed setting whenever some K fits the budget for ≥80% of
   requests; among those, the K within one validation query of the best score with the widest band.
   More budget also buys more attempts and verification rounds. If no K beats zero-shot on
   validation, SIMIT keeps the base answer by default; "Imagine even when confident" still runs it.

Validation scores (64 general questions):

| model | zero-shot | K=1 | K=2 | K=3 | K=4 | used by default |
|---|---|---|---|---|---|---|
| BAGEL-7B-MoT | 0.703 | 0.708 | **0.750** | 0.760 | 0.755 | K=2 from 60 s (20-40 s: zero-shot) |
| Lance | 0.432 | 0.442 | 0.442 | 0.458 | **0.468** | K=1 at 20 s, K=4 from 40 s |
| Qwen3.8-27B-FP8 + FLUX | **0.854** | 0.839 | 0.823 | 0.839 | 0.844 | zero-shot (no gain found) |

Lance's answers are sampled (greedy decoding degenerates on this checkpoint), so its outputs vary
between runs and SIMIT can also hurt a correct answer. Qwen3.8-27B is strong enough that imagined
examples did not improve it on these questions.

The cached examples are real runs of this app (BAGEL-7B-MoT and Lance) on queries from the paper's
figures, plus one product-page screenshot (`examples/bagel_amazon`). Several of them used "Imagine even when confident" (shown in the example), because the base
model was confidently wrong (e.g. "Unanswerable"). Qwen3.8-27B already answered those queries
correctly, so it has no cached example.
