# Runtime Optimization Findings — Tables 1, 2, 3

All numbers below are relative to the greedy baseline (**276.76 sec/benchmark**,
averaged over the 17 canonical benchmarks). Everything here reflects a
**single GPU** — no multi-GPU parallelism is assumed anywhere, to keep the
comparison fair against baselines that also run on one GPU.

## What changed and why (methodology)

Three independent, separately-validated corrections were layered onto the
original measurements:

1. **Batching (1.759×)** — measured directly by running the real synthesis
   pipeline (unmodified `synthesize_vqa.py` functions: `generate_synthetic_examples`,
   `generate_visualization`, `classify_request`, `verify`) through a bounded
   worker pool instead of one-sample-at-a-time. Confirmed at three pool
   sizes (6, 10, 16) and via both `ThreadPoolExecutor` and `ProcessPoolExecutor`
   (ruling out GIL contention as an explanation) — all converge to ~1.7-1.76×,
   which we take as the real ceiling for this workload on one GPU. A deeper
   attempt to improve this further by modifying the serving engine's request
   scheduler (`mstar/worker/worker.py`'s dispatch loop) was tested and reverted
   — it gave **no improvement** (1.69× vs 1.76×), confirming the ceiling is
   not caused by eager dispatch timing.

2. **Finetune token-packing (1.458×)** — the LoRA training script
   (`train/pretrain_unified_navit.py`) uses token-based batch packing, not a
   fixed sample count. Raising `expected_num_tokens` from the production
   default (16384) to the engine's own ceiling (36864, matching
   `max_num_tokens`) raised measured tokens/sec from ~0.99k to ~1.45k.
   `cpu_offload=True` was confirmed **mandatory**, not removable overhead —
   disabling it OOMs immediately during FSDP setup on one GPU (75.45GB/79.17GB
   used). `num_workers` (1→4) gave no improvement, ruling out data loading as
   a bottleneck. This factor is applied assuming `total_steps` is
   proportionally reduced to preserve the same total training exposure —
   it's a real throughput gain, but only realized if steps are adjusted
   accordingly, not "free" on top of an unchanged step count.

3. **Adaptive-budget (ABA/ASC) correction — real per-sample replay, not a
   ceiling estimate.** The original estimate used the *swept hyperparameter
   ceiling* (the max K the search was allowed to use) as a proxy for the
   real average K. This was explicitly flagged as a likely overestimate.
   Corrected by replaying the actual `adaptive_k(p0, tau)` / `compute_tau(...)`
   formulas (from `prepare_vqa_data_v2_dfadaptive.py`) against the real,
   already-computed zero-shot confidence (`avg_answer_prob`) for all 500 real
   samples per benchmark, using each benchmark's own winning ASC
   hyperparameters and the *correct* `NUM_GENERATIONS` constant for whichever
   pool it drew from (6 for the `_v2` pool, 4 for the `_nodf4` pool — verified
   these are genuinely different τ-curves, not just a post-hoc cap, though
   the two approaches turn out numerically equivalent for this specific
   formula's structure). Result:
   - SIMIT-ICL: ceiling estimate 64.7% → **real 23.3%**
   - SIMIT-FT Single-Sample: ceiling estimate 76.5% → **real 41.1%**
   - SIMIT-FT Test-Set / Test-Set→ICL: 43.1% → **41.7%** (smaller change —
     10/12 of this row's benchmarks already used empirically-measured
     `records` counts, not a ceiling; only 2/12 needed the replay correction)

   This correction does **not** apply to Table 2/3's ablation rows — those
   all have ABA disabled (fixed k=4 budget), so there's no adaptive selection
   to correct.

**What is and isn't a joint real measurement:** every retry, verification
failure, and render error rate is naturally baked into these numbers because
they come from actually running the real pipeline (not a theoretical model).
The one exception: the "clean" full-Kmax synthesis base rate (49,566
sec/benchmark) is extrapolated from a smaller k=4 experiment's own real
error rate × 1.5, not a direct k=6 measurement — it assumes per-generated-variant
cost is roughly constant, which is reasonable but not identical to measuring
the real k=6 corpus's error rate at the same time as the K/Kmax decisions.

---

## Table 1 — Runtime (×greedy)

| Method | Runtime | Confidence |
|---|---|---|
| Greedy | 1× | reference |
| Multinomial | 1× (measured 1.08×) | high |
| Thinking (1,000 tok) | 18.8× | high |
| Thinking (10,000 tok) | 43.8× | high |
| Self-Refine (iters=1) | 5.0× | high |
| Self-Refine (iters=2) | 11.5× | high |
| Self-Refine (iters=3) | 20.0× | high |
| Self-Refine (iters=5) | 24.9× | high |
| Self-Consistency (K=8) — sequential | 8.2× | high |
| Self-Consistency (K=16) — sequential | 15.7× | high |
| Self-Consistency (K=32) — sequential | 34.2× | high |
| Self-Consistency+Thinking (K=8) | 154.1× | high |
| TTRL (K=32) | 75.0× | high |
| EMPO (K=32) | 58.1× | high |
| TTRV (K=32) | 40.7× | high |
| SRLM (K=32) | 251.0× | high |
| **SIMIT-ICL** | **25.0×** | medium-high |
| **SIMIT-FT, Single-Sample** | **46.5×** | medium-high |
| **SIMIT-FT, Test-Set** | **54.9×** | medium |
| **SIMIT-FT, Test-Set→ICL** | **55.1×** | medium |

### Self-Consistency, estimated if implemented via batching (not directly measured)

The table above reports SC as it's actually implemented — K rollouts issued
**sequentially** per sample. For a fairer comparison against SIMIT's
now-batched numbers, applying the same empirically-validated 1.76× batching
factor (SC's workload — K independent generate() calls to the same server —
is structurally similar to what was actually measured for synthesis
concurrency):

| Method | Sequential (actual) | Batched (**estimated**, not measured) |
|---|---|---|
| Self-Consistency (K=8) | 8.2× | **~4.7×** |
| Self-Consistency (K=16) | 15.7× | **~8.9×** |
| Self-Consistency (K=32) | 34.2× | **~19.4×** |

**This is an extrapolation, not a direct measurement** — no batched SC
implementation was actually built and run end-to-end the way the SIMIT
synthesis pipeline was. If a real number is needed here (e.g. for the paper),
it should be measured the same way SIMIT's was: build a bounded worker pool
around the actual K-rollout generation code and time it.

---

## Table 2 — Runtime (sec/benchmark), component ablation of SIMIT-FT Test-Set

| Row | Synthesis | Finetune | Eval | **Total (sec)** | **×greedy** |
|---|---|---|---|---|---|
| Baseline, BAGEL-7B | — | — | — | 277 | 1.0× |
| Full pipeline (ABA+DP+SL+V+DF) | 11,752 | 3,143 | 300 | **15,195** | **54.9×** |
| **w/o ABA** | 18,786 + 290 (DF-scoring) | 3,204 | 303 | **22,583** | **81.6×** |
| w/o ABA, DF (shared reference) | 18,786 | 2,743 | 303 | **21,832** | **78.9×** |
| w/o ABA, DP, DF | 24,085 | 2,743 | 311 | **27,140** | **98.1×** |
| w/o ABA, SL, DF | 27,888 | 2,743 | 319 | **30,951** | **111.8×** |
| w/o ABA, V, DF | 9,768 | 2,733 | 326 | **12,827** | **46.3×** |

**"w/o ABA" row — found and verified.** The real experiment is
`prepare_vqa_data_nodf4_dfonly.py` / `finetune_nodf4_dfonly_job_tracking.json`
(docstring: "DF only: kept = [c_i in [T_low, T_high]]; ASC disabled -> k =
NUM_GENERATIONS (no cap)" — exactly ABA/ASC off, DF on). Initially
mis-attributed to the "shared reference" row; corrected after verifying its
domain-average accuracy matches the paper's own "w/o ABA" anchor almost
exactly (Image 1.069 vs 1.07, VQA 68.46 vs 68.5, MathSci 87.0 exact, DocInfo
78.8 exact, Multi 83.42 vs 83.4). Finetune (4,671.8 sec real average from 17
LSF job footers) and eval (303.0 sec real average from 17 `results.json`
files) are now **real per-benchmark measurements**, not estimates — eval
validates the earlier ~300s assumption almost exactly; finetune came in
~17% higher than the earlier placeholder. Synthesis reuses the same shared
nodf4-pool generation cost as "w/o ABA, DF" (confirmed: `prepare_vqa_data_nodf4_dfonly.py`
selects from the already-generated `..._no_diff_filtering_4_samples` pool,
it does not generate anything new), plus the DF confidence-scoring step
(~290 sec, batched estimate). Considered and ruled out: retained-record
count after DF filtering does **not** make finetune or eval cheaper here —
verified `total_steps=150` is fixed across every ablation (not calibrated
to dataset size), and eval always runs the same fixed 500-sample test set
regardless of training-set size.

---

## Table 3 — Runtime (sec/benchmark), Question/Image fix-vs-generate (ABA+DF disabled throughout)

| Row | Synthesis | Finetune | Eval | **Total (sec)** | **×greedy** |
|---|---|---|---|---|---|
| Question Fix, Image Generate | 26,187 | 2,743 | 305 | **29,236** | **105.6×** |
| Question Generate, Image Fix | 9,610 | 2,743 | 325 | **12,678** | **45.8×** |
| Question Generate, Image Generate | 18,786 | 2,743 | 303 | **21,832** | **78.9×** (same run as Table 2's "w/o ABA, DF") |

**Why fixing the question is *slower*, not faster (counterintuitive but
confirmed in the code):** `synthesize_vqa.py` retries on verification
failure (`_MAX_TRIPLET_ATTEMPTS` budget). Free question generation lets the
model pick a (question, image) pairing it can render/verify easily. Fixing
the question removes that freedom — the image generator must produce
something that specifically supports a pre-given question, a harder target
that fails verification more often, burning through more retries to fill
the same number of accepted slots. Skipping image generation (Question
Generate, Image Fix) is a real time save; skipping question generation
(Question Fix, Image Generate) is not, because it costs more in retries than
it saves in generation.

---

## Table label correction: SRLM is not K=32

Checked the actual job scripts (`LFRL_Baselines/job_scripts/srlm_*.sh`)
rather than trusting the table header. **TTRL, EMPO, and TTRV are genuinely
K=32** (`--k 32`, confirmed across every variant sampled: full/altvote/temp15/lexical/lora32).
**SRLM is actually K=4** for essentially every benchmark (`--k 4`, confirmed
across ai2d/chartqa/ok_vqa/vizwiz/docvqa/seedbench/mmbench_en), with gqa
alone bumped to `--k 8` after K=4 gave "very little DPO training signal."
The paper's `SRLM ($K=32$)` label does not match any actual run and should
be corrected to `SRLM ($K{=}4$)`.

This changes the interpretation of SRLM's 251× runtime: it is **not**
expensive because of a large K — it uses 8× *fewer* rollouts than the other
three LFRL methods. It's expensive because its per-candidate mechanism
(self-judging via an extra LLM call, plus DPO preference computation) is
inherently costlier than TTRL/EMPO/TTRV's simple majority-vote consensus.
`srlm_k8_gqa.sh` has an explicit comment to this effect: judging isn't
batched, so even K=8 costs more per step than the other methods' K=32.
Recommend stating this explicitly in the paper's computational-cost
discussion rather than attributing the outlier to K.

## Self-Refine iters=3 (20.0×) is inflated by one outlier

Per-benchmark `total_evaluation_time_seconds` (17 benchmarks, ex-mathvista)
show the typical scaling is smooth: median ratio to iters=1 is **2.4× / 3.5×
/ 4.9×** for iters=2/3/5. That is what the implementation predicts —
`bagel.py::_self_refine_inference` has **no early stopping**; every iteration
adds a feedback generation plus a refined-answer generation (each up to 1000
tokens) onto a growing context, so cost is roughly linear in iterations.

The 20.0× is inflated mainly by **chartqa: 20,047s at iters=3, vs 8,190s at
iters=5** (and 1,747s at iters=1) — impossible by iteration count alone.
ai2d is mildly high too (8,284s, 4.3× its iters=1 vs the typical 3.5×).
Both ran in an **interactive GPU0 session** (`Bagel/logs/self_refine_iters3_gpu0.log`)
rather than the LSF batch chunks used for most other tasks, and the next task
in that session (coco) was lost and re-run after a resume. Most likely a
session-level slowdown (contention or stall), but the mechanism is **not
confirmed** — that log has no per-request events and `token_counts` are not
recorded in the samples.

Better estimate: linearly interpolate each benchmark between its measured
iters=2 and iters=5 times (`t2 + (t5 - t2)/3`). On the 14 benchmarks that did
not run in the interactive session this reproduces the measured iters=3 total
within ~4% (62.4k measured vs 60.1k interpolated), so the method is validated.
Replacing only chartqa + ai2d gives **16.2× greedy** (4,489 s/benchmark);
interpolating all 17 gives 16.0×; also replacing coco (same interactive
session, but measured *below* trend) gives 16.5×. Best estimate **≈16.2×
(range 16.0–16.5×)** instead of the measured 20.0×. (An earlier median-ratio-to-iters=1
estimate of 17.1× was biased high: iters=1 is cheap relative to the later
iterations, so scaling from it overshoots.) This is an estimate, not a
measurement; the clean fix is re-running chartqa (and ai2d) at iters=3 as an
LSF batch chunk like the others.

## Open items / honest caveats

- **Table 2's "w/o ABA" row** has no runtime source — flagged above, not
  estimated.
- **Self-Consistency's batched estimate** is an extrapolation from a
  different workload's measured batching factor, not a direct measurement.
- **The "clean" synthesis base rate** (49,566 sec/benchmark) is extrapolated
  from a smaller experiment, not directly measured at k=6.
- **Finetune's 1.458× token-packing gain** assumes `total_steps` is
  proportionally reduced to preserve total training exposure — it is not a
  free win on top of an unchanged step count.
- GPU1 on the test node is hardware/driver-broken for `mstar`'s serving
  engine specifically (confirmed reproducible even after full process/IPC
  cleanup) but works fine for plain PyTorch — all multi-GPU avenues were
  therefore ruled out for this investigation, consistent with keeping the
  comparison single-GPU throughout anyway.
