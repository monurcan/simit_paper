# vllm-omni patches (not upstream)

`pipeline_sensenova_u1.py.patched` is a full copy of vllm-omni's
`vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py` (installed
at `vllm-omni==0.24.0`) with one addition: a `_score_text` method plus two new
`extra_args` (`score_answer`, `score_greedy`) that expose teacher-forced /
greedy per-token log-probs for confidence computation, since the public
`Omni.generate()` API never exposes logits and `_forward_text`'s own AR loop
discards them after decoding. See the method's docstring in the patched file
for details; callers are `common/model_clients/sensenova_client.py`'s
`score_teacher_forced`/`score_greedy`.

This lives outside upstream vllm-omni (a pip-installed dependency), so a
fresh venv rebuild needs it reapplied:

```bash
cp environment/vllm_omni_patches/pipeline_sensenova_u1.py.patched \
   .venv/lib/python3.12/site-packages/vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py
```

If `vllm-omni` is ever upgraded past 0.24.0, re-diff against the new
`pipeline_sensenova_u1.py` and reapply the same three edits by hand (search
for "SIMIT/Other_UMMs patch") rather than blindly overwriting -- the upstream
file may have changed elsewhere.

## pipeline_lance.py.patched

Same idea, for `vllm_omni/diffusion/models/lance/pipeline_lance.py` (Lance-3B,
`vllm-omni==0.24.1`), backing `common/model_clients/lance_client.py`. Adds,
after `_x2t_prepare_assistant_start`:

  - `_forward_x2t_image_multi` / `_x2t_multi_setup` / `_x2t_prefill_multi` /
    `_split_x2t_prompt_multi`: generalizes upstream's single-image
    `_forward_x2t_image` (one vision block) to N interleaved images, for
    SIMIT-ICL's K-demonstration-image prompts (`LanceClient.chat_multi`).
    Dispatched from `forward()` when `multi_modal_data.image` is a *list*;
    plain single-image requests still go through upstream's own
    `_forward_x2t_image` unchanged.
  - `_score_text_x2t` / `_score_ar_step`: teacher-forced/greedy per-token
    log-probs, same purpose and output shape as
    `pipeline_sensenova_u1.py.patched`'s `_score_text` (ported onto Lance's
    segmented-prefill + `self.bagel.language_model` step call instead of
    SenseNova's single combined-embeds forward). Dispatched from `forward()`
    when `extra_args["score_answer"]` or `extra_args["score_greedy"]` is set.

Reapply the same way after any `vllm-omni` upgrade:

```bash
cp environment/vllm_omni_patches/pipeline_lance.py.patched \
   .venv/lib/python3.12/site-packages/vllm_omni/diffusion/models/lance/pipeline_lance.py
```

## bagel_transformer.py.patched

`vllm_omni/diffusion/models/bagel/bagel_transformer.py` -- one-line fix, not
Lance-only code but only exercised by Lance in this project (SenseNova has
its own separate `pipeline_sensenova_u1.py`/no shared BAGEL code, so this
never touches SenseNova behavior).

`Bagel.forward`'s CFG-branch image-generation path built
`batched_position_ids = torch.cat(cfg_branch_pids)` (default `dim=0`) to
concatenate each CFG branch's (cond / cfg_text / cfg_img) own
`packed_position_ids` sequence into one batched forward call. That's correct
for plain BAGEL's 1-D `(S,)` position ids (dim 0 IS the sequence axis there),
but Lance's checkpoint uses mRoPE, where `LanceBagel.prepare_vae_latent_cfg`/
`prepare_prompts` (`lance_transformer.py`) build 2-D `(3, S)` position ids
(3 stacked t/h/w rope axes) -- concatenating those along dim 0 mixes the
3-axis dimension across branches into e.g. 6, instead of extending the
sequence axis. Confirmed via a plain `generate_image()`/t2i smoke test:
crashed inside `BagelRotaryEmbedding.forward`'s mRoPE branch with "size of
tensor a (3) must match the size of tensor b (6) at non-singleton dimension
1" (2 CFG branches x 3 rope axes = 6, forced against `inv_freq_expanded`'s
hardcoded 3). Fix: `torch.cat(cfg_branch_pids, dim=-1)` -- the sequence axis
in both conventions (the only axis for 1-D, axis 1 for 2-D), so a no-op for
plain 1-D position ids and correct for Lance's 2-D mRoPE ones. Confirmed
fixed: t2i now succeeds at 1024x1024 (Lance's own default) and 400x400 (this
project's GEN_WIDTH/GEN_HEIGHT).

```bash
cp environment/vllm_omni_patches/bagel_transformer.py.patched \
   .venv/lib/python3.12/site-packages/vllm_omni/diffusion/models/bagel/bagel_transformer.py
```
