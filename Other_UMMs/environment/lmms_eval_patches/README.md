# lmms-eval task-config patches (not upstream)

13 of the 17 `_lite` task yamls this project uses (`lmms-eval==0.7.3`, pip-installed)
set `dataset_kwargs: {token: True}`, which forces `huggingface_hub` to require
a real local HF token to be logged in -- and raises `LocalTokenNotFoundError`
outright when none exists, rather than falling back to anonymous access.
The dataset repo (`lmms-lab-encoder/LMMs-Eval-Lite`) is public and loads fine
with no token at all (confirmed: every subset loads via plain
`datasets.load_dataset(...)` with no `token` kwarg), so this requirement is
unnecessary for our use and blocks evaluation with no HF_TOKEN configured.

Fix: removed the `dataset_kwargs: {token: True}` block entirely from each
affected yaml (falls back to anonymous access, which already works). The 4
unaffected tasks (`ok_vqa_val2014_lite`, `refcoco_bbox_val_lite`,
`textvqa_val_lite`, `vizwiz_vqa_val_lite`) never had this key and needed no
change.

Affected tasks: ai2d, chartqa, coco2017_cap_val, docvqa_val, flickr30k_test,
gqa, infovqa_val, mmbench_cn_dev, mmbench_en_dev, nocaps_val, seedbench,
textcaps_val, vqav2_val.

To reapply after a fresh venv rebuild:

```bash
cd environment/lmms_eval_patches
for f in *.yaml; do
  task_dir="${f%__*}"
  task_file="${f#*__}"
  cp "$f" "../../.venv/lib/python3.12/site-packages/lmms_eval/tasks/${task_dir}/${task_file}"
done
```

(filenames here are `<task_dir>__<task_file>.yaml`, e.g.
`ai2d__ai2d_lite.yaml` -> `lmms_eval/tasks/ai2d/ai2d_lite.yaml`.)

See also `environment/env.sh`'s `LMMS_EVAL_DATASETS_CACHE` export -- a
separate, related fix for lmms-eval's dataset-cache-directory resolution
silently redirecting away from this project's HF cache on `/work3` (a remote
filesystem).

`gqa__utils.py` is a separate, related fix of the same underlying problem in
a different place: `gqa/utils.py`'s `gqa_process_docs()` hardcodes
`token=True` directly in a `load_dataset("lmms-lab-encoder/GQA", ...)` call
(a *second*, auxiliary raw-image dataset GQA needs beyond the main
LMMs-Eval-Lite one) -- not in any yaml, so the yaml-level fix above doesn't
touch it. Same fix, same reasoning: that repo is public and loads fine with
no token. Reapply by copying this file back to
`lmms_eval/tasks/gqa/utils.py`.
