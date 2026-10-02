#!/bin/bash
# Apply this project's patches to the pip-installed lmms-eval and vllm-omni in
# Other_UMMs/.venv (see lmms_eval_patches/README.md and vllm_omni_patches/README.md),
# and register the 17 `<task>_val50` validation tasks there.
set -e
: "${SIMIT_ROOT:?source simit_env.sh first}"
SITE="$SIMIT_ROOT/Other_UMMs/.venv/lib/python3.12/site-packages"
HERE="$(cd "$(dirname "$0")" && pwd)"

for f in "$HERE"/lmms_eval_patches/*__*.yaml; do
  name="$(basename "$f")"; cp "$f" "$SITE/lmms_eval/tasks/${name%%__*}/${name#*__}"
done
cp "$HERE/lmms_eval_patches/gqa__utils.py" "$SITE/lmms_eval/tasks/gqa/utils.py"
cp "$HERE/vllm_omni_patches/pipeline_sensenova_u1.py.patched" "$SITE/vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py"
cp "$HERE/vllm_omni_patches/pipeline_lance.py.patched" "$SITE/vllm_omni/diffusion/models/lance/pipeline_lance.py"
cp "$HERE/vllm_omni_patches/bagel_transformer.py.patched" "$SITE/vllm_omni/diffusion/models/bagel/bagel_transformer.py"

# Validation tasks: same yaml as the Lite task, data from rag_retrieval/build_val50_datasets.py.
for y in "$SIMIT_ROOT"/lmms-eval/lmms_eval/tasks/*/*_lite_val50.yaml; do
  dir="$(basename "$(dirname "$y")")"; task="$(basename "$y" _val50.yaml)"
  extra=$(grep -q "^process_docs: null" "$y" && echo "process_docs: null" || true)
  cat > "$SITE/lmms_eval/tasks/$dir/${task}_val50.yaml" <<YAML
include: ${task}.yaml
task: ${task}_val50
dataset_path: $SIMIT_DATA/val50/${task}
dataset_name: null
dataset_kwargs:
  load_from_disk: True
test_split: val
$extra
YAML
done
echo "patches applied to $SITE"
