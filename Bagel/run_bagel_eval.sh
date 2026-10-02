#!/bin/sh
# Run a single lmms-eval evaluation of a BAGEL checkpoint (bagel_umm).
# Usage: ./run_bagel_eval.sh <pretrained_path> <tasks_comma_separated> <output_dir> [mode] [extra_model_args] [extra_lmms_eval_args...]
#   mode: bagel_umm mode, defaults to "understanding" (e.g. "self_refine")
#   extra_model_args: extra comma-separated key=value pairs appended to --model_args (e.g. "self_refine_iters=2",
#                     "lora_ckpt_path=<checkpoint dir>"); pass "" to skip it and still give extra lmms-eval args
set -e
: "${SIMIT_ROOT:?run 'source simit_env.sh' first}"

PRETRAINED="$1"
TASKS="$2"
OUTPUT_DIR="$3"
MODE="${4:-understanding}"
EXTRA_MODEL_ARGS="$5"
shift $(( $# < 5 ? $# : 5 ))

if [ -z "$PRETRAINED" ] || [ -z "$TASKS" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <pretrained_path> <tasks_comma_separated> <output_dir> [mode] [extra_model_args] [extra_lmms_eval_args...]" >&2
  exit 1
fi

cd ${SIMIT_ROOT}/lmms-eval
export PYTHONPATH="${SIMIT_ROOT}/lmms-eval/pylibs:${SIMIT_ROOT}/Bagel:$PYTHONPATH"
mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="pretrained=${PRETRAINED},mode=${MODE}"
if [ -n "$EXTRA_MODEL_ARGS" ]; then
  MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

export API_TYPE=dummy
${SIMIT_ROOT}/Bagel/.venv/bin/python -m lmms_eval \
  --model bagel_umm \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUTPUT_DIR" \
  "$@"

