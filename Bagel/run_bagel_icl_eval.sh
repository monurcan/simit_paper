#!/bin/sh
# SIMIT-ICL evaluation (lmms-eval model `bagel_icl`): each test query is answered
# with its own self-generated demonstrations in context. Demos come from a
# synthetic pool written by similar_samples_llms_eval_lite/run_synthesis.sh and
# are selected per query by difficulty filtering (DF, keep demos with confidence
# in [T_low, T_high]) and adaptive budget allocation (ABA, epsilon/A0/A1/B).
# For `<task>_val50` tasks the validation pool folder (<bench>_val50) is used.
# With a LoRA checkpoint this evaluates SIMIT-FT -> ICL (the finetuned model
# gets the same demonstrations).
#
# Usage: ./run_bagel_icl_eval.sh <pretrained_path> <tasks_comma_separated> <output_dir> \
#            <num_icl_examples> <df_enabled 0|1> <asc_enabled 0|1> \
#            <epsilon> <A0> <A1> <B> <T_low> <T_high> <pool v2|nodf4> [lora_ckpt_path] [icl_image_max_long_side] \
#            [extra lmms_eval args, e.g. --limit 5]
set -e
: "${SIMIT_ROOT:?run 'source simit_env.sh' first}"

PRETRAINED="$1"
TASKS="$2"
OUTPUT_DIR="$3"
NUM_ICL_EXAMPLES="${4:-4}"
DF_ENABLED="${5:-0}"
ASC_ENABLED="${6:-0}"
EPSILON="${7:-0.14}"
A0="${8:-0.35}"
A1="${9:-0.60}"
B="${10:-0.2}"
T_LOW="${11:-0.2}"
T_HIGH="${12:-0.9}"
POOL="${13:-v2}"
LORA_CKPT_PATH="${14}"
ICL_IMAGE_MAX_LONG_SIDE="${15}"

if [ -z "$PRETRAINED" ] || [ -z "$TASKS" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <pretrained_path> <tasks> <output_dir> <num_icl_examples> <df_enabled> <asc_enabled> <epsilon> <A0> <A1> <B> <T_low> <T_high> <pool v2|nodf4> [lora_ckpt_path] [icl_image_max_long_side]" >&2
  exit 1
fi

NPARAMS=15
if [ $# -lt $NPARAMS ]; then
  NPARAMS=$#
fi
shift $NPARAMS

cd ${SIMIT_ROOT}/lmms-eval
export PYTHONPATH="${SIMIT_ROOT}/lmms-eval/pylibs:${SIMIT_ROOT}/Bagel:$PYTHONPATH"
mkdir -p "$OUTPUT_DIR"

if [ "$POOL" = "nodf4" ]; then
  ICL_ROOT=${SIMIT_DATA}/generated_samples_discard_unverified_no_diff_filtering_4_samples
  NUM_GENERATIONS=4
else
  ICL_ROOT=${SIMIT_DATA}/generated_samples_discard_unverified_v2
  NUM_GENERATIONS=6
fi
DF_BOOL=$([ "$DF_ENABLED" = "1" ] && echo "True" || echo "False")
ASC_BOOL=$([ "$ASC_ENABLED" = "1" ] && echo "True" || echo "False")

MODEL_ARGS="pretrained=${PRETRAINED},mode=understanding,icl_samples_root=${ICL_ROOT},num_icl_examples=${NUM_ICL_EXAMPLES},num_generations=${NUM_GENERATIONS},df_enabled=${DF_BOOL},asc_enabled=${ASC_BOOL},df_asc_epsilon=${EPSILON},df_asc_A0=${A0},df_asc_A1=${A1},df_asc_B=${B},df_asc_T_low=${T_LOW},df_asc_T_high=${T_HIGH}"
if [ -n "$LORA_CKPT_PATH" ]; then
  MODEL_ARGS="${MODEL_ARGS},lora_ckpt_path=${LORA_CKPT_PATH}"
fi
if [ -n "$ICL_IMAGE_MAX_LONG_SIDE" ]; then
  MODEL_ARGS="${MODEL_ARGS},icl_image_max_long_side=${ICL_IMAGE_MAX_LONG_SIDE}"
fi

export API_TYPE=dummy

${SIMIT_ROOT}/Bagel/.venv/bin/python -m lmms_eval \
  --model bagel_icl \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUTPUT_DIR" \
  "$@"

