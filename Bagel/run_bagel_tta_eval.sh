#!/bin/sh
# Run a single lmms-eval evaluation of BAGEL's single-sample test-time-adaptation
# model (lmms_eval/models/simple/bagel_tta.py): per doc, LoRA-finetune on
# DF+ASC-selected retrieved/synthesized examples, answer with the plain
# baseline prompt, reset before the next doc. Same interface as
# run_bagel_icl_eval.sh, extended with the LoRA/adaptation hyperparameters.
#
# Usage: ./run_bagel_tta_eval.sh <pretrained_path> <tasks_comma_separated> <output_dir> \
#            <num_adapt_examples> <pool v2|nodf4> <df_enabled 0|1> <asc_enabled 0|1> \
#            <epsilon> <A0> <A1> <B> <T_low> <T_high> \
#            <lora_rank> <lora_alpha> <lora_dropout> <adapt_steps> <adapt_lr> \
#            [extra lmms_eval args, e.g. --limit 100]
set -e
: "${SIMIT_ROOT:?run 'source simit_env.sh' first}"

PRETRAINED="$1"
TASKS="$2"
OUTPUT_DIR="$3"
NUM_ADAPT_EXAMPLES="${4:-4}"
POOL="${5:-v2}"
DF_ENABLED="${6:-0}"
ASC_ENABLED="${7:-0}"
EPSILON="${8:-0.14}"
A0="${9:-0.35}"
A1="${10:-0.60}"
B="${11:-0.2}"
T_LOW="${12:-0.2}"
T_HIGH="${13:-0.9}"
LORA_RANK="${14:-8}"
LORA_ALPHA="${15:-16}"
LORA_DROPOUT="${16:-0.0}"
ADAPT_STEPS="${17:-20}"
ADAPT_LR="${18:-2e-4}"

if [ -z "$PRETRAINED" ] || [ -z "$TASKS" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <pretrained_path> <tasks> <output_dir> <num_adapt_examples> <pool v2|nodf4> <df_enabled> <asc_enabled> <epsilon> <A0> <A1> <B> <T_low> <T_high> <lora_rank> <lora_alpha> <lora_dropout> <adapt_steps> <adapt_lr> [extra lmms_eval args...]" >&2
  exit 1
fi

NPARAMS=18
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

MODEL_ARGS="pretrained=${PRETRAINED},mode=understanding,icl_samples_root=${ICL_ROOT},num_icl_examples=${NUM_ADAPT_EXAMPLES},num_generations=${NUM_GENERATIONS},df_enabled=${DF_BOOL},asc_enabled=${ASC_BOOL},df_asc_epsilon=${EPSILON},df_asc_A0=${A0},df_asc_A1=${A1},df_asc_B=${B},df_asc_T_low=${T_LOW},df_asc_T_high=${T_HIGH},lora_rank=${LORA_RANK},lora_alpha=${LORA_ALPHA},lora_dropout=${LORA_DROPOUT},adapt_steps=${ADAPT_STEPS},adapt_lr=${ADAPT_LR}"

# Per-doc training forward+backward activation footprints vary a lot with
# image/token count (unlike pure inference, autograd must retain every
# intermediate activation until backward()), so the CUDA caching allocator
# sees widely different allocation sizes back-to-back across docs. Confirmed
# in production: a chartqa sweep trial OOM'd on one unlucky doc with 78.9/79.2
# GiB "in use" yet only 790MiB "reserved but unallocated" -- peak memory
# itself was fine most of the time (observed oscillating 30-78GiB doc-to-doc,
# not a monotonic leak), it just couldn't satisfy one more small allocation
# out of a fragmented pool. This is exactly the scenario PyTorch's own
# expandable_segments allocator mode exists for (recommended directly in the
# OOM error message); zero behavior change otherwise.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# No OpenAI judge: MMBench answers are matched by lmms-eval's rule-based extractor.
export API_TYPE=dummy

# Without this, stdout is fully block-buffered when redirected to a file (not
# a tty) -- progress can sit unflushed for a long time, and a killed process
# never flushes it at all, making a .out file's content unreliable evidence
# of whether a job was actually stuck or just hadn't flushed yet (confirmed:
# a trial believed stuck for 25+ minutes with zero visible progress lines
# turned out to have hundreds once its buffer finally flushed). Unbuffered
# output costs a little I/O overhead but makes future diagnosis trustworthy.
export PYTHONUNBUFFERED=1

${SIMIT_ROOT}/Bagel/.venv/bin/python -m lmms_eval \
  --model bagel_tta \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUTPUT_DIR" \
  "$@"
