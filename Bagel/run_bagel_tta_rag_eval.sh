#!/bin/sh
# Run a single lmms-eval evaluation of BAGEL's single-sample test-time-adaptation
# model (lmms_eval/models/simple/bagel_tta.py), but with real-data retrieval
# (bagel_icl.py's RealSampleRetriever / FAISS + Qwen3-VL-Embedding neighbors
# from rag_retrieval/) instead of run_bagel_tta_eval.sh's Bagel-synthesized
# samples: per doc, LoRA-finetune on the k nearest real neighbors from the
# benchmark's capped-2000 retrieval pool, answer with the plain baseline
# prompt, reset before the next doc. Mirrors run_bagel_tta_eval.sh's /
# run_bagel_icl_rag_eval.sh's interface. No DF/ASC concept here -- that's
# specific to SyntheticSampleRetriever's confidence-adaptive-K selection.
#
# Usage: ./run_bagel_tta_rag_eval.sh <pretrained_path> <tasks_comma_separated> <output_dir> \
#            <num_adapt_examples> <lora_rank> <lora_alpha> <lora_dropout> <adapt_steps> <adapt_lr> \
#            [rag_embed_gpu] [extra lmms_eval args, e.g. --limit 100]
#   rag_embed_gpu: CUDA device index (relative to this job's own visible
#     devices, not physical IDs) the Qwen3-VL-Embedding subprocess is pinned
#     to. Default 0 -- these are 1-GPU jobs (Bagel-7B + Qwen3-VL-Embedding-2B
#     fit together on one H100, same as run_bagel_icl_rag_eval.sh's note).
set -e

PRETRAINED="$1"
TASKS="$2"
OUTPUT_DIR="$3"
NUM_ADAPT_EXAMPLES="${4:-4}"
LORA_RANK="${5:-8}"
LORA_ALPHA="${6:-16}"
LORA_DROPOUT="${7:-0.0}"
ADAPT_STEPS="${8:-20}"
ADAPT_LR="${9:-2e-4}"
RAG_EMBED_GPU="${10:-0}"

if [ -z "$PRETRAINED" ] || [ -z "$TASKS" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <pretrained_path> <tasks> <output_dir> <num_adapt_examples> <lora_rank> <lora_alpha> <lora_dropout> <adapt_steps> <adapt_lr> [rag_embed_gpu] [extra lmms_eval args...]" >&2
  exit 1
fi

NPARAMS=10
if [ $# -lt $NPARAMS ]; then
  NPARAMS=$#
fi
shift $NPARAMS

cd ${SIMIT_ROOT}/lmms-eval
export PYTHONPATH="${SIMIT_ROOT}/lmms-eval/pylibs:${SIMIT_ROOT}/Bagel:$PYTHONPATH"
mkdir -p "$OUTPUT_DIR"

RAG_INDEX_ROOT="${RAG_INDEX_ROOT_OVERRIDE:-${SIMIT_ROOT}/rag_retrieval/data_capped2000}"
RAG_EMBED_PYTHON=${SIMIT_ROOT}/rag_retrieval/.venv/bin/python

MODEL_ARGS="pretrained=${PRETRAINED},mode=understanding,retrieval_strategy=rag,rag_index_root=${RAG_INDEX_ROOT},rag_embed_python=${RAG_EMBED_PYTHON},rag_embed_gpu=${RAG_EMBED_GPU},num_icl_examples=${NUM_ADAPT_EXAMPLES},lora_rank=${LORA_RANK},lora_alpha=${LORA_ALPHA},lora_dropout=${LORA_DROPOUT},adapt_steps=${ADAPT_STEPS},adapt_lr=${ADAPT_LR}"

# Same rationale as run_bagel_tta_eval.sh: per-doc training forward+backward
# activation footprints vary a lot with image/token count, so the CUDA
# caching allocator needs expandable_segments to avoid fragmentation OOMs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# No real OpenAI judge in this environment; dummy fallback verified safe for
# rule-based MCQ matching (mathvista excluded from this project's TTA sweeps
# entirely, so not handled here).
export API_TYPE=dummy

# Without this, stdout is fully block-buffered when redirected to a file --
# see run_bagel_tta_eval.sh's note for the production incident this fixes.
export PYTHONUNBUFFERED=1

${SIMIT_ROOT}/Bagel/.venv/bin/python -m lmms_eval \
  --model bagel_tta \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUTPUT_DIR" \
  "$@"
