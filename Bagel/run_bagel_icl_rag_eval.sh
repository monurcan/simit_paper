#!/bin/sh
# Run a single lmms-eval evaluation of a BAGEL checkpoint with the bagel_icl
# model's real-sample RAG retrieval strategy (FAISS + Qwen3-VL-Embedding
# neighbors from rag_retrieval/, instead of run_bagel_icl_eval.sh's
# Bagel-synthesized samples). Mirrors run_bagel_icl_eval.sh's interface.
# Usage: ./run_bagel_icl_rag_eval.sh <pretrained_path> <tasks_comma_separated> <output_dir> [num_icl_examples] [icl_image_max_long_side] [rag_embed_gpu] [extra_lmms_eval_args...]
#   icl_image_max_long_side: caps the long edge (px) of ICL demo images only
#     (never the query image), downscale-only, aspect-ratio-preserving.
#     Empty/unset = unlimited (no resize, prior behavior).
#   rag_embed_gpu: CUDA device index (relative to this job's own visible
#     devices, not physical IDs) the Qwen3-VL-Embedding subprocess is pinned
#     to, independent of whatever GPU the Bagel model itself uses. Default 1
#     (second GPU) -- pass 0 explicitly if this job only has one GPU visible
#     (Bagel-7B + Qwen3-VL-Embedding-2B fit together on one H100).
#
# mathvista_testmini_format automatically gets API_TYPE=dummy, matching the
# methodology used for all other bagel_umm mathvista runs in this project
# (no real OpenAI judge available; see self_refine_chunk.sh).
set -e

PRETRAINED="$1"
TASKS="$2"
OUTPUT_DIR="$3"
NUM_ICL_EXAMPLES="${4:-4}"
ICL_IMAGE_MAX_LONG_SIDE="$5"
RAG_EMBED_GPU="${6:-1}"
shift $(( $# < 6 ? $# : 6 ))

if [ -z "$PRETRAINED" ] || [ -z "$TASKS" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <pretrained_path> <tasks_comma_separated> <output_dir> [num_icl_examples] [icl_image_max_long_side] [rag_embed_gpu] [extra_lmms_eval_args...]" >&2
  exit 1
fi

cd ${SIMIT_ROOT}/lmms-eval
export PYTHONPATH="${SIMIT_ROOT}/lmms-eval/pylibs:${SIMIT_ROOT}/Bagel:$PYTHONPATH"
mkdir -p "$OUTPUT_DIR"

RAG_INDEX_ROOT="${RAG_INDEX_ROOT_OVERRIDE:-${SIMIT_ROOT}/rag_retrieval/data_capped2000}"
RAG_EMBED_PYTHON=${SIMIT_ROOT}/rag_retrieval/.venv/bin/python
MODEL_ARGS="pretrained=${PRETRAINED},mode=understanding,retrieval_strategy=rag,rag_index_root=${RAG_INDEX_ROOT},rag_embed_python=${RAG_EMBED_PYTHON},rag_embed_gpu=${RAG_EMBED_GPU},num_icl_examples=${NUM_ICL_EXAMPLES}"
if [ -n "$ICL_IMAGE_MAX_LONG_SIDE" ]; then
  MODEL_ARGS="${MODEL_ARGS},icl_image_max_long_side=${ICL_IMAGE_MAX_LONG_SIDE}"
fi

case ",$TASKS," in
  *,mathvista_testmini_format,*) export API_TYPE=dummy ;;
esac

${SIMIT_ROOT}/Bagel/.venv/bin/python -m lmms_eval \
  --model bagel_icl \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUTPUT_DIR" \
  "$@"

# mathvista_testmini_format(-family) scoring is broken under API_TYPE=dummy (no
# real judge in this environment): letter-only multi-choice answers never hit
# the exact-match shortcut and fall through to a Levenshtein-vs-"dummy" tiebreak
# unrelated to correctness. Auto-correct via local re-extraction immediately
# after every run so no mathvista result is ever read uncorrected -- see
# scripts_eval_fix/rescore_mathvista.py for the full writeup.
case ",$TASKS," in
  *,mathvista_testmini_format,*|*,mathvista_testmini_format_sc,*|*,mathvista_testmini_format_sc16,*|*,mathvista_testmini_format_sc32,*|*,mathvista_testmini_format_think,*|*,mathvista_testmini_format_think_sc,*)
    echo "[$(date)] mathvista task detected -- auto-correcting the dummy-judge scoring bug"
    find "$OUTPUT_DIR" -name "*_samples_mathvista_testmini_format*.jsonl" ! -name "*__new_corrected*" -print0 \
      | xargs -0 -r env API_TYPE=dummy ${SIMIT_ROOT}/Bagel/.venv/bin/python ${SIMIT_ROOT}/Bagel/scripts_eval_fix/write_corrected_mathvista_outputs.py \
      || echo "[$(date)] WARNING: mathvista auto-correction failed -- run scripts_eval_fix/write_corrected_mathvista_outputs.py manually"
    ;;
esac
