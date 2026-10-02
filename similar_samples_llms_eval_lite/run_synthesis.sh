#!/bin/bash
# Synthesize SIMIT practice samples for one or more benchmarks.
#
# Usage: ./run_synthesis.sh <pool> <split> [benchmark ...] [--synthesize_vqa.py options, e.g. --limit=5]
#   pool   v2               K=6 candidates per query (main pool, Kmax=6)
#          nodf4            K=4 candidates per query (Kmax=4; also the reference pool of the ablations)
#          no_diversity     K=4, without diversity-encouraging prompting   (Tab. 2, w/o DP)
#          force_natural    K=4, every image from BAGEL's native generation (Tab. 2, w/o SL)
#          keep_unverified  K=4, images kept even if the critic rejects them (Tab. 2, w/o V)
#          fixed_question   K=4, the test question is kept, new answers+images (Tab. 3, row 1)
#          fixed_image      K=4, the test image is kept, new questions+answers (Tab. 3, row 2)
#   split  lite   the 500 eval queries -> $SIMIT_DATA/<pool dir>/<bench>/sample{i}
#          val50  the 50 validation queries -> $SIMIT_DATA/<pool dir>/<bench>_val50/sample{i}
#   benchmark: subset names without the "_lite" suffix (default: all 17).
#
# Runs in the foreground on the current machine's (first visible) GPU; the script starts its
# own M* BAGEL server. Set SIMIT_LAUNCHER=bsub to submit one LSF job per benchmark instead
# (queue: $SIMIT_LSF_QUEUE, default gpuh100).
set -e
: "${SIMIT_ROOT:?run 'source simit_env.sh' first}"

POOL="$1"; SPLIT="$2"; shift 2 || true
BENCHES=(); EXTRA=()
for a in "$@"; do case "$a" in --*) EXTRA+=("$a") ;; *) BENCHES+=("$a") ;; esac; done
if [ ${#BENCHES[@]} -eq 0 ]; then
  BENCHES=(ai2d chartqa coco2017_cap_val docvqa_val flickr30k_test gqa infovqa_val mmbench_cn_dev
           mmbench_en_dev nocaps_val ok_vqa_val2014 refcoco_bbox_val seedbench textcaps_val
           textvqa_val vizwiz_vqa_val vqav2_val)
fi

# Paper defaults (App. G.1): direct-answer mode everywhere except the thinking-mode critic,
# critic SCORE >= 50 to pass, failed images are discarded.
VERIFY="--no-think --verify-all --discard-unverified --confidence-threshold=50 --verify-criterion=score --verify-think"
case "$POOL" in
  v2)              DIR=generated_samples_discard_unverified_v2;                              FLAGS="--k=6 $VERIFY" ;;
  nodf4)           DIR=generated_samples_discard_unverified_no_diff_filtering_4_samples;     FLAGS="--k=4 $VERIFY" ;;
  no_diversity)    DIR=generated_samples_no_diversity_prompt_no_diff_filtering_4_samples;    FLAGS="--k=4 --no-diversity-prompt $VERIFY" ;;
  force_natural)   DIR=generated_samples_force_natural_no_diff_filtering_4_samples;          FLAGS="--k=4 --force-natural $VERIFY" ;;
  # w/o V: same critic loop, but the last generated image is kept even if it never passes.
  keep_unverified) DIR=generated_samples_keep_unverified_no_diff_filtering_4_samples;        FLAGS="--k=4 --no-think --verify-all --confidence-threshold=50 --verify-criterion=score --verify-think" ;;
  fixed_question)  DIR=generated_samples_fixed_question_prompt_no_diff_filtering_4_samples;  FLAGS="--k=4 --fixed-question-prompt $VERIFY" ;;
  fixed_image)     DIR=generated_samples_fixed_image_prompt_no_diff_filtering_4_samples;     FLAGS="--k=4 --fixed-image-prompt $VERIFY" ;;
  *) echo "unknown pool '$POOL'" >&2; exit 1 ;;
esac
case "$SPLIT" in lite) LIMIT="--limit=500" ;; val50) LIMIT="" ;; *) echo "split must be lite or val50" >&2; exit 1 ;; esac

OUT_ROOT="$SIMIT_DATA/$DIR"
LOG_DIR="$SIMIT_DATA/logs/synthesis"
mkdir -p "$OUT_ROOT" "$LOG_DIR"
PY="$SIMIT_ROOT/.venv/bin/python"
CMD_BASE="$PY $SIMIT_ROOT/similar_samples_llms_eval_lite/synthesize_vqa.py --port=8000 --out-root=$OUT_ROOT --split=$SPLIT $LIMIT $FLAGS ${EXTRA[*]}"

for bench in "${BENCHES[@]}"; do
  if [ "${SIMIT_LAUNCHER:-local}" = "bsub" ]; then
    bsub <<EOF
#!/bin/sh
#BSUB -q ${SIMIT_LSF_QUEUE:-gpuh100}
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -J synth_${POOL}_${SPLIT}_${bench}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -o ${LOG_DIR}/${POOL}_${SPLIT}_${bench}_%J.out
source $SIMIT_ROOT/simit_env.sh
export PYTHONUNBUFFERED=1
$CMD_BASE $bench
EOF
  else
    echo "=== $POOL / $SPLIT / $bench ==="
    PYTHONUNBUFFERED=1 $CMD_BASE "$bench" 2>&1 | tee -a "$LOG_DIR/${POOL}_${SPLIT}_${bench}.log"
  fi
done
