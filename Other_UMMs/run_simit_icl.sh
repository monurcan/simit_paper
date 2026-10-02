#!/bin/bash
# SIMIT-ICL with SenseNova-U1-8B or Lance-3B (paper App. K, Tab. 9), per benchmark:
#   synth  synthesize K=5 practice samples for the 500 eval queries and the 50 validation queries
#   eval   confidences, zero-shot baseline, Optuna DF/ABA search on <bench>_lite_val50,
#          and one eval-split run of the best validation configuration (eval/run_full_eval_pipeline.py)
# Usage: ./run_simit_icl.sh <sensenova|lance> <synth|eval|all> [benchmark ...]   (default: all 17)
# Runs in the foreground; SIMIT_LAUNCHER=bsub submits one LSF job per benchmark instead
# (synthesis of a large benchmark can take more than 24h: use SIMIT_LSF_QUEUE=p1, it allows 72h).
set -e
: "${SIMIT_ROOT:?source simit_env.sh first}"
MODEL="$1"; STAGE="$2"; shift 2 || true
BENCHES=("$@")
if [ ${#BENCHES[@]} -eq 0 ]; then
  BENCHES=(ai2d chartqa coco2017_cap_val docvqa_val flickr30k_test gqa infovqa_val mmbench_cn_dev
           mmbench_en_dev nocaps_val ok_vqa_val2014 refcoco_bbox_val seedbench textcaps_val
           textvqa_val vizwiz_vqa_val vqav2_val)
fi
case "$MODEL" in
  sensenova) SYNTH_FLAGS="--k=5 --verify-criterion=score --confidence-threshold=50" ;;
  # Lance cannot follow the long triplet prompt; synthesize_vqa.py uses a decomposed prompt,
  # native image generation and no critic for it (see common/model_clients/lance_client.py).
  lance)     SYNTH_FLAGS="--model lance --k=5" ;;
  *) echo "model must be sensenova or lance" >&2; exit 1 ;;
esac
HERE="$SIMIT_ROOT/Other_UMMs"
ROOT="$SIMIT_DATA/other_umms/$MODEL"
SAMPLES="$ROOT/generated_samples"
mkdir -p "$SAMPLES" "$ROOT/eval_results" "$ROOT/logs"
PY="$HERE/.venv/bin/python"
WALL=$([ "${SIMIT_LSF_QUEUE:-p1}" = "p1" ] && echo 72:00 || echo 24:00)

for bench in "${BENCHES[@]}"; do
  CMD="source $HERE/environment/env.sh"
  if [ "$STAGE" = "synth" ] || [ "$STAGE" = "all" ]; then
    CMD="$CMD && $PY $HERE/synthesis/synthesize_vqa.py $bench --out-root=$SAMPLES --split=lite --limit=500 $SYNTH_FLAGS"
    CMD="$CMD && $PY $HERE/synthesis/synthesize_vqa.py $bench --out-root=$SAMPLES --split=val50 $SYNTH_FLAGS"
  fi
  if [ "$STAGE" = "eval" ] || [ "$STAGE" = "all" ]; then
    EVAL_MODEL=$([ "$MODEL" = "lance" ] && echo "--model lance" || echo "")
    CMD="$CMD && $PY $HERE/eval/run_full_eval_pipeline.py --bench $bench $EVAL_MODEL --icl-samples-root=$SAMPLES \
--num-generations=5 --out=$ROOT/eval_results/${bench}.json"
  fi
  if [ "${SIMIT_LAUNCHER:-local}" = "bsub" ]; then
    bsub <<JOB
#!/bin/bash
#BSUB -q ${SIMIT_LSF_QUEUE:-p1}
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W $WALL
#BSUB -J simit_${MODEL}_${STAGE}_${bench}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -o $ROOT/logs/${STAGE}_${bench}_%J.out
source $SIMIT_ROOT/simit_env.sh
$CMD
JOB
  else
    bash -c "$CMD" 2>&1 | tee -a "$ROOT/logs/${STAGE}_${bench}.log"
  fi
done
