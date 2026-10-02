#!/bin/bash
# Zero-shot BAGEL-7B and the test-time-scaling baselines of Tab. 1 (paper App. G.2).
#
# Usage: ./run_baselines.sh <method> [benchmark ...] [--lmms-eval options, e.g. --limit 5]   (default: all 17)
#   zero_shot                      greedy direct answer (the baseline every gain is relative to)
#   think1k | think10k             BAGEL thinking mode, max_think_token_n = 1,000 / 10,000
#   sc8 | sc16 | sc32              Self-Consistency: first response greedy, the rest sampled at
#                                  temperature 0.7, majority vote (lmms-eval *_lite_sc* tasks)
#   self_refine1 | self_refine3 | self_refine5    Self-Refine with 1 / 3 / 5 feedback-refine rounds
# Results: $SIMIT_DATA/baselines/<method>/<benchmark>/. Nothing is tuned, so the eval split is used.
# Runs in the foreground; SIMIT_LAUNCHER=bsub submits one LSF job per benchmark instead.
set -e
: "${SIMIT_ROOT:?run 'source simit_env.sh' first}"
METHOD="$1"; shift || true
BENCHES=(); EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in --*) EXTRA+=("$1"); [ $# -gt 1 ] && [ "${2#--}" = "$2" ] && { EXTRA+=("$2"); shift; } ;; *) BENCHES+=("$1") ;; esac
  shift
done
if [ ${#BENCHES[@]} -eq 0 ]; then
  BENCHES=(ai2d chartqa coco2017_cap_val docvqa_val flickr30k_test gqa infovqa_val mmbench_cn_dev
           mmbench_en_dev nocaps_val ok_vqa_val2014 refcoco_bbox_val seedbench textcaps_val
           textvqa_val vizwiz_vqa_val vqav2_val)
fi
case "$METHOD" in
  zero_shot)     SUFFIX="";       MODE=understanding;       ARGS="" ;;
  think1k)       SUFFIX="_think"; MODE=think_understanding; ARGS="do_sample=False,max_think_token_n=1000" ;;
  think10k)      SUFFIX="_think"; MODE=think_understanding; ARGS="do_sample=False,max_think_token_n=10000" ;;
  sc8)           SUFFIX="_sc";    MODE=understanding;       ARGS="do_sample=True,text_temperature=0.7" ;;
  sc16)          SUFFIX="_sc16";  MODE=understanding;       ARGS="do_sample=True,text_temperature=0.7" ;;
  sc32)          SUFFIX="_sc32";  MODE=understanding;       ARGS="do_sample=True,text_temperature=0.7" ;;
  self_refine[135]) SUFFIX="";    MODE=self_refine;         ARGS="self_refine_iters=${METHOD#self_refine}" ;;
  *) echo "unknown method '$METHOD'" >&2; exit 1 ;;
esac
OUT="$SIMIT_DATA/baselines/$METHOD"
MODEL="$SIMIT_ROOT/Bagel/models/BAGEL-7B-MoT"
mkdir -p "$OUT/jobs"
for bench in "${BENCHES[@]}"; do
  CMD="cd $SIMIT_ROOT/Bagel && ./run_bagel_eval.sh $MODEL ${bench}_lite${SUFFIX} $OUT/$bench $MODE '$ARGS' ${EXTRA[*]}"
  if [ "${SIMIT_LAUNCHER:-local}" = "bsub" ]; then
    bsub <<JOB
#!/bin/bash
#BSUB -q ${SIMIT_LSF_QUEUE:-gpuh100}
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -J ${METHOD}_${bench}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -o $OUT/jobs/${bench}_%J.out
source $SIMIT_ROOT/simit_env.sh
$CMD
JOB
  else
    bash -c "$CMD"
  fi
done
