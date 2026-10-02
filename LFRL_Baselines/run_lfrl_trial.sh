#!/bin/bash
# One LFRL baseline configuration (paper App. G.3): train one LoRA adapter on the benchmark's
# 500 unlabeled eval queries for 10 epochs (one checkpoint per epoch), then score the
# epoch-{3,4,5,6,10} checkpoints on the 50-sample validation split (out_dir/epoch<E>/).
#   ttrl  TTRL: GRPO, reward 1 for agreeing with the exact-match majority vote (K=32)
#   empo  EMPO: GRPO, word-Jaccard clustering of responses, threshold <method_param> (K=32)
#   ttrv  TTRV: GRPO, frequency reward + entropy term weighted by <method_param> (K=32)
#   srlm  SRLM: self-judged best/worst of K=4 responses, DPO with beta=0.1
# Usage: ./run_lfrl_trial.sh <method> <bench> <run_name> <temperature> <lr> <lora_rank> <method_param> <out_dir>
set -e
: "${SIMIT_ROOT:?run 'source simit_env.sh' first}"
METHOD="$1"; BENCH="$2"; RUN="$3"; TEMP="$4"; LR="$5"; RANK="$6"; PARAM="$7"; OUT="$8"
EPOCHS="3 4 5 6 10"
cd "$SIMIT_ROOT/LFRL_Baselines"
PY="$SIMIT_ROOT/Bagel/.venv/bin/python"
export PYTHONUNBUFFERED=1
N=$($PY -c "from common.dataset import load_real_samples; print(len(load_real_samples(benchmarks=['$BENCH'])))")
case "$METHOD" in
  ttrl) MODULE=ttrl.train_ttrl; DIR=ttrl; ARGS="--k 32 --voting_mode naive" ;;
  empo) MODULE=ttrl.train_ttrl; DIR=ttrl; ARGS="--k 32 --voting_mode lexical --lexical_similarity_threshold $PARAM" ;;
  ttrv) MODULE=ttrv.train_ttrv; DIR=ttrv; ARGS="--k 32 --voting_mode naive --entropy_coef $PARAM" ;;
  srlm) MODULE=srlm.train_srlm; DIR=srlm; ARGS="--k 4 --beta 0.1" ;;
  *) echo "unknown method $METHOD" >&2; exit 1 ;;
esac
$PY -m $MODULE --run_name "$RUN" --benchmarks "$BENCH" --num_epochs 10 --temperature "$TEMP" --lr "$LR" \
  --lora_rank "$RANK" --lora_alpha $((2 * RANK)) --save_every "$N" --device cuda:0 $ARGS
TASK=$($PY -c "import sys; sys.path.insert(0, '$SIMIT_ROOT/Bagel'); from simit_sweep import BENCHMARKS; print(BENCHMARKS['$BENCH']['task'])")
for E in $EPOCHS; do
  CKPT="$SIMIT_DATA/checkpoints/lfrl/$DIR/$RUN/step_$(printf %07d $((E * N)))"
  ./eval_lfrl_ckpt.sh "$METHOD" "$CKPT" "$RANK" "$TASK" val50 "$OUT/epoch$E"
done
