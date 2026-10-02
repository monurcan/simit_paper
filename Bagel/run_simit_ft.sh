#!/bin/bash
# One SIMIT-FT (Test-Set) configuration (paper Sec. 3.3 / App. G.1):
#   1. pool the synthetic samples of all unlabeled eval queries of <bench>, keeping, per query,
#      the first K*(p0) candidates (ABA) whose confidence lies in [T_low, T_high] (DF);
#   2. train one LoRA adapter (rank r, alpha 2r, EMA 0.95, clip(N/2, 40, 150) steps);
#   3. evaluate the final EMA checkpoint on <eval_task> (the *_val50 task during tuning).
# The checkpoint is kept in $SIMIT_DATA/checkpoints/simit_ft_ts/<tag> for the final eval-split
# evaluation (run_ft_sweep.py --finalize), for SIMIT-FT -> ICL and for the ablations.
#
# Usage: ./run_simit_ft.sh <bench> <tag> <pool v2|nodf4> <epsilon> <A0> <A1> <B> <T_low> <T_high> \
#                          <lora_rank> <lr> <eval_task> <out_dir>
set -e
: "${SIMIT_ROOT:?run 'source simit_env.sh' first}"
BENCH="$1"; TAG="$2"; POOL="$3"; EPS="$4"; A0="$5"; A1="$6"; B="$7"; T_LOW="$8"; T_HIGH="$9"
RANK="${10}"; LR="${11}"; EVAL_TASK="${12}"; OUT_DIR="${13}"
cd "$SIMIT_ROOT/Bagel"

.venv/bin/python prepare_vqa_data_v2_dfadaptive.py --tag "$TAG" --benches "$BENCH" --pool "$POOL" \
  --epsilon "$EPS" --A0 "$A0" --A1 "$A1" --B "$B" --T-low "$T_LOW" --T-high "$T_HIGH"
NAME="vqa_${BENCH}_v2_dfadaptive_${TAG}"
N=$(.venv/bin/python -c "import json; print(json.load(open('data_root/vqa_benchmarks_v2_dfadaptive/$TAG/registry.json'))['$NAME']['num_total_samples'])")
STEPS=$(( N / 2 )); [ "$STEPS" -lt 40 ] && STEPS=40; [ "$STEPS" -gt 150 ] && STEPS=150
CKPT="$SIMIT_DATA/checkpoints/simit_ft_ts/$TAG"
echo "[run_simit_ft] $BENCH/$TAG: $N training records -> $STEPS steps"
if [ "$N" -gt 0 ]; then
  ./train_lora.sh "data/configs/${NAME}.yaml" "$CKPT" "$STEPS" "$LR" "$RANK" $((2 * RANK)) 0.95
fi
./eval_lora_ckpt.sh "$CKPT" "$EVAL_TASK" "$OUT_DIR"
