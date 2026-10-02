#!/bin/sh
# Evaluate the last checkpoint saved by train_lora.sh under <checkpoint_root>
# (<checkpoint_root>/<step>/ema.safetensors + lora_config.json) on lmms-eval tasks.
# If no checkpoint exists (a training set that DF+ABA filtered down to nothing),
# the unadapted base model is evaluated instead.
# Usage: ./eval_lora_ckpt.sh <checkpoint_root> <tasks_comma_separated> <output_dir> [extra lmms_eval args...]
set -e
: "${SIMIT_ROOT:?run 'source simit_env.sh' first}"
CKPT_ROOT="$1"; TASKS="$2"; OUT_DIR="$3"; shift 3
STEP_DIR=$(ls -d "$CKPT_ROOT"/[0-9]* 2>/dev/null | sort | tail -1)
cd "$SIMIT_ROOT/Bagel"
if [ -n "$STEP_DIR" ]; then
  ./run_bagel_eval.sh "$SIMIT_ROOT/Bagel/models/BAGEL-7B-MoT" "$TASKS" "$OUT_DIR" understanding "lora_ckpt_path=$STEP_DIR" "$@"
else
  echo "[eval_lora_ckpt] no checkpoint under $CKPT_ROOT, evaluating the base model"
  ./run_bagel_eval.sh "$SIMIT_ROOT/Bagel/models/BAGEL-7B-MoT" "$TASKS" "$OUT_DIR" understanding "" "$@"
fi
