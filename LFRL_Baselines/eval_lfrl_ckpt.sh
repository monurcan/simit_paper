#!/bin/bash
# Evaluate one LFRL LoRA checkpoint (paper App. G.3): TTRL, EMPO and TTRV decode with
# majority voting over 32 responses (maj@32, lmms-eval *_sc32 tasks, model bagel_batch);
# SRLM decodes greedily (model bagel_umm).
# Usage: ./eval_lfrl_ckpt.sh <method ttrl|empo|ttrv|srlm> <ckpt_dir> <lora_rank> <lite_task> <split val50|lite> <out_dir>
set -e
: "${SIMIT_ROOT:?run 'source simit_env.sh' first}"
METHOD="$1"; CKPT="$2"; RANK="$3"; TASK="$4"; SPLIT="$5"; OUT="$6"
SUFFIX=$([ "$SPLIT" = "val50" ] && echo "_val50" || echo "")
MODEL="$SIMIT_ROOT/Bagel/models/BAGEL-7B-MoT"
cd "$SIMIT_ROOT/lmms-eval"
if [ "$METHOD" = "srlm" ]; then
  "$SIMIT_ROOT/Bagel/.venv/bin/python" -m lmms_eval --model bagel_umm \
    --model_args "pretrained=$MODEL,mode=understanding,lora_ckpt_path=$CKPT" \
    --tasks "${TASK}${SUFFIX}" --batch_size 1 --log_samples --output_path "$OUT"
else
  "$SIMIT_ROOT/Bagel/.venv/bin/python" -m lmms_eval --model bagel_batch \
    --model_args "pretrained=$MODEL,lora_ckpt_path=$CKPT,lora_rank=$RANK,lora_alpha=$((2 * RANK))" \
    --tasks "${TASK}_sc32${SUFFIX}" --batch_size 1 --log_samples --output_path "$OUT"
fi
