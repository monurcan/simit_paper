#!/bin/sh
# LoRA-finetune BAGEL-7B on one vlm_sft dataset config (SIMIT-FT (Test-Set), its
# ablations, and the real-data SFT reference). Runs in the foreground on 1 GPU
# (FSDP + CPU offload); wrap it in a bsub job for cluster runs.
#
# Usage: ./train_lora.sh <dataset_config_yaml> <checkpoint_dir> <total_steps> <lr> <lora_rank> <lora_alpha> [ema] [save_every]
#
# Paper recipe (App. G.1): AdamW (0.9, 0.95), no weight decay, cosine schedule with
# 10% warmup, grad clip 1.0, LoRA (no dropout) on the language model's
# understanding-expert attention + MLP projections only (ViT, connector and
# generation parts frozen), EMA 0.95 of the weights is what gets saved/evaluated.
# Needs ~170GB host RAM per job because of CPU offloading (bsub: -n 4, rusage[mem=170GB]).
set -e
: "${SIMIT_ROOT:?run 'source simit_env.sh' first}"

DATASET_CONFIG="$1"
CKPT_DIR="$2"
TOTAL_STEPS="$3"
LR="$4"
LORA_RANK="$5"
LORA_ALPHA="$6"
EMA="${7:-0.95}"
SAVE_EVERY="${8:-$TOTAL_STEPS}"
if [ -z "$LORA_ALPHA" ]; then
  echo "Usage: $0 <dataset_config_yaml> <checkpoint_dir> <total_steps> <lr> <lora_rank> <lora_alpha> [ema] [save_every]" >&2
  exit 1
fi
WARMUP_STEPS=$((TOTAL_STEPS / 10))
[ "$WARMUP_STEPS" -lt 1 ] && WARMUP_STEPS=1

cd "$SIMIT_ROOT/Bagel"
export MASTER_ADDR=localhost
export MASTER_PORT=$((20000 + $$ % 20000))
export PYTHONPATH="$SIMIT_ROOT/Bagel:$PYTHONPATH"
RUN_NAME="$(basename "$CKPT_DIR")"

.venv/bin/torchrun \
  --nnodes=1 --node_rank=0 --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file "$DATASET_CONFIG" \
  --model_path models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --num_shard 1 \
  --cpu_offload True \
  --visual_gen False \
  --resume_from models/BAGEL-7B-MoT \
  --finetune_from_hf True \
  --auto_resume False \
  --resume_model_only True \
  --finetune_from_ema True \
  --log_every 10 \
  --lr "$LR" \
  --warmup_steps "$WARMUP_STEPS" \
  --lr_scheduler cosine \
  --ema "$EMA" \
  --freeze_vit True \
  --freeze_connector True \
  --use_lora True \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout 0.0 \
  --num_workers 1 \
  --expected_num_tokens 16384 \
  --max_num_tokens 36864 \
  --max_num_tokens_per_sample 16384 \
  --total_steps "$TOTAL_STEPS" \
  --save_every "$SAVE_EVERY" \
  --results_dir "$SIMIT_DATA/train_logs/$RUN_NAME" \
  --checkpoint_dir "$CKPT_DIR" \
  --wandb_offline True \
  --wandb_name "$RUN_NAME"
