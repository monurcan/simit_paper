#!/bin/sh
# Single GPU only (matches all baselines). Same real training script, same
# cpu_offload=True (proven mandatory on 1 GPU), but a LARGER packed-batch
# token budget to see if amortizing offload's per-step transfer overhead
# over more work improves tokens/sec and MFU.
set -e
cd ${SIMIT_ROOT}/Bagel

EXPECTED_TOKENS="$1"
MAX_TOKENS="$2"
TAG="bigbatch_test_${EXPECTED_TOKENS}"
STEPS=6

export MASTER_ADDR=localhost
export MASTER_PORT=$((22000 + RANDOM % 1000))
export PYTHONPATH="$(pwd):$PYTHONPATH"

.venv/bin/torchrun \
  --nnodes=1 --node_rank=0 --nproc_per_node=1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file "data/configs/vqa_ai2d_v2_dfadaptive_manual_v1.yaml" \
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
  --log_every 1 \
  --lr 2e-4 \
  --warmup_steps 1 \
  --lr_scheduler cosine \
  --ema 0.95 \
  --freeze_vit True \
  --freeze_connector True \
  --use_lora True \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --num_workers 1 \
  --expected_num_tokens ${EXPECTED_TOKENS} \
  --max_num_tokens ${MAX_TOKENS} \
  --max_num_tokens_per_sample 16384 \
  --total_steps ${STEPS} \
  --save_every 999 \
  --results_dir "${SIMIT_ROOT}/speed_experiments/train_results/${TAG}" \
  --checkpoint_dir "${SIMIT_ROOT}/speed_experiments/train_ckpt/${TAG}" \
  --wandb_offline True \
  --wandb_name "${TAG}_$$"
