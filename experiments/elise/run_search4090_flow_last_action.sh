#!/usr/bin/env bash
set -euo pipefail

cd /home/user/agent-behavior-prediction
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate DACON

mkdir -p logs results/screens oof/screens
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

log_path="logs/search4090_flow_last_action_$(date +%Y%m%d_%H%M%S).log"
python -m src.search4090_flow_screen \
  --candidate_variant flow_last_action \
  --candidate_tag search4090_flow_last_action \
  --reference_variant flow_exact \
  --reference_tag search4090_flow_exact \
  --fold 0 \
  --split_seed 42 \
  --model_seed 42 \
  --epochs 4 \
  --primary_epoch 3 \
  --micro_batch 8 \
  --grad_accum 8 \
  --val_batch 64 \
  --max_len 640 \
  --lr 4e-5 \
  --warmup 0.06 \
  --num_workers 2 \
  2>&1 | tee "${log_path}"
