#!/usr/bin/env bash
set -euo pipefail
cd /workspace/agent-behavior-prediction
mkdir -p logs results/screens oof/screens
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
for fold in 2 3 4; do
  tag="cloud_pro6000_distill_l050_t20_fold${fold}"
  log="logs/${tag}_$(date +%Y%m%d_%H%M%S).log"
  python3 -m src.search4090_distill_screen \
    --candidate_tag "${tag}" \
    --reference_tag search4090_flow_exact \
    --reference_macro_f1 0.7719461292810287 \
    --teacher_probs_path oof/search_teacher_pool_flow45_flow47_nested_bias_probs.npy \
    --train_ids_path oof/train_ids.npy \
    --variant flow_exact \
    --lambda_distill 0.50 \
    --temperature 2.0 \
    --fold "${fold}" \
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
    2>&1 | tee "${log}"
done
