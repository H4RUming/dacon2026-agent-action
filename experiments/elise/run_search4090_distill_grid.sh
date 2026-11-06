#!/usr/bin/env bash
set -euo pipefail

cd /home/user/agent-behavior-prediction
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate DACON

mkdir -p logs results/screens oof/screens
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

run_one() {
  local lam="$1"
  local temp="$2"
  local tag_lam="${lam/./}"
  local tag_temp="${temp/./}"
  local tag="search4090_distill_l${tag_lam}_t${tag_temp}"
  local log_path="logs/${tag}_$(date +%Y%m%d_%H%M%S).log"

  python -m src.search4090_distill_screen \
    --candidate_tag "${tag}" \
    --reference_tag search4090_flow_exact \
    --reference_macro_f1 0.7719461292810287 \
    --teacher_probs_path oof/search_teacher_pool_flow45_flow47_nested_bias_probs.npy \
    --train_ids_path oof/train_ids.npy \
    --variant flow_exact \
    --lambda_distill "${lam}" \
    --temperature "${temp}" \
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
}

run_one 0.20 1.5
run_one 0.20 2.0
run_one 0.35 1.5
run_one 0.35 2.0
run_one 0.50 1.5
run_one 0.50 2.0
