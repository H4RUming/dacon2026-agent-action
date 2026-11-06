#!/usr/bin/env bash
set -euo pipefail

cd /workspace/agent-behavior-prediction
mkdir -p logs results/screens oof/screens

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REF=0.7848645177319582

run_one() {
  local tag="$1"
  local teacher="$2"
  local lam="$3"
  local temp="$4"
  local log="logs/${tag}_$(date +%Y%m%d_%H%M%S).log"

  python3 -m src.search4090_distill_screen \
    --candidate_tag "${tag}" \
    --reference_tag distill_l050_t20_fold0 \
    --reference_macro_f1 "${REF}" \
    --teacher_probs_path "${teacher}" \
    --train_ids_path oof/train_ids.npy \
    --variant flow_exact \
    --lambda_distill "${lam}" \
    --temperature "${temp}" \
    --fold 0 \
    --split_seed 42 \
    --model_seed 42 \
    --epochs 3 \
    --primary_epoch 3 \
    --micro_batch 8 \
    --grad_accum 8 \
    --val_batch 64 \
    --max_len 640 \
    --lr 4e-5 \
    --warmup 0.06 \
    --num_workers 2 \
    --force \
    2>&1 | tee "${log}"
}

run_one screen_selfdistill_l035_t20 oof/distill_l050_t20_fullcv_ep3_probs.npy 0.35 2.0
run_one screen_selfdistill_l050_t20 oof/distill_l050_t20_fullcv_ep3_probs.npy 0.50 2.0
run_one screen_selfdistill_l065_t20 oof/distill_l050_t20_fullcv_ep3_probs.npy 0.65 2.0
run_one screen_lastaction_fixed_l050_t20 oof/search_teacher_pool_with_last_action_fixed_probs.npy 0.50 2.0
run_one screen_lastaction_fixed_l065_t20 oof/search_teacher_pool_with_last_action_fixed_probs.npy 0.65 2.0
