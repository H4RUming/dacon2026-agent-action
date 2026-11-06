#!/usr/bin/env bash
set -euo pipefail

repo_dir="${REPO_DIR:-/workspace/agent-behavior-prediction}"
if [ ! -d "${repo_dir}" ]; then
  repo_dir="${HOME}/agent-behavior-prediction"
fi
cd "${repo_dir}"

if [ -f ".venv-flow/bin/activate" ]; then
  source ".venv-flow/bin/activate"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate DACON
fi

mkdir -p logs oof checkpoints results
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

log_path="logs/fields_flow_last_action_seed42_$(date +%Y%m%d_%H%M%S).log"
python -m src.sweep_flow_variant_seed \
  --variant flow_last_action \
  --model jhu-clsp/mmBERT-base \
  --folds 5 \
  --split_seed 42 \
  --model_seed 42 \
  --epochs 4 \
  --batch 64 \
  --max_len 640 \
  --lr 4e-5 \
  --warmup 0.06 \
  --num_workers 3 \
  --tag fields_flow_last_action_seed42 \
  2>&1 | tee "${log_path}"
