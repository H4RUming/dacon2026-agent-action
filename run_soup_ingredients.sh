#!/bin/bash
set -u
cd ~/Desktop/agent-behavior-prediction || exit 1
source .venv/bin/activate

echo "=== START $(date) ==="
# 재료 1: 현 레시피 + 시드 43 (다양성: 시드)
python -m src.sweep_save --max_epochs 3 --batch 64 --max_len 640 --lr 4e-5 --seed 43 --tag soup_s43 2>&1 | tee soup_s43.log
# 재료 2: lr 5e-5 + 시드 42 (다양성: lr)
python -m src.sweep_save --max_epochs 3 --batch 64 --max_len 640 --lr 5e-5 --seed 42 --tag soup_lr5 2>&1 | tee soup_lr5.log
# 재료 3: 현 레시피 + 시드 44 (다양성: 시드)
python -m src.sweep_save --max_epochs 3 --batch 64 --max_len 640 --lr 4e-5 --seed 44 --tag soup_s44 2>&1 | tee soup_s44.log
echo "=== ALL DONE $(date) ==="
