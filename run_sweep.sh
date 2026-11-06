#!/bin/bash
set -u
cd ~/Desktop/agent-behavior-prediction || exit 1
source .venv/bin/activate

echo "=== START $(date) ==="
python -m src.sweep_epochs --max_epochs 4 --batch 64 --max_len 640 --lr 2e-5 --tag hp_lr2e5 2>&1 | tee hp_lr2e5.log
python -m src.sweep_epochs --max_epochs 4 --batch 64 --max_len 640 --lr 3e-5 --tag hp_lr3e5 2>&1 | tee hp_lr3e5.log
python -m src.sweep_epochs --max_epochs 4 --batch 64 --max_len 640 --lr 5e-5 --tag hp_lr5e5 2>&1 | tee hp_lr5e5.log
python -m src.sweep_epochs --max_epochs 4 --batch 64 --max_len 640 --lr 4e-5 --warmup 0.03 --tag hp_wu003 2>&1 | tee hp_wu003.log
python -m src.sweep_epochs --max_epochs 4 --batch 64 --max_len 640 --lr 4e-5 --warmup 0.10 --tag hp_wu010 2>&1 | tee hp_wu010.log
echo "=== ALL DONE $(date) ==="
