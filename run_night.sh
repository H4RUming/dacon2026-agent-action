#!/bin/bash
set -u
cd ~/Desktop/agent-behavior-prediction || exit 1
source .venv/bin/activate

echo "=== START $(date) ==="
python -m src.sweep_epochs --model intfloat/multilingual-e5-large --max_epochs 4 --batch 32 --max_len 512 --tag e5large_ctx 2>&1 | tee e5large_ctx.log
python -m src.sweep_epochs --model BAAI/bge-m3 --max_epochs 4 --batch 32 --max_len 512 --tag bgem3_ctx 2>&1 | tee bgem3_ctx.log
python -m src.sweep_epochs --model nlpai-lab/KURE-v1 --max_epochs 4 --batch 32 --max_len 512 --tag kure_ctx 2>&1 | tee kure_ctx.log
python -m src.sweep_supcon --lam 0.5 --max_epochs 4 --batch 64 --max_len 512 --tag supcon_lam05_ctx 2>&1 | tee supcon_lam05_ctx.log
echo "=== ALL DONE $(date) ==="
