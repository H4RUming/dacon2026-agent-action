#!/bin/bash
set -u
cd ~/Desktop/agent-behavior-prediction || exit 1
source .venv/bin/activate
echo "=== START $(date) ==="
# 1) KD alpha=0.5 (5-fold ~2.5h) — OOF/오프셋 균형점 탐색
python -m src.train_kd --alpha 0.5 --tag kd_a05 2>&1 | tee kd_a05.log
# 2) mmBERT-small + fields (5-fold ~1.5h) — 3모델 시간내 조합 재료
python -m src.sweep_epochs --model jhu-clsp/mmBERT-small --max_epochs 4 --batch 128 --max_len 640 --lr 5e-5 --tag small_fields 2>&1 | tee small_fields.log
echo "=== ALL DONE $(date) ==="
