#!/bin/bash
# 하드-라벨 lr5 full 재학습 (hp_lr5e5_ep3 레시피의 전량학습판).
# 용도: kd_a03+s43+lr5 하드-트리오 — 소프트-노출 상관 함정이 없는 3번째 멤버.
# 근거: 트리오 deploy su 0.79063 (+0.0025 vs 현행 페어), 하드 멤버라 +0.003 잔차 유효.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

echo "=== lr5 full retrain $(date) ==="
python -m src.train_full --lr 5e-5 --epochs 3 --batch 64 --max_len 640 \
  --out submit/model_lr5full 2>&1 | tee train_full_lr5_v2.log

echo "=== quantize $(date) ==="
python -m src.quantize_int8 submit/model_lr5full submit/model_lr5full.npz
echo "=== DONE $(date) ==="
