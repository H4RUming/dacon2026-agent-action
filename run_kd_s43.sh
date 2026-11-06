#!/bin/bash
# 야간: seed-43 α=0.3 ep3 KD 학생 — kd_a03과 독립 시드 페어 재료.
# 근거: kd_a03+kd_a03e4 페어가 honest su +0.0044 (ep4 노출 페널티 포함).
#       같은 다양성을 ep4 페널티 없이 (α=0.3 ep3 × 시드 2개)로 재현하는 시도.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

echo "=== 1) 5-fold OOF (seed 43) $(date) ==="
python -m src.train_kd --alpha 0.3 --epochs 3 --seed 43 --tag kd_a03_s43 2>&1 | tee kd_a03_s43.log

echo "=== 2) full-data retrain $(date) ==="
python -m src.train_kd_full --alpha 0.3 --epochs 3 --seed 43 --out submit/model_kd03_s43 2>&1 | tee train_kd_full_a03s43.log

echo "=== 3) int8 quantize $(date) ==="
python -m src.quantize_int8 submit/model_kd03_s43 submit/model_kd03_s43.npz

echo "=== DONE $(date) ==="
