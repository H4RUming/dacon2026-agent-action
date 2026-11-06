#!/bin/bash
set -u
cd ~/Desktop/agent-behavior-prediction || exit 1
source .venv/bin/activate
echo "=== OVERNIGHT START $(date) ==="

# 1) kd_a05 에포크 연장 (ep4 스케줄, ~3.3h)
python -m src.train_kd --alpha 0.5 --epochs 4 --tag kd_a05_ep4 2>&1 | tee kd_a05_ep4.log

# 2) tricks 스크린 — 가볍고 기대 큰 순서 (아침까지 3~4개 완료 예상)
python -m src.sweep_tricks --rdrop 1.0 --tag trk_rdrop 2>&1 | tee trk_rdrop.log
python -m src.sweep_tricks --ls 0.1     --tag trk_ls    2>&1 | tee trk_ls.log
python -m src.sweep_tricks --gce 0.7    --tag trk_gce   2>&1 | tee trk_gce.log
python -m src.sweep_tricks --elr 1      --tag trk_elr   2>&1 | tee trk_elr.log
python -m src.sweep_tricks --sam 0.05   --tag trk_sam   2>&1 | tee trk_sam.log
python -m src.sweep_tricks --fgm 1.0    --tag trk_fgm   2>&1 | tee trk_fgm.log
python -m src.sweep_tricks --sce 1      --tag trk_sce   2>&1 | tee trk_sce.log
python -m src.sweep_tricks --swa 1 --max_epochs 4 --tag trk_swa 2>&1 | tee trk_swa.log

echo "=== OVERNIGHT ALL DONE $(date) ==="
