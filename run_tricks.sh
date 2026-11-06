#!/bin/bash
set -u
cd ~/Desktop/agent-behavior-prediction || exit 1
source .venv/bin/activate
echo "=== TRICKS START $(date) ==="
python -m src.sweep_tricks --rdrop 1.0 --tag trk_rdrop 2>&1 | tee trk_rdrop.log
python -m src.sweep_tricks --ls 0.1     --tag trk_ls    2>&1 | tee trk_ls.log
python -m src.sweep_tricks --gce 0.7    --tag trk_gce   2>&1 | tee trk_gce.log
python -m src.sweep_tricks --elr 1      --tag trk_elr   2>&1 | tee trk_elr.log
python -m src.sweep_tricks --sam 0.05   --tag trk_sam   2>&1 | tee trk_sam.log
python -m src.sweep_tricks --fgm 1.0    --tag trk_fgm   2>&1 | tee trk_fgm.log
python -m src.sweep_tricks --sce 1      --tag trk_sce   2>&1 | tee trk_sce.log
python -m src.sweep_tricks --swa 1 --max_epochs 4 --tag trk_swa 2>&1 | tee trk_swa.log
echo "=== TRICKS ALL DONE $(date) ==="
