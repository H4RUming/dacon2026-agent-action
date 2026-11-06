#!/bin/bash
set -u
cd ~/Desktop/agent-behavior-prediction || exit 1
source .venv/bin/activate
echo "=== START $(date) ==="
python -m src.sweep_llrd --llrd 0.9 --tag llrd09 2>&1 | tee llrd09.log
python -m src.sweep_llrd --ema 0.999 --tag ema999 2>&1 | tee ema999.log
python -m src.sweep_llrd --llrd 0.9 --ema 0.999 --tag llrd_ema 2>&1 | tee llrd_ema.log
echo "=== ALL DONE $(date) ==="
